"""
Pipeline:
  skull strip -> smoothed border erosion -> grey-HU cleanup -> salt&pepper (tissue) -> bbox
  -> adaptive binarization (HU high FIXED, HU low adaptive) -> tissue-mask multiply -> salt&pepper (mask)

ONE cache: a (5,H,W) float16 .npy per scan with planes
    0 = raw CT (window 0-100, skull kept)
    1 = cleaned CT (skull-stripped)
    2 = guide_mask (binary 0/1)
    3 = guide_mask_extended (clean HU inside the dilated region)
    4 = guide_hu (clean HU inside the binarised region, 0 elsewhere)
The cache folder name contains a fingerprint of every parameter. Writes are atomic.
All parameters come from cfg["preprocessing"]; nothing runs at import time.
"""
import hashlib
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pydicom

PLANE_NAMES = ("raw", "clean", "guide", "ext", "guide_hu")
N_PLANES = len(PLANE_NAMES)

_INTERP = {"nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}

_FP_PARAMS = [
    "bone_hu_threshold", "pocket_hu_low", "pocket_hu_high", "min_pocket_area",
    "border_erode_mm", "border_smooth_sigma", "tissue_hu_threshold", "hu_low_start", "hu_high_fixed",
    "hu_step", "white_pct_target", "min_white_pct_floor", "fine_step", "hu_low_cap", "hu_low_min",
    "salt_pepper_kernel_size", "pre_bbox_salt_pepper_kernel_size", "extend_pixels", "hu_extend_margin",
    "neighbour_hu_low", "neighbour_hu_high", "grey_region_hu_low", "grey_region_hu_cutoff", "mask_interp",
]


def _warm_one(prep, rec):
    cv2.setNumThreads(1)
    before = prep.stats["miss"]
    try:
        prep.get_planes(rec)
        return ("computed" if prep.stats["miss"] > before else "cached"), None
    except Exception as e:
        return "failed", f"{rec['filename']}: {e}"


class Preprocessor:
    def __init__(self, cfg):
        self.p = SimpleNamespace(**cfg["preprocessing"])
        self.p.grey_region_hu_low = self.p.tissue_hu_threshold
        self.img_size = cfg["data"]["img_size"]
        self.version = cfg["cache"]["version"]
        self.mask_interp = _INTERP[self.p.mask_interp]

        self.cache_dir = os.path.join(cfg["paths"]["output_dir"],
                                      f"{cfg['cache']['prefix']}_{self._fingerprint()}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = {"hit": 0, "miss": 0}     # per-process counters

    # ------------------------------------------------------------------
    # pipeline functions
    # ------------------------------------------------------------------
    @staticmethod
    def read_hu(dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        return pixel_array * slope + intercept, ds

    def skull_strip(self, hu):
        """Remove HU >= bone threshold, then blank small disconnected pockets whose HU is fully
        inside [pocket_hu_low, pocket_hu_high]."""
        p = self.p
        bone_mask = (hu >= p.bone_hu_threshold)
        tissue_mask = (~bone_mask).astype(np.uint8) * 255

        if p.min_pocket_area > 0:
            num_labels, labels = cv2.connectedComponents(tissue_mask, connectivity=8)
            if num_labels > 1:
                for lab in range(1, num_labels):
                    comp = (labels == lab)
                    area = int(comp.sum())
                    if area >= p.min_pocket_area:
                        continue
                    comp_hu = hu[comp]
                    if comp_hu.size and comp_hu.min() >= p.pocket_hu_low and comp_hu.max() <= p.pocket_hu_high:
                        tissue_mask[comp] = 0

        clean_hu = hu.copy()
        clean_hu[tissue_mask == 0] = -1000.0
        return clean_hu, tissue_mask

    @staticmethod
    def get_pixel_spacing_mm(ds):
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is None:
            spacing = getattr(ds, "ImagerPixelSpacing", None)
        if spacing is None:
            return 1.0
        return (float(spacing[0]) + float(spacing[1])) / 2.0

    def erode_tissue_border(self, clean_hu, tissue_mask, ds):
        p = self.p
        pixel_spacing_mm = self.get_pixel_spacing_mm(ds)
        erode_px = max(1, round(p.border_erode_mm / pixel_spacing_mm))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))

        mask = (tissue_mask > 0).astype(np.uint8) * 255
        if p.border_smooth_sigma > 0:
            blurred = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), p.border_smooth_sigma)
            smooth_mask = ((blurred >= 0.5) & (mask > 0)).astype(np.uint8) * 255
        else:
            smooth_mask = mask

        eroded_mask = cv2.erode(smooth_mask, kernel, iterations=1)
        out_hu = clean_hu.copy()
        out_hu[eroded_mask == 0] = -1000.0
        return out_hu, eroded_mask

    def remove_high_hu_in_grey(self, clean_hu, tissue_mask):
        grey_region = clean_hu >= self.p.grey_region_hu_low
        high_hu_in_grey = grey_region & (clean_hu > self.p.grey_region_hu_cutoff)
        if not high_hu_in_grey.any():
            return clean_hu, tissue_mask, 0
        out_hu = clean_hu.copy()
        out_hu[high_hu_in_grey] = -1000.0
        out_mask = tissue_mask.copy()
        out_mask[high_hu_in_grey] = 0
        return out_hu, out_mask, int(high_hu_in_grey.sum())

    @staticmethod
    def remove_salt_and_pepper(mask_uint8, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def remove_salt_pepper_from_tissue(self, clean_hu, tissue_mask):
        white = (clean_hu >= self.p.tissue_hu_threshold).astype(np.uint8)
        cleaned = self.remove_salt_and_pepper(white, self.p.pre_bbox_salt_pepper_kernel_size)
        removed = (white == 1) & (cleaned == 0)
        if not removed.any():
            return clean_hu, tissue_mask, 0
        out_hu = clean_hu.copy()
        out_hu[removed] = -1000.0
        out_mask = tissue_mask.copy()
        out_mask[removed] = 0
        return out_hu, out_mask, int(removed.sum())

    def find_tissue_bbox(self, clean_hu):
        tissue = clean_hu >= self.p.tissue_hu_threshold
        rows_any = np.any(tissue, axis=1)
        cols_any = np.any(tissue, axis=0)
        if not rows_any.any() or not cols_any.any():
            return None
        r0 = int(np.argmax(rows_any))
        r1 = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
        c0 = int(np.argmax(cols_any))
        c1 = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
        return (r0, r1, c0, c1)

    @staticmethod
    def _white_pct_at(bbox_hu, hu_low, hu_high, bbox_total):
        if bbox_total == 0:
            return 0.0
        mask = (bbox_hu >= hu_low) & (bbox_hu <= hu_high)
        return (mask.sum() / bbox_total) * 100.0

    def adaptive_binarize_in_bbox(self, clean_hu, bbox):
        """HU high edge fixed, HU low edge adapted so white % of the bbox lands in [floor, target)."""
        p = self.p
        hu_high_fixed = p.hu_high_fixed
        target_pct, floor_pct = p.white_pct_target, p.min_white_pct_floor
        fine_step, coarse_step = p.fine_step, p.hu_step
        hu_low_cap, hu_low_min = p.hu_low_cap, p.hu_low_min

        r0, r1, c0, c1 = bbox
        bbox_hu = clean_hu[r0:r1 + 1, c0:c1 + 1]
        bbox_total = bbox_hu.size

        hu_low = p.hu_low_start
        white_pct = self._white_pct_at(bbox_hu, hu_low, hu_high_fixed, bbox_total)

        if floor_pct <= white_pct < target_pct:
            pass
        elif white_pct < floor_pct:
            while white_pct < floor_pct and hu_low > hu_low_min:
                hu_low -= fine_step
                white_pct = self._white_pct_at(bbox_hu, hu_low, hu_high_fixed, bbox_total)
                if white_pct >= target_pct:
                    hu_low += fine_step
                    white_pct = self._white_pct_at(bbox_hu, hu_low, hu_high_fixed, bbox_total)
                    break
        else:
            prev_hu_low = hu_low
            while white_pct >= target_pct and hu_low < hu_low_cap:
                prev_hu_low = hu_low
                hu_low += coarse_step
                white_pct = self._white_pct_at(bbox_hu, hu_low, hu_high_fixed, bbox_total)
            if white_pct < floor_pct and hu_low > prev_hu_low:
                fine_hu_low = prev_hu_low
                fine_pct = self._white_pct_at(bbox_hu, fine_hu_low, hu_high_fixed, bbox_total)
                while fine_hu_low < hu_low and not (floor_pct <= fine_pct < target_pct):
                    fine_hu_low += fine_step
                    fine_pct = self._white_pct_at(bbox_hu, fine_hu_low, hu_high_fixed, bbox_total)
                hu_low = fine_hu_low
                white_pct = fine_pct

        full_mask = np.zeros(clean_hu.shape, dtype=np.uint8)
        full_mask[r0:r1 + 1, c0:c1 + 1] = ((clean_hu[r0:r1 + 1, c0:c1 + 1] >= hu_low) &
                                           (clean_hu[r0:r1 + 1, c0:c1 + 1] <= hu_high_fixed)).astype(np.uint8)
        return full_mask, hu_low

    def extend_by_hu_similarity(self, guide_mask_u8, clean_hu, hu_low_used):
        p = self.p
        if p.extend_pixels <= 0:
            return guide_mask_u8.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * p.extend_pixels + 1, 2 * p.extend_pixels + 1))
        spatial_candidate = cv2.dilate(guide_mask_u8, kernel, iterations=1)
        intensity_candidate = ((clean_hu >= hu_low_used - p.hu_extend_margin) &
                               (clean_hu <= p.hu_high_fixed)).astype(np.uint8)
        return ((spatial_candidate & intensity_candidate) | guide_mask_u8).astype(np.uint8)

    @staticmethod
    def hu_to_normalized(hu_array, low=0, high=100):
        windowed = np.clip(hu_array, low, high)
        return ((windowed - low) / (high - low)).astype(np.float32)

    def compute_planes(self, filepath, img_size=None):
        """DICOM -> float32 array (5, img_size, img_size): [raw, clean, guide, ext, guide_hu]."""
        p = self.p
        img_size = self.img_size if img_size is None else img_size

        hu, ds = self.read_hu(filepath)
        raw = cv2.resize(self.hu_to_normalized(hu), (img_size, img_size), interpolation=cv2.INTER_AREA)

        clean_hu, tissue_mask = self.skull_strip(hu)
        clean_hu, tissue_mask = self.erode_tissue_border(clean_hu, tissue_mask, ds)
        clean_hu, tissue_mask, _ = self.remove_high_hu_in_grey(clean_hu, tissue_mask)
        clean_hu, tissue_mask, _ = self.remove_salt_pepper_from_tissue(clean_hu, tissue_mask)

        bbox = self.find_tissue_bbox(clean_hu)
        if bbox is None:
            guide_full = np.zeros(clean_hu.shape, dtype=np.uint8)
            hu_low_used = p.hu_low_start
        else:
            guide_full, hu_low_used = self.adaptive_binarize_in_bbox(clean_hu, bbox)
            guide_full = guide_full * tissue_mask.clip(0, 1).astype(np.uint8)
            guide_full = self.remove_salt_and_pepper(guide_full, p.salt_pepper_kernel_size)

        dilated = self.extend_by_hu_similarity(guide_full, clean_hu, hu_low_used)
        clean_norm_full = self.hu_to_normalized(clean_hu, p.neighbour_hu_low, p.neighbour_hu_high)
        ext_full = clean_norm_full * dilated.astype(np.float32)
        guide_hu_full = clean_norm_full * guide_full.astype(np.float32)   # binarised region with real HU values

        clean = cv2.resize(self.hu_to_normalized(clean_hu), (img_size, img_size), interpolation=cv2.INTER_AREA)
        guide = cv2.resize(guide_full.astype(np.float32), (img_size, img_size), interpolation=self.mask_interp)
        ext = cv2.resize(ext_full, (img_size, img_size), interpolation=self.mask_interp)
        guide_hu = cv2.resize(guide_hu_full, (img_size, img_size), interpolation=cv2.INTER_AREA)
        return np.stack([raw, clean, guide, ext, guide_hu]).astype(np.float32)

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _fingerprint(self):
        values = {}
        for k in _FP_PARAMS:
            v = getattr(self.p, k)
            values[k] = float(v) if isinstance(v, (int, float)) else str(v)
        values["version"] = self.version
        return hashlib.md5(json.dumps(values, sort_keys=True).encode()).hexdigest()[:10]

    def cache_path(self, rec, img_size=None):
        img_size = self.img_size if img_size is None else img_size
        try:
            size = os.path.getsize(rec["filepath"])
        except OSError:
            size = 0
        raw = f"{rec['class_name']}|{rec['filename']}|{size}|{img_size}"
        return os.path.join(self.cache_dir, hashlib.md5(raw.encode()).hexdigest() + f"_{img_size}.npy")

    def get_planes(self, rec, img_size=None):
        """float32 (5,H,W). Reads the cache; computes + writes it on a miss."""
        img_size = self.img_size if img_size is None else img_size
        path = self.cache_path(rec, img_size)
        if os.path.exists(path):
            try:
                a = np.load(path)
                if a.shape == (N_PLANES, img_size, img_size):
                    self.stats["hit"] += 1
                    return a.astype(np.float32)
            except Exception:
                pass                                   # corrupt/partial file -> recompute below
        self.stats["miss"] += 1
        planes16 = self.compute_planes(rec["filepath"], img_size).astype(np.float16)
        tmp = path[:-4] + f".tmp{os.getpid()}.npy"
        np.save(tmp, planes16)
        os.replace(tmp, path)                          # atomic: readers never see half-written files
        return planes16.astype(np.float32)             # same values as a later cache hit

    def cache_report(self, records, img_size=None, verbose=True):
        img_size = self.img_size if img_size is None else img_size
        n_have = sum(os.path.exists(self.cache_path(r, img_size)) for r in records)
        if verbose:
            print(f"[cache] dir: {self.cache_dir}")
            print(f"[cache] {n_have}/{len(records)} scans already cached "
                  f"({len(records) - n_have} will be computed)")
        return n_have

    def warm_cache(self, records, img_size=None, n_jobs=4):
        from joblib import Parallel, delayed
        from tqdm import tqdm

        img_size = self.img_size if img_size is None else img_size
        n_jobs = max(1, min(n_jobs, os.cpu_count() or 1))
        self.cache_report(records, img_size)
        print(f"[cache] warming with {n_jobs} process(es)...")
        if n_jobs == 1:
            cv2.setNumThreads(1)
            results = [_warm_one(self, r) for r in tqdm(records, desc="warm-up")]
        else:
            try:
                gen = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16, return_as="generator")(
                    delayed(_warm_one)(self, r) for r in records)
                results = list(tqdm(gen, total=len(records), desc="warm-up"))
            except TypeError:       # old joblib without return_as
                results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16)(
                    delayed(_warm_one)(self, r) for r in records)
        counts = {k: sum(1 for s, _ in results if s == k) for k in ("cached", "computed", "failed")}
        print(f"[cache] done: {counts['cached']} were already cached, {counts['computed']} computed, "
              f"{counts['failed']} failed")
        for s, msg in results:
            if s == "failed":
                print("   FAILED:", msg)
        return counts
