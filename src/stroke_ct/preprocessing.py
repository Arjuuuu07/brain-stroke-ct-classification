"""
Pipeline order (old v13 structure + pre-bbox cleanup):
  skull strip -> smoothed border erosion -> grey-HU cleanup ->
  salt-and-pepper on white tissue (before bbox) -> bounding box ->
  adaptive binarization (HU high FIXED, HU low adaptive) ->
  tissue-mask multiply -> salt-and-pepper on guide mask.
Also: cached image loader.

All parameters come from cfg["preprocessing"]; nothing runs at import time.
"""
import hashlib
import json
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pydicom


class Preprocessor:
    CACHE_RELEVANT_PARAMS = [
        "bone_hu_threshold", "pocket_hu_low", "pocket_hu_high", "min_pocket_area",
        "border_erode_mm", "border_smooth_sigma", "tissue_hu_threshold",
        "pre_bbox_salt_pepper_kernel_size", "grey_region_hu_low", "grey_region_hu_cutoff",
    ]

    def __init__(self, cfg):
        self.p = SimpleNamespace(**cfg["preprocessing"])
        self.p.grey_region_hu_low = self.p.tissue_hu_threshold
        self.img_size = cfg["data"]["img_size"]

        out = cfg["paths"]["output_dir"]
        self.full_cache_dir = os.path.join(out, cfg["cache"]["full_dir"])
        self.image_cache_dir = os.path.join(out, cfg["cache"]["image_only_dir"])
        self._params_file = os.path.join(self.image_cache_dir, "_cache_params.json")
        os.makedirs(self.full_cache_dir, exist_ok=True)

        self._open_kernel = np.ones((3, 3), np.uint8)
        self.overlay_index = self._build_overlay_index(cfg["paths"]["overlay_dir"])
        self.no_grey_region_filenames = []
        self._mem_cache = {}
        self._image_mem_cache = {}
        self._times = []

        self._check_and_invalidate_cache_if_changed()
        print(f"Image-only cache ready. Cache dir: {self.image_cache_dir}")

    # ------------------------------------------------------------------
    # DICOM / HU
    # ------------------------------------------------------------------
    @staticmethod
    def read_hu(dicom_path):
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        return pixel_array * slope + intercept, ds

    def bone_mask_from_hu(self, hu):
        mask = (hu >= self.p.bone_hu_threshold).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)

    @staticmethod
    def hu_to_normalized(hu_array, low=0, high=100):
        windowed = np.clip(hu_array, low, high)
        return ((windowed - low) / (high - low)).astype(np.float32)

    @staticmethod
    def get_pixel_spacing_mm(ds):
        """Row/col pixel spacing in mm, averaged. Falls back to 1.0 mm/px."""
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is None:
            spacing = getattr(ds, "ImagerPixelSpacing", None)
        if spacing is None:
            return 1.0
        row_sp, col_sp = float(spacing[0]), float(spacing[1])
        return (row_sp + col_sp) / 2.0

    # ------------------------------------------------------------------
    # Skull strip / cleanup
    # ------------------------------------------------------------------
    def skull_strip(self, hu):
        """
        1. Remove every pixel with HU >= bone_hu_threshold.
        2. Keep everything else as tissue.
        3. Mask small DISCONNECTED pockets (area < min_pocket_area) whose HU lies
           fully inside [pocket_hu_low, pocket_hu_high].
        Returns clean_hu (removed pixels = -1000) and tissue_mask (uint8, 255 kept / 0 removed).
        """
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

    def erode_tissue_border(self, clean_hu, tissue_mask, ds):
        """
        1. Smooth the stripped-skull border (gaussian blur + re-threshold at 0.5,
           intersected with the original mask so smoothing can only REMOVE pixels).
        2. Erode a border band of border_erode_mm (converted with this scan's pixel spacing).
        """
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
        """Runs before the bbox. Removes only pixels the open+close cleanup deleted."""
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

    # ------------------------------------------------------------------
    # BBox + adaptive binarization
    # ------------------------------------------------------------------
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
        """HU high edge is FIXED. The HU low edge is adapted so the white % of the
        bbox lands in [floor_pct, target_pct). Returns (full_mask uint8 0/1, hu_low_used)."""
        p = self.p
        hu_high_fixed = p.hu_high_fixed
        target_pct = p.white_pct_target
        floor_pct = p.min_white_pct_floor
        fine_step = p.fine_step
        coarse_step = p.hu_step
        hu_low_cap = p.hu_low_cap
        hu_low_min = p.hu_low_min

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

    def extend_by_hu_similarity(self, guide_mask_u8, clean_hu, hu_low_used, extend_pixels, hu_margin):
        if extend_pixels <= 0:
            return guide_mask_u8.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * extend_pixels + 1, 2 * extend_pixels + 1))
        spatial_candidate = cv2.dilate(guide_mask_u8, kernel, iterations=1)
        intensity_candidate = ((clean_hu >= hu_low_used - hu_margin) &
                               (clean_hu <= self.p.hu_high_fixed)).astype(np.uint8)
        extended_mask = (spatial_candidate & intensity_candidate) | guide_mask_u8
        return extended_mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # Overlay (red annotation) helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_overlay_index(overlay_dir):
        index = {}
        if not os.path.isdir(overlay_dir):
            return index
        for f in os.listdir(overlay_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                index[os.path.splitext(f)[0]] = os.path.join(overlay_dir, f)
        return index

    def find_overlay_for_record(self, rec):
        dicom_stem = os.path.splitext(rec["filename"])[0]
        if dicom_stem in self.overlay_index:
            return self.overlay_index[dicom_stem]
        for stem, path in self.overlay_index.items():
            if dicom_stem in stem or stem in dicom_stem:
                return path
        return None

    def extract_red_mask(self, overlay_path, target_shape):
        p = self.p
        overlay_bgr = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
        if overlay_bgr is None:
            return np.zeros(target_shape, dtype=np.float32)
        hsv = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, tuple(p.red_hsv_lower_1), tuple(p.red_hsv_upper_1))
        mask2 = cv2.inRange(hsv, tuple(p.red_hsv_lower_2), tuple(p.red_hsv_upper_2))
        red_mask = cv2.bitwise_or(mask1, mask2)
        if red_mask.shape != target_shape:
            red_mask = cv2.resize(red_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (red_mask > 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Full multi-output preprocessing (image, guide mask, extended mask, seg gt)
    # ------------------------------------------------------------------
    def _preprocess_scan_uncached(self, rec, img_size, extend_pixels, hu_margin):
        p = self.p
        hu, ds = self.read_hu(rec["filepath"])
        clean_hu, tissue_mask = self.skull_strip(hu)
        clean_hu, tissue_mask = self.erode_tissue_border(clean_hu, tissue_mask, ds)
        clean_hu, tissue_mask, _ = self.remove_high_hu_in_grey(clean_hu, tissue_mask)
        clean_hu, tissue_mask, _ = self.remove_salt_pepper_from_tissue(clean_hu, tissue_mask)

        bbox = self.find_tissue_bbox(clean_hu)
        if bbox is None:
            if rec["filename"] not in self.no_grey_region_filenames:
                self.no_grey_region_filenames.append(rec["filename"])
            guide_mask_full = np.zeros(clean_hu.shape, dtype=np.uint8)
            hu_low_used = p.hu_low_start
        else:
            guide_mask_full, hu_low_used = self.adaptive_binarize_in_bbox(clean_hu, bbox)
            guide_mask_full = guide_mask_full * tissue_mask.clip(0, 1).astype(np.uint8)
            guide_mask_full = self.remove_salt_and_pepper(guide_mask_full, p.salt_pepper_kernel_size)

        dilated_mask_full = self.extend_by_hu_similarity(
            guide_mask_full, clean_hu, hu_low_used, extend_pixels, hu_margin)
        clean_hu_norm_full = self.hu_to_normalized(clean_hu, p.neighbour_hu_low, p.neighbour_hu_high)
        guide_mask_extended_full = clean_hu_norm_full * dilated_mask_full.astype(np.float32)

        image_norm_full = self.hu_to_normalized(clean_hu)

        if rec["class_name"] == "Bleeding":
            overlay_path = self.find_overlay_for_record(rec)
            seg_gt_full = self.extract_red_mask(overlay_path, clean_hu.shape) if overlay_path else \
                np.zeros(clean_hu.shape, dtype=np.float32)
        else:
            seg_gt_full = np.zeros(clean_hu.shape, dtype=np.float32)

        image_norm = cv2.resize(image_norm_full, (img_size, img_size), interpolation=cv2.INTER_AREA)
        guide_mask = cv2.resize(guide_mask_full.astype(np.float32), (img_size, img_size),
                                interpolation=cv2.INTER_NEAREST)
        guide_mask_extended = cv2.resize(guide_mask_extended_full, (img_size, img_size),
                                         interpolation=cv2.INTER_NEAREST)
        seg_gt_mask = cv2.resize(seg_gt_full, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        label = int(rec["label"]) if rec["label"] is not None else 0
        return image_norm, guide_mask, guide_mask_extended, seg_gt_mask, label

    def _cache_key(self, rec, img_size, extend_pixels, hu_margin):
        cfg_str = "_".join(f"{k}{v}" for k, v in sorted(vars(self.p).items()))
        raw = f"{rec['filepath']}_{img_size}_ext{extend_pixels}_margin{hu_margin}_{cfg_str}_v23"
        return hashlib.md5(raw.encode()).hexdigest()

    def preprocess_scan(self, rec, img_size=None, extend_pixels=None, hu_margin=None):
        img_size = self.img_size if img_size is None else img_size
        extend_pixels = self.p.extend_pixels if extend_pixels is None else extend_pixels
        hu_margin = self.p.hu_extend_margin if hu_margin is None else hu_margin

        key = self._cache_key(rec, img_size, extend_pixels, hu_margin)

        def _copies(t):
            return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in t)

        if key in self._mem_cache:
            return _copies(self._mem_cache[key])

        cache_path = os.path.join(self.full_cache_dir, key + ".npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                result = (data["image_norm"].astype(np.float32), data["guide_mask"].astype(np.float32),
                          data["guide_mask_extended"].astype(np.float32),
                          data["seg_gt_mask"].astype(np.float32), int(data["label"]))
                self._mem_cache[key] = result
                return _copies(result)
            except Exception as e:
                print(f"  [cache read error, recomputing] {rec['filename']}: {e}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        result = self._preprocess_scan_uncached(rec, img_size, extend_pixels, hu_margin)
        image_norm, guide_mask, guide_mask_extended, seg_gt_mask, label = result

        try:
            tmp_path = os.path.join(self.full_cache_dir, f"{key}.tmp{os.getpid()}.npz")
            np.savez_compressed(tmp_path, image_norm=image_norm, guide_mask=guide_mask,
                                guide_mask_extended=guide_mask_extended, seg_gt_mask=seg_gt_mask, label=label)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"  [cache write error, continuing uncached] {rec['filename']}: {e}")

        self._mem_cache[key] = result
        return _copies(result)

    def preprocess_scan_timed(self, rec, img_size=None, extend_pixels=None, hu_margin=None):
        t0 = time.perf_counter()
        result = self.preprocess_scan(rec, img_size, extend_pixels, hu_margin)
        self._times.append((rec["filename"], time.perf_counter() - t0))
        return result

    def print_preprocess_timing_summary(self):
        if not self._times:
            print("No timed preprocess_scan calls recorded yet.")
            return
        times = np.array([t for _, t in self._times])
        print(f"preprocess_scan calls: {len(times)}")
        print(f"  mean:   {times.mean()*1000:.1f} ms")
        print(f"  median: {np.median(times)*1000:.1f} ms")
        print(f"  min:    {times.min()*1000:.1f} ms")
        print(f"  max:    {times.max()*1000:.1f} ms")
        print("  slowest scans:")
        for fn, t in sorted(self._times, key=lambda x: -x[1])[:5]:
            print(f"    {fn}: {t*1000:.1f} ms")

    # ------------------------------------------------------------------
    # Image-only disk-persistent cache
    # ------------------------------------------------------------------
    def _current_param_fingerprint(self):
        values = {name: getattr(self.p, name, None) for name in self.CACHE_RELEVANT_PARAMS}
        raw = json.dumps(values, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest(), values

    def _check_and_invalidate_cache_if_changed(self):
        os.makedirs(self.image_cache_dir, exist_ok=True)
        current_fp, current_values = self._current_param_fingerprint()

        stored_fp = None
        if os.path.exists(self._params_file):
            try:
                with open(self._params_file) as f:
                    stored_fp = json.load(f).get("fingerprint")
            except Exception:
                stored_fp = None

        if stored_fp == current_fp:
            print(f"Params match cached fingerprint -- reusing existing cache at {self.image_cache_dir}.")
            return

        if stored_fp is None:
            print(f"No previous cache fingerprint found -- starting fresh at {self.image_cache_dir}.")
        else:
            print(f"Parameters CHANGED since last run -- clearing cache and reprocessing.\n"
                  f"  Current params: {current_values}")
            for entry in os.listdir(self.image_cache_dir):
                entry_path = os.path.join(self.image_cache_dir, entry)
                if entry_path == self._params_file:
                    continue
                try:
                    if os.path.isfile(entry_path):
                        os.remove(entry_path)
                except OSError as e:
                    print(f"  [warning] couldn't remove {entry_path}: {e}")

        with open(self._params_file, "w") as f:
            json.dump({"fingerprint": current_fp, "values": current_values}, f, indent=2)

    def _image_only_cache_key(self, rec, img_size):
        raw = f"{rec['filepath']}_{img_size}_imgonly_v4"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_clean_image_norm(self, rec, img_size=None):
        """Returns the preprocessed image (float32, [0,1], img_size x img_size). Cached in memory and on disk."""
        img_size = self.img_size if img_size is None else img_size
        key = self._image_only_cache_key(rec, img_size)

        if key in self._image_mem_cache:
            return self._image_mem_cache[key].copy()

        cache_path = os.path.join(self.image_cache_dir, key + ".npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                image_norm = data["image_norm"].astype(np.float32)
                self._image_mem_cache[key] = image_norm
                return image_norm.copy()
            except Exception as e:
                print(f"  [cache read error, recomputing] {rec['filename']}: {e}")

        hu, ds = self.read_hu(rec["filepath"])
        clean_hu, tissue_mask = self.skull_strip(hu)
        clean_hu, tissue_mask = self.erode_tissue_border(clean_hu, tissue_mask, ds)
        clean_hu, tissue_mask, _ = self.remove_high_hu_in_grey(clean_hu, tissue_mask)
        clean_hu, tissue_mask, _ = self.remove_salt_pepper_from_tissue(clean_hu, tissue_mask)
        image_norm_full = self.hu_to_normalized(clean_hu)
        image_norm = cv2.resize(image_norm_full, (img_size, img_size), interpolation=cv2.INTER_AREA)

        try:
            tmp_path = os.path.join(self.image_cache_dir, f"{key}.tmp{os.getpid()}.npz")
            np.savez_compressed(tmp_path, image_norm=image_norm)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"  [cache write error, continuing uncached] {rec['filename']}: {e}")

        self._image_mem_cache[key] = image_norm
        return image_norm.copy()

    def warm_image_only_cache(self, records, img_size=None, n_jobs=None):
        """Parallel cache warm-up over whatever record set you're about to use."""
        from joblib import Parallel, delayed

        img_size = self.img_size if img_size is None else img_size
        if n_jobs is None:
            n_jobs = min(os.cpu_count() or 4, 12)

        def _warm_one(rec):
            try:
                self.get_clean_image_norm(rec, img_size)
                return None
            except Exception as e:
                return f"{rec['filename']}: {e}"

        print(f"Warming image-only cache for {len(records)} scans using {n_jobs} processes...")
        errors = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_warm_one)(rec) for rec in records)
        n_errors = sum(1 for e in errors if e is not None)
        if n_errors:
            print(f"  {n_errors} scan(s) failed during warm-up:")
            for e in errors:
                if e is not None:
                    print(f"    {e}")
        print("Cache warm-up done.")