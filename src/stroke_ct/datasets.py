import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess_image_only(prep, rec, img_size):
    """Delegates to the shared cache (Preprocessor.get_clean_image_norm)."""
    image_norm = prep.get_clean_image_norm(rec, img_size)
    label = int(rec["label"]) if rec.get("label") is not None else 0
    return image_norm, label


class Image3ChDataset(Dataset):
    """3-channel input: raw CT image duplicated x3, ImageNet-normalized.
    Returns a dummy zero seg_gt tensor purely to keep the training-loop unpacking unchanged."""

    def __init__(self, records, cfg, prep, augment=False):
        self.records = records
        self.prep = prep
        self.augment = augment
        self.img_size = cfg["data"]["img_size"]
        self.mean = np.array(cfg["data"]["imagenet_mean"], dtype=np.float32)
        self.std = np.array(cfg["data"]["imagenet_std"], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            image_norm, label = preprocess_image_only(self.prep, rec, self.img_size)
        except Exception as e:
            print(f"  [dataset error] {rec['filename']}: {e}")
            image_norm = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            label = 0

        if self.augment and np.random.rand() < 0.5:
            image_norm = np.fliplr(image_norm).copy()

        img_3ch = np.stack([image_norm, image_norm, image_norm], axis=-1)
        img_3ch = (img_3ch - self.mean) / self.std
        input_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1)).float()
        dummy_seg_gt = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        return input_tensor, label, dummy_seg_gt


class Image3ChSegDataset(Dataset):
    """Image half comes from the shared cache (identical to what train.py's classifier trained on).
    seg_gt (the bleeding mask) is scan-specific and computed once, then cached in memory."""

    def __init__(self, records, cfg, prep, augment=False):
        self.records = records
        self.prep = prep
        self.augment = augment
        self.img_size = cfg["data"]["img_size"]
        self.mean = np.array(cfg["data"]["imagenet_mean"], dtype=np.float32)
        self.std = np.array(cfg["data"]["imagenet_std"], dtype=np.float32)
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def _preprocess_seg(self, rec):
        k = f"{rec['filepath']}_{self.img_size}_seg_v1"
        if k in self._cache:
            return self._cache[k]

        prep = self.prep
        image_norm = prep.get_clean_image_norm(rec, self.img_size)

        if rec["class_name"] == "Bleeding":
            p = prep.find_overlay_for_record(rec)
            hu, ds = prep.read_hu(rec["filepath"])
            clean_hu, tissue_mask = prep.skull_strip(hu)
            clean_hu, tissue_mask = prep.erode_tissue_border(clean_hu, tissue_mask, ds)
            clean_hu, tissue_mask, _ = prep.remove_high_hu_in_grey(clean_hu, tissue_mask)
            seg_full = prep.extract_red_mask(p, clean_hu.shape) if p is not None else \
                np.zeros(clean_hu.shape, np.float32)
        else:
            seg_full = np.zeros((self.img_size, self.img_size), np.float32)

        seg_gt = cv2.resize(seg_full, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        self._cache[k] = (image_norm, seg_gt)
        return image_norm, seg_gt

    def __getitem__(self, i):
        rec = self.records[i]
        try:
            img, seg = self._preprocess_seg(rec)
        except Exception as e:
            print(f"  [seg dataset err] {rec['filename']}: {e}")
            img = np.zeros((self.img_size, self.img_size), np.float32)
            seg = np.zeros((self.img_size, self.img_size), np.float32)

        if self.augment and np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            seg = np.fliplr(seg).copy()

        c = np.stack([img, img, img], -1)
        c = (c - self.mean) / self.std
        x = torch.from_numpy(c.transpose(2, 0, 1)).float()
        s = torch.from_numpy(seg).float().unsqueeze(0)
        label = int(rec["label"]) if rec.get("label") is not None else 0
        return x, label, s