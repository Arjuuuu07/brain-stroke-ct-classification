"""Discover DICOM file paths + labels (NO pixel data loaded here)."""
import glob
import os
from pathlib import Path

import pandas as pd


def list_dicom_files(folder_path, valid_exts):
    valid = set(valid_exts)
    files = []
    for f in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid:
            files.append(f)
    return sorted(files)


def build_file_manifest(cfg, verbose=True):
    """Builds and returns the file manifest (paths + labels only, cheap)."""
    paths = cfg["paths"]
    class_labels = cfg["data"]["class_labels"]
    valid_exts = cfg["data"]["valid_exts"]
    file_manifest = []

    for class_name, folder_path in paths["folders"].items():
        if not os.path.isdir(folder_path):
            if verbose:
                print(f"Folder not found, skipping: {folder_path}")
            continue
        label = class_labels[class_name]
        files = list_dicom_files(folder_path, valid_exts)
        for f in files:
            file_manifest.append({
                "filepath": Path(f),
                "filename": os.path.basename(f),
                "class_name": class_name,
                "label": label,
                "split": "train",
            })
        if verbose:
            print(f"[{class_name}] found {len(files)} file(s), label={label}")

    if verbose:
        print(f"\nTotal train files in manifest: {len(file_manifest)}")

    # ---------------- EXTERNAL TEST MANIFEST ----------------
    ext_dir = paths["external_test_dicom_dir"]
    if os.path.isdir(ext_dir):
        labels_df = None
        if os.path.isfile(paths["external_test_labels_csv"]):
            labels_df = pd.read_csv(paths["external_test_labels_csv"])
            labels_df.columns = [c.strip().lower() for c in labels_df.columns]

        ext_files = list_dicom_files(ext_dir, valid_exts)
        for f in ext_files:
            fname = os.path.basename(f)
            label = None
            if labels_df is not None:
                key_col = next((c for c in labels_df.columns if "file" in c or "name" in c), labels_df.columns[0])
                label_col = next((c for c in labels_df.columns if "label" in c or "class" in c), labels_df.columns[-1])
                match = labels_df[labels_df[key_col].astype(str).str.contains(os.path.splitext(fname)[0], na=False)]
                if not match.empty:
                    label = match.iloc[0][label_col]

            file_manifest.append({
                "filepath": Path(f),
                "filename": fname,
                "class_name": "External_Test",
                "label": label,
                "split": "external_test",
            })
        if verbose:
            print(f"[External_Test] found {len(ext_files)} file(s)")
    else:
        if verbose:
            print(f"Folder not found, skipping: {ext_dir}")

    if verbose:
        print(f"Total files in manifest (train + external): {len(file_manifest)}")
    return file_manifest