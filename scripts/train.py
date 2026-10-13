"""5-fold CV classification training (3-channel, image-only).
Run: python scripts/train.py [--config configs/default.yaml] [--epochs 15] [--batch-size 8] ..."""
import argparse
import copy
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from thop import profile as thop_profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_ct.config import apply_overrides, load_config
from stroke_ct.datasets import Image3ChDataset
from stroke_ct.manifest import build_file_manifest
from stroke_ct.metrics import binary_metrics, print_confusion
from stroke_ct.models import Image3ChResNet
from stroke_ct.preprocessing import Preprocessor
from stroke_ct.seeding import seed_everything, seeded_generator, worker_init_fn


def build_arg_parser():
    p = argparse.ArgumentParser(description="3-channel image-only classifier, 5-fold CV")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--n-per-class-pool", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--recall-weight", type=float, default=None)
    p.add_argument("--precision-weight", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    return p


def predict(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            class_logit, _ = model(imgs, seg=False)
            probs = torch.sigmoid(class_logit).cpu().numpy().flatten()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().tolist())
    return probs_all, labels_all


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]
    apply_overrides(tc, args, ["n_per_class_pool", "batch_size", "epochs", "patience", "lr", "folds",
                               "num_workers", "recall_weight", "precision_weight", "device"])

    seed = cfg["seed"]
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    N_PER_CLASS_POOL = tc["n_per_class_pool"]
    BATCH_SIZE = tc["batch_size"]
    NUM_EPOCHS = tc["epochs"]
    EARLY_STOP_PATIENCE = tc["patience"]
    LEARNING_RATE = tc["lr"]
    N_FOLDS = tc["folds"]
    NUM_WORKERS = tc["num_workers"]
    RECALL_WEIGHT = tc["recall_weight"]
    PRECISION_WEIGHT = tc["precision_weight"]
    THRESH = tc["decision_threshold"]
    DEVICE = torch.device(tc["device"]) if tc["device"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    seed_everything(seed)
    print(f"Seeded everything with seed={seed}")

    file_manifest = build_file_manifest(cfg)
    prep = Preprocessor(cfg)
    img_size = cfg["data"]["img_size"]
    w_init = partial(worker_init_fn, seed=seed) if NUM_WORKERS > 0 else None

    # ---------------- balanced pool + fixed held-out remaining test set ----------------
    train_pool_all = [r for r in file_manifest if r["split"] == "train"]
    bleed_records = [r for r in train_pool_all if r["label"] == 1]
    nonbleed_records = [r for r in train_pool_all if r["label"] == 0]

    if len(bleed_records) < N_PER_CLASS_POOL or len(nonbleed_records) < N_PER_CLASS_POOL:
        raise ValueError(f"Not enough records for {N_PER_CLASS_POOL}/class pool.")

    rng = np.random.default_rng(seed)
    bleed_idx = rng.choice(len(bleed_records), size=N_PER_CLASS_POOL, replace=False)
    nonbleed_idx = rng.choice(len(nonbleed_records), size=N_PER_CLASS_POOL, replace=False)

    cv_pool = [bleed_records[i] for i in bleed_idx] + [nonbleed_records[i] for i in nonbleed_idx]
    pool_filenames = {r["filename"] for r in cv_pool}
    remaining_test_set = [r for r in train_pool_all if r["filename"] not in pool_filenames]

    print(f"CV pool: {len(cv_pool)} scans ({N_PER_CLASS_POOL} bleeding + {N_PER_CLASS_POOL} non-bleeding)")
    print(f"Held-out remaining test set: {len(remaining_test_set)} scans (NOT touched until the very end)")

    pool_labels = np.array([r["label"] for r in cv_pool])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_results = []
    cv_total_start = time.perf_counter()

    for fold_idx, (train_pos, val_pos) in enumerate(skf.split(np.zeros(len(cv_pool)), pool_labels), start=1):
        print(f"\n{'='*60}\nFOLD {fold_idx}/{N_FOLDS} (classification, 3-channel image-only)\n{'='*60}")

        fold_train_records = [cv_pool[i] for i in train_pos]
        fold_val_records = [cv_pool[i] for i in val_pos]
        print(f"Fold train: {len(fold_train_records)}, fold val: {len(fold_val_records)}")

        train_dataset = Image3ChDataset(fold_train_records, cfg, prep, augment=tc["augment"])
        val_dataset = Image3ChDataset(fold_val_records, cfg, prep, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  generator=seeded_generator(seed + fold_idx),
                                  worker_init_fn=w_init)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        model = Image3ChResNet().to(DEVICE)

        if fold_idx == 1:
            dummy_input = torch.randn(1, 3, img_size, img_size).to(DEVICE)
            macs, n_params = thop_profile(model, inputs=(dummy_input, False), verbose=False)
            print(f"  Model complexity (classification-only forward path): "
                  f"{macs/1e9:.3f} GFLOPs (MACs={macs/1e6:.1f}M), {n_params/1e6:.2f}M parameters")
            for module in model.modules():
                if hasattr(module, "total_ops"):
                    del module.total_ops
                if hasattr(module, "total_params"):
                    del module.total_params

        class_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=tc["weight_decay"])

        best_val_auc = -1.0
        best_state = None
        patience_counter = 0
        stopped_early_at = None

        fold_start_time = time.perf_counter()
        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start_time = time.perf_counter()
            model.train()
            running_loss = 0.0
            for imgs, labels, _ in tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}", leave=False):
                imgs = imgs.to(DEVICE)
                labels_t = labels.float().unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                class_logit, _ = model(imgs, seg=False)
                loss = class_criterion(class_logit, labels_t)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            train_loss = running_loss / len(train_dataset)

            val_probs, val_labels_list = predict(model, val_loader, DEVICE)
            val_auc = roc_auc_score(val_labels_list, val_probs) if len(set(val_labels_list)) > 1 else 0.5
            epoch_elapsed = time.perf_counter() - epoch_start_time
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")
            print(f"    (epoch took {epoch_elapsed:.1f}s)")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    stopped_early_at = epoch
                    print(f"  Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        epochs_trained = stopped_early_at if stopped_early_at is not None else NUM_EPOCHS

        val_probs, val_labels_list = predict(model, val_loader, DEVICE)
        val_preds = (np.array(val_probs) >= THRESH).astype(int)
        val_labels_arr = np.array(val_labels_list)
        vm = binary_metrics(val_labels_arr, val_preds)

        print(f"\nFold {fold_idx} VALIDATION CONFUSION MATRIX (rows=actual, cols=predicted):")
        print_confusion(vm)
        print(f"\nFold {fold_idx} VALIDATION METRICS:")
        print(f"  Accuracy    = {vm['accuracy']:.4f}")
        print(f"  AUC         = {best_val_auc:.4f}")
        print(f"  Precision   = {vm['precision']:.4f}")
        print(f"  Recall      = {vm['recall']:.4f}")
        print(f"  Specificity = {vm['specificity']:.4f}")
        print(f"  F1          = {vm['f1']:.4f}")
        print(f"  ({epochs_trained} epochs, {'early stopped' if stopped_early_at else 'full ceiling'})")

        fold_elapsed = time.perf_counter() - fold_start_time
        print(f"  Fold {fold_idx} total time: {fold_elapsed/60:.2f} min "
              f"({epochs_trained} epochs, {fold_elapsed/epochs_trained:.1f}s/epoch avg)")

        model_path = os.path.join(out_dir, cfg["outputs"]["fold_model_pattern"].format(fold=fold_idx))
        torch.save(model.state_dict(), model_path)

        fold_results.append({
            "fold": fold_idx,
            "epochs_trained": epochs_trained,
            "early_stopped": stopped_early_at is not None,
            "val_accuracy": vm["accuracy"],
            "val_auc": best_val_auc,
            "val_precision": vm["precision"],
            "val_recall": vm["recall"],
            "val_specificity": vm["specificity"],
            "val_f1": vm["f1"],
            "val_TP": vm["TP"], "val_TN": vm["TN"], "val_FP": vm["FP"], "val_FN": vm["FN"],
            "model_path": model_path,
            "fold_time_min": fold_elapsed / 60,
        })

    cv_total_elapsed = time.perf_counter() - cv_total_start
    print(f"\n{'='*90}\nTOTAL {N_FOLDS}-FOLD CV WALL-CLOCK TIME: {cv_total_elapsed/60:.2f} min\n{'='*90}")

    results_df = pd.DataFrame(fold_results)
    print(f"\n{'='*90}\nCLASSIFICATION CV SUMMARY (VALIDATION METRICS ONLY, ALL FOLDS)\n{'='*90}")
    print(results_df.to_string(index=False))

    results_df["combined_score"] = (
        RECALL_WEIGHT * results_df["val_recall"] + PRECISION_WEIGHT * results_df["val_precision"]
    )
    best_fold_row = results_df.sort_values(
        by=["combined_score", "val_f1", "val_recall", "val_auc"],
        ascending=[False, False, False, False]
    ).iloc[0]

    BEST_FOLD_MODEL_PATH = best_fold_row["model_path"]
    print(f"\nBest fold (by VALIDATION combined recall+precision score, "
          f"weights: recall={RECALL_WEIGHT}, precision={PRECISION_WEIGHT}): {int(best_fold_row['fold'])}")
    print(f"  combined_score={best_fold_row['combined_score']:.4f}  "
          f"val_recall={best_fold_row['val_recall']:.4f}  val_precision={best_fold_row['val_precision']:.4f}  "
          f"val_f1={best_fold_row['val_f1']:.4f}  val_auc={best_fold_row['val_auc']:.4f}  "
          f"val_accuracy={best_fold_row['val_accuracy']:.4f}")
    print(f"  -> {BEST_FOLD_MODEL_PATH}")

    summary_path = os.path.join(out_dir, cfg["outputs"]["cv_summary_csv"])
    results_df.to_csv(summary_path, index=False)
    print(f"\nValidation-only summary saved to: {summary_path}")

    # ---------------- FINAL TEST-SET EVALUATION, ONCE, BEST FOLD ONLY ----------------
    print(f"\n{'='*90}\nFINAL TEST-SET EVALUATION (single run, best fold only, "
          f"selected purely from validation)\n{'='*90}")

    test_dataset = Image3ChDataset(remaining_test_set, cfg, prep, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    test_eval_start = time.perf_counter()
    best_model = Image3ChResNet().to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_FOLD_MODEL_PATH, map_location=DEVICE))
    test_probs, test_labels_list = predict(best_model, test_loader, DEVICE)

    test_preds = (np.array(test_probs) >= THRESH).astype(int)
    test_labels_arr = np.array(test_labels_list)
    test_eval_elapsed = time.perf_counter() - test_eval_start
    print(f"\nTest-set inference took {test_eval_elapsed:.1f}s for {len(remaining_test_set)} scans "
          f"({test_eval_elapsed/len(remaining_test_set)*1000:.1f} ms/scan avg)")

    tm = binary_metrics(test_labels_arr, test_preds)
    test_auc = roc_auc_score(test_labels_arr, test_probs) if len(set(test_labels_list)) > 1 else 0.5

    print(f"\nTEST CONFUSION MATRIX (rows=actual, cols=predicted):")
    print_confusion(tm)
    print(f"\nFINAL TEST METRICS (best fold = {int(best_fold_row['fold'])}, chosen by validation only):")
    print(f"  Accuracy    = {tm['accuracy']:.4f}")
    print(f"  AUC         = {test_auc:.4f}")
    print(f"  Precision   = {tm['precision']:.4f}")
    print(f"  Recall      = {tm['recall']:.4f}  (sensitivity -- fraction of actual bleeds caught)")
    print(f"  Specificity = {tm['specificity']:.4f}  (fraction of actual non-bleeds correctly cleared)")
    print(f"  F1          = {tm['f1']:.4f}")

    test_summary = {
        "best_fold": int(best_fold_row["fold"]),
        "test_accuracy": tm["accuracy"], "test_auc": test_auc,
        "test_precision": tm["precision"], "test_recall": tm["recall"],
        "test_specificity": tm["specificity"], "test_f1": tm["f1"],
        "test_TP": tm["TP"], "test_TN": tm["TN"], "test_FP": tm["FP"], "test_FN": tm["FN"],
        "test_eval_time_sec": test_eval_elapsed,
        "cv_total_time_min": cv_total_elapsed / 60,
    }
    test_summary_path = os.path.join(out_dir, cfg["outputs"]["final_test_csv"])
    pd.DataFrame([test_summary]).to_csv(test_summary_path, index=False)
    print(f"\nFinal test summary saved to: {test_summary_path}")

    probs_path = os.path.join(out_dir, cfg["outputs"]["test_probs_npz"])
    np.savez(probs_path, test_probs=np.array(test_probs), test_labels=test_labels_arr)
    print(f"Raw test-set probabilities/labels saved to: {probs_path}")


if __name__ == "__main__":
    main()