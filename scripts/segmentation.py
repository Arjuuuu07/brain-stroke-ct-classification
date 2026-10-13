"""Segmentation fine-tuning seeded from the BEST fold's classifier weights.
Bleeding-only scans, U-Net decoder on the same ResNet18 encoder, Dice-dominant loss.
Requires train.py to have run first (cv summary csv + fold checkpoints).
Run: python scripts/segmentation.py [--config configs/default.yaml] [--epochs 70] ..."""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_ct.config import apply_overrides, load_config
from stroke_ct.datasets import Image3ChSegDataset
from stroke_ct.manifest import build_file_manifest
from stroke_ct.metrics import dice_iou_from_logits, dice_loss, tversky_loss
from stroke_ct.models import Image3ChSegResNet, set_train_mode
from stroke_ct.preprocessing import Preprocessor
from stroke_ct.seeding import seed_everything, seeded_generator


def build_arg_parser():
    p = argparse.ArgumentParser(description="Segmentation fine-tuning")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--bce-weight", type=float, default=None)
    p.add_argument("--dice-weight", type=float, default=None)
    p.add_argument("--use-tversky", action="store_true", help="Use Tversky loss instead of Dice")
    p.add_argument("--tversky-alpha", type=float, default=None, help="FP weight")
    p.add_argument("--tversky-beta", type=float, default=None, help="FN weight (higher = recall-biased)")
    p.add_argument("--lr-encoder", type=float, default=None)
    p.add_argument("--lr-decoder", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--freeze-encoder", action="store_true", help="Train decoder only")
    p.add_argument("--no-keep-bn-eval", action="store_true",
                   help="Let encoder BatchNorm update during training (default: kept frozen in eval)")
    p.add_argument("--encoder-checkpoint", type=str, default=None)
    p.add_argument("--recall-weight", type=float, default=None,
                   help="Must match training.recall_weight for the best-fold pick to agree")
    p.add_argument("--precision-weight", type=float, default=None,
                   help="Must match training.precision_weight for the best-fold pick to agree")
    p.add_argument("--device", type=str, default=None)
    return p


def resolve_encoder_checkpoint(cfg):
    """Finds the best fold checkpoint by re-applying train.py's best-fold selection to its cv summary csv."""
    sc, tc = cfg["segmentation"], cfg["training"]
    if sc["encoder_checkpoint"]:
        return sc["encoder_checkpoint"]

    summary_path = os.path.join(cfg["paths"]["output_dir"], cfg["outputs"]["cv_summary_csv"])
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found. Run train.py first, or pass --encoder-checkpoint explicitly.")

    df = pd.read_csv(summary_path)
    df["combined_score"] = tc["recall_weight"] * df["val_recall"] + tc["precision_weight"] * df["val_precision"]
    best_row = df.sort_values(
        by=["combined_score", "val_f1", "val_recall", "val_auc"],
        ascending=[False, False, False, False]
    ).iloc[0]

    model_path = best_row["model_path"]
    print(f"Resolved best fold from {summary_path}: fold {int(best_row['fold'])} "
          f"(combined_score={best_row['combined_score']:.4f}) -> {model_path}")
    return model_path


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    sc, tc = cfg["segmentation"], cfg["training"]
    apply_overrides(sc, args, ["batch_size", "epochs", "patience", "val_fraction", "threshold", "bce_weight",
                               "dice_weight", "use_tversky", "tversky_alpha", "tversky_beta", "lr_encoder",
                               "lr_decoder", "weight_decay", "freeze_encoder", "encoder_checkpoint", "device"])
    apply_overrides(tc, args, ["recall_weight", "precision_weight"])
    if args.no_keep_bn_eval:
        sc["keep_bn_eval"] = False

    seed = cfg["seed"]
    DEVICE = torch.device(sc["device"]) if sc["device"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    seed_everything(seed)
    print(f"Seeded everything with seed={seed}")

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    file_manifest = build_file_manifest(cfg)
    prep = Preprocessor(cfg)

    encoder_checkpoint = resolve_encoder_checkpoint(cfg)
    print(f"Fine-tuning from encoder checkpoint: {encoder_checkpoint}")

    seg_output_dir = os.path.join(out_dir, cfg["outputs"]["seg_dir"])
    seg_model_dir = os.path.join(seg_output_dir, "models")
    os.makedirs(seg_model_dir, exist_ok=True)

    if sc["use_tversky"]:
        a, b = sc["tversky_alpha"], sc["tversky_beta"]
        region_loss_fn = lambda logit, target: tversky_loss(logit, target, a, b)
        print(f"  Region loss = Tversky(alpha={a:.2f},beta={b:.2f})")
    else:
        region_loss_fn = dice_loss
        print("  Region loss = Dice")
    print(f"  Loss weights: BCE={sc['bce_weight']}, region={sc['dice_weight']}")
    print(f"  LRs: encoder={sc['lr_encoder']}, decoder={sc['lr_decoder']}")

    # ---------------- data: Bleeding-only, single split ----------------
    seg_records_all = [r for r in file_manifest
                       if r["class_name"] == "Bleeding" and r["split"] == "train"]
    print(f"Bleeding scans available for segmentation: {len(seg_records_all)}")

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(seg_records_all))
    n_val = int(len(seg_records_all) * sc["val_fraction"])
    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]

    seg_train_records = [seg_records_all[i] for i in train_idx]
    seg_val_records = [seg_records_all[i] for i in val_idx]
    print(f"Seg train: {len(seg_train_records)}, seg val: {len(seg_val_records)}")

    seg_train_dataset = Image3ChSegDataset(seg_train_records, cfg, prep, augment=sc["augment"])
    seg_val_dataset = Image3ChSegDataset(seg_val_records, cfg, prep, augment=False)

    seg_train_loader = DataLoader(seg_train_dataset, batch_size=sc["batch_size"], shuffle=True,
                                  num_workers=sc["num_workers"], pin_memory=True,
                                  generator=seeded_generator(seed))
    seg_val_loader = DataLoader(seg_val_dataset, batch_size=sc["batch_size"], shuffle=False,
                                num_workers=sc["num_workers"], pin_memory=True)

    # ---------------- load best fold's classifier weights into the encoder ----------------
    model = Image3ChSegResNet().to(DEVICE)
    cls_sd = torch.load(encoder_checkpoint, map_location=DEVICE)
    model_sd = model.state_dict()
    kept, dropped = 0, 0
    for k, v in cls_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            kept += 1
        else:
            dropped += 1
    model.load_state_dict(model_sd)
    print(f"  Loaded {kept} encoder/classifier tensors from checkpoint "
          f"(skipped {dropped} decoder-only tensors, fresh init)")

    if sc["freeze_encoder"]:
        for mod in (model.stem, model.maxpool, model.layer1, model.layer2,
                    model.layer3, model.layer4, model.classifier):
            for p in mod.parameters():
                p.requires_grad = False
        print("  Encoder + classifier frozen; training decoder only.")

    # ---------------- optimizer: differential LR ----------------
    encoder_params = (list(model.stem.parameters()) + list(model.maxpool.parameters())
                      + list(model.layer1.parameters()) + list(model.layer2.parameters())
                      + list(model.layer3.parameters()) + list(model.layer4.parameters())
                      + list(model.classifier.parameters()))
    decoder_params = (list(model.up4.parameters()) + list(model.dec4.parameters())
                      + list(model.up3.parameters()) + list(model.dec3.parameters())
                      + list(model.up2.parameters()) + list(model.dec2.parameters())
                      + list(model.up1.parameters()) + list(model.dec1.parameters())
                      + list(model.seg_head.parameters()))

    param_groups = [
        {"params": [p for p in encoder_params if p.requires_grad], "lr": sc["lr_encoder"]},
        {"params": [p for p in decoder_params if p.requires_grad], "lr": sc["lr_decoder"]},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=sc["weight_decay"])

    cls_criterion = nn.BCEWithLogitsLoss()
    seg_criterion = nn.BCEWithLogitsLoss()

    # ---------------- training loop ----------------
    best_val_dice = -1.0
    best_state = None
    patience_counter = 0
    stopped_early_at = None
    train_start = time.perf_counter()

    for epoch in range(1, sc["epochs"] + 1):
        ep_start = time.perf_counter()

        set_train_mode(model, sc["keep_bn_eval"])
        running_loss, running_bce, running_region, n_seen = 0.0, 0.0, 0.0, 0

        for imgs, labels, seg_gt in tqdm(seg_train_loader, desc=f"Seg Epoch {epoch}", leave=False):
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            seg_gt = seg_gt.to(DEVICE)

            optimizer.zero_grad()
            cls_logit, seg_logit = model(imgs, seg=True)

            cls_loss = cls_criterion(cls_logit, labels)
            bce_seg = seg_criterion(seg_logit, seg_gt)
            reg_seg = region_loss_fn(seg_logit, seg_gt)

            loss = cls_loss + sc["bce_weight"] * bce_seg + sc["dice_weight"] * reg_seg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=sc["grad_clip"])
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_bce += bce_seg.item() * imgs.size(0)
            running_region += reg_seg.item() * imgs.size(0)
            n_seen += imgs.size(0)

        train_loss = running_loss / max(1, n_seen)
        train_bce = running_bce / max(1, n_seen)
        train_region = running_region / max(1, n_seen)

        # ---- validation ----
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for imgs, _, seg_gt in seg_val_loader:
                imgs, seg_gt = imgs.to(DEVICE), seg_gt.to(DEVICE)
                _, seg_logit = model(imgs, seg=True)
                for b in range(imgs.size(0)):
                    d, i = dice_iou_from_logits(seg_logit[b:b+1], seg_gt[b:b+1], sc["threshold"])
                    dices.append(d)
                    ious.append(i)

        val_dice = float(np.mean(dices)) if dices else 0.0
        val_iou = float(np.mean(ious)) if ious else 0.0
        ep_elapsed = time.perf_counter() - ep_start

        print(f"  Epoch {epoch}: loss={train_loss:.4f}  "
              f"(bce={train_bce:.4f}, region={train_region:.4f})  "
              f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  ({ep_elapsed:.1f}s)")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= sc["patience"]:
                stopped_early_at = epoch
                print(f"  Early stopping at epoch {epoch} (best val_dice={best_val_dice:.4f})")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    total_elapsed = time.perf_counter() - train_start
    seg_model_path = os.path.join(seg_model_dir, cfg["outputs"]["seg_model"])
    torch.save(model.state_dict(), seg_model_path)

    print(f"\nSegmentation training done.")
    print(f"  Best val_dice = {best_val_dice:.4f}")
    print(f"  Epochs run    = {stopped_early_at or sc['epochs']}")
    print(f"  Total time    = {total_elapsed/60:.2f} min")
    print(f"  Saved seg model -> {seg_model_path}")


if __name__ == "__main__":
    main()