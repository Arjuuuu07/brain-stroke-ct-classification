"""5-fold CV classification, NO augmentation, on the 5-plane preprocessing.

Planes in the cache: 0 raw, 1 clean, 2 guide (binary), 3 ext (HU in dilated region), 4 guide_hu (HU in binarised region)

  --channels t2       clean, guide_hu, ext     (default, all HU-valued)
  --channels t2clean  clean, guide, ext        (binary guide mask in channel 2)
  --channels t1       clean, clean, clean      (control)

Pool: 500 bleeding + 500 non-bleeding -> every fold: 800 train (400+400) / 200 val (100+100).
Best fold = highest validation AUC (ties: lower val loss). Only that fold is evaluated on the test set.
Test = every scan outside the pool.

Run: python scripts/train.py --channels t2 --tag t2
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_ct.config import apply_overrides, channel_indices, load_config, run_name
from stroke_ct.datasets import PlanesDataset
from stroke_ct.manifest import build_file_manifest
from stroke_ct.metrics import classification_metrics, safe_auc, show, youden_threshold
from stroke_ct.models import ResNetClassifier
from stroke_ct.preprocessing import PLANE_NAMES, Preprocessor
from stroke_ct.seeding import seed_everything, seeded_generator, worker_init_fn


def build_arg_parser():
    p = argparse.ArgumentParser(description="5-fold CV classifier on cached preprocessing planes")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--channels", default=None, help="t1 | t2 | t2clean (see 'channels:' in the config)")
    p.add_argument("--backbone", default=None, choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--n-per-class-pool", type=int, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--exclude-by-filename", action="store_true",
                   help="build the test set by excluding the pool by filename only (old behaviour)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--warm-jobs", type=int, default=None)
    p.add_argument("--no-warm", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default=None)
    return p


def rid(r):
    return f"{r['class_name']}/{r['filename']}"


def make_pool_and_test(manifest, n_per_class, seed, exclude_by_filename=False):
    pool_all = [r for r in manifest if r["split"] == "train"]
    bleed = [r for r in pool_all if r["label"] == 1]
    non = [r for r in pool_all if r["label"] == 0]
    if len(bleed) < n_per_class or len(non) < n_per_class:
        raise ValueError(f"need {n_per_class}/class, have bleeding={len(bleed)}, non-bleeding={len(non)}")
    rng = np.random.default_rng(seed)
    pick = lambda recs: [recs[i] for i in rng.choice(len(recs), size=n_per_class, replace=False)]
    pool = pick(bleed) + pick(non)
    if exclude_by_filename:
        used = {r["filename"] for r in pool}
        test = [r for r in pool_all if r["filename"] not in used]
    else:
        used = {rid(r) for r in pool}
        test = [r for r in pool_all if rid(r) not in used]
    return pool, test


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device)
        logit = model(x)
        loss_sum += F.binary_cross_entropy_with_logits(
            logit, y.float().unsqueeze(1).to(device), reduction="sum").item()
        probs += torch.sigmoid(logit).cpu().numpy().ravel().tolist()
        labels += y.numpy().tolist()
        n += len(y)
    return np.array(probs), np.array(labels), loss_sum / max(n, 1)


def train_one_fold(fold, train_r, val_r, cfg, device, idx, prep):
    tc, seed = cfg["training"], cfg["seed"]
    seed_everything(seed + fold)
    pin = device.type == "cuda"
    train_loader = DataLoader(PlanesDataset(train_r, prep, idx, cfg), batch_size=tc["batch_size"], shuffle=True,
                              num_workers=tc["num_workers"], pin_memory=pin,
                              generator=seeded_generator(seed + fold),
                              worker_init_fn=worker_init_fn if tc["num_workers"] > 0 else None)
    val_loader = DataLoader(PlanesDataset(val_r, prep, idx, cfg), batch_size=tc["batch_size"], shuffle=False,
                            num_workers=tc["num_workers"], pin_memory=pin)

    model = ResNetClassifier(tc["backbone"], pretrained=tc["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    best_auc, best_loss, best_state, bad, epochs_run = -1.0, 1e9, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, tc["epochs"] + 1):
        epochs_run = epoch
        model.train()
        run = 0.0
        for x, y in tqdm(train_loader, desc=f"fold {fold} epoch {epoch}", leave=False):
            x, t = x.to(device), y.float().unsqueeze(1).to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), t)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
        vp, vy, vloss = predict(model, val_loader, device)
        vauc = safe_auc(vy, vp)
        print(f"  fold {fold} epoch {epoch:2d}  train_loss={run / len(train_r):.4f}  "
              f"val_loss={vloss:.4f}  val_auc={vauc:.4f}")
        if vauc > best_auc or (vauc == best_auc and vloss < best_loss):
            best_auc, best_loss, best_state, bad = vauc, vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"  fold {fold} early stop at epoch {epoch} (best val_auc={best_auc:.4f})")
                break
    minutes = (time.perf_counter() - t0) / 60
    print(f"  fold {fold} training time: {minutes:.1f} min")
    model.load_state_dict(best_state)
    return model, val_loader, best_state, best_auc, best_loss, epochs_run, minutes


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]
    apply_overrides(tc, args, ["channels", "backbone", "n_per_class_pool", "folds", "epochs", "patience",
                               "batch_size", "lr", "weight_decay", "exclude_by_filename", "num_workers",
                               "warm_jobs", "device", "tag"])
    if args.no_pretrained:
        tc["pretrained"] = False
    if args.no_warm:
        tc["warm_cache"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = cfg["seed"]
    device = torch.device(tc["device"]) if tc["device"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    idx = channel_indices(cfg)
    names = [PLANE_NAMES[i] for i in idx]
    print(f"device={device}  channels={tc['channels']}={names}  backbone={tc['backbone']}  "
          f"folds={tc['folds']}  (no augmentation)")

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    name = run_name(cfg)
    outs = cfg["outputs"]

    manifest = build_file_manifest(cfg)
    prep = Preprocessor(cfg)
    img_size = cfg["data"]["img_size"]

    pool, test_r = make_pool_and_test(manifest, tc["n_per_class_pool"], seed, tc["exclude_by_filename"])
    cnt = lambda rs: f"{len(rs)} (bleeding={sum(r['label'] == 1 for r in rs)}, non={sum(r['label'] == 0 for r in rs)})"
    print(f"CV pool {cnt(pool)} | held-out test {cnt(test_r)}  (test used once, best fold only)")

    labels = np.array([r["label"] for r in pool])
    skf = StratifiedKFold(n_splits=tc["folds"], shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(pool)), labels))
    n_tr = 2 * tc["n_per_class_pool"] * (tc["folds"] - 1) // tc["folds"]
    n_va = 2 * tc["n_per_class_pool"] // tc["folds"]
    for tr, va in splits:
        assert len(tr) == n_tr and labels[tr].sum() == n_tr // 2, "unexpected train split"
        assert len(va) == n_va and labels[va].sum() == n_va // 2, "unexpected val split"

    rows = [dict(id=rid(r), set="test", fold=-1, label=r["label"]) for r in test_r]
    for f, (tr, va) in enumerate(splits, start=1):
        rows += [dict(id=rid(pool[i]), set="train", fold=f, label=pool[i]["label"]) for i in tr]
        rows += [dict(id=rid(pool[i]), set="val", fold=f, label=pool[i]["label"]) for i in va]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, outs["split_csv"].format(name=name)), index=False)

    if tc["warm_cache"]:
        prep.warm_cache(pool + test_r, img_size, n_jobs=tc["warm_jobs"])

    chk = prep.get_planes(pool[0], img_size)[list(idx)]
    print(f"input check: shape={chk.shape}  per-channel min/max="
          f"{[(round(float(c.min()), 3), round(float(c.max()), 3)) for c in chk]}")

    # ---------------- train every fold, validation only ----------------
    fold_rows, fold_paths, fold_val_thr = [], [], []
    cv_t0 = time.perf_counter()
    for fold, (tr_pos, va_pos) in enumerate(splits, start=1):
        print(f"\n{'=' * 70}\nFOLD {fold}/{tc['folds']}\n{'=' * 70}")
        train_r, val_r = [pool[i] for i in tr_pos], [pool[i] for i in va_pos]
        print(f"train {cnt(train_r)} | val {cnt(val_r)}")

        model, val_loader, best_state, vauc, vloss, epochs_run, minutes = train_one_fold(
            fold, train_r, val_r, cfg, device, idx, prep)
        model_path = os.path.join(out_dir, outs["fold_model"].format(name=name, fold=fold))
        torch.save(best_state, model_path)
        fold_paths.append(model_path)

        vp, vy, _ = predict(model, val_loader, device)
        thr_val = youden_threshold(vy, vp)
        fold_val_thr.append(thr_val)
        m05 = classification_metrics(vy, vp, 0.5)
        mthr = classification_metrics(vy, vp, thr_val)
        show(f"FOLD {fold} VAL@0.5", m05)
        show(f"FOLD {fold} VAL@YOUDEN", mthr)
        fold_rows.append({"fold": fold, "val_auc": vauc, "val_loss": vloss, "val_thr_youden": thr_val,
                          "epochs_trained": epochs_run, "fold_time_min": minutes, "model_path": model_path,
                          **{f"val_{k}": v for k, v in m05.items() if k != "thr"}})

    print(f"\nTOTAL CV TIME: {(time.perf_counter() - cv_t0) / 60:.1f} min")
    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(out_dir, outs["folds_csv"].format(name=name)), index=False)

    print(f"\n{'=' * 70}\nVALIDATION SUMMARY (all folds)\n{'=' * 70}")
    print(df[["fold", "val_auc", "val_loss", "val_accuracy", "val_precision", "val_recall",
              "val_specificity", "val_f1", "epochs_trained"]].to_string(index=False))
    for c in ["val_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_f1"]:
        print(f"  {c:16s} mean={df[c].mean():.4f}  std={df[c].std(ddof=0):.4f}")

    # ---------------- best fold by validation AUC (ties: lower val loss) ----------------
    best = df.sort_values(["val_auc", "val_loss"], ascending=[False, True]).iloc[0]
    bf = int(best["fold"])
    thr_val = fold_val_thr[bf - 1]
    print(f"\nBEST FOLD by validation AUC: {bf}  (val_auc={best['val_auc']:.4f}, val_loss={best['val_loss']:.4f})")
    print(f"  model: {fold_paths[bf - 1]}")

    # ---------------- test set: evaluated ONCE, best fold only ----------------
    print(f"\n{'=' * 70}\nTEST SET (best fold {bf} only, evaluated once)\n{'=' * 70}")
    best_model = ResNetClassifier(tc["backbone"], pretrained=False).to(device)
    best_model.load_state_dict(torch.load(fold_paths[bf - 1], map_location=device))
    test_loader = DataLoader(PlanesDataset(test_r, prep, idx, cfg), batch_size=tc["batch_size"], shuffle=False,
                             num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    t0 = time.perf_counter()
    tp_, ty, _ = predict(best_model, test_loader, device)
    print(f"test inference: {time.perf_counter() - t0:.1f}s for {len(test_r)} scans")

    res = {"test@0.5": classification_metrics(ty, tp_, tc["decision_threshold"]),
           "test@val_youden_thr": classification_metrics(ty, tp_, thr_val)}
    for k, d in res.items():
        show(f"BEST FOLD {bf} {k.upper()}", d)
    pd.DataFrame([{"best_fold": bf, "split": k, **d} for k, d in res.items()]).to_csv(
        os.path.join(out_dir, outs["test_csv"].format(name=name)), index=False)
    np.savez(os.path.join(out_dir, outs["test_probs_npz"].format(name=name)),
             test_probs=tp_, test_labels=ty, best_fold=bf, val_threshold=thr_val)
    print(f"\n[cache] this process: hits={prep.stats['hit']} misses={prep.stats['miss']}")


if __name__ == "__main__":
    main()
