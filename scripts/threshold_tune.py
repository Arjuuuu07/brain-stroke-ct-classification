"""Threshold sweep on the test probabilities saved by train.py.

The sweep is EXPLORATORY: picking a threshold by looking at test data makes the result optimistic.
The unbiased number is train.py's 'test@val_youden_thr' (threshold chosen on validation only).

Run: python scripts/threshold_tune.py --channels t2 --tag t2
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stroke_ct.config import apply_overrides, load_config, run_name
from stroke_ct.metrics import classification_metrics, show


def build_arg_parser():
    p = argparse.ArgumentParser(description="Classification threshold sweep")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--channels", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="threshold for the single check (default: the run's validation Youden threshold)")
    p.add_argument("--sweep-start", type=float, default=None)
    p.add_argument("--sweep-end", type=float, default=None)
    p.add_argument("--sweep-step", type=float, default=None)
    p.add_argument("--probs-path", type=str, default=None, help="override the .npz saved by train.py")
    p.add_argument("--no-plot", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    tc, tt, outs = cfg["training"], cfg["threshold_tune"], cfg["outputs"]
    apply_overrides(tc, args, ["channels", "backbone", "folds", "tag"])
    apply_overrides(tt, args, ["threshold", "sweep_start", "sweep_end", "sweep_step", "probs_path"])
    if args.no_plot:
        tt["plot"] = False

    out_dir = cfg["paths"]["output_dir"]
    name = run_name(cfg)
    probs_path = tt["probs_path"] or os.path.join(out_dir, outs["test_probs_npz"].format(name=name))
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"{probs_path} not found -- run train.py with the same --channels/--tag first.")

    data = np.load(probs_path)
    probs, labels = data["test_probs"], data["test_labels"]

    if tt["threshold"] is not None:
        thr_check = float(tt["threshold"])
    elif "val_threshold" in data.files:
        thr_check = float(data["val_threshold"])
    else:
        thr_check = 0.5

    # 1) single check
    show(f"RUN {name}", classification_metrics(labels, probs, thr_check))

    # 2) sweep
    thresholds = np.arange(tt["sweep_start"], tt["sweep_end"], tt["sweep_step"])
    rows = [classification_metrics(labels, probs, t) for t in thresholds]
    sweep_df = pd.DataFrame(rows).rename(columns={"thr": "threshold"}).drop(columns=["auc"])

    print(f"\n{'=' * 90}\nTHRESHOLD SWEEP ({tt['sweep_start']} -> {tt['sweep_end']})  [exploratory, on the test set]\n{'=' * 90}")
    print(sweep_df.to_string(index=False))

    best = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\nBest by F1: threshold={best['threshold']:.2f} precision={best['precision']:.4f} "
          f"recall={best['recall']:.4f} f1={best['f1']:.4f}")
    print("NOTE: this threshold was chosen on the test set; do not report it as an unbiased result.")

    sweep_csv = os.path.join(out_dir, outs["sweep_csv"].format(name=name))
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSweep table saved to: {sweep_csv}")

    # 3) plot
    if tt["plot"]:
        plt.figure(figsize=(9, 5))
        for col, label in [("precision", "Precision"), ("recall", "Recall"),
                           ("f1", "F1"), ("specificity", "Specificity")]:
            plt.plot(sweep_df["threshold"], sweep_df[col], label=label, marker="o")
        plt.axvline(thr_check, linestyle="--", alpha=0.6, label=f"checked ({thr_check:.3f})")
        plt.xlabel("Decision threshold")
        plt.ylabel("Score")
        plt.title(f"Metrics vs threshold (test set, exploratory) - {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(out_dir, outs["sweep_plot"].format(name=name))
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()
