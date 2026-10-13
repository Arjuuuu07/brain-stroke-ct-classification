"""Classification threshold sweep on the probabilities saved by train.py.
Run: python scripts/threshold_tune.py [--config configs/default.yaml] [--threshold 0.96] ..."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stroke_ct.config import apply_overrides, load_config
from stroke_ct.metrics import binary_metrics, print_confusion


def build_arg_parser():
    p = argparse.ArgumentParser(description="Classification threshold sweep")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--sweep-start", type=float, default=None)
    p.add_argument("--sweep-end", type=float, default=None)
    p.add_argument("--sweep-step", type=float, default=None)
    p.add_argument("--probs-path", type=str, default=None, help="Override path to the .npz saved by train.py")
    p.add_argument("--no-plot", action="store_true", help="Skip the matplotlib plot")
    return p


def confusion_at_threshold(threshold, probs, labels, verbose=True):
    preds = (probs >= threshold).astype(int)
    m = binary_metrics(labels, preds)

    if verbose:
        print(f"\n{'='*60}")
        print(f"THRESHOLD = {threshold:.2f}")
        print(f"{'='*60}")
        print_confusion(m, indent="")
        print(f"\nAccuracy    = {m['accuracy']:.4f}")
        print(f"Precision   = {m['precision']:.4f}")
        print(f"Recall      = {m['recall']:.4f} (sensitivity)")
        print(f"Specificity = {m['specificity']:.4f}")
        print(f"F1          = {m['f1']:.4f}")

    return {"threshold": threshold, **m}


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    tt = cfg["threshold_tune"]
    apply_overrides(tt, args, ["threshold", "sweep_start", "sweep_end", "sweep_step", "probs_path"])
    if args.no_plot:
        tt["plot"] = False

    out_dir = cfg["paths"]["output_dir"]
    probs_path = tt["probs_path"] or os.path.join(out_dir, cfg["outputs"]["test_probs_npz"])
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"{probs_path} not found -- run train.py first.")

    data = np.load(probs_path)
    test_probs_arr = data["test_probs"]
    test_labels_arr = data["test_labels"]

    # 1) Quick single check
    threshold_to_test = tt["threshold"]
    confusion_at_threshold(threshold_to_test, test_probs_arr, test_labels_arr)

    # 2) Full sweep
    sweep_thresholds = np.arange(tt["sweep_start"], tt["sweep_end"], tt["sweep_step"])
    sweep_results = [confusion_at_threshold(t, test_probs_arr, test_labels_arr, verbose=False)
                     for t in sweep_thresholds]
    sweep_df = pd.DataFrame(sweep_results)

    print(f"\n{'='*90}\nTHRESHOLD SWEEP ({tt['sweep_start']} -> {tt['sweep_end']})\n{'='*90}")
    print(sweep_df.to_string(index=False))

    best_by_f1 = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\nBest by F1: threshold={best_by_f1['threshold']:.2f} "
          f"precision={best_by_f1['precision']:.4f} "
          f"recall={best_by_f1['recall']:.4f} "
          f"f1={best_by_f1['f1']:.4f}")

    sweep_csv_path = os.path.join(out_dir, cfg["outputs"]["sweep_csv"])
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\nSweep table saved to: {sweep_csv_path}")

    # 3) Plot
    if tt["plot"]:
        plt.figure(figsize=(9, 5))
        for col, label in [("precision", "Precision"), ("recall", "Recall"),
                           ("f1", "F1"), ("specificity", "Specificity")]:
            plt.plot(sweep_df["threshold"], sweep_df[col], label=label, marker="o")
        plt.axvline(threshold_to_test, linestyle="--", alpha=0.6, label=f"current ({threshold_to_test})")
        plt.xlabel("Decision threshold")
        plt.ylabel("Score")
        plt.title("Classification metrics vs threshold (test set)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(out_dir, cfg["outputs"]["sweep_plot"])
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()