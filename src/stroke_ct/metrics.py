import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def binary_metrics(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def print_confusion(m, indent="  "):
    print(f"{indent}                 Pred Normal/Isch   Pred Bleeding")
    print(f"{indent}Actual Normal/Isch   TN={m['TN']:4d}            FP={m['FP']:4d}")
    print(f"{indent}Actual Bleeding      FN={m['FN']:4d}            TP={m['TP']:4d}")


# ---------------- segmentation losses / metrics ----------------
def dice_loss(pred_logits, target, smooth=1.0):
    p = torch.sigmoid(pred_logits).view(pred_logits.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def tversky_loss(pred_logits, target, alpha, beta, smooth=1.0):
    p = torch.sigmoid(pred_logits).view(pred_logits.size(0), -1)
    t = target.view(target.size(0), -1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)
    return (1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


@torch.no_grad()
def dice_iou_from_logits(seg_logit, seg_gt, threshold, smooth=1e-6):
    p = (torch.sigmoid(seg_logit) >= threshold)
    g = (seg_gt > 0.5)
    inter = (p & g).sum().item()
    union = (p | g).sum().item()
    dice = (2.0 * inter + smooth) / (p.sum().item() + g.sum().item() + smooth)
    iou = (inter + smooth) / (union + smooth)
    return dice, iou