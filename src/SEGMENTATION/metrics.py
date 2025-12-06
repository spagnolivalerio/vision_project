import torch
import numpy as np

EPS = 1e-6


# Utility: convert predictions to class indices
def to_class(preds):
    """
    preds: logits from model, shape (B, C, H, W)
    returns predicted class mask: (B, H, W)
    """
    return preds.argmax(dim=1)


def multiclass_iou(preds, target, num_classes):
    preds = to_class(preds)

    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(np.nan) 
        else:
            ious.append(intersection / (union + EPS))

    mean_iou = np.nanmean(ious)
    return ious, mean_iou

def multiclass_dice(preds, target, num_classes):
    preds = to_class(preds)

    dice_scores = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (target == cls)

        tp = (pred_inds & target_inds).sum().item()
        fp = (pred_inds & ~target_inds).sum().item()
        fn = (~pred_inds & target_inds).sum().item()

        denom = (2 * tp + fp + fn + EPS)

        if denom == 0:
            dice_scores.append(np.nan)
        else:
            dice_scores.append((2 * tp) / denom)

    mean_dice = np.nanmean(dice_scores)
    return dice_scores, mean_dice


def multiclass_precision_recall(preds, target, num_classes):
    preds = to_class(preds)

    precisions = []
    recalls = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (target == cls)

        tp = (pred_inds & target_inds).sum().item()
        fp = (pred_inds & ~target_inds).sum().item()
        fn = (~pred_inds & target_inds).sum().item()

        if (tp + fp) == 0:
            precisions.append(np.nan)
        else:
            precisions.append(tp / (tp + fp + EPS))

        if (tp + fn) == 0:
            recalls.append(np.nan)
        else:
            recalls.append(tp / (tp + fn + EPS))

    return precisions, recalls

def pixel_accuracy(preds, target):
    preds = to_class(preds)
    correct = (preds == target).sum().item()
    total = preds.numel()
    return correct / (total + EPS)
