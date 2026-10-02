import numpy as np
import torch


def expected_age(logits, min_age):
    """Predicted age as the probability-weighted average of the age classes (Fig. 2a)."""
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(logits.shape[1], dtype=probs.dtype, device=logits.device)
    return (probs * classes).sum(1) + min_age


def mean_absolute_error(pred_ages, true_ages):
    return float(np.mean(np.abs(np.asarray(pred_ages) - np.asarray(true_ages))))


def systematic_bias(pred_ages, true_ages):
    """Mean BrainAGE of subjects younger than mu - sigma (SB-L) and older than mu + sigma (SB-R), Sec. 3.2.2."""
    pred_ages, true_ages = np.asarray(pred_ages), np.asarray(true_ages)
    brain_age_gap = pred_ages - true_ages
    mu, sigma = true_ages.mean(), true_ages.std()
    left, right = true_ages < mu - sigma, true_ages > mu + sigma
    sb_l = float(brain_age_gap[left].mean()) if left.any() else float("nan")
    sb_r = float(brain_age_gap[right].mean()) if right.any() else float("nan")
    return sb_l, sb_r


def ordinality(features, true_ages):
    """Ordinality score (Sec. 3.2.1).

    Pearson correlation between the Manhattan distances from the youngest age's feature centroid to
    every other centroid, and the corresponding age differences. When every age in the range is
    present, the age differences are the paper's C = {1, ..., c - 1}.
    """
    features, true_ages = np.asarray(features), np.asarray(true_ages)
    ages = np.unique(true_ages)
    if len(ages) < 3:
        return float("nan")

    centroids = np.stack([features[true_ages == age].mean(0) for age in ages])
    distances = np.abs(centroids[1:] - centroids[0]).sum(1)
    return float(np.corrcoef(distances, ages[1:] - ages[0])[0, 1])
