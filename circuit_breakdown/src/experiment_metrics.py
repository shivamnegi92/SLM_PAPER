"""Baseline-aware intervention outcomes and paired statistical summaries."""
from __future__ import annotations

from math import comb
from statistics import NormalDist

import numpy as np


def binary_flags(values):
    flags = np.asarray(values, dtype=np.float64)
    if flags.ndim != 1 or flags.size == 0 or not np.isin(flags, [0, 1]).all():
        raise ValueError("Expected a nonempty one-dimensional array of binary outcomes")
    return flags


def rate_ci(flags, confidence=0.95):
    """Wilson score interval, including all-zero/all-one samples."""
    flags = binary_flags(flags)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    quantile = NormalDist().inv_cdf((1 + confidence) / 2)
    mean = float(flags.mean())
    correction = quantile ** 2 / flags.size
    center = (mean + correction / 2) / (1 + correction)
    radius = quantile * np.sqrt(
        mean * (1 - mean) / flags.size + quantile ** 2 / (4 * flags.size ** 2)
    ) / (1 + correction)
    return mean, (max(0.0, float(center - radius)), min(1.0, float(center + radius)))


def paired_delta_ci(baseline, edited):
    """Approximate 95% paired bounds using Bonferroni gain/loss Wilson intervals."""
    baseline = binary_flags(baseline)
    edited = binary_flags(edited)
    if baseline.shape != edited.shape:
        raise ValueError("Paired outcomes must have identical lengths and ordering")
    _, gain_ci = rate_ci((edited > baseline).astype(int), confidence=0.975)
    _, loss_ci = rate_ci((edited < baseline).astype(int), confidence=0.975)
    return float((edited - baseline).mean()), (
        gain_ci[0] - loss_ci[1], gain_ci[1] - loss_ci[0]
    )


def mcnemar(first, second):
    first, second = binary_flags(first), binary_flags(second)
    if first.shape != second.shape:
        raise ValueError("Paired outcomes must have identical lengths")
    first_only = int(((first == 1) & (second == 0)).sum())
    second_only = int(((first == 0) & (second == 1)).sum())
    total = first_only + second_only
    probability = (min(1.0, 2 * sum(comb(total, count)
                   for count in range(min(first_only, second_only) + 1)) / (2 ** total))
                   if total else 1.0)
    return first_only, second_only, probability


def prediction_outcome(clean_target, corrupt_target, baseline_clean, baseline_corrupt,
                       steered_corrupt, positive_clean, negative_clean):
    if clean_target == corrupt_target:
        raise ValueError("Counterfactual targets must differ")
    clean_correct = baseline_clean == clean_target
    target_eligible = baseline_corrupt != clean_target
    return {
        "clean_target_id": clean_target, "corrupt_target_id": corrupt_target,
        "baseline_clean_prediction": baseline_clean,
        "baseline_corrupt_prediction": baseline_corrupt,
        "steered_prediction": steered_corrupt,
        "positive_clean_prediction": positive_clean,
        "negative_clean_prediction": negative_clean,
        "baseline_clean_correct": int(clean_correct),
        "baseline_corrupt_correct": int(baseline_corrupt == corrupt_target),
        "baseline_is_target": int(not target_eligible),
        "steer": int(steered_corrupt == clean_target),
        "new_target_eligible": int(target_eligible),
        "new_target_success": int(target_eligible and steered_corrupt == clean_target),
        "pred_is_corrupt": int(steered_corrupt == corrupt_target),
        "third_token": int(steered_corrupt not in (clean_target, corrupt_target)),
        "same_sign_error": int(positive_clean != clean_target),
        "same_sign_damage": int(clean_correct and positive_clean != clean_target),
        "negative_edit_error": int(negative_clean != clean_target),
        "negative_edit_disruption": int(clean_correct and negative_clean != clean_target),
    }


def summarize_predictions(records):
    if not records:
        raise ValueError("No prediction records to summarize")
    correct = [record for record in records if record["baseline_clean_correct"]]
    eligible = [record for record in records if record["new_target_eligible"]]
    summary = {"n": len(records), "n_baseline_clean_correct": len(correct),
               "n_new_target_eligible": len(eligible)}
    metrics = {
        "steer": (records, "steer"),
        "baseline_clean_acc": (records, "baseline_clean_correct"),
        "baseline_corrupt_acc": (records, "baseline_corrupt_correct"),
        "third_token_rate": (records, "third_token"),
        "negative_edit_error": (records, "negative_edit_error"),
        "same_sign_damage": (correct, "same_sign_damage"),
        "negative_edit_disruption": (correct, "negative_edit_disruption"),
        "new_target_rate": (eligible, "new_target_success"),
    }
    for name, (population, key) in metrics.items():
        mean, interval = rate_ci([record[key] for record in population]) if population else (None, None)
        summary[name], summary[f"{name}_ci"] = mean, interval
    return summary


def align_records(first, second):
    def indexed(records):
        result = {}
        for record in records:
            identifier = record.get("sample_id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Missing stable sample_id; legacy arrays cannot be paired safely")
            if identifier in result:
                raise ValueError(f"Duplicate sample_id: {identifier}")
            result[identifier] = record
        if not result:
            raise ValueError("No paired records")
        return result

    left, right = indexed(first), indexed(second)
    if left.keys() != right.keys():
        raise ValueError("Methods must contain exactly the same sample IDs")
    identifiers = sorted(left)
    for identifier in identifiers:
        for key in ("seed", "input_hash", "clean_target_id", "corrupt_target_id"):
            if left[identifier].get(key) != right[identifier].get(key):
                raise ValueError(f"Paired record differs in {key}: {identifier}")
    return [left[identifier] for identifier in identifiers], [right[identifier] for identifier in identifiers]


def paired_comparison(first, second, metric, binary=True, iters=9999, seed=0):
    """Compare first minus second, pairing IDs and stratifying resampling by seed."""
    left, right = align_records(first, second)
    values_left = np.asarray([record[metric] for record in left], dtype=float)
    values_right = np.asarray([record[metric] for record in right], dtype=float)
    if not np.isfinite(values_left).all() or not np.isfinite(values_right).all():
        raise ValueError("Non-finite paired metric")
    difference = values_left - values_right
    if binary:
        mean, interval = paired_delta_ci(values_right, values_left)
        first_only, second_only, probability = mcnemar(values_left, values_right)
        method = "paired_gain_loss_wilson_bonferroni_approx95; exact_McNemar"
    else:
        if iters < 1:
            raise ValueError("Resampling count must be positive")
        rng = np.random.default_rng(seed)
        strata = {}
        for index, record in enumerate(left):
            strata.setdefault(record.get("seed", 0), []).append(index)
        draws = np.zeros(iters)
        for indexes in strata.values():
            draws += difference[rng.choice(indexes, size=(iters, len(indexes)))].sum(axis=1)
        draws /= len(difference)
        mean = float(difference.mean())
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        randomized = (rng.choice([-1, 1], size=(iters, len(difference))) * difference).mean(axis=1)
        probability = float((1 + np.count_nonzero(np.abs(randomized) >= abs(mean))) / (iters + 1))
        first_only = second_only = None
        method = "seed_stratified_paired_percentile_bootstrap; paired_sign_permutation_plus1"
    return {"metric": metric, "n_unique": len(left), "difference": mean,
            "ci": interval, "p": probability, "first_only": first_only,
            "second_only": second_only, "method": method,
            "scope": "conditional_on_observed_seeds; not population equivalence"}