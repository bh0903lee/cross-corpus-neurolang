"""
statistical_analysis.py — statistical significance tests and multi-seed
aggregation (aggregate_seeds, mcnemar_test, bootstrap_permutation_test,
compare_all_models).
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import f1_score


def aggregate_seeds(results_list: List[dict]) -> dict:
    """
    Aggregate results across N seeds: mean ± std for numeric metrics.
    Each element of results_list is a dict returned by evaluate_model().
    Returns {metric: {"mean": float, "std": float, "values": list}}.
    """
    if not results_list:
        return {}

    all_keys = set()
    for r in results_list:
        all_keys.update(r.keys())

    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results_list if key in r]
        if not values:
            continue
        # Aggregate numeric scalars only
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = {
                "mean":   float(np.mean(values)),
                "std":    float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                "values": [float(v) for v in values],
            }

    # Summary format for key metrics
    summary_keys = [
        "test_accuracy", "test_macro_f1", "test_weighted_f1",
        "val_accuracy",  "val_macro_f1",
    ]
    aggregated["summary"] = {
        k: f"{aggregated[k]['mean']:.4f} ± {aggregated[k]['std']:.4f}"
        for k in summary_keys if k in aggregated
    }

    return aggregated


def mcnemar_test(preds_a: list, preds_b: list, labels: list) -> dict:
    """
    McNemar's test: whether two classifiers' error patterns differ significantly
    (predictions reduced to binary correct/incorrect). H0: equal accuracy.
    Returns {statistic, p_value, b (a wrong/b correct), c (a correct/b wrong), significant}.
    """
    from scipy.stats import chi2 as _chi2

    correct_a = np.array(preds_a) == np.array(labels)
    correct_b = np.array(preds_b) == np.array(labels)

    b = int((~correct_a & correct_b).sum())   # a wrong, b correct
    c = int((correct_a & ~correct_b).sum())   # a correct, b wrong

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    # McNemar statistic with Edwards continuity correction
    stat    = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(1.0 - _chi2.cdf(stat, df=1))

    return {
        "statistic":   float(stat),
        "p_value":     p_value,
        "b":           b,
        "c":           c,
        "significant": p_value < 0.05,
    }


def bootstrap_permutation_test(preds_a: list, preds_b: list, labels: list,
                                n_iter: int = 1000, seed: int = 42) -> dict:
    """
    Permutation test on the Macro-F1 difference between two models.
    H0: preds_a and preds_b have equal Macro-F1; predictions are shuffled to
    build the null distribution. Returns {observed_diff, p_value, n_iter, significant}.
    """
    rng = np.random.default_rng(seed)

    def _mf1(preds):
        return f1_score(labels, preds, average="macro", zero_division=0)

    observed_diff = abs(_mf1(preds_a) - _mf1(preds_b))

    pool = np.array(list(preds_a) + list(preds_b))
    n    = len(preds_a)
    count_extreme = 0

    for _ in range(n_iter):
        perm    = rng.permutation(pool)
        diff    = abs(_mf1(perm[:n].tolist()) - _mf1(perm[n:].tolist()))
        if diff >= observed_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_iter + 1)  # +1 for Bayesian smoothing

    return {
        "observed_diff": float(observed_diff),
        "p_value":       float(p_value),
        "n_iter":        n_iter,
        "significant":   p_value < 0.05,
    }


def compare_all_models(results_dict: dict, proposed_key: str = "proposed") -> list:
    """
    Build a comparison table: proposed model vs all baselines/ablations.
    results_dict maps model_name to {"aggregated": {test_macro_f1: {mean, std}, ...}}.
    Returns a list of row dicts sorted by test_macro_f1 mean (descending).
    """
    rows = []
    for name, result in results_dict.items():
        agg = result.get("aggregated", result)
        mf1 = agg.get("test_macro_f1", {})
        acc = agg.get("test_accuracy",  {})

        if isinstance(mf1, dict):
            mf1_mean = mf1.get("mean", 0.0)
            mf1_std  = mf1.get("std",  0.0)
        else:
            mf1_mean = float(mf1) if mf1 else 0.0
            mf1_std  = 0.0

        if isinstance(acc, dict):
            acc_mean = acc.get("mean", 0.0)
            acc_std  = acc.get("std",  0.0)
        else:
            acc_mean = float(acc) if acc else 0.0
            acc_std  = 0.0

        rows.append({
            "model":         name,
            "is_proposed":   name == proposed_key,
            "macro_f1_mean": mf1_mean,
            "macro_f1_std":  mf1_std,
            "accuracy_mean": acc_mean,
            "accuracy_std":  acc_std,
            "macro_f1_str":  f"{mf1_mean:.4f} ± {mf1_std:.4f}",
            "accuracy_str":  f"{acc_mean:.4f} ± {acc_std:.4f}",
        })

    rows.sort(key=lambda x: x["macro_f1_mean"], reverse=True)
    return rows
