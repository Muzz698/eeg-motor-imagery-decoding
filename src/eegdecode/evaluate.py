"""Evaluation schemes.

- Within-subject: train and test on the same person (5-fold CV on their trials).
- Cross-subject: train on some people, test on people the model has never seen
  (GroupKFold by subject). This is the harder, more honest test, and the one a
  real brain-computer interface needs if it should work without calibration.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
from scipy.stats import binom, wilcoxon
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold


def chance_level(n_trials: int, n_classes: int = 2, alpha: float = 0.05) -> float:
    """Accuracy needed to beat chance at significance `alpha` for `n_trials` trials
    (Combrisson & Jerbi, 2015). With few trials, 50% is NOT the right bar."""
    return float(binom.ppf(1 - alpha, n_trials, 1 / n_classes) / n_trials)


def _fit(model, X, y, groups=None):
    if groups is not None and "groups" in inspect.signature(model.fit).parameters:
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def within_subject(X, y, groups, make_model: Callable, n_splits: int = 5,
                   seed: int = 42) -> dict:
    per_subject = {}
    for subject in np.unique(groups):
        mask = groups == subject
        Xs, ys = X[mask], y[mask]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        correct = 0
        for train, test in cv.split(Xs, ys):
            model = _fit(make_model(), Xs[train], ys[train])
            correct += int((model.predict(Xs[test]) == ys[test]).sum())
        per_subject[int(subject)] = correct / len(ys)
    accs = np.array(list(per_subject.values()))
    n_trials = int(np.median([np.sum(groups == s) for s in per_subject]))
    return {
        "scheme": "within_subject",
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "per_subject": per_subject,
        "chance_level_per_subject": chance_level(n_trials),
        "n_subjects_above_chance": int((accs > chance_level(n_trials)).sum()),
    }


def cross_subject(X, y, groups, make_model: Callable, n_splits: int = 5) -> dict:
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    predictions = np.empty_like(y)
    fold_accs = []
    for fold, (train, test) in enumerate(cv.split(X, y, groups)):
        model = _fit(make_model(), X[train], y[train], groups[train])
        predictions[test] = model.predict(X[test])
        fold_accs.append(float((predictions[test] == y[test]).mean()))
        share = max(float((predictions[test] == c).mean()) for c in np.unique(y))
        warn = "  <- predicts one class almost always" if share > 0.9 else ""
        print(f"    fold {fold + 1}/{n_splits}: {fold_accs[-1]:.3f}{warn}", flush=True)

    per_subject = {int(s): float((predictions[groups == s] == y[groups == s]).mean())
                   for s in np.unique(groups)}
    return {
        "scheme": "cross_subject",
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
        "per_subject": per_subject,
        "overall_accuracy": float((predictions == y).mean()),
        "chance_level_overall": chance_level(len(y)),
    }


def clone_factory(estimator) -> Callable:
    """Return a function that gives a fresh, unfitted copy of `estimator`."""
    return lambda: clone(estimator)


# Pairs of methods compared across subjects. Each pair is tested only if both are in the results.
DEFAULT_PAIRS = [
    ("eegnet_cross", "csp_lda_cross"),
    ("eegnet_cross", "c3c4_lda_cross"),
    ("csp_lda_cross", "c3c4_lda_cross"),
    ("eegnet_within", "csp_lda_within"),
    ("csp_lda_within", "c3c4_lda_within"),
    ("csp_lda_within", "csp_lda_cross"),
    ("eegnet_within", "eegnet_cross"),
]


def compare_methods(results: dict, pairs=DEFAULT_PAIRS) -> list[dict]:
    """Two-sided Wilcoxon signed-rank test between methods, paired by subject.

    Each subject contributes one accuracy per method, so the test asks whether one
    method is consistently better across people. p-values are Holm-corrected for the
    number of pairs tested.
    """
    tests = []
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        pa, pb = results[a]["per_subject"], results[b]["per_subject"]
        subjects = sorted(set(map(int, pa)) & set(map(int, pb)))
        xa = np.array([pa[s] if s in pa else pa[str(s)] for s in subjects])
        xb = np.array([pb[s] if s in pb else pb[str(s)] for s in subjects])
        diff = xa - xb
        if len(subjects) < 2 or np.all(diff == 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(xa, xb, zero_method="wilcox")
        tests.append({"a": a, "b": b, "n_subjects": len(subjects),
                      "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                      "median_difference": float(np.median(diff)),
                      "a_better_in": int((diff > 0).sum()), "b_better_in": int((diff < 0).sum()),
                      "W": float(stat), "p": float(p)})
    # Holm-Bonferroni correction
    order = np.argsort([t["p"] for t in tests])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running
    return tests


def format_tests(tests: list[dict]) -> str:
    lines = [f"{'A':16s} {'B':16s} {'n':>4s} {'mean A':>7s} {'mean B':>7s} "
             f"{'A>B':>4s} {'B>A':>4s} {'W':>8s} {'p':>9s} {'p(Holm)':>9s}"]
    for t in tests:
        lines.append(f"{t['a']:16s} {t['b']:16s} {t['n_subjects']:4d} {t['mean_a']:7.3f} "
                     f"{t['mean_b']:7.3f} {t['a_better_in']:4d} {t['b_better_in']:4d} "
                     f"{t['W']:8.1f} {t['p']:9.2g} {t['p_holm']:9.2g}")
    return "\n".join(lines)
