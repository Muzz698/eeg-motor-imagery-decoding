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
from scipy.stats import binom
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
