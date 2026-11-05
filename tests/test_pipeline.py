"""Tests run on synthetic EEG, so they need no download and run in CI.

The synthetic data mimics motor imagery: imagining one hand lowers 10 Hz (mu)
power over the opposite side of the head. A working pipeline must pick this up.
"""
import numpy as np
import pytest

from eegdecode.data import EXCLUDED_SUBJECTS, parse_subjects
from eegdecode.evaluate import (
    chance_level,
    clone_factory,
    compare_methods,
    cross_subject,
    within_subject,
)
from eegdecode.models import EEGNetClassifier, _scale, make_c3c4_lda, make_csp_lda

SFREQ = 128


def make_fake_eeg(n_subjects=6, n_trials=40, n_channels=8, n_samples=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SFREQ
    X, y, groups = [], [], []
    for subject in range(1, n_subjects + 1):
        labels = np.tile([0, 1], n_trials // 2)
        for label in labels:
            trial = rng.normal(0, 1, (n_channels, n_samples))
            phase = rng.uniform(0, 2 * np.pi)
            mu = np.sin(2 * np.pi * 10 * t + phase)
            # label 0 (left hand) -> weak mu on channel 1, strong on channel 0
            trial[0] += (3.0 if label == 0 else 0.5) * mu
            trial[1] += (0.5 if label == 0 else 3.0) * mu
            X.append(trial)
        y.extend(labels)
        groups.extend([subject] * len(labels))
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def test_parse_subjects_ranges_and_exclusions():
    assert parse_subjects("1-3,7") == [1, 2, 3, 7]
    assert parse_subjects([5, 5, 2]) == [2, 5]
    assert not set(parse_subjects("85-105")) & EXCLUDED_SUBJECTS
    with pytest.raises(ValueError):
        parse_subjects("0-2")


def test_chance_level_is_above_half_for_small_samples():
    assert chance_level(45) > 0.6      # one subject in this dataset
    assert chance_level(10_000) < 0.51  # almost 50% with lots of trials


def make_power_only_eeg(n_subjects=8, n_trials=50, n_channels=8, n_samples=256, seed=1):
    """Class information ONLY in relative channel power, as in real motor imagery:
    imagining the left hand changes power on one side, the right hand on the other."""
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_subjects * n_trials // 2)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_trials)
    X = rng.normal(0, 1, (len(y), n_channels, n_samples)).astype(np.float32)
    X[y == 0, 0] *= 2.0
    X[y == 1, 1] *= 2.0
    return X, y, groups


def test_scaling_keeps_power_differences_between_channels():
    X, y, _ = make_power_only_eeg()
    Z = _scale(X, float(X.std()))
    power_c0 = (Z[y == 0, 0] ** 2).mean() / (Z[y == 1, 0] ** 2).mean()
    assert power_c0 > 3  # channel 0 is still ~4x stronger for class 0
    assert np.allclose(Z.mean(axis=-1), 0, atol=1e-5)


def test_csp_lda_learns_synthetic_motor_imagery():
    X, y, groups = make_fake_eeg()
    res = within_subject(X, y, groups, clone_factory(make_csp_lda(n_components=2)))
    assert res["mean_accuracy"] > 0.9
    assert res["n_subjects_above_chance"] == len(np.unique(groups))


def test_c3c4_lda_learns_lateralization_within_and_across_subjects():
    X, y, groups = make_fake_eeg()  # the class signal is on channels 0 and 1
    make_model = clone_factory(make_c3c4_lda(c3_index=0, c4_index=1))
    assert within_subject(X, y, groups, make_model)["mean_accuracy"] > 0.9
    assert cross_subject(X, y, groups, make_model, n_splits=3)["mean_accuracy"] > 0.9

    wrong_channels = clone_factory(make_c3c4_lda(c3_index=2, c4_index=3))
    assert cross_subject(X, y, groups, wrong_channels, n_splits=3)["mean_accuracy"] < 0.65


def test_cross_subject_never_tests_on_training_subjects():
    X, y, groups = make_fake_eeg()
    X[:, 0, 0] = np.arange(len(X))  # tag each trial so the spy can trace it to its subject
    train_seen, test_seen = [], []

    def subjects_in(X_fold):
        return set(groups[X_fold[:, 0, 0].astype(int)].tolist())

    class Spy:
        def fit(self, X, y, groups=None):
            train_seen.append(subjects_in(X))
            return self

        def predict(self, X):
            test_seen.append(subjects_in(X))
            return np.zeros(len(X), dtype=int)

    res = cross_subject(X, y, groups, lambda: Spy(), n_splits=3)
    all_subjects = set(np.unique(groups).tolist())
    assert len(train_seen) == len(test_seen) == 3
    for train_subjects, test_subjects in zip(train_seen, test_seen, strict=True):
        assert not train_subjects & test_subjects  # no subject on both sides of a fold
        assert train_subjects | test_subjects == all_subjects
    # every subject is tested in exactly one fold
    assert sum(len(s) for s in test_seen) == len(all_subjects)
    assert set().union(*test_seen) == all_subjects
    assert set(res["per_subject"]) == all_subjects


@pytest.mark.slow
def test_eegnet_learns_power_differences_on_unseen_subjects():
    """Regression test: an earlier version z-scored every channel separately, which
    erased this signal and left EEGNet at chance on real data."""
    X, y, groups = make_power_only_eeg()
    res = cross_subject(X, y, groups, lambda: EEGNetClassifier(
        epochs=30, batch_size=32, kern_length=32, seed=0), n_splits=4)
    assert res["mean_accuracy"] > 0.8


@pytest.mark.slow
def test_eegnet_trains_and_predicts():
    X, y, _ = make_fake_eeg(n_subjects=5, n_trials=40)
    clf = EEGNetClassifier(epochs=15, batch_size=16, kern_length=32, seed=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert (clf.predict(X) == y).mean() > 0.7


@pytest.mark.slow
def test_eegnet_keeps_training_when_new_data_does_not_improve():
    """Regression test: early stopping on validation loss used to restore a near-untrained
    model on real data, leaving EEGNet at chance. Labels here are pure noise, so only
    fitting the training set can raise accuracy, and only if training keeps going."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (120, 8, 128)).astype(np.float32)
    y = rng.permutation(np.tile([0, 1], 60))
    clf = EEGNetClassifier(epochs=70, batch_size=16, kern_length=32, dropout=0.25, seed=0)
    clf.fit(X, y)
    assert len(clf.history_["loss"]) == 70
    assert (clf.predict(X) == y).mean() > 0.8


def test_cli_run_end_to_end(tmp_path, monkeypatch):
    """Full run (load -> evaluate -> results.json + figure) with fake data."""
    import yaml

    from eegdecode import cli

    X, y, groups = make_fake_eeg(n_subjects=5)
    monkeypatch.setattr(cli, "load_dataset", lambda subjects, cfg: (X, y, groups))
    cfg = yaml.safe_load(open("configs/quick.yaml"))
    cfg["output_dir"] = str(tmp_path)
    cfg["csp_lda"]["n_components"] = 2
    cfg["c3c4_lda"] = {"c3_index": 0, "c4_index": 1}  # the fake data has only 8 channels
    cfg["eegnet"].update(epochs=3, kern_length=32)

    summary = cli.run(cfg)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "accuracy_by_subject.png").exists()
    assert set(summary["results"]) == {"csp_lda_within", "csp_lda_cross", "c3c4_lda_within",
                                       "c3c4_lda_cross", "eegnet_within", "eegnet_cross"}
    pairs = {(t["a"], t["b"]) for t in summary["tests"]}
    assert ("eegnet_cross", "csp_lda_cross") in pairs and ("eegnet_within", "eegnet_cross") in pairs


def test_load_subject_preprocessing_offline(monkeypatch):
    """Runs the real MNE preprocessing on a fake recording shaped like PhysioNet
    (64 channels, 160 Hz, T0/T1/T2 annotations), without downloading anything."""
    import mne

    from eegdecode import data

    ch_names = [f"EEG{i:03d}" for i in range(64)]
    sfreq, n_trials = 160, 15

    def fake_raw():
        rng = np.random.default_rng(0)
        info = mne.create_info(ch_names, sfreq, "eeg")
        raw = mne.io.RawArray(rng.normal(0, 1e-5, (64, sfreq * 4 * n_trials * 2)), info,
                              verbose="error")
        onsets = np.arange(n_trials * 2) * 8.0
        labels = ["T1" if i % 2 == 0 else "T2" for i in range(n_trials)]
        descriptions = [d for lab in labels for d in ("T0", lab)]
        raw.set_annotations(mne.Annotations(onsets[: len(descriptions)] / 2, 4.0,
                                            descriptions))
        return raw

    monkeypatch.setattr(mne.datasets.eegbci, "load_data",
                        lambda *a, **k: ["r04.edf", "r08.edf", "r12.edf"])
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *a, **k: fake_raw())

    cfg = {"l_freq": 7.0, "h_freq": 30.0, "tmin": 0.5, "tmax": 2.5, "resample": 128}
    X, y = data.load_subject(1, cfg)
    assert X.shape == (3 * n_trials, 64, 257)  # 2 s at 128 Hz, inclusive
    assert set(np.unique(y)) == {0, 1}
    assert X.dtype == np.float32


def test_compare_methods_finds_consistently_better_method():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 0.7, 30)
    results = {"csp_lda_cross": {"per_subject": {s: float(a) for s, a in enumerate(base)}},
               "eegnet_cross": {"per_subject": {str(s): float(a + 0.05 + rng.uniform(0, 0.02))
                                                for s, a in enumerate(base)}},
               "c3c4_lda_cross": {"per_subject": {s: float(a + rng.normal(0, 0.05))
                                                  for s, a in enumerate(base)}}}
    tests = {(t["a"], t["b"]): t for t in compare_methods(results)}
    assert set(tests) == {("eegnet_cross", "csp_lda_cross"), ("eegnet_cross", "c3c4_lda_cross"),
                          ("csp_lda_cross", "c3c4_lda_cross")}
    better = tests[("eegnet_cross", "csp_lda_cross")]
    assert better["a_better_in"] == 30 and better["p_holm"] < 0.001
    assert tests[("csp_lda_cross", "c3c4_lda_cross")]["p_holm"] > 0.05
    assert all(t["p_holm"] >= t["p"] for t in tests.values())
