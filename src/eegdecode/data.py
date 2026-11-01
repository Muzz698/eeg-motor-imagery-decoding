"""Load and preprocess the PhysioNet EEG Motor Movement/Imagery dataset.

We use the motor *imagery* runs (4, 8, 12): the subject imagines opening and
closing the left fist (T1) or the right fist (T2). Each subject gives ~45 trials.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

IMAGERY_RUNS = [4, 8, 12]
# Subjects with a different sampling rate or broken annotations in this dataset.
EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104}
CLASS_NAMES = ["left_hand", "right_hand"]


def parse_subjects(spec: str | list[int]) -> list[int]:
    """Turn "1-10,15" into [1, ..., 10, 15], dropping excluded subjects."""
    if isinstance(spec, list):
        subjects = [int(s) for s in spec]
    else:
        subjects = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = (int(p) for p in part.split("-"))
                subjects.extend(range(start, end + 1))
            else:
                subjects.append(int(part))
    bad = [s for s in subjects if not 1 <= s <= 109]
    if bad:
        raise ValueError(f"Subject IDs must be between 1 and 109, got {bad}")
    return sorted(set(subjects) - EXCLUDED_SUBJECTS)


def _cache_key(subject: int, data_cfg: dict) -> str:
    keys = ("l_freq", "h_freq", "tmin", "tmax", "resample")
    params = json.dumps({k: data_cfg.get(k) for k in keys}, sort_keys=True)
    digest = hashlib.md5(params.encode()).hexdigest()[:8]
    return f"S{subject:03d}_{digest}.npz"


def load_subject(subject: int, data_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Download (if needed), filter and epoch one subject.

    Returns X with shape (n_trials, n_channels, n_samples) in microvolts,
    and y with 0 = left hand, 1 = right hand.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    files = eegbci.load_data(subject, IMAGERY_RUNS, path=data_cfg.get("mne_data_dir"),
                             update_path=False, verbose="error")
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="error") for f in files])
    eegbci.standardize(raw)

    # Band-pass to the mu/beta range, where motor imagery changes power.
    raw.filter(data_cfg["l_freq"], data_cfg["h_freq"], fir_design="firwin",
               skip_by_annotation="edge", verbose="error")

    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, {"left_hand": 1, "right_hand": 2},
                        tmin=data_cfg["tmin"], tmax=data_cfg["tmax"], picks=picks,
                        baseline=None, proj=False, preload=True, verbose="error")
    if data_cfg.get("resample"):
        epochs.resample(data_cfg["resample"], verbose="error")

    X = (epochs.get_data() * 1e6).astype(np.float32)  # volts -> microvolts
    y = (epochs.events[:, -1] - 1).astype(np.int64)
    return X, y


def load_dataset(subjects: list[int], data_cfg: dict,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load many subjects. Returns X, y and groups (the subject ID of each trial).

    Preprocessed subjects are cached as .npz files so later runs are fast.
    """
    cache_dir = Path(data_cfg.get("cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    Xs, ys, groups = [], [], []
    for subject in subjects:
        cache_file = cache_dir / _cache_key(subject, data_cfg)
        if cache_file.exists():
            with np.load(cache_file) as f:
                X, y = f["X"], f["y"]
        else:
            if verbose:
                print(f"  preprocessing subject {subject} ...", flush=True)
            X, y = load_subject(subject, data_cfg)
            np.savez_compressed(cache_file, X=X, y=y)
        Xs.append(X)
        ys.append(y)
        groups.append(np.full(len(y), subject))

    n_samples = {x.shape[-1] for x in Xs}
    if len(n_samples) != 1:
        raise RuntimeError(f"Subjects have different epoch lengths: {n_samples}")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)
