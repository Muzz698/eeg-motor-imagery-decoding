"""Command-line entry point.

    eegdecode run --config configs/default.yaml
    eegdecode run --config configs/default.yaml --subjects 1-5 --skip-eegnet
    eegdecode add-eegnet-within --config configs/default.yaml --subjects 1-109
    eegdecode compare results/results.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .data import load_dataset, parse_subjects
from .evaluate import clone_factory, compare_methods, cross_subject, format_tests, within_subject
from .models import EEGNetClassifier, make_c3c4_lda, make_csp_lda
from .plots import plot_results


def run(cfg: dict, skip_eegnet: bool = False) -> dict:
    import mne

    mne.set_log_level("WARNING")  # hide MNE's per-fold rank messages
    np.random.seed(cfg["seed"])
    subjects = parse_subjects(cfg["data"]["subjects"])
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(subjects)} subjects ...")
    X, y, groups = load_dataset(subjects, cfg["data"])
    print(f"Data: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples; "
          f"class balance {np.bincount(y).tolist()}")

    csp = make_csp_lda(cfg["csp_lda"]["n_components"], cfg["csp_lda"].get("reg"))
    ev = cfg["evaluation"]
    results = {}

    print("CSP+LDA, within-subject ...")
    results["csp_lda_within"] = within_subject(X, y, groups, clone_factory(csp),
                                               ev["within_folds"], cfg["seed"])
    print(f"  mean accuracy {results['csp_lda_within']['mean_accuracy']:.3f}")

    print("CSP+LDA, cross-subject ...")
    results["csp_lda_cross"] = cross_subject(X, y, groups, clone_factory(csp), ev["cross_folds"])

    c3c4 = make_c3c4_lda(cfg["c3c4_lda"]["c3_index"], cfg["c3c4_lda"]["c4_index"])

    print("C3/C4+LDA, within-subject ...")
    results["c3c4_lda_within"] = within_subject(X, y, groups, clone_factory(c3c4),
                                                ev["within_folds"], cfg["seed"])
    print(f"  mean accuracy {results['c3c4_lda_within']['mean_accuracy']:.3f}")

    print("C3/C4+LDA, cross-subject ...")
    results["c3c4_lda_cross"] = cross_subject(X, y, groups, clone_factory(c3c4),
                                              ev["cross_folds"])

    if not skip_eegnet:
        results["eegnet_within"] = _eegnet_within(X, y, groups, cfg)
        eegnet = EEGNetClassifier(seed=cfg["seed"], **cfg["eegnet"])
        print("EEGNet, cross-subject ...")
        results["eegnet_cross"] = cross_subject(X, y, groups, clone_factory(eegnet),
                                                ev["cross_folds"])

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "config": cfg,
        "n_trials": len(y),
        "subjects": subjects,
        "results": results,
        "tests": compare_methods(results),
    }
    _save(summary, out_dir)

    print("\nSummary")
    for name, r in results.items():
        print(f"  {name:18s} {r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f}")
    print(f"Saved results to {out_dir}/")
    return summary


def _eegnet_within(X, y, groups, cfg: dict) -> dict:
    """EEGNet trained and tested on each person separately (5-fold CV per person).
    Settings in `eegnet_within` override `eegnet`; with only ~36 training trials,
    Lawhern et al. (2018) suggest a higher dropout (0.5) for within-subject use."""
    params = {**cfg["eegnet"], **cfg.get("eegnet_within", {})}
    eegnet = EEGNetClassifier(seed=cfg["seed"], **params)
    print("EEGNet, within-subject ...", flush=True)
    r = within_subject(X, y, groups, clone_factory(eegnet), cfg["evaluation"]["within_folds"],
                       cfg["seed"])
    print(f"  mean accuracy {r['mean_accuracy']:.3f}")
    return r


def _save(summary: dict, out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    plot_results(summary["results"], out_dir / "accuracy_by_subject.png")
    print("\nPaired Wilcoxon signed-rank tests across subjects")
    print(format_tests(summary["tests"]))


def add_eegnet_within(cfg: dict) -> dict:
    """Add the EEGNet within-subject result to an existing results.json, without
    re-running the other methods (which takes hours for all subjects)."""
    import mne

    mne.set_log_level("WARNING")
    out_dir = Path(cfg.get("output_dir", "results"))
    summary = json.loads((out_dir / "results.json").read_text())
    subjects = parse_subjects(cfg["data"]["subjects"])
    if subjects != summary["subjects"]:
        raise ValueError(f"{out_dir}/results.json was made with different subjects; "
                         "pass the same --subjects as the original run")
    X, y, groups = load_dataset(subjects, cfg["data"])
    summary["results"]["eegnet_within"] = _eegnet_within(X, y, groups, cfg)
    summary["tests"] = compare_methods(summary["results"])
    _save(summary, out_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eegdecode",
                                     description="Motor imagery EEG decoding benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Run the full benchmark")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--subjects", help='Override subjects, e.g. "1-5" or "1,2,7"')
    p.add_argument("--skip-eegnet", action="store_true", help="Only run CSP+LDA (fast)")
    w = sub.add_parser("add-eegnet-within",
                       help="Add EEGNet within-subject results to an existing results.json")
    w.add_argument("--config", default="configs/default.yaml")
    w.add_argument("--subjects", help="Must match the subjects of the existing run")
    c = sub.add_parser("compare", help="Wilcoxon tests between methods from a results.json")
    c.add_argument("results", nargs="?", default="results/results.json")
    args = parser.parse_args(argv)

    if args.command == "compare":
        summary = json.loads(Path(args.results).read_text())
        summary["tests"] = compare_methods(summary["results"])
        Path(args.results).write_text(json.dumps(summary, indent=2))
        print(format_tests(summary["tests"]))
        return

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.subjects:
        cfg["data"]["subjects"] = args.subjects
    start = time.time()
    if args.command == "add-eegnet-within":
        add_eegnet_within(cfg)
    else:
        run(cfg, skip_eegnet=args.skip_eegnet)
    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
