"""Results figure: accuracy of every subject under every method, with chance level."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "csp_lda_within": "CSP+LDA\nwithin-subject",
    "csp_lda_cross": "CSP+LDA\ncross-subject",
    "c3c4_lda_within": "C3/C4+LDA\nwithin-subject",
    "c3c4_lda_cross": "C3/C4+LDA\ncross-subject",
    "eegnet_cross": "EEGNet\ncross-subject",
}
COLORS = {"csp_lda_within": "#4C72B0", "csp_lda_cross": "#55A868",
          "c3c4_lda_within": "#8172B3", "c3c4_lda_cross": "#CCB974",
          "eegnet_cross": "#C44E52"}


def plot_results(results: dict, path: str | Path) -> None:
    names = [n for n in LABELS if n in results]
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 2, 4.5))
    rng = np.random.default_rng(0)

    for i, name in enumerate(names):
        accs = np.array(list(results[name]["per_subject"].values()))
        jitter = rng.uniform(-0.12, 0.12, len(accs))
        ax.scatter(np.full(len(accs), i) + jitter, accs, s=18, alpha=0.6,
                   color=COLORS[name], edgecolor="none")
        ax.errorbar(i + 0.28, accs.mean(), yerr=accs.std(), fmt="o", color="black",
                    capsize=4, markersize=6)
        ax.text(i + 0.36, accs.mean(), f"{accs.mean():.2f}", va="center", fontsize=9)

    chance = results.get("csp_lda_within", {}).get("chance_level_per_subject")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="50% (theoretical chance)")
    if chance:
        ax.axhline(chance, color="grey", linestyle="--", linewidth=1,
                   label=f"{chance:.0%} (significant chance level, one subject)")

    ax.set_xticks(range(len(names)), [LABELS[n] for n in names])
    ax.set_xlim(-0.5, len(names) - 0.3)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("Accuracy (per subject)")
    ax.set_title("Left vs right hand motor imagery decoding")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
