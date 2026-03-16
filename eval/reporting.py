"""Metrics reporting: console tables, JSON persistence, and plots."""
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print a single evaluation result to stdout."""
    print(f"\nModel:   {results['model']}")
    print(
        f"Dataset: {results['dataset']}  "
        f"({results['n_samples']} samples, {results['n_classes']} classes)"
    )
    print("-" * 56)
    for probe_name, probe_res in results["probes"].items():
        mean = probe_res["mean_acc"] * 100
        std = probe_res["std_acc"] * 100
        folds = "  ".join(f"{a * 100:.1f}" for a in probe_res["fold_accs"])
        print(f"  {probe_name:<20}  {mean:.2f}% ± {std:.2f}%    [{folds}]")
    print()


def save_results(results: Dict[str, Any], path: Path) -> None:
    """Write results dict as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def make_comparison_table(results_dir: Path) -> str:
    """Read all JSON result files and return a Markdown comparison table.

    Rows are (model, dataset) pairs; columns are probe names.
    Values are "mean% ± std%".
    """
    rows: List[Dict] = []
    for p in sorted(results_dir.glob("*.json")):
        data = json.loads(p.read_text())
        row: Dict[str, Any] = {
            "model": data["model"],
            "dataset": data["dataset"],
        }
        for probe_name, probe_res in data["probes"].items():
            row[probe_name] = (
                f"{probe_res['mean_acc'] * 100:.2f} ± {probe_res['std_acc'] * 100:.2f}"
            )
        rows.append(row)

    if not rows:
        return "No results found."

    # Deduplicate: keep only the latest run per (model, dataset) pair
    seen: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row["model"], row["dataset"])
        seen[key] = row   # later files overwrite earlier ones (sorted by name = timestamp)
    rows = list(seen.values())

    all_probes = sorted({k for r in rows for k in r if k not in ("model", "dataset")})
    headers = ["model", "dataset"] + all_probes
    col_w = [
        max(len(h), max(len(str(r.get(h, "-"))) for r in rows)) + 2
        for h in headers
    ]

    def fmt_row(vals: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    lines = [fmt_row(headers), sep] + [fmt_row([r.get(h, "-") for h in headers]) for r in rows]
    return "\n".join(lines)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cmap="Blues",
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
