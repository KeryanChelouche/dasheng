#!/usr/bin/env python3
"""Plot sample-efficiency curves from few-shot evaluation results.

One curve per model.  X-axis = total training samples (log scale).
Y-axis = accuracy or macro-F1.  Shaded band = ±1 std.

Examples
--------
# All models found in results/few_shot/, both probes, accuracy
python scripts/plot_sample_efficiency.py

# Only specific models, F1 metric, linear probe only
python scripts/plot_sample_efficiency.py \\
    --models dasheng_base mae_imagenet \\
    --metric f1 \\
    --probe linear \\
    --output results/plots/sample_efficiency_linear_f1.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_DEFAULT_INPUT_DIR  = REPO_ROOT / "results" / "few_shot"
_DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "plots"

# ── Aesthetics ────────────────────────────────────────────────────────────────

# Ordered colour palette — deterministic, colour-blind friendly
_PALETTE = [
    "#2196F3",  # blue       dasheng_base
    "#4CAF50",  # green      dasheng_06B
    "#009688",  # teal       dasheng_12B
    "#FF9800",  # orange     audiomae
    "#9C27B0",  # purple     beats_iter3
    "#E91E63",  # pink       beats_iter3+
    "#F44336",  # red        fisher_small
    "#607D8B",  # blue-grey  mae_imagenet
]

_DISPLAY_NAMES = {
    "dasheng_base":      "Dasheng-base",
    "dasheng_06B":       "Dasheng-0.6B",
    "dasheng_12B":       "Dasheng-1.2B",
    "audiomae":          "AudioMAE",
    "beats_iter3":       "BEATs iter3",
    "beats_iter3+":      "BEATs iter3+",
    "fisher_small":      "FISHER-small",
    "fisher_small_4band":"FISHER-small",
    "mae_imagenet":      "MAE-ImageNet",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(input_dir: Path, models: list[str] | None) -> dict:
    """Return {model_key: result_dict} keeping only the latest run per model×dataset."""
    latest: dict = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("run_"):
            continue
        data = json.loads(path.read_text())
        if "n_repeats" not in data:
            continue  # skip non-few-shot files
        key = (data["model"], data["dataset"])
        latest[key] = data  # sorted by name → latest timestamp wins

    if models:
        # normalise: "beats_iter3+" in registry → "beats_iter3+" in JSON model field
        latest = {k: v for k, v in latest.items() if _registry_key(v["model"]) in models}

    return latest


def _registry_key(model_name: str) -> str:
    """Map JSON model field back to registry key (handles fisher_small_4band → fisher_small)."""
    _MAP = {"fisher_small_4band": "fisher_small"}
    return _MAP.get(model_name, model_name)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(
    results: dict,
    probe_names: list[str],
    metric: str,            # "acc" or "f1"
    dataset_filter: str | None,
    output_path: Path,
) -> None:
    # Group by dataset
    datasets = sorted({v["dataset"] for v in results.values()})
    if dataset_filter:
        datasets = [d for d in datasets if d == dataset_filter]
    if not datasets:
        print("No matching data found.")
        return

    metric_key  = f"mean_{metric}"
    std_key     = f"std_{metric}"
    metric_label = "Accuracy (%)" if metric == "acc" else "Macro-F1 (%)"

    n_datasets = len(datasets)
    n_probes   = len(probe_names)
    n_cols = n_probes
    n_rows = n_datasets

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    # Assign deterministic colours by model name
    all_models = sorted({_registry_key(v["model"]) for v in results.values()})
    colour_map = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(all_models)}

    for row_idx, dataset_name in enumerate(datasets):
        dataset_results = {
            k: v for k, v in results.items() if v["dataset"] == dataset_name
        }

        for col_idx, probe_name in enumerate(probe_names):
            ax = axes[row_idx][col_idx]

            for model_key, data in sorted(dataset_results.items(),
                                          key=lambda kv: _registry_key(kv[1]["model"])):
                probe_data = data["probes"].get(probe_name)
                if probe_data is None:
                    continue

                xs, ys, errs = [], [], []
                for entry in probe_data:
                    if entry[metric_key] is None:
                        continue
                    xs.append(entry["n_total_train"])
                    ys.append(entry[metric_key] * 100)
                    errs.append(entry[std_key] * 100)

                xs   = np.array(xs)
                ys   = np.array(ys)
                errs = np.array(errs)

                rkey   = _registry_key(data["model"])
                colour = colour_map[rkey]
                label  = _DISPLAY_NAMES.get(rkey, rkey)

                # Separate "few-shot" points from the "full" point
                few_mask  = np.array([e["n_shots"] is not None for e in probe_data
                                      if e[metric_key] is not None])
                full_mask = ~few_mask

                if few_mask.any():
                    ax.plot(xs[few_mask], ys[few_mask],
                            marker="o", markersize=4, linewidth=1.8,
                            color=colour, label=label)
                    ax.fill_between(
                        xs[few_mask],
                        ys[few_mask] - errs[few_mask],
                        ys[few_mask] + errs[few_mask],
                        alpha=0.15, color=colour,
                    )
                if full_mask.any():
                    ax.scatter(xs[full_mask], ys[full_mask],
                               marker="*", s=120, color=colour, zorder=5,
                               label=f"{label} (full)")

            ax.set_xscale("log")
            ax.set_xlabel("Training samples", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)

            probe_title = {
                "knn_k10":    "k-NN (k=10)",
                "linear_C1.0":"Linear probe",
            }.get(probe_name, probe_name)
            ax.set_title(
                f"{dataset_name.upper()} — {probe_title}",
                fontsize=12, fontweight="bold",
            )
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(True, which="major", linestyle="--", alpha=0.4)
            ax.grid(True, which="minor", linestyle=":",  alpha=0.2)
            ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"Sample efficiency — {metric_label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", default=str(_DEFAULT_INPUT_DIR))
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these model registry keys (default: all found)",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Restrict to this dataset (default: all found)",
    )
    p.add_argument(
        "--probe", nargs="+",
        default=["knn_k10", "linear_C1.0"],
        choices=["knn_k10", "linear_C1.0"],
        help="Which probe columns to plot (default: both)",
    )
    p.add_argument(
        "--metric", default="acc", choices=["acc", "f1"],
        help="Metric to plot on y-axis (default: acc)",
    )
    p.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR / "sample_efficiency.png"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.input_dir), args.models)
    if not results:
        print(f"No few-shot result files found in {args.input_dir}")
        return
    plot(
        results=results,
        probe_names=args.probe,
        metric=args.metric,
        dataset_filter=args.dataset,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
