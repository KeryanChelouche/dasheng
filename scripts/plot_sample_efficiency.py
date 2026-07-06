#!/usr/bin/env python3
"""Plot sample-efficiency curves from few-shot evaluation results.

One curve per model.  X-axis = total training samples (log scale).
Y-axis = accuracy or macro-F1.  Shaded band = ±1 std.
A horizontal dashed line shows MAE-ImageNet's full-data performance as a
reference, making it easy to read off when audio models surpass it.

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

_PALETTE = {
    # ── Audio foundation models (cool tones) ──────────────────────
    "dasheng_base":       "#2196F3",  # blue
    "dasheng_06B":        "#4CAF50",  # green
    "dasheng_12B":        "#009688",  # teal
    "audiomae":           "#FF9800",  # orange
    "beats_iter3":        "#9C27B0",  # purple
    "beats_iter3+":       "#E91E63",  # pink
    "fisher_small":       "#795548",  # brown
    "mae_imagenet":       "#607D8B",  # blue-grey
    "whisper_small":      "#3F51B5",  # indigo
    "whisper_large_v3":   "#00BCD4",  # cyan
    "qwen2_audio":        "#FF5722",  # deep orange
    "vjepa2.1_vitb":      "#8D6E63",  # warm grey
    # ── DINOv3 variants (red-orange family) ───────────────────────
    "dinov3_vitb16":                  "#D32F2F",  # red
    "dinov3_lora":                    "#F44336",  # lighter red
    "dinov3_lora_ft_glasgow":         "#F44336",  # lighter red
    "dinov3_lora_ft_glasgow_5":       "#F44336",  # lighter red
    "dinov3_lora_ft_mad_5":           "#FF7043",  # deep orange
    "dinov3_lora_ft_mad_sub12":       "#FFB300",  # amber
    "dinov3_lora_ft_glasgow_young":   "#AD1457",  # dark pink
    "dinov3_vits16":                  "#EF9A9A",  # pale red
    "dinov3_vits16_lora":             "#EF5350",  # soft red
    "dinov3_vits16_dora":             "#E57373",  # muted red
    # ── ImageNet-supervised ViT (purple-violet family) ────────────
    "vit_base_imagenet":              "#7E57C2",  # violet
    "vit_base_imagenet_lora":         "#9575CD",  # lighter violet
    "vit_base_imagenet_dora":         "#B39DDB",  # pale violet
    "vit_small_imagenet":             "#5E35B1",  # deep violet
    "vit_small_imagenet_lora":        "#673AB7",  # purple
    "vit_small_imagenet_dora":        "#7E57C2",  # violet
    # ── ResNet variants (green family) ────────────────────────────
    "resnet50":                  "#2E7D32",  # dark green
    "resnet50_imagenet":         "#43A047",  # green
    "resnet50_ft_glasgow":       "#66BB6A",  # medium green
    "resnet50_ft_glasgow_5":     "#66BB6A",  # medium green
    "resnet50_ft_mad_5":         "#81C784",  # light green
    "resnet50_ft_mad_5_sub1":    "#A5D6A7",  # pale green
    "resnet50_ft_mad_sub1":      "#A5D6A7",  # pale green
    "resnet50_ft_mad_sub23":     "#C8E6C9",  # very pale green
    "resnet50_ft_mad_sub12":     "#1B5E20",  # very dark green
    "resnet50_ft_glasgow_young": "#00BFA5",  # teal-green
}
_PALETTE_DEFAULT = "#333333"

_DISPLAY_NAMES = {
    "dasheng_base":       "Dasheng-base",
    "dasheng_06B":        "Dasheng-0.6B",
    "dasheng_12B":        "Dasheng-1.2B",
    "audiomae":           "AudioMAE",
    "beats_iter3":        "BEATs iter3",
    "beats_iter3+":       "BEATs iter3+",
    "fisher_small":       "FISHER-small",
    "fisher_small_4band": "FISHER-small",
    "mae_imagenet":       "MAE-ImageNet",
    "whisper_small":      "Whisper-small",
    "whisper_large_v3":   "Whisper-large-v3",
    "qwen2_audio":        "Qwen2-Audio",
    "vjepa2.1_vitb":      "V-JEPA 2.1",
    "dinov3_vitb16":                  "DINOv3 (frozen)",
    "dinov3_lora":                    "DINOv3 LoRA",
    "dinov3_lora_ft_glasgow":         "DINOv3 LoRA (ft Glasgow)",
    "dinov3_lora_ft_glasgow_5":       "DINOv3 LoRA (ft Glasgow-5)",
    "dinov3_lora_ft_mad_5":           "DINOv3 LoRA (ft MAD-5)",
    "dinov3_lora_ft_mad_sub12":       "DINOv3 LoRA (ft MAD-sub12)",
    "dinov3_lora_ft_glasgow_young":   "DINOv3 LoRA (ft Glasgow-young)",
    "dinov3_vits16":                  "DINOv3-S (frozen)",
    "dinov3_vits16_lora":             "DINOv3-S LoRA",
    "dinov3_vits16_dora":             "DINOv3-S DoRA",
    "vit_base_imagenet":              "ViT-B/16 (ImageNet)",
    "vit_base_imagenet_lora":         "ViT-B/16 LoRA (ImageNet)",
    "vit_base_imagenet_dora":         "ViT-B/16 DoRA (ImageNet)",
    "vit_small_imagenet":             "ViT-S/16 (ImageNet)",
    "vit_small_imagenet_lora":        "ViT-S/16 LoRA (ImageNet)",
    "vit_small_imagenet_dora":        "ViT-S/16 DoRA (ImageNet)",
    "resnet50":                  "ResNet-50 (sup.)",
    "resnet50_imagenet":         "ResNet-50 (ImageNet)",
    "resnet50_ft_glasgow":       "ResNet-50 (ft Glasgow)",
    "resnet50_ft_glasgow_5":     "ResNet-50 (ft Glasgow-5)",
    "resnet50_ft_mad_5":         "ResNet-50 (ft MAD-5)",
    "resnet50_ft_mad_5_sub1":    "ResNet-50 (ft MAD-5-sub1)",
    "resnet50_ft_mad_sub1":      "ResNet-50 (ft MAD-sub1)",
    "resnet50_ft_mad_sub23":     "ResNet-50 (ft MAD-sub23)",
    "resnet50_ft_mad_sub12":     "ResNet-50 (ft MAD-sub12)",
    "resnet50_ft_glasgow_young": "ResNet-50 (ft Glasgow-young)",
}

_PROBE_TITLES = {
    "knn_k10":     "k-NN  (k = 10)",
    "linear_C1.0": "Linear probe",
    "supervised":  "Supervised (fine-tuned)",
}

_MAE_IMAGENET_KEY = "mae_imagenet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _registry_key(model_name: str) -> str:
    _MAP = {"fisher_small_4band": "fisher_small"}
    return _MAP.get(model_name, model_name)


def load_results(input_dirs: list[Path], models: list | None) -> dict:
    """Return {(model, dataset): result_dict}, keeping latest run per pair."""
    latest: dict = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.json")):
            if path.name.startswith("run_"):
                continue
            data = json.loads(path.read_text())
            if "n_repeats" not in data:
                continue
            key = (data["model"], data["dataset"])
            latest[key] = data

    if models:
        latest = {k: v for k, v in latest.items()
                  if _registry_key(v["model"]) in models}
    return latest


def _get_full_value(probe_data: list, metric_key: str) -> float | None:
    """Return the 'full' (n_shots=None) entry's metric value, or None."""
    for entry in probe_data:
        if entry["n_shots"] is None and entry[metric_key] is not None:
            return entry[metric_key] * 100
    return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_dataset(
    dataset_name: str,
    ds_results: dict,
    probe_names: list,
    metric: str,
    output_path: Path,
) -> None:
    """Render a single-dataset figure with one column per probe."""
    metric_key   = f"mean_{metric}"
    std_key      = f"std_{metric}"
    metric_label = "Accuracy (%)" if metric == "acc" else "Macro-F1 (%)"

    n_cols = len(probe_names)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(6.5 * n_cols, 4.8),
        squeeze=False,
    )

    legend_handles: dict = {}

    for col_idx, probe_name in enumerate(probe_names):
        ax = axes[0][col_idx]

        mae_full_val: float | None = None
        for data in ds_results.values():
            if _registry_key(data["model"]) == _MAE_IMAGENET_KEY:
                pd = data["probes"].get(probe_name)
                if pd:
                    mae_full_val = _get_full_value(pd, metric_key)

        for data in sorted(ds_results.values(),
                           key=lambda d: _registry_key(d["model"])):
            probe_data = data["probes"].get(probe_name)
            if probe_data is None:
                continue

            rkey   = _registry_key(data["model"])
            colour = _PALETTE.get(rkey, _PALETTE_DEFAULT)
            label  = _DISPLAY_NAMES.get(rkey, rkey)

            few_entries  = [e for e in probe_data
                            if e["n_shots"] is not None and e[metric_key] is not None]
            full_entries = [e for e in probe_data
                            if e["n_shots"] is None  and e[metric_key] is not None]

            xs   = np.array([e["n_total_train"] for e in few_entries])
            ys   = np.array([e[metric_key] * 100  for e in few_entries])
            errs = np.array([e[std_key]    * 100  for e in few_entries])

            line, = ax.plot(
                xs, ys,
                marker="o", markersize=3.5, linewidth=1.8,
                color=colour, label=label,
            )
            ax.fill_between(xs, ys - errs, ys + errs,
                            alpha=0.12, color=colour)

            if full_entries:
                fx = full_entries[0]["n_total_train"]
                fy = full_entries[0][metric_key] * 100
                ax.scatter([fx], [fy], marker="*", s=130,
                           color=colour, zorder=6)

            if rkey != _MAE_IMAGENET_KEY and label not in legend_handles:
                legend_handles[label] = (line, fy if full_entries else ys[-1] if len(ys) else 0)

        if mae_full_val is not None:
            hline = ax.axhline(
                y=mae_full_val,
                linestyle="--", linewidth=1.4,
                color=_PALETTE.get(_MAE_IMAGENET_KEY, "#607D8B"),
                alpha=0.85, zorder=3,
            )
            ax.text(
                0.98, mae_full_val,
                f" MAE-ImageNet (full)  {mae_full_val:.1f}%",
                transform=ax.get_yaxis_transform(),
                fontsize=7.5,
                color=_PALETTE.get(_MAE_IMAGENET_KEY, "#607D8B"),
                va="bottom", ha="right",
            )
            mae_label = "MAE-ImageNet (full data)"
            if mae_label not in legend_handles:
                legend_handles[mae_label] = (hline, -1)

        ax.set_xscale("log")
        ax.set_xlabel("Training samples", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(
            _PROBE_TITLES.get(probe_name, probe_name),
            fontsize=12, fontweight="bold", pad=8,
        )
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":",  alpha=0.15)
        ax.tick_params(labelsize=9)

    sorted_items = sorted(legend_handles.items(), key=lambda kv: -kv[1][1])
    handles = [v[0] for _, v in sorted_items]
    labels  = [k     for k, _ in sorted_items]

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
        borderaxespad=0.5,
    )

    fig.suptitle(
        f"Sample efficiency — {dataset_name.upper()} — {metric_label}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def plot(
    results: dict,
    probe_names: list,
    metric: str,
    dataset_filter: str | None,
    output_path: Path,
) -> None:
    datasets = sorted({v["dataset"] for v in results.values()})
    if dataset_filter:
        datasets = [d for d in datasets if d == dataset_filter]
    if not datasets:
        print("No matching data found.")
        return

    for dataset_name in datasets:
        ds_results = {k: v for k, v in results.items()
                      if v["dataset"] == dataset_name}

        # Suffix the output path with the dataset name so each figure is distinct.
        ds_output = output_path.with_name(
            f"{output_path.stem}_{dataset_name}{output_path.suffix}"
        )
        _plot_dataset(
            dataset_name=dataset_name,
            ds_results=ds_results,
            probe_names=probe_names,
            metric=metric,
            output_path=ds_output,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", nargs="+",
                   default=[str(_DEFAULT_INPUT_DIR),
                            str(REPO_ROOT / "results" / "cross_few_shot")])
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these registry keys (default: all found)")
    p.add_argument("--dataset", default=None,
                   help="Restrict to this dataset (default: all)")
    p.add_argument("--probe", nargs="+", default=["linear_C1.0"],
                   choices=["knn_k10", "linear_C1.0", "supervised"],
                   help="Which probe columns to plot (default: linear)")
    p.add_argument("--metric", default="acc", choices=["acc", "f1"],
                   help="Metric to plot (default: acc)")
    p.add_argument("--output",
                   default=str(_DEFAULT_OUTPUT_DIR / "sample_efficiency.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(d) for d in args.input_dir if Path(d).is_dir()]
    results = load_results(input_dirs, args.models)
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
