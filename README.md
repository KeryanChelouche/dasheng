# Dasheng — Cross-Domain Spectrogram Evaluation Fork

> **Research fork of [Dasheng (大声)](https://github.com/XiaoMi/dasheng)**
> (Dinkel et al., Interspeech 2024)
>
> **Hypothesis:** Pretrained audio foundation models are not merely
> audio-specific encoders — they are general-purpose spectrogram
> foundation models whose representations transfer to any domain where
> data can be expressed as a 2D time-frequency image (radar, EEG,
> vibration, seismology, …).

---

## Research goal

Audio foundation models (AudioMAE, Dasheng, BEATs, …) operate on
ViT-style patch embeddings of Mel-spectrograms.  They never process
raw waveforms — they process 2D images.  If the features they learn
are truly about time-frequency structure, those features should
generalise beyond audio to other spectrogram-like domains.

This fork provides a modular evaluation framework to test that
hypothesis systematically, across multiple models and multiple
non-audio spectrogram datasets.

---

## Repository layout

```
dasheng/                      # Upstream model code (unchanged)
│   ├── __init__.py           # Factory functions for all model variants
│   ├── pretrained/
│   │   └── pretrained.py     # Dasheng, DashengSpectrogram, factory fns
│   └── train/                # Training infrastructure (MAE, WebDataset)
│
eval/                         # Evaluation framework (this fork)
│   ├── models/
│   │   ├── base.py           # FeatureExtractor ABC
│   │   └── dasheng.py        # DashengExtractor wrapper
│   ├── datasets/
│   │   ├── base.py           # SpectrogramDataset ABC
│   │   ├── glasgow.py        # Glasgow micro-Doppler HAR
│   │   └── esc50.py          # ESC-50 audio (Mel-spec conversion)
│   ├── probes/
│   │   ├── base.py           # Probe ABC
│   │   ├── knn.py            # k-NN probe
│   │   └── linear.py         # Linear (logistic regression) probe
│   ├── features.py           # Feature extraction + NPZ caching
│   ├── evaluation.py         # Cross-validation pipeline
│   └── reporting.py          # Tables, JSON persistence, plots
│
scripts/
│   ├── run_eval.py           # Main CLI: model × dataset × probe(s)
│   └── compare_results.py    # Aggregate JSON results → Markdown table
│
configs/
│   └── experiments/          # Reference YAML configs for reproducibility
│       ├── glasgow_dasheng_base.yaml
│       └── esc50_dasheng_base.yaml
│
data/                         # Git-ignored datasets
│   ├── Glasgow/              # 2081 micro-Doppler .npy files, 6 classes
│   └── ESC-50-master/        # 2000 .wav files, 50 classes
│
results/                      # Git-ignored experiment outputs
    ├── features/             # Cached .npz feature matrices
    ├── metrics/              # Per-run .json results + logs
    └── plots/                # PNG figures
```

---

## Quickstart

**Environment:** Python 3.14, `.venv/` (torch 2.10+cu130).

```bash
# Activate the virtual environment
source .venv/bin/activate

# k-NN + linear probe on Glasgow micro-Doppler with Dasheng-base
python scripts/run_eval.py --model dasheng_base --dataset glasgow

# Grid: compare two model scales on two datasets
python scripts/run_eval.py \
    --model dasheng_base dasheng_06B \
    --dataset glasgow esc50 \
    --probes knn linear

# Print a comparison table of all saved results
python scripts/compare_results.py
```

Results are saved as JSON files in `results/metrics/` and feature
matrices are cached in `results/features/` so re-runs skip extraction.

---

## Adding a new model

1. Create `eval/models/<name>.py` implementing `FeatureExtractor`:

```python
from eval.models.base import FeatureExtractor

class MyModelExtractor(FeatureExtractor):
    @property
    def name(self) -> str: return "my_model"

    @property
    def embed_dim(self) -> int: return 768

    def extract(self, x: torch.Tensor) -> np.ndarray:
        # x: (B, F, T) spectrogram — return (B, D) features
        ...
```

2. Register it in `scripts/run_eval.py`:

```python
MODEL_REGISTRY["my_model"] = lambda: MyModelExtractor()
```

That's it.  No other files need to change.

---

## Adding a new dataset

1. Create `eval/datasets/<name>.py` implementing `SpectrogramDataset`.
   The `load_item()` method must return a `(1, F, T)` float32 tensor —
   audio files should be converted to spectrograms there.

2. Register it in `scripts/run_eval.py`:

```python
DATASET_REGISTRY["my_dataset"] = lambda: MyDataset(ROOT / "data" / "MyDataset")
```

---

## Design notes

- **All models receive spectrograms.**  Audio datasets (ESC-50) convert
  to log-Mel spectrograms inside `load_item()`; model wrappers handle
  any frequency-bin resizing.  This keeps the pipeline uniform and
  makes future models (AudioMAE, ViT, …) easy to drop in.

- **Feature caching.**  Extraction is the expensive step.
  `results/features/<model>__<dataset>.npz` is written after the first
  run.  Delete the file or pass `--no-cache` to force re-extraction.

- **No training.**  The framework evaluates frozen pretrained features
  only (k-NN and linear probing).  Fine-tuning experiments are planned
  for a later phase.

---

## Upstream: Dasheng

Dasheng is a scaled-up masked audio encoder (AudioMAE-style) pretrained
on 272 k hours of audio.  See the
[original paper](https://arxiv.org/abs/2406.06992) and
[upstream repo](https://github.com/XiaoMi/dasheng) for full details.

| Model | Params | HEAR avg (env / speech / music) |
|---|---|---|
| Dasheng-Base | 86 M | 80.2 / 72.5 / 84.0 |
| Dasheng-0.6B | 600 M | 82.4 / 74.9 / 84.0 |
| Dasheng-1.2B | 1200 M | 83.2 / 75.7 / 84.9 |

Pretrained checkpoints are downloaded automatically from
[Zenodo](https://zenodo.org/records/11511780) on first use.

---

## Citation

```bibtex
@inproceedings{dinkel2024dasheng,
  title={Scaling up masked audio encoder learning for general audio classification},
  author={Dinkel, Heinrich and Yan, Zhiyong and Wang, Yongqing and Zhang, Junbo
          and Wang, Yujun and Wang, Bin},
  booktitle={Interspeech 2024},
  year={2024}
}
```
