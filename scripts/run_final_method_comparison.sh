#!/usr/bin/env bash
# Re-run missing/under-spec experiments for the final method-comparison table.
#
# Final hyperparameters:
#   FT methods:  ep=60, lr=1e-4, n_repeats=10, seeds={0,1,2}
#     PiSSA  → DINOv3 ViT-S, r=8, targets=all
#     LoRA   → DINOv3 ViT-S, r=8, targets=attn
#     SelaFD → ImageNet ViT-S + LoRA(q,v) r=alpha=4 + serial/parallel
#              adapters (ratio=0.5, s=0.2) — paper-pinned (run_peft_sweep
#              ignores --ranks/--targets for this method).
#     ResNet50 → standard ft (no LoRA), ep=60
#   Frozen:      no FT, seed=42, n_repeats=10  (deterministic — one run)
#
# Output:
#   PEFT runs       → results/peft_sweep/ep60/*.json
#   ResNet50 runs   → results/cross_few_shot/*_s{0,1,2}_*.json
#   Frozen runs     → results/cross_few_shot/*_frozen_*.json
#
# Already-cached results are skipped automatically by run_peft_sweep.py.
# run_cross_few_shot.py overwrites — re-runs everything every time.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
EP60_OUT=results/peft_sweep/ep60
CFS_OUT=results/cross_few_shot
mkdir -p "$EP60_OUT" "$CFS_OUT"

PAIRS=(
  "glasgow_young  glasgow_mature"
  "mad_sub12      mad_sub3"
  "mad_5          glasgow_5"
  "glasgow_5      mad_5"
)

# ── 1. PEFT: PiSSA all  (Ours)  ──────────────────────────────────────────
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> PiSSA all  $SRC -> $TGT"
  $PY scripts/run_peft_sweep.py \
      --methods pissa --targets all \
      --ranks 8 --lrs 1e-4 \
      --epochs 60 --seeds 0 1 2 \
      --source "$SRC" --target "$TGT" \
      --output-dir "$EP60_OUT"
done

# ── 2. PEFT: LoRA attn  (PEFT baseline)  ─────────────────────────────────
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> LoRA attn  $SRC -> $TGT"
  $PY scripts/run_peft_sweep.py \
      --methods lora --targets attn \
      --ranks 8 --lrs 1e-4 \
      --epochs 60 --seeds 0 1 2 \
      --source "$SRC" --target "$TGT" \
      --output-dir "$EP60_OUT"
done

# ── 3. PEFT: ImageNet ViT-S + LoRA attn  (final spec, ablation comparator) ─
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> lora_in attn  $SRC -> $TGT"
  $PY scripts/run_peft_sweep.py \
      --methods lora_in --targets attn \
      --ranks 8 --lrs 1e-4 \
      --epochs 60 --seeds 0 1 2 \
      --variant vits16 \
      --source "$SRC" --target "$TGT" \
      --output-dir "$EP60_OUT"
done

# ── 4. PEFT: SelaFD  (ImageNet ViT-S, paper-pinned)  ─────────────────────
# rank/alpha=4, adapter ratio=0.5, parallel scale=0.2 are pinned inside
# run_peft_sweep.py — only (lr, seed) vary.
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> SelaFD  $SRC -> $TGT"
  $PY scripts/run_peft_sweep.py \
      --methods selafd \
      --lrs 1e-4 \
      --epochs 60 --seeds 0 1 2 \
      --variant vits16 \
      --source "$SRC" --target "$TGT" \
      --output-dir "$EP60_OUT"
done

# ── 5. ResNet50 FT  (3 seeds, ep=60)  ────────────────────────────────────
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> ResNet50 ft  $SRC -> $TGT"
  $PY scripts/run_cross_few_shot.py \
      --model resnet50 \
      --source-dataset "$SRC" \
      --target-dataset "$TGT" \
      --epochs 60 --lr 1e-4 \
      --seeds 0 1 2 \
      --output-dir "$CFS_OUT"
done

# ── 6. Frozen DINOv3 ViT-S  (no FT, one run per pair)  ───────────────────
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> DINOv3 ViT-S frozen  $SRC -> $TGT"
  $PY scripts/run_cross_few_shot.py \
      --model dinov3_vits16_frozen \
      --source-dataset "$SRC" \
      --target-dataset "$TGT" \
      --freeze \
      --output-dir "$CFS_OUT"
done

# ── 7. Frozen ViT-S ImageNet  (no FT, one run per pair)  ─────────────────
for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  echo ">>> ViT-S ImageNet frozen  $SRC -> $TGT"
  $PY scripts/run_cross_few_shot.py \
      --model vit_small_imagenet_frozen \
      --source-dataset "$SRC" \
      --target-dataset "$TGT" \
      --freeze \
      --output-dir "$CFS_OUT"
done

echo
echo "All method-comparison experiments queued."
