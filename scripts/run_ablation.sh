#!/usr/bin/env bash
# Ablation table reproducer (glasgow_5 → mad_5).
#
# Rows (each is a leave-one-out from "Ours" unless otherwise noted):
#   1. Ours: DINOv3 + PiSSA (all)               method=pissa     targets=all  variant=vits16
#   2. — w/o MLP adaptation (PiSSA, attn)       method=pissa     targets=attn variant=vits16
#   3. — w/o PiSSA init   (LoRA, all)           method=lora      targets=all  variant=vits16
#   4. — w/o both         (LoRA, attn)          method=lora      targets=attn variant=vits16
#   5. — w/o SSL          (ImageNet + PiSSA all) method=pissa_in targets=all  variant=vits16
#   6. — w/o SSL + LoRA attn  (ImageNet + LoRA attn) method=lora_in targets=attn variant=vits16
#   7. — w/o PEFT         (DINOv3 + full FT)     method=full_ft  (rank/targets ignored)
#   8. Frozen anchor      (DINOv3 ViT-S frozen)  via run_cross_few_shot.py --freeze
#
# Shared hyperparameters: ep=60, lr=1e-4, r=8 (where applicable), 3 seeds, n_repeats=10.
# Idempotent — run_peft_sweep.py and run_cross_few_shot.py both skip on existing files.
#
# Output:
#   PEFT rows  → results/peft_sweep/ep60/{cfg}__glasgow_5__mad_5.json
#   Frozen row → results/cross_few_shot/dinov3_vits16_frozen_frozen_glasgow_5__mad_5__{ts}.json

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
EP60_OUT=results/peft_sweep/ep60
CFS_OUT=results/cross_few_shot
mkdir -p "$EP60_OUT" "$CFS_OUT"

SRC=glasgow_5
TGT=mad_5

# ── Rows 1–4: existing PEFT cells (skipped if already present) ──────────
for cfg in "pissa all" "pissa attn" "lora all" "lora attn"; do
  read -r METHOD TARGETS <<<"$cfg"
  echo ">>> $METHOD ($TARGETS)  $SRC -> $TGT"
  $PY scripts/run_peft_sweep.py \
      --methods "$METHOD" --targets "$TARGETS" \
      --ranks 8 --lrs 1e-4 \
      --epochs 60 --seeds 0 1 2 \
      --variant vits16 \
      --source "$SRC" --target "$TGT" \
      --output-dir "$EP60_OUT"
done

# ── Row 5: w/o SSL  (ImageNet ViT-S + PiSSA all) ────────────────────────
echo ">>> pissa_in (all)  $SRC -> $TGT"
$PY scripts/run_peft_sweep.py \
    --methods pissa_in --targets all \
    --ranks 8 --lrs 1e-4 \
    --epochs 60 --seeds 0 1 2 \
    --variant vits16 \
    --source "$SRC" --target "$TGT" \
    --output-dir "$EP60_OUT"

# ── Row 6: w/o SSL + LoRA attn  (ImageNet ViT-S + LoRA attn) ────────────
echo ">>> lora_in (attn)  $SRC -> $TGT"
$PY scripts/run_peft_sweep.py \
    --methods lora_in --targets attn \
    --ranks 8 --lrs 1e-4 \
    --epochs 60 --seeds 0 1 2 \
    --variant vits16 \
    --source "$SRC" --target "$TGT" \
    --output-dir "$EP60_OUT"

# ── Row 7: w/o PEFT  (DINOv3 ViT-S full FT) ─────────────────────────────
# rank/targets are ignored by the full_ft factory but must be passed for the
# sweep arg grammar; the dedup logic collapses them to one (lr, seed, variant).
echo ">>> full_ft  $SRC -> $TGT"
$PY scripts/run_peft_sweep.py \
    --methods full_ft --targets all \
    --ranks 8 --lrs 1e-4 \
    --epochs 60 --seeds 0 1 2 \
    --variant vits16 \
    --source "$SRC" --target "$TGT" \
    --output-dir "$EP60_OUT"

# ── Row 8: Frozen anchor  (run_cross_few_shot.py --freeze) ──────────────
# Already produced by run_final_method_comparison.sh; re-run if missing.
# run_cross_few_shot.py does NOT skip-by-filename so we guard here.
FROZEN_FILES=( "$CFS_OUT"/dinov3_vits16_frozen_frozen_${SRC}__${TGT}__*.json )
if [[ ! -e "${FROZEN_FILES[0]}" ]]; then
  echo ">>> dinov3_vits16 frozen  $SRC -> $TGT"
  $PY scripts/run_cross_few_shot.py \
      --model dinov3_vits16_frozen \
      --source-dataset "$SRC" \
      --target-dataset "$TGT" \
      --freeze \
      --output-dir "$CFS_OUT"
else
  echo "skip (exists): ${FROZEN_FILES[0]##*/}"
fi

echo
echo "Ablation runs complete.  Aggregate with the chat-side analysis script."
