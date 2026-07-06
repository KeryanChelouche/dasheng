#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Re-run all Glasgow experiments with subject-wise CV splits.
#
# Resumable: completed steps are recorded in a progress file.
# Kill with Ctrl-C at any time; re-run the same script to pick up
# where you left off.  To start over, delete the progress file:
#   rm results/.rerun_progress
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

PROGRESS_FILE="results/.rerun_progress"
touch "$PROGRESS_FILE"

run_step() {
    local tag="$1"; shift
    if grep -qxF "$tag" "$PROGRESS_FILE"; then
        echo "  [skip] $tag (already done)"
        return 0
    fi
    echo "  [run]  $tag"
    "$@"
    echo "$tag" >> "$PROGRESS_FILE"
}

# ── 1. Standard evaluation (metrics/) ────────────────────────────────
echo "=== [1/4] Standard evaluation (metrics/) ==="

run_step "eval:dasheng_base+06B+dinov3:glasgow" \
    python scripts/run_eval.py \
        --model dasheng_base dasheng_06B dinov3_vitb16 \
        --dataset glasgow

run_step "eval:audiomae+beats:glasgow" \
    python scripts/run_eval.py \
        --model audiomae beats_iter3 "beats_iter3+" \
        --dataset glasgow

run_step "eval:fisher+mae:glasgow" \
    python scripts/run_eval.py \
        --model fisher_small mae_imagenet \
        --dataset glasgow

run_step "eval:whisper+qwen2:glasgow" \
    python scripts/run_eval.py \
        --model whisper_small whisper_large_v3 qwen2_audio \
        --dataset glasgow

run_step "eval:vjepa2:glasgow" \
    python scripts/run_eval.py \
        --model "vjepa2.1_vitb" \
        --dataset glasgow

run_step "eval:resnet50:glasgow" \
    python scripts/run_eval.py \
        --model resnet50 \
        --dataset glasgow

# ── 2. Few-shot sample efficiency (few_shot/) ────────────────────────
echo "=== [2/4] Few-shot evaluation (few_shot/) ==="

run_step "fewshot:dasheng_base:glasgow" \
    python scripts/run_few_shot.py \
        --model dasheng_base \
        --dataset glasgow

run_step "fewshot:dinov3:glasgow+glasgow_5" \
    python scripts/run_few_shot.py \
        --model dinov3_vitb16 \
        --dataset glasgow glasgow_5

run_step "fewshot:audiomae+beats+fisher:glasgow" \
    python scripts/run_few_shot.py \
        --model audiomae "beats_iter3+" beats_iter3 fisher_small \
        --dataset glasgow

run_step "fewshot:mae+whisper:glasgow" \
    python scripts/run_few_shot.py \
        --model mae_imagenet whisper_small \
        --dataset glasgow

# ── 3. Cross few-shot (cross_few_shot/) ──────────────────────────────
echo "=== [3/4] Cross few-shot (cross_few_shot/) ==="

run_step "xfewshot:resnet50:mad5->glasgow5" \
    python scripts/run_cross_few_shot.py \
        --model resnet50 \
        --source-dataset mad_5 \
        --target-dataset glasgow_5

run_step "xfewshot:resnet50:imagenet->glasgow5" \
    python scripts/run_cross_few_shot.py \
        --model resnet50 \
        --target-dataset glasgow_5

run_step "xfewshot:dinov3_lora:mad5->glasgow5" \
    python scripts/run_cross_few_shot.py \
        --model dinov3_lora \
        --source-dataset mad_5 \
        --target-dataset glasgow_5

run_step "xfewshot:dinov3_lora:glasgow5->mad5" \
    python scripts/run_cross_few_shot.py \
        --model dinov3_lora \
        --source-dataset glasgow_5 \
        --target-dataset mad_5

# ── 4. Done ──────────────────────────────────────────────────────────
echo ""
echo "=== All Glasgow experiments re-run with subject-wise CV ==="
echo "New results are in results/{metrics,few_shot,cross_few_shot}/."
echo "Old (defective) results remain in results/defective_random_kfold/."
echo "Progress file: $PROGRESS_FILE"
