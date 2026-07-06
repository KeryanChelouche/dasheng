#!/usr/bin/env bash
# Full evaluation sweep for ONE model — orchestrates the existing CLI
# scripts (run_eval / run_few_shot / run_cross_eval / run_cross_few_shot).
#
# Stages (skip individually with the SKIP_* env vars):
#   1. in-domain        Glasgow + MAD: full kNN/linear + few-shot
#   2. cross-direct     4 transfer pairs, probe trained on source
#   3. cross-fewshot    4 transfer pairs, few-shot probe on target
#
# Supervised models (resnet50, dinov3_lora) auto-trigger source-side
# adaptation (full FT for ResNet, LoRA for DINOv3) in the cross stages
# because that is the default for the underlying scripts.
#
# Usage:
#   scripts/run_full_sweep.sh <model> [extra args forwarded to every step]
#
# Examples:
#   scripts/run_full_sweep.sh dasheng_base
#   scripts/run_full_sweep.sh resnet50 --batch-size 8
#   scripts/run_full_sweep.sh dinov3_vitb16 --n-shots 1 5 10 50
#
# Skip stages by setting any of:
#   SKIP_IN_DOMAIN=1 SKIP_CROSS_DIRECT=1 SKIP_CROSS_FEWSHOT=1
#
# n-shots / n-repeats can be overridden by setting:
#   N_SHOTS="1 2 5 10 20 50 100 200"   N_REPEATS=10
# (these only apply to the few-shot stages — they are passed positionally
# below and respected by run_few_shot.py and run_cross_few_shot.py).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model> [extra args]" >&2
    exit 1
fi

MODEL="$1"
shift
EXTRA=("$@")

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv if present (no-op if we're already in one).
if [[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

N_SHOTS="${N_SHOTS:-1 2 5 10 20 50 100 200}"
N_REPEATS="${N_REPEATS:-10}"

# Models that go through SUPERVISED_REGISTRY (source-side adaptation
# is the default for run_cross_few_shot.py / run_cross_eval.py).
# Keep in sync with scripts/run_cross_few_shot.py SUPERVISED_REGISTRY.
SUPERVISED_MODELS=(
    resnet50
    # LoRA — auto-discovered as the "${ssl_model}_lora" counterpart
    dinov3_lora dinov3_vits16_lora
    vit_base_imagenet_lora vit_small_imagenet_lora
    dasheng_lora dasheng_06B_lora dasheng_12B_lora
    # DoRA — opt-in: pass the *_dora name explicitly to run_full_sweep.sh
    # for an apples-to-apples ablation against LoRA. Not auto-discovered.
    dinov3_dora dinov3_vits16_dora
    vit_base_imagenet_dora vit_small_imagenet_dora
    dasheng_dora dasheng_06B_dora dasheng_12B_dora
)

is_supervised() {
    local m
    for m in "${SUPERVISED_MODELS[@]}"; do
        [[ "$m" == "$1" ]] && return 0
    done
    return 1
}

# For an SSL ViT base model, the natural source-adapted counterpart.
# Naming isn't strictly "${model}_lora" — the default variant is dropped
# in the LoRA registry (e.g. dasheng_base → dasheng_lora,
# dinov3_vitb16 → dinov3_lora) while non-default variants keep the
# tag (dasheng_06B → dasheng_06B_lora).  Try the literal name first,
# then fall back to stripping the last "_<variant>" suffix.
lora_counterpart() {
    local cand="${1}_lora"
    if is_supervised "$cand"; then
        echo "$cand"
        return
    fi
    cand="${1%_*}_lora"
    if [[ "$cand" != "${1}_lora" ]] && is_supervised "$cand"; then
        echo "$cand"
    fi
}

IN_DOMAIN_DATASETS=(glasgow mad)
# (source, target) pairs for the cross stages.
CROSS_PAIRS=(
    "glasgow_young glasgow_mature"
    "mad_sub12     mad_sub3"
    "glasgow_5     mad_5"
    "mad_5         glasgow_5"
)

banner() { printf '\n━━━ %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n' "$*"; }

run() { echo "+ $*"; "$@"; }

# ── Stage 1: in-domain ──────────────────────────────────────────────────────
if [[ -z "${SKIP_IN_DOMAIN:-}" ]]; then
    for ds in "${IN_DOMAIN_DATASETS[@]}"; do
        banner "IN-DOMAIN FULL: $MODEL on $ds"
        run python scripts/run_eval.py \
            --model "$MODEL" --dataset "$ds" "${EXTRA[@]}"

        banner "IN-DOMAIN FEW-SHOT: $MODEL on $ds"
        # shellcheck disable=SC2086
        run python scripts/run_few_shot.py \
            --model "$MODEL" --dataset "$ds" \
            --n-shots $N_SHOTS --n-repeats "$N_REPEATS" "${EXTRA[@]}"
    done
fi

# ── Stage 2: cross-direct ───────────────────────────────────────────────────
# Always run the base model (frozen probe + scheme remap, or supervised FT).
# When the base model is an SSL ViT and a "${MODEL}_lora" counterpart exists,
# also run the source-adapted version for free.
if [[ -z "${SKIP_CROSS_DIRECT:-}" ]]; then
    LORA_NAME="$(lora_counterpart "$MODEL")"
    for pair in "${CROSS_PAIRS[@]}"; do
        read -r SRC TGT <<<"$pair"
        banner "CROSS-DIRECT: $MODEL  $SRC → $TGT"
        run python scripts/run_cross_eval.py \
            --model "$MODEL" \
            --train-dataset "$SRC" --test-dataset "$TGT" \
            "${EXTRA[@]}"

        if [[ -n "$LORA_NAME" ]]; then
            banner "CROSS-DIRECT (LoRA): $LORA_NAME  $SRC → $TGT"
            run python scripts/run_cross_eval.py \
                --model "$LORA_NAME" \
                --train-dataset "$SRC" --test-dataset "$TGT" \
                "${EXTRA[@]}"
        fi
    done
fi

# ── Stage 3: cross-fewshot ──────────────────────────────────────────────────
# Supervised: source-FT then few-shot on target via run_cross_few_shot.py.
# SSL:        no source adaptation — equivalent is frozen-feature few-shot
#             on the target via run_few_shot.py (target labels already
#             match the shared scheme for all four registered pairs).
if [[ -z "${SKIP_CROSS_FEWSHOT:-}" ]]; then
    LORA_NAME="$(lora_counterpart "$MODEL")"
    if is_supervised "$MODEL"; then
        for pair in "${CROSS_PAIRS[@]}"; do
            read -r SRC TGT <<<"$pair"
            banner "CROSS-FEWSHOT (FT): $MODEL  $SRC → $TGT"
            # shellcheck disable=SC2086
            run python scripts/run_cross_few_shot.py \
                --model "$MODEL" \
                --source-dataset "$SRC" --target-dataset "$TGT" \
                --n-shots $N_SHOTS --n-repeats "$N_REPEATS" \
                "${EXTRA[@]}"
        done
    else
        # Unique targets only — frozen few-shot on the same target gives
        # the same numbers regardless of which source it's paired with.
        SEEN=()
        for pair in "${CROSS_PAIRS[@]}"; do
            read -r _ TGT <<<"$pair"
            if [[ " ${SEEN[*]} " == *" $TGT "* ]]; then continue; fi
            SEEN+=("$TGT")
            banner "CROSS-FEWSHOT (frozen): $MODEL on target $TGT"
            # shellcheck disable=SC2086
            run python scripts/run_few_shot.py \
                --model "$MODEL" --dataset "$TGT" \
                --n-shots $N_SHOTS --n-repeats "$N_REPEATS" \
                "${EXTRA[@]}"
        done
    fi

    # Source-adapted counterpart (if registered) — runs the per-pair
    # supervised cross-fewshot path with checkpoint caching on disk.
    if [[ -n "$LORA_NAME" ]]; then
        for pair in "${CROSS_PAIRS[@]}"; do
            read -r SRC TGT <<<"$pair"
            banner "CROSS-FEWSHOT (LoRA): $LORA_NAME  $SRC → $TGT"
            # shellcheck disable=SC2086
            run python scripts/run_cross_few_shot.py \
                --model "$LORA_NAME" \
                --source-dataset "$SRC" --target-dataset "$TGT" \
                --n-shots $N_SHOTS --n-repeats "$N_REPEATS" \
                "${EXTRA[@]}"
        done
    fi
fi

banner "DONE: $MODEL"
