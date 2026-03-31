#!/usr/bin/env bash
# TODO: debug this 
# Retarget a source motion to all skeletons in a target group.
#
# Usage:
#   bash scripts/retarget_group.sh \
#       --model_path  save/.../model000189999.pt \
#       --source_motion dataset/.../Alligator_Alligator_BigMouth_11.npy \
#       --source_type Alligator \
#       --target_group quadropeds \
#       [--num_repetitions 1]

# bash scripts/retarget_group.sh \
#     --model_path save/20260325_Retarget_1_latentdim_128_src_BrownBear_selfRecon_cycle/model000500000.pt \
#     --source_motion dataset/truebones/zoo/truebones_processed/motions/Alligator_Alligator_BigMouth_11.npy \
#     --source_type BrownBear \
#     --target_group quadropeds \
#     --num_repetitions 3

# target_group choices: quadropeds, bipeds, flying, millipeds,
#                       millipeds_snakes, quadropeds_clean, bipeds_clean,
#                       flying_clean, millipeds_clean, all, all_clean

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL_PATH=""
SOURCE_MOTION=""
SOURCE_TYPE=""
TARGET_GROUP="quadropeds"
NUM_REPETITIONS=1

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)        MODEL_PATH="$2";        shift 2 ;;
        --source_motion)     SOURCE_MOTION="$2";     shift 2 ;;
        --source_type)       SOURCE_TYPE="$2";       shift 2 ;;
        --target_group)      TARGET_GROUP="$2";      shift 2 ;;
        --num_repetitions)   NUM_REPETITIONS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ───────────────────────────────────────────────────
if [[ -z "$MODEL_PATH" || -z "$SOURCE_MOTION" || -z "$SOURCE_TYPE" ]]; then
    echo "Error: --model_path, --source_motion, --source_type are required."
    exit 1
fi

# ── Resolve skeleton list from group ────────────────────────────────────────
SKELETONS=$(python3 - <<EOF
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
group = "$TARGET_GROUP"
if group not in OBJECT_SUBSETS_DICT:
    raise ValueError(f"Unknown group '{group}'. Choices: {list(OBJECT_SUBSETS_DICT)}")
print(" ".join(OBJECT_SUBSETS_DICT[group]))
EOF
)

echo "============================================================"
echo "Model      : $MODEL_PATH"
echo "Source     : $SOURCE_TYPE  ($SOURCE_MOTION)"
echo "Group      : $TARGET_GROUP"
echo "Targets    : $SKELETONS"
echo "Repetitions: $NUM_REPETITIONS"
echo "============================================================"

# ── Run retargeting ──────────────────────────────────────────────────────────
# sample.retarget accepts --object_type as a space-separated list,
# so we pass all targets in a single call (batched inference).
python -m sample.retarget \
    --model_path "$MODEL_PATH" \
    --source_motion "$SOURCE_MOTION" \
    --source_type "$SOURCE_TYPE" \
    --num_repetitions "$NUM_REPETITIONS" \
    --object_type $SKELETONS

echo "Done."
