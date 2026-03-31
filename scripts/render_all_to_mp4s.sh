#!/bin/bash
# Convert all BVH files under bvhs/ into mp4s/, preserving folder structure.
#
# bvhs/<Animal>/*.bvh  →  mp4s/<Animal>/*.mp4
#
# Usage: ./render_all_to_mp4s.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BVH_ROOT="$SCRIPT_DIR/dataset/truebones/zoo/truebones_processed/bvhs"
MP4_ROOT="$SCRIPT_DIR/dataset/truebones/zoo/truebones_processed/mp4s"
RENDER_SCRIPT="$HOME/Github/BVHView/render_bvhs.sh"

if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "ERROR: render script not found at $RENDER_SCRIPT"
    exit 1
fi

if [ ! -d "$BVH_ROOT" ]; then
    echo "ERROR: BVH root not found: $BVH_ROOT"
    exit 1
fi

FOLDERS=($(find "$BVH_ROOT" -mindepth 1 -maxdepth 1 -type d | sort))
TOTAL=${#FOLDERS[@]}

echo "============================================"
echo "BVH Root : $BVH_ROOT"
echo "MP4 Root : $MP4_ROOT"
echo "Folders  : $TOTAL"
echo "============================================"

COUNT=0
ERRORS=0

for BVH_DIR in "${FOLDERS[@]}"; do
    ANIMAL=$(basename "$BVH_DIR")
    OUT_DIR="$MP4_ROOT/$ANIMAL"
    COUNT=$((COUNT + 1))

    echo ""
    echo "[$COUNT/$TOTAL] $ANIMAL"
    echo "  BVH : $BVH_DIR"
    echo "  MP4 : $OUT_DIR"

    mkdir -p "$OUT_DIR"

    if bash "$RENDER_SCRIPT" "$BVH_DIR" "$OUT_DIR"; then
        echo "  OK"
    else
        echo "  ERROR"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "============================================"
echo "Done."
echo "  Success : $((COUNT - ERRORS))"
echo "  Errors  : $ERRORS"
echo "  Output  : $MP4_ROOT"
echo "============================================"
