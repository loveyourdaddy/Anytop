#!/bin/bash
# Render all BVH folders in dataset/truebones/zoo/truebones_processed/bvhs/ to MP4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BVH_ROOT="$SCRIPT_DIR/dataset/truebones/zoo/truebones_processed/bvhs"
RENDER_SCRIPT="$HOME/Github/BVHView/render_bvhs.sh"

if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "ERROR: render script not found at $RENDER_SCRIPT"
    exit 1
fi

if [ ! -d "$BVH_ROOT" ]; then
    echo "ERROR: BVH root directory not found: $BVH_ROOT"
    exit 1
fi

FOLDERS=($(find "$BVH_ROOT" -mindepth 1 -maxdepth 1 -type d | sort))
TOTAL=${#FOLDERS[@]}

echo "============================================"
echo "BVH Root : $BVH_ROOT"
echo "Total    : $TOTAL folders"
echo "============================================"

COUNT=0
ERRORS=0

for FOLDER in "${FOLDERS[@]}"; do
    NAME=$(basename "$FOLDER")
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] $NAME"

    if bash "$RENDER_SCRIPT" "$FOLDER"; then
        echo "  OK: $NAME"
    else
        echo "  ERROR: failed for $NAME"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "============================================"
echo "Done."
echo "  Success : $((COUNT - ERRORS))"
echo "  Errors  : $ERRORS"
echo "============================================"
