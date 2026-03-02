#!/bin/bash
# ./organize_bvhs.sh
# Copy BVH files into per-animal subdirectories

BVH_ROOT="$(cd "$(dirname "$0")" && pwd)/dataset/truebones/zoo/truebones_processed/bvhs"

echo "BVH Root: $BVH_ROOT"

for BVH_FILE in "$BVH_ROOT"/*.bvh; do
    [ -f "$BVH_FILE" ] || continue

    BASENAME=$(basename "$BVH_FILE")
    ANIMAL="${BASENAME%%_*}"
    DEST_DIR="$BVH_ROOT/$ANIMAL"

    mkdir -p "$DEST_DIR"
    cp "$BVH_FILE" "$DEST_DIR/"
done

echo "Done."
echo "Folders created:"
find "$BVH_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r DIR; do
    COUNT=$(ls "$DIR"/*.bvh 2>/dev/null | wc -l)
    echo "  $(basename "$DIR"): $COUNT files"
done
