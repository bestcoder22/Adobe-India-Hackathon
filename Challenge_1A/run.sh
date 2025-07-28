#!/usr/bin/env bash
set -e

INPUT_DIR=/app/input
OUTPUT_DIR=/app/output

# Create output dir if missing
mkdir -p "$OUTPUT_DIR"

echo "🚀 Running batch inference on \$INPUT_DIR, writing to \$OUTPUT_DIR"
python src/inference.py "$INPUT_DIR" "$OUTPUT_DIR" --batch
