#!/bin/bash
# run.sh — Entrypoint for the RAG system.
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

if [ -z "$QUESTIONS_PATH" ] || [ -z "$PREDICTIONS_PATH" ]; then
    echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/rag.py" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
