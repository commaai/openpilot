#!/bin/bash
# Quick script to profile UI and generate flamegraph

PATTERN=${4:-"selfdrive/ui/ui.py"}  # pgrep -f pattern to locate process
UI_PID=$(pgrep -f "$PATTERN" | head -1)

if [ -z "$UI_PID" ]; then
    echo "Error: process not found (pgrep -f \"$PATTERN\")"
    exit 1
fi

DURATION=${1:-10}  # Default 10 seconds
OUTPUT=${2:-/tmp/ui_flamegraph.svg}
MODE=${3:-""}  # Optional: "gil" to sample only when GIL is held

echo "Profiling UI process $UI_PID for ${DURATION}s..."
echo "Output: $OUTPUT"
echo "Pattern: $PATTERN"
if [ "$MODE" = "gil" ]; then
    echo "Mode: GIL-only (--gil)"
fi

# Check if py-spy is installed
if ! command -v py-spy &> /dev/null; then
    echo "Installing py-spy..."
    pip install py-spy
fi

# Generate flamegraph
PYSPY_ARGS=()
if [ "$MODE" = "gil" ]; then
    PYSPY_ARGS+=(--gil)
fi

py-spy record "${PYSPY_ARGS[@]}" \
    -o "$OUTPUT" \
    --pid "$UI_PID" \
    --duration "$DURATION" \
    --rate 100 \
    --subprocesses

if [ $? -eq 0 ]; then
    echo "Flamegraph saved to: $OUTPUT"
    echo "View with: xdg-open $OUTPUT"
else
    echo "Error generating flamegraph"
    exit 1
fi

