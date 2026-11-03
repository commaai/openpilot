#!/bin/bash
# Build the patched small driving model from the master model, one patch at a time.
# patch_model.py refuses any base that is not the known master model (sha256 check).
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$CODE" rev-parse --show-toplevel)
MODEL=openpilot/selfdrive/modeld/models/driving_supercombo.onnx
BASE=$(mktemp --suffix=.onnx)
trap 'rm -f "$BASE"' EXIT
# master's model from git LFS, not the (patched) file in the working tree
git -C "$ROOT" show "master:$MODEL" | git -C "$ROOT" lfs smudge > "$BASE"
PYTHON=${PYTHON:-python3}
$PYTHON "$CODE/patch_model.py" --base "$BASE" --out "$ROOT/$MODEL" --traffic-output \
  "$CODE/patches/stop_gain_1p8.npz" \
  "$CODE/patches/soft_brake_v1.npz" \
  "$CODE/patches/lat_readout_wlo20.npz"
