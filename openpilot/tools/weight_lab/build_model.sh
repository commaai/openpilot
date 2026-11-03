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
# No soft brake: on the road (2026-11-03) it braked at greens, stopped 3-6 m early, did not resume, and slowed for
# speed bumps and things like intersections. Offline it adds > 0.1 m/s² of braking on 63% of open-road frames.
# Corner-cut readout at half strength: the full fit was twitchy at highway speed (+9% y jitter at > 20 m/s).
# Stop gain x1.4: x1.8 stopped a little too hard at red lights and slowed a little at some greens.
$PYTHON "$CODE/patch_model.py" --base "$BASE" --out "$ROOT/$MODEL" --traffic-output \
  "$CODE/patches/stop_gain_1p4.npz" \
  "$CODE/patches/lat_readout_wlo20_half.npz"
