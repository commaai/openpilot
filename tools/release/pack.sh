#!/usr/bin/env bash
# Builds the setup, reset and updater UIs that ship in AGNOS into standalone executables
# for comma devices. They bundle their own Python, raylib and dependencies.
#
#   tools/release/pack.sh [--onefile] OUTPUT_DIR
#
# Builds for linux/arm64 with docker buildx, e.g. on a remote builder (nsc docker buildx setup --use).
set -euo pipefail

ONEFILE=0
if [ "${1:-}" = "--onefile" ]; then
  ONEFILE=1
  shift
fi
OUTPUT=$(realpath -m "${1:?usage: pack.sh [--onefile] OUTPUT_DIR}")

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
git -C "$ROOT" submodule update --init opendbc_repo

# only send what the apps need to the builder
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT
mkdir -p "$CONTEXT/opendbc_repo" "$CONTEXT/tools/release"
tar -C "$ROOT" -cf - --exclude=__pycache__ --exclude='*.so' --exclude=openpilot/cereal/gen \
  --exclude=openpilot/selfdrive/assets/training \
  --exclude=openpilot/selfdrive/modeld/models --exclude=openpilot/common/hardware/comma/updater \
  openpilot opendbc_repo/opendbc tools/release/pack | tar -C "$CONTEXT" -xf -

docker buildx build --platform linux/arm64 --build-arg "ONEFILE=$ONEFILE" \
  -f "$CONTEXT/tools/release/pack/Dockerfile" --output "type=local,dest=$OUTPUT" "$CONTEXT"
echo "built $(ls "$OUTPUT" | grep -v report.xml | tr '\n' ' ')in $OUTPUT"
