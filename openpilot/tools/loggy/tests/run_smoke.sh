#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

JOBS="${LOGGY_SMOKE_JOBS:-}"
SKIP_BUILD=0
WITH_ROUTE=0
WITH_CAPTURE=0

usage() {
  cat <<'USAGE'
Usage: openpilot/tools/loggy/tests/run_smoke.sh [options]

Build and run the Loggy smoke suite from the repository root.

Options:
  --skip-build       Run existing binaries without invoking SCons.
  --jobs N           SCons parallelism. Defaults to LOGGY_SMOKE_JOBS or nproc.
  --with-route       Include route_ingest_smoke --demo. This may download data.
  --with-capture     Capture Cabana/Jotpluggler presets under a virtual display.
  --full             Equivalent to --with-route --with-capture.
  -h, --help         Show this help.

The default run is deterministic and non-GUI. The capture step delegates to
capture_presets.sh, which starts a private Xvfb display or uses xvfb-run.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      ;;
    --jobs)
      if [[ $# -lt 2 ]]; then
        echo "Error: --jobs requires a value" >&2
        exit 2
      fi
      JOBS="$2"
      shift
      ;;
    --with-route)
      WITH_ROUTE=1
      ;;
    --with-capture)
      WITH_CAPTURE=1
      ;;
    --full)
      WITH_ROUTE=1
      WITH_CAPTURE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS=4
  fi
fi

cd "$REPO_ROOT"

echo "Running Loggy style ratchet"
bash openpilot/tools/loggy/tests/style_ratchet.sh

LOCAL_SMOKES=(
  openpilot/tools/loggy/tests/workspace_smoke
  openpilot/tools/loggy/tests/transport_smoke
  openpilot/tools/loggy/tests/dbc_parser
  openpilot/tools/loggy/tests/dbc_commands
  openpilot/tools/loggy/tests/settings_smoke
  openpilot/tools/loggy/tests/store_scheduler
  openpilot/tools/loggy/tests/live_smoke
  openpilot/tools/loggy/tests/computed_smoke
  openpilot/tools/loggy/tests/panes_smoke
  openpilot/tools/loggy/tests/extract_smoke
)

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "Building Loggy smoke targets with scons -j$JOBS loggy_smoke_build"
  scons -j"$JOBS" loggy_smoke_build
fi

for smoke in "${LOCAL_SMOKES[@]}"; do
  echo "Running $smoke"
  "$smoke"
done

if [[ "$WITH_ROUTE" -eq 1 ]]; then
  echo "Running openpilot/tools/loggy/tests/route_ingest_smoke --demo"
  openpilot/tools/loggy/tests/route_ingest_smoke --demo
else
  echo "Skipping route_ingest_smoke; pass --with-route or --full to include it."
fi

if [[ "$WITH_CAPTURE" -eq 1 ]]; then
  echo "Running preset captures under a virtual display"
  bash openpilot/tools/loggy/tests/capture_presets.sh
else
  echo "Skipping GUI capture; pass --with-capture or --full to include it."
fi

echo "Loggy smoke suite passed."
