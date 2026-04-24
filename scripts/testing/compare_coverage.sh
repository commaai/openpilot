#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd)"
cd "$ROOT"

COV_TARGET="selfdrive/modeld"
BASELINE_TESTS="selfdrive/modeld/tests/test_modeld.py"
# Default "ours" matches docs/testing/LOW-LEVEL-TEST-PLAN.md §6 (parser + fill contracts + Phase C extension).
OUR_TESTS="selfdrive/modeld/tests/test_parse_model_outputs.py selfdrive/modeld/tests/test_parse_model_outputs_vision_contracts.py selfdrive/modeld/tests/test_parse_model_outputs_policy_contracts.py selfdrive/modeld/tests/test_fill_model_msg.py selfdrive/modeld/tests/test_fill_model_msg_frame_ids.py selfdrive/modeld/tests/test_fill_model_msg_modelv2_dimensions.py selfdrive/modeld/tests/test_fill_model_msg_pose_odometry.py selfdrive/modeld/tests/test_fill_model_msg_driving_model_data.py selfdrive/modeld/tests/test_fill_model_msg_raw_predictions.py selfdrive/modeld/tests/test_fill_model_msg_fcw_hard_brake.py selfdrive/modeld/tests/test_get_model_metadata_unit.py selfdrive/modeld/tests/test_modeld_phase_c_contracts.py"
OUT_DIR=".coverage-compare/modeld"
COV_CONFIG="$ROOT/scripts/testing/coverage-modeld-compare.ini"

function usage() {
  echo "Compare baseline vs new-test coverage for a target directory."
  echo ""
  echo "Usage:"
  echo "  bash scripts/testing/compare_coverage.sh [options]"
  echo ""
  echo "Options:"
  echo "  --cov-target <path>   Coverage source target (default: selfdrive/modeld)"
  echo "  --baseline <tests>    Quoted baseline test paths/globs"
  echo "  --ours <tests>        Quoted new-test paths/globs"
  echo "  --out-dir <path>      Output directory for coverage artifacts"
  echo "  -h, --help            Show this help"
  echo ""
  echo "See scripts/testing/coverage-modeld-compare.ini: omits tests/ and subprocess daemons"
  echo "(modeld.py, dmonitoringmodeld.py); pytest runs with -n 0 for stable cov combine."
  echo ""
  echo "Example:"
  echo "  bash scripts/testing/compare_coverage.sh \\"
  echo "    --cov-target selfdrive/modeld \\"
  echo "    --baseline \"selfdrive/modeld/tests/test_modeld.py\" \\"
  echo "    --ours \"selfdrive/modeld/tests/test_parse_model_outputs.py selfdrive/modeld/tests/test_fill_model_msg.py\""
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cov-target) COV_TARGET="$2"; shift 2 ;;
    --baseline) BASELINE_TESTS="$2"; shift 2 ;;
    --ours) OUR_TESTS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

function run_group() {
  local name="$1"
  local tests_str="$2"
  local coverage_file="$OUT_DIR/.coverage.${name}"
  local xml_file="$OUT_DIR/coverage-${name}.xml"
  local html_dir="$OUT_DIR/html-${name}"
  local report_file="$OUT_DIR/summary-${name}.txt"

  read -r -a test_args <<< "$tests_str"
  if [[ ${#test_args[@]} -eq 0 ]]; then
    echo "No tests specified for ${name}"
    exit 1
  fi

  echo ""
  echo "=== Running ${name} coverage ==="
  echo "Tests: ${tests_str}"
  # -n 0: disable xdist for this run so pytest-cov combines a single data file and
  #        worker processes cannot re-introduce test modules into the trace.
  COVERAGE_FILE="$coverage_file" python -m pytest "${test_args[@]}" -n 0 \
    --cov="$COV_TARGET" \
    --cov-config="$COV_CONFIG" \
    --cov-report=xml:"$xml_file" \
    --cov-report=html:"$html_dir"

  echo ""
  echo "--- ${name} coverage summary ---"
  coverage report --rcfile="$COV_CONFIG" --data-file="$coverage_file" --precision=2 --sort=cover | tee "$report_file"
}

run_group "baseline" "$BASELINE_TESTS"
run_group "ours" "$OUR_TESTS"

echo ""
echo "Coverage artifacts written to: $OUT_DIR"
echo " - baseline data: $OUT_DIR/.coverage.baseline"
echo " - baseline xml : $OUT_DIR/coverage-baseline.xml"
echo " - baseline html: $OUT_DIR/html-baseline/index.html"
echo " - ours data    : $OUT_DIR/.coverage.ours"
echo " - ours xml     : $OUT_DIR/coverage-ours.xml"
echo " - ours html    : $OUT_DIR/html-ours/index.html"
