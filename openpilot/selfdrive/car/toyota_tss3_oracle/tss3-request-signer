#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PY="${TSS3_PYTHON:-/usr/local/venv/bin/python}"
OPENPILOT_ROOT="${TSS3_OPENPILOT_ROOT:-/data/openpilot}"
TOOL="${TSS3_REQUEST_SIGNER_PATH:-$KIT_ROOT/runtime/tsk/ram_runtime/tss3_request_signer.py}"
PAYLOAD="${TSS3_REQUEST_SIGNER_PAYLOAD_PATH:-$KIT_ROOT/bundle/payload.bin}"
META="${TSS3_REQUEST_SIGNER_META_PATH:-$KIT_ROOT/bundle/request_signer.json}"
LEASE_LIB="${TSS3_PANDA_LEASE_LIB:-$KIT_ROOT/runtime/tsk/ram_runtime/tss3_panda_lease.sh}"
RECOVERY_TOOL="$KIT_ROOT/runtime/tsk/ram_runtime/tss3_post_install_recovery.py"
UI_BRINGUP_TOOL="$KIT_ROOT/runtime/tsk/ram_runtime/tss3_request_signer_ui_bringup.py"

export PYTHONPATH="$OPENPILOT_ROOT:$KIT_ROOT/runtime"
cd "$KIT_ROOT"

die() { printf 'tss3-request-signer: %s\n' "$*" >&2; exit 2; }
[[ -f "$LEASE_LIB" ]] || die "Panda lease helper missing: $LEASE_LIB"
source "$LEASE_LIB"

usage() {
  cat <<'USAGE'
usage: ./tss3-request-signer COMMAND [ARGS]

Offline:
  doctor

After power-up bringup (install via ui-bringup, or NRTD cold start):
  install [OUTPUT_JSON]
  status [OUTPUT_JSON]
  self-test [OUTPUT_JSON]
  benchmark [COUNT] [OUTPUT_JSON]
  benchmark-100hz [COUNT] [OUTPUT_JSON]
  recover-peers [OUTPUT_DIR]

Guided UI bringup (shared status protocol; the manual-arm startup race runs
on the shared repinned topology, Panda bus 0):
  ui-bringup [OUTPUT_DIR]
  ui-resume NATIVE_CATCH [OUTPUT_DIR]
  ui-worker SOCKET PARENT_PID
  ui-resume-warm SOCKET NATIVE_CATCH OUTPUT_DIR PANDAD_PID

This kit is exact-target-bound. Do not use it on another EPS identity.
The default request uses four ordered standard-0x777 classic frames and carries
all 28 application bytes. The response is one standard-0x7A9 classic frame.
The resident is volatile; a full EPS power cycle removes it.
No CodeFlash write is performed.
USAGE
}

doctor() {
  [[ -x "$PY" ]] || die "Python unavailable: $PY"
  [[ -d "$OPENPILOT_ROOT" ]] || die "openpilot root unavailable: $OPENPILOT_ROOT"
  [[ -f "$TOOL" && -f "$PAYLOAD" && -f "$META" ]] || die "request-signer kit incomplete"
  "$PY" - "$PAYLOAD" "$META" <<'PY'
import hashlib, json, pathlib, sys
from Crypto.Cipher import AES
from panda import Panda
from tsk.common.ram_exec import _import_isotp_send, _import_uds

payload = pathlib.Path(sys.argv[1]).read_bytes()
meta = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assert meta["schema"] == "tss3-request-signer-build-v2"
got = hashlib.sha256(payload).hexdigest()
assert len(payload) == 0x1000
assert got == meta["authenticated_payload"]["sha256"], (got, meta["authenticated_payload"]["sha256"])
assert AES is not None and callable(Panda)
_import_uds()
_import_isotp_send()
print("tss3-request-signer doctor: PASS")
print("target=" + meta["target"]["name"])
print("payload_sha256=" + got)
PY
}

run_bounded() {
  local seconds="$1" rc
  shift
  set +e
  timeout --signal=TERM --kill-after=2 "${seconds}s" "$PY" "$TOOL" "$@"
  rc=$?
  set -e
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    die "command exceeded ${seconds}s absolute runtime bound"
  fi
  return "$rc"
}
require_ui_backend() {
  # No kit-target gate here: the backend shares one status protocol across
  # targets and itself refuses manual arm when the kit target has no
  # recovered startup race. Install-time F181 identity binding is the
  # vehicle boundary.
  [[ -f "$UI_BRINGUP_TOOL" ]] || die "UI bringup backend missing: $UI_BRINGUP_TOOL"
}

require_empty_output_dir() {
  local out_dir="$1"
  if [[ -d "$out_dir" ]] && find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "output directory is not empty: $out_dir"
  fi
  [[ ! -e "$out_dir" || -d "$out_dir" ]] || die "output path is not a directory: $out_dir"
}

cmd="${1:-}"
[[ -n "$cmd" ]] || { usage; exit 2; }
shift || true

case "$cmd" in
  -h|--help|help) usage ;;
  doctor)
    [[ $# -eq 0 ]] || die "doctor takes no arguments"
    doctor
    ;;
  install)
    [[ $# -le 1 ]] || die "install accepts [OUTPUT_JSON]"
    quiesce_panda_owner
    out="${1:-/tmp/tss3-request-signer-install.json}"
    run_bounded 25 install --payload "$PAYLOAD" --meta "$META" --execute --output "$out"
    ;;
  status)
    [[ $# -le 1 ]] || die "status accepts [OUTPUT_JSON]"
    quiesce_panda_owner
    args=(status --meta "$META")
    [[ $# -eq 1 ]] && args+=(--output "$1")
    run_bounded 12 "${args[@]}"
    ;;
  self-test)
    [[ $# -le 1 ]] || die "self-test accepts [OUTPUT_JSON]"
    quiesce_panda_owner
    args=(self-test --meta "$META" --execute)
    [[ $# -eq 1 ]] && args+=(--output "$1")
    run_bounded 12 "${args[@]}"
    ;;
  benchmark)
    [[ $# -le 2 ]] || die "benchmark accepts [COUNT] [OUTPUT_JSON]"
    count="${1:-20}"; out="${2:-}"
    quiesce_panda_owner
    args=(benchmark --meta "$META" --count "$count" --period-ms 25 --execute)
    [[ -n "$out" ]] && args+=(--output "$out")
    run_bounded 60 "${args[@]}"
    ;;
  benchmark-100hz)
    [[ $# -le 2 ]] || die "benchmark-100hz accepts [COUNT] [OUTPUT_JSON]"
    count="${1:-100}"; out="${2:-}"
    quiesce_panda_owner
    args=(benchmark-pipelined --meta "$META" --count "$count" --period-ms 10 --execute)
    [[ -n "$out" ]] && args+=(--output "$out")
    run_bounded 60 "${args[@]}"
    ;;
  ui-bringup)
    [[ $# -le 1 ]] || die "ui-bringup accepts [OUTPUT_DIR]"
    require_ui_backend
    out_dir="${1:-/tmp/tss3-request-signer-ui-bringup-$(date +%Y%m%dT%H%M%S)}"
    require_empty_output_dir "$out_dir"
    quiesce_panda_owner
    "$PY" "$UI_BRINGUP_TOOL" --payload "$PAYLOAD" --meta "$META" --output-dir "$out_dir"
    ;;
  ui-resume)
    [[ $# -ge 1 && $# -le 2 ]] || die "ui-resume requires NATIVE_CATCH and optional OUTPUT_DIR"
    require_ui_backend
    native_catch="$1"
    [[ -f "$native_catch" ]] || die "native catch marker unavailable: $native_catch"
    out_dir="${2:-/tmp/tss3-request-signer-ui-resume-$(date +%Y%m%dT%H%M%S)}"
    require_empty_output_dir "$out_dir"
    quiesce_panda_owner
    "$PY" "$UI_BRINGUP_TOOL" --payload "$PAYLOAD" --meta "$META" \
      --output-dir "$out_dir" --native-catch "$native_catch"
    ;;
  ui-worker)
    [[ $# -eq 2 ]] || die "ui-worker requires SOCKET PARENT_PID"
    require_ui_backend
    exec "$PY" "$UI_BRINGUP_TOOL" --payload "$PAYLOAD" --meta "$META" \
      --worker-socket "$1" --worker-parent-pid "$2"
    ;;
  ui-resume-warm)
    [[ $# -eq 4 ]] || die "ui-resume-warm requires SOCKET NATIVE_CATCH OUTPUT_DIR PANDAD_PID"
    require_ui_backend
    worker_socket="$1"; native_catch="$2"; out_dir="$3"; pandad_pid="$4"
    [[ -S "$worker_socket" ]] || die "warm request-signer worker unavailable: $worker_socket"
    [[ -f "$native_catch" ]] || die "native catch marker unavailable: $native_catch"
    require_empty_output_dir "$out_dir"
    quiesce_panda_owner "$pandad_pid" caught-handoff
    "$PY" - "$worker_socket" "$native_catch" "$out_dir" \
      "$DIRECT_PANDA_LEASE_SUPERVISOR_PID" "$DIRECT_PANDA_LEASE_ID" <<'PY'
import json, socket, sys
sock_path, native_catch, out_dir, supervisor_pid, lease_id = sys.argv[1:]
request = {
    "native_catch": native_catch,
    "output_dir": out_dir,
    "DIRECT_PANDA_LEASE_SUPERVISOR_PID": supervisor_pid,
    "DIRECT_PANDA_LEASE_ID": lease_id,
}
result = None
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    sock.connect(sock_path)
    stream = sock.makefile("rw", encoding="utf-8")
    stream.write(json.dumps(request, separators=(",", ":")) + "\n")
    stream.flush()
    for line in stream:
        print(line, end="", flush=True)
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") == "tss3-request-signer-warm-worker-result-v1":
            result = int(row["returncode"])
raise SystemExit(2 if result is None else result)
PY
    ;;
  recover-peers)
    [[ $# -le 1 ]] || die "recover-peers accepts [OUTPUT_DIR]"
    out_dir="${1:-/tmp/tss3-peer-recovery}"
    [[ ! -e "$out_dir" ]] || die "recover-peers output already exists: $out_dir"
    mkdir -p "$out_dir"
    quiesce_panda_owner
    "$PY" "$RECOVERY_TOOL" restart-domain --domain brake --meta "$META" --execute --output "$out_dir/brake-restart.json"
    printf 'Brake application returned. Wait for network quiet, then press Enter to restart FRC... ' >&2
    IFS= read -r _
    "$PY" "$RECOVERY_TOOL" restart-domain --domain frc --meta "$META" --execute --output "$out_dir/frc-restart.json"
    "$PY" "$RECOVERY_TOOL" state --meta "$META" --output "$out_dir/control-domain-state.json"
    ;;
  *) usage; exit 2 ;;
esac
