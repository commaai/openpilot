#!/usr/bin/env python3
"""On-device guided TSS3 request-signer bringup backend.

Stdout is a small JSONL status protocol intended for the openpilot UI. The
protocol is target-neutral: every request-signer kit reports through it. The
caller owns the cooperative direct-Panda lease for this process lifetime.
The normal path performs no peer resets: it verifies the post-install control
domain state directly and reports DRCC state as telemetry while requiring healthy
Brake/EPS communication monitors and a successful fresh-signing self-test.
Manual arm (no native catch) runs the shared startup race on the repinned
topology; the Camry was the first vehicle proven on it.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, TextIO
from tsk.common.ram_exec import _import_isotp_send, _import_uds
from tsk.ram_runtime.tss3_request_signer import install, self_test
from tsk.ram_runtime.tss3_post_install_recovery import control_domain_state, recovery_target
from tsk.ram_runtime.tss3_startup_programming import race_to_bootloader


STATUS_SCHEMA = "tss3-request-signer-ui-status-v1"
WORKER_RESULT_SCHEMA = "tss3-request-signer-warm-worker-result-v1"
_emit_stream: TextIO = sys.stdout


class UiBringupError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def emit(stage: str, title: str, detail: str, *, progress: int,
         done: bool = False, error: bool = False, output_dir: Path | None = None) -> None:
    row = {
        "schema": STATUS_SCHEMA,
        "stage": stage,
        "title": title,
        "detail": detail,
        "progress": progress,
        "done": done,
        "error": error,
    }
    if output_dir is not None:
        row["output_dir"] = str(output_dir)
    print(json.dumps(row, separators=(",", ":")), file=_emit_stream, flush=True)


def load_native_catch(path: Path) -> dict[str, Any]:
    # pandad writes this marker only after CarParamsPersistent passes the
    # Toyota TSS3 safety check, and install() below binds the kit's exact EPS
    # F181 identity before anything is uploaded.
    native = json.loads(path.read_text(encoding="utf-8"))
    if (
        native.get("schema") != "tss3-oracle-native-catch-v1"
        or not isinstance(native.get("target"), str)
        or not native["target"]
        or native.get("verdict") != "programming_request_sent_after_exact_50_03"
    ):
        raise UiBringupError(f"invalid native startup-catch marker: {path}")
    try:
        positive_frame = bytes.fromhex(native["positive_extended_frame_hex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UiBringupError(f"invalid native startup-catch positive frame: {path}") from exc
    if positive_frame != bytes.fromhex("065003003201f400"):
        raise UiBringupError(f"invalid native startup-catch positive frame: {path}")
    for key in (
        "pandad_wrapper_pid",
        "ignition_monotonic_ns",
        "first_extended_tx_monotonic_ns",
        "positive_extended_monotonic_ns",
        "programming_tx_monotonic_ns",
    ):
        if not isinstance(native.get(key), int) or native[key] <= 0:
            raise UiBringupError(f"native startup-catch marker has invalid {key}: {path}")
    if native["programming_tx_monotonic_ns"] < native["positive_extended_monotonic_ns"]:
        raise UiBringupError(f"native startup-catch marker has reversed response/programming order: {path}")
    return native


def manual_startup_race(meta: Path, *, cancel_requested: Any) -> dict[str, Any]:
    """Run the shared startup race on the repinned topology; kit target is provenance."""
    target = json.loads(meta.read_text(encoding="utf-8"))["target"]["name"]
    return race_to_bootloader(cancel_requested=cancel_requested, target=target)


def run(payload: Path, meta: Path, output_dir: Path, *,
        native_catch: Path | None = None, panda: Any | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UiBringupError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if native_catch is None:
        cancel_path = output_dir / "cancel-requested"
        emit(
            "armed", "Request-signer bringup armed", "Press the brake and POWER normally. Keep the vehicle in Park.",
            progress=5, output_dir=output_dir,
        )
        # UI cancellation is cooperative.  The startup catcher checks this path
        # while waiting and immediately before 10 02; after that boundary the UI
        # leaves the process alive so the EPS cannot be stranded mid-handoff.
        race = manual_startup_race(meta, cancel_requested=cancel_path.exists)
    else:
        native = load_native_catch(native_catch)
        race = {
            "schema": "tss3-startup-programming-native-v1",
            "native_catch": native,
            "verdict": "native_pandad_programming_request_dispatched",
        }
    write_json(output_dir / "startup-programming.json", race)

    emit(
        "programming", "EPS programming caught", "Installing the volatile 0x08A request signer into EPS RAM.",
        progress=20, output_dir=output_dir,
    )
    install_result = install(payload, meta, direct_boot=True, panda=panda)
    write_json(output_dir / "request-signer-install.json", install_result)
    if install_result.get("verdict") != "request_signer_live_probe_pass":
        raise UiBringupError(f"request-signer install/probe failed: {install_result.get('verdict')}")

    emit(
        "verifying", "Verifying bringup",
        "Checking FRC/Brake current faults without resets, then requesting one independently fresh signed 0x08A trailer.",
        progress=70, output_dir=output_dir,
    )
    state = control_domain_state(
        output_dir / "control-domain-state.json", recovery_target(meta), panda=panda,
    )
    health = state["peer_health"]
    if health["state"] == "fault":
        raise UiBringupError(f"peer fault observed: {', '.join(health['faults'])}")
    signer_test = self_test(meta, panda=panda)
    write_json(output_dir / "request-signer-self-test.json", signer_test)
    if signer_test.get("passed") is not True:
        raise UiBringupError("classic 0x08A fresh-signing self-test failed")

    peer_health_inconclusive = health["state"] != "healthy"
    unavailable = ", ".join(health["unavailable"])
    verdict = (
        "startup_caught_fresh_signer_peer_state_inconclusive_self_test_pass"
        if peer_health_inconclusive
        else "startup_caught_fresh_signer_peer_state_healthy_self_test_pass"
    )

    result = {
        "schema": "tss3-request-signer-ui-bringup-v1",
        "startup_programming": race,
        "request_signer_install": install_result,
        "control_domain_state": state,
        "request_signer_self_test": signer_test,
        "verdict": verdict,
    }
    write_json(output_dir / "summary.json", result)
    title = "Signer live; peer checks incomplete" if peer_health_inconclusive else "Request-signer bringup complete"
    detail = (
        f"EPS fresh signer self-test passed; unavailable peer diagnostics: {unavailable}."
        if peer_health_inconclusive
        else "EPS fresh signer and FRC/Brake current-fault checks passed without peer resets."
    )
    emit(
        "done", title, detail,
        progress=100, done=True, output_dir=output_dir,
    )
    return result


def run_warm_worker(payload: Path, meta: Path, worker_socket: Path, *, parent_pid: int) -> int:
    """Pre-import the backend, then open a fresh Panda after the caught handoff."""
    global _emit_stream
    try:
        from panda import Panda
    except ImportError as exc:
        raise UiBringupError("panda module unavailable") from exc

    # Resolve the lazy protocol imports before advertising readiness. They cost
    # hundreds of milliseconds on-device and must not sit between 10 02 and the
    # first bootloader request.
    _import_uds()
    _import_isotp_send()
    worker_socket.parent.mkdir(parents=True, exist_ok=True)
    worker_socket.unlink(missing_ok=True)
    panda = None
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(worker_socket))
        server.listen(1)
        server.settimeout(1.0)
        while True:
            try:
                conn, _ = server.accept()
                break
            except TimeoutError:
                try:
                    os.kill(parent_pid, 0)
                except ProcessLookupError as exc:
                    raise UiBringupError("warm worker parent exited") from exc
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            _emit_stream = writer
            request = json.loads(reader.readline())
            native_catch = Path(request["native_catch"])
            output_dir = Path(request["output_dir"])
            for key in ("DIRECT_PANDA_LEASE_SUPERVISOR_PID", "DIRECT_PANDA_LEASE_ID"):
                value = request.get(key)
                if not isinstance(value, str) or not value:
                    raise UiBringupError(f"warm worker request missing {key}")
                os.environ[key] = value

            try:
                # Match the proven manual-arm lifecycle: native pandad closes its
                # catch connection, then the installer opens and configures a new
                # Panda connection. A handle opened before ignition intermittently
                # delayed both boot and application F181 by seconds.
                panda = Panda(cli=False)
                run(
                    payload, meta, output_dir,
                    native_catch=native_catch, panda=panda,
                )
                returncode = 0
            except Exception as exc:  # noqa: BLE001 - forwarded to UI protocol
                emit(
                    "error", "Request-signer bringup failed", f"{type(exc).__name__}: {exc}",
                    progress=0, error=True, output_dir=output_dir,
                )
                returncode = 2
            print(json.dumps({"schema": WORKER_RESULT_SCHEMA, "returncode": returncode}), file=writer, flush=True)
            return returncode
    finally:
        _emit_stream = sys.stdout
        server.close()
        if panda is not None:
            panda.close()
        worker_socket.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--native-catch", type=Path)
    ap.add_argument("--worker-socket", type=Path)
    ap.add_argument("--worker-parent-pid", type=int)
    args = ap.parse_args()

    if args.worker_socket is not None:
        if args.output_dir is not None or args.native_catch is not None:
            raise UiBringupError("warm worker does not accept output-dir/native-catch on its command line")
        if args.worker_parent_pid is None or args.worker_parent_pid <= 1:
            raise UiBringupError("warm worker requires --worker-parent-pid")
        return run_warm_worker(
            args.payload, args.meta, args.worker_socket,
            parent_pid=args.worker_parent_pid,
        )
    if args.output_dir is None:
        raise UiBringupError("--output-dir is required outside warm-worker mode")

    try:
        run(args.payload, args.meta, args.output_dir, native_catch=args.native_catch)
        return 0
    except Exception as exc:  # noqa: BLE001 - UI must surface any bounded bringup failure
        emit(
            "error", "Request-signer bringup failed", f"{type(exc).__name__}: {exc}",
            progress=0, error=True, output_dir=args.output_dir,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
