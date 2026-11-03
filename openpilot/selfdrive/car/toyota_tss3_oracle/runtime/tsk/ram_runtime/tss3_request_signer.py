#!/usr/bin/env python3
"""Install and exercise an exact-target TSS3 RAM-resident request signer."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import statistics
import subprocess

import threading
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from tsk.common.ram_exec import (
    ELM327_SAFETY_MODE,
    RAM_LOAD_ADDR,
    RX_ADDR,
    TX_ADDR,
    _import_uds,
    execute_ram_payload,
    explicit_ram_exec_geometry,
    explicit_route,
    load_security_secret,
)
from tsk.ram_runtime.panda_eps import wait_for_f181
BUILD_SCHEMA = "tss3-request-signer-build-v2"

REQUEST_ID = 0x777
RESPONSE_ID = 0x7A9
BUS = 0
SEQUENCE_MAX = 0xFF
RUNTIME_MAX_PENDING_GENERATIONS = 8
RUNTIME_PUBLICATION_DEADLINE_MS = 90.0
RESPONSE_LEN = 8
RESPONSE_MAGIC = 0xC9
FOUR_FRAME_CODEC = "four-frame"
REQUEST_CARRIER = "functional-nibble4"
FRAGMENT_COUNT = 4
STATE_SIZE = 0x24
STATE_MAGIC = 0x4C433841
STATE_VERSION = 3
APPLICATION_REAPPEAR_TIMEOUT_SECONDS = 3.0
SIGNER_PROBE_TIMEOUT_SECONDS = 3.0
SAFE_APPLICATION = bytes.fromhex("0000000880002d4700000000007fff007fff0000c000100032000000")

# Panda echoes each transmitted message back on can_recv(): bus + 0x80 was
# accepted onto CAN, bus + 0xC0 was rejected by the active safety model.
TX_RECEIPT_ACCEPTED_BUS_OFFSET = 0x80
TX_RECEIPT_REJECTED_BUS_OFFSET = 0xC0


class RequestSignerError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_meta(path: Path) -> dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != BUILD_SCHEMA:
        raise RequestSignerError(f"unexpected metadata schema: {meta.get('schema')}")
    if meta.get("state", {}).get("version") != STATE_VERSION:
        raise RequestSignerError(f"unexpected state schema version: {meta.get('state', {}).get('version')}")
    return meta


def meta_transport(meta: dict[str, Any]) -> dict[str, Any]:
    request = meta["request"]
    response = meta["response"]
    carrier = request.get("carrier")
    if carrier != REQUEST_CARRIER:
        raise RequestSignerError(f"unsupported request-signer carrier: {carrier!r}")
    if request.get("codec") != FOUR_FRAME_CODEC:
        raise RequestSignerError(
            f"request-signer codec must be {FOUR_FRAME_CODEC!r}: {request.get('codec')!r}"
        )
    if request.get("frame_count") != FRAGMENT_COUNT:
        raise RequestSignerError(
            f"{FOUR_FRAME_CODEC} request signer requires {FRAGMENT_COUNT} request frames: "
            f"{request.get('frame_count')!r}"
        )
    return {
        "request_id": int(request["can_id"], 0), "request_bus": int(request["bus"]),
        "response_id": int(response["can_id"], 0), "response_bus": int(response["bus"]),
        "codec": FOUR_FRAME_CODEC, "carrier": carrier,
    }


def decode_state(raw: bytes) -> dict[str, Any]:
    if len(raw) != STATE_SIZE:
        raise RequestSignerError(f"state returned {len(raw)} bytes; expected {STATE_SIZE}")
    magic = int.from_bytes(raw[0:4], "little")
    return {
        "raw_hex": raw.hex(),
        "magic": f"0x{magic:08X}",
        "magic_ok": magic == STATE_MAGIC,
        "version": raw[4],
        "version_ok": raw[4] == STATE_VERSION,
        "initialized": raw[5] == 1,
        "sequence": raw[6],
        "last_seq": raw[6],
        "assembly_seq": raw[6],
        "next_fragment": raw[7],
        "reserved_transport": raw[7],
        "request_count": int.from_bytes(raw[8:12], "little"),
        "success_count": int.from_bytes(raw[12:16], "little"),
        "response_count": int.from_bytes(raw[16:20], "little"),
        "last_command5_rc": raw[20],
        "last_done_flag": raw[21],
        "last_command_status": raw[22],
        "last_output_length": raw[23],
        "freshness_reset": int.from_bytes(raw[24:28], "little"),
        "freshness_trip": int.from_bytes(raw[28:30], "little"),
        "freshness_message": raw[30],
        "freshness_initialized": raw[31] == 1,
        "tap_cursor": int.from_bytes(raw[32:34], "little"),
        "last_fv4": raw[34],
    }


def read_resident_state(meta: dict[str, Any], panda: Any) -> dict[str, Any]:
    """Read the resident's 36-byte state through application SID 0x23.

    Use this only to distinguish failures. The target metadata binds the state
    location selected by the exact CodeFlash contract. Report an unsupported
    application-memory read to the caller while preserving the signer verdict.
    """
    state = meta.get("state", {})
    address = int(state["base"], 0)
    size = int(state["size"])
    if size != STATE_SIZE:
        raise RequestSignerError(f"metadata state size {size} != {STATE_SIZE}")
    transport = meta_transport(meta)
    uds_mod = _import_uds()
    client = uds_mod.UdsClient(
        panda, TX_ADDR, RX_ADDR, transport["request_bus"],
        timeout=0.2, response_pending_timeout=1.0,
    )
    client.diagnostic_session_control(uds_mod.SESSION_TYPE.EXTENDED_DIAGNOSTIC)
    request = bytes((0x15, 0x01)) + address.to_bytes(4, "big") + bytes((size,))
    payload = client._uds_request(uds_mod.SERVICE_TYPE.READ_MEMORY_BY_ADDRESS, data=request)
    return decode_state(bytes(payload))


def build_request(application: bytes, seq: int) -> tuple[bytes, ...]:
    """Encode all 28 application bytes in the universal four-frame carrier."""
    if len(application) != 28:
        raise ValueError("application must be exactly 28 bytes")
    if not 1 <= seq <= SEQUENCE_MAX:
        raise ValueError("seq must be 1..255")
    low = seq & 0x0F
    high = (seq >> 4) & 0x0F
    headers = (0x80 | low, 0x90 | high, 0xA0 | low, 0xB0 | high)
    return tuple(
        bytes((header,)) + application[index * 7:(index + 1) * 7]
        for index, header in enumerate(headers)
    )



@dataclass(frozen=True)
class Response:
    seq: int
    status: int
    trailer: bytes


def parse_response(data: bytes, expected_seq: int | None = None) -> Response:
    if len(data) != RESPONSE_LEN:
        raise ValueError("response must be exactly 8 bytes")
    if data[0] != RESPONSE_MAGIC:
        raise ValueError(f"bad response magic 0x{data[0]:02X}")
    seq, status = data[1], data[2]
    if data[3] != (seq ^ 0xFF):
        raise ValueError("response sequence complement mismatch")
    if expected_seq is not None and seq != expected_seq:
        raise ValueError(f"response seq {seq} != expected {expected_seq}")
    return Response(seq, status, data[4:8])


def transact(panda, application: bytes, seq: int,
             timeout: float = 0.050, *, transport: dict[str, Any] | None = None,
             telemetry: dict[str, Any] | None = None) -> tuple[Response, float]:
    transport = transport or {
        "request_id": REQUEST_ID, "request_bus": BUS,
        "response_id": RESPONSE_ID, "response_bus": BUS,
        "codec": FOUR_FRAME_CODEC, "carrier": REQUEST_CARRIER,
    }
    frames = build_request(application, seq)
    start = time.monotonic()
    panda.can_send_many([
        (transport["request_id"], frame, transport["request_bus"]) for frame in frames
    ])
    deadline = start + timeout
    while time.monotonic() < deadline:
        for row in panda.can_recv():
            if len(row) < 3:
                continue
            address, data, bus = int(row[0]), bytes(row[-2]), int(row[-1])
            if telemetry is not None and address == transport["request_id"]:
                if bus == transport["request_bus"] + TX_RECEIPT_ACCEPTED_BUS_OFFSET:
                    telemetry["tx_receipts_accepted"] += 1
                elif bus == transport["request_bus"] + TX_RECEIPT_REJECTED_BUS_OFFSET:
                    telemetry["tx_receipts_rejected"] += 1
            if address == transport["response_id"] and bus == transport["response_bus"]:
                try:
                    return parse_response(data, seq), (time.monotonic() - start)
                except ValueError:
                    continue
        time.sleep(0.0005)
    raise TimeoutError(f"no request-signer response for seq {seq}")


def _checkout_identity(path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(path), "branch": None, "commit": None}
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True, timeout=1.0,
        )
        toplevel, commit, branch = result.stdout.splitlines()
        # An uninitialized submodule must not inherit its parent's identity.
        if Path(toplevel).resolve() != path.resolve():
            raise ValueError("path is not a Git checkout root")
        identity["commit"] = commit
        identity["branch"] = None if branch == "HEAD" else branch
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    return identity


class RequestSignerSession:
    def __init__(self, meta: dict[str, Any], *, panda: Any | None = None):
        self.meta = meta
        self.transport = meta_transport(meta)
        self._owns_panda = panda is None
        if panda is None:
            from panda import Panda
            panda = Panda()
        self.panda = panda
        # 0x777 is an 8-byte diagnostic-address carrier, so ELM327 already
        # permits it while preserving the normal relay path. SAFETY_ALLOUTPUT
        # is debug-only and falls back to SILENT on release Panda firmware.
        self.panda.set_safety_mode(ELM327_SAFETY_MODE, 1)
        # set_safety_mode never reports rejection: firmware without the
        # requested mode silently falls back to SILENT, so health() is the
        # only witness that the request took effect.
        self.telemetry: dict[str, Any] = {
            "safety_mode_requested": ELM327_SAFETY_MODE,
            "safety_param_requested": 1,
            "tx_receipts_accepted": 0,
            "tx_receipts_rejected": 0,
        }
        self._record_panda_state()

    def _record_panda_state(self) -> None:
        try:
            health = self.panda.health()
            self.telemetry["safety_mode_reported"] = health.get("safety_mode")
            self.telemetry["safety_param_reported"] = health.get("safety_param")
            self.telemetry["safety_mode_ok"] = (
                health.get("safety_mode") == ELM327_SAFETY_MODE
                and health.get("safety_param") == 1
            )
        except Exception as exc:  # noqa: BLE001
            self.telemetry["health_error"] = f"{type(exc).__name__}: {exc}"
        try:
            self.telemetry["panda_version"] = self.panda.get_version()
        except Exception as exc:  # noqa: BLE001
            self.telemetry["panda_version_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from panda import FW_PATH, McuType, Panda
            fw_file = os.path.join(FW_PATH, McuType.H7.config.app_fn)
            self.telemetry["fw_file"] = fw_file
            self.telemetry["fw_file_signature_hex_prefix"] = (
                Panda.get_signature_from_firmware(fw_file).hex()[:16]
            )
            self.telemetry["panda_signature_hex_prefix"] = self.panda.get_signature().hex()[:16]
            self.telemetry["panda_signature_matches_fw_file"] = self.panda.up_to_date()
        except Exception as exc:  # noqa: BLE001
            self.telemetry["signature_error"] = f"{type(exc).__name__}: {exc}"
        root = Path(os.environ.get("TSS3_OPENPILOT_ROOT") or "/data/openpilot")
        self.telemetry["source_checkouts"] = {
            name: _checkout_identity(path)
            for name, path in (
                ("openpilot", root),
                ("opendbc_repo", root / "opendbc_repo"),
                ("panda", root / "panda"),
            )
        }

    def close(self) -> None:
        if self._owns_panda:
            try:
                self.panda.close()
            except Exception:
                pass


def safe_application(seq: int) -> bytes:
    if not 1 <= seq <= SEQUENCE_MAX:
        raise ValueError("seq must be 1..255")
    application = bytearray(SAFE_APPLICATION)
    application[26] = seq & 0x3F
    return bytes(application)


def probe_signer(session: RequestSignerSession, *, timeout: float, start_seq: int = 1) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("signer probe timeout must be positive")
    if not 1 <= start_seq <= SEQUENCE_MAX:
        raise ValueError("signer probe start sequence must be 1..255")

    deadline = time.monotonic() + timeout
    seq = start_seq
    attempts = 0
    last_error = "no request sent"
    while time.monotonic() < deadline:
        attempts += 1
        application = safe_application(seq)
        frames = build_request(application, seq)
        try:
            response, elapsed = transact(
                session.panda, application, seq,
                timeout=min(0.080, max(0.001, deadline - time.monotonic())),
                transport=session.transport,
                telemetry=session.telemetry,
            )
            if response.status == 0:
                return {
                    "passed": True,
                    "attempts": attempts,
                    "request": {
                        "can_id": f"0x{session.transport['request_id']:08X}",
                        "bus": session.transport["request_bus"],
                        "fd": False,
                        "codec": session.transport["codec"],
                        "carrier": session.transport["carrier"],
                        "application_hex": application.hex(),
                        "frames_hex": [frame.hex() for frame in frames],
                    },
                    "response": {
                        "can_id": f"0x{session.transport['response_id']:08X}",
                        "bus": session.transport["response_bus"],
                        "seq": response.seq,
                        "status": response.status,
                        "trailer": response.trailer.hex(),
                        "fv4": response.trailer[0] >> 4,
                        "mac28": f"{int.from_bytes(response.trailer, 'big') & 0x0FFFFFFF:07x}",
                        "elapsed_ms": elapsed * 1000.0,
                    },
                }
            last_error = f"request-signer status {response.status} for seq {seq}"
        except TimeoutError as exc:
            last_error = str(exc)
        seq = (seq % SEQUENCE_MAX) + 1

    raise RequestSignerError(
        f"no successful request-signer response after {attempts} attempts: {last_error}"
    )


def install(payload_path: Path, meta_path: Path, *, direct_boot: bool = False, panda: Any | None = None) -> dict[str, Any]:
    install_started_ns = time.monotonic_ns()
    meta = load_meta(meta_path)
    transport = meta_transport(meta)
    route = explicit_route(bus=transport["request_bus"], elm327_param=1, uds_variant="old", cpu_index=0)
    expected_f181 = meta["target"]["application_f181_hex"]
    expected_boot_f181 = meta["target"]["boot_f181_hex"]
    payload = payload_path.read_bytes()
    expected = meta["authenticated_payload"]["sha256"]
    if len(payload) != 0x1000 or sha256(payload) != expected:
        raise RequestSignerError("authenticated request-signer payload identity mismatch")
    secret, _ = load_security_secret()
    execution = execute_ram_payload(
        payload, route=route, security_secret=secret, timeout=0.08,
        expected_f181_hex=expected_f181, expected_boot_f181_hex=expected_boot_f181,
        geometry=explicit_ram_exec_geometry(load_addr=RAM_LOAD_ADDR, evidence=f"dynamic:{meta['target']['name']}"),
        allow_direct_boot=direct_boot, programming_already_requested=direct_boot, panda=panda,
    )
    execution_complete_ns = time.monotonic_ns()
    pre = wait_for_f181(
        timeout=APPLICATION_REAPPEAR_TIMEOUT_SECONDS, route=route, panda=panda,
        expected_f181_hex=expected_f181,
    )
    application_probe_complete_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "schema": "tss3-request-signer-install-v1",
        "payload_sha256": expected,
        "entry_condition": "exact_bootloader_f181" if direct_boot else "application_programming_entry",
        "execution": execution,
        "pre_helper_f181": pre,
    }
    result["timing"] = {
        "install_started_monotonic_ns": install_started_ns,
        "execution_complete_monotonic_ns": execution_complete_ns,
        "application_probe_complete_monotonic_ns": application_probe_complete_ns,
        "execution_ms": (execution_complete_ns - install_started_ns) / 1e6,
        "application_probe_ms": (application_probe_complete_ns - execution_complete_ns) / 1e6,
    }
    if not pre["ok"]:
        result["verdict"] = "abi_preserving_runtime_failed_before_helper_initialization"
        return result
    session: RequestSignerSession | None = None
    try:
        session = RequestSignerSession(meta, panda=panda)
        result["signer_probe"] = probe_signer(session, timeout=SIGNER_PROBE_TIMEOUT_SECONDS)
        result["verdict"] = "request_signer_live_probe_pass"
    except Exception as exc:  # noqa: BLE001
        result["signer_probe_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "application_live_but_request_signer_probe_failed"
        if session is not None:
            try:
                result["resident_state_after_probe"] = read_resident_state(meta, session.panda)
            except Exception as state_exc:  # noqa: BLE001
                result["resident_state_readback_error"] = f"{type(state_exc).__name__}: {state_exc}"
    finally:
        if session is not None:
            result["signer_session"] = dict(session.telemetry)
            session.close()
    result["timing"]["signer_probe_complete_monotonic_ns"] = time.monotonic_ns()
    install_complete_ns = time.monotonic_ns()
    result["timing"]["install_complete_monotonic_ns"] = install_complete_ns
    result["timing"]["total_ms"] = (install_complete_ns - install_started_ns) / 1e6
    return result


def self_test(meta_path: Path, *, panda: Any | None = None) -> dict[str, Any]:
    meta = load_meta(meta_path)
    session = RequestSignerSession(meta, panda=panda)
    try:
        probe = probe_signer(session, timeout=SIGNER_PROBE_TIMEOUT_SECONDS)
        return {
            "schema": "tss3-request-signer-self-test-v1",
            "passed": True,
            "request": probe["request"],
            "response": probe["response"],
        }
    finally:
        session.close()


def status(meta_path: Path) -> dict[str, Any]:
    meta = load_meta(meta_path)
    session = RequestSignerSession(meta)
    try:
        probe = probe_signer(session, timeout=SIGNER_PROBE_TIMEOUT_SECONDS)
        return {
            "schema": "tss3-request-signer-status-v1",
            "live": True,
            "signer_probe": probe,
        }
    finally:
        session.close()


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2

    def percentile(p: float) -> float:
        position = (count - 1) * p
        low = int(position)
        high = min(low + 1, count - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2,
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


def benchmark(meta_path: Path, count: int, period_ms: float) -> dict[str, Any]:
    if not 1 <= count <= 1000:
        raise RequestSignerError("benchmark count must be 1..1000")
    if not 20.0 <= period_ms <= 1000.0:
        raise RequestSignerError("benchmark period must be 20..1000 ms")
    meta = load_meta(meta_path)
    session = RequestSignerSession(meta)
    rows = []
    try:
        seq = 1
        deadline = time.monotonic()
        for i in range(count):
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            start = time.monotonic()
            application = safe_application(seq)
            response, elapsed = transact(
                session.panda, application, seq, timeout=0.080,
                transport=session.transport,
                telemetry=session.telemetry,
            )
            rows.append({
                "index": i, "seq": seq, "codec": session.transport["codec"],
                "status": response.status, "trailer": response.trailer.hex(),
                "fv4": response.trailer[0] >> 4, "successful": response.status == 0,
                "elapsed_ms": elapsed * 1000.0,
            })
            seq = (seq % SEQUENCE_MAX) + 1
            deadline = start + period_ms / 1000.0
        elapsed_values = [float(r["elapsed_ms"]) for r in rows]
        successful = sum(bool(r["successful"]) for r in rows)
        return {
            "schema": "tss3-request-signer-benchmark-v1",
            "count": count,
            "period_ms": period_ms,
            "successful": successful,
            "complete": successful == count,
            **latency_summary(elapsed_values),
            "rows": rows,
        }
    finally:
        session.close()


def _drain_can(panda: Any, *, max_batches: int = 32) -> None:
    """Discard bounded stale host-facing CAN before a timed benchmark."""
    for _ in range(max_batches):
        rows = panda.can_recv() or []
        if not rows:
            return


def _reduce_pipelined_benchmark(
    count: int,
    period_ms: float,
    sends: list[dict[str, Any]],
    responses: dict[int, dict[str, Any]],
    *,
    unsolicited_count: int,
    rx_errors: list[str],
    outstanding_after_drain: int,
    max_outstanding: int,
    sequence_window_exhausted: bool,
) -> dict[str, Any]:
    """Reduce a recorded pipelined run to the published benchmark report.

    A run only counts as a complete target-rate run when every scheduled request
    was sent and answered with status 0, the receiver never errored, the sequence
    window never exhausted, nothing stayed outstanding after the drain, the
    pending backlog never passed RUNTIME_MAX_PENDING_GENERATIONS, no publication
    missed its slot by a full period, and no reply exceeded
    RUNTIME_PUBLICATION_DEADLINE_MS.
    """
    rows: list[dict[str, Any]] = []
    rtts: list[float] = []
    lateness: list[float] = []
    response_ns: list[int] = []
    for sent in sends:
        index = int(sent["index"])
        response_row = responses.get(index)
        row = {key: value for key, value in sent.items() if key != "started_ns"}
        if response_row is None:
            row["error"] = "missing_response"
        else:
            response_time_ns = int(response_row["monotonic_ns"])
            row.update({
                "response_frame_hex": response_row["frame_hex"],
                "status": response_row["status"],
                "trailer": response_row["trailer"],
                "fv4": response_row["fv4"],
                "request_start_to_response_ms": round(
                    (response_time_ns - int(sent["started_ns"])) / 1e6, 3
                ),
                "response_monotonic_ns": response_time_ns,
            })
        rows.append(row)
        lateness.append(float(row["scheduled_start_lateness_ms"]))
        if row.get("status") == 0:
            rtts.append(float(row["request_start_to_response_ms"]))
            response_ns.append(int(row["response_monotonic_ns"]))

    success_count = len(rtts)
    response_ns.sort()
    response_intervals_ms = [
        (later - earlier) / 1e6 for earlier, later in pairwise(response_ns)
    ]
    effective_response_rate_hz = None
    if len(response_ns) >= 2 and response_ns[-1] > response_ns[0]:
        effective_response_rate_hz = (len(response_ns) - 1) * 1e9 / (response_ns[-1] - response_ns[0])

    def summary_or_none(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {key: round(value, 3) for key, value in latency_summary(values).items()}

    result: dict[str, Any] = {
        "schema": "tss3-request-signer-pipelined-benchmark-v1",
        "count_requested": count,
        "count_sent": len(sends),
        "period_ms": period_ms,
        "target_rate_hz": 1000.0 / period_ms,
        "success_count": success_count,
        "success_rate": success_count / count,
        "responses_received": len(responses),
        "outstanding_after_drain": outstanding_after_drain,
        "max_outstanding": max_outstanding,
        "runtime_max_pending_generations": RUNTIME_MAX_PENDING_GENERATIONS,
        "runtime_publication_deadline_ms": RUNTIME_PUBLICATION_DEADLINE_MS,
        "sequence_window_exhausted": sequence_window_exhausted,
        "unsolicited_response_count": unsolicited_count,
        "rx_errors": list(rx_errors),
        "request_start_to_response_ms": summary_or_none(rtts),
        "scheduled_start_lateness_ms": summary_or_none(lateness),
        "response_interarrival_ms": summary_or_none(response_intervals_ms),
        "effective_success_response_rate_hz": (
            round(effective_response_rate_hz, 3) if effective_response_rate_hz is not None else None
        ),
        "rows": rows,
    }
    max_lateness_ms = max(lateness, default=0.0)
    max_rtt_ms = max(rtts, default=float("inf"))
    result["complete_target_rate_run"] = bool(
        len(sends) == count and success_count == count and not rx_errors and not sequence_window_exhausted and
        outstanding_after_drain == 0 and max_outstanding <= RUNTIME_MAX_PENDING_GENERATIONS and
        max_lateness_ms < period_ms and max_rtt_ms <= RUNTIME_PUBLICATION_DEADLINE_MS
    )
    return result


def benchmark_pipelined(
    meta_path: Path, count: int, period_ms: float = 10.0, *, drain_timeout_s: float = 0.5,
) -> dict[str, Any]:
    """Measure signer throughput while publishing one request every period."""
    if not 1 <= count <= 1000:
        raise RequestSignerError("pipelined benchmark count must be 1..1000")
    if not 10.0 <= period_ms <= 1000.0:
        raise RequestSignerError("pipelined benchmark period must be 10..1000 ms")
    if not 0.1 <= drain_timeout_s <= 5.0:
        raise RequestSignerError("pipelined benchmark drain timeout must be 0.1..5 seconds")

    meta = load_meta(meta_path)
    session = RequestSignerSession(meta)
    try:
        next_seq = 1
        _drain_can(session.panda)

        rx_lock = threading.Lock()
        stop_rx = threading.Event()
        pending_by_seq: dict[int, deque[int]] = {}
        responses: dict[int, dict[str, Any]] = {}
        unsolicited: list[dict[str, Any]] = []
        rx_errors: list[str] = []
        max_outstanding = 0

        def receiver() -> None:
            while not stop_rx.is_set():
                try:
                    can_rows = session.panda.can_recv() or []
                except Exception as exc:  # noqa: BLE001
                    with rx_lock:
                        rx_errors.append(f"{type(exc).__name__}: {exc}")
                    return
                if not can_rows:
                    time.sleep(0.0005)
                    continue

                now_ns = time.monotonic_ns()
                with rx_lock:
                    for row in can_rows:
                        if len(row) < 3:
                            continue
                        address, data, bus = int(row[0]), bytes(row[-2]), int(row[-1])
                        if address != session.transport["response_id"] or bus != session.transport["response_bus"]:
                            continue
                        try:
                            parsed = parse_response(data)
                        except ValueError:
                            continue
                        queue = pending_by_seq.get(parsed.seq)
                        if not queue:
                            unsolicited.append({"monotonic_ns": now_ns, "frame_hex": data.hex()})
                            continue
                        index = queue.popleft()
                        if not queue:
                            del pending_by_seq[parsed.seq]
                        responses[index] = {
                            "monotonic_ns": now_ns,
                            "frame_hex": data.hex(),
                            "seq": parsed.seq,
                            "status": parsed.status,
                            "trailer": parsed.trailer.hex(),
                            "fv4": parsed.trailer[0] >> 4,
                        }

        rx_thread = threading.Thread(target=receiver, name="tss3-request-signer-rx", daemon=True)
        rx_thread.start()

        sends: list[dict[str, Any]] = []
        schedule_zero = time.monotonic()
        period_s = period_ms / 1000.0
        sequence_window_exhausted = False
        try:
            for index in range(count):
                target = schedule_zero + index * period_s
                now = time.monotonic()
                if now < target:
                    time.sleep(target - now)
                actual_start = time.monotonic()
                started_ns = time.monotonic_ns()

                with rx_lock:
                    seq = None
                    for _ in range(SEQUENCE_MAX):
                        candidate = next_seq
                        next_seq = (candidate % SEQUENCE_MAX) + 1
                        if candidate not in pending_by_seq:
                            seq = candidate
                            break
                    if seq is None:
                        sequence_window_exhausted = True
                        break
                    pending_by_seq[seq] = deque((index,))
                    outstanding = sum(len(queue) for queue in pending_by_seq.values())
                    max_outstanding = max(max_outstanding, outstanding)

                # Match native B26 progression while transporting the full application.
                application = safe_application(seq)
                frames = build_request(application, seq)
                submit_start_ns = time.monotonic_ns()
                try:
                    session.panda.can_send_many([
                        (session.transport["request_id"], frame, session.transport["request_bus"])
                        for frame in frames
                    ])
                except Exception:
                    with rx_lock:
                        queue = pending_by_seq.get(seq)
                        if queue is not None:
                            try:
                                queue.remove(index)
                            except ValueError:
                                pass
                            if not queue:
                                pending_by_seq.pop(seq, None)
                    raise
                submit_end_ns = time.monotonic_ns()
                sends.append({
                    "index": index,
                    "seq": seq,
                    "codec": session.transport["codec"],
                    "request_sequence": application[26],
                    "scheduled_start_lateness_ms": round(max(0.0, (actual_start - target) * 1000.0), 3),
                    "started_ns": started_ns,
                    "submit_ms": round((submit_end_ns - submit_start_ns) / 1e6, 3),
                    "frames_hex": [frame.hex() for frame in frames],
                })

            deadline = time.monotonic() + drain_timeout_s
            while time.monotonic() < deadline:
                with rx_lock:
                    if len(responses) >= len(sends) or rx_errors:
                        break
                time.sleep(0.001)
        finally:
            stop_rx.set()
            rx_thread.join(timeout=1.0)

        with rx_lock:
            responses_copy = dict(responses)
            errors_copy = list(rx_errors)
            outstanding_after_drain = sum(len(queue) for queue in pending_by_seq.values())

        return _reduce_pipelined_benchmark(
            count, period_ms, sends, responses_copy,
            unsolicited_count=len(unsolicited),
            rx_errors=errors_copy,
            outstanding_after_drain=outstanding_after_drain,
            max_outstanding=max_outstanding,
            sequence_window_exhausted=sequence_window_exhausted,
        )
    finally:
        session.close()


def _emit(value: Any, output: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("install")
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--direct-boot-confirmed", action="store_true")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("status")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("self-test")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("benchmark")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--period-ms", type=float, default=25.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("benchmark-pipelined")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--period-ms", type=float, default=10.0)
    p.add_argument("--drain-timeout-s", type=float, default=0.5)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("codec")
    p.add_argument("--application", required=True)
    p.add_argument("--seq", type=lambda x: int(x, 0), default=1)

    args = ap.parse_args()
    if args.cmd == "install":
        if not args.execute:
            raise RequestSignerError("live install requires --execute")
        result = install(args.payload, args.meta, direct_boot=args.direct_boot_confirmed)
        _emit(result, args.output)
        return 0 if result["verdict"] == "request_signer_live_probe_pass" else 2
    elif args.cmd == "status":
        _emit(status(args.meta), args.output)
    elif args.cmd == "self-test":
        if not args.execute:
            raise RequestSignerError("self-test requires --execute")
        _emit(self_test(args.meta), args.output)
    elif args.cmd == "benchmark":
        if not args.execute:
            raise RequestSignerError("benchmark requires --execute")
        _emit(benchmark(args.meta, args.count, args.period_ms), args.output)
    elif args.cmd == "benchmark-pipelined":
        if not args.execute:
            raise RequestSignerError("pipelined benchmark requires --execute")
        _emit(benchmark_pipelined(args.meta, args.count, args.period_ms,
                                  drain_timeout_s=args.drain_timeout_s), args.output)
    else:
        application = bytes.fromhex(args.application)
        frames = build_request(application, args.seq)
        print(
            f"request_id=0x{REQUEST_ID:08X} bus={BUS} fd=0 "
            f"codec={FOUR_FRAME_CODEC} carrier={REQUEST_CARRIER} frames={len(frames)}"
        )
        for index, frame in enumerate(frames):
            print(f"frame={index} len={len(frame)} data={frame.hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
