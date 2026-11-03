#!/usr/bin/env python3
"""TSS3 post-bootstrap diagnostic clear and control-domain recovery.

The volatile resident is installed through the EPS programming/NRTD path. That
transition can leave a peer fail-safe chain active after the EPS application
returns. The recovery sequence keeps EPS powered, resets FRC, then Brake/EPB,
then FRC again, and validates the resulting peer state.

The caller supplies canonical request-signer metadata. The EPS application
identity stays exact-target-bound; the shared repinned topology supplies the
diagnostic bus. Peer roles use their topology-defined addresses and bind any
restart to the identity observed before it, rather than one vehicle's F181.

The legacy ``clear`` command remains diagnostic/forensic tooling. Recovery
performs no SecurityAccess, RoutineControl, firmware write, flash erase,
steering command, EPS reset, or EPS power cycle.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

EPS_TX = 0x7A1
EPS_RX = 0x7A9
DIAG_SAFETY = 3  # ELM327
DIAG_SAFETY_PARAM = 1  # CAN_MODE_NORMAL
SILENT_SAFETY = 0

FAULT_MASK = 0xAF
CURRENT_DTC_FAILURE_MASK = 0x01  # UDS testFailed; history has separate status bits.
DTC_RESPONDERS = (0x700, 0x724, 0x7D2, 0x747, 0x7A1, 0x7B0, 0x7B3, 0x7C4, 0x7D0, 0x792, 0x7A2)
PHYSICAL_CLEAR = (0x7A1, 0x7B3, 0x7C4, 0x7D0, 0x792, 0x7A2)
OBD_CLEAR_TX = 0x7DF
OBD_CLEAR_BUS = 0
OBD_CLEAR_FRAME = bytes.fromhex("0104000000000000")
OBD_CLEAR_RESPONSES = (0x7E8, 0x7EA, 0x7EB, 0x7ED, 0x7EE)
FRC_TX = 0x792
FRC_BUS = 0
DRCC_DIDS = (0x1903, 0x1905, 0x1906)
# 0x1B09 is an ancillary six-byte fail-safe-factor monitor. Current GTS+ places
# it in the FRC ISA/speed-limiter block, and P6 names the homologous factors
# "for Speed Limiter"; do not interpret it as the DRCC latch itself.
FRC_DIDS = (0x1B09, *DRCC_DIDS)
BRAKE_TX = 0x7B0
BRAKE_BUS = 0
BLANK_BOOT_F181 = bytes.fromhex("02" + "21" * 32)
# GTS+ ABS_P5 type61 mode 1: each value bit has a support bit in the
# preceding byte. Type62 supplies these MSB0 positions (monitor keys 81/82/500).
BRAKE_MONITORS = {
    0x102D: {"fail_status": 57, "fail_control": 58},
    0x102F: {"eps_communication_open": 74},
}
BRAKE_DIDS = tuple(BRAKE_MONITORS)
FLOW_CONTROL = bytes.fromhex("3000000000000000")
# ECUReset returns application identity before the rest of the peer domain has
# necessarily reinitialized. Keep reset handling quiet, then give the network a
# deliberate settle window before reading fault state or resetting another ECU.
RESET_APP_RETURN_INITIAL_WAIT_SECONDS = 2.0
DOMAIN_SETTLE_SECONDS = 5.0
FRC_RESET_RESPONSE_TIMEOUT_SECONDS = 1.5
BRAKE_RESET_RESPONSE_TIMEOUT_SECONDS = 1.0


class RecoveryError(RuntimeError):
    pass

@dataclass(frozen=True)
class RecoveryTarget:
    name: str
    eps_f181: bytes
    eps_bus: int


def recovery_target(meta_path: Path) -> RecoveryTarget:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target = meta.get("target", {})
    transport = meta.get("request", meta.get("control", {}))
    try:
        return RecoveryTarget(
            name=str(target["name"]),
            eps_f181=bytes.fromhex(target["application_f181_hex"]),
            eps_bus=int(transport["bus"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryError("metadata lacks target application F181 or diagnostic bus") from exc


def _single_frame(pdu: bytes) -> bytes:
    if not 0 < len(pdu) <= 7:
        raise ValueError("request PDU must fit one classic-CAN ISO-TP frame")
    return bytes((len(pdu),)) + pdu + bytes(7 - len(pdu))


def _drain(panda) -> list[tuple[int, bytes, int]]:
    return [(int(addr), bytes(data), int(bus)) for addr, data, bus in (panda.can_recv() or [])]


def isotp_request(panda, tx_addr: int, bus: int, pdu: bytes, *, timeout: float = 0.35) -> bytes:
    """Finite classic-CAN ISO-TP request for Toyota's ordinary +8 response IDs."""
    rx_addr = tx_addr + 8
    # Discard stale diagnostic traffic before binding a response to this request.
    _drain(panda)
    panda.can_send(tx_addr, _single_frame(pdu), bus)
    deadline = time.monotonic() + timeout
    assembly: bytearray | None = None
    total = 0
    next_sn = 1

    def matching_response(response: bytes) -> bytes | None:
        if response[:2] == bytes((0x7F, pdu[0])):
            if len(response) != 3:
                raise RecoveryError("malformed UDS negative response")
            # Response-pending is not the completed transaction. Keep the
            # original absolute deadline; a peer cannot extend this forever.
            return None if response[2] == 0x78 else response
        return response if response[:1] == bytes(((pdu[0] + 0x40) & 0xFF,)) else None

    while time.monotonic() < deadline:
        messages = _drain(panda)
        if not messages:
            time.sleep(0.001)
            continue
        for addr, frame, src in messages:
            if addr != rx_addr or src != bus or not frame:
                continue
            pci = frame[0] >> 4
            if pci == 0:
                length = frame[0] & 0x0F
                if not 0 < length <= 7 or length > len(frame) - 1:
                    raise RecoveryError("truncated or invalid classic ISO-TP single frame")
                response = matching_response(frame[1:1 + length])
                if response is not None:
                    return response
                continue
            if pci == 1 and len(frame) >= 2:
                total = ((frame[0] & 0x0F) << 8) | frame[1]
                if len(frame) != 8 or total <= 7:
                    raise RecoveryError("malformed classic ISO-TP first frame")
                next_sn = 1
                assembly = bytearray(frame[2:])
                panda.can_send(tx_addr, FLOW_CONTROL, bus)
                if len(assembly) >= total:
                    response = matching_response(bytes(assembly[:total]))
                    assembly = None
                    if response is not None:
                        return response
                continue
            if pci == 2 and assembly is not None:
                sn = frame[0] & 0x0F
                if sn != next_sn:
                    raise RecoveryError(
                        f"ISO-TP sequence mismatch for 0x{tx_addr:03X}: expected {next_sn}, got {sn}"
                    )
                assembly.extend(frame[1:])
                next_sn = (next_sn + 1) & 0x0F
                if len(assembly) >= total:
                    response = matching_response(bytes(assembly[:total]))
                    assembly = None
                    if response is not None:
                        return response
    raise TimeoutError(f"0x{tx_addr:03X}/bus{bus} timed out for {pdu.hex()}")


def _positive(response: bytes, request_sid: int, *, subfunction: int | None = None) -> bytes:
    if len(response) >= 3 and response[0] == 0x7F:
        raise RecoveryError(
            f"UDS 0x{request_sid:02X} negative response: NRC 0x{response[2]:02X}"
        )
    expected = (request_sid + 0x40) & 0xFF
    if not response or response[0] != expected:
        raise RecoveryError(f"UDS 0x{request_sid:02X} invalid response: {response.hex()}")
    if subfunction is not None:
        if len(response) < 2 or response[1] != subfunction:
            raise RecoveryError(
                f"UDS 0x{request_sid:02X} subfunction response mismatch: {response.hex()}"
            )
        return response[2:]
    return response[1:]


def parse_dtc_by_status(response: bytes) -> dict[str, Any]:
    data = _positive(response, 0x19, subfunction=0x02)
    if not data:
        raise RecoveryError("SID19/02 response has no status-availability mask")
    availability = data[0]
    records_raw = data[1:]
    if len(records_raw) % 4:
        raise RecoveryError(f"SID19/02 malformed record length: {len(records_raw)}")
    records = []
    for i in range(0, len(records_raw), 4):
        raw = records_raw[i:i + 4]
        records.append({
            "dtc": raw[:3].hex().upper(),
            "status": raw[3],
            "status_hex": f"0x{raw[3]:02X}",
            "fault_bits": raw[3] & FAULT_MASK,
        })
    return {
        "status_availability_mask": f"0x{availability:02X}",
        "records": records,
        "fault_records": [record for record in records if record["fault_bits"]],
    }


def decode_frc_did(did: int, response: bytes) -> dict[str, Any]:
    data = _positive(response, 0x22)
    if len(data) < 2 or int.from_bytes(data[:2], "big") != did:
        raise RecoveryError(f"FRC DID 0x{did:04X} response mismatch: {response.hex()}")
    value = data[2:]
    minimum_length = {0x1B09: 6, 0x1903: 1, 0x1905: 2, 0x1906: 6}.get(did)
    if minimum_length is None or len(value) < minimum_length:
        raise RecoveryError(f"FRC DID 0x{did:04X} has unsupported/truncated data: {value.hex()}")
    row: dict[str, Any] = {"did": f"0x{did:04X}", "raw": value.hex()}
    if did == 0x1B09:
        names = ("b1a", "b1b", "b2", "c1", "c2", "d1")
        row["fail_safe_factors"] = {name: value[index] for index, name in enumerate(names)}
    elif did == 0x1903 and value:
        row["control_mode"] = value[0]
        row["control_mode_label"] = {
            1: "DRCC all speed",
            2: "DRCC high speed",
            3: "constant speed",
            4: "DRCC all speed without brake hold",
        }.get(value[0], "unknown")
    elif did == 0x1905 and len(value) >= 2:
        row["cruise_control_allowed"] = bool(value[1] & 0x80)
    elif did == 0x1906 and len(value) >= 6:
        row["main_switch_recognized"] = bool(value[1] & 0x80)
        row["acc_not_available_icon"] = bool(value[5] & 0x80)
    return row


def decode_brake_did(did: int, response: bytes) -> dict[str, Any]:
    data = _positive(response, 0x22)
    if len(data) < 2 or int.from_bytes(data[:2], "big") != did:
        raise RecoveryError(f"Brake DID 0x{did:04X} response mismatch: {response.hex()}")
    value = data[2:]
    if did not in BRAKE_MONITORS:
        raise RecoveryError(f"unsupported Brake DID 0x{did:04X}")
    if not value:
        raise RecoveryError(f"Brake DID 0x{did:04X} has no value data")
    row: dict[str, Any] = {"did": f"0x{did:04X}", "raw": value.hex()}
    for name, bit in BRAKE_MONITORS[did].items():
        byte_index, bit_index = divmod(bit, 8)
        mask = 0x80 >> bit_index
        supported = byte_index < len(value) and bool(value[byte_index - 1] & mask)
        row[name] = bool(value[byte_index] & mask) if supported else None
    return row


def read_fault_state(panda, buses: dict[int, int]) -> dict[str, Any]:
    frc_bus = buses[FRC_TX]
    brake_bus = buses[BRAKE_TX]
    frc = {}
    for did in FRC_DIDS:
        key = f"0x{did:04X}"
        try:
            response = isotp_request(panda, FRC_TX, frc_bus, bytes((0x22, did >> 8, did & 0xFF)))
            frc[key] = {"available": True, **decode_frc_did(did, response)}
        except (TimeoutError, RecoveryError) as exc:
            # GTS+ DDB membership is not live-support proof. Keep newly added
            # fault-state oracles best-effort so an unsupported DID does not
            # destroy the already-qualified DTC-clear evidence path.
            frc[key] = {"did": key, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    brake = {}
    for did in BRAKE_DIDS:
        key = f"0x{did:04X}"
        try:
            response = isotp_request(panda, BRAKE_TX, brake_bus, bytes((0x22, did >> 8, did & 0xFF)))
            brake[key] = {"available": True, **decode_brake_did(did, response)}
        except (TimeoutError, RecoveryError) as exc:
            brake[key] = {"did": key, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"frc": frc, "brake": brake}


def read_peer_dtcs(panda, buses: dict[int, int]) -> dict[str, Any]:
    """Read FRC/Brake DTC status without clearing or changing diagnostic sessions."""
    peers = {}
    for name, address in (("frc", FRC_TX), ("brake", BRAKE_TX)):
        row: dict[str, Any] = {"address": f"0x{address:03X}", "bus": buses[address]}
        try:
            response = isotp_request(panda, address, buses[address], bytes.fromhex("1902ff"))
            row["raw"] = response.hex()
            row.update(parse_dtc_by_status(response))
            row["available"] = True
        except (TimeoutError, RecoveryError) as exc:
            row.update(available=False, error=f"{type(exc).__name__}: {exc}")
        peers[name] = row
    return peers


def drcc_permission_observed(oracles: dict[str, Any]) -> bool:
    # Current FRC_P5 monitor 199 distinguishes distance control (1/2/4) from
    # conventional constant-speed cruise (3). MAIN alone is not permission.
    return (oracles["0x1903"]["control_mode"] in (1, 2, 4)
            and oracles["0x1905"]["cruise_control_allowed"]
            and not oracles["0x1906"]["acc_not_available_icon"])


def classify_peer_health(state: dict[str, Any], dtcs: dict[str, Any]) -> dict[str, Any]:
    """Require supported FRC/Brake current-DTC checks; retain explicit monitor faults."""
    faults = []
    unavailable = []
    brake = state.get("brake", {})
    for did, monitors in BRAKE_MONITORS.items():
        key = f"0x{did:04X}"
        row = brake.get(key, {})
        for name in monitors:
            if row.get("available") is True and row.get(name) is True:
                faults.append(f"{key}.{name}")
    for peer in ("frc", "brake"):
        row = dtcs.get(peer, {})
        if row.get("available") is not True:
            unavailable.append(f"{peer}.dtcs")
            continue
        if not int(row["status_availability_mask"], 16) & CURRENT_DTC_FAILURE_MASK:
            unavailable.append(f"{peer}.dtcs.test_failed")
            continue
        for record in row["records"]:
            if record["status"] & CURRENT_DTC_FAILURE_MASK:
                faults.append(f"{peer}.dtc.{record['dtc']}")
    return {
        "state": "fault" if faults else "inconclusive" if unavailable else "healthy",
        "basis": "peer_dtc_test_failed_and_supported_brake_monitors",
        "faults": faults,
        "unavailable": unavailable,
    }


def _save_result(output: Path | None, result: dict[str, Any]) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)


def read_f181(
    panda, tx_addr: int, bus: int, label: str, *, timeout: float = 0.6,
) -> dict[str, Any]:
    """Read and record an ECU identity without imposing one vehicle's part number."""
    response = isotp_request(panda, tx_addr, bus, bytes.fromhex("22f181"), timeout=timeout)
    data = _positive(response, 0x22)
    if len(data) <= 2 or data[:2] != bytes.fromhex("f181"):
        raise RecoveryError(f"{label} F181 response malformed: {response.hex()}")
    return {
        "address": f"0x{tx_addr:03X}",
        "bus": bus,
        "f181": data[2:].hex(),
    }


def read_exact_f181(
    panda, tx_addr: int, bus: int, expected: bytes, label: str, *, timeout: float = 0.6,
) -> dict[str, Any]:
    result = read_f181(panda, tx_addr, bus, label, timeout=timeout)
    observed = bytes.fromhex(result["f181"])
    if observed != expected:
        raise RecoveryError(f"{label} exact identity mismatch: {observed.hex()}")
    return result


def _poll_exact_f181(
    panda, *, tx_addr: int, bus: int, expected_f181: bytes, label: str,
    timeout_s: float, poll_timeout_s: float = 0.18,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_s
    misses: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            return (
                read_exact_f181(panda, tx_addr, bus, expected_f181, label, timeout=poll_timeout_s),
                misses,
            )
        except (TimeoutError, RecoveryError) as exc:
            misses.append({
                "elapsed_ms": round((time.monotonic() - (deadline - timeout_s)) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
            time.sleep(0.05)
    return None, misses


def restart_exact_ecu(
    panda, *, tx_addr: int, bus: int, label: str, expected_f181: bytes | None = None,
    return_timeout: float = 5.0,
    reset_response_timeout: float = FRC_RESET_RESPONSE_TIMEOUT_SECONDS,
    verify_application_return: bool = True,
    retry_reset_once_on_timeout: bool = False,
) -> dict[str, Any]:
    """Enter programming and reset the ECU observed at one topology-defined address.

    With no expected identity, bind the restart to the exact F181 observed
    immediately before programming. Supplying an expected identity remains
    available for a caller with target-specific evidence.

    Brake/EPB is special. Live evidence shows its first 11 01 can time out while
    leaving it in programming context, while any F181/DID polling during restart
    can itself provoke peer faults. For Brake, the only permitted recovery action
    after a first 11 01 timeout is one immediate second 11 01. There are no reads,
    sleeps, session changes, or other requests between those two reset attempts,
    and no post-reset diagnostics when ``verify_application_return`` is false.
    """
    if expected_f181 is None:
        before = read_f181(panda, tx_addr, bus, label)
        expected_f181 = bytes.fromhex(before["f181"])
    else:
        before = read_exact_f181(panda, tx_addr, bus, expected_f181, label)
    programming_response = isotp_request(panda, tx_addr, bus, bytes.fromhex("1002"), timeout=1.5)
    _positive(programming_response, 0x10, subfunction=0x02)

    attempts: list[dict[str, Any]] = []
    max_attempts = 2 if retry_reset_once_on_timeout else 1
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        response_hex: str | None = None
        error: str | None = None
        try:
            reset_response = isotp_request(
                panda, tx_addr, bus, bytes.fromhex("1101"), timeout=reset_response_timeout,
            )
            _positive(reset_response, 0x11, subfunction=0x01)
            response_hex = reset_response.hex()
        except TimeoutError as exc:
            error = f"{type(exc).__name__}: {exc}"

        attempts.append({
            "attempt": attempt,
            "response": response_hex,
            "error": error,
            "call_ms": round((time.monotonic() - started) * 1000, 3),
        })

        if response_hex is not None:
            break
        if attempt < max_attempts:
            # CRITICAL: immediate 11 01 retry only. No F181, DID, sleep, or any
            # other diagnostic traffic is allowed between Brake reset attempts.
            continue

    result: dict[str, Any] = {
        "label": label,
        "before": before,
        "programming_request": "1002",
        "programming_response": programming_response.hex(),
        "reset_request": "1101",
        "reset_attempts": attempts,
        "security_access_used": False,
        "application_return_verified": False,
    }
    if not verify_application_return:
        result["post_reset_diagnostics"] = "none"
        return result

    time.sleep(RESET_APP_RETURN_INITIAL_WAIT_SECONDS)
    returned, misses = _poll_exact_f181(
        panda, tx_addr=tx_addr, bus=bus, expected_f181=expected_f181, label=label,
        timeout_s=return_timeout, poll_timeout_s=0.25,
    )
    if returned is None:
        raise RecoveryError(f"{label} application F181 did not return after reset")
    result.update({
        "quiet_wait_before_f181_s": RESET_APP_RETURN_INITIAL_WAIT_SECONDS,
        "f181_poll_misses": misses,
        "after": returned,
        "application_return_verified": True,
    })
    return result


def restart_brake_known_good(output: Path | None, target: RecoveryTarget) -> dict[str, Any]:
    """Run the field-proven Brake reset procedure without changing its timing."""
    try:
        from panda import Panda  # type: ignore
        from opendbc.car.uds import (  # type: ignore
            UdsClient, SESSION_TYPE, RESET_TYPE, DATA_IDENTIFIER_TYPE,
        )
    except ImportError as exc:
        raise RecoveryError("panda/opendbc UDS unavailable") from exc

    tx = BRAKE_TX
    rx = BRAKE_TX + 8
    bus = BRAKE_BUS
    # Preserve the field-proven no-read pre-reset sequence. Any non-blank F181
    # at the fixed Brake address proves that an application returned; record
    # the vehicle-specific identity rather than imposing Camry's part number.
    panda = Panda()
    result: dict[str, Any] = {
        "schema": "tss3-brake-reset-known-good-v1",
        "target": target.name,
        "domain": "brake",
        "procedure": "field-proven-udsclient-block",
        "codeflash_write": False,
        "security_access_used": False,
        "eps_reset": False,
        "eps_power_cycle": False,
        "reset_attempts": [],
        "f181_samples": [],
    }
    try:
        # Keep this sequence identical to the manually proven block. Do not add
        # heartbeat changes, canfd_auto changes, pre-reset F181 reads, or custom
        # ISO-TP traffic here.
        panda.set_safety_mode(DIAG_SAFETY, DIAG_SAFETY_PARAM)

        client = UdsClient(
            panda, tx, rx_addr=rx, bus=bus,
            timeout=1.0, response_pending_timeout=2.0,
        )
        client.diagnostic_session_control(SESSION_TYPE.PROGRAMMING)
        result["programming_session_accepted"] = True

        reset_accepted = False
        for attempt in (1, 2):
            reset_client = UdsClient(
                panda, tx, rx_addr=rx, bus=bus,
                timeout=0.35, response_pending_timeout=0.35,
            )
            started = time.monotonic()
            row: dict[str, Any] = {"attempt": attempt}
            try:
                reset_client.ecu_reset(RESET_TYPE.HARD)
                row["accepted"] = True
                row["error"] = None
                reset_accepted = True
            except Exception as exc:  # exact live behavior: any exception triggers attempt 2
                row["accepted"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["call_ms"] = round((time.monotonic() - started) * 1000, 3)
            result["reset_attempts"].append(row)
            if row["accepted"]:
                break
            if attempt == 1:
                time.sleep(0.05)

        result["reset_response_observed"] = reset_accepted

        time.sleep(2.0)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            try:
                query = UdsClient(
                    panda, tx, rx_addr=rx, bus=bus,
                    timeout=0.25, response_pending_timeout=0.25,
                )
                f181 = bytes(query.read_data_by_identifier(
                    DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION
                ))
                result["f181_samples"].append(f181.hex())
                if f181 and f181 != BLANK_BOOT_F181:
                    result["application_f181"] = f181.hex()
                    result["verdict"] = "brake_application_returned"
                    break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            result["verdict"] = "brake_application_did_not_return"
            raise RecoveryError("Brake application F181 did not return")
    except Exception as exc:
        if "verdict" not in result:
            result["verdict"] = "brake_restart_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            panda.close()
        except Exception:
            pass
        _save_result(output, result)
    return result


def restart_one_domain(
    domain: str, output: Path | None, target: RecoveryTarget,
) -> dict[str, Any]:
    """Reset one peer ECU; Brake uses the field-proven UdsClient block."""
    if domain not in {"brake", "frc"}:
        raise RecoveryError(f"unsupported control domain: {domain}")
    if domain == "brake":
        return restart_brake_known_good(output, target)
    try:
        from panda import Panda  # type: ignore
    except ImportError as exc:
        raise RecoveryError("panda module unavailable; run from the packaged comma/openpilot environment") from exc

    serials = Panda.list()
    if len(serials) != 1:
        raise RecoveryError(f"expected exactly one Panda, found {len(serials)}: {serials}")
    panda = Panda(serials[0])
    result: dict[str, Any] = {
        "schema": "tss3-single-control-domain-restart-v1",
        "target": target.name,
        "domain": domain,
        "codeflash_write": False,
        "security_access_used": False,
        "eps_reset": False,
        "eps_power_cycle": False,
    }
    try:
        panda.set_heartbeat_disabled()
        panda.set_safety_mode(DIAG_SAFETY, DIAG_SAFETY_PARAM)
        panda.set_canfd_auto(0, False)
        panda.set_canfd_auto(1, False)
        result["eps_identity_before"] = read_exact_f181(
            panda, EPS_TX, target.eps_bus, target.eps_f181, "EPS", timeout=0.8,
        )
        result["restart"] = restart_exact_ecu(
            panda, tx_addr=FRC_TX, bus=FRC_BUS, label="FRC",
            reset_response_timeout=FRC_RESET_RESPONSE_TIMEOUT_SECONDS,
        )
        result["eps_identity_after"] = read_exact_f181(
            panda, EPS_TX, target.eps_bus, target.eps_f181, "EPS", timeout=0.8,
        )
        result["verdict"] = "frc_application_restarted"
    except Exception as exc:
        result["verdict"] = f"{domain}_restart_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            panda.set_safety_mode(SILENT_SAFETY)
        except Exception:
            pass
        try:
            panda.close()
        except Exception:
            pass
        _save_result(output, result)
    return result


def control_domain_state(
    output: Path | None, target: RecoveryTarget, *, panda: Any | None = None,
) -> dict[str, Any]:
    """Read peer identities and fault state after the EPS signer takes over."""
    try:
        from panda import Panda  # type: ignore
    except ImportError as exc:
        raise RecoveryError("panda module unavailable; run from the packaged comma/openpilot environment") from exc
    owns_panda = panda is None
    if panda is None:
        serials = Panda.list()
        if len(serials) != 1:
            raise RecoveryError(f"expected exactly one Panda, found {len(serials)}: {serials}")
        panda = Panda(serials[0])
    result: dict[str, Any] = {
        "schema": "tss3-control-domain-state-v4",
        "target": target.name,
        "eps_identity": {
            "verification": "bound_by_install_pre_helper_f181",
            "expected_f181_hex": target.eps_f181.hex(),
        },
    }
    try:
        panda.set_heartbeat_disabled()
        panda.set_safety_mode(DIAG_SAFETY, DIAG_SAFETY_PARAM)
        panda.set_canfd_auto(0, False)
        panda.set_canfd_auto(1, False)
        result["frc_identity"] = read_f181(panda, FRC_TX, FRC_BUS, "FRC", timeout=0.8)
        result["brake_identity"] = read_f181(panda, BRAKE_TX, BRAKE_BUS, "Brake/EPB", timeout=0.8)
        buses = {FRC_TX: FRC_BUS, BRAKE_TX: BRAKE_BUS}
        dtcs = read_peer_dtcs(panda, buses)
        result["dtc_state"] = dtcs
        state = read_fault_state(panda, buses)
        result["fault_state"] = state
        health = classify_peer_health(state, dtcs)
        result["peer_health"] = health
        result["peer_health_observed"] = health["state"] == "healthy"
        result["verdict"] = {
            "healthy": "control_domains_healthy",
            "fault": "control_domain_fault_observed",
            "inconclusive": "control_domain_state_inconclusive",
        }[health["state"]]
    finally:
        if owns_panda:
            try:
                panda.set_safety_mode(SILENT_SAFETY)
            except Exception:
                pass
            try:
                panda.close()
            except Exception:
                pass
        _save_result(output, result)
    return result


def restart_control_domains(output: Path | None, target: RecoveryTarget) -> dict[str, Any]:
    """Keep EPS powered and reset FRC -> Brake -> FRC with quiet settles."""
    try:
        from panda import Panda  # type: ignore
    except ImportError as exc:
        raise RecoveryError("panda module unavailable; run from the packaged comma/openpilot environment") from exc

    serials = Panda.list()
    if len(serials) != 1:
        raise RecoveryError(f"expected exactly one Panda, found {len(serials)}: {serials}")
    panda = Panda(serials[0])
    result: dict[str, Any] = {
        "schema": "tss3-control-domain-restart-v1",
        "target": target.name,
        "sequence": [
            "FRC 0x792: 10 02 -> 11 01",
            "Brake/EPB 0x7B0: 10 02 -> 11 01",
            "FRC 0x792: 10 02 -> 11 01",
        ],
        "codeflash_write": False,
        "security_access_used": False,
        "eps_reset": False,
        "eps_power_cycle": False,
    }
    try:
        panda.set_heartbeat_disabled()
        panda.set_safety_mode(DIAG_SAFETY, DIAG_SAFETY_PARAM)
        panda.set_canfd_auto(0, False)
        panda.set_canfd_auto(1, False)

        result["eps_identity_before"] = read_exact_f181(
            panda, EPS_TX, target.eps_bus, target.eps_f181, "EPS", timeout=0.8,
        )
        result["frc_identity_before"] = read_f181(
            panda, FRC_TX, FRC_BUS, "FRC", timeout=0.8,
        )
        result["brake_identity_before"] = read_f181(
            panda, BRAKE_TX, BRAKE_BUS, "Brake/EPB", timeout=0.8,
        )
        frc_f181 = bytes.fromhex(result["frc_identity_before"]["f181"])
        brake_f181 = bytes.fromhex(result["brake_identity_before"]["f181"])
        fixed_buses = {FRC_TX: FRC_BUS, BRAKE_TX: BRAKE_BUS}
        result["pre_restart_fault_state"] = read_fault_state(panda, fixed_buses)
        _save_result(output, result)

        result["frc_prereset"] = restart_exact_ecu(
            panda, tx_addr=FRC_TX, bus=FRC_BUS, expected_f181=frc_f181, label="FRC pre-reset",
            reset_response_timeout=FRC_RESET_RESPONSE_TIMEOUT_SECONDS,
        )
        time.sleep(DOMAIN_SETTLE_SECONDS)
        result["frc_prereset"]["domain_settle_s"] = DOMAIN_SETTLE_SECONDS
        result["after_frc_prereset_fault_state"] = read_fault_state(panda, fixed_buses)
        _save_result(output, result)

        result["brake_restart"] = restart_exact_ecu(
            panda, tx_addr=BRAKE_TX, bus=BRAKE_BUS, expected_f181=brake_f181, label="Brake/EPB",
            reset_response_timeout=BRAKE_RESET_RESPONSE_TIMEOUT_SECONDS, verify_application_return=False,
            retry_reset_once_on_timeout=True,
        )
        time.sleep(DOMAIN_SETTLE_SECONDS)
        result["brake_restart"]["domain_settle_s"] = DOMAIN_SETTLE_SECONDS
        result["after_brake_fault_state"] = read_fault_state(panda, fixed_buses)
        _save_result(output, result)

        result["frc_final_restart"] = restart_exact_ecu(
            panda, tx_addr=FRC_TX, bus=FRC_BUS, expected_f181=frc_f181, label="FRC final reset",
            reset_response_timeout=FRC_RESET_RESPONSE_TIMEOUT_SECONDS,
        )
        time.sleep(DOMAIN_SETTLE_SECONDS)
        result["frc_final_restart"]["domain_settle_s"] = DOMAIN_SETTLE_SECONDS
        result["after_frc_fault_state"] = read_fault_state(panda, fixed_buses)
        oracles = {
            f"0x{did:04X}": result["after_frc_fault_state"]["frc"][f"0x{did:04X}"]
            for did in DRCC_DIDS
        }
        unavailable = [key for key, row in oracles.items() if not row.get("available")]
        if unavailable:
            raise RecoveryError("required post-restart FRC DRCC oracle unavailable: " + ", ".join(unavailable))
        result["frc_oracles"] = oracles
        result["drcc_permission_observed"] = drcc_permission_observed(oracles)
        result["eps_identity_after"] = read_exact_f181(
            panda, EPS_TX, target.eps_bus, target.eps_f181, "EPS", timeout=0.8,
        )
        result["verdict"] = (
            "control_domains_restarted_drcc_permission_restored"
            if result["drcc_permission_observed"]
            else "control_domains_restarted_drcc_permission_still_denied"
        )
    except Exception as exc:
        result["verdict"] = "control_domain_restart_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            panda.set_safety_mode(SILENT_SAFETY)
        except Exception:
            pass
        try:
            panda.close()
        except Exception:
            pass
        _save_result(output, result)
    return result


def _discover_dtc_bus(panda, address: int, eps_bus: int) -> tuple[int, bytes]:
    preferred = (eps_bus, OBD_CLEAR_BUS) if address in (0x7A1, 0x7B0) else (OBD_CLEAR_BUS, eps_bus)
    errors = []
    for bus in preferred:
        try:
            response = isotp_request(panda, address, bus, bytes.fromhex("1902ff"))
            parse_dtc_by_status(response)
            return bus, response
        except (TimeoutError, RecoveryError) as exc:
            errors.append(f"bus{bus}: {exc}")
    raise RecoveryError(f"0x{address:03X} did not answer SID19/02 on stock diagnostic routes ({'; '.join(errors)})")


def _functional_mode04(panda) -> dict[str, str]:
    _drain(panda)
    panda.can_send(OBD_CLEAR_TX, OBD_CLEAR_FRAME, OBD_CLEAR_BUS)
    deadline = time.monotonic() + 1.0
    responses: dict[int, bytes] = {}
    expected = set(OBD_CLEAR_RESPONSES)
    while time.monotonic() < deadline and set(responses) != expected:
        messages = _drain(panda)
        if not messages:
            time.sleep(0.002)
            continue
        for addr, data, bus in messages:
            if bus == OBD_CLEAR_BUS and addr in expected and len(data) == 8 and data[:2] == b"\x01\x44":
                responses[addr] = data
    missing = sorted(expected - set(responses))
    if missing:
        raise RecoveryError("functional Mode04 missing positive responses: " + ", ".join(hex(x) for x in missing))
    return {f"0x{addr:03X}": responses[addr].hex() for addr in sorted(responses)}


def plan(target: RecoveryTarget) -> dict[str, Any]:
    return {
        "schema": "tss3-post-install-recovery-plan-v1",
        "target": target.name,
        "eps_identity_route": {"request": "22F181", "address": "0x7A1", "bus": target.eps_bus},
        "preserve_before_clear": "physical SID19 02 FF snapshot from all 11 known responders",
        "physical_clear": {"request": "14FFFFFF", "addresses": [f"0x{x:03X}" for x in PHYSICAL_CLEAR]},
        "functional_clear": {
            "bus": OBD_CLEAR_BUS,
            "address": f"0x{OBD_CLEAR_TX:03X}",
            "frame": OBD_CLEAR_FRAME.hex(),
            "required_positive_responders": [f"0x{x:03X}" for x in OBD_CLEAR_RESPONSES],
        },
        "acceptance": {
            "fault_mask": f"0x{FAULT_MASK:02X}",
            "all_reachable_dtc_fault_bits_clear": True,
            "frc_oracles": [f"0x{x:04X}" for x in FRC_DIDS],
            "brake_oracles": [f"0x{x:04X}" for x in BRAKE_DIDS],
            "capture_pre_and_post_clear_fault_state": True,
            "drcc_permission_must_be_observed": True,
        },
        "mutations_excluded": [
            "ECU reset", "SecurityAccess", "RoutineControl", "firmware write", "flash erase",
            "steering command", "EPS power cycle",
        ],
    }


def execute(output: Path | None, target: RecoveryTarget) -> dict[str, Any]:
    try:
        from panda import Panda  # type: ignore
    except ImportError as exc:
        raise RecoveryError("panda module unavailable; run from the packaged comma/openpilot environment") from exc

    serials = Panda.list()
    if len(serials) != 1:
        raise RecoveryError(f"expected exactly one Panda, found {len(serials)}: {serials}")
    panda = Panda(serials[0])
    result: dict[str, Any] = {
        "schema": "tss3-post-install-recovery-v1",
        "target": target.name,
        "codeflash_write": False,
        "diagnostic_memory_mutated": True,
        "eps_reset": False,
    }
    try:
        panda.set_heartbeat_disabled()
        panda.set_safety_mode(DIAG_SAFETY, DIAG_SAFETY_PARAM)
        panda.set_canfd_auto(0, False)
        panda.set_canfd_auto(1, False)

        f181 = isotp_request(panda, EPS_TX, target.eps_bus, bytes.fromhex("22f181"), timeout=0.6)
        f181_data = _positive(f181, 0x22)
        if len(f181_data) < 2 or f181_data[:2] != bytes.fromhex("f181"):
            raise RecoveryError(f"EPS F181 response malformed: {f181.hex()}")
        observed_f181 = f181_data[2:]
        if observed_f181 != target.eps_f181:
            raise RecoveryError(f"target EPS identity mismatch: {observed_f181.hex()}")
        result["eps_f181"] = observed_f181.hex()

        buses: dict[int, int] = {}
        before: dict[str, Any] = {}
        for address in DTC_RESPONDERS:
            bus, response = _discover_dtc_bus(panda, address, target.eps_bus)
            buses[address] = bus
            before[f"0x{address:03X}"] = {"bus": bus, **parse_dtc_by_status(response)}
        result["pre_clear"] = before
        result["discovered_buses"] = {f"0x{k:03X}": v for k, v in sorted(buses.items())}
        result["pre_clear_fault_state"] = read_fault_state(panda, buses)

        # Save the pre-clear evidence before the first mutating request.
        _save_result(output, result)
        physical = {}
        result["physical_clear"] = physical
        for address in PHYSICAL_CLEAR:
            response = isotp_request(panda, address, buses[address], bytes.fromhex("14ffffff"), timeout=2.0)
            _positive(response, 0x14)
            physical[f"0x{address:03X}"] = {"bus": buses[address], "response": response.hex()}
            _save_result(output, result)
        result["functional_mode04"] = _functional_mode04(panda)

        after: dict[str, Any] = {}
        uncleared: dict[str, Any] = {}
        for address in DTC_RESPONDERS:
            response = isotp_request(panda, address, buses[address], bytes.fromhex("1902ff"))
            decoded = parse_dtc_by_status(response)
            after[f"0x{address:03X}"] = {"bus": buses[address], **decoded}
            if decoded["fault_records"]:
                uncleared[f"0x{address:03X}"] = decoded["fault_records"]
        result["post_clear"] = after
        result["remaining_fault_records"] = uncleared
        result["post_clear_fault_state"] = read_fault_state(panda, buses)
        if uncleared:
            result["verdict"] = "dtc_fault_bits_remain"
        else:
            oracles = {
                f"0x{did:04X}": result["post_clear_fault_state"]["frc"][f"0x{did:04X}"]
                for did in DRCC_DIDS
            }
            unavailable = [key for key, row in oracles.items() if not row.get("available")]
            if unavailable:
                raise RecoveryError("required FRC DRCC oracle unavailable: " + ", ".join(unavailable))
            result["frc_oracles"] = oracles
            result["drcc_permission_observed"] = drcc_permission_observed(oracles)
            result["verdict"] = "drcc_permission_restored" if result["drcc_permission_observed"] else "dtcs_clear_drcc_permission_still_denied"
    except Exception as exc:
        result["verdict"] = "recovery_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            panda.set_safety_mode(SILENT_SAFETY)
        except Exception:
            pass
        try:
            panda.close()
        except Exception:
            pass
        _save_result(output, result)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    clear = sub.add_parser("clear")
    clear.add_argument("--execute", action="store_true")
    clear.add_argument("--output", type=Path)
    restart = sub.add_parser("restart-control-domains")
    restart.add_argument("--execute", action="store_true")
    restart.add_argument("--output", type=Path)
    one = sub.add_parser("restart-domain")
    one.add_argument("--execute", action="store_true")
    one.add_argument("--domain", choices=("brake", "frc"), required=True)
    one.add_argument("--output", type=Path)
    state = sub.add_parser("state")
    state.add_argument("--output", type=Path)
    for command_parser in (plan_parser, clear, restart, one, state):
        command_parser.add_argument("--meta", type=Path, required=True)
    args = ap.parse_args()
    target = recovery_target(args.meta)

    if args.command == "plan":
        print(json.dumps(plan(target), indent=2, sort_keys=True))
        return 0
    if args.command == "state":
        result = control_domain_state(args.output, target)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("peer_health_observed") else 2
    if not args.execute:
        raise SystemExit(f"{args.command} requires --execute")
    if args.command == "clear":
        result = execute(args.output, target)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("verdict") == "drcc_permission_restored" else 2
    if args.command == "restart-domain":
        result = restart_one_domain(args.domain, args.output, target)
        print(json.dumps(result, indent=2, sort_keys=True))
        expected = "brake_application_returned" if args.domain == "brake" else "frc_application_restarted"
        return 0 if result.get("verdict") == expected else 2
    result = restart_control_domains(args.output, target)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("verdict") == "control_domains_restarted_drcc_permission_restored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
