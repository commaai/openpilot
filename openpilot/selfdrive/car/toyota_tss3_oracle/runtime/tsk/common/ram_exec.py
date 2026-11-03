#!/usr/bin/env python3
"""Shared authenticated 4 KiB Toyota EPS RAM-exec bootstrap.

This module owns only diagnostic/session/RAM-upload mechanics. It contains no
CodeFlash read protocol and no flash erase/program primitive. Callers provide a
configured 4 KiB payload and interpret its raw 0x7A9 telemetry.
"""

from __future__ import annotations

import importlib
import json
import os
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from Crypto.Cipher import AES

TX_ADDR = 0x7A1
RX_ADDR = 0x7A9
ELM327_SAFETY_MODE = 3
RAM_LOAD_ADDR = 0xFEBF0000
RAM_LOAD_SIZE = 0x1000
TRIGGER_ADDR = 0x000E0000
TRIGGER_LENGTH = 0x8000
DID_201_KEY = b"\x00" * 16
DID_202_IV = b"\x00" * 16
# These P1M-E bootstrap roots are byte-identical in the tracked Sienna
# 8965B4512000 and Camry 8965F3307000 CodeFlash images. The same bytes are
# public in LoChuan's RH850_P1M-E image at CodeFlash offsets 0xBFD8/0xBFE8.
TOYOTA_P1ME_PAYLOAD_BUILD_SECRET = bytes.fromhex("ba052435f8843f985fd1329d2b6117b0")
TOYOTA_P1ME_BOOT_SECURITY_ACCESS_SECRET = bytes.fromhex("f05f36b7d78c03e24ab4faef2a57d044")
TOYOTA_P1ME_PAYLOAD_BUILD_SECRET_SOURCE = "builtin:toyota-p1me-codeflash-0xBFD8"
TOYOTA_P1ME_BOOT_SECURITY_ACCESS_SECRET_SOURCE = "builtin:toyota-p1me-codeflash-0xBFE8"
DIRECT_PANDA_LEASE_PATH = Path("/tmp/openpilot-pandad-direct-lease")
DIRECT_PANDA_LEASE_READY_PATH = Path("/tmp/openpilot-pandad-direct-ready")


class RamExecError(RuntimeError):
    pass


@dataclass(frozen=True)
class RamExecGeometry:
    load_addr: int = RAM_LOAD_ADDR
    load_size: int = RAM_LOAD_SIZE
    evidence: str = "firmware-static:sienna-8965b4512000"

    def validate(self) -> None:
        if isinstance(self.load_addr, bool) or not isinstance(self.load_addr, int):
            raise RamExecError("RAM load address must be an integer")
        if self.load_addr < 0 or self.load_addr + self.load_size > 0x1_0000_0000:
            raise RamExecError("RAM load address/range must fit uint32")
        if self.load_addr & 0x3:
            raise RamExecError("RAM load address must be 4-byte aligned")
        # The authenticated package layout, TransferData loop, callback slot, and
        # CRC/CMAC envelope are all currently fixed at one 4 KiB payload. A target
        # may move that window, but changing its size requires a separate backend.
        if self.load_size != RAM_LOAD_SIZE:
            raise RamExecError(f"RAM payload geometry requires size 0x{RAM_LOAD_SIZE:X}")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise RamExecError("RAM-exec geometry requires a non-empty evidence label")


SIENNA_RAM_EXEC_GEOMETRY = RamExecGeometry()


def explicit_ram_exec_geometry(
    *,
    load_addr: int,
    evidence: str,
    load_size: int = RAM_LOAD_SIZE,
) -> RamExecGeometry:
    geometry = RamExecGeometry(load_addr=load_addr, load_size=load_size, evidence=evidence)
    geometry.validate()
    return geometry


@dataclass(frozen=True)
class RamExecRoute:
    bus: int
    elm327_param: int
    uds_variant: str
    cpu_index: int = 0

    def validate(self) -> None:
        if self.bus not in (0, 1, 2):
            raise RamExecError("Panda logical bus must be 0, 1, or 2")
        if not 0 <= self.elm327_param <= 0xFFFF:
            raise RamExecError("ELM327 safety parameter must fit uint16")
        if self.uds_variant not in ("old", "new"):
            raise RamExecError("UDS variant must be explicitly 'old' or 'new'")
        if self.cpu_index not in (0, 1):
            raise RamExecError("cpu_index must be 0 or 1")


def route_from_session(
    session_dir: Path,
    *,
    uds_variant: str,
    cpu_index: int = 0,
    bus: int | None = None,
    elm327_param: int | None = None,
) -> tuple[RamExecRoute, dict]:
    state = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    recorded_bus = state.get("routing", {}).get("diagnostic_bus")
    if not isinstance(recorded_bus, int):
        raise RamExecError("session has no resolved diagnostic_bus; record a read-only F181 probe first")
    recorded_elm = state.get("routing", {}).get("elm327_param")
    if not isinstance(recorded_elm, int):
        raise RamExecError("session has no explicit elm327_param")
    if bus is not None and bus != recorded_bus:
        raise RamExecError("explicit bus disagrees with recorded session routing")
    if elm327_param is not None and elm327_param != recorded_elm:
        raise RamExecError("explicit ELM327 parameter disagrees with recorded session routing")
    route = RamExecRoute(recorded_bus, recorded_elm, uds_variant, cpu_index)
    route.validate()
    return route, state


def explicit_route(*, bus: int, elm327_param: int, uds_variant: str, cpu_index: int = 0) -> RamExecRoute:
    route = RamExecRoute(bus, elm327_param, uds_variant, cpu_index)
    route.validate()
    return route


def bootstrap_protocol_values(route: RamExecRoute) -> tuple[bytes, bytes, bytes]:
    """Return DID 0203, RequestDownload memory ID, and routine magic.

    Toyota's old boot stack uses an all-zero DID 0203 regardless of CPU index.
    The 01 00 00 00 00 CPU0 offset selector belongs to the new-stack grammar.
    RequestDownload memory ID remains CPU-indexed on both stacks.
    """
    route.validate()
    did_203 = (
        b"\x01\x00\x00\x00\x00"
        if route.uds_variant == "new" and route.cpu_index == 0
        else b"\x00" * 5
    )
    mem_id = b"\x01" if route.cpu_index == 0 else b"\x00"
    routine_magic = b"\x45\x01" if route.uds_variant == "new" else b"\x45\x00"
    return did_203, mem_id, routine_magic


def load_security_secret() -> tuple[bytes, str]:
    return TOYOTA_P1ME_BOOT_SECURITY_ACCESS_SECRET, TOYOTA_P1ME_BOOT_SECURITY_ACCESS_SECRET_SOURCE


def load_payload_secret() -> tuple[bytes, str]:
    return TOYOTA_P1ME_PAYLOAD_BUILD_SECRET, TOYOTA_P1ME_PAYLOAD_BUILD_SECRET_SOURCE


def _process_state(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    if len(fields) < 3:
        raise OSError("short /proc stat")
    return fields[2]


def ensure_boardd_stopped() -> None:
    # Modern openpilot runs Panda ownership under pandad; older environments may
    # still expose boardd. Direct Panda diagnostics must not race either owner.
    paused_supervisor = os.environ.get("DIRECT_PANDA_PAUSED_SUPERVISOR_PID")
    lease_supervisor = os.environ.get("DIRECT_PANDA_LEASE_SUPERVISOR_PID")
    lease_id = os.environ.get("DIRECT_PANDA_LEASE_ID")
    allowed_pid: int | None = None
    if paused_supervisor and (lease_supervisor or lease_id):
        raise RamExecError("conflicting paused and cooperative pandad ownership declarations")
    if paused_supervisor:
        try:
            allowed_pid = int(paused_supervisor)
            state = _process_state(allowed_pid)
        except (OSError, ValueError) as exc:
            raise RamExecError("declared paused pandad supervisor is unavailable") from exc
        if state not in ("T", "t"):
            raise RamExecError("declared pandad supervisor is not stopped")
    elif lease_supervisor or lease_id:
        if not lease_supervisor or not lease_id:
            raise RamExecError("incomplete cooperative pandad lease declaration")
        try:
            allowed_pid = int(lease_supervisor)
            _process_state(allowed_pid)
            lease = DIRECT_PANDA_LEASE_PATH.read_text(encoding="utf-8").strip()
            ready = DIRECT_PANDA_LEASE_READY_PATH.read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as exc:
            raise RamExecError("declared cooperative pandad lease is unavailable") from exc
        if lease != lease_id or ready != lease_id:
            raise RamExecError("cooperative pandad lease is not acknowledged")
        # The managed wrapper writes the exact ready token only after its native
        # pandad child has exited. Re-running three process-table commands here
        # adds hundreds of milliseconds to the caught-boot hot path without
        # strengthening that ownership proof; the lease grantor already did the
        # global owner check before publishing ready.
        return
    checks = (
        ["pgrep", "-f", r"selfdrive\.pandad\.pandad"],
        ["pidof", "pandad"],
        ["pidof", "boardd"],
    )
    for argv in checks:
        try:
            result = subprocess.run(argv, capture_output=True, check=False, text=True)
        except FileNotFoundError:
            continue
        found = [int(token) for token in result.stdout.split() if token.isdigit()]
        if allowed_pid is not None:
            found = [pid for pid in found if pid != allowed_pid]
        if result.returncode == 0 and found:
            raise RamExecError(
                "boardd/pandad is running; stop openpilot before direct Panda diagnostics"
            )


def _import_uds():
    for name in ("opendbc.car.uds", "panda.python.uds", "panda.uds"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RamExecError("cannot import a supported UDS module")


def _import_isotp_send():
    for name in ("opendbc.car.isotp", "panda.python.isotp", "panda.isotp"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(module, "isotp_send", None)
        if fn is not None:
            return fn
    raise RamExecError("cannot import isotp_send")


def _make_uds_client(panda, uds_mod, route: RamExecRoute, *, timeout: float = 0.5, response_pending_timeout: float = 3.0):
    return uds_mod.UdsClient(
        panda, TX_ADDR, RX_ADDR, route.bus,
        timeout=timeout, response_pending_timeout=response_pending_timeout,
    )


def _boot_route_dict(route: RamExecRoute) -> dict[str, object]:
    return {
        "tx": TX_ADDR,
        "rx": RX_ADDR,
        "tx_bus": route.bus,
        "rx_bus": route.bus,
        "elm327_param": route.elm327_param,
        "semantic_path": "obd" if route.elm327_param == 0 else "normal-harness",
    }


def _enter_programming(uds_client, uds_mod, variant: str) -> None:
    session = uds_mod.SESSION_TYPE
    uds_client.diagnostic_session_control(session.DEFAULT)
    uds_client.diagnostic_session_control(session.EXTENDED_DIAGNOSTIC)
    uds_client.diagnostic_session_control(session.PROGRAMMING)
    # The external community uploader repeats the old-stack transition. Preserve
    # that observed transport behavior explicitly rather than guessing by SW ID.
    if variant == "old":
        uds_client.diagnostic_session_control(session.DEFAULT)
        uds_client.diagnostic_session_control(session.EXTENDED_DIAGNOSTIC)
        uds_client.diagnostic_session_control(session.PROGRAMMING)


def _prepare_boot_client(
    panda,
    uds_mod,
    route: RamExecRoute,
    boot_route: dict,
    *,
    expected_boot_f181_hex: str | None,
    confirmed_boot_f181: tuple[str, str | None] | None = None,
):
    """Bind exact boot identity, preserve the field-proven probe, then enter PROGRAMMING."""
    if not (
        int(boot_route.get("tx", -1)) == TX_ADDR
        and int(boot_route.get("rx", -1)) == RX_ADDR
        and int(boot_route.get("tx_bus", -1)) == route.bus
        and int(boot_route.get("rx_bus", -1)) == route.bus
        and int(boot_route.get("elm327_param", -1)) == route.elm327_param
    ):
        raise RamExecError(f"bootloader route changed from the preserved route: {boot_route}")

    boot = _make_uds_client(panda, uds_mod, route, timeout=0.5, response_pending_timeout=3.0)
    if confirmed_boot_f181 is None:
        boot_f181_hex, boot_f181_ascii = _read_f181(boot, uds_mod)
    else:
        boot_f181_hex, boot_f181_ascii = confirmed_boot_f181
    if boot_f181_hex is None:
        raise RamExecError("bootloader endpoint is present but F181 could not be read")
    if expected_boot_f181_hex is not None and boot_f181_hex.lower() != expected_boot_f181_hex.lower():
        raise RamExecError(
            f"boot F181 mismatch: expected {expected_boot_f181_hex}, observed {boot_f181_hex}"
        )

    # Preserve the exact successful 2026-08-26 boot-session choreography. The F33 boot
    # rejected this read-only SID 0x23 probe with NRC 0x7F; the important contract is
    # that the proven host path enters EXTENDED and performs the probe before rebuilding
    # one clean DEFAULT -> EXTENDED -> PROGRAMMING ladder for SecurityAccess.
    sid23_probe = {"status": "not-run"}
    try:
        boot.diagnostic_session_control(uds_mod.SESSION_TYPE.EXTENDED_DIAGNOSTIC)
        data = bytes(boot.read_memory_by_address(0, 0x10))
        sid23_probe = {"status": "accepted", "length": len(data)}
    except Exception as exc:
        sid23_probe = {"status": "rejected", "error": type(exc).__name__}

    boot = _make_uds_client(panda, uds_mod, route, timeout=0.5, response_pending_timeout=3.0)
    boot.diagnostic_session_control(uds_mod.SESSION_TYPE.DEFAULT)
    boot.diagnostic_session_control(uds_mod.SESSION_TYPE.EXTENDED_DIAGNOSTIC)
    boot.diagnostic_session_control(uds_mod.SESSION_TYPE.PROGRAMMING)
    return boot, boot_f181_hex, boot_f181_ascii, sid23_probe


def _enter_programming_bootloader(
    panda,
    uds_mod,
    route: RamExecRoute,
    *,
    expected_boot_f181_hex: str | None,
    settle_extended: float = 0.7,
    reappearance_timeout: float = 8.0,
):
    """Use the exact F33 field-proven application -> bootloader handoff.

    The retained 2026-08-26 acquisition succeeded through
    ``tsk.lib.programming.enter_programming_bootloader``. Do not maintain a second
    approximation here: that duplicate path produced authenticated uploads and accepted
    0x10F0 but no callback telemetry on the same F33. The field-proven helper re-enters
    DEFAULT while rediscovering the preserved route, confirms F181, and records the
    transition before the final boot programming ladder.
    """
    try:
        from tsk.lib.programming import enter_programming_bootloader, uds_client
    except ImportError as exc:
        raise RamExecError(
            "field-proven Toyota programming helper is unavailable on PYTHONPATH; use the complete self-contained field kit"
        ) from exc

    route_dict = _boot_route_dict(route)
    try:
        boot_route, handoff = enter_programming_bootloader(
            panda,
            route_dict,
            prepare_sessions=True,
            settle_extended=settle_extended,
            reappearance_timeout=reappearance_timeout,
        )
    except Exception as exc:
        raise RamExecError(f"field-proven programming handoff failed: {type(exc).__name__}: {exc}") from exc

    boot, boot_f181_hex, boot_f181_ascii, sid23_probe = _prepare_boot_client(
        panda, uds_mod, route, boot_route, expected_boot_f181_hex=expected_boot_f181_hex
    )
    return boot, boot_f181_hex, boot_f181_ascii, handoff, sid23_probe


def _prepare_direct_bootloader(
    panda,
    uds_mod,
    route: RamExecRoute,
    *,
    expected_boot_f181_hex: str,
    confirmed_boot_f181: tuple[str, str | None],
):
    """Prepare a bootloader that is already running, e.g. after an invalid app CRC.

    This is recovery-only. It does not attempt an application handoff and therefore
    requires the exact expected boot F181 on the preserved physical route.
    """
    boot_route = _boot_route_dict(route)
    boot, boot_f181_hex, boot_f181_ascii, sid23_probe = _prepare_boot_client(
        panda, uds_mod, route, boot_route, expected_boot_f181_hex=expected_boot_f181_hex,
        confirmed_boot_f181=confirmed_boot_f181,
    )
    handoff = {
        "direct_bootloader": True,
        "route": boot_route,
        "boot_f181_hex": boot_f181_hex,
    }
    return boot, boot_f181_hex, boot_f181_ascii, handoff, sid23_probe



def _security_access(uds_client, uds_mod, secret: bytes) -> bytes:
    seed_payload = b"\x00" * 16
    seed = uds_client.security_access(uds_mod.ACCESS_TYPE.REQUEST_SEED, data_record=seed_payload)
    intermediate = AES.new(secret, AES.MODE_ECB).decrypt(seed_payload)
    key = AES.new(intermediate, AES.MODE_ECB).encrypt(seed)
    uds_client.security_access(uds_mod.ACCESS_TYPE.SEND_KEY, key)
    return bytes(seed)


def _read_f181(uds_client, uds_mod) -> tuple[str | None, str | None]:
    try:
        payload = bytes(
            uds_client.read_data_by_identifier(
                uds_mod.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION
            )
        )
    except Exception:
        return None, None
    try:
        ascii_value = payload.rstrip(b"\x00").decode("ascii") or None
    except UnicodeDecodeError:
        ascii_value = None
    return payload.hex(), ascii_value


def _wait_for_f181_response(
    panda,
    uds_mod,
    route: RamExecRoute,
    *,
    timeout: float,
    expected_f181_hex: str | None = None,
) -> tuple[Any, str | None, str | None, int]:
    """Wait through an endpoint transition, optionally for one exact F181.

    A caught startup PROGRAMMING request is asynchronous from the host's point of
    view: the application may still answer F181 briefly after 10 02 has already
    been put on the bus.  When ``expected_f181_hex`` is supplied, keep polling
    through those stale application replies and return only after the exact boot
    identity appears (or the timeout expires).  Every retry uses a fresh ISO-TP
    transport so a timed-out transaction cannot poison the next probe.
    """
    deadline = time.monotonic() + timeout
    attempts = 0
    last_hex: str | None = None
    last_ascii: str | None = None
    client = _make_uds_client(panda, uds_mod, route, timeout=0.12, response_pending_timeout=0.25)
    while True:
        attempts += 1
        f181_hex, f181_ascii = _read_f181(client, uds_mod)
        if f181_hex is not None:
            last_hex, last_ascii = f181_hex, f181_ascii
            if expected_f181_hex is None or f181_hex.lower() == expected_f181_hex.lower():
                return client, f181_hex, f181_ascii, attempts
        if time.monotonic() >= deadline:
            return client, last_hex, last_ascii, attempts
        time.sleep(0.01)
        client = _make_uds_client(panda, uds_mod, route, timeout=0.12, response_pending_timeout=0.25)


def _upload_and_trigger(
    panda,
    uds_client,
    uds_mod,
    route: RamExecRoute,
    payload: bytes,
    *,
    geometry: RamExecGeometry = SIENNA_RAM_EXEC_GEOMETRY,
) -> None:
    geometry.validate()
    if len(payload) != geometry.load_size:
        raise RamExecError(f"RAM payload must be exactly 0x{geometry.load_size:X} bytes")

    did_203, mem_id, routine_magic = bootstrap_protocol_values(route)

    uds_client.write_data_by_identifier(0x203, did_203)
    uds_client.write_data_by_identifier(0x201, DID_201_KEY)
    uds_client.write_data_by_identifier(0x202, DID_202_IV)

    request_download = b"\x01\x46" + mem_id + b"\x00"
    request_download += struct.pack("!I", geometry.load_addr)
    request_download += struct.pack("!I", geometry.load_size)
    uds_client._uds_request(uds_mod.SERVICE_TYPE.REQUEST_DOWNLOAD, data=request_download)

    chunk_size = 0x400
    for index in range(geometry.load_size // chunk_size):
        start = index * chunk_size
        uds_client.transfer_data(index + 1, payload[start:start + chunk_size])
    uds_client.request_transfer_exit()

    verify_data = routine_magic + struct.pack("!I", geometry.load_addr) + struct.pack("!I", geometry.load_size)
    uds_client.routine_control(uds_mod.ROUTINE_CONTROL_TYPE.START, 0x10F0, verify_data)

    trigger_data = routine_magic + struct.pack("!I", TRIGGER_ADDR) + struct.pack("!I", TRIGGER_LENGTH)
    # Exact F33 acquisition clears Panda's host-facing RX ring immediately before
    # the one-shot multi-frame FF00 trigger. opendbc.car.isotp.isotp_send() treats
    # the first matching 0x7A9 frame as flow-control, so stale diagnostic traffic
    # can otherwise be mistaken for the current FC and corrupt trigger framing.
    # This changes no ECU state; it only discards already-received host backlog.
    panda.can_clear(0xFFFF)
    time.sleep(0.01)
    isotp_send = _import_isotp_send()
    isotp_send(
        panda,
        b"\x31\x01\xFF\x00" + trigger_data,
        TX_ADDR,
        bus=route.bus,
    )


def execute_ram_payload(
    payload: bytes,
    *,
    route: RamExecRoute,
    security_secret: bytes,
    timeout: float = 120.0,
    frame_handler: Callable[[bytes], bool] | None = None,
    expected_f181_hex: str | None = None,
    expected_boot_f181_hex: str | None = None,
    geometry: RamExecGeometry = SIENNA_RAM_EXEC_GEOMETRY,
    allow_direct_boot: bool = False,
    programming_already_requested: bool = False,
    allow_unidentified_app_handoff: bool = False,
    panda: Any | None = None,
) -> dict:
    phase_ns: dict[str, int] = {"started": time.monotonic_ns()}
    route.validate()
    geometry.validate()
    ensure_boardd_stopped()
    try:
        from panda import Panda
    except ImportError as exc:
        raise RamExecError("cannot import panda; execute on a comma/openpilot environment") from exc

    uds_mod = _import_uds()
    panda = Panda() if panda is None else panda
    phase_ns["panda_connected"] = time.monotonic_ns()
    panda.set_safety_mode(ELM327_SAFETY_MODE, route.elm327_param)
    phase_ns["safety_configured"] = time.monotonic_ns()
    if programming_already_requested and not allow_direct_boot:
        raise RamExecError("programming_already_requested requires allow_direct_boot")
    if programming_already_requested and expected_boot_f181_hex is None:
        raise RamExecError("programming_already_requested requires exact expected boot F181")

    identity_attempts = 1
    # A native startup catcher can hand us an application that has already seen
    # 10 02 but has not disappeared yet. In that case application F181 is a
    # transitional observation. Do not send a second PROGRAMMING request.
    # Wait only for the exact boot identity and never fall back through
    # the ordinary application -> boot handoff.
    if programming_already_requested:
        app, initial_f181_hex, initial_f181_ascii, identity_attempts = _wait_for_f181_response(
            panda, uds_mod, route, timeout=3.0, expected_f181_hex=expected_boot_f181_hex,
        )
    elif allow_direct_boot:
        app, initial_f181_hex, initial_f181_ascii, identity_attempts = _wait_for_f181_response(
            panda, uds_mod, route, timeout=3.0,
        )
    else:
        app = _make_uds_client(panda, uds_mod, route)
        try:
            app.diagnostic_session_control(uds_mod.SESSION_TYPE.DEFAULT)
        except Exception:
            # Match the exact F33 acquisition guard: a stale session-reset response is
            # non-authoritative; the identity read below is the required application gate.
            pass
        initial_f181_hex, initial_f181_ascii = _read_f181(app, uds_mod)
    phase_ns["initial_f181_complete"] = time.monotonic_ns()
    direct_bootloader = False
    if programming_already_requested:
        if initial_f181_hex is None or initial_f181_hex.lower() != expected_boot_f181_hex.lower():
            raise RamExecError(
                "caught PROGRAMMING transition did not reach the exact boot endpoint: "
                f"expected {expected_boot_f181_hex}, last observed {initial_f181_hex}"
            )
        direct_bootloader = True
        f181_hex, f181_ascii = None, None
        boot, boot_f181_hex, boot_f181_ascii, handoff, sid23_probe = _prepare_direct_bootloader(
            panda, uds_mod, route, expected_boot_f181_hex=expected_boot_f181_hex,
            confirmed_boot_f181=(initial_f181_hex, initial_f181_ascii),
        )
    elif expected_f181_hex is None or (
        initial_f181_hex is not None and initial_f181_hex.lower() == expected_f181_hex.lower()
    ) or (allow_unidentified_app_handoff and initial_f181_hex is None):
        f181_hex, f181_ascii = initial_f181_hex, initial_f181_ascii
        boot, boot_f181_hex, boot_f181_ascii, handoff, sid23_probe = _enter_programming_bootloader(
            panda, uds_mod, route, expected_boot_f181_hex=expected_boot_f181_hex
        )
    elif (
        allow_direct_boot
        and expected_boot_f181_hex is not None
        and initial_f181_hex is not None
        and initial_f181_hex.lower() == expected_boot_f181_hex.lower()
    ):
        # Recovery path for the unavoidable two-block persistent-write window: an
        # invalid application CRC may leave the ECU in bootloader after power loss.
        # Accept only the exact expected boot identity; never treat an arbitrary F181
        # mismatch as permission to upload a recovery payload.
        direct_bootloader = True
        f181_hex, f181_ascii = None, None
        boot, boot_f181_hex, boot_f181_ascii, handoff, sid23_probe = _prepare_direct_bootloader(
            panda, uds_mod, route, expected_boot_f181_hex=expected_boot_f181_hex,
            confirmed_boot_f181=(initial_f181_hex, initial_f181_ascii),
        )
    else:
        raise RamExecError(
            f"F181 mismatch before RAM upload: expected application {expected_f181_hex}, "
            f"observed {initial_f181_hex}"
        )
    phase_ns["boot_prepared"] = time.monotonic_ns()
    seed = _security_access(boot, uds_mod, security_secret)
    phase_ns["security_access_complete"] = time.monotonic_ns()
    _upload_and_trigger(panda, boot, uds_mod, route, payload, geometry=geometry)
    phase_ns["payload_trigger_complete"] = time.monotonic_ns()

    deadline = time.monotonic() + timeout
    frames = 0
    handler_done = False
    while time.monotonic() < deadline and not handler_done:
        for row in panda.can_recv():
            # Current Panda Python returns (address, data, bus). Older call sites in
            # this project also encountered wider tuples, so preserve the robust
            # first/last-two-field decode instead of imposing a tuple length of 4+.
            # The old length guard silently discarded *every* current Panda frame.
            if len(row) < 3:
                continue
            can_addr = int(row[0])
            can_data = bytes(row[-2])
            bus = int(row[-1])
            # On relay-correct Toyota B topologies the same physical EPS segment can
            # be visible through multiple Panda logical buses. Do not discard valid
            # raw payload telemetry solely because it arrived on the relay mate.
            if can_addr != RX_ADDR or len(can_data) < 8:
                continue
            if can_data == b"\x03\x7F\x31\x78\x00\x00\x00\x00":
                continue
            frames += 1
            if frame_handler is not None and frame_handler(can_data[:8]):
                handler_done = True
                break
        if not handler_done:
            time.sleep(0.002)
    phase_ns["complete"] = time.monotonic_ns()
    started_ns = phase_ns["started"]

    return {
        "route": asdict(route),
        "ram_exec_geometry": asdict(geometry),
        "f181_hex": f181_hex,
        "f181_ascii": f181_ascii,
        "initial_f181_hex": initial_f181_hex,
        "initial_f181_ascii": initial_f181_ascii,
        "initial_f181_attempts": identity_attempts,
        "direct_bootloader": direct_bootloader,
        "programming_already_requested": programming_already_requested,
        "unidentified_app_handoff_allowed": allow_unidentified_app_handoff,
        "boot_f181_hex": boot_f181_hex,
        "boot_f181_ascii": boot_f181_ascii,
        "programming_handoff": handoff,
        "boot_sid23_probe": sid23_probe,
        "security_access": {
            "seed_length": len(seed),
            "secret_value_recorded": False,
        },
        "telemetry_frames": frames,
        "handler_done": handler_done,
        "timed_out": not handler_done,
        "timing": {
            "phase_monotonic_ns": phase_ns,
            "phase_after_start_ms": {
                key: (value - started_ns) / 1e6 for key, value in phase_ns.items()
            },
        },
    }
