#!/usr/bin/env python3
"""Catch TSS3 EPS application startup and enter stock bootloader PROGRAMMING.

This is the minimum race, first proven on the Camry F33 EPS: it does not read
application identity before the race, perform SecurityAccess, upload RAM,
reset the ECU, or write persistent storage. It repeatedly offers EXTENDED
while the EPS is asleep, waits for the exact completed positive response shared
by every registered target, then sends exactly one 10 02. The accepted frame is
`06 50 03 00 32 01 F4 00`; it is also recorded in the result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from tsk.common.ram_exec import (
    ELM327_SAFETY_MODE,
    _import_uds,
    _make_uds_client,
    _read_f181,
    ensure_boardd_stopped,
    explicit_route,
)

TX_ADDR = 0x7A1
RX_ADDR = 0x7A9
# Panda bus 0: the shared repinned TSS3 topology (vehicle network and EPS
# diagnostics), identical on every maintained TSS3 vehicle.
BUS = 0
# Blank-bootloader F181 shared by every registered TSS3 EPS target
# (target_profiles.BOOT_F181_HEX).
EXPECTED_BOOT_F181_HEX = "02" + "21" * 32
EXTENDED_FRAME = bytes.fromhex("0210030000000000")
PROGRAMMING_FRAME = bytes.fromhex("0210020000000000")
# Exact completed positive EXTENDED response recovered from every registered
# Camry, Crown, and Corolla CodeFlash image.
POSITIVE_EXTENDED_FRAME = bytes.fromhex("065003003201f400")
EXTENDED_PERIOD_SECONDS = 0.020
DEFAULT_RACE_TIMEOUT_SECONDS = 35.0
BOOT_IDENTITY_TIMEOUT_SECONDS = 3.0


class StartupProgrammingError(RuntimeError):
    pass


def _event(now_ns: int, start_ns: int, *, address: int, bus: int, data: bytes) -> dict[str, Any]:
    return {
        "t_ms": (now_ns - start_ns) / 1e6,
        "address": f"0x{address:03X}",
        "bus": bus,
        "data_hex": data.hex(),
    }


def _read_boot_identity(
    panda: Any, *, start_ns: int, timeout: float, route: Any,
) -> tuple[str, list[dict[str, Any]]]:
    uds_mod = _import_uds()
    deadline = time.monotonic() + timeout
    samples: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        client = _make_uds_client(panda, uds_mod, route, timeout=0.12, response_pending_timeout=0.25)
        sample: dict[str, Any] = {"t_ms": (time.monotonic_ns() - start_ns) / 1e6}
        f181_hex, f181_ascii = _read_f181(client, uds_mod)
        sample.update({"f181_hex": f181_hex, "f181_ascii": f181_ascii})
        samples.append(sample)
        if f181_hex is not None and f181_hex.lower() == EXPECTED_BOOT_F181_HEX:
            return f181_hex.lower(), samples
        time.sleep(0.04)
    observed = [row["f181_hex"] for row in samples if row.get("f181_hex")]
    raise StartupProgrammingError(f"bootloader F181 did not appear; observed {observed[-3:]}")


def race_to_bootloader(
    *,
    timeout: float = DEFAULT_RACE_TIMEOUT_SECONDS,
    cancel_requested: Callable[[], bool] | None = None,
    bus: int = BUS,
    target: str = "",
) -> dict[str, Any]:
    """Enter the EPS bootloader from a pre-armed OFF-state worker.

    ``bus`` defaults to the shared repinned topology (Panda bus 0); ``target``
    is recorded provenance. First proven on the Camry EPS; the mechanism and
    bus topology are shared across maintained TSS3 targets.

    ``cancel_requested`` is cooperative by design.  It is honored while the
    catcher is still waiting and once more immediately before the one-way 10 02
    transition.  After that final check the caller must let the handoff finish;
    killing the worker after PROGRAMMING has been transmitted can strand the EPS
    between application and bootloader ownership.
    """
    route = explicit_route(bus=bus, elm327_param=1, uds_variant="old", cpu_index=0)
    ensure_boardd_stopped()
    try:
        from panda import Panda
    except ImportError as exc:
        raise StartupProgrammingError("panda module unavailable") from exc

    panda = Panda()
    start_ns = time.monotonic_ns()
    tx_rows: list[dict[str, Any]] = []
    rx_rows: list[dict[str, Any]] = []
    try:
        if hasattr(panda, "set_power_save"):
            panda.set_power_save(0)
        panda.set_safety_mode(ELM327_SAFETY_MODE, route.elm327_param)

        # Discard any already-queued frames exactly once, without delaying the
        # ignition-triggered race. A timed pre-arm drain is only useful when the
        # worker is started before POWER; in auto-arm it burns the startup window.
        panda.can_recv()
        armed_ns = time.monotonic_ns()
        deadline = time.monotonic() + timeout
        next_extended = time.monotonic()
        positive_extended_ns: int | None = None
        positive_extended_event: dict[str, Any] | None = None

        while time.monotonic() < deadline and positive_extended_ns is None:
            if cancel_requested is not None and cancel_requested():
                raise StartupProgrammingError("startup race cancelled before PROGRAMMING")
            now = time.monotonic()
            if now >= next_extended:
                sent_ns = time.monotonic_ns()
                panda.can_send(TX_ADDR, EXTENDED_FRAME, bus)
                tx_rows.append({"t_ms": (sent_ns - start_ns) / 1e6, "kind": "10_03"})
                next_extended = now + EXTENDED_PERIOD_SECONDS

            rows = panda.can_recv()
            recv_ns = time.monotonic_ns()
            for row in rows:
                if len(row) < 3:
                    continue
                address, data, frame_bus = int(row[0]), bytes(row[-2]), int(row[-1])
                if address in (0x030, 0x101, 0x51E, RX_ADDR) and frame_bus < 128:
                    rx_rows.append(_event(recv_ns, start_ns, address=address, bus=frame_bus, data=data))
                if address == RX_ADDR and frame_bus == bus and data == POSITIVE_EXTENDED_FRAME:
                    positive_extended_ns = recv_ns
                    positive_extended_event = _event(recv_ns, start_ns, address=address, bus=frame_bus, data=data)
                    break
            if positive_extended_ns is None:
                time.sleep(0.0005)

        if positive_extended_ns is None or positive_extended_event is None:
            raise StartupProgrammingError("no exact 50 03 response before startup race timeout")

        # This is the point of no return for cooperative cancellation.  A cancel
        # request that races with this check is intentionally resolved in favor
        # of completing bringup: the caller never force-kills us after 10 02.
        if cancel_requested is not None and cancel_requested():
            raise StartupProgrammingError("startup race cancelled before PROGRAMMING")
        programming_ns = time.monotonic_ns()
        panda.can_send(TX_ADDR, PROGRAMMING_FRAME, bus)
        tx_rows.append({"t_ms": (programming_ns - start_ns) / 1e6, "kind": "10_02"})

        # No further SessionControl traffic. Give the asynchronous handoff a short
        # quiet interval, then identify only the exact known bootloader endpoint.
        quiet_deadline = time.monotonic() + 0.080
        while time.monotonic() < quiet_deadline:
            rows = panda.can_recv()
            recv_ns = time.monotonic_ns()
            for row in rows:
                if len(row) < 3:
                    continue
                address, data, frame_bus = int(row[0]), bytes(row[-2]), int(row[-1])
                if address in (0x030, 0x101, 0x51E, RX_ADDR) and frame_bus < 128:
                    rx_rows.append(_event(recv_ns, start_ns, address=address, bus=frame_bus, data=data))
            time.sleep(0.0005)

        boot_f181_hex, f181_samples = _read_boot_identity(
            panda, start_ns=start_ns, timeout=BOOT_IDENTITY_TIMEOUT_SECONDS, route=route,
        )
        return {
            "schema": "tss3-startup-programming-v1",
            "start_monotonic_ns": start_ns,
            "armed_monotonic_ns": armed_ns,
            "positive_extended_monotonic_ns": positive_extended_ns,
            "programming_tx_monotonic_ns": programming_ns,
            "target": target,
            "route": {"tx": f"0x{TX_ADDR:03X}", "rx": f"0x{RX_ADDR:03X}", "bus": bus, "elm327_param": route.elm327_param},
            "armed_t_ms": (armed_ns - start_ns) / 1e6,
            "positive_extended": positive_extended_event,
            "programming_tx_t_ms": (programming_ns - start_ns) / 1e6,
            "programming_after_50_03_ms": (programming_ns - positive_extended_ns) / 1e6,
            "boot_f181_hex": boot_f181_hex,
            "f181_samples": f181_samples,
            "tx": tx_rows,
            "rx": rx_rows,
            "security_access_used": False,
            "payload_upload": False,
            "ecu_reset": False,
            "persistent_flash_writes": False,
            "verdict": "bootloader_caught_during_normal_startup",
        }
    finally:
        try:
            panda.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=float, default=DEFAULT_RACE_TIMEOUT_SECONDS)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--bus", type=int, default=BUS, help="EPS diagnostic bus from kit metadata")
    ap.add_argument("--target", default="", help="kit target name recorded as provenance")
    args = ap.parse_args()
    if not args.execute:
        raise StartupProgrammingError("startup race requires --execute")
    result = race_to_bootloader(timeout=args.timeout, bus=args.bus, target=args.target)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
