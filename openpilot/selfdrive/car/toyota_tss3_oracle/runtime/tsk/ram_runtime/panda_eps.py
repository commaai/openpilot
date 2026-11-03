"""Target-neutral Panda-side EPS diagnostics shared by maintained RAM runtimes.

Safety-mode selection and F181 identity polling only. Vehicle-state guards
(NRTD/READY/Park/stationary) are deliberately not here: the power-up bringup
path loads during EPS startup and does not require them.
"""
from __future__ import annotations

import time
from typing import Any

from tsk.common.ram_exec import (
    ELM327_SAFETY_MODE,
    _import_uds,
    _make_uds_client,
    _read_f181,
)


class PandaEpsError(RuntimeError):
    pass


def set_alloutput_mode(panda: Any) -> None:
    try:
        from opendbc.car import structs
    except ImportError as exc:
        raise PandaEpsError("opendbc.car.structs unavailable") from exc
    panda.set_safety_mode(structs.CarParams.SafetyModel.allOutput, 1)


def try_f181(
    *, timeout: float = 0.16, route, panda: Any | None = None,
    expected_f181_hex: str | None = None,
) -> dict[str, Any]:
    """Attempt one bounded F181 read; report identity without raising."""
    try:
        from panda import Panda
    except ImportError as exc:
        return {"ok": False, "error": f"ImportError: {exc}"}
    uds_mod = _import_uds()
    owns_panda = panda is None
    panda = Panda() if panda is None else panda
    if owns_panda:
        panda.set_safety_mode(ELM327_SAFETY_MODE, route.elm327_param)
    client = _make_uds_client(panda, uds_mod, route, timeout=timeout, response_pending_timeout=0.5)
    try:
        f181_hex, f181_ascii = _read_f181(client, uds_mod)
        return {
            "ok": (
                expected_f181_hex is not None
                and f181_hex is not None
                and f181_hex.lower() == expected_f181_hex.lower()
            ),
            "f181_hex": f181_hex,
            "f181_ascii": f181_ascii,
        }
    except Exception as exc:  # noqa: BLE001 - evidence capture
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if owns_panda:
            panda.close()


def wait_for_f181(
    *, timeout: float, route, panda: Any | None = None,
    expected_f181_hex: str,
) -> dict[str, Any]:
    """Poll F181 until the exact expected application identity answers."""
    deadline = time.monotonic() + timeout
    attempts: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        row = try_f181(route=route, panda=panda, expected_f181_hex=expected_f181_hex)
        f181_hex = row.get("f181_hex")
        row["ok"] = isinstance(f181_hex, str) and f181_hex.lower() == expected_f181_hex.lower()
        row["at_monotonic"] = time.monotonic()
        attempts.append(row)
        if row["ok"]:
            return {"ok": True, "attempts": attempts, "first_success": row}
        time.sleep(0.02)
    return {"ok": False, "attempts": attempts}
