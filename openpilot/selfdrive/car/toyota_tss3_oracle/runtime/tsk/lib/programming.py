#!/usr/bin/env python3
"""Shared Toyota EPS application -> bootloader programming handoff.

On the analyzed Sienna application, ``10 02`` is asynchronous: a positive final
response is not required before shutdown/reset overtakes the application endpoint.
A timeout after response-pending is therefore not sufficient evidence of failure.
The reliable discriminator is whether the diagnostic endpoint reappears on the same
Panda physical route, with Panda/CAN health captured around the transition.
"""
from __future__ import annotations

import time

from tsk.lib.diagnostic_route import panda_health_snapshot, rediscover_route, route_fields


class ProgrammingHandoffError(Exception):
  def __init__(self, message: str, *, nrc: int | None = None, telemetry: dict | None = None):
    super().__init__(message)
    self.nrc = nrc
    self.telemetry = telemetry or {}


def uds_client(panda, route: dict, *, timeout: float = 0.3,
               response_pending_timeout: float = 3.0):
  from opendbc.car.uds import UdsClient
  return UdsClient(panda, route["tx"], route["rx"], route["tx_bus"],
                   timeout=timeout, response_pending_timeout=response_pending_timeout)


def enter_programming_bootloader(panda, route: dict, *, prepare_sessions: bool = True,
                                 settle_extended: float = 0.0,
                                 reappearance_timeout: float = 6.0) -> tuple[dict, dict]:
  """Request PROGRAMMING and require the endpoint to reappear on the same route.

  Returns ``(boot_route, telemetry)``. ``MessageTimeoutError`` from the initial
  ``10 02`` is recorded but not treated as failure. A negative response is a real
  rejection and is raised with its NRC. Physical routing never falls back to another
  ELM parameter during this stateful transition.
  """
  from opendbc.car.uds import SESSION_TYPE, InvalidServiceIdError, MessageTimeoutError, NegativeResponseError

  bus = int(route["tx_bus"])
  telemetry = {
    "route_before": route_fields(route),
    "health_before": panda_health_snapshot(panda, bus),
    "programming_response_timeout": False,
  }

  client = uds_client(panda, route, timeout=0.3, response_pending_timeout=5.5)
  if prepare_sessions:
    try:
      client.diagnostic_session_control(SESSION_TYPE.DEFAULT)
      client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
      if settle_extended:
        time.sleep(settle_extended)
    except (InvalidServiceIdError, MessageTimeoutError, NegativeResponseError) as e:
      telemetry["prepare_error"] = type(e).__name__
      raise ProgrammingHandoffError("Could not establish DEFAULT -> EXTENDED before PROGRAMMING",
                                    telemetry=telemetry) from e

  try:
    client.diagnostic_session_control(SESSION_TYPE.PROGRAMMING)
  except MessageTimeoutError:
    telemetry["programming_response_timeout"] = True
  except NegativeResponseError as e:
    telemetry["programming_nrc"] = int(e.error_code)
    telemetry["health_after_request"] = panda_health_snapshot(panda, bus)
    raise ProgrammingHandoffError(f"PROGRAMMING rejected with NRC 0x{e.error_code:02x}",
                                  nrc=int(e.error_code), telemetry=telemetry) from e
  except InvalidServiceIdError as e:
    telemetry["health_after_request"] = panda_health_snapshot(panda, bus)
    raise ProgrammingHandoffError("PROGRAMMING returned an invalid response", telemetry=telemetry) from e

  telemetry["health_after_request"] = panda_health_snapshot(panda, bus)

  deadline = time.monotonic() + reappearance_timeout
  while time.monotonic() < deadline:
    found = rediscover_route(panda, route, buses=[bus], preferred_timeout=0.35, scan_timeout=0.1)
    if found is not None:
      telemetry["route_after"] = route_fields(found)
      telemetry["health_after_reappearance"] = panda_health_snapshot(panda, bus)
      return found, telemetry
    time.sleep(0.1)

  telemetry["health_after_reappearance"] = panda_health_snapshot(panda, bus)
  raise ProgrammingHandoffError("EPS did not reappear on the preserved route after PROGRAMMING",
                                telemetry=telemetry)
