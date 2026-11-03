#!/usr/bin/env python3
"""Toyota EPS diagnostic route and Panda physical-routing helpers.

A Panda logical CAN bus number is not a complete route. In ELM327 safety mode,
logical bus 1 is MCU FDCAN2 and the safety parameter selects which physical path
FDCAN2 reaches:

  param 1/non-zero -> normal harness
  param 0          -> OBD-II mux

Normal-harness routing is therefore the default discovery state. OBD routing is
an explicit fallback. A discovered route records the parameter together with the
logical bus and UDS endpoint, and stateful operations must preserve that exact
route instead of rediscovering under a different mux state.
"""
from __future__ import annotations

import time

DIAGNOSTIC_ADDRESS_RANGE = range(0x700, 0x800)
ELM327_NORMAL_PARAM = 1
ELM327_OBD_PARAM = 0
DEFAULT_ROUTING_PARAMS = (ELM327_NORMAL_PARAM, ELM327_OBD_PARAM)
DEFAULT_BUS_ORDER = (1, 0, 2)  # prefer the distinct FDCAN2 normal-harness path before relay-side 0/2

# Firmware-verified on Sienna 8965B4512000. Keep calibration-scoped: other EPS
# variants may differ and must be observed rather than inferred.
SIENNA_PHYSICAL_ADDR = 0x7A1
SIENNA_FUNCTIONAL_ADDR = 0x777
SIENNA_RESPONSE_ADDR = 0x7A9


class AmbiguousDiagnosticRouteError(RuntimeError):
  """More than one EPS-identity responder was observed in one physical route state."""


def configure_elm327(panda, param: int = ELM327_NORMAL_PARAM) -> None:
  """Select ELM327 safety and persist the intended FDCAN2 physical mux state."""
  from opendbc.car.structs import CarParams
  panda.set_safety_mode(CarParams.SafetyModel.elm327, int(param))


def routing_semantic(param: int) -> str:
  return "normal-harness" if int(param) != ELM327_OBD_PARAM else "obd"


def parse_first_isotp(data: bytes) -> bytes | None:
  if not data:
    return None
  pci = data[0] >> 4
  if pci == 0x0:
    size = data[0] & 0x0F
    return bytes(data[1:1 + size])
  if pci == 0x1:
    return bytes(data[2:])
  return None


def matches_uds_response(body: bytes, request_sid: int) -> bool:
  if not body:
    return False
  if body[0] == 0x7F:
    return len(body) >= 2 and body[1] == request_sid
  return body[0] == ((request_sid + 0x40) & 0xFF)


def probe_response_route(panda, tx: int, bus: int, payload: bytes,
                         timeout: float = 0.3) -> dict | None:
  """Return the first matching ``{tx, rx, tx_bus, rx_bus, body, ms}`` route."""
  from opendbc.car.isotp import isotp_send

  try:
    panda.can_recv()
  except Exception:
    pass
  started = time.monotonic()
  try:
    isotp_send(panda, payload, tx, bus=bus)
  except Exception:
    return None

  deadline = started + timeout
  while time.monotonic() < deadline:
    try:
      frames = panda.can_recv()
    except Exception:
      return None
    for addr, *_rest, data, rx_bus in frames:
      if rx_bus >= 0x80:
        continue
      body = parse_first_isotp(bytes(data))
      if body is None or not matches_uds_response(body, payload[0]):
        continue
      return {
        "tx": int(tx),
        "rx": int(addr),
        "tx_bus": int(bus),
        "rx_bus": int(rx_bus),
        "body": body.hex(),
        "ms": int((time.monotonic() - started) * 1000),
      }
    time.sleep(0.001)
  return None


def discover_known_route(panda, tx: int, buses, payload: bytes = b"\x10\x01",
                         timeout: float = 0.3) -> dict | None:
  """Try a known request address on every candidate bus without assuming its RX ID."""
  for bus in buses:
    route = probe_response_route(panda, tx, bus, payload, timeout)
    if route is not None:
      route["source"] = "prior_address_hypothesis"
      return route
  return None


def read_f181(panda, route: dict, timeout: float = 0.2) -> bytes:
  """Read the full F181 value on a same-bus route using openpilot's ISO-TP client."""
  if route["tx_bus"] != route["rx_bus"]:
    return b""
  try:
    from opendbc.car.uds import UdsClient
    client = UdsClient(panda, route["tx"], route["rx"], route["tx_bus"],
                       timeout=timeout, response_pending_timeout=timeout)
    return bytes(client.read_data_by_identifier(0xF181))
  except Exception:
    return b""


def discover_eps_route(panda, buses, preferred_tx: int = SIENNA_PHYSICAL_ADDR,
                       addresses=None, preferred_timeout: float = 0.3,
                       scan_timeout: float = 0.05,
                       require_eps_identity: bool = False) -> dict | None:
  """Try the prior EPS address first, then scan and identify a Toyota 8965 responder.

  This function discovers only the logical UDS route under the Panda mux state the
  caller already selected. Prefer :func:`discover_eps_route_with_routing` for a new
  Panda takeover so the physical routing parameter is also part of the result.
  """
  addresses = DIAGNOSTIC_ADDRESS_RANGE if addresses is None else addresses
  preferred = discover_known_route(panda, preferred_tx, buses, timeout=preferred_timeout)
  fallback = None
  if preferred is not None:
    identity = read_f181(panda, preferred, timeout=max(preferred_timeout, 0.2))
    if identity:
      preferred["identity"] = identity.hex()
      if b"8965" in identity:
        preferred["source"] = "prior_address_f181_8965"
        return preferred
    if not require_eps_identity:
      return preferred
    fallback = preferred

  for tx in addresses:
    if tx == 0x7DF:
      continue
    for bus in buses:
      route = probe_response_route(panda, tx, bus, b"\x10\x01", scan_timeout)
      if route is None:
        continue
      route["source"] = "address_scan"
      if fallback is None:
        fallback = route
      identity = read_f181(panda, route, timeout=max(scan_timeout, 0.2))
      if not identity:
        continue
      route["identity"] = identity.hex()
      if b"8965" in identity:
        route["source"] = "address_scan_f181_8965"
        return route
  return None if require_eps_identity else fallback


def _preferred_eps_candidates(panda, tx: int, buses, timeout: float) -> list[dict]:
  """Identify every same-bus ``8965`` responder at one request address."""
  candidates: list[dict] = []
  for bus in buses:
    observed = probe_response_route(panda, tx, bus, b"\x10\x01", timeout)
    if observed is None:
      continue
    # With the intercept relay in physical pass-through, the first returned copy of a
    # response can be reported on another Panda controller. Use its observed response
    # arbitration ID, but confirm the typed F181 exchange on the actual transmit bus
    # before accepting a stateful same-bus route.
    route = dict(observed)
    route["rx_bus"] = int(bus)
    identity = read_f181(panda, route, timeout=max(timeout, 0.2))
    if not identity or b"8965" not in identity:
      continue
    route["identity"] = identity.hex()
    route["source"] = "prior_address_f181_8965_same_bus_confirmed"
    candidates.append(route)
  return candidates


def discover_eps_route_with_routing(panda, buses, preferred_tx: int = SIENNA_PHYSICAL_ADDR,
                                    routing_params=DEFAULT_ROUTING_PARAMS, addresses=None,
                                    preferred_timeout: float = 0.3,
                                    scan_timeout: float = 0.05) -> dict | None:
  """Discover EPS route across explicit Panda physical-routing states.

  Normal-harness (ELM param 1) is always attempted first by default. At the preferred
  EPS address, every candidate logical bus is F181-probed and discovery fails closed if
  more than one same-state ``8965`` responder exists. The returned route includes
  ``elm327_param`` and ``semantic_path`` and leaves the Panda in that exact state.
  Stateful callers should preserve it for the rest of the operation.
  """
  for param in routing_params:
    configure_elm327(panda, param)
    # Give the board a short settling interval after FDCAN2 mux changes and drain any
    # frames queued from the previous route before probing.
    time.sleep(0.03)
    try:
      panda.can_recv()
    except Exception:
      pass
    preferred = _preferred_eps_candidates(panda, preferred_tx, buses, preferred_timeout)
    identities = {r["identity"] for r in preferred}
    if len(identities) > 1:
      detail = ", ".join(f"b{r['tx_bus']}->b{r['rx_bus']} F181={r['identity']}" for r in preferred)
      raise AmbiguousDiagnosticRouteError(
        f"multiple distinct 8965 responders at 0x{preferred_tx:03x} under {routing_semantic(param)} routing: {detail}"
      )
    if preferred:
      route = preferred[0]
      if len(preferred) > 1:
        route["alternate_routes"] = [
          {"tx_bus": int(r["tx_bus"]), "rx_bus": int(r["rx_bus"]), "identity": r["identity"]}
          for r in preferred[1:]
        ]
    else:
      route = discover_eps_route(panda, buses, preferred_tx=preferred_tx,
                                 addresses=addresses, preferred_timeout=preferred_timeout,
                                 scan_timeout=scan_timeout, require_eps_identity=True)
    if route is None:
      continue
    route["elm327_param"] = int(param)
    route["semantic_path"] = routing_semantic(param)
    # Re-apply so remembered current_safety_param is unquestionably the selected
    # route even if a probe helper changed board state in the future.
    configure_elm327(panda, param)
    return route
  return None


def rediscover_route(panda, route: dict, buses=None, preferred_timeout: float = 0.4,
                     scan_timeout: float = 0.08) -> dict | None:
  """Rediscover an endpoint without changing its physical Panda routing state."""
  param = int(route.get("elm327_param", ELM327_NORMAL_PARAM))
  configure_elm327(panda, param)
  candidates = [route["tx_bus"]] if buses is None else buses
  found = discover_eps_route(panda, candidates, preferred_tx=route["tx"],
                             preferred_timeout=preferred_timeout,
                             scan_timeout=scan_timeout)
  if found is None:
    return None
  # The first raw copy may be reported on a relay-mirrored controller. Confirm a
  # complete typed F181 exchange on the preserved transmit bus before declaring the
  # endpoint alive after reset.
  found["rx_bus"] = int(route["tx_bus"])
  identity = read_f181(panda, found, timeout=max(preferred_timeout, 0.3))
  if not identity:
    return None
  found["identity"] = identity.hex()
  found["source"] = "preserved_route_f181_confirmed"
  found["elm327_param"] = param
  found["semantic_path"] = routing_semantic(param)
  return found


def route_fields(route: dict | None) -> dict:
  """Small JSON-safe route payload for result objects/evidence records."""
  if not route:
    return {"eps_bus": -1, "eps_rx_bus": -1, "eps_tx": "", "eps_rx": "",
            "elm327_param": -1, "semantic_path": ""}
  fields = {
    "eps_bus": int(route["tx_bus"]),
    "eps_rx_bus": int(route["rx_bus"]),
    "eps_tx": f"0x{int(route['tx']):03x}",
    "eps_rx": f"0x{int(route['rx']):03x}",
    "elm327_param": int(route.get("elm327_param", ELM327_NORMAL_PARAM)),
    "semantic_path": str(route.get("semantic_path", routing_semantic(route.get("elm327_param", ELM327_NORMAL_PARAM)))),
  }
  if route.get("alternate_routes"):
    fields["alternate_routes"] = list(route["alternate_routes"])
  return fields


def _mapping(value) -> dict:
  if value is None:
    return {}
  if isinstance(value, dict):
    return dict(value)
  if hasattr(value, "_asdict"):
    try:
      return dict(value._asdict())
    except Exception:
      pass
  try:
    return dict(value)
  except Exception:
    return {"value": str(value)}


def panda_health_snapshot(panda, bus: int) -> dict:
  """Best-effort Panda + CAN-controller telemetry around a route transition."""
  snapshot = {"bus": int(bus), "health": {}, "can_health": {}}
  try:
    snapshot["health"] = _mapping(panda.health())
  except Exception as e:
    snapshot["health_error"] = type(e).__name__
  try:
    snapshot["can_health"] = _mapping(panda.can_health(int(bus)))
  except Exception as e:
    snapshot["can_health_error"] = type(e).__name__
  return snapshot
