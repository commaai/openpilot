# a test route's window the UP plays: its first engagement, or its first 30 s when it has none
from dataclasses import dataclass

from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.logreader import LogReader
from opendbc.car.tests.routes import CarTestRoute
from opendbc.car.tests.test_models import get_cached_segment
from opendbc.car.vin import VIN_UNKNOWN

DT = 0.01
# seconds of CAN before the panda's controls_allowed edge, and of controls_allowed after it
LEAD = 15.
ENGAGED = 10.
NO_EDGE = 30.
BUSES = 3

Frame = list[tuple[int, bytes, int]]


@dataclass
class Window:
  segment: int
  # segment seconds from its first CAN
  start: float
  seconds: float
  edge: bool
  frames: list[Frame]
  # the route's carParams for CarParamsCache, None when card wouldn't take its firmware
  cache: bytes | None
  # card can't tell the car from the route's firmware, it's given
  given: bool
  alpha_long: bool


def segments_to_try(route: CarTestRoute) -> list[int]:
  # test_models' order, then the route's other segments
  first = [route.segment] if route.segment is not None else [2, 1, 0]
  return first + [s for s in range(10) if s not in first]


def allowed_samples(msgs) -> list[tuple[float, bool]]:
  samples = []
  for m in msgs:
    if m.which() == 'pandaStates' and len(m.pandaStates):
      samples.append((m.logMonoTime * 1e-9, m.pandaStates[0].controlsAllowed))
    elif m.which() == 'pandaStateDEPRECATED':
      samples.append((m.logMonoTime * 1e-9, m.pandaStateDEPRECATED.controlsAllowed))
  return samples


def find_edge(samples: list[tuple[float, bool]], can_start: float, can_end: float) -> float | None:
  for (_, before), (t, allowed) in zip(samples, samples[1:], strict=False):
    if before or not allowed or t - LEAD < can_start or t + ENGAGED > can_end:
      continue
    if all(a for s, a in samples if t <= s <= t + ENGAGED):
      return t
  return None


def bin_frames(msgs, start: float, seconds: float) -> list[Frame]:
  frames: list[Frame] = [[] for _ in range(round(seconds / DT))]
  for m in msgs:
    if m.which() != 'can':
      continue
    i = int((m.logMonoTime * 1e-9 - start) / DT)
    if 0 <= i < len(frames):
      frames[i] += [(c.address, bytes(c.dat), c.src) for c in m.can if c.src < BUSES]
  return frames


def load_segment(route: CarTestRoute, segment: int) -> list:
  return list(LogReader(str(get_cached_segment(route.route, segment)), only_union_types=True, sort_by_time=True))


def make_window(route: CarTestRoute, segment: int, msgs, start: float, seconds: float, edge: bool) -> Window:
  can_start = next(m.logMonoTime * 1e-9 for m in msgs if m.which() == 'can')
  logged = next((m.carParams for m in reversed(msgs) if m.which() == 'carParams'), None)
  cache = None
  if logged is not None and len(logged.carFw) and logged.carVin != VIN_UNKNOWN:
    cache = logged.as_builder().to_bytes()
  given = cache is None or match_fw_to_car(logged.carFw, logged.carVin, log=False)[1] != {route.car_model}
  alpha_long = logged is not None and logged.openpilotLongitudinalControl
  return Window(segment, start - can_start, seconds, edge, bin_frames(msgs, start, seconds), cache, given, alpha_long)


def find_window(route: CarTestRoute) -> Window:
  fallback = None
  for segment in segments_to_try(route):
    try:
      msgs = load_segment(route, segment)
    except OSError:
      continue
    can = [m.logMonoTime * 1e-9 for m in msgs if m.which() == 'can']
    if not can:
      continue
    if fallback is None:
      fallback = (segment, msgs, can[0])
    edge = find_edge(allowed_samples(msgs), can[0], can[-1])
    if edge is not None:
      return make_window(route, segment, msgs, edge - LEAD, LEAD + ENGAGED, True)
  if fallback is None:
    raise FileNotFoundError(f"no segment of {route.route} with CAN")
  segment, msgs, can_start = fallback
  return make_window(route, segment, msgs, can_start, NO_EDGE, False)
