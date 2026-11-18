# a test route's window the UP plays: its first engagement after the route's panda was in the car's safety mode
from dataclasses import dataclass

from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.logreader import LogReader
from opendbc.car.tests.routes import CarTestRoute
from opendbc.car.tests.test_models import get_cached_segment
from opendbc.car.vin import VIN_UNKNOWN

DT = 0.01
# seconds of CAN before the panda's controls_allowed edge, openpilot is ready ~8 s after the first CAN, and after it
LEAD = 10.
ENGAGED = 5.
BUSES = 3
# before the route's panda is in a car's safety mode the recording is the old device's relay and forwarding, not the car's
NO_CAR_MODES = ('silent', 'noOutput', 'elm327', 'allOutput')

Frame = list[tuple[int, bytes, int]]


@dataclass
class Window:
  segment: int
  # segment seconds from its first CAN
  start: float
  frames: list[Frame]
  # the route's carParams for CarParamsCache, None when card wouldn't take its firmware
  cache: bytes | None
  # card can't tell the car from the route's firmware, it's given
  given: bool
  alpha_long: bool


def car_mode_samples(msgs) -> list[tuple[float, bool]]:
  # (time, controls_allowed) of the route's panda while in a car's safety mode
  samples = []
  for m in msgs:
    if m.which() == 'pandaStates' and len(m.pandaStates):
      state = m.pandaStates[0]
    elif m.which() == 'pandaStateDEPRECATED':
      state = m.pandaStateDEPRECATED
    else:
      continue
    if str(state.safetyModel) not in NO_CAR_MODES:
      samples.append((m.logMonoTime * 1e-9, state.controlsAllowed))
  return samples


def find_edge(samples: list[tuple[float, bool]], can_end: float) -> float | None:
  # the first controls_allowed rising edge with LEAD s in the car's safety mode before it and ENGAGED s allowed after it
  for (_, before), (t, allowed) in zip(samples, samples[1:], strict=False):
    if before or not allowed or t - LEAD < samples[0][0] or t + ENGAGED > can_end:
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


def make_window(route: CarTestRoute, segment: int, msgs, start: float) -> Window:
  can_start = next(m.logMonoTime * 1e-9 for m in msgs if m.which() == 'can')
  logged = next((m.carParams for m in reversed(msgs) if m.which() == 'carParams'), None)
  cache = None
  if logged is not None and len(logged.carFw) and logged.carVin != VIN_UNKNOWN:
    cache = logged.as_builder().to_bytes()
  given = cache is None or match_fw_to_car(logged.carFw, logged.carVin, log=False)[1] != {route.car_model}
  alpha_long = logged is not None and logged.openpilotLongitudinalControl
  return Window(segment, start - can_start, bin_frames(msgs, start, LEAD + ENGAGED), cache, given, alpha_long)


def find_window(route: CarTestRoute) -> Window | None:
  # in the segments test_models uses
  for segment in [route.segment] if route.segment is not None else [2, 1, 0]:
    try:
      msgs = list(LogReader(str(get_cached_segment(route.route, segment)), only_union_types=True, sort_by_time=True))
    except OSError:
      continue
    samples = car_mode_samples(msgs)
    can = [m.logMonoTime * 1e-9 for m in msgs if m.which() == 'can']
    edge = find_edge(samples, can[-1]) if samples and can else None
    if edge is not None:
      return make_window(route, segment, msgs, edge - LEAD)
  return None
