# a test route's window the UP plays: its first engagement after the route's panda was in the car's safety mode
import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from opendbc.car.car_helpers import interfaces
from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.structs import car
from opendbc.car.tests.routes import CarTestRoute
from opendbc.car.tests.test_models import DOWNLOAD_CACHE_ROOT, get_cached_segment
from opendbc.car.vin import VIN_UNKNOWN
from openpilot.tools.lib.logreader import LogReader

DT = 0.01
# seconds of CAN before the panda's controls_allowed edge, openpilot is ready ~8 s after the first CAN, and after it
LEAD = 10.
ENGAGED = 3.
BUSES = 3
# before the route's panda is in a car's safety mode the recording is the old device's relay and forwarding, not the car's
NO_CAR_MODES = ('silent', 'noOutput', 'elm327', 'allOutput')

Frame = list[tuple[int, bytes, int]]
# parsing a route's segments takes seconds, what they give is kept, per version of this file. its own dir, so CI can cache it without the logs
WINDOW_CACHE = Path(os.environ.get("SAFETY_MODES_CACHE", DOWNLOAD_CACHE_ROOT))
CACHE_KEY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


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


def cruise_times(msgs, pcm_cruise: bool) -> tuple[list[float], list[float]]:
  # when the car's cruise was on, and when resume was pressed in a car with openpilot's longitudinal, which blocks resume
  # until cruise was set once, never on a freshly started DUT
  on, resumes = [], []
  for m in msgs:
    if m.which() == 'carState':
      if m.carState.cruiseState.enabled:
        on.append(m.logMonoTime * 1e-9)
      if not pcm_cruise and any(str(b.type) in ('accelCruise', 'resumeCruise') for b in m.carState.buttonEvents):
        resumes.append(m.logMonoTime * 1e-9)
  return on, resumes


def find_edge(samples: list[tuple[float, bool]], can_end: float, cruise_on: list[float] = (), resumes: list[float] = ()) -> float | None:
  # the first controls_allowed rising edge with LEAD s in the car's safety mode before it with cruise off until 1 s before,
  # not engaged by a resume, and ENGAGED s allowed after it
  for (_, before), (t, allowed) in zip(samples, samples[1:], strict=False):
    if before or not allowed or t - LEAD < samples[0][0] or t + ENGAGED > can_end:
      continue
    if any(t - LEAD <= c < t - 1. for c in cruise_on) or any(t - 1. <= r <= t for r in resumes):
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


def make_window(route: CarTestRoute, segment: int, start: float, frames: list[Frame], logged_bytes: bytes | None) -> Window | None:
  if logged_bytes is None:
    return Window(segment, start, frames, None, True, False)
  with car.CarParams.from_bytes(logged_bytes) as logged:
    has_fw = len(logged.carFw) > 0 and logged.carVin != VIN_UNKNOWN
    given = not has_fw or match_fw_to_car(logged.carFw, logged.carVin, log=False)[1] != {route.car_model}
    # the car's CAN depends on who did longitudinal, e.g. its radar is off for openpilot's. the DUT must get the same
    fingerprint: dict[int, dict[int, int]] = {bus: {} for bus in range(8)}
    for frame in frames:
      for addr, dat, bus in frame:
        fingerprint[bus][addr] = len(dat)
    car_fw = list(logged.carFw) if has_fw else []
    CP = interfaces[route.car_model].get_params(route.car_model, fingerprint, car_fw, logged.openpilotLongitudinalControl, False, docs=False)
    if CP.openpilotLongitudinalControl != logged.openpilotLongitudinalControl:
      return None
    return Window(segment, start, frames, logged_bytes if has_fw else None, given, logged.openpilotLongitudinalControl)


def parse_window(route: CarTestRoute) -> tuple[int, float, list[Frame], bytes | None] | None:
  # in the segments test_models uses
  for segment in [route.segment] if route.segment is not None else [2, 1, 0]:
    try:
      msgs = list(LogReader(str(get_cached_segment(route.route, segment)), sort_by_time=True))
    except OSError:
      continue
    samples = car_mode_samples(msgs)
    can = [m.logMonoTime * 1e-9 for m in msgs if m.which() == 'can']
    logged = next((m.carParams for m in reversed(msgs) if m.which() == 'carParams'), None)
    cruise_on, resumes = cruise_times(msgs, logged is None or logged.pcmCruise)
    edge = find_edge(samples, can[-1], cruise_on, resumes) if samples and can else None
    if edge is not None:
      start = edge - LEAD
      logged_bytes = None if logged is None else logged.as_builder().to_bytes()
      return segment, start - can[0], bin_frames(msgs, start, LEAD + ENGAGED), logged_bytes
  return None


def find_window(route: CarTestRoute) -> Window | None:
  path = WINDOW_CACHE / f"safety_modes_{CACHE_KEY}_{route.route.replace('/', '|')}.pkl"
  if path.exists():
    parsed = pickle.loads(path.read_bytes())
  else:
    parsed = parse_window(route)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(parsed))
  return None if parsed is None else make_window(route, *parsed)
