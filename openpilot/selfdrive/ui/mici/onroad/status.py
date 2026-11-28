"""
Engaged and override can flap every frame (gas taps, steer override on driver torque near the threshold),
so anything that looks different between them fades with a StatusFade instead of switching on ui_state.status.
Override holds until engaged for OVERRIDE_HOLD, so flapping shows a steady override.
"""
import numpy as np
import raylib_nb as rl
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app

OVERRIDE_HOLD = 0.3
STATUSES = tuple(UIStatus)
GearShifter = car.CarState.GearShifter


def lerp_color(a: rl.Color, b: rl.Color, t: float) -> rl.Color:
  return rl.Color(*(round(x + (y - x) * t) for x, y in zip((a.r, a.g, a.b, a.a), (b.r, b.g, b.b, b.a), strict=True)))


class StatusFade:
  """ui_state.status as weights, faded with rc and stepped once per frame"""
  def __init__(self, rc: float):
    self._weights = FirstOrderFilter(np.zeros(len(STATUSES)), rc, 1 / gui_app.target_fps, initialized=False)
    self._frame = -1
    self._override_time = float('-inf')

  def _update(self) -> np.ndarray:
    if ui_state.sm.frame != self._frame:
      self._frame = ui_state.sm.frame
      now = rl.get_time()
      if ui_state.status == UIStatus.OVERRIDE:
        self._override_time = now
      held = ui_state.status == UIStatus.ENGAGED and now - self._override_time < OVERRIDE_HOLD
      status = UIStatus.OVERRIDE if held else ui_state.status
      self._weights.update(np.array([status == s for s in STATUSES], dtype=float))
    return self._weights.x

  def weight(self, status: UIStatus) -> float:
    return float(self._update()[STATUSES.index(status)])

  def color(self, colors: dict[UIStatus, rl.Color]) -> rl.Color:
    rgba = sum(w * np.array([c.r, c.g, c.b, c.a], dtype=float) for w, c in zip(self._update(), (colors[s] for s in STATUSES), strict=True))
    return rl.Color(*np.round(rgba).astype(int).tolist())


def openpilot_available() -> bool:
  """Whether openpilot could drive right now. The path and lead bars only show while it could.
  Unlike selfdriveState.engageable, what the driver is doing (braking, steering, cruise buttons) doesn't count."""
  sm, CP = ui_state.sm, ui_state.CP
  if sm['selfdriveState'].enabled:
    return True
  if not ui_state.started or CP is None or CP.passive:  # passive is dashcam mode
    return False

  cs = sm['carState']
  unavailable = (
    sm['extrinsicsCalibration'].calStatus != log.ExtrinsicsCalibration.Status.calibrated,
    cs.gearShifter in (GearShifter.park, GearShifter.neutral, GearShifter.reverse),
    cs.parkingBrake,
    cs.accFaulted,
    cs.steerFaultPermanent,
    cs.espDisabled,
    cs.invalidLkasSetting,
  )
  return not any(unavailable)
