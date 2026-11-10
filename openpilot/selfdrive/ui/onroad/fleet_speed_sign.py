import raylib_nb as rl
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

FleetSource = log.LongitudinalPlan.LongitudinalPlanSource.fleetSpeed
ACTIVE_BORDER = rl.Color(255, 149, 0, 255)


def fleet_speed_state():
  # (speed in display units, map is limiting, gap-filled) or None when the map has nothing here
  sm = ui_state.sm
  if sm.recv_frame['longitudinalPlan'] < ui_state.started_frame:
    return None
  plan = sm['longitudinalPlan']
  active = plan.longitudinalPlanSource == FleetSource
  v = plan.fleetSpeedTarget if active and plan.fleetSpeedTarget > 0 else plan.fleetSpeed
  if v <= 0:
    return None
  return round(v * (CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH)), active, plan.fleetSpeedGapFilled


def draw_fleet_speed_sign(x: float, y: float, w: float, h: float) -> None:
  state = fleet_speed_state()
  if state is None:
    return
  value, active, gap_filled = state
  alpha = 140 if gap_filled else 255
  ink = rl.Color(0, 0, 0, alpha)
  rect = rl.Rectangle(x, y, w, h)
  rl.draw_rectangle_rounded(rect, 0.18, 10, rl.Color(255, 255, 255, alpha))
  rl.draw_rectangle_rounded_lines_ex(rect, 0.18, 10, max(4.0, w * 0.06), ACTIVE_BORDER if active else ink)

  label_font, number_font = gui_app.font(FontWeight.SEMI_BOLD), gui_app.font(FontWeight.BOLD)
  label, label_size = "FLEET", h * 0.17
  label_w = measure_text_cached(label_font, label, int(label_size)).x
  rl.draw_text_ex(label_font, label, rl.Vector2(x + (w - label_w) / 2, y + h * 0.12), label_size, 0, ink)
  text, number_size = str(value), h * (0.5 if value < 100 else 0.4)
  text_size = measure_text_cached(number_font, text, int(number_size))
  rl.draw_text_ex(number_font, text, rl.Vector2(x + (w - text_size.x) / 2, y + h * 0.36), number_size, 0, ink)
