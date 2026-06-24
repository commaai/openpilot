import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label

# onroad debug overlay: live control numbers (lat accel, curvature, ...) on the driving view
# gated by the OnroadDebugOverlay param, mici only

FONT_SIZE = 14
PAD = 8
BOX_W = 196
BOX_H = 128


class DebugOverlay(Widget):
  def __init__(self):
    super().__init__()
    self._lines: list[str] = []

  def _update_state(self):
    if not ui_state.onroad_debug_overlay or not ui_state.started:
      self._lines = []
      return

    sm = ui_state.sm
    # skip until each service has arrived since going onroad, else we'd show stale numbers as live
    if any(sm.recv_frame[s] < ui_state.started_frame for s in ('carState', 'controlsState', 'carControl')):
      self._lines = []
      return

    cs = sm['carState']
    controls_state = sm['controlsState']
    car_control = sm['carControl']

    v_ego = cs.vEgo
    actual_lat_accel = controls_state.curvature * v_ego ** 2
    desired_lat_accel = controls_state.desiredCurvature * v_ego ** 2

    self._lines = [
      f"v_ego      {v_ego:5.1f} m/s",
      f"steer ang  {cs.steeringAngleDeg:5.1f} deg",
      f"lat acc    {actual_lat_accel:5.2f}",
      f"des acc    {desired_lat_accel:5.2f}",
      f"acc err    {desired_lat_accel - actual_lat_accel:5.2f}",
      f"lat act    {'Y' if car_control.latActive else 'N'}",
    ]

  def _render(self, rect: rl.Rectangle):
    if not self._lines:
      return

    # left edge, vertically centered: clears alerts/set-speed (top), wheel and torque bar (bottom), path (center)
    box = rl.Rectangle(rect.x + 6, rect.y + (rect.height - BOX_H) / 2, BOX_W, BOX_H)
    rl.draw_rectangle_rounded(box, 0.15, 10, rl.Color(0, 0, 0, 150))

    row_h = (box.height - PAD * 2) / len(self._lines)
    for i, line in enumerate(self._lines):
      row = rl.Rectangle(box.x + PAD, box.y + PAD + i * row_h, box.width - PAD * 2, row_h)
      gui_label(row, line, FONT_SIZE, rl.Color(255, 255, 255, 235), font_weight=FontWeight.MEDIUM)
