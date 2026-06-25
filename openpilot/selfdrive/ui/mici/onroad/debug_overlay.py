import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

# onroad debug overlay: live actual vs desired lateral accel and model exec timing on the driving view
# gated by the OnroadDebugOverlay param, mici only. small label over a big value so it reads at arm's
# length on the tiny screen. top-right corner clears every other element. plain curvature * v^2, no roll comp.

LABEL_SIZE = 18
VALUE_SIZE = 37
PAD = 3
ROW_GAP = 2
TOP_MARGIN = 4
RIGHT_GAP = 74  # right edge sits just past the widest label, clear of the confidence ball
BOX_W = 142

WHITE = rl.Color(255, 255, 255, 235)


class DebugOverlay(Widget):
  def __init__(self):
    super().__init__()
    self._label_font = gui_app.font(FontWeight.MEDIUM)
    self._value_font = gui_app.font(FontWeight.BOLD)
    self._rows: list[tuple[str, str]] = []

  def _update_state(self):
    if not ui_state.onroad_debug_overlay or not ui_state.started:
      self._rows = []
      return

    sm = ui_state.sm
    # skip until each service is alive and has arrived since going onroad, else we'd show stale numbers as live
    services = ('carState', 'controlsState', 'modelV2')
    if any(sm.recv_frame[s] < ui_state.started_frame or not sm.alive[s] for s in services):
      self._rows = []
      return

    controls_state = sm['controlsState']
    v_ego_sq = sm['carState'].vEgo ** 2
    actual_lat_accel = controls_state.curvature * v_ego_sq
    desired_lat_accel = controls_state.desiredCurvature * v_ego_sq
    model_ms = sm['modelV2'].modelExecutionTime * 1000.0

    self._rows = [
      ("actual lat accel", f"{actual_lat_accel:.2f}"),
      ("desired lat accel", f"{desired_lat_accel:.2f}"),
      ("model ms", f"{model_ms:.1f}"),
    ]

  def _render(self, rect: rl.Rectangle):
    if not self._rows:
      return

    row_h = LABEL_SIZE + VALUE_SIZE + ROW_GAP
    box_h = len(self._rows) * row_h - ROW_GAP + PAD * 2
    box = rl.Rectangle(rect.x + rect.width - BOX_W - RIGHT_GAP, rect.y + TOP_MARGIN, BOX_W, box_h)
    rl.draw_rectangle_rounded(box, 0.12, 10, rl.Color(0, 0, 0, 150))

    y = box.y + PAD
    for label, value in self._rows:
      rl.draw_text_ex(self._label_font, label, rl.Vector2(box.x + PAD, y), LABEL_SIZE, 0, WHITE)
      rl.draw_text_ex(self._value_font, value, rl.Vector2(box.x + PAD, y + LABEL_SIZE + 2), VALUE_SIZE, 0, WHITE)
      y += row_h
