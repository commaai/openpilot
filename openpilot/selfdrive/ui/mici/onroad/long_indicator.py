import raylib_nb as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.onroad import DIMMED_ICON_ALPHA
from openpilot.selfdrive.ui.mici.onroad.status import status_fade
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


class LongIndicator(Widget):
  def __init__(self):
    super().__init__()
    # lead car, then the distance bars, each with its center y
    self._icons = [(self._texture('car', 40, 31), 96), (self._texture('distance_1', 33, 8), 123),
                   (self._texture('distance_2', 40, 9), 138), (self._texture('distance_3', 48, 10), 155)]
    self._lit_filters = [FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps) for _ in self._icons]
    self._alpha_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    # stock ACC cars show a static speed dial instead
    self._txt_speed_dial = self._texture('speed_dial', 56, 48)
    self._should_draw = self._has_lead = False

  @staticmethod
  def _texture(name: str, width: int, height: int) -> rl.Texture:
    return gui_app.texture(f'icons_mici/longitudinal/{name}.png', width, height, keep_aspect_ratio=False)

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  def set_has_lead(self, has_lead: bool):
    self._has_lead = has_lead

  def _render(self, rect: rl.Rectangle) -> None:
    # offroad shows the disengaged state
    if ui_state.started and not ui_state.received_since_started('selfdriveState'):
      self._alpha_filter.x = 0.0
      return

    alpha = self._alpha_filter.update(self._should_draw)
    disengaged = status_fade.weight(UIStatus.DISENGAGED)
    if not ui_state.has_longitudinal_control:
      # the needle's arrowhead overhangs the left, so the dial sits 2px left of center
      dial_rect = rl.Rectangle(rect.x - 2, rect.y, rect.width, rect.height)
      self._draw_centered(self._txt_speed_dial, dial_rect, 120, (0.9 - (0.9 - DIMMED_ICON_ALPHA) * disengaged) * alpha)
      return

    # white car while openpilot has a lead, lit bars up to the personality. lit icons dim while disengaged, unlit ones stay grey
    personality = ui_state.sm['selfdriveState'].personality.raw if ui_state.started else ui_state.personality_param
    lit_alpha = 0.9 + (0.63 - 0.9) * disengaged
    for i, ((txt, y), f) in enumerate(zip(self._icons, self._lit_filters, strict=True)):
      lit = f.update(self._has_lead if i == 0 else i <= personality + 1)
      self._draw_centered(txt, rect, y, (0.35 + (lit_alpha - 0.35) * lit) * alpha)

  @staticmethod
  def _draw_centered(texture: rl.Texture, rect: rl.Rectangle, y: float, alpha: float) -> None:
    pos = rl.Vector2(rect.x + (rect.width - texture.width) / 2, rect.y + y - texture.height / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, rl.Color(255, 255, 255, round(255 * alpha)))
