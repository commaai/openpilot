import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


INACTIVE_COLOR = rl.Color(86, 86, 86, 255)
INACTIVE_TEXTURE_TINT = rl.Color(140, 140, 140, 255)
PlanSource = log.LongitudinalPlan.LongitudinalPlanSource
PERSONALITY_BARS = {
  log.LongitudinalPersonality.relaxed: 1,
  log.LongitudinalPersonality.standard: 2,
  log.LongitudinalPersonality.aggressive: 3,
}


class StatusIconColumn(Widget):
  """Four-slot status rail for experimental mode, speed limiter, eGPU use, and personality."""

  def __init__(self):
    super().__init__()

    self._experimental_texture = gui_app.texture("icons_mici/experimental_mode.png", 48, 48)
    self._cruise_texture = gui_app.texture("icons_mici/speed_limiter_cruise.png", 48, 48)
    self._lead_texture = gui_app.texture("icons_mici/speed_limiter_lead.png", 48, 48)
    self._egpu_texture = gui_app.texture("icons_mici/egpu.png", 50, 37)
    self._egpu_gray_texture = gui_app.texture("icons_mici/egpu_gray.png", 50, 37)
    self._personality_textures = tuple(
      gui_app.texture(f"icons_mici/personality_{bar_count}.png", 56, 20)
      for bar_count in range(1, 4)
    )

    self._experimental_mode = False
    self._speed_limiter_texture = self._cruise_texture
    self._speed_limiter_active = False
    self._using_egpu = False
    self._personality_bar_count = 0

  def _update_state(self):
    sm = ui_state.sm
    selfdrive_state_valid = sm.alive["selfdriveState"] and sm.valid["selfdriveState"]
    self._experimental_mode = selfdrive_state_valid and sm["selfdriveState"].experimentalMode
    personality_available = selfdrive_state_valid and ui_state.has_longitudinal_control
    self._personality_bar_count = PERSONALITY_BARS.get(sm["selfdriveState"].personality, 0) if personality_available else 0
    source = sm["longitudinalPlan"].longitudinalPlanSource
    if source in (PlanSource.lead0, PlanSource.lead1, PlanSource.lead2):
      self._speed_limiter_texture = self._lead_texture
    elif source == PlanSource.e2e:
      self._speed_limiter_texture = self._experimental_texture
    else:
      self._speed_limiter_texture = self._cruise_texture
    self._speed_limiter_active = (sm.alive["carControl"] and sm.valid["carControl"] and
                                  sm["carControl"].longActive and
                                  sm.alive["longitudinalPlan"] and sm.valid["longitudinalPlan"])
    self._using_egpu = ui_state.usbgpu and ui_state.usbgpu_compiled

  @staticmethod
  def _draw_texture_centered(texture: rl.Texture, center: rl.Vector2, tint: rl.Color = rl.WHITE):
    pos = rl.Vector2(center.x - texture.width / 2, center.y - texture.height / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, tint)

  def _render(self, _):
    panel = rl.Rectangle(
      self.rect.x + self.rect.width - SIDE_PANEL_WIDTH,
      self.rect.y,
      SIDE_PANEL_WIDTH,
      self.rect.height,
    )
    center_x = panel.x + panel.width / 2
    centers = [rl.Vector2(center_x, panel.y + panel.height * (slot + 0.5) / 4) for slot in range(4)]

    experimental_tint = rl.WHITE if self._experimental_mode else INACTIVE_COLOR
    self._draw_texture_centered(self._experimental_texture, centers[0], experimental_tint)
    limiter_tint = rl.WHITE if self._speed_limiter_active else INACTIVE_COLOR
    self._draw_texture_centered(self._speed_limiter_texture, centers[1], limiter_tint)
    egpu_texture = self._egpu_texture if self._using_egpu else self._egpu_gray_texture
    egpu_tint = rl.WHITE if self._using_egpu else INACTIVE_TEXTURE_TINT
    self._draw_texture_centered(egpu_texture, centers[2], egpu_tint)
    personality_index = self._personality_bar_count - 1 if self._personality_bar_count else 2
    personality_tint = rl.WHITE if self._personality_bar_count else INACTIVE_COLOR
    self._draw_texture_centered(self._personality_textures[personality_index], centers[3], personality_tint)
