import pyray as rl

from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.personality import personality_bar_count
from openpilot.selfdrive.ui.plan_source import SpeedLimiter, speed_limiter_from_source
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


INACTIVE_COLOR = rl.Color(86, 86, 86, 255)
INACTIVE_TEXTURE_TINT = rl.Color(140, 140, 140, 255)
CRUISE_COLOR = rl.Color(0, 163, 255, 255)
LEAD_COLOR = rl.Color(255, 190, 0, 255)
E2E_COLOR = rl.Color(0, 255, 204, 255)
PERSONALITY_ACTIVE_COLOR = rl.Color(255, 255, 255, 230)


class StatusIconColumn(Widget):
  """Four-slot status rail for experimental mode, speed limiter, eGPU use, and personality."""

  def __init__(self):
    super().__init__()

    self._experimental_texture = gui_app.texture("icons_mici/experimental_mode.png", 48, 48)
    self._egpu_texture = gui_app.texture("icons_mici/egpu.png", 50, 37)
    self._egpu_gray_texture = gui_app.texture("icons_mici/egpu_gray.png", 50, 37)

    self._experimental_mode = False
    self._speed_limiter = SpeedLimiter.CRUISE
    self._speed_limiter_active = False
    self._using_egpu = False
    self._personality_bar_count = 0

  def _update_state(self):
    sm = ui_state.sm
    selfdrive_state_valid = sm.alive["selfdriveState"] and sm.valid["selfdriveState"]
    self._experimental_mode = selfdrive_state_valid and sm["selfdriveState"].experimentalMode
    personality_available = selfdrive_state_valid and ui_state.has_longitudinal_control
    self._personality_bar_count = personality_bar_count(sm["selfdriveState"].personality, personality_available)
    self._speed_limiter = speed_limiter_from_source(sm["longitudinalPlan"].longitudinalPlanSource)
    self._speed_limiter_active = (sm.alive["carControl"] and sm.valid["carControl"] and
                                  sm["carControl"].longActive and
                                  sm.alive["longitudinalPlan"] and sm.valid["longitudinalPlan"])
    self._using_egpu = ui_state.usbgpu and ui_state.usbgpu_compiled

  @staticmethod
  def _draw_texture_centered(texture: rl.Texture, center: rl.Vector2, tint: rl.Color = rl.WHITE):
    pos = rl.Vector2(center.x - texture.width / 2, center.y - texture.height / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, tint)

  @staticmethod
  def _draw_cruise_icon(center: rl.Vector2, color: rl.Color):
    gauge_center = rl.Vector2(center.x, center.y + 4)
    rl.draw_ring(gauge_center, 14, 19, 180, 360, 32, color)
    rl.draw_line_ex(gauge_center, rl.Vector2(center.x + 11, center.y - 6), 4, color)
    rl.draw_circle(int(gauge_center.x), int(gauge_center.y), 4, color)

  @staticmethod
  def _draw_lead_icon(center: rl.Vector2, color: rl.Color, active: bool):
    # Front-view car silhouette.
    rl.draw_rectangle_rounded(rl.Rectangle(center.x - 12, center.y - 18, 24, 17), 0.35, 8, color)
    rl.draw_rectangle_rounded(rl.Rectangle(center.x - 18, center.y - 8, 36, 23), 0.3, 8, color)
    rl.draw_rectangle_rec(rl.Rectangle(center.x - 8, center.y - 14, 16, 9), rl.BLACK)
    detail_color = rl.WHITE if active else color
    rl.draw_circle(int(center.x - 10), int(center.y + 7), 3, detail_color)
    rl.draw_circle(int(center.x + 10), int(center.y + 7), 3, detail_color)
    rl.draw_rectangle_rec(rl.Rectangle(center.x - 19, center.y + 5, 4, 10), color)
    rl.draw_rectangle_rec(rl.Rectangle(center.x + 15, center.y + 5, 4, 10), color)

  @staticmethod
  def _draw_e2e_icon(center: rl.Vector2, color: rl.Color):
    # A compact network glyph for the end-to-end model.
    start = rl.Vector2(center.x, center.y + 18)
    left = rl.Vector2(center.x - 13, center.y)
    right = rl.Vector2(center.x + 13, center.y)
    end = rl.Vector2(center.x, center.y - 18)
    for first, second in ((start, left), (start, right), (left, end), (right, end)):
      rl.draw_line_ex(first, second, 4, color)
    for point in (start, left, right, end):
      rl.draw_circle(int(point.x), int(point.y), 5, color)

  @staticmethod
  def _draw_personality_icon(center: rl.Vector2, filled_bars: int):
    heights = (18, 26, 34)
    bar_width = 8
    bar_gap = 5
    total_width = len(heights) * bar_width + (len(heights) - 1) * bar_gap
    left = center.x - total_width / 2
    baseline = center.y + 17

    for index, height in enumerate(heights):
      rect = rl.Rectangle(left + index * (bar_width + bar_gap), baseline - height, bar_width, height)
      color = PERSONALITY_ACTIVE_COLOR if index < filled_bars else INACTIVE_COLOR
      rl.draw_rectangle_rounded(rect, 0.35, 6, color)

  def _draw_speed_limiter(self, center: rl.Vector2):
    if not self._speed_limiter_active:
      color = INACTIVE_COLOR
    elif self._speed_limiter == SpeedLimiter.LEAD:
      color = LEAD_COLOR
    elif self._speed_limiter == SpeedLimiter.E2E:
      color = E2E_COLOR
    else:
      color = CRUISE_COLOR

    if self._speed_limiter == SpeedLimiter.LEAD:
      self._draw_lead_icon(center, color, self._speed_limiter_active)
    elif self._speed_limiter == SpeedLimiter.E2E:
      self._draw_e2e_icon(center, color)
    else:
      self._draw_cruise_icon(center, color)

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
    self._draw_speed_limiter(centers[1])
    egpu_texture = self._egpu_texture if self._using_egpu else self._egpu_gray_texture
    egpu_tint = rl.WHITE if self._using_egpu else INACTIVE_TEXTURE_TINT
    self._draw_texture_centered(egpu_texture, centers[2], egpu_tint)
    self._draw_personality_icon(centers[3], self._personality_bar_count)
