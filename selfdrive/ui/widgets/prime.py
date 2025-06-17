import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.widget import Widget


DEFAULT_TEXT_SIZE  = 40
DEFAULT_TEXT_COLOR = rl.WHITE


class Label(Widget):
  """Immediate-mode QLabel clone: automatic elide + alignment."""

  def __init__(self,
               text: str,
               font_size: int = DEFAULT_TEXT_SIZE,
               color: rl.Color = DEFAULT_TEXT_COLOR,
               font_weight: FontWeight = FontWeight.NORMAL,
               align_h: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
               align_v: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
               elide_right: bool = True):
    super().__init__()
    self.text = text
    self.font_size = font_size
    self.color = color
    self.font_weight = font_weight
    self.align_h = align_h
    self.align_v = align_v
    self.elide_right = elide_right

  # ------------------------------------------------------------
  def set_text(self, txt: str):
    self.text = txt

  # ------------------------------------------------------------
  def _render(self, rect: rl.Rectangle):
    font = gui_app.font(self.font_weight)
    text_size = measure_text_cached(font, self.text, self.font_size)
    display = self.text

    # -------- elide if too wide ---------------------------------
    if self.elide_right and text_size.x > rect.width:
      ellipsis = "..."
      left, right = 0, len(self.text)
      while left < right:
        mid = (left + right) // 2
        candidate = self.text[:mid] + ellipsis
        if measure_text_cached(font, candidate, self.font_size).x <= rect.width:
          left = mid + 1
        else:
          right = mid
      display = (self.text[: left - 1] + ellipsis) if left > 0 else ellipsis
      text_size = measure_text_cached(font, display, self.font_size)

    # -------- horizontal alignment ------------------------------
    text_x = rect.x + {
      rl.GuiTextAlignment.TEXT_ALIGN_LEFT: 0,
      rl.GuiTextAlignment.TEXT_ALIGN_CENTER: (rect.width - text_size.x) / 2,
      rl.GuiTextAlignment.TEXT_ALIGN_RIGHT: rect.width - text_size.x,
    }.get(self.align_h, 0)

    # -------- vertical alignment --------------------------------
    text_y = rect.y + {
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP: 0,
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE: (rect.height - text_size.y) / 2,
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM: rect.height - text_size.y,
    }.get(self.align_v, 0)

    rl.draw_text_ex(font, display,
                    rl.Vector2(text_x, text_y),
                    self.font_size, 0, self.color)


class PrimeWidget(Widget):
  """Widget for displaying comma prime subscription status"""

  PRIME_BG_COLOR = rl.Color(51, 51, 51, 255)

  def __init__(self):
    super().__init__()
    self._upgrade_now = Label("Upgrade Now", font_size=75, font_weight=FontWeight.BOLD)


  def _render(self, rect):
    if ui_state.prime_state.is_prime() and False:
      self._render_for_prime_user(rect)
    else:
      self._render_for_non_prime_users(rect)

  def _render_for_non_prime_users(self, rect: rl.Rectangle):
    """Renders the advertisement for non-Prime users."""

    rl.draw_rectangle_rounded(rect, 0.02, 10, self.PRIME_BG_COLOR)

    # Layout
    x, y = rect.x + 80, rect.y + 90
    w = rect.width - 160

    # Title
    # gui_label(rl.Rectangle(x, y, w, 90), "Upgrade Now", 75, font_weight=FontWeight.BOLD)
    self._upgrade_now.render(rl.Rectangle(x, y, w, 90))

    # Description with wrapping
    desc_y = y + 140
    font = gui_app.font(FontWeight.LIGHT)
    wrapped_text = "\n".join(wrap_text(font, "Become a comma prime member at connect.comma.ai", 56, int(w)))
    text_size = measure_text_cached(font, wrapped_text, 56)
    rl.draw_text_ex(font, wrapped_text, rl.Vector2(x, desc_y), 56, 0, rl.Color(255, 255, 255, 255))

    # Features section
    features_y = desc_y + text_size.y + 50
    gui_label(rl.Rectangle(x, features_y, w, 50), "PRIME FEATURES:", 41, font_weight=FontWeight.BOLD)

    # Feature list
    features = ["Remote access", "24/7 LTE connectivity", "1 year of drive storage", "Remote snapshots"]
    for i, feature in enumerate(features):
      item_y = features_y + 80 + i * 65
      gui_label(rl.Rectangle(x, item_y, 50, 60), "✓", 50, color=rl.Color(70, 91, 234, 255))
      gui_label(rl.Rectangle(x + 60, item_y, w - 60, 60), feature, 50)

  def _render_for_prime_user(self, rect: rl.Rectangle):
    """Renders the prime user widget with subscription status."""

    rl.draw_rectangle_rounded(rl.Rectangle(rect.x, rect.y, rect.width, 230), 0.02, 10, self.PRIME_BG_COLOR)

    x = rect.x + 56
    y = rect.y + 40

    font = gui_app.font(FontWeight.BOLD)
    rl.draw_text_ex(font, "✓ SUBSCRIBED", rl.Vector2(x, y), 41, 0, rl.Color(134, 255, 78, 255))
    rl.draw_text_ex(font, "comma prime", rl.Vector2(x, y + 61), 75, 0, rl.WHITE)
