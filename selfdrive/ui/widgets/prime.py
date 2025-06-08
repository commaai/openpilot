import pyray as rl
from openpilot.system.ui.lib.application import gui_app, Widget, FontWeight
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached


class PrimeAdWidget(Widget):
  """Advertisement widget for non-Prime users."""

  def __init__(self):
    super().__init__()

  def _render(self, rect):
    # Layout
    x, y = rect.x + 80, rect.y + 90
    w = rect.width - 160

    # Title
    gui_label(rl.Rectangle(x, y, w, 90), "Upgrade Now", 75, font_weight=FontWeight.BOLD)

    # Description with wrapping
    desc_y = y + 140
    font = gui_app.font(FontWeight.LIGHT)
    wrapped_text = "\n".join(wrap_text(font, "Become a comma prime member at connect.comma.ai", 56, w))
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
