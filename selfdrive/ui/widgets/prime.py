import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label

PRIME_BG_COLOR = rl.Color(51, 51, 51, 255)
SUBSCRIBED_TEXT_COLOR = rl.Color(134, 255, 78, 255)

FONT_SIZE_LARGE = 75
FONT_SIZE_MEDIUM = 56
FONT_SIZE_SMALL = 41
FEATURES_FONT_SIZE = 50


class PrimeWidget(Widget):
  """Widget for displaying comma prime subscription status"""

  def __init__(self):
    super().__init__()
    self._upgrade_now = Label("Upgrade Now", FONT_SIZE_LARGE, FontWeight.BOLD)
    self._prime_features = Label("PRIME FEATURES:", FONT_SIZE_SMALL, FontWeight.BOLD)
    self._check_mark_label = Label("✓", FEATURES_FONT_SIZE, FontWeight.BOLD, color=rl.Color(70, 91, 234, 255))
    self._feature_labels = [
      Label("Remote access", FEATURES_FONT_SIZE),
      Label("24/7 LTE connectivity", FEATURES_FONT_SIZE),
      Label("1 year of drive storage", FEATURES_FONT_SIZE),
      Label("Remote snapshots", FEATURES_FONT_SIZE),
    ]
    self._subscribed_label = Label("✓ SUBSCRIBED", FONT_SIZE_SMALL, FontWeight.BOLD, color=SUBSCRIBED_TEXT_COLOR)
    self._prime_label = Label("comma prime", FONT_SIZE_LARGE, FontWeight.BOLD)

  def _render(self, rect):
    if ui_state.prime_state.is_prime():
      self._render_for_prime_user(rect)
    else:
      self._render_for_non_prime_users(rect)

  def _render_for_non_prime_users(self, rect: rl.Rectangle):
    """Renders the advertisement for non-Prime users."""

    # Draw background
    rl.draw_rectangle_rounded(rect, 0.02, 10, PRIME_BG_COLOR)

    # Layout
    x, y = rect.x + 80, rect.y + 90
    w = rect.width - 160

    # Title
    self._upgrade_now.render(rl.Rectangle(x, y, w, self._upgrade_now.font_size))

    # Description with wrapping
    desc_y = y + 140
    font = gui_app.font(FontWeight.LIGHT)
    wrapped_text = "\n".join(wrap_text(font, "Become a comma prime member at connect.comma.ai", FONT_SIZE_MEDIUM, int(w)))
    text_size = measure_text_cached(font, wrapped_text, FONT_SIZE_MEDIUM)
    rl.draw_text_ex(font, wrapped_text, rl.Vector2(x, desc_y), FONT_SIZE_MEDIUM, 0, rl.Color(255, 255, 255, 255))

    # Features section
    features_y = desc_y + text_size.y + self._prime_features.font_size
    self._prime_features.render(rl.Rectangle(x, features_y, w, self._prime_features.font_size))

    # Feature list
    for i, feature_label in enumerate(self._feature_labels):
      item_y = features_y + 80 + i * 65
      # Draw check mark
      self._check_mark_label.render(rl.Rectangle(x, item_y, 50, self._check_mark_label.font_size))
      # Draw feature label
      feature_label.render(rl.Rectangle(x + 60, item_y, w - 60, feature_label.font_size))

  def _render_for_prime_user(self, rect: rl.Rectangle):
    """Renders the prime user widget with subscription status."""

    # Draw background
    rl.draw_rectangle_rounded(rl.Rectangle(rect.x, rect.y, rect.width, 230), 0.02, 10, PRIME_BG_COLOR)

    # Calculate positions
    x = rect.x + 56
    y = rect.y + 40
    w = rect.width

    # Render the labels
    self._subscribed_label.render(rl.Rectangle(x, y, w, 61))
    self._prime_label.render(rl.Rectangle(x, y + 61, w, 90))
