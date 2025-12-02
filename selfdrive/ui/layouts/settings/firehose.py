import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr, trn, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.selfdrive.ui.mici.layouts.settings.firehose import FirehoseLayoutBase

TITLE = tr_noop("Firehose Mode")
DESCRIPTION = tr_noop(
  "openpilot learns to drive by watching humans, like you, drive.\n\n"
  + "Firehose Mode allows you to maximize your training data uploads to improve "
  + "openpilot's driving models. More data means bigger models, which means better Experimental Mode."
)
INSTRUCTIONS = tr_noop(
  "For maximum effectiveness, bring your device inside and connect to a good USB-C adapter and Wi-Fi weekly.\n\n"
  + "Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card.\n\n\n"
  + "Frequently Asked Questions\n\n"
  + "Does it matter how or where I drive? Nope, just drive as you normally would.\n\n"
  + "Do all of my segments get pulled in Firehose Mode? No, we selectively pull a subset of your segments.\n\n"
  + "What's a good USB-C adapter? Any fast phone or laptop charger should be fine.\n\n"
  + "Does it matter which software I run? Yes, only upstream openpilot (and particular forks) are able to be used for training."
)


class FirehoseLayout(FirehoseLayoutBase):
  def __init__(self):
    super().__init__()
    self._scroll_panel = GuiScrollPanel()

  def _render(self, rect: rl.Rectangle):
    # Calculate content dimensions
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, self._content_height)

    # Handle scrolling and render with clipping
    scroll_offset = self._scroll_panel.update(rect, content_rect)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._content_height = self._render_content(rect, scroll_offset)
    rl.end_scissor_mode()

  def _render_content(self, rect: rl.Rectangle, scroll_offset: float) -> int:
    x = int(rect.x + 40)
    y = int(rect.y + 40 + scroll_offset)
    w = int(rect.width - 80)

    # Title (centered)
    title_text = tr(TITLE)  # live translate
    title_font = gui_app.font(FontWeight.MEDIUM)
    text_width = measure_text_cached(title_font, title_text, 100).x
    title_x = rect.x + (rect.width - text_width) / 2
    rl.draw_text_ex(title_font, title_text, rl.Vector2(title_x, y), 100, 0, rl.WHITE)
    y += 200

    # Description
    y = self._draw_wrapped_text(x, y, w, tr(DESCRIPTION), gui_app.font(FontWeight.NORMAL), 45, rl.WHITE)
    y += 40 + 20

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 30 + 20

    # Status
    status_text, status_color = self._get_status()
    y = self._draw_wrapped_text(x, y, w, status_text, gui_app.font(FontWeight.BOLD), 60, status_color)
    y += 20 + 20

    # Contribution count (if available)
    if self._segment_count > 0:
      contrib_text = trn("{} segment of your driving is in the training dataset so far.",
                         "{} segments of your driving is in the training dataset so far.", self._segment_count).format(self._segment_count)
      y = self._draw_wrapped_text(x, y, w, contrib_text, gui_app.font(FontWeight.BOLD), 52, rl.WHITE)
      y += 20 + 20

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 30 + 20

    # Instructions
    y = self._draw_wrapped_text(x, y, w, tr(INSTRUCTIONS), gui_app.font(FontWeight.NORMAL), 40, self.LIGHT_GRAY)

    # bottom margin + remove effect of scroll offset
    return int(round(y - self._scroll_panel.offset + 40))

  def _draw_wrapped_text(self, x, y, width, text, font, font_size, color):
    wrapped = wrap_text(font, text, font_size, width)
    for line in wrapped:
      rl.draw_text_ex(font, line, rl.Vector2(x, y), font_size, 0, color)
      y += font_size * FONT_SCALE
    return round(y)
