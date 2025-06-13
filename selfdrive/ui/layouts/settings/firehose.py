import pyray as rl
import json
import time
import threading

from openpilot.common.api import Api, api_get
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.widget import Widget
from openpilot.selfdrive.ui.ui_state import ui_state


TITLE = "Firehose Mode"
DESCRIPTION = (
  "openpilot learns to drive by watching humans, like you, drive.\n\n"
  + "Firehose Mode allows you to maximize your training data uploads to improve "
  + "openpilot's driving models. More data means bigger models, which means better Experimental Mode."
)
INSTRUCTIONS = (
  "For maximum effectiveness, bring your device inside and connect to a good USB-C adapter and Wi-Fi weekly.\n\n"
  + "Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card.\n\n"
  + "Frequently Asked Questions\n\n"
  + "Does it matter how or where I drive? Nope, just drive as you normally would.\n\n"
  + "Do all of my segments get pulled in Firehose Mode? No, we selectively pull a subset of your segments.\n\n"
  + "What's a good USB-C adapter? Any fast phone or laptop charger should be fine.\n\n"
  + "Does it matter which software I run? Yes, only upstream openpilot (and particular forks) are able to be used for training."
)


class FirehoseLayout(Widget):
  PARAM_KEY = "ApiCache_FirehoseStats"
  GREEN = rl.Color(46, 204, 113, 255)
  RED = rl.Color(231, 76, 60, 255)
  GRAY = rl.Color(68, 68, 68, 255)
  LIGHT_GRAY = rl.Color(228, 228, 228, 255)
  UPDATE_INTERVAL = 30  # seconds

  def __init__(self):
    super().__init__()
    self.params = Params()
    self.segment_count = self._get_segment_count()
    self.scroll_panel = GuiScrollPanel()

    self.running = True
    self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
    self.update_thread.start()
    self.last_update_time = 0

  def _get_segment_count(self) -> int:
    stats = self.params.get(self.PARAM_KEY, encoding='utf8')
    try:
      return int(json.loads(stats).get("firehose", 0))
    except Exception:
      cloudlog.exception(f"Failed to decode firehose stats: {stats}")
      return 0

  def __del__(self):
    self.running = False
    if self.update_thread and self.update_thread.is_alive():
      self.update_thread.join(timeout=1.0)

  def _render(self, rect: rl.Rectangle):
    # Calculate content dimensions
    content_width = rect.width - 80
    content_height = self._calculate_content_height(int(content_width))
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, content_height)

    # Handle scrolling and render with clipping
    scroll_offset = self.scroll_panel.handle_scroll(rect, content_rect)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._render_content(rect, scroll_offset)
    rl.end_scissor_mode()

  def _calculate_content_height(self, content_width: int) -> int:
    height = 80  # Top margin

    # Title
    height += 100 + 40

    # Description
    desc_font = gui_app.font(FontWeight.NORMAL)
    desc_lines = wrap_text(desc_font, DESCRIPTION, 45, content_width)
    height += len(desc_lines) * 45 + 40

    # Status section
    height += 32  # Separator
    status_text, _ = self._get_status()
    status_lines = wrap_text(gui_app.font(FontWeight.BOLD), status_text, 60, content_width)
    height += len(status_lines) * 60 + 20

    # Contribution count (if available)
    if self.segment_count > 0:
      contrib_text = f"{self.segment_count} segment(s) of your driving is in the training dataset so far."
      contrib_lines = wrap_text(gui_app.font(FontWeight.BOLD), contrib_text, 52, content_width)
      height += len(contrib_lines) * 52 + 20

    # Instructions section
    height += 32  # Separator
    inst_lines = wrap_text(gui_app.font(FontWeight.NORMAL), INSTRUCTIONS, 40, content_width)
    height += len(inst_lines) * 40 + 40  # Bottom margin

    return height

  def _render_content(self, rect: rl.Rectangle, scroll_offset: rl.Vector2):
    x = int(rect.x + 40)
    y = int(rect.y + 40 + scroll_offset.y)
    w = int(rect.width - 80)

    # Title
    title_font = gui_app.font(FontWeight.MEDIUM)
    rl.draw_text_ex(title_font, TITLE, rl.Vector2(x, y), 100, 0, rl.WHITE)
    y += 140

    # Description
    y = self._draw_wrapped_text(x, y, w, DESCRIPTION, gui_app.font(FontWeight.NORMAL), 45, rl.WHITE)
    y += 40

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 30

    # Status
    status_text, status_color = self._get_status()
    y = self._draw_wrapped_text(x, y, w, status_text, gui_app.font(FontWeight.BOLD), 60, status_color)
    y += 20

    # Contribution count (if available)
    if self.segment_count > 0:
      contrib_text = f"{self.segment_count} segment(s) of your driving is in the training dataset so far."
      y = self._draw_wrapped_text(x, y, w, contrib_text, gui_app.font(FontWeight.BOLD), 52, rl.WHITE)
      y += 20

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 30

    # Instructions
    self._draw_wrapped_text(x, y, w, INSTRUCTIONS, gui_app.font(FontWeight.NORMAL), 40, self.LIGHT_GRAY)

  def _draw_wrapped_text(self, x, y, width, text, font, size, color):
    wrapped = wrap_text(font, text, size, width)
    for line in wrapped:
      rl.draw_text_ex(font, line, rl.Vector2(x, y), size, 0, color)
      y += size
    return y

  def _get_status(self) -> tuple[str, rl.Color]:
    network_type = ui_state.sm["deviceState"].networkType
    network_metered = ui_state.sm["deviceState"].networkMetered

    if not network_metered and network_type != 0:  # Not metered and connected
      return "ACTIVE", self.GREEN
    else:
      return "INACTIVE: connect to an unmetered network", self.RED

  def _fetch_firehose_stats(self):
    try:
      dongle_id = self.params.get("DongleId", encoding='utf8') or ""
      identity_token = Api(dongle_id).get_token()
      response = api_get(f"v1/devices/{dongle_id}/firehose_stats", access_token=identity_token)
      if response.status_code == 200:
        data = response.json()
        self.segment_count = data.get("firehose", 0)
        self.params.put(self.PARAM_KEY, json.dumps(data))
    except Exception as e:
      cloudlog.error(f"Failed to fetch firehose stats: {e}")

  def _update_loop(self):
    while self.running:
      if not ui_state.started:
        self._fetch_firehose_stats()
      time.sleep(self.UPDATE_INTERVAL)
