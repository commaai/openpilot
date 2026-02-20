import requests
import threading
import time
import pyray as rl

from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.lib.api_helpers import get_token
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.system.ui.lib.multilang import tr, trn, tr_noop
from openpilot.system.ui.widgets import Widget, NavWidget

TITLE = tr_noop("Firehose Mode")
DESCRIPTION = tr_noop(
  "openpilot learns to drive by watching humans, like you, drive.\n\n"
  + "Firehose Mode allows you to maximize your training data uploads to improve "
  + "openpilot's driving models. More data means bigger models, which means better Experimental Mode."
)
INSTRUCTIONS_INTRO = tr_noop(
  "For maximum effectiveness, bring your device inside and connect to a good USB-C adapter and Wi-Fi weekly.\n\n"
  + "Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card."
)
FAQ_HEADER = tr_noop("Frequently Asked Questions")
FAQ_ITEMS = [
  (tr_noop("Does it matter how or where I drive?"), tr_noop("Nope, just drive as you normally would.")),
  (tr_noop("Do all of my segments get pulled in Firehose Mode?"), tr_noop("No, we selectively pull a subset of your segments.")),
  (tr_noop("What's a good USB-C adapter?"), tr_noop("Any fast phone or laptop charger should be fine.")),
  (tr_noop("Does it matter which software I run?"), tr_noop("Yes, only upstream openpilot (and particular forks) are able to be used for training.")),
]


class FirehoseLayoutBase(Widget):
  PARAM_KEY = "ApiCache_FirehoseStats"
  GREEN = rl.Color(46, 204, 113, 255)
  RED = rl.Color(231, 76, 60, 255)
  GRAY = rl.Color(68, 68, 68, 255)
  LIGHT_GRAY = rl.Color(228, 228, 228, 255)
  UPDATE_INTERVAL = 30  # seconds

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._session = requests.Session()  # reuse session to reduce SSL handshake overhead
    self._segment_count = self._get_segment_count()

    self._scroll_panel = GuiScrollPanel2(horizontal=False)
    self._content_height = 0

    self._running = True
    self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
    self._update_thread.start()

  def __del__(self):
    self._running = False
    try:
      if self._update_thread and self._update_thread.is_alive():
        self._update_thread.join(timeout=1.0)
    except Exception:
      pass

  def show_event(self):
    super().show_event()
    self._scroll_panel.set_offset(0)

  def _get_segment_count(self) -> int:
    stats = self._params.get(self.PARAM_KEY)
    if not stats:
      return 0
    try:
      return int(stats.get("firehose", 0))
    except Exception:
      cloudlog.exception(f"Failed to decode firehose stats: {stats}")
      return 0

  def _render(self, rect: rl.Rectangle):
    # compute total content height for scrolling
    content_height = self._measure_content_height(rect)
    scroll_offset = round(self._scroll_panel.update(rect, content_height))

    # start drawing with offset
    x = int(rect.x + 40)
    y = int(rect.y + 40 + scroll_offset)
    w = int(rect.width - 80)

    # Title
    title_text = tr(TITLE)
    title_font = gui_app.font(FontWeight.BOLD)
    title_size = 64
    rl.draw_text_ex(title_font, title_text, rl.Vector2(x, y), title_size, 0, rl.WHITE)
    y += int(title_size * FONT_SCALE) + 20

    # Description
    y = self._draw_wrapped_text(x, y, w, tr(DESCRIPTION), gui_app.font(FontWeight.ROMAN), 36, rl.WHITE)
    y += 20

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 20

    # Status
    status_text, status_color = self._get_status()
    y = self._draw_wrapped_text(x, y, w, status_text, gui_app.font(FontWeight.BOLD), 48, status_color)
    y += 20

    # Contribution count (if available)
    if self._segment_count > 0:
      contrib_text = trn("{} segment of your driving is in the training dataset so far.",
                         "{} segments of your driving is in the training dataset so far.", self._segment_count).format(self._segment_count)
      y = self._draw_wrapped_text(x, y, w, contrib_text, gui_app.font(FontWeight.BOLD), 42, rl.WHITE)
      y += 20

    # Separator
    rl.draw_rectangle(x, y, w, 2, self.GRAY)
    y += 20

    # Instructions intro
    y = self._draw_wrapped_text(x, y, w, tr(INSTRUCTIONS_INTRO), gui_app.font(FontWeight.ROMAN), 32, self.LIGHT_GRAY)
    y += 20

    # FAQ Header
    y = self._draw_wrapped_text(x, y, w, tr(FAQ_HEADER), gui_app.font(FontWeight.BOLD), 44, rl.WHITE)
    y += 20

    # FAQ Items
    for question, answer in FAQ_ITEMS:
      y = self._draw_wrapped_text(x, y, w, tr(question), gui_app.font(FontWeight.BOLD), 32, self.LIGHT_GRAY)
      y = self._draw_wrapped_text(x, y, w, tr(answer), gui_app.font(FontWeight.ROMAN), 32, self.LIGHT_GRAY)
      y += 20

    # return value not used by NavWidget
    return -1

  def _draw_wrapped_text(self, x, y, width, text, font, font_size, color):
    wrapped = wrap_text(font, text, font_size, width)
    for line in wrapped:
      rl.draw_text_ex(font, line, rl.Vector2(x, y), font_size, 0, color)
      y += int(font_size * FONT_SCALE)
    return y

  def _measure_content_height(self, rect: rl.Rectangle) -> int:
    # Rough measurement using the same wrapping as rendering
    w = int(rect.width - 80)
    y = 40

    # Title
    title_size = 72
    y += int(title_size * FONT_SCALE) + 20

    # Description
    desc_lines = wrap_text(gui_app.font(FontWeight.ROMAN), tr(DESCRIPTION), 36, w)
    y += int(len(desc_lines) * 36 * FONT_SCALE) + 20

    # Separator + Status
    y += 2 + 20
    status_text, _ = self._get_status()
    status_lines = wrap_text(gui_app.font(FontWeight.BOLD), status_text, 48, w)
    y += int(len(status_lines) * 48 * FONT_SCALE) + 20

    # Contribution count
    if self._segment_count > 0:
      contrib_text = trn("{} segment of your driving is in the training dataset so far.",
                         "{} segments of your driving is in the training dataset so far.", self._segment_count).format(self._segment_count)
      contrib_lines = wrap_text(gui_app.font(FontWeight.BOLD), contrib_text, 42, w)
      y += int(len(contrib_lines) * 42 * FONT_SCALE) + 20

    # Separator + Instructions
    y += 2 + 20

    # Instructions intro
    intro_lines = wrap_text(gui_app.font(FontWeight.ROMAN), tr(INSTRUCTIONS_INTRO), 32, w)
    y += int(len(intro_lines) * 32 * FONT_SCALE) + 20

    # FAQ Header
    faq_header_lines = wrap_text(gui_app.font(FontWeight.BOLD), tr(FAQ_HEADER), 44, w)
    y += int(len(faq_header_lines) * 44 * FONT_SCALE) + 20

    # FAQ Items
    for question, answer in FAQ_ITEMS:
      q_lines = wrap_text(gui_app.font(FontWeight.BOLD), tr(question), 32, w)
      y += int(len(q_lines) * 32 * FONT_SCALE)
      a_lines = wrap_text(gui_app.font(FontWeight.ROMAN), tr(answer), 32, w)
      y += int(len(a_lines) * 32 * FONT_SCALE) + 20

    # bottom padding
    y += 40
    return y

  def _get_status(self) -> tuple[str, rl.Color]:
    network_type = ui_state.sm["deviceState"].networkType
    network_metered = ui_state.sm["deviceState"].networkMetered

    if not network_metered and network_type != 0:  # Not metered and connected
      return tr("ACTIVE"), self.GREEN
    else:
      return tr("INACTIVE: connect to an unmetered network"), self.RED

  def _fetch_firehose_stats(self):
    try:
      dongle_id = self._params.get("DongleId")
      if not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
        return
      identity_token = get_token(dongle_id)
      response = api_get(f"v1/devices/{dongle_id}/firehose_stats", access_token=identity_token, session=self._session)
      if response.status_code == 200:
        data = response.json()
        self._segment_count = data.get("firehose", 0)
        self._params.put(self.PARAM_KEY, data)
    except Exception as e:
      cloudlog.error(f"Failed to fetch firehose stats: {e}")

  def _update_loop(self):
    while self._running:
      if not ui_state.started and device._awake:
        self._fetch_firehose_stats()
      time.sleep(self.UPDATE_INTERVAL)


class FirehoseLayout(FirehoseLayoutBase, NavWidget):
  BACK_TOUCH_AREA_PERCENTAGE = 0.1

  def __init__(self, back_callback):
    super().__init__()
    self.set_back_callback(back_callback)
