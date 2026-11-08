import requests
import threading
import time
import raylib_nb as rl

from openpilot.common.api import api_get
from openpilot.common.params import Params
from openpilot.common.realtime import drop_realtime
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.lib.api_helpers import get_token
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID
from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, trn, tr_noop
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import NavScroller

CARDS = [
  tr_noop("openpilot learns to drive by watching humans, like you, drive."),
  tr_noop("Firehose Mode maximizes your training data uploads to improve openpilot."),
  tr_noop("More data means bigger models, which means better driving."),
  tr_noop("For maximum effectiveness, bring your device inside weekly."),
  tr_noop("Connect it to a good USB-C adapter and Wi-Fi."),
  tr_noop("Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card."),
]
FAQ_HEADER = tr_noop("frequently asked\nquestions")
FAQ_ITEMS = [
  tr_noop("Does it matter how or where I drive? Nope, just drive as you normally would."),
  tr_noop("Do all of my segments get pulled? No, we selectively pull a subset of your segments."),
  tr_noop("What's a good USB-C adapter? Any fast phone or laptop charger should be fine."),
  tr_noop("Does it matter which software I run? Yes, only upstream openpilot (and some forks) can be used."),
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

  def _get_segment_count(self) -> int:
    stats = self._params.get(self.PARAM_KEY)
    if not stats:
      return 0
    try:
      return int(stats.get("firehose", 0))
    except Exception:
      cloudlog.exception(f"Failed to decode firehose stats: {stats}")
      return 0

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
    drop_realtime()
    while self._running:
      if not ui_state.started and device._awake:
        self._fetch_firehose_stats()
      time.sleep(self.UPDATE_INTERVAL)


class FirehoseLayout(NavScroller, FirehoseLayoutBase):
  def __init__(self):
    super().__init__()

    self._status_card = GreyBigButton("firehose mode", "", gui_app.texture("icons_mici/settings/firehose.png", 52, 62))
    self._segments_card = GreyBigButton("", "")
    self._segments_card.set_visible(lambda: self._segment_count > 0)

    self._scroller.add_widgets([
      self._status_card,
      self._segments_card,
      *[GreyBigButton("", tr(text)) for text in CARDS],
      GreyBigButton(tr(FAQ_HEADER), "", gui_app.texture("icons_mici/setup/green_info.png", 64, 64)),
      *[GreyBigButton("", tr(text)) for text in FAQ_ITEMS],
    ])

  def _update_state(self):
    super()._update_state()

    status_text, status_color = self._get_status()
    if self._status_card.get_value() != status_text:
      self._status_card.set_value(status_text)
      self._status_card._sub_label.set_text_color(status_color)

    segments_text = trn("{} segment of your driving is in the training dataset so far.",
                        "{} segments of your driving is in the training dataset so far.", self._segment_count).format(self._segment_count)
    if self._segments_card.get_value() != segments_text:
      self._segments_card.set_value(segments_text)
