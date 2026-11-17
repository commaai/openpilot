"""
DashC settings panel.

Place at: openpilot/selfdrive/ui/layouts/settings/dashcam.py

Shows up as "Dashcam" in the settings menu:
  - Recording Quality: Standard / High / Max (applies from the next drive)
  - Upload on Mobile Hotspot: allow uploads over a metered connection
  - Google Drive: signed in or not
  - Free Storage on the device
"""
import os

from openpilot.common.params import Params
from openpilot.system.loggerd.config import get_available_bytes
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import multiple_button_item, text_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller

TOKEN_PATH = "/data/dashcam/google_token.json"

QUALITY_DESCRIPTION = (
  "Video bitrate for all three cameras. Standard is about 11 GB per hour of driving, "
  "High about 22 GB, Max about 34 GB. Higher quality fills storage faster and takes longer to upload. "
  "Changes apply from the next drive."
)
METERED_DESCRIPTION = (
  "Upload to Google Drive over a phone hotspot or other metered connection. "
  "Recordings are large, so this uses a lot of mobile data."
)


class DashcamLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    self._quality = multiple_button_item(
      lambda: "Recording Quality",
      lambda: QUALITY_DESCRIPTION,
      buttons=[lambda: "Standard", lambda: "High", lambda: "Max"],
      button_width=255,
      callback=self._set_quality,
      selected_index=self._get_quality(),
    )

    self._metered = toggle_item(
      lambda: "Upload on Mobile Hotspot",
      lambda: METERED_DESCRIPTION,
      self._params.get_bool("DashcamAllowMetered"),
      callback=self._set_metered,
    )

    self._drive = text_item(lambda: "Google Drive", lambda: self._drive_status())
    self._storage = text_item(lambda: "Free Storage", lambda: self._storage_status())

    self._scroller = Scroller([self._quality, self._metered, self._drive, self._storage],
                              line_separator=True, spacing=0)

  # ---- helpers -----------------------------------------------------------
  def _get_quality(self) -> int:
    try:
      return max(0, min(2, int(self._params.get("DashcamQuality", return_default=True))))
    except (TypeError, ValueError):
      return 1

  @staticmethod
  def _drive_status() -> str:
    return "Signed in" if os.path.exists(TOKEN_PATH) else "Not signed in"

  @staticmethod
  def _storage_status() -> str:
    return f"{get_available_bytes(default=0) / 1e9:.0f} GB"

  # ---- callbacks ---------------------------------------------------------
  def _set_quality(self, button_index: int):
    self._params.put("DashcamQuality", button_index, block=True)

  def _set_metered(self, state: bool):
    self._params.put_bool("DashcamAllowMetered", state, block=True)

  # ---- widget ------------------------------------------------------------
  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._quality.action_item.set_selected_button(self._get_quality())
    self._metered.action_item.set_state(self._params.get_bool("DashcamAllowMetered"))

  def _render(self, rect):
    self._scroller.render(rect)
