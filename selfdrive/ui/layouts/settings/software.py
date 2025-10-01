import os
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item, text_item
from openpilot.system.ui.widgets.scroller import Scroller


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._init_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _init_items(self):
    self._current_version_item = text_item("Current Version", lambda: self._params.get("UpdaterCurrentDescription") or "")

    self._download_item = button_item(
      "Download",
      self._download_button_text,
      description=self._download_button_description,
      callback=self._on_download_button_clicked,
      enabled=self._download_button_enabled,
    )

    self._install_item = button_item("Install Update", "INSTALL", callback=self._on_install_update)

    items = [
      self._current_version_item,
      self._download_item,
      self._install_item,
      button_item("Target Branch", "SELECT", callback=self._on_select_branch),
      button_item("Uninstall", "UNINSTALL", callback=self._on_uninstall),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  # --- Download/Check logic (mirrors Qt SoftwarePanel::checkForUpdates + updateLabels) ---
  def _download_button_enabled(self) -> bool:
    if not ui_state.is_offroad():
      return False
    updater_state = (self._params.get("UpdaterState") or "").strip()
    return updater_state == "idle"

  def _download_button_text(self) -> str:
    updater_state = (self._params.get("UpdaterState") or "").strip()
    if updater_state != "idle":
      # Button text stays as last actionable state; show state in description
      # Default to CHECK when idle
      fetch_available = self._params.get_bool("UpdaterFetchAvailable")
      return "DOWNLOAD" if fetch_available else "CHECK"
    # idle: decide between CHECK or DOWNLOAD
    return "DOWNLOAD" if self._params.get_bool("UpdaterFetchAvailable") else "CHECK"

  def _download_button_description(self) -> str:
    updater_state = (self._params.get("UpdaterState") or "").strip()
    if updater_state != "idle":
      return updater_state

    failed = False
    try:
      failed = int(self._params.get("UpdateFailedCount") or "0") > 0
    except Exception:
      failed = False

    if failed:
      return "failed to check for update"

    if self._params.get_bool("UpdaterFetchAvailable"):
      return "update available"

    last = self._params.get("LastUpdateTime") or ""
    # Keep it simple; Qt shows relative time, we can show the raw timestamp
    return f"up to date, last checked {last or 'never'}"

  def _on_download_button_clicked(self):
    text = self._download_button_text()
    try:
      if text == "CHECK":
        os.system("pkill -SIGUSR1 -f system.updated.updated")
      elif text == "DOWNLOAD":
        os.system("pkill -SIGHUP -f system.updated.updated")
    except Exception:
      pass

  # ---

  def _on_install_update(self):
    # Mirrors Qt: set DoReboot to start install flow handled by updater
    self._params.put_bool("DoReboot", True)
  def _on_install_update(self): pass
  def _on_select_branch(self): pass

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result):
      if result == DialogResult.CONFIRM:
        self._params.put_bool("DoUninstall", True)

    dialog = ConfirmDialog("Are you sure you want to uninstall?", "Uninstall")
    gui_app.set_modal_overlay(dialog, callback=handle_uninstall_confirmation)
