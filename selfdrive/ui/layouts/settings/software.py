import os
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item, text_item, ListItem
from openpilot.system.ui.widgets.scroller import Scroller


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()

    # Create onroad warning label
    self._onroad_label = ListItem(title="Updates are only downloaded while the car is off.")

    # Create text item for current version
    self._version_item = text_item("Current Version", ui_state.params.get("UpdaterCurrentDescription") or "Unknown")

    # Create download button with initial state
    self._download_btn = button_item("Download", "CHECK", callback=self._on_download_update)

    # Create install button (initially hidden)
    self._install_btn = button_item("Install Update", "INSTALL", callback=self._on_install_update)
    self._install_btn.set_visible(False)

    items = self._init_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _init_items(self):
    items = [
      self._onroad_label,
      self._version_item,
      self._download_btn,
      self._install_btn,
      button_item("Uninstall", "UNINSTALL", callback=self._on_uninstall),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def _update_state(self):
    # Show/hide onroad warning
    self._onroad_label.set_visible(ui_state.is_onroad())

    # Update current version and release notes
    current_desc = ui_state.params.get("UpdaterCurrentDescription") or "Unknown"
    current_release_notes = (ui_state.params.get("UpdaterCurrentReleaseNotes") or b"").decode("utf-8", "replace")
    self._version_item.action_item.set_text(current_desc)
    self._version_item.description = current_release_notes

    # Update download button visibility and state
    self._download_btn.set_visible(ui_state.is_offroad())

    updater_state = ui_state.params.get("UpdaterState") or "idle"
    failed_count = ui_state.params.get("UpdateFailedCount") or 0
    fetch_available = ui_state.params.get_bool("UpdaterFetchAvailable")
    update_available = ui_state.params.get_bool("UpdateAvailable")

    if updater_state != "idle":
      self._download_btn.set_enabled(False)
      self._download_btn.action_item.set_value(updater_state)
    else:
      if failed_count > 0:
        self._download_btn.action_item.set_value("failed to check for update")
        self._download_btn.action_item.set_text("CHECK")
      elif fetch_available:
        self._download_btn.action_item.set_value("update available")
        self._download_btn.action_item.set_text("DOWNLOAD")
      else:
        last_update = ui_state.params.get("LastUpdateTime")
        if last_update:
          # TODO: Format time ago like Qt does
          self._download_btn.action_item.set_value(f"up to date, last checked {last_update}")
        else:
          self._download_btn.action_item.set_value("up to date, last checked never")
        self._download_btn.action_item.set_text("CHECK")
      self._download_btn.set_enabled(True)

    # Update install button
    self._install_btn.set_visible(ui_state.is_offroad() and update_available)
    if update_available:
      new_desc = ui_state.params.get("UpdaterNewDescription") or ""
      new_release_notes = (ui_state.params.get("UpdaterNewReleaseNotes") or b"").decode("utf-8", "replace")
      self._install_btn.action_item.set_text("INSTALL")
      self._install_btn.action_item.set_value(new_desc)
      self._install_btn.description = new_release_notes
      # Enable install button for testing (like Qt showEvent)
      self._install_btn.set_enabled(True)
    else:
      self._install_btn.set_visible(False)

  def _on_download_update(self):
    # Check if we should start checking or start downloading
    if self._download_btn.action_item.text == "CHECK":
      print('Starting update check...')
      # Start checking for updates
      os.system("pkill -SIGUSR1 -f system.updated.updated")
    else:
      # Start downloading
      print('Starting update download...')
      os.system("pkill -SIGHUP -f system.updated.updated")

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result):
      if result == DialogResult.CONFIRM:
        ui_state.params.put_bool("DoUninstall", True)

    dialog = ConfirmDialog("Are you sure you want to uninstall?", "Uninstall")
    gui_app.set_modal_overlay(dialog, callback=handle_uninstall_confirmation)

  def _on_install_update(self):
    # Trigger reboot to install update
    ui_state.params.put_bool("DoReboot", True)

