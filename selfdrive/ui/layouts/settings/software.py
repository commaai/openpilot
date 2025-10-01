import os
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item, TextAction, ListItem
from openpilot.system.ui.widgets.scroller import Scroller


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()

    # Create text item for current version
    version_action = TextAction(ui_state.params.get("UpdaterCurrentDescription", "Unknown"))
    self._version_item = ListItem(title="Current Version", action_item=version_action)

    # Create download button with initial state
    self._download_btn = button_item("Download", "CHECK", callback=self._on_download_update)

    # Create install button (initially hidden)
    self._install_btn = button_item("Install Update", "INSTALL", callback=self._on_install_update)
    self._install_btn.set_visible(False)

    items = self._init_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _init_items(self):
    items = [
      self._version_item,
      self._download_btn,
      self._install_btn,
      button_item("Target Branch", "SELECT", callback=self._on_select_branch),
      button_item("Uninstall", "UNINSTALL", callback=self._on_uninstall),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def _update_state(self):
    # Update current version
    current_desc = ui_state.params.get("UpdaterCurrentDescription", "Unknown")
    self._version_item.action_item.set_text(current_desc)

    # Update download button state
    updater_state = ui_state.params.get("UpdaterState", "idle")
    failed_count = int(ui_state.params.get("UpdateFailedCount", "0"))
    fetch_available = ui_state.params.get_bool("UpdaterFetchAvailable")
    update_available = ui_state.params.get_bool("UpdateAvailable")

    if updater_state != "idle":
      self._download_btn.set_enabled(False)
      self._download_btn.description = updater_state
    else:
      if failed_count > 0:
        self._download_btn.description = "failed to check for update"
        self._download_btn.action_item._text_source = "CHECK"
      elif fetch_available:
        self._download_btn.description = "update available"
        self._download_btn.action_item._text_source = "DOWNLOAD"
      else:
        last_update = ui_state.params.get("LastUpdateTime", "")
        if last_update:
          # TODO: Format time ago
          self._download_btn.description = f"up to date, last checked {last_update}"
        else:
          self._download_btn.description = "up to date, last checked never"
        self._download_btn.action_item._text_source = "CHECK"
      self._download_btn.set_enabled(True)

    # Update install button
    if update_available:
      new_desc = ui_state.params.get("UpdaterNewDescription", "")
      self._install_btn.description = new_desc
      self._install_btn.set_visible(True)
    else:
      self._install_btn.set_visible(False)

  def _on_download_update(self):
    # Check if we should start checking or stop downloading
    if self._download_btn.action_item.text == "CHECK":
      # Start checking for updates
      os.system("pkill -SIGUSR1 -f system.updated.updated")
    else:
      # Stop downloading
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

  def _on_select_branch(self):
    pass
