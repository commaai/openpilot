import datetime
import os
import threading
import time

import pyray as rl

from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, trn
from openpilot.system.ui.widgets.html_render import HtmlRenderer
from openpilot.system.ui.widgets.scroller import NavRawScrollPanel, NavScroller

UPDATED_TIMEOUT = 10.0  # seconds to wait for updated to respond

STATE_TO_DISPLAY_TEXT = {
  "checking...": tr("checking..."),
  "downloading...": tr("downloading..."),
  "finalizing update...": tr("finalizing update..."),
}


def time_ago(date: datetime.datetime | None) -> str:
  if not date:
    return tr("never")

  if not system_time_valid():
    return date.strftime("%a %b %d %Y")

  now = datetime.datetime.now(datetime.UTC)
  if date.tzinfo is None:
    date = date.replace(tzinfo=datetime.UTC)

  diff_seconds = int((now - date).total_seconds())
  if diff_seconds < 60:
    return tr("now")
  if diff_seconds < 3600:
    minutes = diff_seconds // 60
    return trn("{} minute ago", "{} minutes ago", minutes).format(minutes)
  if diff_seconds < 86400:
    hours = diff_seconds // 3600
    return trn("{} hour ago", "{} hours ago", hours).format(hours)
  if diff_seconds < 604800:
    days = diff_seconds // 86400
    return trn("{} day ago", "{} days ago", days).format(days)
  return date.strftime("%a %b %d %Y")


def _get_release_notes(param_key: str) -> str:
  notes = (ui_state.params.get(param_key) or b"").decode("utf-8", "replace").strip()
  if not notes:
    return f"<h2>{tr('No release notes available.')}</h2>"
  return notes


def _ordered_branches() -> list[str]:
  current_git_branch = ui_state.params.get("GitBranch") or ""
  branches_str = ui_state.params.get("UpdaterAvailableBranches") or ""
  branches = [b for b in branches_str.split(",") if b]

  for branch in [current_git_branch, "devel-staging", "devel", "nightly", "nightly-dev", "master"]:
    if branch in branches:
      branches.remove(branch)
      branches.insert(0, branch)

  return branches


class SoftwareReleaseNotesMici(NavRawScrollPanel):
  def __init__(self, title: str, description: str, release_notes: str):
    super().__init__()
    self._header = GreyBigButton(title, description)
    self._content = HtmlRenderer(text=release_notes)

  def _render(self, rect: rl.Rectangle):
    content_height = 20 + self._header.rect.height + 20 + self._content.get_total_height(int(rect.width))
    scroll_content_rect = rl.Rectangle(rect.x, rect.y, rect.width, content_height)
    scroll_offset = round(self._scroll_panel.update(rect, scroll_content_rect.height))

    header_rect = rl.Rectangle(
      rect.x + (rect.width - self._header.rect.width) / 2,
      rect.y + 20 + scroll_offset,
      self._header.rect.width,
      self._header.rect.height,
    )
    self._header.render(header_rect)

    body_rect = rl.Rectangle(rect.x, header_rect.y + header_rect.height + 20, rect.width, content_height)
    self._content.render(body_rect)


class UpdateDownloadBigButton(BigButton):
  def __init__(self):
    self._txt_update_icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    super().__init__("download", "", self._txt_update_icon)

    self._waiting_for_updater = False
    self._waiting_start_ts = 0.0

  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)

    self.set_enabled(False)
    self._waiting_for_updater = True
    self._waiting_start_ts = time.monotonic()

    def run():
      if ui_state.params.get_bool("UpdaterFetchAvailable"):
        os.system("pkill -SIGHUP -f system.updated.updated")
      else:
        os.system("pkill -SIGUSR1 -f system.updated.updated")

    threading.Thread(target=run, daemon=True).start()

  def _update_state(self):
    super()._update_state()

    updater_state = ui_state.params.get("UpdaterState") or "idle"
    failed_count = int(ui_state.params.get("UpdateFailedCount") or 0)
    fetch_available = ui_state.params.get_bool("UpdaterFetchAvailable")

    if updater_state != "idle":
      self._waiting_for_updater = False
      self.set_enabled(False)
      self.set_rotate_icon(True)
      self.set_value(STATE_TO_DISPLAY_TEXT.get(updater_state, updater_state))
      return

    self.set_rotate_icon(False)

    if self._waiting_for_updater and time.monotonic() - self._waiting_start_ts > UPDATED_TIMEOUT:
      self._waiting_for_updater = False

    self.set_enabled(ui_state.is_offroad() and not self._waiting_for_updater)

    if failed_count > 0:
      self.set_value(tr("failed to check for update"))
    elif fetch_available:
      self.set_value(tr("update available"))
    else:
      last_update = ui_state.params.get("LastUpdateTime")
      if last_update:
        self.set_value(tr("up to date, last checked {}").format(time_ago(last_update)))
      else:
        self.set_value(tr("up to date, last checked never"))


class BranchPickerMici(NavScroller):
  def __init__(self, current_target: str, branches: list[str]):
    super().__init__()

    for branch in branches:
      branch_btn = BigButton(branch, icon=None, scroll=True)
      branch_btn.set_enabled(branch != current_target)
      branch_btn.set_click_callback(lambda b=branch: self._on_select_branch(b))
      self._scroller.add_widget(branch_btn)

  def _on_select_branch(self, branch: str):
    ui_state.params.put("UpdaterTargetBranch", branch)
    os.system("pkill -SIGUSR1 -f system.updated.updated")
    self.dismiss()


class SoftwareLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    info_icon = gui_app.texture("icons_mici/settings/device/info.png", 64, 64)
    update_icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    reboot_icon = gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70)
    uninstall_icon = gui_app.texture("icons_mici/settings/device/uninstall.png", 64, 64)

    self._onroad_card = GreyBigButton("", tr("Updates are only downloaded while the car is off."))
    self._onroad_card.set_visible(ui_state.is_onroad)

    self._current_version_btn = BigButton("current\nversion", "", info_icon, scroll=True)
    self._current_version_btn.set_click_callback(lambda: self._show_release_notes(
      tr("Current Version"),
      ui_state.params.get("UpdaterCurrentDescription") or "",
      _get_release_notes("UpdaterCurrentReleaseNotes"),
    ))

    self._download_btn = UpdateDownloadBigButton()
    self._download_btn.set_visible(ui_state.is_offroad)

    self._new_version_btn = BigButton("new\nversion", "", info_icon, scroll=True)
    self._new_version_btn.set_click_callback(lambda: self._show_release_notes(
      tr("Install Update"),
      ui_state.params.get("UpdaterNewDescription") or "",
      _get_release_notes("UpdaterNewReleaseNotes"),
    ))
    self._new_version_btn.set_visible(False)

    self._install_btn = BigButton("install\nupdate", "", reboot_icon, scroll=True)
    self._install_btn.set_click_callback(self._install_update)
    self._install_btn.set_visible(False)

    self._branch_btn = BigButton("target\nbranch", "", update_icon, scroll=True)
    self._branch_btn.set_click_callback(self._show_branch_picker)
    self._branch_btn.set_enabled(lambda: len(_ordered_branches()) > 0)
    self._branch_btn.set_visible(lambda: not ui_state.params.get_bool("IsTestedBranch"))

    self._uninstall_btn = BigButton("uninstall\nopenpilot", "", uninstall_icon)
    self._uninstall_btn.set_click_callback(self._uninstall)

    self._scroller.add_widgets([
      self._onroad_card,
      self._current_version_btn,
      self._download_btn,
      self._new_version_btn,
      self._install_btn,
      self._branch_btn,
      self._uninstall_btn,
    ])

  def _update_state(self):
    super()._update_state()

    update_available = ui_state.params.get_bool("UpdateAvailable")

    self._current_version_btn.set_value(ui_state.params.get("UpdaterCurrentDescription") or "")
    self._new_version_btn.set_value(ui_state.params.get("UpdaterNewDescription") or "")
    self._new_version_btn.set_visible(update_available)

    self._install_btn.set_value(ui_state.params.get("UpdaterNewDescription") or "")
    self._install_btn.set_visible(ui_state.is_offroad() and update_available)
    self._install_btn.set_enabled(ui_state.is_offroad() and update_available)

    self._branch_btn.set_value(ui_state.params.get("UpdaterTargetBranch") or "")

  def _show_release_notes(self, title: str, description: str, release_notes: str):
    gui_app.push_widget(SoftwareReleaseNotesMici(title, description, release_notes))

  def _show_branch_picker(self):
    gui_app.push_widget(BranchPickerMici(ui_state.params.get("UpdaterTargetBranch") or "", _ordered_branches()))

  def _install_update(self):
    self._install_btn.set_enabled(False)
    ui_state.params.put_bool("DoReboot", True)

  def _uninstall(self):
    if ui_state.engaged:
      gui_app.push_widget(BigDialog("", tr("Disengage to Uninstall")))
      return

    gui_app.push_widget(BigConfirmationDialog(
      "slide to\nuninstall",
      gui_app.texture("icons_mici/settings/device/uninstall.png", 64, 64),
      lambda: ui_state.params.put_bool("DoUninstall", True),
      exit_on_confirm=False,
      red=True,
    ))
