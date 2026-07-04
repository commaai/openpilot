import subprocess
import threading
import pyray as rl
from enum import IntEnum
from collections.abc import Callable

from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.mici.layouts.settings.device import EngagedConfirmationButton
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller

UPDATER_TIMEOUT = 10.0  # seconds to wait for updater to respond


def _split_description(desc: str) -> tuple[str, str, str, str] | None:
  # UpdaterCurrentDescription/UpdaterNewDescription format: "version / branch / commit / date"
  parts = [p.strip() for p in desc.split(" / ")]
  if len(parts) != 4:
    return None
  version, branch, commit, date = parts
  return version, branch, commit, date


class UpdaterState(IntEnum):
  IDLE = 0
  WAITING_FOR_UPDATER = 1
  UPDATER_RESPONDING = 2


class SoftwareInfoLayoutMici(Widget):
  def __init__(self):
    super().__init__()

    self.set_rect(rl.Rectangle(0, 0, 360, 180))

    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    self._version_label = UnifiedLabel("version", 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self._version_text_label = UnifiedLabel("", 32, max_width=max_width, text_color=subheader_color,
                                            font_weight=FontWeight.ROMAN, wrap_text=False)

    self._branch_label = UnifiedLabel("branch", 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self._branch_text_label = UnifiedLabel("", 32, max_width=max_width, text_color=subheader_color,
                                           font_weight=FontWeight.ROMAN, wrap_text=False)

  def _update_state(self):
    desc = _split_description(ui_state.params.get("UpdaterCurrentDescription") or "")
    if desc is not None:
      version, branch, commit, date = desc
      self._version_text_label.set_text(f"{version} ({date})")
      self._branch_text_label.set_text(f"{branch} ({commit})")
    else:
      self._version_text_label.set_text(ui_state.params.get("Version") or "N/A")
      self._branch_text_label.set_text(ui_state.params.get("GitBranch") or "N/A")

  def _render(self, _):
    self._version_label.set_position(self._rect.x + 20, self._rect.y - 10)
    self._version_label.render()

    self._version_text_label.set_position(self._rect.x + 20, self._rect.y + 68 - 25)
    self._version_text_label.render()

    self._branch_label.set_position(self._rect.x + 20, self._rect.y + 114 - 30)
    self._branch_label.render()

    self._branch_text_label.set_position(self._rect.x + 20, self._rect.y + 161 - 25)
    self._branch_text_label.render()


class CheckUpdateButton(BigButton):
  def __init__(self):
    self._txt_update_icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    self._txt_up_to_date_icon = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)
    super().__init__("check for update", "", self._txt_update_icon)

    self._waiting_for_updater_t: float | None = None
    self._hide_value_t: float | None = None
    self._state: UpdaterState = UpdaterState.IDLE

    ui_state.add_offroad_transition_callback(self.offroad_transition)

  def offroad_transition(self):
    if ui_state.is_offroad():
      self.set_enabled(True)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    if not system_time_valid():
      dlg = BigDialog("", tr("Please connect to Wi-Fi to update."))
      gui_app.push_widget(dlg)
      return

    self.set_enabled(False)
    self._state = UpdaterState.WAITING_FOR_UPDATER
    self.set_icon(self._txt_update_icon)

    def run():
      if self.get_value() == "download update":
        subprocess.run("pkill -SIGHUP -f openpilot.system.updated.updated", shell=True)
      else:
        subprocess.run("pkill -SIGUSR1 -f openpilot.system.updated.updated", shell=True)

    threading.Thread(target=run, daemon=True).start()

  def set_value(self, value: str):
    super().set_value(value)
    if value:
      self.set_text("")
    else:
      self.set_text("check for update")

  def _update_state(self):
    super()._update_state()

    if ui_state.started:
      self.set_enabled(False)
      return

    updater_state = ui_state.params.get("UpdaterState") or ""
    failed_count = ui_state.params.get("UpdateFailedCount") or 0
    failed = int(failed_count) > 0

    if self._state == UpdaterState.WAITING_FOR_UPDATER:
      self.set_rotate_icon(True)
      if updater_state != "idle":
        self._state = UpdaterState.UPDATER_RESPONDING

      # Recover from updater not responding (time invalid shortly after boot)
      if self._waiting_for_updater_t is None:
        self._waiting_for_updater_t = rl.get_time()

      if self._waiting_for_updater_t is not None and rl.get_time() - self._waiting_for_updater_t > UPDATER_TIMEOUT:
        self.set_rotate_icon(False)
        self.set_value("updater failed\nto respond")
        self._state = UpdaterState.IDLE
        self._hide_value_t = rl.get_time()

    elif self._state == UpdaterState.UPDATER_RESPONDING:
      if updater_state == "idle":
        self.set_rotate_icon(False)
        self._state = UpdaterState.IDLE
        self._hide_value_t = rl.get_time()
      else:
        if self.get_value() != updater_state:
          self.set_value(updater_state)

    elif self._state == UpdaterState.IDLE:
      self.set_rotate_icon(False)
      if failed:
        self.set_enabled(True)  # allow retry when failure came from updater param
        if self.get_value() != "failed to update":
          self.set_value("failed to update")

      elif ui_state.params.get_bool("UpdaterFetchAvailable"):
        self.set_enabled(True)
        if self.get_value() != "download update":
          self.set_value("download update")

      elif self._hide_value_t is not None:
        self.set_enabled(True)
        if self.get_value() == "checking...":
          self.set_value("up to date")
          self.set_icon(self._txt_up_to_date_icon)

        # Hide previous text after short amount of time (up to date or failed)
        if rl.get_time() - self._hide_value_t > 3.0:
          self._hide_value_t = None
          self.set_value("")
          self.set_icon(self._txt_update_icon)
      else:
        if self.get_value() != "":
          self.set_value("")

    if self._state != UpdaterState.WAITING_FOR_UPDATER:
      self._waiting_for_updater_t = None


class InstallUpdateButton(BigButton):
  def __init__(self):
    super().__init__("install update", "", gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70))
    self.set_visible(lambda: ui_state.is_offroad() and ui_state.params.get_bool("UpdateAvailable"))

  def _update_state(self):
    super()._update_state()

    desc = _split_description(ui_state.params.get("UpdaterNewDescription") or "")
    value = f"{desc[0]} ({desc[1]})" if desc is not None else ""
    if self.get_value() != value:
      self.set_value(value)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    self.set_enabled(False)

    def run():
      ui_state.params.put_bool("DoReboot", True, block=True)

    threading.Thread(target=run, daemon=True).start()


class BranchSelectPage(NavScroller):
  def __init__(self, on_select: Callable[[str], None]):
    super().__init__()

    params = ui_state.params
    current_git_branch = params.get("GitBranch") or ""
    branches_str = params.get("UpdaterAvailableBranches") or ""
    branches = [b for b in branches_str.split(",") if b]

    for b in [current_git_branch, "devel-staging", "devel", "nightly", "nightly-dev", "master"]:
      if b in branches:
        branches.remove(b)
        branches.insert(0, b)

    current_target = params.get("UpdaterTargetBranch") or ""
    check_icon = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)

    buttons = []
    for branch in branches:
      btn = BigButton(branch, "", check_icon if branch == current_target else None, scroll=True)
      btn.set_click_callback(lambda b=branch: self.dismiss(lambda: on_select(b)))
      buttons.append(btn)
    self._scroller.add_widgets(buttons)


class TargetBranchButton(BigButton):
  def __init__(self):
    super().__init__("target branch", ui_state.params.get("UpdaterTargetBranch") or "")
    self.set_click_callback(self._on_click)
    self.set_visible(not ui_state.params.get_bool("IsTestedBranch"))
    self.set_enabled(lambda: ui_state.is_offroad())

  def _update_state(self):
    super()._update_state()

    target = ui_state.params.get("UpdaterTargetBranch") or ""
    if self.get_value() != target:
      self.set_value(target)

  def _on_click(self):
    gui_app.push_widget(BranchSelectPage(self._on_select))

  def _on_select(self, branch: str):
    ui_state.params.put("UpdaterTargetBranch", branch, block=True)
    self.set_value(branch)
    subprocess.run("pkill -SIGUSR1 -f openpilot.system.updated.updated", shell=True)


class SoftwareLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    def uninstall_openpilot_callback():
      ui_state.params.put_bool("DoUninstall", True, block=True)

    uninstall_openpilot_btn = EngagedConfirmationButton("uninstall openpilot", "uninstall",
                                                        gui_app.texture("icons_mici/settings/device/uninstall.png", 64, 64),
                                                        uninstall_openpilot_callback, exit_on_confirm=False)

    self._scroller.add_widgets([
      SoftwareInfoLayoutMici(),
      CheckUpdateButton(),
      InstallUpdateButton(),
      TargetBranchButton(),
      uninstall_openpilot_btn,
    ])
