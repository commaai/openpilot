import re
import subprocess
import threading
import raylib_nb as rl
from collections.abc import Callable

from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.mici.layouts.settings.device.device_layout import EngagedConfirmationButton
from openpilot.selfdrive.ui.mici.widgets.info import InfoLayoutMici
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigPillButton, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigConfirmationDialog
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, MousePos
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import NavScroller

UPDATER_TIMEOUT = 30.0  # seconds to wait for updater to respond, retries rebuild the staging area first


def _split_description(desc: str) -> tuple[str, str, str, str] | None:
  # UpdaterCurrentDescription/UpdaterNewDescription format: "version / branch / commit / date"
  parts = [p.strip() for p in desc.split(" / ")]
  if len(parts) != 4:
    return None
  version, branch, commit, date = parts
  return version, branch, commit, date


def _parse_updater_state(state: str) -> tuple[str, float | None]:
  # updated writes "<stage>..." or "<stage>... <detail>", where detail may start with a percent
  stage, sep, detail = state.partition("... ")
  if not sep:
    return state, None
  m = re.match(r"(\d+)%", detail)
  return f"{stage}\n{detail}", int(m[1]) / 100 if m else None


class SoftwareInfoLayoutMici(InfoLayoutMici):
  def __init__(self):
    super().__init__("version", "", "branch", "")

  def _update_state(self):
    super()._update_state()

    desc = _split_description(ui_state.params.get("UpdaterCurrentDescription") or "")
    if desc is not None:
      version, branch, commit, date = desc
      self.subtext1.set_text(f"{version} ({date})")
      self.subtext2.set_text(f"{branch} ({commit})")
    else:
      self.subtext1.set_text(ui_state.params.get("Version") or "N/A")
      self.subtext2.set_text(ui_state.params.get("GitBranch") or "N/A")


class CheckUpdateButton(BigButton):
  UPDATER_PROC = "openpilot.system.updated.updated"
  CHECK_FOR_UPDATE = "SIGUSR1"
  DOWNLOAD_UPDATE = "SIGHUP"
  RESULT_DURATION = 3.0  # seconds to show the result of a check

  def __init__(self):
    self._txt_update_icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    self._txt_up_to_date_icon = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)
    super().__init__("check for update", "", self._txt_update_icon)

    self._requested_t: float | None = None  # signaled the updater, waiting for it to start
    self._result_snapshot: tuple | None = None  # last result before the current check, None when not watching
    self._result: str = ""
    self._result_t = 0.0
    self._progress: float | None = None

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    if not system_time_valid():
      dlg = BigDialog("", tr("Please connect to Wi-Fi to update."))
      gui_app.push_widget(dlg)
      return

    self._signal_updater(self.DOWNLOAD_UPDATE if ui_state.params.get_bool("UpdaterFetchAvailable") else self.CHECK_FOR_UPDATE)

  def check_for_update(self):
    self._signal_updater(self.CHECK_FOR_UPDATE)

  @property
  def waiting_for_updater(self) -> bool:
    return self._requested_t is not None

  def _last_result(self) -> tuple:
    return ui_state.params.get("LastUpdateTime"), int(ui_state.params.get("UpdateFailedCount") or 0)

  def _check_finished(self) -> bool:
    # a finished check either succeeded or failed once more. the count also resets when updated restarts, which isn't a result
    assert self._result_snapshot is not None
    last_time, failed_count = self._last_result()
    return last_time != self._result_snapshot[0] or failed_count > self._result_snapshot[1]

  def _signal_updater(self, sig: str):
    self._requested_t = rl.get_time()
    self._result_snapshot = self._last_result()
    self._result = ""

    def run():
      subprocess.run(f"pkill -{sig} -f {self.UPDATER_PROC}", shell=True)

    threading.Thread(target=run, daemon=True).start()

  def _update_state(self):
    super()._update_state()

    params = ui_state.params
    updater_state = params.get("UpdaterState") or "idle"
    now = rl.get_time()

    if ui_state.started:
      # updated doesn't run onroad, forget about any check in progress
      self._requested_t = self._result_snapshot = None

    elif updater_state != "idle":
      self._requested_t = None
      if self._result_snapshot is None:
        self._result_snapshot = self._last_result()
      status, self._progress = _parse_updater_state(updater_state)
      self._show("", status, self._txt_update_icon, enabled=False, spinning=True)
      return
    self._progress = None

    failed = int(params.get("UpdateFailedCount") or 0) > 0
    if self._result_snapshot is not None and self._check_finished():
      self._requested_t = self._result_snapshot = None
      if not failed:
        self._result = "update ready" if params.get_bool("UpdateAvailable") else "up to date"
        self._result_t = now
    elif self._requested_t is not None:
      if now - self._requested_t < UPDATER_TIMEOUT:
        self._show(self.text or "check for update", "", self._txt_update_icon, enabled=False, spinning=True)
        return
      self._requested_t = self._result_snapshot = None
      self._result, self._result_t = "updater not responding", now
    elif self._result_snapshot is not None and not ui_state.started:
      self._result_snapshot = None  # went idle without a result, e.g. restarted

    enabled = not ui_state.started
    showing_result = self._result and now - self._result_t < self.RESULT_DURATION
    if showing_result and self._result == "updater not responding":
      self._show("check for update", self._result, self._txt_update_icon, enabled)
    elif params.get_bool("UpdaterFetchAvailable"):
      self._show("download update", "last attempt failed" if failed else "", self._txt_update_icon, enabled)
    elif failed:
      self._show("check for update", "last attempt failed", self._txt_update_icon, enabled)
    elif showing_result:
      self._show("check for update", self._result, self._txt_up_to_date_icon, enabled)
    else:
      self._show("check for update", "", self._txt_update_icon, enabled)

  def _show(self, text: str, value: str, icon: rl.Texture, enabled: bool, spinning: bool = False):
    if self.text != text:
      self.set_text(text)
    if self.value != value:
      self.set_value(value)
    self.set_icon(icon)
    self.set_enabled(enabled)
    self.set_rotate_icon(spinning)

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    if self._progress is not None:
      bar = rl.Rectangle(self._rect.x + self.LABEL_HORIZONTAL_PADDING, btn_y + self._rect.height - 16,
                         self._rect.width - self.LABEL_HORIZONTAL_PADDING * 2, 6)
      rl.draw_rectangle_rounded(bar, 1.0, 6, rl.Color(255, 255, 255, 40))
      bar.width *= self._progress
      if bar.width > 0:
        rl.draw_rectangle_rounded(bar, 1.0, 6, rl.Color(255, 255, 255, 200))


def _apply_update():
  threading.Thread(target=lambda: ui_state.params.put_bool("DoReboot", True, block=True), daemon=True).start()


class UpdatePage(NavScroller):
  """What the downloaded update is, a QR code to its release notes, and a slider to apply it"""
  RELEASE_NOTES_URL = "https://github.com/commaai/openpilot/blob/{}/RELEASES.md"

  def __init__(self):
    super().__init__()
    params = ui_state.params
    version, branch, commit, date = _split_description(params.get("UpdaterNewDescription") or "") or ("", "", "", "")

    cards: list[Widget] = [GreyBigButton(f"openpilot {version}" if version else "update ready", f"{branch}, {date}" if version else "",
                                         gui_app.texture("icons_mici/settings/device/update.png", 64, 75))]
    if commit:
      cards += [GreyBigButton("release notes", "swipe for QR code",
                              gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 56, flip_x=True)),
                QR(self.RELEASE_NOTES_URL.format(commit))]
    apply_button = BigPillButton("apply update", green=True)
    apply_button.set_click_callback(lambda: gui_app.push_widget(BigConfirmationDialog(
      "slide to\nupdate", gui_app.texture("icons_mici/settings/device/update.png", 64, 75), _apply_update, exit_on_confirm=False)))
    cards.append(apply_button)
    self._scroller.add_widgets(cards)


class UpdateButton(BigButton):
  """The downloaded update, opens UpdatePage"""

  def __init__(self):
    super().__init__("update", "", gui_app.texture("icons_mici/settings/device/update.png", 64, 75))
    self.set_visible(lambda: ui_state.is_offroad() and ui_state.params.get_bool("UpdateAvailable"))
    self.set_click_callback(lambda: gui_app.push_widget(UpdatePage()))

  def _update_state(self):
    super()._update_state()

    desc = _split_description(ui_state.params.get("UpdaterNewDescription") or "")
    value = f"openpilot {desc[0]}" if desc is not None else ""
    if self.get_value() != value:
      self.set_value(value)


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
  def __init__(self, check_update_btn: CheckUpdateButton):
    super().__init__("target branch", ui_state.params.get("UpdaterTargetBranch") or "",
                     description="Select the software branch to download on the next update check.")
    self._check_update_btn = check_update_btn
    self.set_click_callback(self._on_click)
    self.set_visible(not ui_state.params.get_bool("IsTestedBranch"))
    self.set_enabled(lambda: ui_state.is_offroad() and not check_update_btn.waiting_for_updater and
                     (ui_state.params.get("UpdaterState") or "idle") == "idle")

  def _update_state(self):
    super()._update_state()

    target = ui_state.params.get("UpdaterTargetBranch") or ""
    if self.get_value() != target:
      self.set_value(target)

  def _on_click(self):
    if not ui_state.params.get("UpdaterAvailableBranches"):
      gui_app.push_widget(BigDialog("", tr("Failed to get available branches. Ensure you're connected to the internet and try again.")))
      return
    gui_app.push_widget(BranchSelectPage(self._on_select))

  def _on_select(self, branch: str):
    ui_state.params.put("UpdaterTargetBranch", branch, block=True)
    self.set_value(branch)
    self._check_update_btn.check_for_update()


class SoftwareLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    def uninstall_openpilot_callback():
      ui_state.params.put_bool("DoUninstall", True, block=True)

    uninstall_openpilot_btn = EngagedConfirmationButton("uninstall openpilot", "uninstall",
                                                        gui_app.texture("icons_mici/settings/device/uninstall.png", 64, 64),
                                                        uninstall_openpilot_callback, exit_on_confirm=False,
                                                        description="Remove openpilot from this device.",
                                                        description_icon=gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))

    check_update_btn = CheckUpdateButton()
    self._scroller.add_widgets([
      SoftwareInfoLayoutMici(),
      UpdateButton(),
      check_update_btn,
      TargetBranchButton(check_update_btn),
      uninstall_openpilot_btn,
    ])
