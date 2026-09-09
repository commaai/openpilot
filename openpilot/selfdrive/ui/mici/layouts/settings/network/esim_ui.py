import threading

import numpy as np
import pyray as rl
from collections.abc import Callable
from openpilot.cereal import log
from openpilot.cereal.visionipc import VisionStreamType

from openpilot.common import qrcode
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.mici.onroad.cabin_camera_dialog import CabinCameraView
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.nav_widget import NavWidget

from openpilot.selfdrive.ui.mici.widgets.button import BigButton, LABEL_COLOR
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigInputDialog, BigConfirmationDialog
from openpilot.common.esim.base import Profile
from openpilot.common.esim.lpa import parse_lpa_activation_code
from openpilot.system.ui.lib.application import DEFAULT_TEXT_COLOR, FontWeight, MousePos, TextAlignment, gui_app
from openpilot.system.ui.lib.cellular_manager import CellularManager
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel, gui_label
from openpilot.system.ui.widgets.scroller import NavRawScrollPanel, NavScroller


class ProfileActionButton(Widget):
  SIZE = 68
  MARGIN = 10
  HORIZONTAL_MARGIN = 4

  def __init__(self, callback: Callable, delete: bool = False):
    super().__init__()
    self.set_click_callback(callback)
    self._delete = delete
    self._trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 25, 30) if delete else None

    self._bg_txt = gui_app.texture("icons_mici/buttons/button_circle.png", self.SIZE, self.SIZE)
    self._bg_pressed_txt = gui_app.texture("icons_mici/buttons/button_circle_pressed.png", self.SIZE, self.SIZE)
    self.set_rect(rl.Rectangle(0, 0, self.SIZE + self.HORIZONTAL_MARGIN * 2, self.SIZE + self.MARGIN * 2))

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture_ex(bg_txt, (self._rect.x + (self._rect.width - self._bg_txt.width) / 2,
                                self._rect.y + (self._rect.height - self._bg_txt.height) / 2), 0, 1.0, rl.WHITE)
    color = rl.Color(255, 105, 115, 255) if self._delete else DEFAULT_TEXT_COLOR
    if not self.enabled:
      color = rl.Color(color.r, color.g, color.b, 90)
    if self._trash_txt:
      rl.draw_texture_ex(self._trash_txt, (self._rect.x + (self._rect.width - self._trash_txt.width) / 2,
                                          self._rect.y + (self._rect.height - self._trash_txt.height) / 2), 0, 1.0, color)
    else:
      gui_label(self._rect, "Aa", 30, color=color, alignment=TextAlignment.CENTER)


class QRScannerDialog(NavWidget):
  SCAN_INTERVAL_S = 0.25
  INVALID_CODE_DURATION_S = 1.0

  def __init__(self, on_qr_detected: Callable[[str], None]):
    super().__init__()
    self._on_qr_detected = on_qr_detected
    self._camera_view = CabinCameraView("camerad", VisionStreamType.VISION_STREAM_CABIN)
    self._detected = False
    self._last_scan_time = 0.0
    self._invalid_code_until = 0.0
    self._scan_thread: threading.Thread | None = None
    self._scan_result: str | None = None
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  def show_event(self):
    super().show_event()
    ui_state.params.put_bool("DisableDriverCameraIR", True)
    ui_state.params.put_bool("IsDriverViewEnabled", True)

  def hide_event(self):
    super().hide_event()
    ui_state.params.put_bool("IsDriverViewEnabled", False)
    ui_state.params.put_bool("DisableDriverCameraIR", False)

  def __del__(self):
    self._camera_view.close()

  def _update_state(self):
    super()._update_state()

    now = rl.get_time()
    if self._detected or not self._camera_view.frame or now < self._invalid_code_until:
      return

    if self._scan_thread is not None:
      if self._scan_thread.is_alive():
        return
      self._scan_thread = None
      data = self._scan_result
      if data is not None:
        try:
          parse_lpa_activation_code(data)
        except ValueError:
          self._invalid_code_until = now + self.INVALID_CODE_DURATION_S
          self._last_scan_time = self._invalid_code_until
        else:
          self._detected = True
          self.dismiss(lambda: self._on_qr_detected(data))
        return

    if now - self._last_scan_time < self.SCAN_INTERVAL_S:
      return
    self._last_scan_time = now

    frame = self._camera_view.frame
    y = np.frombuffer(frame.data, dtype=np.uint8, count=frame.height * frame.stride).reshape(frame.height, frame.stride)
    gray = y[:, :frame.width].copy()  # the vision buffer is recycled under the scan thread
    self._scan_thread = threading.Thread(target=self._scan, args=(gray,), daemon=True)
    self._scan_thread.start()

  def _scan(self, gray: np.ndarray):
    self._scan_result = qrcode.decode(gray)

  def _render(self, rect):
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._camera_view._render(rect)

    if not self._camera_view.frame:
      gui_label(rect, tr("camera starting"), font_size=54, font_weight=FontWeight.BOLD,
                alignment=TextAlignment.CENTER)
    else:
      label_y = rect.y + rect.height * 3 / 4
      label_rect = rl.Rectangle(rect.x, label_y + (rect.height - label_y) / 2 - 20, rect.width, 40)
      text = "not an LPA code" if rl.get_time() < self._invalid_code_until else "hold QR code to camera"
      gui_label(label_rect, text, font_size=32, font_weight=FontWeight.MEDIUM,
                alignment=TextAlignment.CENTER,
                color=rl.Color(255, 255, 255, int(255 * 0.9)))

    rl.end_scissor_mode()


class InstallingProfileDialog(BigDialog):
  DOT_STEP = 0.6

  def __init__(self):
    super().__init__("installing profile", "please wait...")
    self._show_time = 0.0

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0
    self._show_time = rl.get_time()

  def _back_enabled(self) -> bool:
    return False

  def _render(self, _):
    t = (rl.get_time() - self._show_time) % (self.DOT_STEP * 2)
    dots = "." * min(int(t / (self.DOT_STEP / 4)), 3)
    self._card.set_value(f"please wait{dots}")
    super()._render(_)


class EsimProfileButton(BigButton):
  SUB_LABEL_DISABLED = rl.Color(255, 255, 255, int(255 * 0.585))
  CHECK_ICON_COLOR = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
  LABEL_PADDING = 98
  LABEL_WIDTH = 402 - 98 - 28
  SUB_LABEL_WIDTH = 402 - BigButton.LABEL_HORIZONTAL_PADDING * 2

  def __init__(self, profile: Profile, cellular_manager: CellularManager, profiles_enabled: Callable[[], bool]):
    self._cellular_manager = cellular_manager
    self._profiles_enabled = profiles_enabled
    super().__init__(profile.display_name, scroll=True)

    self._profile = profile

    self._cell_full_txt = gui_app.texture("icons_mici/settings/network/cell_strength_full.png", 48, 36)
    self._cell_none_txt = gui_app.texture("icons_mici/settings/network/cell_strength_none.png", 48, 36)
    self._check_txt = gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 32, 32)
    self._comma_txt = gui_app.texture("icons_mici/settings/comma_icon.png", 36, 36) if profile.is_comma else None

    self._delete_btn = ProfileActionButton(self._on_delete, delete=True)
    self._rename_btn = ProfileActionButton(self._on_rename) if not profile.is_comma else None
    self._delete_btn.set_enabled(lambda: not self._locked and not self._cellular_manager.busy and self._show_delete_btn)
    if self._rename_btn:
      self._rename_btn.set_enabled(lambda: not self._locked and not self._cellular_manager.busy)
    self.set_enabled(lambda: not self._profile.enabled and self._profiles_enabled() and not self._cellular_manager.busy)
    self.update_profile(profile)

  @property
  def profile(self) -> Profile:
    return self._profile

  def update_profile(self, profile: Profile):
    self._profile = profile
    active = profile.enabled
    self.set_text(profile.display_name)
    self.set_value("active" if active else "switch")

  def _update_state(self):
    super()._update_state()
    self._sub_label.set_color(DEFAULT_TEXT_COLOR if self.enabled else self.SUB_LABEL_DISABLED)
    self._sub_label.set_font_weight(FontWeight.SEMI_BOLD if self.enabled else FontWeight.ROMAN)

  @property
  def _locked(self) -> bool:
    return not self._profile.is_comma and not self._profiles_enabled()

  @property
  def _show_delete_btn(self) -> bool:
    return not self._profile.enabled and not self._profile.is_comma

  def _on_rename(self):
    current = self._profile.nickname or ""
    dlg = BigInputDialog("nickname", default_text=current, confirm_callback=self._on_nickname_entered,
                         text_validator=lambda text: bool(text.strip()))
    gui_app.push_widget(dlg)

  def _on_delete(self):
    icon = gui_app.texture("icons_mici/settings/network/new/trash.png", 54, 64)
    gui_app.push_widget(BigConfirmationDialog("slide to delete", icon, self._delete_profile, red=True))

  def _delete_profile(self):
    if not self._locked and not self._cellular_manager.busy and self._show_delete_btn:
      self._cellular_manager.delete_profile(self._profile.iccid)

  def _on_nickname_entered(self, nickname: str):
    if not self._locked and not self._cellular_manager.busy:
      self._cellular_manager.nickname_profile(self._profile.iccid, nickname.strip())

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._show_delete_btn and rl.check_collision_point_rec(mouse_pos, self._delete_btn.rect):
      return
    if self._rename_btn is not None and rl.check_collision_point_rec(mouse_pos, self._rename_btn.rect):
      return
    super()._handle_mouse_release(mouse_pos)

  def _get_label_font_size(self):
    return 48

  def _draw_content(self, btn_y: float):
    self._label.set_color(self.SUB_LABEL_DISABLED if self._locked else LABEL_COLOR)
    label_rect = rl.Rectangle(self._rect.x + self.LABEL_PADDING, btn_y + self.LABEL_VERTICAL_PADDING,
                              self.LABEL_WIDTH, self._rect.height - self.LABEL_VERTICAL_PADDING * 2)
    self._label.render(label_rect)

    active = self._profile.enabled

    if self.value:
      sub_label_x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
      label_y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING
      action_w = self._rename_btn.rect.width if self._rename_btn is not None else 0
      action_w += self._delete_btn.rect.width if self._show_delete_btn else 0
      sub_label_w = self.SUB_LABEL_WIDTH - action_w
      sub_label_height = self._sub_label.get_content_height(sub_label_w)

      if active:
        check_y = int(label_y - sub_label_height + (sub_label_height - self._check_txt.height) / 2)
        rl.draw_texture_ex(self._check_txt, rl.Vector2(sub_label_x, check_y), 0.0, 1.0, self.CHECK_ICON_COLOR)
        sub_label_x += self._check_txt.width + 14

      sub_label_rect = rl.Rectangle(sub_label_x, label_y - sub_label_height, sub_label_w, sub_label_height)
      self._sub_label.render(sub_label_rect)

    if self._comma_txt:
      rl.draw_texture_ex(self._comma_txt, (self._rect.x + 36, btn_y + 38), 0.0, 1.0, rl.WHITE)
    else:
      cell_icon = self._cell_full_txt if active else self._cell_none_txt
      rl.draw_texture_ex(cell_icon, (self._rect.x + 30, btn_y + 38), 0.0, 1.0, rl.WHITE)

    btn_x = self._rect.x + self._rect.width - (ProfileActionButton.MARGIN - ProfileActionButton.HORIZONTAL_MARGIN)
    btn_bottom = btn_y + self._rect.height
    if self._show_delete_btn:
      btn_x -= self._delete_btn.rect.width
      self._delete_btn.render(rl.Rectangle(
        btn_x, btn_bottom - self._delete_btn.rect.height,
        self._delete_btn.rect.width, self._delete_btn.rect.height,
      ))
    if self._rename_btn is not None:
      btn_x -= self._rename_btn.rect.width
      self._rename_btn.render(rl.Rectangle(
        btn_x, btn_bottom - self._rename_btn.rect.height,
        self._rename_btn.rect.width, self._rename_btn.rect.height,
      ))

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    def action_pressed() -> bool:
      return self._delete_btn.is_pressed or (self._rename_btn is not None and self._rename_btn.is_pressed)
    super().set_touch_valid_callback(lambda: touch_callback() and not action_pressed())
    self._delete_btn.set_touch_valid_callback(touch_callback)
    if self._rename_btn:
      self._rename_btn.set_touch_valid_callback(touch_callback)


class EsimErrorDialog(NavRawScrollPanel):
  def __init__(self, error: str):
    super().__init__()
    self._title = UnifiedLabel("esim error", font_size=64, font_weight=FontWeight.BOLD)
    self._error = UnifiedLabel(error, font_size=36, elide=False)

  def _render(self, rect: rl.Rectangle):
    width = int(rect.width - 80)
    title_height = self._title.get_content_height(width)
    error_height = self._error.get_content_height(width)
    offset = self._scroll_panel.update(rect, title_height + error_height + 100)
    y = rect.y + 40 + offset

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._title.render(rl.Rectangle(rect.x + 40, y, width, title_height))
    self._error.render(rl.Rectangle(rect.x + 40, y + title_height + 20, width, error_height))
    rl.end_scissor_mode()


class EsimUI(NavScroller):
  def __init__(self, cellular_manager: CellularManager, profiles_enabled: Callable[[], bool]):
    super().__init__()

    self._cellular_manager = cellular_manager
    self._profiles_enabled = profiles_enabled

    self._add_profile_btn = BigButton("add profile", "scan QR code")
    self._add_profile_btn.set_click_callback(self._on_add_profile)
    self._scroller.add_widget(self._add_profile_btn)
    self._installing_dialog: InstallingProfileDialog | None = None

    self._cellular_manager.on_profiles_updated = self._on_profiles_updated
    self._cellular_manager.on_operation_error = self._on_error

  def show_event(self):
    super().show_event()
    self._update_buttons(re_sort=True)
    self._cellular_manager.refresh_profiles()

  def _on_profiles_updated(self):
    if self._installing_dialog:
      existing = {btn.profile.iccid for btn in self._scroller.items if isinstance(btn, EsimProfileButton)}
      added = [profile for profile in self._cellular_manager.profiles if profile.iccid not in existing]
      # Start the normal tap-to-activate flow once the profile list is visible again.
      self._installing_dialog.dismiss(lambda: self._on_profile_clicked(added[0]) if len(added) == 1 else None)
      self._installing_dialog = None

    self._update_buttons()

  def _update_buttons(self, re_sort: bool = False):
    existing = {btn.profile.iccid: btn for btn in self._scroller.items if isinstance(btn, EsimProfileButton)}
    buttons = []
    for profile in self._cellular_manager.profiles:
      btn = existing.get(profile.iccid)
      if btn is None:
        btn = EsimProfileButton(profile, self._cellular_manager, self._profiles_enabled)
        btn.set_click_callback(lambda btn=btn: self._on_profile_clicked(btn.profile))
        self._scroller.add_widget(btn)
      else:
        btn.update_profile(profile)
      buttons.append(btn)

    if re_sort:
      self._scroller.items[:] = sorted(buttons, key=lambda b: not b.profile.enabled)
    else:
      self._scroller.items[:] = [btn for btn in self._scroller.items if btn in buttons]

    self._scroller.items.append(self._add_profile_btn)

  def _move_profile_to_front(self, iccid: str | None, scroll: bool = False):
    front_btn_idx = next((i for i, btn in enumerate(self._scroller.items)
                          if isinstance(btn, EsimProfileButton) and btn.profile.iccid == iccid), None) if iccid else None

    if front_btn_idx is not None and front_btn_idx > 0:
      self._scroller.move_item(front_btn_idx, 0)

      if scroll:
        self._scroller.scroll_to(self._scroller.scroll_panel.get_offset(), smooth=True)

  def _update_state(self):
    super()._update_state()

    self._add_profile_btn.set_enabled(not self._cellular_manager.busy and self._profiles_enabled())
    active = self._cellular_manager.active_profile
    self._move_profile_to_front(active.iccid if active else None)

  def _on_add_profile(self):
    if self._cellular_manager.busy or not self._profiles_enabled():
      return
    if ui_state.sm["deviceState"].networkType == log.DeviceState.NetworkType.none:
      gui_app.push_widget(BigDialog("", tr("Ensure you're connected to the internet and try again.")))
      return
    gui_app.push_widget(QRScannerDialog(on_qr_detected=self._on_qr_scanned))

  def _on_qr_scanned(self, lpa_code: str):
    dlg = BigInputDialog("enter a nickname...", text_validator=lambda text: bool(text.strip()),
                         confirm_callback=lambda nickname: self._download_profile(lpa_code, nickname))
    gui_app.push_widget(dlg)

  def _download_profile(self, lpa_code: str, nickname: str):
    self._installing_dialog = InstallingProfileDialog()
    gui_app.push_widget(self._installing_dialog)
    self._cellular_manager.download_profile(lpa_code, nickname.strip())

  def _on_error(self, error: str):
    cloudlog.error("eSIM error: %s", error)
    dlg = EsimErrorDialog(error)
    if self._installing_dialog:
      self._installing_dialog.dismiss(lambda: gui_app.push_widget(dlg))
      self._installing_dialog = None
    else:
      gui_app.push_widget(dlg)

  def _on_profile_clicked(self, profile: Profile):
    if self._cellular_manager.busy or not self._profiles_enabled():
      return
    self._cellular_manager.switch_profile(profile.iccid)
    self._move_profile_to_front(profile.iccid, scroll=True)
