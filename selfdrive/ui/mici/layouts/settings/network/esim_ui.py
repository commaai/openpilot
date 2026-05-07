import pyray as rl
from collections.abc import Callable

from openpilot.selfdrive.ui.mici.widgets.button import BigButton, LABEL_COLOR
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, BigDialog, BigInputDialog
from openpilot.system.hardware.base import Profile
from openpilot.system.ui.lib.application import DEFAULT_TEXT_COLOR, FontWeight, MousePos, gui_app
from openpilot.system.ui.lib.cellular_manager import CellularManager
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.scroller import NavScroller

SUB_LABEL_DISABLED = rl.Color(255, 255, 255, int(255 * 0.585))
CHECK_ICON_COLOR = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))


class DeleteButton(Widget):
  SIZE = 63
  MARGIN = 12

  def __init__(self, delete_callback: Callable):
    super().__init__()
    self._delete_callback = delete_callback

    self._bg_txt = gui_app.texture("icons_mici/settings/network/new/forget_button.png", self.SIZE, self.SIZE)
    self._bg_pressed_txt = gui_app.texture("icons_mici/settings/network/new/forget_button_pressed.png", self.SIZE, self.SIZE)
    self._trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 22, 26)
    self._dialog_trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 54, 64)
    self.set_rect(rl.Rectangle(0, 0, self.SIZE + self.MARGIN * 2, self.SIZE + self.MARGIN * 2))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    dlg = BigConfirmationDialog("slide to delete", self._dialog_trash_txt, self._delete_callback, red=True)
    gui_app.push_widget(dlg)

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture_ex(bg_txt, (self._rect.x + (self._rect.width - self._bg_txt.width) / 2,
                                self._rect.y + (self._rect.height - self._bg_txt.height) / 2), 0, 1.0, rl.WHITE)

    trash_x = self._rect.x + (self._rect.width - self._trash_txt.width) / 2
    trash_y = self._rect.y + (self._rect.height - self._trash_txt.height) / 2
    rl.draw_texture_ex(self._trash_txt, (trash_x, trash_y), 0, 1.0, rl.WHITE)


class RenameButton(Widget):
  SIZE = 84
  MARGIN = 12

  def __init__(self, rename_callback: Callable):
    super().__init__()
    self._rename_callback = rename_callback

    self._bg_txt = gui_app.texture("icons_mici/buttons/button_circle.png", self.SIZE, self.SIZE)
    self._bg_pressed_txt = gui_app.texture("icons_mici/buttons/button_circle_pressed.png", self.SIZE, self.SIZE)
    self.set_rect(rl.Rectangle(0, 0, self.SIZE + self.MARGIN * 2, self.SIZE + self.MARGIN * 2))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    self._rename_callback()

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture_ex(bg_txt, (self._rect.x + (self._rect.width - self._bg_txt.width) / 2,
                                self._rect.y + (self._rect.height - self._bg_txt.height) / 2), 0, 1.0, rl.WHITE)
    icon_rect = rl.Rectangle(self._rect.x, self._rect.y, self._rect.width, self._rect.height)
    gui_label(icon_rect, "Aa", 32, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)


class EsimProfileButton(BigButton):
  LABEL_PADDING = 98
  LABEL_WIDTH = 402 - 98 - 28
  SUB_LABEL_WIDTH = 402 - BigButton.LABEL_HORIZONTAL_PADDING * 2

  def __init__(self, profile: Profile, cellular_manager: CellularManager):
    self._cellular_manager = cellular_manager
    super().__init__(profile.display_name, scroll=True)

    self._profile = profile
    self._deleting = False
    self._switching = False

    self._cell_full_txt = gui_app.texture("icons_mici/settings/network/cell_strength_full.png", 48, 36)
    self._cell_none_txt = gui_app.texture("icons_mici/settings/network/cell_strength_none.png", 48, 36)
    self._check_txt = gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 32, 32)
    self._comma_txt = gui_app.texture("icons_mici/settings/comma_icon.png", 36, 36) if profile.is_comma else None

    self._delete_btn = DeleteButton(self._on_delete)
    self._rename_btn = RenameButton(self._on_rename) if not profile.is_comma else None

  @property
  def profile(self) -> Profile:
    return self._profile

  def update_profile(self, profile: Profile):
    self._profile = profile
    self._deleting = False
    self._switching = False
    if profile.display_name != self.text:
      self.set_text(profile.display_name)

  def mark_switching(self):
    self._switching = True

  @property
  def _show_rename_btn(self) -> bool:
    if self._deleting:
      return False
    return self._rename_btn is not None

  @property
  def _show_delete_btn(self) -> bool:
    if self._deleting or self._profile.enabled:
      return False
    return not self._profile.is_comma

  def _on_delete(self):
    if self._deleting:
      return
    self._deleting = True
    self._cellular_manager.delete_profile(self._profile.iccid)

  def _on_rename(self):
    current = self._profile.nickname or ""
    dlg = BigInputDialog("nickname", default_text=current, minimum_length=0, confirm_callback=self._on_nickname_entered)
    gui_app.push_widget(dlg)

  def _on_nickname_entered(self, nickname: str):
    self._cellular_manager.nickname_profile(self._profile.iccid, nickname.strip())

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._show_delete_btn and rl.check_collision_point_rec(mouse_pos, self._delete_btn.rect):
      return
    if self._show_rename_btn and rl.check_collision_point_rec(mouse_pos, self._rename_btn.rect):
      return
    super()._handle_mouse_release(mouse_pos)

  def _get_label_font_size(self):
    return 48

  def _draw_content(self, btn_y: float):
    self._label.set_color(LABEL_COLOR)
    label_rect = rl.Rectangle(self._rect.x + self.LABEL_PADDING, btn_y + self.LABEL_VERTICAL_PADDING,
                              self.LABEL_WIDTH, self._rect.height - self.LABEL_VERTICAL_PADDING * 2)
    self._label.render(label_rect)

    active = self._profile.enabled

    if self.value:
      sub_label_x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
      label_y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING
      action_w = self._rename_btn.rect.width if self._show_rename_btn else 0
      # delete sits just inside rename's left edge (overlap their inner margins) so the sub_label has more room
      action_w += self._delete_btn.rect.width - DeleteButton.MARGIN if self._show_delete_btn else 0
      sub_label_w = self.SUB_LABEL_WIDTH - action_w
      sub_label_height = self._sub_label.get_content_height(sub_label_w)

      if active and not self._deleting:
        check_y = int(label_y - sub_label_height + (sub_label_height - self._check_txt.height) / 2)
        rl.draw_texture_ex(self._check_txt, rl.Vector2(sub_label_x, check_y), 0.0, 1.0, CHECK_ICON_COLOR)
        sub_label_x += self._check_txt.width + 14

      sub_label_rect = rl.Rectangle(sub_label_x, label_y - sub_label_height, sub_label_w, sub_label_height)
      self._sub_label.render(sub_label_rect)

    if self._comma_txt:
      rl.draw_texture_ex(self._comma_txt, (self._rect.x + 36, btn_y + 38), 0.0, 1.0, rl.WHITE)
    else:
      cell_icon = self._cell_full_txt if active else self._cell_none_txt
      rl.draw_texture_ex(cell_icon, (self._rect.x + 30, btn_y + 38), 0.0, 1.0, rl.WHITE)

    btn_x = self._rect.x + self._rect.width
    btn_bottom = btn_y + self._rect.height
    if self._show_rename_btn:
      btn_x -= self._rename_btn.rect.width
      self._rename_btn.render(rl.Rectangle(
        btn_x, btn_bottom - self._rename_btn.rect.height,
        self._rename_btn.rect.width, self._rename_btn.rect.height,
      ))
    if self._show_delete_btn:
      btn_x -= self._delete_btn.rect.width - DeleteButton.MARGIN
      self._delete_btn.render(rl.Rectangle(
        btn_x, btn_bottom - self._delete_btn.rect.height,
        self._delete_btn.rect.width, self._delete_btn.rect.height,
      ))

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    def action_pressed() -> bool:
      return self._delete_btn.is_pressed or (self._rename_btn is not None and self._rename_btn.is_pressed)
    super().set_touch_valid_callback(lambda: touch_callback() and not action_pressed())
    self._delete_btn.set_touch_valid_callback(touch_callback)
    if self._rename_btn:
      self._rename_btn.set_touch_valid_callback(touch_callback)

  def _update_state(self):
    super()._update_state()

    if self._deleting or self._switching:
      self.set_enabled(False)
      self._sub_label.set_color(SUB_LABEL_DISABLED)
      self._sub_label.set_font_weight(FontWeight.ROMAN)
      self.set_value("deleting..." if self._deleting else "switching...")
    elif self._profile.enabled:
      self.set_value("active")
      self.set_enabled(False)
      self._sub_label.set_color(SUB_LABEL_DISABLED)
      self._sub_label.set_font_weight(FontWeight.ROMAN)
    else:
      self.set_value("switch")
      self.set_enabled(True)
      self._sub_label.set_color(DEFAULT_TEXT_COLOR)
      self._sub_label.set_font_weight(FontWeight.SEMI_BOLD)


class EsimUIMici(NavScroller):
  def __init__(self, cellular_manager: CellularManager):
    super().__init__()

    self._cellular_manager = cellular_manager

    self._cellular_manager.add_callbacks(
      profiles_updated=self._on_profiles_updated,
      operation_error=self._on_error,
    )

  def show_event(self):
    super().show_event()
    self._update_buttons(re_sort=True)
    self._cellular_manager.refresh_profiles()

  def _on_profiles_updated(self, profiles: list[Profile]):
    self._update_buttons()

  def _update_buttons(self, re_sort: bool = False):
    existing = {btn.profile.iccid: btn for btn in self._scroller.items if isinstance(btn, EsimProfileButton)}
    profiles = self._cellular_manager.profiles
    current_iccids = {p.iccid for p in profiles}

    for profile in profiles:
      if profile.iccid in existing:
        existing[profile.iccid].update_profile(profile)
      else:
        btn = EsimProfileButton(profile, self._cellular_manager)
        btn.set_click_callback(lambda iccid=profile.iccid: self._on_profile_clicked(iccid))
        self._scroller.add_widget(btn)

    if re_sort:
      btn_map = {btn.profile.iccid: btn for btn in self._scroller.items if isinstance(btn, EsimProfileButton)}
      self._scroller.items[:] = sorted(
        [btn_map[iccid] for iccid in current_iccids if iccid in btn_map],
        key=lambda b: not b.profile.enabled,
      )
    else:
      self._scroller.items[:] = [
        btn for btn in self._scroller.items
        if not isinstance(btn, EsimProfileButton) or btn.profile.iccid in current_iccids
      ]

  def _move_profile_to_front(self, iccid: str | None, scroll: bool = False):
    front_btn_idx = next((i for i, btn in enumerate(self._scroller.items)
                          if isinstance(btn, EsimProfileButton) and
                          btn.profile.iccid == iccid), None) if iccid else None

    if front_btn_idx is not None and front_btn_idx > 0:
      self._scroller.move_item(front_btn_idx, 0)

      if scroll:
        self._scroller.scroll_to(self._scroller.scroll_panel.get_offset(), smooth=True)

  def _update_state(self):
    super()._update_state()

    active = next((p for p in self._cellular_manager.profiles if p.enabled), None)
    self._move_profile_to_front(active.iccid if active else None)

  def _on_error(self, error: str):
    dlg = BigDialog("esim error", error)
    gui_app.push_widget(dlg)

  def _on_profile_clicked(self, iccid: str):
    if self._cellular_manager.busy:
      return
    profile = next((p for p in self._cellular_manager.profiles if p.iccid == iccid), None)
    if profile is None or profile.enabled:
      return

    btn = next((b for b in self._scroller.items
                if isinstance(b, EsimProfileButton) and b.profile.iccid == iccid), None)
    if btn is not None:
      btn.mark_switching()

    self._cellular_manager.switch_profile(iccid)
    self._move_profile_to_front(iccid, scroll=True)
