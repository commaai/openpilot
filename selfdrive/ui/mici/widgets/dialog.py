import abc
import math
import pyray as rl
from typing import Union
from collections.abc import Callable
from typing import cast
from openpilot.selfdrive.ui.mici.widgets.side_button import SideButton
from openpilot.system.ui.widgets import Widget, NavWidget, DialogResult
from openpilot.system.ui.widgets.label import UnifiedLabel, gui_label
from openpilot.system.ui.widgets.mici_keyboard import MiciKeyboard
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.slider import RedBigSlider, BigSlider
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.widgets.button import BigButton

DEBUG = False

PADDING = 20


class BigDialogBase(NavWidget, abc.ABC):
  def __init__(self, right_btn: str | None = None, right_btn_callback: Callable | None = None):
    super().__init__()
    self._ret = DialogResult.NO_ACTION
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    self.set_back_callback(lambda: setattr(self, '_ret', DialogResult.CANCEL))

    self._right_btn = None
    if right_btn:
      def right_btn_callback_wrapper():
        gui_app.set_modal_overlay(None)
        if right_btn_callback:
          right_btn_callback()

      self._right_btn = SideButton(right_btn)
      self._right_btn.set_click_callback(right_btn_callback_wrapper)
      # move to right side
      self._right_btn._rect.x = self._rect.x + self._rect.width - self._right_btn._rect.width

  def _render(self, _) -> DialogResult:
    """
    Allows `gui_app.set_modal_overlay(BigDialog(...))`.
    The overlay runner keeps calling until result != NO_ACTION.
    """
    if self._right_btn:
      self._right_btn.set_position(self._right_btn._rect.x, self._rect.y)
      self._right_btn.render()

    return self._ret


class BigDialog(BigDialogBase):
  def __init__(self,
               title: str,
               description: str,
               right_btn: str | None = None,
               right_btn_callback: Callable | None = None):
    super().__init__(right_btn, right_btn_callback)
    self._title = title
    self._description = description

  def _render(self, _) -> DialogResult:
    super()._render(_)

    # draw title
    # TODO: we desperately need layouts
    # TODO: coming up with these numbers manually is a pain and not scalable
    # TODO: no clue what any of these numbers mean. VBox and HBox would remove all of this shite
    max_width = self._rect.width - PADDING * 2
    if self._right_btn:
      max_width -= self._right_btn._rect.width

    title_wrapped = '\n'.join(wrap_text(gui_app.font(FontWeight.BOLD), self._title, 50, int(max_width)))
    title_size = measure_text_cached(gui_app.font(FontWeight.BOLD), title_wrapped, 50)
    text_x_offset = 0
    title_rect = rl.Rectangle(int(self._rect.x + text_x_offset + PADDING),
                              int(self._rect.y + PADDING),
                              int(max_width),
                              int(title_size.y))
    gui_label(title_rect, title_wrapped, 50, font_weight=FontWeight.BOLD,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

    # draw description
    desc_wrapped = '\n'.join(wrap_text(gui_app.font(FontWeight.MEDIUM), self._description, 30, int(max_width)))
    desc_size = measure_text_cached(gui_app.font(FontWeight.MEDIUM), desc_wrapped, 30)
    desc_rect = rl.Rectangle(int(self._rect.x + text_x_offset + PADDING),
                             int(self._rect.y + self._rect.height / 3),
                             int(max_width),
                             int(desc_size.y))
    # TODO: text align doesn't seem to work properly with newlines
    gui_label(desc_rect, desc_wrapped, 30, font_weight=FontWeight.MEDIUM,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

    return self._ret


class BigConfirmationDialogV2(BigDialogBase):
  def __init__(self, title: str, icon: str, red: bool = False,
               exit_on_confirm: bool = True,
               confirm_callback: Callable | None = None):
    super().__init__()
    self._confirm_callback = confirm_callback
    self._exit_on_confirm = exit_on_confirm

    icon_txt = gui_app.texture(icon, 64, 53)
    self._slider: BigSlider | RedBigSlider
    if red:
      self._slider = RedBigSlider(title, icon_txt, confirm_callback=self._on_confirm)
    else:
      self._slider = BigSlider(title, icon_txt, confirm_callback=self._on_confirm)
    self._slider.set_enabled(lambda: not self._swiping_away)

  def _on_confirm(self):
    if self._confirm_callback:
      self._confirm_callback()
    if self._exit_on_confirm:
      self._ret = DialogResult.CONFIRM

  def _update_state(self):
    super()._update_state()
    if self._swiping_away and not self._slider.confirmed:
      self._slider.reset()

  def _render(self, _) -> DialogResult:
    self._slider.render(self._rect)
    return self._ret


class BigInputDialog(BigDialogBase):
  BACK_TOUCH_AREA_PERCENTAGE = 0.2
  BACKSPACE_RATE = 25  # hz

  def __init__(self,
               hint: str,
               default_text: str = "",
               minimum_length: int = 1,
               confirm_callback: Callable[[str], None] = None):
    super().__init__(None, None)
    self._hint_label = UnifiedLabel(hint, font_size=35, text_color=rl.Color(255, 255, 255, int(255 * 0.35)),
                                    font_weight=FontWeight.MEDIUM)
    self._keyboard = MiciKeyboard()
    self._keyboard.set_text(default_text)
    self._minimum_length = minimum_length

    self._backspace_held_time: float | None = None

    self._backspace_img = gui_app.texture("icons_mici/settings/keyboard/backspace.png", 44, 44)
    self._backspace_img_alpha = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._enter_img = gui_app.texture("icons_mici/settings/keyboard/confirm.png", 44, 44)
    self._enter_img_alpha = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    # rects for top buttons
    self._top_left_button_rect = rl.Rectangle(0, 0, 0, 0)
    self._top_right_button_rect = rl.Rectangle(0, 0, 0, 0)

    def confirm_callback_wrapper():
      self._ret = DialogResult.CONFIRM
      if confirm_callback:
        confirm_callback(self._keyboard.text())
    self._confirm_callback = confirm_callback_wrapper

  def _update_state(self):
    super()._update_state()

    last_mouse_event = gui_app.last_mouse_event
    if last_mouse_event.left_down and rl.check_collision_point_rec(last_mouse_event.pos, self._top_right_button_rect) and self._backspace_img_alpha.x > 1:
      if self._backspace_held_time is None:
        self._backspace_held_time = rl.get_time()

      if rl.get_time() - self._backspace_held_time > 0.5:
        if gui_app.frame % round(gui_app.target_fps / self.BACKSPACE_RATE) == 0:
          self._keyboard.backspace()

    else:
      self._backspace_held_time = None

  def _render(self, _):
    text_input_size = 35

    # draw current text so far below everything. text floats left but always stays in view
    text = self._keyboard.text()
    candidate_char = self._keyboard.get_candidate_character()
    text_size = measure_text_cached(gui_app.font(FontWeight.ROMAN), text + candidate_char or self._hint_label.text, text_input_size)
    text_x = PADDING * 2 + self._enter_img.width

    # text needs to move left if we're at the end where right button is
    text_rect = rl.Rectangle(text_x,
                             int(self._rect.y + PADDING),
                             # clip width to right button when in view
                             int(self._rect.width - text_x - PADDING * 2 - self._enter_img.width + 5),  # TODO: why 5?
                             int(text_size.y))

    # draw rounded background for text input
    bg_block_margin = 5
    text_field_rect = rl.Rectangle(text_rect.x - bg_block_margin, text_rect.y - bg_block_margin,
                                   text_rect.width + bg_block_margin * 2, text_input_size + bg_block_margin * 2)

    # draw text input
    # push text left with a gradient on left side if too long
    if text_size.x > text_rect.width:
      text_x -= text_size.x - text_rect.width

    rl.begin_scissor_mode(int(text_rect.x), int(text_rect.y), int(text_rect.width), int(text_rect.height))
    rl.draw_text_ex(gui_app.font(FontWeight.ROMAN), text, rl.Vector2(text_x, text_rect.y), text_input_size, 0, rl.WHITE)

    # draw grayed out character user is hovering over
    if candidate_char:
      candidate_char_size = measure_text_cached(gui_app.font(FontWeight.ROMAN), candidate_char, text_input_size)
      rl.draw_text_ex(gui_app.font(FontWeight.ROMAN), candidate_char,
                      rl.Vector2(min(text_x + text_size.x, text_rect.x + text_rect.width) - candidate_char_size.x, text_rect.y),
                      text_input_size, 0, rl.Color(255, 255, 255, 128))

    rl.end_scissor_mode()

    # draw gradient on left side to indicate more text
    if text_size.x > text_rect.width:
      rl.draw_rectangle_gradient_h(int(text_rect.x), int(text_rect.y), 80, int(text_rect.height),
                                   rl.BLACK, rl.BLANK)

    # draw cursor
    if text:
      blink_alpha = (math.sin(rl.get_time() * 6) + 1) / 2
      cursor_x = min(text_x + text_size.x + 3, text_rect.x + text_rect.width)
      rl.draw_rectangle_rounded(rl.Rectangle(int(cursor_x), int(text_rect.y), 4, int(text_size.y)),
                                1, 4, rl.Color(255, 255, 255, int(255 * blink_alpha)))

    # draw backspace icon with nice fade
    self._backspace_img_alpha.update(255 * bool(text))
    if self._backspace_img_alpha.x > 1:
      color = rl.Color(255, 255, 255, int(self._backspace_img_alpha.x))
      rl.draw_texture(self._backspace_img, int(self._rect.width - self._enter_img.width - 15), int(text_field_rect.y), color)

    if not text and self._hint_label.text and not candidate_char:
      # draw description if no text entered yet and not drawing candidate char
      self._hint_label.render(text_field_rect)

    # TODO: move to update state
    # make rect take up entire area so it's easier to click
    self._top_left_button_rect = rl.Rectangle(self._rect.x, self._rect.y, text_field_rect.x, self._rect.height - self._keyboard.get_keyboard_height())
    self._top_right_button_rect = rl.Rectangle(text_field_rect.x + text_field_rect.width, self._rect.y,
                                               self._rect.width - (text_field_rect.x + text_field_rect.width), self._top_left_button_rect.height)

    self._enter_img_alpha.update(255 if (len(text) >= self._minimum_length) else 255 * 0.35)
    if self._enter_img_alpha.x > 1:
      color = rl.Color(255, 255, 255, int(self._enter_img_alpha.x))
      rl.draw_texture(self._enter_img, int(self._rect.x + 15), int(text_field_rect.y), color)

    # keyboard goes over everything
    self._keyboard.render(self._rect)

    # draw debugging rect bounds
    if DEBUG:
      rl.draw_rectangle_lines_ex(text_field_rect, 1, rl.Color(100, 100, 100, 255))
      rl.draw_rectangle_lines_ex(text_rect, 1, rl.Color(0, 255, 0, 255))
      rl.draw_rectangle_lines_ex(self._top_right_button_rect, 1, rl.Color(0, 255, 0, 255))
      rl.draw_rectangle_lines_ex(self._top_left_button_rect, 1, rl.Color(0, 255, 0, 255))

    return self._ret

  def _handle_mouse_press(self, mouse_pos: MousePos):
    super()._handle_mouse_press(mouse_pos)
    # TODO: need to track where press was so enter and back can activate on release rather than press
    #  or turn into icon widgets :eyes_open:
    # handle backspace icon click
    if rl.check_collision_point_rec(mouse_pos, self._top_right_button_rect) and self._backspace_img_alpha.x > 254:
      self._keyboard.backspace()
    elif rl.check_collision_point_rec(mouse_pos, self._top_left_button_rect) and self._enter_img_alpha.x > 254:
      # handle enter icon click
      self._confirm_callback()


class BigDialogOptionButton(Widget):
  def __init__(self, option: str):
    super().__init__()
    self.option = option
    self.set_rect(rl.Rectangle(0, 0, int(gui_app.width / 2 + 220), 64))

    self._selected = False

    self._label = UnifiedLabel(option, font_size=70, text_color=rl.Color(255, 255, 255, int(255 * 0.58)),
                               font_weight=FontWeight.DISPLAY_REGULAR, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)

  def set_selected(self, selected: bool):
    self._selected = selected

  def _render(self, _):
    if DEBUG:
      rl.draw_rectangle_lines_ex(self._rect, 1, rl.Color(0, 255, 0, 255))

    # FIXME: offset x by -45 because scroller centers horizontally
    if self._selected:
      self._label.set_font_size(74)
      self._label.set_color(rl.Color(255, 255, 255, int(255 * 0.9)))
      self._label.set_font_weight(FontWeight.DISPLAY)
    else:
      self._label.set_font_size(70)
      self._label.set_color(rl.Color(255, 255, 255, int(255 * 0.58)))
      self._label.set_font_weight(FontWeight.DISPLAY_REGULAR)

    self._label.render(self._rect)


class BigMultiOptionDialog(BigDialogBase):
  BACK_TOUCH_AREA_PERCENTAGE = 0.1

  def __init__(self, options: list[str], default: str | None,
               right_btn: str | None = 'check', right_btn_callback: Callable[[], None] = None):
    super().__init__(right_btn, right_btn_callback=right_btn_callback)
    self._options = options
    if default is not None:
      assert default in options

    self._default_option: str = default or (options[0] if len(options) > 0 else "")
    self._selected_option: str = self._default_option
    self._last_selected_option: str = self._selected_option

    self._scroller = Scroller([], horizontal=False, pad_start=100, pad_end=100, spacing=0)
    if self._right_btn is not None:
      self._scroller.set_enabled(lambda: not cast(Widget, self._right_btn).is_pressed)

    for option in options:
      self.add_button(BigDialogOptionButton(option))

  def add_button(self, button: BigDialogOptionButton):
    og_callback = button._click_callback

    def wrapped_callback(btn=button):
      self._on_option_selected(btn.option)
      if og_callback:
        og_callback()

    button.set_click_callback(wrapped_callback)
    self._scroller.add_widget(button)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._on_option_selected(self._default_option)

  def get_selected_option(self) -> str:
    return self._selected_option

  def _on_option_selected(self, option: str):
    y_pos = 0.0
    for btn in self._scroller._items:
      if cast(BigDialogOptionButton, btn).option == option:
        y_pos = btn.rect.y

    self._scroller.scroll_to(y_pos, smooth=True)

  def _selected_option_changed(self):
    pass

  def _update_state(self):
    super()._update_state()

    # get selection by whichever button is closest to center
    center_y = self._rect.y + self._rect.height / 2
    closest_btn = (None, float('inf'))
    for btn in self._scroller._items:
      dist_y = abs((btn.rect.y + btn.rect.height / 2) - center_y)
      if dist_y < closest_btn[1]:
        closest_btn = (btn, dist_y)

    if closest_btn[0]:
      for btn in self._scroller._items:
        btn.set_selected(btn.option == closest_btn[0].option)
      self._selected_option = closest_btn[0].option

    # Signal to subclasses if selection changed
    if self._selected_option != self._last_selected_option:
      self._selected_option_changed()
      self._last_selected_option = self._selected_option

  def _render(self, _):
    super()._render(_)
    self._scroller.render(self._rect)

    return self._ret


class BigDialogButton(BigButton):
  def __init__(self, text: str, value: str = "", icon: Union[str, rl.Texture] = "", description: str = ""):
    super().__init__(text, value, icon)
    self._description = description

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    dlg = BigDialog(self.text, self._description)
    gui_app.set_modal_overlay(dlg)
