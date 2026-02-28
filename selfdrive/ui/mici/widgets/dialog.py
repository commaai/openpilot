import abc
import math
import pyray as rl
from typing import Union
from collections.abc import Callable
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import UnifiedLabel, gui_label
from openpilot.system.ui.widgets.mici_keyboard import MiciKeyboard
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.widgets.slider import RedBigSlider, BigSlider
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.widgets.button import BigButton

DEBUG = False

PADDING = 20


class BigDialogBase(NavWidget, abc.ABC):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    self.set_back_callback(gui_app.pop_widget)


class BigDialog(BigDialogBase):
  def __init__(self,
               title: str,
               description: str):
    super().__init__()
    self._title = title
    self._description = description

  def _render(self, _):
    super()._render(_)

    # draw title
    # TODO: we desperately need layouts
    # TODO: coming up with these numbers manually is a pain and not scalable
    # TODO: no clue what any of these numbers mean. VBox and HBox would remove all of this shite
    max_width = self._rect.width - PADDING * 2

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
    self._slider.set_enabled(lambda: self.enabled and not self._swiping_away)  # self.enabled for nav stack

  def _on_confirm(self):
    if self._exit_on_confirm:
      gui_app.pop_widget()
    if self._confirm_callback:
      self._confirm_callback()

  def _update_state(self):
    super()._update_state()
    if self._swiping_away and not self._slider.confirmed:
      self._slider.reset()

  def _render(self, _):
    self._slider.render(self._rect)


class BigInputDialog(BigDialogBase):
  BACK_TOUCH_AREA_PERCENTAGE = 0.2
  BACKSPACE_RATE = 25  # hz
  TEXT_INPUT_SIZE = 35

  def __init__(self,
               hint: str,
               default_text: str = "",
               minimum_length: int = 1,
               confirm_callback: Callable[[str], None] | None = None):
    super().__init__()
    self._hint_label = UnifiedLabel(hint, font_size=35, text_color=rl.Color(255, 255, 255, int(255 * 0.35)),
                                    font_weight=FontWeight.MEDIUM)
    self._keyboard = MiciKeyboard()
    self._keyboard.set_text(default_text)
    self._keyboard.set_enabled(lambda: self.enabled)  # for nav stack
    self._minimum_length = minimum_length

    self._backspace_held_time: float | None = None

    self._backspace_img = gui_app.texture("icons_mici/settings/keyboard/backspace.png", 42, 36)
    self._backspace_img_alpha = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._enter_img = gui_app.texture("icons_mici/settings/keyboard/enter.png", 76, 62)
    self._enter_disabled_img = gui_app.texture("icons_mici/settings/keyboard/enter_disabled.png", 76, 62)
    self._enter_img_alpha = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    # rects for top buttons
    self._top_left_button_rect = rl.Rectangle(0, 0, 0, 0)
    self._top_right_button_rect = rl.Rectangle(0, 0, 0, 0)

    def confirm_callback_wrapper():
      text = self._keyboard.text()
      gui_app.pop_widget()
      if confirm_callback:
        confirm_callback(text)
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
    # draw current text so far below everything. text floats left but always stays in view
    text = self._keyboard.text()
    candidate_char = self._keyboard.get_candidate_character()
    text_size = measure_text_cached(gui_app.font(FontWeight.ROMAN), text + candidate_char or self._hint_label.text, self.TEXT_INPUT_SIZE)

    bg_block_margin = 5
    text_x = PADDING / 2 + self._enter_img.width + PADDING
    text_field_rect = rl.Rectangle(text_x, int(self._rect.y + PADDING) - bg_block_margin,
                                   int(self._rect.width - text_x * 2),
                                   int(text_size.y))

    # draw text input
    # push text left with a gradient on left side if too long
    if text_size.x > text_field_rect.width:
      text_x -= text_size.x - text_field_rect.width

    rl.begin_scissor_mode(int(text_field_rect.x), int(text_field_rect.y), int(text_field_rect.width), int(text_field_rect.height))
    rl.draw_text_ex(gui_app.font(FontWeight.ROMAN), text, rl.Vector2(text_x, text_field_rect.y), self.TEXT_INPUT_SIZE, 0, rl.WHITE)

    # draw grayed out character user is hovering over
    if candidate_char:
      candidate_char_size = measure_text_cached(gui_app.font(FontWeight.ROMAN), candidate_char, self.TEXT_INPUT_SIZE)
      rl.draw_text_ex(gui_app.font(FontWeight.ROMAN), candidate_char,
                      rl.Vector2(min(text_x + text_size.x, text_field_rect.x + text_field_rect.width) - candidate_char_size.x, text_field_rect.y),
                      self.TEXT_INPUT_SIZE, 0, rl.Color(255, 255, 255, 128))

    rl.end_scissor_mode()

    # draw gradient on left side to indicate more text
    if text_size.x > text_field_rect.width:
      rl.draw_rectangle_gradient_h(int(text_field_rect.x), int(text_field_rect.y), 80, int(text_field_rect.height),
                                   rl.BLACK, rl.BLANK)

    # draw cursor
    blink_alpha = (math.sin(rl.get_time() * 6) + 1) / 2
    if text:
      cursor_x = min(text_x + text_size.x + 3, text_field_rect.x + text_field_rect.width)
    else:
      cursor_x = text_field_rect.x - 6
    rl.draw_rectangle_rounded(rl.Rectangle(int(cursor_x), int(text_field_rect.y), 4, int(text_size.y)),
                              1, 4, rl.Color(255, 255, 255, int(255 * blink_alpha)))

    # draw backspace icon with nice fade
    self._backspace_img_alpha.update(255 * bool(text))
    if self._backspace_img_alpha.x > 1:
      color = rl.Color(255, 255, 255, int(self._backspace_img_alpha.x))
      rl.draw_texture(self._backspace_img, int(self._rect.width - self._backspace_img.width - 27), int(self._rect.y + 14), color)

    if not text and self._hint_label.text and not candidate_char:
      # draw description if no text entered yet and not drawing candidate char
      hint_rect = rl.Rectangle(text_field_rect.x, text_field_rect.y,
                               self._rect.width - text_field_rect.x - PADDING,
                               text_field_rect.height)
      self._hint_label.render(hint_rect)

    # TODO: move to update state
    # make rect take up entire area so it's easier to click
    self._top_left_button_rect = rl.Rectangle(self._rect.x, self._rect.y, text_field_rect.x, self._rect.height - self._keyboard.get_keyboard_height())
    self._top_right_button_rect = rl.Rectangle(text_field_rect.x + text_field_rect.width, self._rect.y,
                                               self._rect.width - (text_field_rect.x + text_field_rect.width), self._top_left_button_rect.height)

    # draw enter button
    self._enter_img_alpha.update(255 if len(text) >= self._minimum_length else 0)
    color = rl.Color(255, 255, 255, int(self._enter_img_alpha.x))
    rl.draw_texture(self._enter_img, int(self._rect.x + PADDING / 2), int(self._rect.y), color)
    color = rl.Color(255, 255, 255, 255 - int(self._enter_img_alpha.x))
    rl.draw_texture(self._enter_disabled_img, int(self._rect.x + PADDING / 2), int(self._rect.y), color)

    # keyboard goes over everything
    self._keyboard.render(self._rect)

    # draw debugging rect bounds
    if DEBUG:
      rl.draw_rectangle_lines_ex(text_field_rect, 1, rl.Color(100, 100, 100, 255))
      rl.draw_rectangle_lines_ex(self._top_right_button_rect, 1, rl.Color(0, 255, 0, 255))
      rl.draw_rectangle_lines_ex(self._top_left_button_rect, 1, rl.Color(0, 255, 0, 255))

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


class BigDialogButton(BigButton):
  def __init__(self, text: str, value: str = "", icon: Union[str, rl.Texture] = "", description: str = ""):
    super().__init__(text, value, icon)
    self._description = description

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    dlg = BigDialog(self.text, self._description)
    gui_app.push_widget(dlg)
