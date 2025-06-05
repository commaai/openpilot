import pyray as rl

from openpilot.system.ui.lib.button import gui_button, ButtonStyle, TextAlignment
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel


class MultiOptionDialog:
  def __init__(self, title, options, current=""):
    self._title = title
    self._options = options
    self._current = current if current in options else ""
    self._selection = self._current
    self._option_height = 80
    self._padding = 20
    self.scroll_panel = GuiScrollPanel()

  @property
  def selection(self):
    return self._selection

  def render(self, rect):
    title_rect = rl.Rectangle(rect.x + self._padding, rect.y + self._padding, rect.width - 2 * self._padding, 70)
    gui_label(title_rect, self._title, 70)

    options_y_start = rect.y + 120
    options_height = len(self._options) * (self._option_height + 10)
    options_rect = rl.Rectangle(rect.x + self._padding, options_y_start, rect.width - 2 * self._padding, options_height)

    view_rect = rl.Rectangle(
      rect.x + self._padding, options_y_start, rect.width - 2 * self._padding, rect.height - 200 - 2 * self._padding
    )

    offset = self.scroll_panel.handle_scroll(view_rect, options_rect)
    is_click_valid = self.scroll_panel.is_click_valid()

    rl.begin_scissor_mode(int(view_rect.x), int(view_rect.y), int(view_rect.width), int(view_rect.height))

    for i, option in enumerate(self._options):
      y_pos = view_rect.y + i * (self._option_height + 10) + offset.y
      item_rect = rl.Rectangle(view_rect.x, y_pos, view_rect.width, self._option_height)

      if not rl.check_collision_recs(item_rect, view_rect):
        continue

      is_selected = option == self._selection
      button_style = ButtonStyle.PRIMARY if is_selected else ButtonStyle.NORMAL

      if gui_button(item_rect, option, button_style=button_style, text_alignment=TextAlignment.LEFT) and is_click_valid:
        self._selection = option

    rl.end_scissor_mode()

    button_y = rect.y + rect.height - 80 - self._padding
    button_width = (rect.width - 3 * self._padding) / 2

    cancel_rect = rl.Rectangle(rect.x + self._padding, button_y, button_width, 80)
    if gui_button(cancel_rect, "Cancel"):
      return 0  # Canceled

    select_rect = rl.Rectangle(rect.x + 2 * self._padding + button_width, button_y, button_width, 80)
    has_new_selection = self._selection != "" and self._selection != self._current

    if gui_button(select_rect, "Select", is_enabled=has_new_selection, button_style=ButtonStyle.PRIMARY):
      return 1  # Selected

    return -1  # Still active


if __name__ == "__main__":
  from openpilot.system.ui.lib.application import gui_app

  gui_app.init_window("Multi Option Dialog Example")
  options = [f"Option {i}" for i in range(1, 11)]
  dialog = MultiOptionDialog("Choose an option", options, options[0])

  for _ in gui_app.render():
    result = dialog.render(rl.Rectangle(100, 100, 1024, 800))
    if result >= 0:
      print(f"Selected: {dialog.selection}" if result > 0 else "Canceled")
      break
