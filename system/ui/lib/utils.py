import pyray as rl


class GuiStyleContext:
  def __init__(self, styles: list[tuple[int, int, int]]):
    """styles is a list of tuples (control, prop, new_value)"""
    self.styles = styles
    self.prev_styles: list[tuple[int, int, int]] = []

  def __enter__(self):
    for control, prop, new_value in self.styles:
      prev_value = rl.gui_get_style(control, prop)
      self.prev_styles.append((control, prop, prev_value))
      rl.gui_set_style(control, prop, new_value)

  def __exit__(self, exc_type, exc_value, traceback):
    for control, prop, prev_value in self.prev_styles:
      rl.gui_set_style(control, prop, prev_value)
