import os
import pyray as rl

DEBUG_FPS = os.getenv("DEBUG_FPS") == '1'

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


class DrawingContext:
  def __enter__(self):
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if DEBUG_FPS:
      rl.draw_text(f"FPS: {rl.get_fps()}", 10, 10, 20, rl.RED)
    rl.end_drawing()
