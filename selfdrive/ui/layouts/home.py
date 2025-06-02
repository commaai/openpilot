import pyray as rl
from openpilot.system.ui.lib.label import gui_text_box


class HomeLayout:
  def __init__(self):
    pass

  def render(self, rect: rl.Rectangle):
    gui_text_box(
      rect,
      "Demo Home Layout",
      font_size=170,
      color=rl.WHITE,
      alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
    )
