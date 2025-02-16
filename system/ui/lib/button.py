
import pyray as rl
from openpilot.system.ui.lib.utils import GuiStyleContext

BUTTON_DEFAULT_BG_COLOR = rl.Color(51, 51, 51, 255)

def gui_button(rect, text, bg_color=BUTTON_DEFAULT_BG_COLOR):
  styles = [
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE),
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(bg_color))
  ]

  with GuiStyleContext(styles):
    return rl.gui_button(rect, text)
