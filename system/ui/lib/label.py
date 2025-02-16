import pyray as rl
from openpilot.system.ui.lib.utils import GuiStyleContext

def gui_label(rect, text, font_size):
  styles = [
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_LINE_SPACING, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_WRAP_MODE, rl.GuiTextWrapMode.TEXT_WRAP_WORD)
  ]

  with GuiStyleContext(styles):
    rl.gui_label(rect, text)
