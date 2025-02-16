import pyray as rl
from openpilot.system.ui.lib.utils import GuiStyleContext


def gui_label(rect, text, font_size, color: rl.Color = None, alignment: rl.GuiTextAlignment = rl.GuiTextAlignment.TEXT_ALIGN_LEFT):
  styles = [
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_LINE_SPACING, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_WRAP_MODE, rl.GuiTextWrapMode.TEXT_WRAP_WORD),
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_ALIGNMENT, alignment)
  ]
  if color:
    styles.append((rl.GuiControl.LABEL, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(color)))

  with GuiStyleContext(styles):
    rl.gui_label(rect, text)
