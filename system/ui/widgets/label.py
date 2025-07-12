import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_TEXT_SIZE, DEFAULT_TEXT_COLOR, DEFAULT_TEXT_SPACING
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.utils import GuiStyleContext
from openpilot.system.ui.widgets import Widget


class Label(Widget):
  def __init__(
    self,
    text: str,
    font_size: int = DEFAULT_TEXT_SIZE,
    font_weight: FontWeight = FontWeight.NORMAL,
    color: rl.Color = DEFAULT_TEXT_COLOR,
    alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
    alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
    spacing: int = DEFAULT_TEXT_SPACING,
    truncated: bool = True,
    truncate_suffix: str = "...",
  ):
    """Text label widget with customizable text properties (font size, font weight, color, alignment, spacing, etc.).
    The `truncated` parameter (default "true") controls whether the text is cut on the right if it exceeds the widget's width,
    while the `truncate_suffix` parameter (default "...") controls the text suffix that is displayed when the text is truncated.
    """
    super().__init__()
    self.text = text
    self.font_size = font_size
    self.font_weight = font_weight
    self.color = color
    self.alignment = alignment
    self.alignment_vertical = alignment_vertical
    self.spacing = spacing
    self.truncated = truncated
    self.truncate_suffix = truncate_suffix

  def _render(self, rect: rl.Rectangle):
    font = gui_app.font(self.font_weight)
    text_size = measure_text_cached(font, self.text, self.font_size)
    display_text = self.text

    # Truncate text to fit within the rectangle if needed
    if self.truncated and text_size.x > rect.width:
      left, right = 0, len(self.text)
      while left < right:
        mid = (left + right) // 2
        candidate = self.text[:mid] + self.truncate_suffix
        candidate_size = measure_text_cached(font, candidate, self.font_size)
        if candidate_size.x <= rect.width:
          left = mid + 1
        else:
          right = mid
      display_text = self.text[: left - 1] + self.truncate_suffix if left > 0 else self.truncate_suffix
      text_size = measure_text_cached(font, display_text, self.font_size)

    # Calculate horizontal position based on alignment
    horizontal_alignment_map = {
      rl.GuiTextAlignment.TEXT_ALIGN_LEFT: 0,
      rl.GuiTextAlignment.TEXT_ALIGN_CENTER: (rect.width - text_size.x) / 2,
      rl.GuiTextAlignment.TEXT_ALIGN_RIGHT: rect.width - text_size.x,
    }
    text_x = rect.x + horizontal_alignment_map.get(self.alignment, 0)

    # Calculate vertical position based on alignment
    vertical_alignment_map = {
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP: 0,
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE: (rect.height - text_size.y) / 2,
      rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM: rect.height - text_size.y,
    }
    text_y = rect.y + vertical_alignment_map.get(self.alignment_vertical, 0)

    # Draw the text in the specified rectangle
    rl.draw_text_ex(font, display_text, rl.Vector2(text_x, text_y), self.font_size, self.spacing, self.color)


# TODO: Remove
def gui_label(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_TEXT_SIZE,
  color: rl.Color = DEFAULT_TEXT_COLOR,
  font_weight: FontWeight = FontWeight.NORMAL,
  alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
  alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
  elide_right: bool = True,
):
  font = gui_app.font(font_weight)
  text_size = measure_text_cached(font, text, font_size)
  display_text = text

  # Elide text to fit within the rectangle
  if elide_right and text_size.x > rect.width:
    ellipsis = "..."
    left, right = 0, len(text)
    while left < right:
      mid = (left + right) // 2
      candidate = text[:mid] + ellipsis
      candidate_size = measure_text_cached(font, candidate, font_size)
      if candidate_size.x <= rect.width:
        left = mid + 1
      else:
        right = mid
    display_text = text[: left - 1] + ellipsis if left > 0 else ellipsis
    text_size = measure_text_cached(font, display_text, font_size)

  # Calculate horizontal position based on alignment
  text_x = rect.x + {
    rl.GuiTextAlignment.TEXT_ALIGN_LEFT: 0,
    rl.GuiTextAlignment.TEXT_ALIGN_CENTER: (rect.width - text_size.x) / 2,
    rl.GuiTextAlignment.TEXT_ALIGN_RIGHT: rect.width - text_size.x,
  }.get(alignment, 0)

  # Calculate vertical position based on alignment
  text_y = rect.y + {
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP: 0,
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE: (rect.height - text_size.y) / 2,
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM: rect.height - text_size.y,
  }.get(alignment_vertical, 0)

  # Draw the text in the specified rectangle
  rl.draw_text_ex(font, display_text, rl.Vector2(text_x, text_y), font_size, 0, color)


# TODO: Remove
def gui_text_box(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_TEXT_SIZE,
  color: rl.Color = DEFAULT_TEXT_COLOR,
  alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
  alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
  font_weight: FontWeight = FontWeight.NORMAL,
):
  """Draw text with word wrapping and alignment."""
  if font_weight != FontWeight.NORMAL:
    rl.gui_set_font(gui_app.font(font_weight))

  styles = [
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(color)),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_LINE_SPACING, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_ALIGNMENT, alignment),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, alignment_vertical),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_WRAP_MODE, rl.GuiTextWrapMode.TEXT_WRAP_WORD),
  ]

  with GuiStyleContext(styles):
    rl.gui_label(rect, text)

  if font_weight != FontWeight.NORMAL:
    rl.gui_set_font(gui_app.font(FontWeight.NORMAL))


class Text(Widget):
  def __init__(
    self,
    text: str,
    font_size: int = DEFAULT_TEXT_SIZE,
    font_weight: FontWeight = FontWeight.NORMAL,
    color: rl.Color = DEFAULT_TEXT_COLOR,
    alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
    alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
    spacing: int = DEFAULT_TEXT_SPACING,
    wrap_mode: int = rl.GuiTextWrapMode.TEXT_WRAP_WORD,
  ):
    super().__init__()
    self.text = text
    self.font_size = font_size
    self.font_weight = font_weight
    self.color = color
    self.alignment = alignment
    self.alignment_vertical = alignment_vertical
    self.spacing = spacing
    self.wrap_mode = wrap_mode

  def _get_styles(self):
    """Return the custom styles for this widget."""
    return [
      (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(self.color)),
      (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, self.font_size),
      (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_LINE_SPACING, self.font_size),
      (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_ALIGNMENT, self.alignment),
      (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, self.alignment_vertical),
      (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SPACING, self.spacing),
      (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_WRAP_MODE, self.wrap_mode),
    ]

  def _render(self, rect: rl.Rectangle):
    # Override font for this render
    if self.font_weight != FontWeight.NORMAL:
      rl.gui_set_font(gui_app.font(self.font_weight))

    ret = None
    # Render the label with the custom styles (overrides the default styles)
    with GuiStyleContext(self._get_styles()):
      ret = rl.gui_label(rect, self.text)

    # Restore default font
    if self.font_weight != FontWeight.NORMAL:
      rl.gui_set_font(gui_app.font(FontWeight.NORMAL))

    return ret
