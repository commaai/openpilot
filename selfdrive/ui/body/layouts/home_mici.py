import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout

PAIR_MESSAGE_FONT_SIZE = 24
PAIR_MESSAGE_MARGIN = 16
PAIR_MESSAGE_WIDTH = 260


class MiciBodyHomeLayout(MiciHomeLayout):
  def __init__(self):
    super().__init__()
    self._branch_label.set_visible(False)
    self._pair_message = self._child(UnifiedLabel("", font_size=PAIR_MESSAGE_FONT_SIZE, font_weight=FontWeight.SEMI_BOLD,
                                                  text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                                  alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                                                  alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM,
                                                  text_padding=0, elide=False, wrap_text=True, line_height=0.95))

  def _get_pair_message(self) -> str:
    if not ui_state.prime_state.is_paired():
      return "pair to this device in settings"
    if ui_state.ignition:
      return "use at: connect.comma.ai"
    return "turn on ignition to use"

  def _render(self, rect: rl.Rectangle):
    super()._render(rect)
    self._pair_message.set_text(self._get_pair_message())
    msg_h = int(self._pair_message.get_content_height(PAIR_MESSAGE_WIDTH) + 6)
    msg_x = self._rect.x + self._rect.width - PAIR_MESSAGE_WIDTH - PAIR_MESSAGE_MARGIN
    msg_y = self._rect.y + self._rect.height - msg_h - PAIR_MESSAGE_MARGIN
    self._pair_message.render(rl.Rectangle(msg_x, msg_y, PAIR_MESSAGE_WIDTH, msg_h))
