import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout

PAIR_MESSAGE_FONT_SIZE = 24
PAIR_MESSAGE_MARGIN = 16
PAIR_MESSAGE_WIDTH = 260


class MiciBodyHomeLayout(MiciHomeLayout):
  def __init__(self):
    super().__init__()
    self.set_visible(False)
    self._branch_label.set_visible(False)
    self._pair_message = UnifiedLabel("", font_size=36, text_color=rl.WHITE, font_weight=FontWeight.ROMAN, scroll=True)

  def _render(self, rect: rl.Rectangle):
    super()._render(rect)
    # TODO: replace this with nice icon
    rl.draw_rectangle_rec(self._branch_label._rect, rl.BLACK)
    self._pair_message.set_max_width(self._branch_label._max_width)
    self._pair_message._rect = self._branch_label._rect
    self._pair_message.set_text(" comma body ")
    self._pair_message.render()
