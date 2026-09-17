from collections.abc import Sequence
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BaseButton


class SettingsPanel(NavScroller):
  def add_widgets(self, items: Sequence[Widget]) -> None:
    # Settings without a description shake on long press
    for item in items:
      if isinstance(item, BaseButton):
        item.enable_long_press_shake()
    self._scroller.add_widgets(items)
