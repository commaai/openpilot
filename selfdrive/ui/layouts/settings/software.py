from openpilot.common.params import Params
import os
import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item, text_item
from openpilot.system.ui.widgets.scroller import Scroller


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._init_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _init_items(self):
    items = [
      text_item("Current Version", ""),
      button_item("Download", "CHECK", callback=self._on_download_update),
      button_item("Install Update", "INSTALL", callback=self._on_install_update),
      button_item("Target Branch", "SELECT", callback=self._on_select_branch),
      button_item("Uninstall", "UNINSTALL", callback=self._on_uninstall),
    ]
    return items

  def _render(self, rect):
    self._scroller.render(rect)
    if os.getenv("DEBUG_TEXT_COMPARE") == "1":
      self._draw_text_compare()

  def _on_download_update(self): pass
  def _on_install_update(self): pass
  def _on_select_branch(self): pass

  def _draw_text_compare(self):
    try:
      font = gui_app.font(FontWeight.NORMAL)
      samples = [(50, "Hg"), (40, "Hg"), (35, "Hg")]
      left_x = 40
      top_y = 40
      row_h = 120

      for i, (size, text) in enumerate(samples):
        y = top_y + i * row_h
        rl.draw_line_ex(rl.Vector2(left_x, y), rl.Vector2(left_x, y + size), 2, rl.RED)
        text_pos = rl.Vector2(left_x + 10, y)
        rl.draw_text_ex(font, text, text_pos, size, 0, rl.WHITE)
        ts = measure_text_cached(font, text, size)
        rl.draw_rectangle_lines(int(text_pos.x), int(text_pos.y), int(ts.x), int(ts.y), rl.GREEN)
    except Exception:
      pass

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result):
      if result == DialogResult.CONFIRM:
        self._params.put_bool("DoUninstall", True)

    dialog = ConfirmDialog("Are you sure you want to uninstall?", "Uninstall")
    gui_app.set_modal_overlay(dialog, callback=handle_uninstall_confirmation)
