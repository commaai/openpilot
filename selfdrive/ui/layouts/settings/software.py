from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog
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

  def _on_download_update(self): pass
  def _on_install_update(self): pass
  def _on_select_branch(self): pass

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result):
      if result == DialogResult.CONFIRM:
        self._params.put_bool("DoUninstall", True)

    gui_app.set_modal_overlay(
      lambda: confirm_dialog("Are you sure you want to uninstall?", "Uninstall"),
      callback=handle_uninstall_confirmation,
    )
