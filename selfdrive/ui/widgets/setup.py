import pyray as rl
from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.confirm_dialog import alert_dialog
from openpilot.system.ui.widgets.button import Button, ButtonStyle


class SetupWidget(Widget):
  def __init__(self):
    super().__init__()
    self._open_settings_callback = None
    self._pair_device_btn = Button(lambda: tr("Pair device"), self._show_pairing, button_style=ButtonStyle.PRIMARY)

  def set_open_settings_callback(self, callback):
    self._open_settings_callback = callback

  def _render(self, rect: rl.Rectangle):
    self._render_registration(rect)

  def _render_registration(self, rect: rl.Rectangle):
    """Render registration prompt."""

    rl.draw_rectangle_rounded(rl.Rectangle(rect.x, rect.y, rect.width, rect.height), 0.03, 20, rl.Color(51, 51, 51, 255))

    x = rect.x + 64
    y = rect.y + 48
    w = rect.width - 128

    # Title
    font = gui_app.font(FontWeight.BOLD)
    rl.draw_text_ex(font, tr("Finish Setup"), rl.Vector2(x, y), 75, 0, rl.WHITE)
    y += 113  # 75 + 38 spacing

    # Description
    desc = tr("Pair your device with Asius Connect to manage it from the app.")
    light_font = gui_app.font(FontWeight.NORMAL)
    wrapped = wrap_text(light_font, desc, 50, int(w))
    for line in wrapped:
      rl.draw_text_ex(light_font, line, rl.Vector2(x, y), 50, 0, rl.WHITE)
      y += 50 * FONT_SCALE

    button_rect = rl.Rectangle(x, y + 30, w, 200)
    self._pair_device_btn.render(button_rect)

  @staticmethod
  def _show_pairing():
    if not system_time_valid():
      dlg = alert_dialog(tr("Please connect to Wi-Fi to complete initial pairing"))
      gui_app.push_widget(dlg)
      return

    gui_app.push_widget(PairingDialog())
