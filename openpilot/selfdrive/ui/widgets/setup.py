import pyray as rl
from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.confirm_dialog import alert_dialog
from openpilot.system.ui.widgets.button import Button, ButtonStyle


class SetupWidget(Widget):
  def __init__(self):
    super().__init__()
    self._open_settings_callback = None
    self._pair_device_btn = Button(lambda: tr("Pair device"), self._show_pairing, button_style=ButtonStyle.PRIMARY)
    self._open_settings_btn = Button(lambda: tr("Open"), lambda: self._open_settings_callback() if self._open_settings_callback else None,
                                     button_style=ButtonStyle.PRIMARY)
    self._fire_icon = gui_app.texture("icons/fire.png", 64, 64)

  def set_open_settings_callback(self, callback):
    self._open_settings_callback = callback

  def _render(self, rect: rl.Rectangle):
    if not ui_state.prime_state.is_paired():
      self._render_registration(rect)
    else:
      self._render_firehose_prompt(rect)

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
    desc = tr("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer.")
    light_font = gui_app.font(FontWeight.NORMAL)
    wrapped = wrap_text(light_font, desc, 50, int(w))
    for line in wrapped:
      rl.draw_text_ex(light_font, line, rl.Vector2(x, y), 50, 0, rl.WHITE)
      y += 50 * FONT_SCALE

    button_rect = rl.Rectangle(x, y + 30, w, 200)
    self._pair_device_btn.render(button_rect)

  def _render_firehose_prompt(self, rect: rl.Rectangle):
    """Render firehose prompt widget."""

    rl.draw_rectangle_rounded(rl.Rectangle(rect.x, rect.y, rect.width, 500), 0.04, 20, rl.Color(51, 51, 51, 255))

    # Content margins (56, 40, 56, 40)
    x = rect.x + 56
    y = rect.y + 40
    w = rect.width - 112
    spacing = 42

    # Title with fire icons
    title_text = tr("Firehose Mode")
    title_font = gui_app.font(FontWeight.MEDIUM)
    title_size = measure_text_cached(title_font, title_text, 64)
    icon = self._fire_icon
    icon_size = 64 * FONT_SCALE
    icon_gap = 15
    total_width = icon_size + icon_gap + title_size.x + icon_gap + icon_size
    title_y = y + (64 - icon_size) / 2
    title_x = rect.x + (rect.width - total_width) / 2
    rl.draw_texture_ex(icon, rl.Vector2(title_x, title_y), 0.0, FONT_SCALE, rl.WHITE)
    title_x += icon_size + icon_gap
    rl.draw_text_ex(title_font, title_text, rl.Vector2(title_x, title_y), 64, 0, rl.WHITE)
    title_x += title_size.x + icon_gap
    rl.draw_texture_ex(icon, rl.Vector2(title_x, title_y), 0.0, FONT_SCALE, rl.WHITE)
    y += 64 + spacing

    # Description
    desc_font = gui_app.font(FontWeight.NORMAL)
    desc_text = tr("Maximize your training data uploads to improve openpilot's driving models.")
    wrapped_desc = wrap_text(desc_font, desc_text, 40, int(w))

    for line in wrapped_desc:
      rl.draw_text_ex(desc_font, line, rl.Vector2(x, y), 40, 0, rl.WHITE)
      y += 40 * FONT_SCALE

    y += spacing

    # Open button
    button_height = 48 + 64  # font size + padding
    button_rect = rl.Rectangle(x, y, w, button_height)
    self._open_settings_btn.render(button_rect)

  @staticmethod
  def _show_pairing():
    if not system_time_valid():
      dlg = alert_dialog(tr("Please connect to Wi-Fi to complete initial pairing"))
      gui_app.push_widget(dlg)
      return

    gui_app.push_widget(PairingDialog())
