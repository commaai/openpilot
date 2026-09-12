from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.network.network_layout import NetworkLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.device.device_layout import DeviceLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.software import SoftwareLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.firehose import FirehoseLayout
from openpilot.system.ui.lib.application import gui_app, FontWeight


class SettingsBigButton(BigButton):
  def _get_label_font_size(self):
    return 64


class SettingsLayout(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = Params()

    toggles_panel = TogglesLayoutMici()
    toggles_btn = SettingsBigButton("toggles", "", gui_app.texture("icons_mici/settings.png", 64, 64))
    toggles_btn.set_click_callback(lambda: gui_app.push_widget(toggles_panel))

    network_panel = NetworkLayoutMici()
    network_btn = SettingsBigButton("network", "", gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 76, 56))
    network_btn.set_click_callback(lambda: gui_app.push_widget(network_panel))

    self._device_panel = DeviceLayoutMici()
    self._device_button = SettingsBigButton("device", "", gui_app.texture("icons_mici/settings/device_icon.png", 72, 58))
    self._device_button.set_click_callback(self._open_device)

    software_panel = SoftwareLayoutMici()
    software_btn = SettingsBigButton("software", "", gui_app.texture("icons_mici/settings/software.png", 64, 75))
    software_btn.set_click_callback(lambda: gui_app.push_widget(software_panel))

    developer_panel = DeveloperLayoutMici()
    developer_btn = SettingsBigButton("developer", "", gui_app.texture("icons_mici/settings/developer_icon.png", 64, 60))
    developer_btn.set_click_callback(lambda: gui_app.push_widget(developer_panel))

    firehose_panel = FirehoseLayout()
    firehose_btn = SettingsBigButton("firehose", "", gui_app.texture("icons_mici/settings/firehose.png", 52, 62))
    firehose_btn.set_click_callback(lambda: gui_app.push_widget(firehose_panel))

    self._scroller.add_widgets([
      toggles_btn,
      network_btn,
      self._device_button,
      software_btn,
      firehose_btn,
      developer_btn,
    ])

    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def show_pairing(self):
    gui_app.push_widget(self)
    # Keep Settings in the back stack without showing its entrance animation.
    self._y_pos_filter.x = 0.0
    self.set_visible(lambda: self.enabled or self._device_panel.is_dismissing)
    self._scroller._layout()
    offset = (self._device_button.rect.x + self._device_button.rect.width / 2) - (self._rect.x + self._rect.width / 2)
    self._scroller.scroll_to(offset, smooth=False)
    self._open_device()
    self._device_panel.set_shown_callback(self._on_pairing_shown)

  def _on_pairing_shown(self):
    self.set_visible(True)
    self._device_panel.scroll_to_pairing()

  def _open_device(self, highlight_pairing: bool = False):
    if highlight_pairing:
      self._device_panel.set_shown_callback(self._device_panel.scroll_to_pairing)
    gui_app.push_widget(self._device_panel)

  def _update_state(self):
    super()._update_state()
    # Also restore Settings if the user dismisses Device before it finishes entering.
    if self.enabled:
      self.set_visible(True)
