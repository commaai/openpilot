from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.network.network_layout import NetworkLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici
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
    toggles_btn = SettingsBigButton(
      'toggles',
      '',
      gui_app.texture('icons_mici/settings.png', 64, 64),
      description='Adjust driving features, recording, and display units.',
    )
    toggles_btn.set_click_callback(lambda: gui_app.push_widget(toggles_panel))

    network_panel = NetworkLayoutMici()
    network_btn = SettingsBigButton(
      'network',
      '',
      gui_app.texture('icons_mici/settings/network/wifi_strength_full.png', 76, 56),
      description='Manage Wi-Fi, cellular connectivity, and tethering.',
    )
    network_btn.set_click_callback(lambda: gui_app.push_widget(network_panel))

    device_panel = DeviceLayoutMici()
    device_btn = SettingsBigButton(
      'device',
      '',
      gui_app.texture('icons_mici/settings/device_icon.png', 72, 58),
      description='Pair your device, review training, manage calibration, and access power controls.',
    )
    device_btn.set_click_callback(lambda: gui_app.push_widget(device_panel))

    software_panel = SoftwareLayoutMici()
    software_btn = SettingsBigButton(
      'software',
      '',
      gui_app.texture('icons_mici/settings/software.png', 64, 75),
      description='Check for updates, select a branch, or uninstall openpilot.',
    )
    software_btn.set_click_callback(lambda: gui_app.push_widget(software_panel))

    developer_panel = DeveloperLayoutMici()
    developer_btn = SettingsBigButton(
      'developer',
      '',
      gui_app.texture('icons_mici/settings/developer_icon.png', 64, 60),
      description='Configure remote access and development tools.',
    )
    developer_btn.set_click_callback(lambda: gui_app.push_widget(developer_panel))

    firehose_panel = FirehoseLayout()
    firehose_btn = SettingsBigButton(
      'firehose',
      '',
      gui_app.texture('icons_mici/settings/firehose.png', 52, 62),
      description='Upload more driving data to help improve openpilot.',
    )
    firehose_btn.set_click_callback(lambda: gui_app.push_widget(firehose_panel))

    self._scroller.add_widgets([
      toggles_btn,
      network_btn,
      device_btn,
      software_btn,
      firehose_btn,
      developer_btn,
    ])

    self._font_medium = gui_app.font(FontWeight.MEDIUM)
