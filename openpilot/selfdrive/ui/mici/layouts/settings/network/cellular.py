
from openpilot.cereal import log
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.hardware import HARDWARE
from openpilot.system.ui.lib.cellular_manager import CellularManager

NetworkStrength = log.DeviceState.NetworkStrength
NetworkType = log.DeviceState.NetworkType


class EsimNetworkButton(BigButton):
  def __init__(self, cellular_manager: CellularManager, *, description: str = ""):
    self._cellular_manager = cellular_manager
    self._cell_icons = {
      NetworkStrength.unknown: gui_app.texture("icons_mici/settings/network/cell_strength_none.png", 64, 47),
      NetworkStrength.poor: gui_app.texture("icons_mici/settings/network/cell_strength_low.png", 64, 47),
      NetworkStrength.moderate: gui_app.texture("icons_mici/settings/network/cell_strength_medium.png", 64, 47),
      NetworkStrength.good: gui_app.texture("icons_mici/settings/network/cell_strength_high.png", 64, 47),
      NetworkStrength.great: gui_app.texture("icons_mici/settings/network/cell_strength_full.png", 64, 47),
    }
    super().__init__("esim", "loading...", self._cell_icons[NetworkStrength.unknown], scroll=True, description=description)

  def _update_state(self):
    super()._update_state()
    self.set_enabled(self._cellular_manager.is_euicc is not False)
    text, value, icon = self._compute_state()
    self.set_text(text)
    self.set_value(value)
    self.set_icon(icon)

  def _compute_state(self):
    cm = self._cellular_manager
    none_icon = self._cell_icons[NetworkStrength.unknown]
    ip = cm.modem_state.get("ip_address") or "connecting..."
    if cm.is_euicc is False:
      iccid = cm.modem_state.get("iccid") or ""
      if not iccid:
        return "sim", "no sim", none_icon
      return f"sim (...{iccid[-4:]})", ip, self._cell_icon()

    active = cm.active_profile
    if active is None:
      return "esim", "loading...", none_icon
    return active.display_name, ip, self._cell_icon()

  def _cell_icon(self):
    # read directly from HARDWARE so it reflects modem state even when wifi is the active connection
    strength = HARDWARE.get_network_strength(NetworkType.cell4G)
    return self._cell_icons.get(strength, self._cell_icons[NetworkStrength.unknown])


