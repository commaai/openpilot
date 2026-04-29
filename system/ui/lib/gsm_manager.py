"""GSM/cellular settings manager via NetworkManager DBus."""
from jeepney import DBusAddress, new_method_call
from jeepney.io.threading import DBusRouter, open_dbus_connection
from jeepney.low_level import MessageType
from jeepney.wrappers import Properties

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.networkmanager import (NM, NM_PATH, NM_SETTINGS_PATH, NM_SETTINGS_IFACE,
                                                    NM_CONNECTION_IFACE, NM_DEVICE_IFACE, NM_DEVICE_TYPE_MODEM)


# NM NMMetered values
# https://networkmanager.dev/docs/api/latest/nm-dbus-types.html#NMMetered
NM_METERED_UNKNOWN = 0
NM_METERED_NO = 2


class _GsmManager:
  """Manages cellular/GSM via NetworkManager DBus."""

  def __init__(self):
    self._router: DBusRouter | None = None
    self._nm = DBusAddress(NM_PATH, bus_name=NM, interface=NM)

  def _ensure_router(self) -> bool:
    if self._router is not None:
      return True
    try:
      self._router = DBusRouter(open_dbus_connection(bus="SYSTEM"))
      return True
    except Exception:
      cloudlog.exception("Failed to connect to system D-Bus for GSM")
      return False

  def update_gsm_settings(self, roaming: bool, apn: str, metered: bool):
    """Update GSM settings for cellular connection"""
    if not self._ensure_router():
      return
    try:
      lte_connection_path = self._get_lte_connection_path()
      if not lte_connection_path:
        cloudlog.warning("No LTE connection found")
        return

      settings = self._get_connection_settings(lte_connection_path)

      if len(settings) == 0:
        cloudlog.warning(f"Failed to get connection settings for {lte_connection_path}")
        return

      # Ensure dicts exist
      if 'gsm' not in settings:
        settings['gsm'] = {}
      if 'connection' not in settings:
        settings['connection'] = {}

      changes = False
      auto_config = apn == ""

      if settings['gsm'].get('auto-config', ('b', False))[1] != auto_config:
        cloudlog.warning(f'Changing gsm.auto-config to {auto_config}')
        settings['gsm']['auto-config'] = ('b', auto_config)
        changes = True

      if settings['gsm'].get('apn', ('s', ''))[1] != apn:
        cloudlog.warning(f'Changing gsm.apn to {apn}')
        settings['gsm']['apn'] = ('s', apn)
        changes = True

      if settings['gsm'].get('home-only', ('b', False))[1] == roaming:
        cloudlog.warning(f'Changing gsm.home-only to {not roaming}')
        settings['gsm']['home-only'] = ('b', not roaming)
        changes = True

      # Unknown means NetworkManager decides
      metered_int = NM_METERED_UNKNOWN if metered else NM_METERED_NO
      if settings['connection'].get('metered', ('i', 0))[1] != metered_int:
        cloudlog.warning(f'Changing connection.metered to {metered_int}')
        settings['connection']['metered'] = ('i', metered_int)
        changes = True

      if changes:
        # Update the connection settings (temporary update)
        conn_addr = DBusAddress(lte_connection_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
        reply = self._router.send_and_get_reply(new_method_call(conn_addr, 'UpdateUnsaved', 'a{sa{sv}}', (settings,)))

        if reply.header.message_type == MessageType.error:
          cloudlog.warning(f"Failed to update GSM settings: {reply}")
          return

        self._activate_modem_connection(lte_connection_path)
    except Exception as e:
      cloudlog.exception(f"Error updating GSM settings: {e}")

  def _get_lte_connection_path(self) -> str | None:
    try:
      settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
      known_connections = self._router.send_and_get_reply(new_method_call(settings_addr, 'ListConnections')).body[0]

      for conn_path in known_connections:
        settings = self._get_connection_settings(conn_path)
        if settings and settings.get('connection', {}).get('id', ('s', ''))[1] == 'lte':
          return str(conn_path)
    except Exception as e:
      cloudlog.exception(f"Error finding LTE connection: {e}")
    return None

  def _activate_modem_connection(self, connection_path: str):
    try:
      modem_device = self._get_adapter(NM_DEVICE_TYPE_MODEM)
      if modem_device and connection_path:
        self._router.send_and_get_reply(new_method_call(self._nm, 'ActivateConnection', 'ooo',
                                                        (connection_path, modem_device, "/")))
    except Exception as e:
      cloudlog.exception(f"Error activating modem connection: {e}")

  def _get_connection_settings(self, conn_path: str) -> dict:
    conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
    reply = self._router.send_and_get_reply(new_method_call(conn_addr, 'GetSettings'))
    if reply.header.message_type == MessageType.error:
      cloudlog.warning(f'Failed to get connection settings: {reply}')
      return {}
    return dict(reply.body[0])

  def _get_adapter(self, adapter_type: int) -> str | None:
    # Return the first NetworkManager device path matching adapter_type
    try:
      device_paths = self._router.send_and_get_reply(new_method_call(self._nm, 'GetDevices')).body[0]
      for device_path in device_paths:
        dev_addr = DBusAddress(device_path, bus_name=NM, interface=NM_DEVICE_IFACE)
        dev_type = self._router.send_and_get_reply(Properties(dev_addr).get('DeviceType')).body[0][1]
        if dev_type == adapter_type:
          return str(device_path)
    except Exception as e:
      cloudlog.exception(f"Error getting adapter type {adapter_type}: {e}")
    return None

  def close(self):
    if self._router is not None:
      try:
        self._router.close()
        self._router.conn.close()
      except Exception:
        pass
      self._router = None
