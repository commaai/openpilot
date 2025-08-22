# NetworkManager constants
NM = "org.freedesktop.NetworkManager"
NM_PATH = '/org/freedesktop/NetworkManager'
NM_IFACE = 'org.freedesktop.NetworkManager'
NM_SETTINGS_PATH = '/org/freedesktop/NetworkManager/Settings'
NM_SETTINGS_IFACE = 'org.freedesktop.NetworkManager.Settings'
NM_CONNECTION_IFACE = 'org.freedesktop.NetworkManager.Settings.Connection'
NM_WIRELESS_IFACE = 'org.freedesktop.NetworkManager.Device.Wireless'
NM_PROPERTIES_IFACE = 'org.freedesktop.DBus.Properties'
NM_DEVICE_IFACE = "org.freedesktop.NetworkManager.Device"

NM_DEVICE_STATE_UNKNOWN = 0
NM_DEVICE_STATE_ACTIVATED = 100
NM_DEVICE_STATE_NEED_AUTH = 60
NM_DEVICE_TYPE_WIFI = 2
NM_DEVICE_TYPE_MODEM = 8
NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT = 8
DBUS_TIMEOUT = 100

# https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NM80211ApFlags
NM_802_11_AP_FLAGS_NONE = 0x0
NM_802_11_AP_FLAGS_PRIVACY = 0x1
NM_802_11_AP_FLAGS_WPS = 0x2

# https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NM80211ApSecurityFlags
NM_802_11_AP_SEC_PAIR_WEP40 = 0x00000001
NM_802_11_AP_SEC_PAIR_WEP104 = 0x00000002
NM_802_11_AP_SEC_GROUP_WEP40 = 0x00000010
NM_802_11_AP_SEC_GROUP_WEP104 = 0x00000020
NM_802_11_AP_SEC_KEY_MGMT_PSK = 0x00000100
NM_802_11_AP_SEC_KEY_MGMT_802_1X = 0x00000200
