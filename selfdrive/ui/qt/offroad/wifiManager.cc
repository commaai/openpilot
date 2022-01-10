#include "selfdrive/ui/qt/offroad/wifiManager.h"

#include <arpa/inet.h>
#include <algorithm>

#include <QHostAddress>
#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/util.h"

WifiManager::WifiManager(QObject *parent) : QObject(parent) {
  qDBusRegisterMetaType<Connection>();
  qDBusRegisterMetaType<IpConfig>();

  // Set tethering ssid as "weedle" + first 4 characters of a dongle id
  tethering_ssid = "weedle";
  if (auto dongle_id = getDongleId()) {
    tethering_ssid += "-" + dongle_id->left(4);
  }

  adapter = getAdapter();
  if (!adapter.isEmpty()) {
    setup();
  } else {
    bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE, "DeviceAdded", this, SLOT(deviceAdded(QDBusObjectPath)));
  }
  timer.callOnTimeout(this, &WifiManager::requestScan);
}

void WifiManager::setup() {
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE, "StateChanged", this, SLOT(stateChange(unsigned int, unsigned int, unsigned int)));
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, "PropertiesChanged", this, SLOT(propertyChange(QString, QVariantMap, QStringList)));

  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "ConnectionRemoved", this, SLOT(connectionRemoved(QDBusObjectPath)));
  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "NewConnection", this, SLOT(newConnection(QDBusObjectPath)));

  activeAp = call<QDBusObjectPath>(adapter, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_DEVICE_WIRELESS, "ActiveAccessPoint").path();

  QDBusReply<QMap<QString, QVariant>> replay = call(adapter, NM_DBUS_INTERFACE_PROPERTIES, "GetAll", NM_DBUS_INTERFACE_DEVICE);
  auto properties = replay.value();
  raw_adapter_state = properties["State"].toUInt();
  setIP4Address(properties["Ip4Address"].toUInt());

  initConnections();
  requestScan();
}

void WifiManager::start() {
  timer.start(5000);
  refreshNetworks();
}

void WifiManager::stop() {
  timer.stop();
}

void WifiManager::refreshNetworks() {
  if (adapter.isEmpty() || !timer.isActive()) return;

  QDBusInterface nm = QDBusInterface(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, bus);
  nm.setTimeout(DBUS_TIMEOUT);
  QDBusPendingCall pending_call = nm.asyncCall("GetAllAccessPoints");
  QDBusPendingCallWatcher *watcher = new QDBusPendingCallWatcher(pending_call);
  QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, &WifiManager::activationFinished);
}

void WifiManager::activationFinished(QDBusPendingCallWatcher *watcher) {
  seenNetworks.clear();
  const QDBusReply<QList<QDBusObjectPath>> wather_reply = *watcher;

  for (const QDBusObjectPath &path : wather_reply.value()) {
    QDBusReply<QMap<QString, QVariant>> replay = call(path.path(), NM_DBUS_INTERFACE_PROPERTIES, "GetAll", NM_DBUS_INTERFACE_ACCESS_POINT);
    auto properties = replay.value();
    const QByteArray ssid = properties["Ssid"].toByteArray();
    uint32_t strength = properties["Strength"].toUInt();
    if (ssid.isEmpty() || (seenNetworks.contains(ssid) && strength <= seenNetworks[ssid].strength)) continue;

    SecurityType security = getSecurityType(properties);
    ConnectedType ctype = ConnectedType::DISCONNECTED;
    if (path.path() == activeAp) {
      ctype = (ssid == connecting_to_network) ? ConnectedType::CONNECTING : ConnectedType::CONNECTED;
    }
    seenNetworks[ssid] = {ssid, strength, ctype, security};
  }

  emit refreshSignal();
  watcher->deleteLater();
}

void WifiManager::setIP4Address(uint address) {
  ipv4_address = QHostAddress(htonl(address)).toString();
  emit ipAddressChanged(ipv4_address);
}

SecurityType WifiManager::getSecurityType(const QMap<QString,QVariant> &properties) {
  int sflag = properties["Flags"].toUInt();
  int wpaflag = properties["WpaFlags"].toUInt();
  int rsnflag = properties["RsnFlags"].toUInt();
  int wpa_props = wpaflag | rsnflag;

  // obtained by looking at flags of networks in the office as reported by an Android phone
  const int supports_wpa = NM_802_11_AP_SEC_PAIR_WEP40 | NM_802_11_AP_SEC_PAIR_WEP104 | NM_802_11_AP_SEC_GROUP_WEP40 | NM_802_11_AP_SEC_GROUP_WEP104 | NM_802_11_AP_SEC_KEY_MGMT_PSK;

  if ((sflag == NM_802_11_AP_FLAGS_NONE) || ((sflag & NM_802_11_AP_FLAGS_WPS) && !(wpa_props & supports_wpa))) {
    return SecurityType::OPEN;
  } else if ((sflag & NM_802_11_AP_FLAGS_PRIVACY) && (wpa_props & supports_wpa) && !(wpa_props & NM_802_11_AP_SEC_KEY_MGMT_802_1X)) {
    return SecurityType::WPA;
  } else {
    LOGW("Unsupported network! sflag: %d, wpaflag: %d, rsnflag: %d", sflag, wpaflag, rsnflag);
    return SecurityType::UNSUPPORTED;
  }
}

void WifiManager::connect(const Network &n, const QString &password, const QString &username) {
  connecting_to_network = n.ssid;
  // disconnect();
  forgetConnection(n.ssid); //Clear all connections that may already exist to the network we are connecting
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "openpilot connection "+QString::fromStdString(n.ssid.toStdString());
  connection["connection"]["autoconnect-retries"] = 0;

  connection["802-11-wireless"]["ssid"] = n.ssid;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  if (n.security_type == SecurityType::WPA) {
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["auth-alg"] = "open";
    connection["802-11-wireless-security"]["psk"] = password;
  }

  connection["ipv4"]["method"] = "auto";
  connection["ipv4"]["dns-priority"] = 600;
  connection["ipv6"]["method"] = "ignore";

  call(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "AddConnection", QVariant::fromValue(connection));
}

void WifiManager::deactivateConnectionBySsid(const QString &ssid) {
  for (QDBusObjectPath active_connection : get_active_connections()) {
    auto pth = call<QDBusObjectPath>(active_connection.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "SpecificObject");
    if (pth.path() != "" && pth.path() != "/" && get_property(pth.path(), "Ssid") == ssid) {
      deactivateConnection(active_connection);
    }
  }
}

void WifiManager::deactivateConnection(const QDBusObjectPath &path) {
  call(NM_DBUS_PATH, NM_DBUS_INTERFACE, "DeactivateConnection", QVariant::fromValue(path));
}

QVector<QDBusObjectPath> WifiManager::get_active_connections() {
  QVector<QDBusObjectPath> conns;
  QDBusObjectPath path;
  const QDBusArgument &arr = call<QDBusArgument>(NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE, "ActiveConnections");
  arr.beginArray();
  while (!arr.atEnd()) {
    arr >> path;
    conns.push_back(path);
  }
  arr.endArray();
  return conns;
}

bool WifiManager::isKnownConnection(const QString &ssid) {
  return (bool)getConnectionPath(ssid);
}

void WifiManager::forgetConnection(const QString &ssid) {
  if (auto path = getConnectionPath(ssid)) {
    call(path->path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "Delete");
  }
}

uint WifiManager::getAdapterType(const QDBusObjectPath &path) {
  return call<uint>(path.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_DEVICE, "DeviceType");
}

void WifiManager::requestScan() {
  if (!adapter.isEmpty()) {
    QDBusInterface nm = QDBusInterface(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, bus);
    nm.asyncCall("RequestScan", QVariantMap());
  }
}

QByteArray WifiManager::get_property(const QString &network_path , const QString &property) {
  return call<QByteArray>(network_path, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACCESS_POINT, property);
}

QString WifiManager::getAdapter(const uint adapter_type) {
  const QDBusReply<QList<QDBusObjectPath>> &response = call(NM_DBUS_PATH, NM_DBUS_INTERFACE, "GetDevices");
  for (const QDBusObjectPath &path : response.value()) {
    if (getAdapterType(path) == adapter_type) {
      return path.path();
    }
  }
  return "";
}

void WifiManager::stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason) {
  raw_adapter_state = new_state;
  if (new_state == NM_DEVICE_STATE_NEED_AUTH && change_reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT && !connecting_to_network.isEmpty()) {
    forgetConnection(connecting_to_network);
    emit wrongPassword(connecting_to_network);
  } else if (new_state == NM_DEVICE_STATE_ACTIVATED) {
    connecting_to_network = "";
    refreshNetworks();
  }
}

// https://developer.gnome.org/NetworkManager/stable/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html
void WifiManager::propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props) {
  if (interface == NM_DBUS_INTERFACE_DEVICE_WIRELESS) {
    if (props.contains("LastScan")) {
      refreshNetworks();
    } else if (props.contains("ActiveAccessPoint")) {
      activeAp = props.value("ActiveAccessPoint").value<QDBusObjectPath>().path();
    }
  } else if (props.contains("Ip4Address")) {
    setIP4Address(props["Ip4Address"].toUInt());
  }
}

void WifiManager::deviceAdded(const QDBusObjectPath &path) {
  if (getAdapterType(path) == NM_DEVICE_TYPE_WIFI && (adapter.isEmpty() || adapter == "/")) {
    adapter = path.path();
    setup();
  }
}

void WifiManager::connectionRemoved(const QDBusObjectPath &path) {
  knownConnections.remove(path);
}

void WifiManager::newConnection(const QDBusObjectPath &path) {
  const Connection &settings = getConnectionSettings(path);
  if (settings.value("connection").value("type") == "802-11-wireless") {
    knownConnections[path] = settings.value("802-11-wireless").value("ssid").toString();
    if (knownConnections[path] != tethering_ssid) {
      activateWifiConnection(knownConnections[path]);
    }
  }
}

void WifiManager::disconnect() {
  if (activeAp != "" && activeAp != "/") {
    deactivateConnectionBySsid(get_property(activeAp, "Ssid"));
  }
}

std::optional<QDBusObjectPath> WifiManager::getConnectionPath(const QString &ssid) {
  auto path = knownConnections.key(ssid);
  return path.path().isEmpty() ? std::nullopt : std::make_optional(path);
}

Connection WifiManager::getConnectionSettings(const QDBusObjectPath &path) {
  return QDBusReply<Connection>(call(path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "GetSettings")).value();
}

void WifiManager::initConnections() {
  const QDBusReply<QList<QDBusObjectPath>> response = call(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "ListConnections");
  for (const QDBusObjectPath &path : response.value()) {
    const Connection &settings = getConnectionSettings(path);
    if (settings.value("connection").value("type") == "802-11-wireless") {
      knownConnections[path] = settings.value("802-11-wireless").value("ssid").toString();
    } else if (path.path() != "/") {
      lteConnectionPath = path;
    }
  }
}

void WifiManager::activateWifiConnection(const QString &ssid) {
  if (auto path = getConnectionPath(ssid)) {
    connecting_to_network = ssid;
    call(NM_DBUS_PATH, NM_DBUS_INTERFACE, "ActivateConnection", QVariant::fromValue(*path), QVariant::fromValue(QDBusObjectPath(adapter)), QVariant::fromValue(QDBusObjectPath("/")));
  }
}

void WifiManager::activateModemConnection(const QDBusObjectPath &path) {
  QString modem = getAdapter(NM_DEVICE_TYPE_MODEM);
  if (!path.path().isEmpty() && !modem.isEmpty()) {
    call(NM_DBUS_PATH, NM_DBUS_INTERFACE, "ActivateConnection", QVariant::fromValue(path), QVariant::fromValue(QDBusObjectPath(modem)), QVariant::fromValue(QDBusObjectPath("/")));
  }
}

// function matches tici/hardware.py
NetworkType WifiManager::currentNetworkType() {
  auto primary_conn = call<QDBusObjectPath>(NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE, "PrimaryConnection");
  auto primary_type = call<QString>(primary_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");

  if (primary_type == "802-3-ethernet") {
    return NetworkType::ETHERNET;
  } else if (primary_type == "802-11-wireless" && !isTetheringEnabled()) {
    return NetworkType::WIFI;
  } else {
    for (const QDBusObjectPath &conn : get_active_connections()) {
      auto type = call<QString>(conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");
      if (type == "gsm") {
        return NetworkType::CELL;
      }
    }
  }
  return NetworkType::NONE;
}

void WifiManager::updateGsmSettings(bool roaming, QString apn) {
  if (!lteConnectionPath.path().isEmpty()) {
    bool changes = false;
    bool auto_config = apn.isEmpty();
    Connection settings = QDBusReply<Connection>(call(lteConnectionPath.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "GetSettings")).value();
    if (settings.value("gsm").value("auto-config").toBool() != auto_config) {
      qWarning() << "Changing gsm.auto-config to" << auto_config;
      settings["gsm"]["auto-config"] = auto_config;
      changes = true;
    }

    if (settings.value("gsm").value("apn").toString() != apn) {
      qWarning() << "Changing gsm.apn to" << apn;
      settings["gsm"]["apn"] = apn;
      changes = true;
    }

    if (settings.value("gsm").value("home-only").toBool() == roaming) {
      qWarning() << "Changing gsm.home-only to" << !roaming;
      settings["gsm"]["home-only"] = !roaming;
      changes = true;
    }

    if (changes) {
      call(lteConnectionPath.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "UpdateUnsaved", QVariant::fromValue(settings));  // update is temporary
      deactivateConnection(lteConnectionPath);
      activateModemConnection(lteConnectionPath);
    }
  }
}

// Functions for tethering
void WifiManager::addTetheringConnection() {
  Connection connection;
  connection["connection"]["id"] = "Hotspot";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["interface-name"] = "wlan0";
  connection["connection"]["autoconnect"] = false;

  connection["802-11-wireless"]["band"] = "bg";
  connection["802-11-wireless"]["mode"] = "ap";
  connection["802-11-wireless"]["ssid"] = tethering_ssid.toUtf8();

  connection["802-11-wireless-security"]["group"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["pairwise"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["proto"] = QStringList("rsn");
  connection["802-11-wireless-security"]["psk"] = defaultTetheringPassword;

  connection["ipv4"]["method"] = "shared";
  QMap<QString,QVariant> address;
  address["address"] = "192.168.43.1";
  address["prefix"] = 24u;
  connection["ipv4"]["address-data"] = QVariant::fromValue(IpConfig() << address);
  connection["ipv4"]["gateway"] = "192.168.43.1";
  connection["ipv4"]["route-metric"] = 1100;
  connection["ipv6"]["method"] = "ignore";

  call(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "AddConnection", QVariant::fromValue(connection));
}

void WifiManager::setTetheringEnabled(bool enabled) {
  if (enabled) {
    if (!isKnownConnection(tethering_ssid)) {
      addTetheringConnection();
    }
    activateWifiConnection(tethering_ssid);
  } else {
    deactivateConnectionBySsid(tethering_ssid);
  }
}

bool WifiManager::isTetheringEnabled() {
  if (activeAp != "" && activeAp != "/") {
    return get_property(activeAp, "Ssid") == tethering_ssid;
  }
  return false;
}

QString WifiManager::getTetheringPassword() {
  if (!isKnownConnection(tethering_ssid)) {
    addTetheringConnection();
  }
  if (auto path = getConnectionPath(tethering_ssid)) {
    const QDBusReply<QMap<QString, QMap<QString, QVariant>>> response = call(path->path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "GetSecrets", "802-11-wireless-security");
    return response.value().value("802-11-wireless-security").value("psk").toString();
  }
  return "";
}

void WifiManager::changeTetheringPassword(const QString &newPassword) {
  if (auto path = getConnectionPath(tethering_ssid)) {
    Connection settings = QDBusReply<Connection>(call(path->path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION,"GetSettings")).value();
    settings["802-11-wireless-security"]["psk"] = newPassword;
    call(path->path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "Update", QVariant::fromValue(settings));
    if (isTetheringEnabled()) {
      activateWifiConnection(tethering_ssid);
    }
  }
}
