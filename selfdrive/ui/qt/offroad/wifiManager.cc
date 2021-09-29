#include "selfdrive/ui/qt/offroad/wifiManager.h"

#include <algorithm>
#include <set>
#include <cstdlib>

#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/ui/qt/util.h"

template <typename T>
T get_response(QDBusMessage response) {
  QVariant first =  response.arguments().at(0);
  QDBusVariant dbvFirst = first.value<QDBusVariant>();
  QVariant vFirst = dbvFirst.variant();
  if (vFirst.canConvert<T>()) {
    return vFirst.value<T>();
  } else {
    LOGE("Variant unpacking failure");
    return T();
  }
}

bool compare_by_strength(const Network &a, const Network &b) {
  if (a.connected == ConnectedType::CONNECTED) return true;
  if (b.connected == ConnectedType::CONNECTED) return false;
  if (a.connected == ConnectedType::CONNECTING) return true;
  if (b.connected == ConnectedType::CONNECTING) return false;
  return a.strength > b.strength;
}

WifiManager::WifiManager(QWidget* parent) : QWidget(parent) {
  qDBusRegisterMetaType<Connection>();
  qDBusRegisterMetaType<IpConfig>();
  connecting_to_network = "";

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

  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, [=]() {
    if (!adapter.isEmpty() && this->isVisible()) {
      requestScan();
    }
  });
  timer->start(5000);
}

void WifiManager::setup() {
  QDBusInterface nm(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE, bus);
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE, "StateChanged", this, SLOT(stateChange(unsigned int, unsigned int, unsigned int)));
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, "PropertiesChanged", this, SLOT(propertyChange(QString, QVariantMap, QStringList)));

  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "ConnectionRemoved", this, SLOT(connectionRemoved(QDBusObjectPath)));
  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "NewConnection", this, SLOT(newConnection(QDBusObjectPath)));

  QDBusInterface device_props(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);
  QDBusMessage response = device_props.call("Get", NM_DBUS_INTERFACE_DEVICE, "State");
  raw_adapter_state = get_response<uint>(response);

  initActiveAp();
  initConnections();
  requestScan();
}

void WifiManager::refreshNetworks() {
  if (adapter.isEmpty()) {
    return;
  }
  seenNetworks.clear();
  ipv4_address = get_ipv4_address();

  QDBusInterface nm(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, bus);
  nm.setTimeout(DBUS_TIMEOUT);

  const QDBusReply<QList<QDBusObjectPath>> &response = nm.call("GetAllAccessPoints");
  for (const QDBusObjectPath &path : response.value()) {
    const QByteArray &ssid = get_property(path.path(), "Ssid");
    unsigned int strength = get_ap_strength(path.path());
    if (ssid.isEmpty() || (seenNetworks.contains(ssid) &&
        strength <= seenNetworks.value(ssid).strength)) {
      continue;
    }
    SecurityType security = getSecurityType(path.path());
    ConnectedType ctype;
    QString activeSsid = (activeAp != "" && activeAp != "/") ? get_property(activeAp, "Ssid") : "";
    if (ssid != activeSsid) {
      ctype = ConnectedType::DISCONNECTED;
    } else {
      if (ssid == connecting_to_network) {
        ctype = ConnectedType::CONNECTING;
      } else {
        ctype = ConnectedType::CONNECTED;
      }
    }
    Network network = {ssid, strength, ctype, security};
    seenNetworks[ssid] = network;
  }
}

QString WifiManager::get_ipv4_address() {
  if (raw_adapter_state != NM_DEVICE_STATE_ACTIVATED) {
    return "";
  }
  QVector<QDBusObjectPath> conns = get_active_connections();
  for (auto &p : conns) {
    QDBusInterface nm(NM_DBUS_SERVICE, p.path(), NM_DBUS_INTERFACE_PROPERTIES, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Ip4Config"));
    QString ip4config = pth.path();

    QString type = get_response<QString>(nm.call("Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type"));

    if (type == "802-11-wireless") {
      QDBusInterface nm2(NM_DBUS_SERVICE, ip4config, NM_DBUS_INTERFACE_PROPERTIES, bus);
      nm2.setTimeout(DBUS_TIMEOUT);

      const QDBusArgument &arr = get_response<QDBusArgument>(nm2.call("Get", NM_DBUS_INTERFACE_IP4_CONFIG, "AddressData"));
      QMap<QString, QVariant> pth2;
      arr.beginArray();
      while (!arr.atEnd()) {
        arr >> pth2;
        QString ipv4 = pth2.value("address").value<QString>();
        arr.endArray();
        return ipv4;
      }
      arr.endArray();
    }
  }
  return "";
}

SecurityType WifiManager::getSecurityType(const QString &path) {
  int sflag = get_property(path, "Flags").toInt();
  int wpaflag = get_property(path, "WpaFlags").toInt();
  int rsnflag = get_property(path, "RsnFlags").toInt();
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

void WifiManager::connect(const Network &n) {
  return connect(n, "", "");
}

void WifiManager::connect(const Network &n, const QString &password) {
  return connect(n, "", password);
}

void WifiManager::connect(const Network &n, const QString &username, const QString &password) {
  connecting_to_network = n.ssid;
  // disconnect();
  forgetConnection(n.ssid); //Clear all connections that may already exist to the network we are connecting
  connect(n.ssid, username, password, n.security_type);
}

void WifiManager::connect(const QByteArray &ssid, const QString &username, const QString &password, SecurityType security_type) {
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "openpilot connection "+QString::fromStdString(ssid.toStdString());
  connection["connection"]["autoconnect-retries"] = 0;

  connection["802-11-wireless"]["ssid"] = ssid;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  if (security_type == SecurityType::WPA) {
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["auth-alg"] = "open";
    connection["802-11-wireless-security"]["psk"] = password;
  }

  connection["ipv4"]["method"] = "auto";
  connection["ipv4"]["dns-priority"] = 600;
  connection["ipv6"]["method"] = "ignore";

  QDBusInterface nm_settings(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, bus);
  nm_settings.setTimeout(DBUS_TIMEOUT);

  nm_settings.call("AddConnection", QVariant::fromValue(connection));
}

void WifiManager::deactivateConnection(const QString &ssid) {
  for (QDBusObjectPath active_connection_raw : get_active_connections()) {
    QString active_connection = active_connection_raw.path();
    QDBusInterface nm(NM_DBUS_SERVICE, active_connection, NM_DBUS_INTERFACE_PROPERTIES, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "SpecificObject"));
    if (pth.path() != "" && pth.path() != "/") {
      QString Ssid = get_property(pth.path(), "Ssid");
      if (Ssid == ssid) {
        QDBusInterface nm2(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE, bus);
        nm2.setTimeout(DBUS_TIMEOUT);
        nm2.call("DeactivateConnection", QVariant::fromValue(active_connection_raw));
      }
    }
  }
}

QVector<QDBusObjectPath> WifiManager::get_active_connections() {
  QDBusInterface nm(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, bus);
  nm.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = nm.call("Get", NM_DBUS_INTERFACE, "ActiveConnections");
  const QDBusArgument &arr = get_response<QDBusArgument>(response);
  QVector<QDBusObjectPath> conns;

  QDBusObjectPath path;
  arr.beginArray();
  while (!arr.atEnd()) {
    arr >> path;
    conns.push_back(path);
  }
  arr.endArray();
  return conns;
}

bool WifiManager::isKnownConnection(const QString &ssid) {
  return !getConnectionPath(ssid).path().isEmpty();
}

void WifiManager::forgetConnection(const QString &ssid) {
  const QDBusObjectPath &path = getConnectionPath(ssid);
  if (!path.path().isEmpty()) {
    QDBusInterface nm2(NM_DBUS_SERVICE, path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, bus);
    nm2.call("Delete");
  }
}

bool WifiManager::isWirelessAdapter(const QDBusObjectPath &path) {
  QDBusInterface device_props(NM_DBUS_SERVICE, path.path(), NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);
  const uint deviceType = get_response<uint>(device_props.call("Get", NM_DBUS_INTERFACE_DEVICE, "DeviceType"));
  return deviceType == NM_DEVICE_TYPE_WIFI;
}

void WifiManager::requestScan() {
  QDBusInterface nm(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, bus);
  nm.setTimeout(DBUS_TIMEOUT);
  nm.call("RequestScan", QVariantMap());
}

uint WifiManager::get_wifi_device_state() {
  QDBusInterface device_props(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", NM_DBUS_INTERFACE_DEVICE, "State");
  uint resp = get_response<uint>(response);
  return resp;
}

QByteArray WifiManager::get_property(const QString &network_path , const QString &property) {
  QDBusInterface device_props(NM_DBUS_SERVICE, network_path, NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", NM_DBUS_INTERFACE_ACCESS_POINT, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(const QString &network_path) {
  QDBusInterface device_props(NM_DBUS_SERVICE, network_path, NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", NM_DBUS_INTERFACE_ACCESS_POINT, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::getAdapter() {
  QDBusInterface nm(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE, bus);
  nm.setTimeout(DBUS_TIMEOUT);

  const QDBusReply<QList<QDBusObjectPath>> &response = nm.call("GetDevices");
  for (const QDBusObjectPath &path : response.value()) {
    if (isWirelessAdapter(path)) {
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
    if (this->isVisible()) {
      refreshNetworks();
      emit refreshSignal();
    }
  }
}

// https://developer.gnome.org/NetworkManager/stable/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html
void WifiManager::propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props) {
  if (interface == NM_DBUS_INTERFACE_DEVICE_WIRELESS && props.contains("LastScan")) {
    if (this->isVisible() || firstScan) {
      refreshNetworks();
      emit refreshSignal();
      firstScan = false;
    }
  } else if (interface == NM_DBUS_INTERFACE_DEVICE_WIRELESS && props.contains("ActiveAccessPoint")) {
    const QDBusObjectPath &path = props.value("ActiveAccessPoint").value<QDBusObjectPath>();
    activeAp = path.path();
  }
}

void WifiManager::deviceAdded(const QDBusObjectPath &path) {
  if (isWirelessAdapter(path) && (adapter.isEmpty() || adapter == "/")) {
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
    deactivateConnection(get_property(activeAp, "Ssid"));
  }
}

QDBusObjectPath WifiManager::getConnectionPath(const QString &ssid) {
  for (const QString &conn_ssid : knownConnections) {
    if (ssid == conn_ssid) {
      return knownConnections.key(conn_ssid);
    }
  }
  return QDBusObjectPath();
}

Connection WifiManager::getConnectionSettings(const QDBusObjectPath &path) {
  QDBusInterface nm(NM_DBUS_SERVICE, path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, bus);
  nm.setTimeout(DBUS_TIMEOUT);
  return QDBusReply<Connection>(nm.call("GetSettings")).value();
}

void WifiManager::initConnections() {
  QDBusInterface nm(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, bus);
  nm.setTimeout(DBUS_TIMEOUT);

  const QDBusReply<QList<QDBusObjectPath>> response = nm.call("ListConnections");
  for (const QDBusObjectPath &path : response.value()) {
    const Connection &settings = getConnectionSettings(path);
    if (settings.value("connection").value("type") == "802-11-wireless") {
      knownConnections[path] = settings.value("802-11-wireless").value("ssid").toString();
    } else if (path.path() != "/") {
      lteConnectionPath = path.path();
    }
  }
}

void WifiManager::activateWifiConnection(const QString &ssid) {
  const QDBusObjectPath &path = getConnectionPath(ssid);
  if (!path.path().isEmpty()) {
    connecting_to_network = ssid;
    QDBusInterface nm3(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE, bus);
    nm3.setTimeout(DBUS_TIMEOUT);
    nm3.call("ActivateConnection", QVariant::fromValue(path), QVariant::fromValue(QDBusObjectPath(adapter)), QVariant::fromValue(QDBusObjectPath("/")));
  }
}

// function matches tici/hardware.py
NetworkType WifiManager::currentNetworkType() {
  QDBusInterface nm(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, bus);
  nm.setTimeout(DBUS_TIMEOUT);
  const QDBusObjectPath &path = get_response<QDBusObjectPath>(nm.call("Get", NM_DBUS_INTERFACE, "PrimaryConnection"));

  QDBusInterface nm2(NM_DBUS_SERVICE, path.path(), NM_DBUS_INTERFACE_PROPERTIES, bus);
  nm.setTimeout(DBUS_TIMEOUT);
  const QString &type = get_response<QString>(nm2.call("Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type"));

  if (type == "802-3-ethernet") {
    return NetworkType::ETHERNET;
  } else if (type == "802-11-wireless" && !isTetheringEnabled()) {
    return NetworkType::WIFI;
  } else {
    for (const QDBusObjectPath &path : get_active_connections()) {
      QDBusInterface nm3(NM_DBUS_SERVICE, path.path(), NM_DBUS_INTERFACE_PROPERTIES, bus);
      nm3.setTimeout(DBUS_TIMEOUT);
      const QString &type = get_response<QString>(nm3.call("Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type"));
      if (type == "gsm") {
        return NetworkType::CELL;
      }
    }
  }
  return NetworkType::NONE;
}

void WifiManager::setRoamingEnabled(bool roaming) {
  if (!lteConnectionPath.isEmpty()) {
    QDBusInterface nm(NM_DBUS_SERVICE, lteConnectionPath, NM_DBUS_INTERFACE_SETTINGS_CONNECTION, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    Connection settings = QDBusReply<Connection>(nm.call("GetSettings")).value();
    if (settings.value("gsm").value("home-only").toBool() == roaming) {
      settings["gsm"]["home-only"] = !roaming;
      nm.call("UpdateUnsaved", QVariant::fromValue(settings));  // update is temporary
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

  QDBusInterface nm_settings(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, bus);
  nm_settings.setTimeout(DBUS_TIMEOUT);
  nm_settings.call("AddConnection", QVariant::fromValue(connection));
}

void WifiManager::setTetheringEnabled(bool enabled) {
  if (enabled) {
    if (!isKnownConnection(tethering_ssid)) {
      addTetheringConnection();
    }
    activateWifiConnection(tethering_ssid);
  } else {
    deactivateConnection(tethering_ssid);
  }
}

void WifiManager::initActiveAp() {
  QDBusInterface device_props(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  const QDBusMessage &response = device_props.call("Get", NM_DBUS_INTERFACE_DEVICE_WIRELESS, "ActiveAccessPoint");
  activeAp = get_response<QDBusObjectPath>(response).path();
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
  const QDBusObjectPath &path = getConnectionPath(tethering_ssid);
  if (!path.path().isEmpty()) {
    QDBusInterface nm(NM_DBUS_INTERFACE, path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    const QDBusReply<QMap<QString, QMap<QString, QVariant>>> response = nm.call("GetSecrets", "802-11-wireless-security");
    return response.value().value("802-11-wireless-security").value("psk").toString();
  }
  return "";
}

void WifiManager::changeTetheringPassword(const QString &newPassword) {
  const QDBusObjectPath &path = getConnectionPath(tethering_ssid);
  if (!path.path().isEmpty()) {
    QDBusInterface nm(NM_DBUS_INTERFACE, path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    Connection settings = QDBusReply<Connection>(nm.call("GetSettings")).value();
    settings["802-11-wireless-security"]["psk"] = newPassword;
    nm.call("Update", QVariant::fromValue(settings));

    if (isTetheringEnabled()) {
      activateWifiConnection(tethering_ssid);
    }
  }
}
