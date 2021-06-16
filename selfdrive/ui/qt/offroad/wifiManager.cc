#include "selfdrive/ui/qt/offroad/wifiManager.h"

#include <algorithm>
#include <set>
#include <cstdlib>

#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"

/**
 * We are using a NetworkManager DBUS API : https://developer.gnome.org/NetworkManager/1.26/spec.html
 * */

// https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NM80211ApFlags
const int NM_802_11_AP_FLAGS_PRIVACY = 0x00000001;

// https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NM80211ApSecurityFlags
const int NM_802_11_AP_SEC_PAIR_WEP40      = 0x00000001;
const int NM_802_11_AP_SEC_PAIR_WEP104     = 0x00000002;
const int NM_802_11_AP_SEC_GROUP_WEP40     = 0x00000010;
const int NM_802_11_AP_SEC_GROUP_WEP104    = 0x00000020;
const int NM_802_11_AP_SEC_KEY_MGMT_PSK    = 0x00000100;
const int NM_802_11_AP_SEC_KEY_MGMT_802_1X = 0x00000200;

const QString nm_path                = "/org/freedesktop/NetworkManager";
const QString nm_settings_path       = "/org/freedesktop/NetworkManager/Settings";

const QString nm_iface               = "org.freedesktop.NetworkManager";
const QString props_iface            = "org.freedesktop.DBus.Properties";
const QString nm_settings_iface      = "org.freedesktop.NetworkManager.Settings";
const QString nm_settings_conn_iface = "org.freedesktop.NetworkManager.Settings.Connection";
const QString device_iface           = "org.freedesktop.NetworkManager.Device";
const QString wireless_device_iface  = "org.freedesktop.NetworkManager.Device.Wireless";
const QString ap_iface               = "org.freedesktop.NetworkManager.AccessPoint";
const QString connection_iface       = "org.freedesktop.NetworkManager.Connection.Active";
const QString ipv4config_iface       = "org.freedesktop.NetworkManager.IP4Config";

const QString nm_service             = "org.freedesktop.NetworkManager";

const int state_connected = 100;
const int state_need_auth = 60;
const int reason_wrong_password = 8;
const int dbus_timeout = 100;

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

void WifiThread::run() {
  unsigned int i = 10;
  while (!isInterruptionRequested()) {
    // Process incoming signals from networking UI
    eventDispatcher()->processEvents(QEventLoop::AllEvents);
    i++;
    QThread::msleep(100);
    if (i > 10) {
      i = 0;
      qDebug() << "WifiThread::run()";
      qDebug() << "connecting:" << wifi->connecting_to_network;
      // TODO move this updating code to change() so we only update on a state change
      // TODO then impose a minimum rate of updating (2s?) where not refreshed in this duration, refresh here
      wifi->request_scan();
      wifi->refreshNetworks();
      emit updateNetworking(wifi->seen_networks, wifi->ipv4_address);
    } else {
//      wifi->updateNetworks();
      emit updateNetworking(wifi->seen_networks, wifi->ipv4_address);
    }
  }
}

void WifiThread::connectToNetwork(const Network n, const QString pass) {
  qDebug() << "WIFITHREAD::connectToNetwork";
//  QThread::sleep(5);
  if (n.known && pass.isEmpty()) {  // we can have a connection with an incorrect password
    wifi->activateWifiConnection(n.ssid);
  } else if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    qDebug() << "WIFITHREAD::connectToNetwork::WPA";
    wifi->connect(n, pass);
  }
}

void WifiThread::toggleTethering(const bool enabled) {
  qDebug() << "toggleTETHERING!" << enabled;
  if (enabled) {
    wifi->enableTethering();
  } else {
    wifi->disableTethering();
  }
  emit tetheringStateChange();
}

void WifiThread::changeTetheringPassword(const QString newPassword) {
  wifi->changeTetheringPassword(newPassword);
}


//void WifiThread::wrongPassword(const QString ssid) {
//  emit wrongPasswordSignal(ssid);
//}


WifiManager::WifiManager() : QObject() {
  qDBusRegisterMetaType<Connection>();
  qDBusRegisterMetaType<IpConfig>();
  connecting_to_network = "";
  adapter = get_adapter();

  bool has_adapter = adapter != "";
  if (!has_adapter) {
    throw std::runtime_error("Error connecting to NetworkManager");
  }

  QDBusInterface nm(nm_service, adapter, device_iface, bus);
  bus.connect(nm_service, adapter, device_iface, "StateChanged", this, SLOT(change(unsigned int, unsigned int, unsigned int)));

  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(dbus_timeout);
  QDBusMessage response = device_props.call("Get", device_iface, "State");
  raw_adapter_state = get_response<uint>(response);
  change(raw_adapter_state, 0, 0);

  // Set tethering ssid as "weedle" + first 4 characters of a dongle id
  tethering_ssid = "weedle";
  std::string bytes = Params().get("DongleId");
  if (bytes.length() >= 4) {
    tethering_ssid+="-"+QString::fromStdString(bytes.substr(0,4));
  }

  // Create dbus interface for tethering button. This populates the introspection cache,
  // making sure all future creations are non-blocking
  // https://bugreports.qt.io/browse/QTBUG-14485
  QDBusInterface(nm_service, nm_settings_path, nm_settings_iface, bus);
}

// if scan is false, it only updates the seen_ssids and seen_networks lists for UI, if it is true then it scans for networks
void WifiManager::updateNetworks() {  // TODO assuming false
//  seen_networks.clear();
//  seen_ssids.clear();
  ipv4_address = get_ipv4_address();  // TODO remove?
  QString active_ap = get_active_ap();
  for (Network &network : seen_networks) {
    ConnectedType ctype;
    if (network.path != active_ap) {
      ctype = ConnectedType::DISCONNECTED;
    } else {
      if (network.ssid == connecting_to_network) {
        ctype = ConnectedType::CONNECTING;
      } else {
        ctype = ConnectedType::CONNECTED;
      }
    }
    network.connected = ctype;
  }
}

void WifiManager::refreshNetworks() {
  seen_networks.clear();
  seen_ssids.clear();  // TODO combine these two
  ipv4_address = get_ipv4_address();  // TODO: remove!
  for (Network &network : get_networks()) {
    if (seen_ssids.count(network.ssid)) {
      continue;
    }
    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
}

QString WifiManager::get_ipv4_address() {
  if (raw_adapter_state != state_connected) {
    return "";
  }
  QVector<QDBusObjectPath> conns = get_active_connections();
  for (auto &p : conns) {
    QString active_connection = p.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    nm.setTimeout(dbus_timeout);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "Ip4Config"));
    QString ip4config = pth.path();

    QString type = get_response<QString>(nm.call("Get", connection_iface, "Type"));

    if (type == "802-11-wireless") {
      QDBusInterface nm2(nm_service, ip4config, props_iface, bus);
      nm2.setTimeout(dbus_timeout);

      const QDBusArgument &arr = get_response<QDBusArgument>(nm2.call("Get", ipv4config_iface, "AddressData"));
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

QList<Network> WifiManager::get_networks() {
  QList<Network> r;
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  nm.setTimeout(dbus_timeout);

  QString active_ap = get_active_ap();
  QDBusReply<QList<QDBusObjectPath>> response = nm.call("GetAllAccessPoints");

  foreach (const QDBusObjectPath& path, response.value()) {  // TODO: can we replace with for (path : response.value()) { } ?
    QString ssid = get_property(path.path(), "Ssid");
    if (ssid.isEmpty()) {
      continue;
    }

    unsigned int strength = get_ap_strength(path.path());
    SecurityType security = getSecurityType(path.path());
    ConnectedType ctype;
    if (path.path() != active_ap) {
      ctype = ConnectedType::DISCONNECTED;
    } else {
      if (ssid == connecting_to_network) {
        ctype = ConnectedType::CONNECTING;
      } else {
        ctype = ConnectedType::CONNECTED;
      }
    }

    Network network = {path.path(), ssid, strength, ctype, security, isKnownNetwork(ssid)};
    r.push_back(network);
  }

  std::sort(r.begin(), r.end(), compare_by_strength);
  return r;
}

SecurityType WifiManager::getSecurityType(const QString &path) {
  int sflag = get_property(path, "Flags").toInt();
  int wpaflag = get_property(path, "WpaFlags").toInt();
  int rsnflag = get_property(path, "RsnFlags").toInt();
  int wpa_props = wpaflag | rsnflag;

  // obtained by looking at flags of networks in the office as reported by an Android phone
  const int supports_wpa = NM_802_11_AP_SEC_PAIR_WEP40 | NM_802_11_AP_SEC_PAIR_WEP104 | NM_802_11_AP_SEC_GROUP_WEP40 | NM_802_11_AP_SEC_GROUP_WEP104 | NM_802_11_AP_SEC_KEY_MGMT_PSK;

  if (sflag == 0) {
    return SecurityType::OPEN;
  } else if ((sflag & NM_802_11_AP_FLAGS_PRIVACY) && (wpa_props & supports_wpa) && !(wpa_props & NM_802_11_AP_SEC_KEY_MGMT_802_1X)) {
    return SecurityType::WPA;
  } else {
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
  connecting_to_network = n.ssid;  // TODO: copy this to activate function too
  // disconnect();
  forgetNetwork(n.ssid); //Clear all connections that may already exist to the network we are connecting
  connect(n.ssid, username, password, n.security_type);
}

void WifiManager::connect(const QString &ssid, const QString &username, const QString &password, SecurityType security_type) {
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "openpilot connection "+QString::fromStdString(ssid.toStdString());
  connection["connection"]["autoconnect-retries"] = 0;

  connection["802-11-wireless"]["ssid"] = ssid.toUtf8();
  connection["802-11-wireless"]["mode"] = "infrastructure";

  if (security_type == SecurityType::WPA) {
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["auth-alg"] = "open";
    connection["802-11-wireless-security"]["psk"] = password;
  }

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";

  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  nm_settings.setTimeout(dbus_timeout);

  nm_settings.call("AddConnection", QVariant::fromValue(connection));
  qDebug() << "Here";

  activateWifiConnection(QString(ssid));
}

void WifiManager::deactivateConnection(const QString &ssid) {
  for (QDBusObjectPath active_connection_raw : get_active_connections()) {
    QString active_connection = active_connection_raw.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    nm.setTimeout(dbus_timeout);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "SpecificObject"));
    if (pth.path() != "" && pth.path() != "/") {
      QString Ssid = get_property(pth.path(), "Ssid");
      if (Ssid == ssid) {
        QDBusInterface nm2(nm_service, nm_path, nm_iface, bus);
        nm2.setTimeout(dbus_timeout);
        nm2.call("DeactivateConnection", QVariant::fromValue(active_connection_raw));
      }
    }
  }
}

QVector<QDBusObjectPath> WifiManager::get_active_connections() {
  QDBusInterface nm(nm_service, nm_path, props_iface, bus);
  nm.setTimeout(dbus_timeout);

  QDBusMessage response = nm.call("Get", nm_iface, "ActiveConnections");
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

bool WifiManager::isKnownNetwork(const QString &ssid) {
  return !pathFromSsid(ssid).path().isEmpty();
}

void WifiManager::forgetNetwork(const QString &ssid) {
  QDBusObjectPath path = pathFromSsid(ssid);
  if (!path.path().isEmpty()) {
    QDBusInterface nm2(nm_service, path.path(), nm_settings_conn_iface, bus);
    nm2.call("Delete");
  }
}

void WifiManager::request_scan() {
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  nm.setTimeout(dbus_timeout);
  nm.call("RequestScan",  QVariantMap());
}

uint WifiManager::get_wifi_device_state() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(dbus_timeout);

  QDBusMessage response = device_props.call("Get", device_iface, "State");
  uint resp = get_response<uint>(response);
  return resp;
}

QString WifiManager::get_active_ap() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(dbus_timeout);

  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}

QByteArray WifiManager::get_property(const QString &network_path , const QString &property) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  device_props.setTimeout(dbus_timeout);

  QDBusMessage response = device_props.call("Get", ap_iface, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(const QString &network_path) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  device_props.setTimeout(dbus_timeout);

  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::get_adapter() {
  QDBusInterface nm(nm_service, nm_path, nm_iface, bus);
  nm.setTimeout(dbus_timeout);

  QDBusMessage response = nm.call("GetDevices");
  QVariant first =  response.arguments().at(0);

  QString adapter_path = "";

  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    // Get device type
    QDBusInterface device_props(nm_service, path.path(), props_iface, bus);
    device_props.setTimeout(dbus_timeout);

    QDBusMessage response = device_props.call("Get", device_iface, "DeviceType");
    uint device_type = get_response<uint>(response);

    if (device_type == 2) { // Wireless
      adapter_path = path.path();
      break;
    }
  }
  args.endArray();

  return adapter_path;
}

void WifiManager::change(unsigned int new_state, unsigned int previous_state, unsigned int change_reason) {
  updateNetworks();
  raw_adapter_state = new_state;
  if (new_state == state_need_auth && change_reason == reason_wrong_password) {
    qDebug() << "CHANGE::WRONG PASSWORD!";
    Network network;
    for (const Network n : seen_networks) {
      if (n.ssid == connecting_to_network) {
        connecting_to_network = "";  // TODO unify connecting_to_network with seen_networks<network>.connecting
        emit wrongPassword(n);
        return;
      }
    }
  } else if (new_state == state_connected) {
    qDebug() << "CHANGE::CONNECTED!";
    emit successfulConnection(connecting_to_network);  // TODO: use Network
    connecting_to_network = "";  // TODO unify connecting_to_network with seen_networks<network>.connecting
  }
}

void WifiManager::disconnect() {
  QString active_ap = get_active_ap();
  if (active_ap != "" && active_ap != "/") {
    deactivateConnection(get_property(active_ap, "Ssid"));
  }
}

QDBusObjectPath WifiManager::pathFromSsid(const QString &ssid) {
  QDBusObjectPath path;  // returns uninitialized path if network is not known
  for (auto const& [conn_ssid, conn_path] : listConnections()) {
    if (conn_ssid == ssid) {
      path = conn_path;
    }
  }
  return path;
}

QVector<QPair<QString, QDBusObjectPath>> WifiManager::listConnections() {
  QVector<QPair<QString, QDBusObjectPath>> connections;
  QDBusInterface nm(nm_service, nm_settings_path, nm_settings_iface, bus);
  nm.setTimeout(dbus_timeout);

  QDBusReply<QList<QDBusObjectPath>> response = nm.call("ListConnections");
  foreach (const QDBusObjectPath& path, response.value()) {
    QDBusInterface nm2(nm_service, path.path(), nm_settings_conn_iface, bus);
    nm2.setTimeout(dbus_timeout);

    const QDBusReply<QMap<QString, QMap<QString, QVariant>>> map = nm2.call("GetSettings");
    const QString ssid = map.value().value("802-11-wireless").value("ssid").toString();

    connections.push_back(qMakePair(ssid, path));
  }
  return connections;
}

void WifiManager::activateWifiConnection(const QString &ssid) {
  connecting_to_network = ssid;
  QDBusObjectPath path = pathFromSsid(ssid);
  if (!path.path().isEmpty()) {
    QString devicePath = get_adapter();
    QDBusInterface nm3(nm_service, nm_path, nm_iface, bus);
    nm3.setTimeout(dbus_timeout);
    nm3.call("ActivateConnection", QVariant::fromValue(path), QVariant::fromValue(QDBusObjectPath(devicePath)), QVariant::fromValue(QDBusObjectPath("/")));
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
  connection["802-11-wireless-security"]["psk"] = tetheringPassword;

  connection["ipv4"]["method"] = "shared";
  QMap<QString,QVariant> address;
  address["address"] = "192.168.43.1";
  address["prefix"] = 24u;
  connection["ipv4"]["address-data"] = QVariant::fromValue(IpConfig() << address);
  connection["ipv4"]["gateway"] = "192.168.43.1";
  connection["ipv4"]["route-metric"] = 1100;
  connection["ipv6"]["method"] = "ignore";

  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  nm_settings.setTimeout(dbus_timeout);
  nm_settings.call("AddConnection", QVariant::fromValue(connection));
}

void WifiManager::enableTethering() {
  if (!isKnownNetwork(tethering_ssid)) {
    addTetheringConnection();
  }
  activateWifiConnection(tethering_ssid);
}

void WifiManager::disableTethering() {
  deactivateConnection(tethering_ssid);
}

bool WifiManager::tetheringEnabled() {
  QString active_ap = get_active_ap();
  return get_property(active_ap, "Ssid") == tethering_ssid;
}

void WifiManager::changeTetheringPassword(const QString &newPassword) {
  qDebug() << "CHANGING PASSWORD!";
  tetheringPassword = newPassword;
  if (isKnownNetwork(tethering_ssid)) {
    forgetNetwork(tethering_ssid);
  }
  addTetheringConnection();
}
