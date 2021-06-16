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


// States: https://developer.gnome.org/NetworkManager/stable/nm-dbus-types.html#NMDeviceState
const int STATE_DISCONNECTED = 30;
const int STATE_CONNECTING = 40;
const int STATE_NEED_AUTH = 60;
const int STATE_CONNECTED = 100;

// Reasons: https://developer.gnome.org/NetworkManager/stable/nm-dbus-types.html#NMDeviceStateReason
const int REASON_WRONG_PASSWORD = 8;
const int DBUS_TIMEOUT = 100;

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

//void WifiThread::run() {
//  QThread::sleep(1);
//  const unsigned int min_refresh_rate = 5000;
//  const unsigned int event_loop_rate = 100;
//  unsigned int ms_since_scan = 5000;
////  wifi->requestScan();
//  while (!isInterruptionRequested()) {
//    QThread::msleep(event_loop_rate);
//    ms_since_scan += event_loop_rate;
//    if (ms_since_scan >= min_refresh_rate) {
//      qDebug() << "Requesting scan";
//	    wifi->requestScan();
//	    ms_since_scan = 0;
//	  }
//    eventDispatcher()->processEvents(QEventLoop::AllEvents);
//
////    qDebug() << "WifiThread::run()";
////    qDebug() << wifi->connecting_to_network;
////    wifi->requestScan();
////    wifi->refreshNetworks();
////    emit updateNetworking(wifi->seen_networks, wifi->ipv4_address);
////
////    // Process incoming signals from networking UI
////    eventDispatcher()->processEvents(QEventLoop::AllEvents);
////    QThread::sleep(1);
//  }
//}

void WifiManager::connectToNetwork(const Network n, const QString pass) {
  if (connecting_to_network != "") {
    qDebug() << "YOU'RE ALREADY CONNECTING TO A NETWORK, CHILL!!";
    return;
  }
  qDebug() << "WIFITHREAD::connectToNetwork";
//  QThread::sleep(5);
  if (n.known) {  // check network n
    activateWifiConnection(n.ssid);
  } else if (n.security_type == SecurityType::OPEN) {
    connect(n);
  } else if (n.security_type == SecurityType::WPA && !pass.isEmpty()) {
    qDebug() << "WIFITHREAD::connectToNetwork::WPA";
    connect(n, pass);
  }
}

void WifiManager::toggleTethering(const bool enabled) {
  qDebug() << "toggleTETHERING!" << enabled;
  if (enabled) {
    enableTethering();
  } else {
    disableTethering();
  }
  emit tetheringStateChange();
}

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
  bus.connect(nm_service, adapter, device_iface, "StateChanged", this, SLOT(state_change(unsigned int, unsigned int, unsigned int)));
  bus.connect(nm_service, adapter, props_iface, "PropertiesChanged", this, SLOT(property_change(QString, QVariantMap, QStringList)));

  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(DBUS_TIMEOUT);
  QDBusMessage response = device_props.call("Get", device_iface, "State");
  raw_adapter_state = get_response<uint>(response);
  state_change(raw_adapter_state, 0, 0);

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

QString WifiManager::get_ipv4_address() {
  if (raw_adapter_state != STATE_CONNECTED) {
    return "";
  }
  QVector<QDBusObjectPath> conns = get_active_connections();
  for (auto &p : conns) {
    QString active_connection = p.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "Ip4Config"));
    QString ip4config = pth.path();

    QString type = get_response<QString>(nm.call("Get", connection_iface, "Type"));

    if (type == "802-11-wireless") {
      QDBusInterface nm2(nm_service, ip4config, props_iface, bus);
      nm2.setTimeout(DBUS_TIMEOUT);

      const QDBusArgument &arr = get_response<QDBusArgument>(nm2.call("Get", ipv4config_iface, "AddressData")); // TODO: Clean this up with new dbus syntax
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

// Resets and creates seen_networks vector. Only to be called after a scan is complete
void WifiManager::refreshNetworks() {
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  nm.setTimeout(DBUS_TIMEOUT);

  QString active_ap = get_active_ap();
  QDBusReply<QList<QDBusObjectPath>> response = nm.call("GetAllAccessPoints");

  ipv4_address = get_ipv4_address();  // TODO decide if this should get called between refreshes, or if it doesn't change at all
  seen_networks.clear();
  seen_ssids.clear();

  foreach (const QDBusObjectPath& path, response.value()) {  // TODO: can we replace with for (path : response.value()) { } ?
    QString ssid = get_property(path.path(), "Ssid");
    if (ssid.isEmpty() || seen_ssids.count(ssid) != 0) {  // TODO: refactor so we can check seen_networks for ssid and remove seen_ssids
      continue;
    }

    unsigned int strength = get_ap_strength(path.path());
    SecurityType security = getSecurityType(path.path());
    ConnectedType ctype = getConnectedType(path.path(), ssid, active_ap);

    Network network = {path.path(), ssid, strength, ctype, security, isKnownNetwork(ssid)};
    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
  std::sort(seen_networks.begin(), seen_networks.end(), compare_by_strength);
}

void WifiManager::updateNetworks() {
  ipv4_address = get_ipv4_address();  // TODO remove?
  const QString active_ap = get_active_ap();
  for (Network &network : seen_networks) {
    network.connected = getConnectedType(network.path, network.ssid, active_ap);
  }
  std::sort(seen_networks.begin(), seen_networks.end(), compare_by_strength);
}

ConnectedType WifiManager::getConnectedType(const QString &path, const QString &ssid, const QString &active_ap) {
  if (path != active_ap) {
    return ConnectedType::DISCONNECTED;
  } else {
    if (ssid == connecting_to_network) {
      return ConnectedType::CONNECTING;
    } else {
      return ConnectedType::CONNECTED;
    }
  }
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
  nm_settings.setTimeout(DBUS_TIMEOUT);

  nm_settings.call("AddConnection", QVariant::fromValue(connection));
  activateWifiConnection(QString(ssid));
}

void WifiManager::deactivateConnection(const QString &ssid) {
  for (QDBusObjectPath active_connection_raw : get_active_connections()) {
    QString active_connection = active_connection_raw.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    nm.setTimeout(DBUS_TIMEOUT);

    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "SpecificObject"));
    if (pth.path() != "" && pth.path() != "/") {
      QString Ssid = get_property(pth.path(), "Ssid");
      if (Ssid == ssid) {
        QDBusInterface nm2(nm_service, nm_path, nm_iface, bus);
        nm2.setTimeout(DBUS_TIMEOUT);
        nm2.call("DeactivateConnection", QVariant::fromValue(active_connection_raw));
      }
    }
  }
}

QVector<QDBusObjectPath> WifiManager::get_active_connections() {
  QDBusInterface nm(nm_service, nm_path, props_iface, bus);
  nm.setTimeout(DBUS_TIMEOUT);

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

void WifiManager::requestScan() {
  if (connecting_to_network == "" && !scanning) {  // TODO don't scan when tethering
    scanning = true;
    QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
    nm.setTimeout(DBUS_TIMEOUT);
    nm.call("RequestScan", QVariantMap());  // a signal is sent to WifiManager::property_change once the scan is complete
  }
}

uint WifiManager::get_wifi_device_state() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", device_iface, "State");
  uint resp = get_response<uint>(response);
  return resp;
}

QString WifiManager::get_active_ap() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}

QByteArray WifiManager::get_property(const QString &network_path , const QString &property) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", ap_iface, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(const QString &network_path) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  device_props.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::get_adapter() {
  QDBusInterface nm(nm_service, nm_path, nm_iface, bus);
  nm.setTimeout(DBUS_TIMEOUT);

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
    device_props.setTimeout(DBUS_TIMEOUT);

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

void WifiManager::state_change(unsigned int new_state, unsigned int old_state, unsigned int reason) {  // TODO prevent double running
  bool updateUI = false;
  raw_adapter_state = new_state;
//  qDebug() << "NEW:" << new_state << "OLD:" << old_state << "CHANGE:" << reason;

  // when connecting, we go into STATE_CONNECTING twice, skip second
  if (new_state == STATE_CONNECTING && old_state != STATE_NEED_AUTH) {
    qDebug() << "STATE CONNECTING!";  // TODO: add some delay where this won't trigger until 500ms pass and we haven't connected, to avoid double UI update calls
    updateUI = true;

  } else if (new_state == STATE_DISCONNECTED && connecting_to_network == "") {  // wait for connecting signal to update
    qDebug() << "STATE DISCONNECTED!";
    updateUI = true;

  } else if (new_state == STATE_NEED_AUTH && reason == REASON_WRONG_PASSWORD) {
    for (const Network n : seen_networks) {
      if (n.ssid == connecting_to_network) {
        qDebug() << "WRONG PASSWORD!";
        connecting_to_network = "";  // TODO unify connecting_to_network with seen_networks<network>.connecting
        emit wrongPassword(n);
        updateUI = true;
        break;  // TODO break and emit update networking at bottom
      }
    }

  } else if (new_state == STATE_CONNECTED) {
    qDebug() << "STATE CONNECTED!";
    updateUI = true;
    connecting_to_network = "";  // TODO unify connecting_to_network with seen_networks<network>.connecting
    emit successfulConnection(connecting_to_network);
  }

  if (updateUI) {
//    requestScan(); // TODO: only update seen_networks and emit
//    refreshNetworks();
    updateNetworks();
    emit updateNetworking(seen_networks, ipv4_address);
  }
}

// https://developer.gnome.org/NetworkManager/stable/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html
void WifiManager::property_change(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props) {
  if (interface == wireless_device_iface && props.contains("LastScan")) {
    refreshNetworks();
    emit updateNetworking(seen_networks, ipv4_address);
    qDebug() << "Scan complete";
    scanning = false;
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
  nm.setTimeout(DBUS_TIMEOUT);

  QDBusReply<QList<QDBusObjectPath>> response = nm.call("ListConnections");
  foreach (const QDBusObjectPath& path, response.value()) {
    QDBusInterface nm2(nm_service, path.path(), nm_settings_conn_iface, bus);
    nm2.setTimeout(DBUS_TIMEOUT);

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
    nm3.setTimeout(DBUS_TIMEOUT);
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
  nm_settings.setTimeout(DBUS_TIMEOUT);
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

void WifiManager::changeTetheringPassword(const QString newPassword) {
  qDebug() << "CHANGING PASSWORD!";
  tetheringPassword = newPassword;
  if (isKnownNetwork(tethering_ssid)) {
    forgetNetwork(tethering_ssid);
  }
  addTetheringConnection();
}
