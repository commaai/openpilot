#include <algorithm>
#include <set>
#include <stdlib.h>
#include <iostream>
#include "wifiManager.hpp"
#include "common/params.h"

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


QString nm_path                = "/org/freedesktop/NetworkManager";
QString nm_settings_path       = "/org/freedesktop/NetworkManager/Settings";

QString nm_iface               = "org.freedesktop.NetworkManager";
QString props_iface            = "org.freedesktop.DBus.Properties";
QString nm_settings_iface      = "org.freedesktop.NetworkManager.Settings";
QString nm_settings_conn_iface = "org.freedesktop.NetworkManager.Settings.Connection";
QString device_iface           = "org.freedesktop.NetworkManager.Device";
QString wireless_device_iface  = "org.freedesktop.NetworkManager.Device.Wireless";
QString ap_iface               = "org.freedesktop.NetworkManager.AccessPoint";
QString connection_iface       = "org.freedesktop.NetworkManager.Connection.Active";
QString ipv4config_iface       = "org.freedesktop.NetworkManager.IP4Config";

QString nm_service             = "org.freedesktop.NetworkManager";

const int state_connected = 100;
const int state_need_auth = 60;
const int reason_wrong_password = 8;

template <typename T>
T get_response(QDBusMessage response) {
  QVariant first =  response.arguments().at(0);
  QDBusVariant dbvFirst = first.value<QDBusVariant>();
  QVariant vFirst = dbvFirst.variant();
  return vFirst.value<T>();
}

bool compare_by_strength(const Network &a, const Network &b) {
  if (a.connected == ConnectedType::CONNECTED) return true;
  if (b.connected == ConnectedType::CONNECTED) return false;
  if (a.connected == ConnectedType::CONNECTING) return true;
  if (b.connected == ConnectedType::CONNECTING) return false;
  return a.strength > b.strength;
}


WifiManager::WifiManager() {
  qDBusRegisterMetaType<Connection>();
  qDBusRegisterMetaType<IpConfig>();
  connecting_to_network = "";
  adapter = get_adapter();

  bool has_adapter = adapter != "";
  if (!has_adapter){
    throw std::runtime_error("Error connecting to panda");
  }

  QDBusInterface nm(nm_service, adapter, device_iface, bus);
  bus.connect(nm_service, adapter, device_iface, "StateChanged", this, SLOT(change(unsigned int, unsigned int, unsigned int)));

  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", device_iface, "State");
  raw_adapter_state = get_response<uint>(response);
  change(raw_adapter_state, 0, 0);

  // Compute tethering ssid as "Weedle" + first 4 characters of a dongle id
  tethering_ssid = "weedle";
  std::string bytes = Params().get("DongleId");
  if (bytes.length() >= 4) {
    tethering_ssid+="-"+QString::fromStdString(bytes.substr(0,4));
  }
}

void WifiManager::refreshNetworks() {
  bus = QDBusConnection::systemBus();
  seen_networks.clear();
  seen_ssids.clear();
  ipv4_address = get_ipv4_address();
  for (Network &network : get_networks()) {
    if (seen_ssids.count(network.ssid)) {
      continue;
    }
    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
}

QString WifiManager::get_ipv4_address(){
  if (raw_adapter_state != state_connected){
    return "";
  }
  QVector<QDBusObjectPath> conns = get_active_connections();
  for (auto p : conns){
    QString active_connection = p.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "Ip4Config"));
    QString ip4config = pth.path();

    QDBusInterface nm2(nm_service, ip4config, props_iface, bus);
    const QDBusArgument &arr = get_response<QDBusArgument>(nm2.call("Get", ipv4config_iface, "AddressData"));
    QMap<QString, QVariant> pth2;
    arr.beginArray();
    while (!arr.atEnd()){
      arr >> pth2;
      QString ipv4 = pth2.value("address").value<QString>();
      arr.endArray();
      return ipv4;
    }
    arr.endArray();
  }
  return "";
}

QList<Network> WifiManager::get_networks() {
  QList<Network> r;
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("GetAllAccessPoints");
  QVariant first =  response.arguments().at(0);

  QString active_ap = get_active_ap();
  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    QByteArray ssid = get_property(path.path(), "Ssid");
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
    Network network = {path.path(), ssid, strength, ctype, security};

    if (ssid.length()) {
      r.push_back(network);
    }
  }
  args.endArray();

  std::sort(r.begin(), r.end(), compare_by_strength);
  return r;
}

SecurityType WifiManager::getSecurityType(QString path) {
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

void WifiManager::connect(Network n) {
  return connect(n, "", "");
}

void WifiManager::connect(Network n, QString password) {
  return connect(n, "", password);
}

void WifiManager::connect(Network n, QString username, QString password) {
  connecting_to_network = n.ssid;
  disconnect();
  clear_connections(n.ssid); //Clear all connections that may already exist to the network we are connecting
  connect(n.ssid, username, password, n.security_type);
}

void WifiManager::connect(QByteArray ssid, QString username, QString password, SecurityType security_type) {
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "OpenPilot connection "+QString::fromStdString(ssid.toStdString());
  connection["connection"]["autoconnect-retries"] = 0;

  connection["802-11-wireless"]["ssid"] = ssid;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  if (security_type == SecurityType::WPA) {
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["auth-alg"] = "open";
    connection["802-11-wireless-security"]["psk"] = password;
  }

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";

  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  nm_settings.call("AddConnection", QVariant::fromValue(connection));
}

void WifiManager::deactivate_connections(QString ssid) {
  for (QDBusObjectPath active_connection_raw : get_active_connections()) {
    QString active_connection = active_connection_raw.path();
    QDBusInterface nm(nm_service, active_connection, props_iface, bus);
    QDBusObjectPath pth = get_response<QDBusObjectPath>(nm.call("Get", connection_iface, "SpecificObject"));
    QString Ssid = get_property(pth.path(), "Ssid");
    if (Ssid == ssid) {
      QDBusInterface nm2(nm_service, nm_path, nm_iface, bus);
      nm2.call("DeactivateConnection", QVariant::fromValue(active_connection_raw));
    }
  }
}

QVector<QDBusObjectPath> WifiManager::get_active_connections() {
  QDBusInterface nm(nm_service, nm_path, props_iface, bus);
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

void WifiManager::clear_connections(QString ssid) {
  QDBusInterface nm(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusMessage response = nm.call("ListConnections");
  QVariant first =  response.arguments().at(0);
  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;
    QDBusInterface nm2(nm_service, path.path(), nm_settings_conn_iface, bus);
    QDBusMessage response = nm2.call("GetSettings");

    const QDBusArgument &dbusArg = response.arguments().at(0).value<QDBusArgument>();

    QMap<QString,QMap<QString,QVariant> > map;
    dbusArg >> map;
    for (QString outer_key : map.keys()) {
      QMap<QString,QVariant> innerMap = map.value(outer_key);
      for (QString inner_key : innerMap.keys()) {
        if (inner_key == "ssid") {
          QString value = innerMap.value(inner_key).value<QString>();
          if (value == ssid) {
            nm2.call("Delete");
          }
        }
      }
    }
  }
}

void WifiManager::request_scan() {
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  nm.call("RequestScan",  QVariantMap());
}

uint WifiManager::get_wifi_device_state() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", device_iface, "State");
  uint resp = get_response<uint>(response);
  return resp;
}

QString WifiManager::get_active_ap() {
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}

QByteArray WifiManager::get_property(QString network_path ,QString property) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(QString network_path) {
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::get_adapter() {

  QDBusInterface nm(nm_service, nm_path, nm_iface, bus);
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

void WifiManager::change(unsigned int new_state,unsigned int previous_state,unsigned int change_reason) {
  raw_adapter_state = new_state;
  if (new_state == state_need_auth && change_reason == reason_wrong_password) {
    emit wrongPassword(connecting_to_network);
  } else if (new_state == state_connected) {
    connecting_to_network = "";
  }
}

void WifiManager::disconnect() {
  QString active_ap = get_active_ap();
  if (active_ap!="" && active_ap!="/") {
    deactivate_connections(get_property(active_ap, "Ssid"));
  }
}

//Functions for tethering

void WifiManager::enableTethering() {
  disconnect();
  Connection connection;
  connection["connection"]["id"] = "Hotspot";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["interface-name"] = "wlan0";

  connection["802-11-wireless"]["band"] = "bg";
  connection["802-11-wireless"]["mode"] = "ap";
  connection["802-11-wireless"]["ssid"] = tethering_ssid.toUtf8();

  connection["802-11-wireless-security"]["group"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["pairwise"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["proto"] = QStringList("rsn");
  connection["802-11-wireless-security"]["psk"] = "swagswagcomma";

  connection["ipv4"]["method"] = "shared";
  QMap<QString,QVariant> address;
  address["address"] = "192.168.43.1";
  address["prefix"] = 24u;
  connection["ipv4"]["address-data"] = QVariant::fromValue(IpConfig() << address);
  connection["ipv4"]["gateway"] = "192.168.43.1";
  connection["ipv6"]["method"] = "ignore";

  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  nm_settings.call("AddConnection", QVariant::fromValue(connection));

}

void WifiManager::disableTethering() {
  clear_connections(tethering_ssid);
}

bool WifiManager::tetheringEnabled() {
  QString active_ap = get_active_ap();
  return get_property(active_ap, "Ssid") == tethering_ssid;
}
