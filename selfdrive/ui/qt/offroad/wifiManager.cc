#include <algorithm>
#include <set>

#include "wifiManager.hpp"
#include "wifi.hpp"
typedef QMap<QString, QMap<QString, QVariant> > Connection;

QString nm_path = "/org/freedesktop/NetworkManager";
QString nm_settings_path = "/org/freedesktop/NetworkManager/Settings";

QString nm_iface = "org.freedesktop.NetworkManager";
QString props_iface = "org.freedesktop.DBus.Properties";
QString nm_settings_iface = "org.freedesktop.NetworkManager.Settings";
QString device_iface = "org.freedesktop.NetworkManager.Device";
QString wireless_device_iface = "org.freedesktop.NetworkManager.Device.Wireless";
QString ap_iface = "org.freedesktop.NetworkManager.AccessPoint";

QString nm_service = "org.freedesktop.NetworkManager";


template <typename T>
T get_response(QDBusMessage response){
  QVariant first =  response.arguments().at(0);
  QDBusVariant dbvFirst = first.value<QDBusVariant>();
  QVariant vFirst = dbvFirst.variant();
  return vFirst.value<T>();
}

bool compare_by_strength(const Network &a, const Network &b){
  return a.strength > b.strength;
}
WifiManager::WifiManager(){
  refreshNetworks();
}

void WifiManager::refreshNetworks(){
  seen_networks.clear();
  seen_ssids.clear();

  qDBusRegisterMetaType<Connection>();
  QString adapter = get_adapter();
  request_scan(adapter);
  QString active_ap = get_active_ap(adapter);

  QList<Network> all_networks = get_networks(adapter);
  QByteArray active_ssid = get_property(active_ap, "Ssid");

  for (Network &network : all_networks){
    if(seen_ssids.count(network.ssid)){
      continue;
    }
    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
  qDebug() <<"Adding networks ";
}

QList<Network> WifiManager::get_networks(QString adapter){
  QList<Network> r;
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("GetAllAccessPoints");
  QVariant first =  response.arguments().at(0);

  QString active_ap = get_active_ap(adapter);

  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    QByteArray ssid = get_property(path.path(), "Ssid");
    unsigned int strength = get_ap_strength(path.path());
    Network network = {path.path(), ssid, strength, path.path()==active_ap};

    if (ssid.length()){
      r.push_back(network);
    }
  }
  args.endArray();

  // Sort by strength
  std::sort(r.begin(), r.end(), compare_by_strength);

  return r;
}

void WifiManager::connect_to_open(QByteArray ssid){

  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "Connection open";

  connection["802-11-wireless"]["ssid"] = ssid;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";

  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusReply<QDBusObjectPath> result = nm_settings.call("AddConnection", QVariant::fromValue(connection));
  if (!result.isValid()) {
    qDebug() << result.error().name() << result.error().message();
  } else {
    qDebug() << result.value().path();
  }

}

void WifiManager::connect_to_WPA(QByteArray ssid, QString password){
  // TODO: handle different authentication types, None, WEP, WPA, WPA Enterprise
  // TODO: hande exisiting connection for same ssid

  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "Connection WPA";

  connection["802-11-wireless"]["ssid"] = ssid;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["auth-alg"] = "open";
  connection["802-11-wireless-security"]["psk"] = password;

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";


  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusReply<QDBusObjectPath> result = nm_settings.call("AddConnection", QVariant::fromValue(connection));
  if (!result.isValid()) {
    qDebug() << result.error().name() << result.error().message();
  } else {
    qDebug() << result.value().path();
  }

}

void WifiManager::request_scan(QString adapter){
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("RequestScan",  QVariantMap());

  qDebug() << response;
}
QString WifiManager::get_active_ap(QString adapter){
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}
QByteArray WifiManager::get_property(QString network_path ,QString property){
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(QString network_path){
  // TODO: abstract get propery function with template
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::get_adapter(){
  QDBusConnection bus = QDBusConnection::systemBus();

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

    if (device_type == 2){ // Wireless
      adapter_path = path.path();
      break;
    }
  }
  args.endArray();

  return adapter_path;
}