#include <algorithm>
#include <set>

#include "wifiManager.hpp"
#include "wifi.hpp"
typedef QMap<QString, QMap<QString, QVariant> > Connection;

QString nm_path               = "/org/freedesktop/NetworkManager";
QString nm_settings_path      = "/org/freedesktop/NetworkManager/Settings";

QString nm_iface                    = "org.freedesktop.NetworkManager";
QString props_iface                 = "org.freedesktop.DBus.Properties";
QString nm_settings_iface           = "org.freedesktop.NetworkManager.Settings";
QString nm_settings_conn_iface      = "org.freedesktop.NetworkManager.Settings.Connection";
QString device_iface                = "org.freedesktop.NetworkManager.Device";
QString wireless_device_iface       = "org.freedesktop.NetworkManager.Device.Wireless";
QString ap_iface                    = "org.freedesktop.NetworkManager.AccessPoint";
QString connection_iface            = "org.freedesktop.NetworkManager.Connection.Active";

QString nm_service            = "org.freedesktop.NetworkManager";


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
  adapter = get_adapter();
  bus = QDBusConnection::systemBus();
  seen_networks.clear();
  seen_ssids.clear();
  qDebug() << "Device path" << adapter ;
  
  QDBusInterface nm(nm_service, nm_path, props_iface, bus);
  QDBusMessage response = nm.call("Get", nm_iface, "ActiveConnections");
  qDebug() << response;
  QVariant step1 = response.arguments().at(0);
  qDebug() << step1;
  QDBusVariant dbvFirst = step1.value<QDBusVariant>();
  QVariant converted = dbvFirst.variant();
  qDebug()<<converted;
  QDBusArgument step4 = converted.value<QDBusArgument>();
  qDebug() << "QDBusArgument current type is" << step4.currentType();
  QDBusObjectPath path;
  step4.beginArray();
  while (!step4.atEnd())
  {
      step4 >> path;
      qDebug()<<path.path();
  }
  step4.endArray();



  qDBusRegisterMetaType<Connection>();
  request_scan();
  QString active_ap = get_active_ap();
  QByteArray active_ssid = get_property(active_ap, "Ssid");
  qDebug() << "Currently active network is:" << active_ssid;

  for (Network &network : get_networks()){
    if(seen_ssids.count(network.ssid)){
      continue;
    }
    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
  qDebug() <<"Adding networks ";
}

QList<Network> WifiManager::get_networks(){
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
    int security = getSecurityType(path.path());
    // qDebug() << "AP "<<ssid<<"has adress"<<get_property(path.path(), "HwAddress");
    Network network = {path.path(), ssid, strength, path.path()==active_ap, security};

    if (ssid.length()){
      r.push_back(network);
    }
  }
  args.endArray();

  // Sort by strength
  std::sort(r.begin(), r.end(), compare_by_strength);
  return r;
}

int WifiManager::getSecurityType(QString path){
  int sflag = get_property(path, "Flags").toInt();
  int wpaflag = get_property(path, "WpaFlags").toInt();
  // int rsnflag = get_property(path, "RsnFlags").toInt();

  if(sflag == 0){
    return 0;
  }else if(sflag == 1 && wpaflag < 400){
    return 1;
  }else{
    // qDebug() << "Cannot determine security type for " << get_property(path, "Ssid") << " with flags"; 
    // qDebug() << "flag    " << sflag;
    // qDebug() << "WpaFlag " << wpaflag;
    // qDebug() << "RsnFlag " << rsnflag;
    return -1;
  }
}
void WifiManager::connect(Network n){
  return connect(n,"","");
}
void WifiManager::connect(Network n, QString password){
  return connect(n, "", password);
}

void WifiManager::connect(Network n, QString username, QString password){
  clear_connections(n.ssid);
  qDebug() << "Connecting to"<< n.ssid << "with username, password =" << username << "," <<password;
  if(n.security_type==0){
    connect_to_open(n.ssid);
  }else if(n.security_type == 1){
    connect_to_WPA(n.ssid, password);
  }else{
    qDebug() << "Network cannot be connected to; unknown security type";
  }
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


  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusReply<QDBusObjectPath> result = nm_settings.call("AddConnection", QVariant::fromValue(connection));
  if (!result.isValid()) {
    qDebug() << result.error().name() << result.error().message();
  } else {
    qDebug() << result.value().path();
  }

}

void WifiManager::clear_connections(QString ssid){
  QDBusInterface nm(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusMessage response = nm.call("ListConnections");
  QVariant first =  response.arguments().at(0);
  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  // qDebug()<<"Printing list of connections; ";
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;
    // qDebug()<<path.path();
    QDBusInterface nm2(nm_service, path.path(), nm_settings_conn_iface, bus);
    QDBusMessage response = nm2.call("GetSettings");

    const QDBusArgument &dbusArg = response.arguments().at(0).value<QDBusArgument>();

    QMap<QString,QMap<QString,QVariant> > map;
    dbusArg >> map;
    for( QString outer_key : map.keys() ){
        QMap<QString,QVariant> innerMap = map.value(outer_key);

        // qDebug() << "Key: " << outer_key;
        for( QString inner_key : innerMap.keys() ){
            // qDebug() << "    " << inner_key << ":" << innerMap.value(inner_key);
            if(inner_key=="ssid"){
              QString value = innerMap.value(inner_key).value<QString>();
              if(value == ssid){
                qDebug()<<"Deleting "<<value;
                nm2.call("Delete");
              }
            }
        }
    }
  }
}
void WifiManager::request_scan(){
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("RequestScan",  QVariantMap());

  qDebug() << response;
}
uint WifiManager::get_wifi_device_state(){
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", device_iface, "State");
  uint resp = get_response<uint>(response);
  return resp;
}
QString WifiManager::get_active_ap(){
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}
QByteArray WifiManager::get_property(QString network_path ,QString property){
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, property);
  return get_response<QByteArray>(response);
}

unsigned int WifiManager::get_ap_strength(QString network_path){
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiManager::get_adapter(){

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