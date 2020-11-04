#include "qt/wifi.hpp"

#include <algorithm>
#include <set>

#include <QDebug>
#include <QListWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QInputDialog>
#include <QLineEdit>
#include <QCoreApplication>
#include <QButtonGroup>
#include <QStackedLayout>

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

WifiSettings::WifiSettings(QWidget *parent) : QWidget(parent) {
  vlayout = new QVBoxLayout;
  refresh();
  setLayout(vlayout);

  setStyleSheet(R"(
    QLabel { font-size: 40px }
    QPushButton { font-size: 40px }
    * {
      background-color: #114265;
    }
  )");

  // TODO: Handle NetworkManager not running
  // TODO: Handle no wireless adapter found
  // TODO: periodically request scan
  // TODO: periodically update network list
  // TODO: implement connecting (including case with wrong password)

  qDebug() << "Running";
}

void WifiSettings::refresh(){
  qDBusRegisterMetaType<Connection>();
  QString adapter = get_adapter();
  request_scan(adapter);

  QList<Network> networks = get_networks(adapter);
  QString active_ap = get_active_ap(adapter);
  QByteArray active_ssid = get_ap_ssid(active_ap);

  QButtonGroup* myButtongroup=new QButtonGroup(this);
  QObject::connect(myButtongroup, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));
  int i=0;
  for (Network &network : networks){
    bool connected = active_ap == network.path;
    unsigned int strength = std::round(network.strength / 25.0) * 25;

    if (seen_ssids.count(network.ssid) && !connected){
      continue;
    }

    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->addWidget(new QLabel(QString::fromUtf8(network.ssid)));
    QPixmap pix("../assets/offroad/indicator_wifi_" + QString::number(strength) + ".png");
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->addWidget(icon);
    hlayout->addSpacing(20);

    QPushButton* m_button = new QPushButton((connected ? "Connected" : "Connect")+(QString(i, QChar(' '))));
    m_button->setFixedWidth(250);
    m_button->setDisabled(connected);
    // connect(m_button, SIGNAL (released()), this, SLOT (handleButton(m_button)));
    myButtongroup->addButton(m_button,i++);

    hlayout->addWidget(m_button);
    hlayout->addSpacing(20);
    vlayout->addLayout(hlayout);

    seen_ssids.push_back(network.ssid);
    seen_networks.push_back(network);
  }
  qDebug() <<"Adding networks "<< i;
  // QPushButton* refreshButton = new QPushButton("Refresh networks");
  // connect(refreshButton, SIGNAL (released()), this, SLOT (refresh()));
  // vlayout->addWidget(refreshButton);
}

void WifiSettings::handleButton(QAbstractButton* m_button){
  int id = m_button->text().length()-7;  //7="Connect".length()
  qDebug() << "Clicked a button:" << id;
  qDebug() << get_ap_ssid(seen_networks[id].path);
  qDebug() << get_ap_security(seen_networks[id].path);
  QByteArray ssid = seen_ssids[id];
  bool ok;
  QString password = QInputDialog::getText(this, "Password for "+ssid, "Password", QLineEdit::Normal, QDir::home().dirName(), &ok);
  if(ok){
    connect_to(ssid, password);
  }else{
    qDebug() << "Connection cancelled, user not willing to provide a password.";
  }
}

QList<Network> WifiSettings::get_networks(QString adapter){
  QList<Network> r;
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("GetAllAccessPoints");
  QVariant first =  response.arguments().at(0);

  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    QByteArray ssid = get_ap_ssid(path.path());
    unsigned int strength = get_ap_strength(path.path());
    Network network = {path.path(), ssid, strength};

    if (ssid.length()){
      r.push_back(network);
    }
  }
  args.endArray();

  // Sort by strength
  std::sort(r.begin(), r.end(), compare_by_strength);

  return r;
}

void WifiSettings::connect_to(QByteArray ssid, QString password){
  // TODO: handle different authentication types, None, WEP, WPA, WPA Enterprise
  // TODO: hande exisiting connection for same ssid

  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "Connection 2";

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

void WifiSettings::request_scan(QString adapter){
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface nm(nm_service, adapter, wireless_device_iface, bus);
  QDBusMessage response = nm.call("RequestScan",  QVariantMap());

  qDebug() << response;
}

QString WifiSettings::get_active_ap(QString adapter){
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, adapter, props_iface, bus);
  QDBusMessage response = device_props.call("Get", wireless_device_iface, "ActiveAccessPoint");
  QDBusObjectPath r = get_response<QDBusObjectPath>(response);
  return r.path();
}

QByteArray WifiSettings::get_ap_ssid(QString network_path){
  // TODO: abstract get propery function with template
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "Ssid");
  return get_response<QByteArray>(response);
}
QByteArray WifiSettings::get_ap_security(QString network_path){
  // TODO: abstract get propery function with template
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "WpaFlags");
  return get_response<QByteArray>(response);
}

unsigned int WifiSettings::get_ap_strength(QString network_path){
  qDebug() << network_path;
  // TODO: abstract get propery function with template
  QDBusConnection bus = QDBusConnection::systemBus();
  QDBusInterface device_props(nm_service, network_path, props_iface, bus);
  QDBusMessage response = device_props.call("Get", ap_iface, "Strength");
  return get_response<unsigned int>(response);
}

QString WifiSettings::get_adapter(){
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
