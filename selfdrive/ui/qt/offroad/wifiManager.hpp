#pragma once

#include <QWidget>
#include <QtDBus>

enum class SecurityType {
  OPEN,
  WPA,
  UNSUPPORTED
};

typedef QMap<QString, QMap<QString, QVariant>> Connection;

struct Network {
  QString path;
  QByteArray ssid;
  unsigned int strength;
  bool connected;
  SecurityType security_type;
};

class WifiManager{
public:
  explicit WifiManager();

  bool has_adapter;
  void request_scan();
  QVector<Network> seen_networks;

  void refreshNetworks();
  void connect(Network ssid);
  void connect(Network ssid, QString password);
  void connect(Network ssid, QString username, QString password);

private:
  QVector<QByteArray> seen_ssids;
  QString adapter;//Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();

  QString get_adapter();
  QList<Network> get_networks();
  void connect(QByteArray ssid, QString username, QString password, SecurityType security_type);
  QString get_active_ap();
  void clear_connections(QString ssid);
  void print_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(QString network_path, QString property);
  unsigned int get_ap_strength(QString network_path);
  SecurityType getSecurityType(QString ssid);
};
