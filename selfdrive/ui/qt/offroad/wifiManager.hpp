#pragma once

#include <QWidget>
#include <QtDBus>

enum class SecurityType {
  OPEN,
  WPA,
  UNSUPPORTED
};
enum class ConnectedType{
  DISCONNECTED,
  CONNECTING,
  CONNECTED
};

typedef QMap<QString, QMap<QString, QVariant>> Connection;
typedef QVector<QMap<QString, QVariant>> IpConfig;

struct Network {
  QString path;
  QByteArray ssid;
  unsigned int strength;
  ConnectedType connected;
  SecurityType security_type;
};

class WifiManager : public QWidget {
  Q_OBJECT
public:
  explicit WifiManager();

  void request_scan();
  QVector<Network> seen_networks;
  QString ipv4_address;

  void refreshNetworks();
  void connect(Network ssid);
  void connect(Network ssid, QString password);
  void connect(Network ssid, QString username, QString password);
  // Tethering functions

  void enableTethering();
  void disableTethering();
  bool tetheringEnabled();

private:
  QVector<QByteArray> seen_ssids;
  QString adapter;//Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  unsigned int raw_adapter_state;//Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;

  QString get_adapter();
  QString get_ipv4_address();
  QList<Network> get_networks();
  void connect(QByteArray ssid, QString username, QString password, SecurityType security_type);
  QString get_active_ap();
  void deactivate_connections(QString ssid);
  void clear_connections(QString ssid);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(QString network_path, QString property);
  unsigned int get_ap_strength(QString network_path);
  SecurityType getSecurityType(QString ssid);
  void disconnect();

private slots:
  void change(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
signals:
  void wrongPassword(QString ssid);
  void refresh();
};
