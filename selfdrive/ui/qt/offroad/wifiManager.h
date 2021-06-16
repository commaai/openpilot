#pragma once

#include <QtDBus>
#include <QWidget>

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
  explicit WifiManager(QWidget* parent);

  void request_scan();
  QVector<Network> seen_networks;
  QString ipv4_address;

  void refreshNetworks();
  bool isKnownNetwork(const QString &ssid);

  void connect(const Network &ssid);
  void connect(const Network &ssid, const QString &password);
  void connect(const Network &ssid, const QString &username, const QString &password);
  void disconnect();

  // Tethering functions
  void enableTethering();
  void disableTethering();
  bool tetheringEnabled();

  void addTetheringConnection();
  void activateWifiConnection(const QString &ssid);
  void changeTetheringPassword(const QString &newPassword);

private:
  QVector<QByteArray> seen_ssids;
  QString adapter;//Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  unsigned int raw_adapter_state;//Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  QString tetheringPassword = "swagswagcommma";

  QString get_adapter();
  QString get_ipv4_address();
  QList<Network> get_networks();
  void connect(const QByteArray &ssid, const QString &username, const QString &password, SecurityType security_type);
  QString get_active_ap();
  void deactivateConnection(const QString &ssid);
  void forgetNetwork(const QString &ssid);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(const QString &network_path, const QString &property);
  unsigned int get_ap_strength(const QString &network_path);
  SecurityType getSecurityType(const QString &ssid);
  QDBusObjectPath pathFromSsid(const QString &ssid);
  QVector<QPair<QString, QDBusObjectPath>> listConnections();

private slots:
  void change(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
signals:
  void wrongPassword(const QString &ssid);
  void successfulConnection(const QString &ssid);
  void refresh();
};
