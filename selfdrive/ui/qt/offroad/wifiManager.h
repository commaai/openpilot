#pragma once

#include <QtDBus>
#include <QWidget>

#include "selfdrive/ui/qt/offroad/networkmanager.h"

enum class SecurityType {
  OPEN,
  WPA,
  UNSUPPORTED
};
enum class ConnectedType {
  DISCONNECTED,
  CONNECTING,
  CONNECTED
};
enum class NetworkType {
  NONE,
  WIFI,
  CELL,
  ETHERNET
};

typedef QMap<QString, QMap<QString, QVariant>> Connection;
typedef QVector<QMap<QString, QVariant>> IpConfig;

struct Network {
  QByteArray ssid;
  unsigned int strength;
  ConnectedType connected;
  SecurityType security_type;
};
bool compare_by_strength(const Network &a, const Network &b);

class WifiManager : public QWidget {
  Q_OBJECT

public:
  explicit WifiManager(QWidget* parent);

  void requestScan();
  QMap<QString, Network> seenNetworks;
  QMap<QDBusObjectPath, QString> knownConnections;
  QString lteConnectionPath;
  QString ipv4_address;

  void refreshNetworks();
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);
  void activateWifiConnection(const QString &ssid);
  NetworkType currentNetworkType();
  void setRoamingEnabled(bool roaming);

  void connect(const Network &ssid);
  void connect(const Network &ssid, const QString &password);
  void connect(const Network &ssid, const QString &username, const QString &password);
  void disconnect();

  // Tethering functions
  void setTetheringEnabled(bool enabled);
  bool isTetheringEnabled();
  void addTetheringConnection();
  void changeTetheringPassword(const QString &newPassword);
  QString getTetheringPassword();

private:
  QString adapter;  // Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  unsigned int raw_adapter_state;  // Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  const QString defaultTetheringPassword = "swagswagcomma";

  bool firstScan = true;
  QString getAdapter();
  bool isWirelessAdapter(const QDBusObjectPath &path);
  QString get_ipv4_address();
  void connect(const QByteArray &ssid, const QString &username, const QString &password, SecurityType security_type);
  QString activeAp;
  void initActiveAp();
  void deactivateConnection(const QString &ssid);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(const QString &network_path, const QString &property);
  unsigned int get_ap_strength(const QString &network_path);
  SecurityType getSecurityType(const QString &path);
  QDBusObjectPath getConnectionPath(const QString &ssid);
  Connection getConnectionSettings(const QDBusObjectPath &path);
  void initConnections();
  void setup();

signals:
  void wrongPassword(const QString &ssid);
  void refreshSignal();

private slots:
  void stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
  void propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props);
  void deviceAdded(const QDBusObjectPath &path);
  void connectionRemoved(const QDBusObjectPath &path);
  void newConnection(const QDBusObjectPath &path);
};
