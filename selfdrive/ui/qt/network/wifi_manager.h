#pragma once

#include <optional>
#include <QtDBus>
#include <QTimer>

#include "selfdrive/ui/qt/network/networkmanager.h"

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

typedef QMap<QString, QVariantMap> Connection;
typedef QVector<QVariantMap> IpConfig;

struct Network {
  QByteArray ssid;
  unsigned int strength;
  ConnectedType connected;
  SecurityType security_type;
};
bool compare_by_strength(const Network &a, const Network &b);
inline int strengthLevel(unsigned int strength) { return std::clamp((int)round(strength / 33.), 0, 3); }

class WifiManager : public QObject {
  Q_OBJECT

public:
  QMap<QString, Network> seenNetworks;
  QMap<QDBusObjectPath, QString> knownConnections;
  QString ipv4_address;

  explicit WifiManager(QObject* parent);
  void start();
  void stop();
  void requestScan();
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);
  std::optional<QDBusPendingCall> activateWifiConnection(const QString &ssid);
  NetworkType currentNetworkType();
  void updateGsmSettings(bool roaming, QString apn, bool metered);
  void connect(const Network &ssid, const QString &password = {}, const QString &username = {});

  // Tethering functions
  void setTetheringEnabled(bool enabled);
  bool isTetheringEnabled();
  void changeTetheringPassword(const QString &newPassword);
  QString getTetheringPassword();

private:
  QString adapter;  // Path to network manager wifi-device
  QTimer timer;
  unsigned int raw_adapter_state;  // Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  const QString defaultTetheringPassword = "swagswagcomma";
  QString activeAp;
  QDBusObjectPath lteConnectionPath;

  QString getAdapter(const uint = NM_DEVICE_TYPE_WIFI);
  uint getAdapterType(const QDBusObjectPath &path);
  QString getIp4Address();
  void deactivateConnectionBySsid(const QString &ssid);
  void deactivateConnection(const QDBusObjectPath &path);
  QVector<QDBusObjectPath> getActiveConnections();
  QByteArray get_property(const QString &network_path, const QString &property);
  SecurityType getSecurityType(const QVariantMap &properties);
  QDBusObjectPath getConnectionPath(const QString &ssid);
  Connection getConnectionSettings(const QDBusObjectPath &path);
  void initConnections();
  void setup();
  void refreshNetworks();
  void activateModemConnection(const QDBusObjectPath &path);
  void addTetheringConnection();
  void setCurrentConnecting(const QString &ssid);

signals:
  void wrongPassword(const QString &ssid);
  void refreshSignal();

private slots:
  void stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
  void propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props);
  void deviceAdded(const QDBusObjectPath &path);
  void connectionRemoved(const QDBusObjectPath &path);
  void newConnection(const QDBusObjectPath &path);
  void refreshFinished(QDBusPendingCallWatcher *call);
  void tetheringActivated(QDBusPendingCallWatcher *call);
};
