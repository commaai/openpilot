#pragma once

#include <optional>

#include <QDBusPendingCallWatcher>
#include <QtDBus>
#include <QTimer>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
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
inline bool compare_by_strength(const Network &a, const Network &b) {
  return a.connected > b.connected || a.strength > b.strength;
}

class WifiManager : public QObject {
  Q_OBJECT

public:
  explicit WifiManager(QObject *parent);
  void requestScan();
  QMap<QString, Network> seenNetworks;
  QMap<QDBusObjectPath, QString> knownConnections;
  QDBusObjectPath lteConnectionPath;
  QString ipv4_address;

  void refreshNetworks();
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);
  void activateWifiConnection(const QString &ssid);
  void activateModemConnection(const QDBusObjectPath &path);
  NetworkType currentNetworkType();
  void updateGsmSettings(bool roaming, QString apn);
  void start();
  void stop();
  void connect(const Network &ssid, const QString &password = {}, const QString &username = {});
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
  bool stop_ = true;
  bool firstScan = true;
  QTimer timer;
  QString getAdapter(const uint = NM_DEVICE_TYPE_WIFI);
  uint getAdapterType(const QDBusObjectPath &path);
  QString get_ipv4_address();
  QString activeAp;
  void deactivateConnectionBySsid(const QString &ssid);
  void deactivateConnection(const QDBusObjectPath &path);
  QVector<QDBusObjectPath> get_active_connections();
  QByteArray get_property(const QString &network_path, const QString &property);
  SecurityType getSecurityType(const QMap<QString,QVariant> &properties);
  std::optional<QDBusObjectPath> getConnectionPath(const QString &ssid);
  Connection getConnectionSettings(const QDBusObjectPath &path);
  void initConnections();
  void setup();
  template <typename T = QDBusMessage, typename... Args>
  T call(const QString &path, const QString &interface, const QString &method, Args&&... args) {
    QDBusInterface nm = QDBusInterface(NM_DBUS_SERVICE, path, interface, bus);
    nm.setTimeout(DBUS_TIMEOUT);
    QDBusMessage response = nm.call(method, args...);
    if constexpr (std::is_same_v<T, QDBusMessage>) {
      return response;
    } else {
      if (response.arguments().count() >= 1) {
        QVariant vFirst = response.arguments().at(0).value<QDBusVariant>().variant();
        if (vFirst.canConvert<T>()) {
          return vFirst.value<T>();
        }
        QDebug critical = qCritical();
        critical << "Variant unpacking failure :" << method << ',';
        (critical << ... << args);
      }

      return T();
    }
  }

signals:
  void wrongPassword(const QString &ssid);
  void refreshSignal();

private slots:
  void stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
  void propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props);
  void deviceAdded(const QDBusObjectPath &path);
  void connectionRemoved(const QDBusObjectPath &path);
  void newConnection(const QDBusObjectPath &path);
  void activationFinished(QDBusPendingCallWatcher *call);
};
