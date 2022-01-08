#pragma once

#include <QtDBus>

#include "selfdrive/common/swaglog.h"
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
  uint32_t strength;
  ConnectedType connected;
  SecurityType security_type;
};
bool compare_by_strength(const Network &a, const Network &b);

class WifiManager : public QObject {
  Q_OBJECT

public:
  explicit WifiManager(QObject* parent);
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);
  void activateWifiConnection(const QString &ssid);
  void activateModemConnection(const QDBusObjectPath &path);
  NetworkType currentNetworkType();
  void updateGsmSettings(bool roaming, QString apn);
  void start();
  inline void stop() { stop_ = true; }
  void connect(const Network &ssid, const QString &password = {}, const QString &username = {});
  void disconnect();
  // Tethering functions
  void setTetheringEnabled(bool enabled);
  bool isTetheringEnabled();
  void addTetheringConnection();
  void changeTetheringPassword(const QString &newPassword);
  QString getTetheringPassword();

  QMap<QString, Network> seenNetworks;
  QString ipv4_address;

private:
  QString adapter;  // Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  uint32_t raw_adapter_state;  // Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  const QString defaultTetheringPassword = "swagswagcomma";
  bool stop_ = true;
  QMap<QDBusObjectPath, QString> knownConnections;
  QDBusObjectPath lteConnectionPath;
  bool firstScan = true;
  void refreshNetworks();
  void requestScan();
  QString getAdapter(const uint = NM_DEVICE_TYPE_WIFI);
  uint getAdapterType(const QDBusObjectPath &path);
  bool isWirelessAdapter(const QDBusObjectPath &path);
  QString get_ipv4_address();
  QString activeAp;
  void initActiveAp();
  void deactivateConnectionBySsid(const QString &ssid);
  void deactivateConnection(const QDBusObjectPath &path);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(const QString &network_path, const QString &property);
  uint32_t get_ap_strength(const QString &network_path);
  SecurityType getSecurityType(const QString &path);
  QDBusObjectPath getConnectionPath(const QString &ssid);
  Connection getConnectionSettings(const QDBusObjectPath &path);
  void initConnections();
  void setup();
  template <typename T = QDBusMessage, typename... Args>
  T call(const QString &path, const QString &interface, const QString &method, Args... args) {
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
      }
      LOGE("Variant unpacking failure");
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
};
