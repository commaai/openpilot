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
  QString ssid;
  unsigned int strength;
  ConnectedType connected;
  SecurityType security_type;
  bool known;
};

class WifiManager : public QObject {
  Q_OBJECT

public:
  explicit WifiManager();

  void request_scan();
  QVector<Network> seen_networks;
  QString ipv4_address;
  QString connecting_to_network;

  void refreshNetworks();
  void updateNetworks();
  bool isKnownNetwork(const QString &ssid);
  QVector<QPair<QString, QDBusObjectPath>> listConnections();

  void connect(const Network &ssid);
  void connect(const Network &ssid, const QString &password);
  void connect(const Network &ssid, const QString &username, const QString &password);
  void disconnect();

  // Tethering functions
  bool tetheringEnabled();
  void enableTethering();
  void disableTethering();

  void addTetheringConnection();
  void activateWifiConnection(const QString &ssid);
  void changeTetheringPassword(const QString &newPassword);

private:
  QVector<QString> seen_ssids;
  QString adapter;//Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  unsigned int raw_adapter_state;//Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString tethering_ssid;
  QString tetheringPassword = "swagswagcommma";

  QString get_adapter();
  QString get_ipv4_address();
  QList<Network> get_networks();
  void connect(const QString &ssid, const QString &username, const QString &password, SecurityType security_type);
  QString get_active_ap();
  void deactivateConnection(const QString &ssid);
  void forgetNetwork(const QString &ssid);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray get_property(const QString &network_path, const QString &property);
  unsigned int get_ap_strength(const QString &network_path);
  SecurityType getSecurityType(const QString &ssid);
  QDBusObjectPath pathFromSsid(const QString &ssid);

signals:  // Signals to communicate with Networking UI
  void wrongPassword(const Network n);
  void successfulConnection(const QString &ssid);

  void updateNetworking(const QVector<Network> seen_networks, const QString ipv4_address);

private slots:
  void change(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
};


class WifiThread : public QThread {
    Q_OBJECT

public:
  using QThread::QThread;
  void run() override;
  WifiManager *wifi;

  WifiThread() {
    wifi = new WifiManager();
    connect(wifi, &WifiManager::wrongPassword, this, &WifiThread::wrongPassword);

    moveToThread(this);
  }

public slots:
  void connectToNetwork(const Network n, const QString pass);
  void toggleTethering(const bool enabled);
  void changeTetheringPassword(const QString newPassword);
//  void wrongPassword(const QString ssid);

signals:
  // Callback to main UI thread
  void updateNetworking(const QVector<Network> seen_networks, const QString ipv4_address);
  void wrongPassword(const Network n);

  // Advanced networking signals
  void tetheringStateChange();

};
