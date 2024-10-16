#pragma once

#include <cassert>
#include <optional>
#include <QtDBus>
#include <QTimer>
#include <iostream>
#include <dbus/dbus.h>

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
  bool tethering_on = false;
  bool ipv4_forward = false;

  explicit WifiManager(QObject* parent);
  void start();
  void stop();
  void requestScan();
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);
  std::optional<QDBusPendingCall> activateWifiConnection(const QString &ssid);
  NetworkType currentNetworkType();
  void updateGsmSettings(bool roaming, QString apn, bool metered);
  void connect(const Network &ssid, const bool is_hidden = false, const QString &password = {}, const QString &username = {});

  // Tethering functions
  void setTetheringEnabled(bool enabled);
  bool isTetheringEnabled();
  void changeTetheringPassword(const QString &newPassword);
  QString getTetheringPassword();

private:

  DBusConnection *dbus;
  DBusError error;

  template <typename T>
  T extractFromMessage(DBusMessage *message) {
    DBusMessageIter args;
    dbus_bool_t ret = dbus_message_iter_init(message, &args);
    assert(ret);

    if constexpr (std::is_same_v<T, uint32_t>) {
      assert(DBUS_TYPE_VARIANT == dbus_message_iter_get_arg_type(&args) && "Argument is not an variant");
      DBusMessageIter variantIter;
      dbus_message_iter_recurse(&args, &variantIter);
      assert(DBUS_TYPE_UINT32 == dbus_message_iter_get_arg_type(&variantIter) && "Argument is not an uint32.");
      uint32_t value;
      dbus_message_iter_get_basic(&variantIter, &value);
      return value;

    } else if constexpr (std::is_same_v<T, std::string>) {
      assert(DBUS_TYPE_STRING == dbus_message_iter_get_arg_type(&args) && "Argument is not a string.");
      const char *str;
      dbus_message_iter_get_basic(&args, &str);
      return std::string(str);

    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      assert(DBUS_TYPE_ARRAY == dbus_message_iter_get_arg_type(&args) && "Argument is not an array.");
      std::vector<std::string> result;
      DBusMessageIter subIter;
      dbus_message_iter_recurse(&args, &subIter);

      // Iterate over the array of strings
      while (dbus_message_iter_get_arg_type(&subIter) != DBUS_TYPE_INVALID) {
        const char *str;
        dbus_message_iter_get_basic(&subIter, &str);
        result.emplace_back(str);
        dbus_message_iter_next(&subIter);
      }

      return result;

    } else {
      // If T is unsupported, trigger a static assert
      assert(0 && "Unsupported type for extraction.");
    }
  }

  template <typename T=void, typename... Args>
  T sendMethodCall(const char *path, const char *interface, const char *method, Args &&...args) {
    DBusMessage *msg = dbus_message_new_method_call(NM_DBUS_SERVICE, path, interface, method);
    assert(msg != nullptr && "Failed to create D-Bus message");

    // Function to append the arguments one by one
    auto appendArgs = [&msg](auto &&arg) {
      using ArgType = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<ArgType, int>) {
        int v = arg;
        return dbus_message_append_args(msg, DBUS_TYPE_INT32, &v, DBUS_TYPE_INVALID);
      } else if constexpr (std::is_same_v<ArgType, const char *>) {
        const char *v = arg;
        return dbus_message_append_args(msg, DBUS_TYPE_STRING, &v, DBUS_TYPE_INVALID);
      } else {
        return false;  // Unsupported type
      }
    };

    bool success = (appendArgs(args) && ...);
    assert(success);

    DBusMessage *reply = dbus_connection_send_with_reply_and_block(dbus, msg, DBUS_TIMEOUT_INFINITE, &error);
    dbus_message_unref(msg);

    if (dbus_error_is_set(&error)) {
      std::cerr << "Error in D-Bus method call: " << error.message << std::endl;
      dbus_error_free(&error);
      assert(0);
    }
    if constexpr (std::is_void_v<T>) {
      dbus_message_unref(reply);
      return T();
    } else {
      T result = extractFromMessage<T>(reply);
      dbus_message_unref(reply);
      return result;
    }
  }


  QString adapter;  // Path to network manager wifi-device
  QTimer timer;
  unsigned int raw_adapter_state = NM_DEVICE_STATE_UNKNOWN;  // Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  const QString defaultTetheringPassword = "swagswagcomma";
  QString activeAp;
  QDBusObjectPath lteConnectionPath;

  QString getAdapter(const uint = NM_DEVICE_TYPE_WIFI);
  uint getAdapterType(const QDBusObjectPath &path);
  uint getAdapterType(const std::string &path);
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
