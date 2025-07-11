#include "selfdrive/ui/qt/network/wifi_manager.h"

#include <utility>

#include "common/swaglog.h"
#include "selfdrive/ui/qt/util.h"

bool compare_by_strength(const Network &a, const Network &b) {
  return std::tuple(a.connected, strengthLevel(a.strength), b.ssid) >
         std::tuple(b.connected, strengthLevel(b.strength), a.ssid);
}

template <typename T = QDBusMessage, typename... Args>
T call(const QString &path, const QString &interface, const QString &method, Args &&...args) {
  QDBusInterface nm(NM_DBUS_SERVICE, path, interface, QDBusConnection::systemBus());
  nm.setTimeout(DBUS_TIMEOUT);

  QDBusMessage response = nm.call(method, std::forward<Args>(args)...);
  if (response.type() == QDBusMessage::ErrorMessage) {
    qCritical() << "DBus call error:" << response.errorMessage();
    return T();
  }

  if constexpr (std::is_same_v<T, QDBusMessage>) {
    return response;
  } else if (response.arguments().count() >= 1) {
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

template <typename... Args>
QDBusPendingCall asyncCall(const QString &path, const QString &interface, const QString &method, Args &&...args) {
  QDBusInterface nm = QDBusInterface(NM_DBUS_SERVICE, path, interface, QDBusConnection::systemBus());
  return nm.asyncCall(method, args...);
}

bool emptyPath(const QString &path) {
  return path == "" || path == "/";
}

WifiManager::WifiManager(QObject *parent) : QObject(parent) {
  qDBusRegisterMetaType<Connection>();
  qDBusRegisterMetaType<IpConfig>();

  // Set tethering ssid as "weedle" + first 4 characters of a dongle id
  tethering_ssid = "weedle";
  if (auto dongle_id = getDongleId()) {
    tethering_ssid += "-" + dongle_id->left(4);
  }

  adapter = getAdapter();
  if (!adapter.isEmpty()) {
    setup();
  } else {
    QDBusConnection::systemBus().connect(NM_DBUS_SERVICE, NM_DBUS_PATH, NM_DBUS_INTERFACE, "DeviceAdded", this, SLOT(deviceAdded(QDBusObjectPath)));
  }

  timer.callOnTimeout(this, &WifiManager::requestScan);

  initConnections();
}

void WifiManager::setup() {
  auto bus = QDBusConnection::systemBus();
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_DEVICE, "StateChanged", this, SLOT(stateChange(unsigned int, unsigned int, unsigned int)));
  bus.connect(NM_DBUS_SERVICE, adapter, NM_DBUS_INTERFACE_PROPERTIES, "PropertiesChanged", this, SLOT(propertyChange(QString, QVariantMap, QStringList)));

  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "ConnectionRemoved", this, SLOT(connectionRemoved(QDBusObjectPath)));
  bus.connect(NM_DBUS_SERVICE, NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "NewConnection", this, SLOT(newConnection(QDBusObjectPath)));

  raw_adapter_state = call<uint>(adapter, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_DEVICE, "State");
  activeAp = call<QDBusObjectPath>(adapter, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_DEVICE_WIRELESS, "ActiveAccessPoint").path();

  requestScan();
}

void WifiManager::start() {
  timer.start(5000);
  refreshNetworks();
}

void WifiManager::stop() {
  timer.stop();
}

void WifiManager::refreshNetworks() {
  if (adapter.isEmpty() || !timer.isActive()) return;

  QDBusPendingCall pending_call = asyncCall(adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, "GetAllAccessPoints");
  QDBusPendingCallWatcher *watcher = new QDBusPendingCallWatcher(pending_call);
  QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, &WifiManager::refreshFinished);
}

void WifiManager::refreshFinished(QDBusPendingCallWatcher *watcher) {
  ipv4_address = getIp4Address();
  seenNetworks.clear();

  const QDBusReply<QList<QDBusObjectPath>> watcher_reply = *watcher;
  if (!watcher_reply.isValid()) {
    qCritical() << "Failed to refresh";
    watcher->deleteLater();
    return;
  }

  for (const QDBusObjectPath &path : watcher_reply.value()) {
    QDBusReply<QVariantMap> reply = call(path.path(), NM_DBUS_INTERFACE_PROPERTIES, "GetAll", NM_DBUS_INTERFACE_ACCESS_POINT);
    if (!reply.isValid()) {
      qCritical() << "Failed to retrieve properties for path:" << path.path();
      continue;
    }

    auto properties = reply.value();
    const QByteArray ssid = properties["Ssid"].toByteArray();
    if (ssid.isEmpty()) continue;

    // May be multiple access points for each SSID.
    // Use first for ssid and security type, then update connected status and strength using all
    if (!seenNetworks.contains(ssid)) {
      seenNetworks[ssid] = {ssid, 0U, ConnectedType::DISCONNECTED, getSecurityType(properties)};
    }

    if (path.path() == activeAp) {
      seenNetworks[ssid].connected = (ssid == connecting_to_network) ? ConnectedType::CONNECTING : ConnectedType::CONNECTED;
    }

    uint32_t strength = properties["Strength"].toUInt();
    if (seenNetworks[ssid].strength < strength) {
      seenNetworks[ssid].strength = strength;
    }
  }

  emit refreshSignal();
  watcher->deleteLater();
}

QString WifiManager::getIp4Address() {
  if (raw_adapter_state != NM_DEVICE_STATE_ACTIVATED) return "";

  for (const auto &p : getActiveConnections()) {
    QString type = call<QString>(p.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");
    if (type == "802-11-wireless") {
      auto ip4config = call<QDBusObjectPath>(p.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Ip4Config");
      const auto &arr = call<QDBusArgument>(ip4config.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_IP4_CONFIG, "AddressData");
      QVariantMap path;
      arr.beginArray();
      while (!arr.atEnd()) {
        arr >> path;
        arr.endArray();
        return path.value("address").value<QString>();
      }
      arr.endArray();
    }
  }
  return "";
}

SecurityType WifiManager::getSecurityType(const QVariantMap &properties) {
  int sflag = properties["Flags"].toUInt();
  int wpaflag = properties["WpaFlags"].toUInt();
  int rsnflag = properties["RsnFlags"].toUInt();
  int wpa_props = wpaflag | rsnflag;

  // obtained by looking at flags of networks in the office as reported by an Android phone
  const int supports_wpa = NM_802_11_AP_SEC_PAIR_WEP40 | NM_802_11_AP_SEC_PAIR_WEP104 | NM_802_11_AP_SEC_GROUP_WEP40 | NM_802_11_AP_SEC_GROUP_WEP104 | NM_802_11_AP_SEC_KEY_MGMT_PSK;

  if ((sflag == NM_802_11_AP_FLAGS_NONE) || ((sflag & NM_802_11_AP_FLAGS_WPS) && !(wpa_props & supports_wpa))) {
    return SecurityType::OPEN;
  } else if ((sflag & NM_802_11_AP_FLAGS_PRIVACY) && (wpa_props & supports_wpa) && !(wpa_props & NM_802_11_AP_SEC_KEY_MGMT_802_1X)) {
    return SecurityType::WPA;
  } else {
    LOGW("Unsupported network! sflag: %d, wpaflag: %d, rsnflag: %d", sflag, wpaflag, rsnflag);
    return SecurityType::UNSUPPORTED;
  }
}

void WifiManager::connect(const Network &n, const bool is_hidden, const QString &password, const QString &username) {
  setCurrentConnecting(n.ssid);
  forgetConnection(n.ssid);  // Clear all connections that may already exist to the network we are connecting
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "openpilot connection " + QString::fromStdString(n.ssid.toStdString());
  connection["connection"]["autoconnect-retries"] = 0;

  connection["802-11-wireless"]["ssid"] = n.ssid;
  connection["802-11-wireless"]["hidden"] = is_hidden;
  connection["802-11-wireless"]["mode"] = "infrastructure";

  if (n.security_type == SecurityType::WPA) {
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["auth-alg"] = "open";
    connection["802-11-wireless-security"]["psk"] = password;
  }

  connection["ipv4"]["method"] = "auto";
  connection["ipv4"]["dns-priority"] = 600;
  connection["ipv6"]["method"] = "ignore";

  asyncCall(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "AddConnection", QVariant::fromValue(connection));
}

void WifiManager::deactivateConnectionBySsid(const QString &ssid) {
  for (QDBusObjectPath active_connection : getActiveConnections()) {
    auto pth = call<QDBusObjectPath>(active_connection.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "SpecificObject");
    if (!emptyPath(pth.path())) {
      QString Ssid = get_property(pth.path(), "Ssid");
      if (Ssid == ssid) {
        deactivateConnection(active_connection);
        return;
      }
    }
  }
}

void WifiManager::deactivateConnection(const QDBusObjectPath &path) {
  asyncCall(NM_DBUS_PATH, NM_DBUS_INTERFACE, "DeactivateConnection", QVariant::fromValue(path));
}

QVector<QDBusObjectPath> WifiManager::getActiveConnections() {
  auto result = call<QDBusArgument>(NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE, "ActiveConnections");
  return qdbus_cast<QVector<QDBusObjectPath>>(result);
}

bool WifiManager::isKnownConnection(const QString &ssid) {
  return !getConnectionPath(ssid).path().isEmpty();
}

void WifiManager::forgetConnection(const QString &ssid) {
  const QDBusObjectPath &path = getConnectionPath(ssid);
  if (!path.path().isEmpty()) {
    call(path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "Delete");
  }
}

void WifiManager::setCurrentConnecting(const QString &ssid) {
  connecting_to_network = ssid;
  for (auto &network : seenNetworks) {
    network.connected = (network.ssid == ssid) ? ConnectedType::CONNECTING : ConnectedType::DISCONNECTED;
  }
  emit refreshSignal();
}

uint WifiManager::getAdapterType(const QDBusObjectPath &path) {
  return call<uint>(path.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_DEVICE, "DeviceType");
}

void WifiManager::requestScan() {
  if (!adapter.isEmpty()) {
    asyncCall(adapter, NM_DBUS_INTERFACE_DEVICE_WIRELESS, "RequestScan", QVariantMap());
  }
}

QByteArray WifiManager::get_property(const QString &network_path , const QString &property) {
  return call<QByteArray>(network_path, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACCESS_POINT, property);
}

QString WifiManager::getAdapter(const uint adapter_type) {
  QDBusReply<QList<QDBusObjectPath>> response = call(NM_DBUS_PATH, NM_DBUS_INTERFACE, "GetDevices");
  for (const QDBusObjectPath &path : response.value()) {
    if (getAdapterType(path) == adapter_type) {
      return path.path();
    }
  }
  return "";
}

void WifiManager::stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason) {
  raw_adapter_state = new_state;
  if (new_state == NM_DEVICE_STATE_NEED_AUTH && change_reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT && !connecting_to_network.isEmpty()) {
    forgetConnection(connecting_to_network);
    emit wrongPassword(connecting_to_network);
  } else if (new_state == NM_DEVICE_STATE_ACTIVATED) {
    connecting_to_network = "";
    refreshNetworks();
  }
}

// https://developer.gnome.org/NetworkManager/stable/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html
void WifiManager::propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props) {
  if (interface == NM_DBUS_INTERFACE_DEVICE_WIRELESS && props.contains("LastScan")) {
    refreshNetworks();
  } else if (interface == NM_DBUS_INTERFACE_DEVICE_WIRELESS && props.contains("ActiveAccessPoint")) {
    activeAp = props.value("ActiveAccessPoint").value<QDBusObjectPath>().path();
  }
}

void WifiManager::deviceAdded(const QDBusObjectPath &path) {
  if (getAdapterType(path) == NM_DEVICE_TYPE_WIFI && emptyPath(adapter)) {
    adapter = path.path();
    setup();
  }
}

void WifiManager::connectionRemoved(const QDBusObjectPath &path) {
  knownConnections.remove(path);
}

void WifiManager::newConnection(const QDBusObjectPath &path) {
  Connection settings = getConnectionSettings(path);
  if (settings.value("connection").value("type") == "802-11-wireless") {
    knownConnections[path] = settings.value("802-11-wireless").value("ssid").toString();
    if (knownConnections[path] != tethering_ssid) {
      activateWifiConnection(knownConnections[path]);
    }
  }
}

QDBusObjectPath WifiManager::getConnectionPath(const QString &ssid) {
  return knownConnections.key(ssid);
}

Connection WifiManager::getConnectionSettings(const QDBusObjectPath &path) {
  return QDBusReply<Connection>(call(path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "GetSettings")).value();
}

void WifiManager::initConnections() {
  const QDBusReply<QList<QDBusObjectPath>> response = call(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "ListConnections");
  for (const QDBusObjectPath &path : response.value()) {
    const Connection settings = getConnectionSettings(path);
    if (settings.value("connection").value("type") == "802-11-wireless") {
      knownConnections[path] = settings.value("802-11-wireless").value("ssid").toString();
    } else if (settings.value("connection").value("id") == "lte") {
      lteConnectionPath = path;
    }
  }

  if (!isKnownConnection(tethering_ssid)) {
    addTetheringConnection();
  }
}

std::optional<QDBusPendingCall> WifiManager::activateWifiConnection(const QString &ssid) {
  const QDBusObjectPath &path = getConnectionPath(ssid);
  if (!path.path().isEmpty()) {
    setCurrentConnecting(ssid);
    return asyncCall(NM_DBUS_PATH, NM_DBUS_INTERFACE, "ActivateConnection", QVariant::fromValue(path), QVariant::fromValue(QDBusObjectPath(adapter)), QVariant::fromValue(QDBusObjectPath("/")));
  }
  return std::nullopt;
}

void WifiManager::activateModemConnection(const QDBusObjectPath &path) {
  QString modem = getAdapter(NM_DEVICE_TYPE_MODEM);
  if (!path.path().isEmpty() && !modem.isEmpty()) {
    asyncCall(NM_DBUS_PATH, NM_DBUS_INTERFACE, "ActivateConnection", QVariant::fromValue(path), QVariant::fromValue(QDBusObjectPath(modem)), QVariant::fromValue(QDBusObjectPath("/")));
  }
}

// function matches tici/hardware.py
// FIXME: it can mistakenly show CELL when connected to WIFI
NetworkType WifiManager::currentNetworkType() {
  auto primary_conn = call<QDBusObjectPath>(NM_DBUS_PATH, NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE, "PrimaryConnection");
  auto primary_type = call<QString>(primary_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");

  if (primary_type == "802-3-ethernet") {
    return NetworkType::ETHERNET;
  } else if (primary_type == "802-11-wireless" && !isTetheringEnabled()) {
    return NetworkType::WIFI;
  } else {
    for (const QDBusObjectPath &conn : getActiveConnections()) {
      auto type = call<QString>(conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");
      if (type == "gsm") {
        return NetworkType::CELL;
      }
    }
  }
  return NetworkType::NONE;
}

MeteredType WifiManager::currentNetworkMetered() {
  MeteredType metered = MeteredType::UNKNOWN;
  for (const auto &active_conn : getActiveConnections()) {
    QString type = call<QString>(active_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");
    if (type == "802-11-wireless") {
      QDBusObjectPath conn = call<QDBusObjectPath>(active_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Connection");
      if (!conn.path().isEmpty()) {
        Connection settings = getConnectionSettings(conn);
        int metered_prop = settings.value("connection").value("metered").toInt();
        if (metered_prop == NM_METERED_YES) {
          metered = MeteredType::YES;
        } else if (metered_prop == NM_METERED_NO) {
          metered = MeteredType::NO;
        }
      }
      break;
    }
  }
  return metered;
}

std::optional<QDBusPendingCall> WifiManager::setCurrentNetworkMetered(MeteredType metered) {
  for (const auto &active_conn : getActiveConnections()) {
    QString type = call<QString>(active_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Type");
    if (type == "802-11-wireless") {
      if (!isTetheringEnabled()) {
        QDBusObjectPath conn = call<QDBusObjectPath>(active_conn.path(), NM_DBUS_INTERFACE_PROPERTIES, "Get", NM_DBUS_INTERFACE_ACTIVE_CONNECTION, "Connection");
        if (!conn.path().isEmpty()) {
          Connection settings = getConnectionSettings(conn);
          settings["connection"]["metered"] = static_cast<int>(metered);
          return asyncCall(conn.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "Update", QVariant::fromValue(settings));
        }
      }
    }
  }
  return std::nullopt;
}

void WifiManager::updateGsmSettings(bool roaming, QString apn, bool metered) {
  if (!lteConnectionPath.path().isEmpty()) {
    bool changes = false;
    bool auto_config = apn.isEmpty();
    Connection settings = getConnectionSettings(lteConnectionPath);
    if (settings.value("gsm").value("auto-config").toBool() != auto_config) {
      qWarning() << "Changing gsm.auto-config to" << auto_config;
      settings["gsm"]["auto-config"] = auto_config;
      changes = true;
    }

    if (settings.value("gsm").value("apn").toString() != apn) {
      qWarning() << "Changing gsm.apn to" << apn;
      settings["gsm"]["apn"] = apn;
      changes = true;
    }

    if (settings.value("gsm").value("home-only").toBool() == roaming) {
      qWarning() << "Changing gsm.home-only to" << !roaming;
      settings["gsm"]["home-only"] = !roaming;
      changes = true;
    }

    int meteredInt = metered ? NM_METERED_UNKNOWN : NM_METERED_NO;
    if (settings.value("connection").value("metered").toInt() != meteredInt) {
      qWarning() << "Changing connection.metered to" << meteredInt;
      settings["connection"]["metered"] = meteredInt;
      changes = true;
    }

    if (changes) {
      QDBusPendingCall pending_call = asyncCall(lteConnectionPath.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "UpdateUnsaved", QVariant::fromValue(settings));  // update is temporary
      QDBusPendingCallWatcher *watcher = new QDBusPendingCallWatcher(pending_call);
      QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, watcher]() {
        deactivateConnection(lteConnectionPath);
        activateModemConnection(lteConnectionPath);
        watcher->deleteLater();
      });
    }
  }
}

// Functions for tethering
void WifiManager::addTetheringConnection() {
  Connection connection;
  connection["connection"]["id"] = "Hotspot";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["interface-name"] = "wlan0";
  connection["connection"]["autoconnect"] = false;

  connection["802-11-wireless"]["band"] = "bg";
  connection["802-11-wireless"]["mode"] = "ap";
  connection["802-11-wireless"]["ssid"] = tethering_ssid.toUtf8();

  connection["802-11-wireless-security"]["group"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["pairwise"] = QStringList("ccmp");
  connection["802-11-wireless-security"]["proto"] = QStringList("rsn");
  connection["802-11-wireless-security"]["psk"] = defaultTetheringPassword;

  connection["ipv4"]["method"] = "shared";
  QVariantMap address;
  address["address"] = "192.168.43.1";
  address["prefix"] = 24u;
  connection["ipv4"]["address-data"] = QVariant::fromValue(IpConfig() << address);
  connection["ipv4"]["gateway"] = "192.168.43.1";
  connection["ipv4"]["never-default"] = true;
  connection["ipv6"]["method"] = "ignore";

  asyncCall(NM_DBUS_PATH_SETTINGS, NM_DBUS_INTERFACE_SETTINGS, "AddConnection", QVariant::fromValue(connection));
}

void WifiManager::tetheringActivated(QDBusPendingCallWatcher *call) {
  if (!ipv4_forward) {
    QTimer::singleShot(5000, this, [=] {
      qWarning() << "net.ipv4.ip_forward = 0";
      std::system("sudo sysctl net.ipv4.ip_forward=0");
    });
  }
  call->deleteLater();
  tethering_on = true;
}

void WifiManager::setTetheringEnabled(bool enabled) {
  if (enabled) {
    auto pending_call = activateWifiConnection(tethering_ssid);

    if (pending_call) {
      QDBusPendingCallWatcher *watcher = new QDBusPendingCallWatcher(*pending_call);
      QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, &WifiManager::tetheringActivated);
    }

  } else {
    deactivateConnectionBySsid(tethering_ssid);
    tethering_on = false;
  }
}

bool WifiManager::isTetheringEnabled() {
  if (!emptyPath(activeAp)) {
    return get_property(activeAp, "Ssid") == tethering_ssid;
  }
  return false;
}

QString WifiManager::getTetheringPassword() {
  const QDBusObjectPath &path = getConnectionPath(tethering_ssid);
  if (!path.path().isEmpty()) {
    QDBusReply<QMap<QString, QVariantMap>> response = call(path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "GetSecrets", "802-11-wireless-security");
    return response.value().value("802-11-wireless-security").value("psk").toString();
  }
  return "";
}

void WifiManager::changeTetheringPassword(const QString &newPassword) {
  const QDBusObjectPath &path = getConnectionPath(tethering_ssid);
  if (!path.path().isEmpty()) {
    Connection settings = getConnectionSettings(path);
    settings["802-11-wireless-security"]["psk"] = newPassword;
    call(path.path(), NM_DBUS_INTERFACE_SETTINGS_CONNECTION, "Update", QVariant::fromValue(settings));
    if (isTetheringEnabled()) {
      activateWifiConnection(tethering_ssid);
    }
  }
}
