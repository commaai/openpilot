#include <QtDBus>
#include <QDebug>

typedef QMap<QString, QMap<QString, QVariant> > Connection;
Q_DECLARE_METATYPE(Connection)

void wifi_stuff(){
  qDBusRegisterMetaType<Connection>();

  QString nm_path = "/org/freedesktop/NetworkManager";
  QString nm_settings_path = "/org/freedesktop/NetworkManager/Settings";

  QString nm_iface = "org.freedesktop.NetworkManager";
  QString props_iface = "org.freedesktop.DBus.Properties";
  QString nm_settings_iface = "org.freedesktop.NetworkManager.Settings";

  QString nm_service = "org.freedesktop.NetworkManager";
  QString device_service = "org.freedesktop.NetworkManager.Device";

  QDBusConnection bus = QDBusConnection::systemBus();

  // Get devices
  QDBusInterface nm(nm_service, nm_path, nm_iface, bus);
  QDBusMessage response = nm.call("GetDevices");
  QVariant first =  response.arguments().at(0);

  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    // Get device type
    QDBusInterface device_props(nm_service, path.path(), props_iface, bus);
    QDBusMessage response = device_props.call("Get", device_service, "DeviceType");
    QVariant first =  response.arguments().at(0);
    QDBusVariant dbvFirst = first.value<QDBusVariant>();
    QVariant vFirst = dbvFirst.variant();
    uint device_type = vFirst.value<uint>();
    qDebug() << path.path() << device_type;
  }
  args.endArray();


  // Add connection
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "Connection 1";

  connection["802-11-wireless"]["ssid"] = QByteArray("<ssid>");
  connection["802-11-wireless"]["mode"] = "infrastructure";

  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["auth-alg"] = "open";
  connection["802-11-wireless-security"]["psk"] = "<password>";

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";


  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusReply<QDBusObjectPath> result = nm_settings.call("AddConnection", QVariant::fromValue(connection));
  if (!result.isValid()) {
    qDebug() << result.error().name() << result.error().message();
  } else {
    qDebug() << result.value().path();
  }
}
