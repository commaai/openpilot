#pragma once
#include <QWidget>

struct Network {
  QString path;
  QByteArray ssid;
  unsigned int strength;
  bool connected;
};

class WifiManager{
  private:
    QVector<QByteArray> seen_ssids;

    QString get_adapter();
    QList<Network> get_networks(QString adapter);
    void connect_to_open(QByteArray ssid);
    void connect_to_WPA(QByteArray ssid, QString password);
    void request_scan(QString adapter);
    QString get_active_ap(QString adapter);
    QByteArray get_ap_ssid(QString network_path);
    QByteArray get_property(QString network_path, QString property);
    unsigned int get_ap_strength(QString network_path);

  public:
    QVector<Network> seen_networks;

    explicit WifiManager();
    void refreshNetworks();
    
};
