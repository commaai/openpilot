#pragma once
#include <QWidget>
#include <QtDBus>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedLayout>

struct Network {
  QString path;
  QByteArray ssid;
  unsigned int strength;
  bool connected;
};


class WifiSettingsModel{
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

    explicit WifiSettingsModel();
    void refreshNetworks();
};


class WifiSettings : public QWidget {
  Q_OBJECT

  private:
    WifiSettingsModel* wifi;
    QVBoxLayout* vlayout;

  public:
    explicit WifiSettings(QWidget *parent = 0);

  private slots:
    void handleButton(QAbstractButton* m_button);
    void refresh();
};
