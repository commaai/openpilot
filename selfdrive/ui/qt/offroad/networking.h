#pragma once

#include <QButtonGroup>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/offroad/wifiManager.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

class NetworkStrengthWidget : public QWidget {
  Q_OBJECT

public:
  explicit NetworkStrengthWidget(int strength, QWidget* parent = nullptr) : strength_(strength), QWidget(parent) { setFixedSize(100, 15); }

private:
  void paintEvent(QPaintEvent* event) override;
  int strength_ = 0;
};

class WifiUI : public QWidget {
  Q_OBJECT

public:
  explicit WifiUI(QWidget *parent = 0, WifiManager* wifi = 0);
  void refresh(QVector<Network> _seen_networks);

private:
  WifiManager *wifi = nullptr;
  QVBoxLayout* main_layout;

  QButtonGroup *connectButtons;
  bool tetheringEnabled;

  QVector<Network> seen_networks;

signals:
  void connectToNetwork(const Network n, const QString pass);

public slots:
  void handleButton(QAbstractButton* m_button);
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0, WifiManager* wifi = 0);
  void refresh(const QString ipv4_address);
  ToggleControl *tetheringToggle;

private:
  LabelControl* ipLabel;
  ButtonControl* editPasswordButton;
  WifiManager* wifi = nullptr;

signals:
  void backPress();
  void enableTethering();
  void disableTethering();

public slots:
  void toggleTethering(bool enable);
  void tetheringStateChange();
};

class Networking : public QWidget {
  Q_OBJECT
  QThread wifiThread;

public:
  explicit Networking(QWidget* parent = 0, bool show_advanced = true);

private:
//    QThread *wifi_thread;
  QStackedLayout* main_layout = nullptr; // nm_warning, wifiScreen, advanced
  QWidget* wifiScreen = nullptr;
  AdvancedNetworking* an = nullptr;
  bool ui_setup_complete = false;
  bool show_advanced;


  Network selectedNetwork;

  WifiUI* wifiWidget;
  WifiManager* wifiManager = nullptr;
  void attemptInitialization();

signals:
  void refreshNetworks();
  void refreshWifiManager();

public slots:
  void refresh(const QVector<Network> seen_networks, const QString ipv4_address);

private slots:
//  void connectToNetwork(const Network &n);
  void wrongPassword(const QString &ssid);

};


//class WifiWorker : public QObject
//{
//    Q_OBJECT
//    QThread wifiThread;
//
//public:
//  WifiManager* wifi = nullptr;
//  WifiWorker(WifiManager* _wifi) {
//    wifi = _wifi;
//  }
//
//public slots:
//    void run();
////    void connectToNetwork(const Network &n);
//
//signals:
//    void update();
//};


//class Controller : public QObject
//{
//    Q_OBJECT
//    QThread workerThread;
//public:
//    Controller() {
//        Worker *worker = new Worker;
//        worker->moveToThread(&workerThread);
//        connect(&workerThread, SIGNAL(finished()), worker, SLOT(deleteLater()));
//        connect(this, SIGNAL(operate(QString)), worker, SLOT(doWork(QString)));
//        connect(worker, SIGNAL(resultReady(QString)), this, SLOT(handleResults(QString)));
//        workerThread.start();
//    }
//    ~Controller() {
//        workerThread.quit();
//        workerThread.wait();
//    }
//public slots:
//    void handleResults(const QString &);
//signals:
//    void operate(const QString &);
//};
