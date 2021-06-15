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

private:
  WifiManager *wifi = nullptr;
  QVBoxLayout* main_layout;

  QButtonGroup *connectButtons;
  bool tetheringEnabled;

signals:
  void connectToNetwork(const Network &n);

public slots:
  void refresh();
  void handleButton(QAbstractButton* m_button);
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0, WifiManager* wifi = 0);

private:
  LabelControl* ipLabel;
  ButtonControl* editPasswordButton;
  WifiManager* wifi = nullptr;

signals:
  void backPress();

public slots:
  void toggleTethering(bool enable);
  void refresh();
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
  WifiManager* wifi = nullptr;
  void attemptInitialization();

public slots:
  void update();

private slots:
  void connectToNetwork(const Network &n);
  void refreshed();
  void wrongPassword(const QString &ssid);

signals:
  void refreshNetworks();
  void startWifiThread();

};


class WifiWorker : public QObject
{
    Q_OBJECT
    QThread wifiThread;

public:
  WifiManager* wifi = nullptr;
  WifiWorker(WifiManager* _wifi) {
    wifi = _wifi;
  }

public slots:
    void run();

signals:
    void update();
};


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
