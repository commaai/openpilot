#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPushButton>

#include "wifiManager.hpp"
#include "widgets/input.hpp"
#include "widgets/ssh_keys.hpp"
#include "widgets/toggle.hpp"

class WifiUI : public QWidget {
  Q_OBJECT

public:
  int page;
  explicit WifiUI(QWidget *parent = 0, WifiManager* wifi = 0);

private:
  WifiManager *wifi = nullptr;
  QVBoxLayout *vlayout;

  QButtonGroup *connectButtons;
  bool tetheringEnabled;

signals:
  void connectToNetwork(Network n);

public slots:
  void handleButton(QAbstractButton* m_button);
  void refresh();
  void prevPage();
  void nextPage();
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0, WifiManager* wifi = 0);

private:
  QLabel* ipLabel;
  QPushButton* editPasswordButton;
  WifiManager* wifi = nullptr;

signals:
  void backPress();

public slots:
  void toggleTethering(bool enable);
  void refresh();
};

class Networking : public QWidget {
  Q_OBJECT

public:
  explicit Networking(QWidget* parent = 0, bool show_advanced = true);

private:
  QStackedLayout* s = nullptr; // nm_warning, keyboard, wifiScreen, advanced
  QWidget* wifiScreen = nullptr;
  AdvancedNetworking* an = nullptr;
  bool ui_setup_complete = false;
  bool show_advanced;

  Network selectedNetwork;

  WifiUI* wifiWidget;
  WifiManager* wifi = nullptr;
  void attemptInitialization();

private slots:
  void connectToNetwork(Network n);
  void refresh();
  void wrongPassword(QString ssid);
};

