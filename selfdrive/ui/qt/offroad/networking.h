#pragma once

#include <QButtonGroup>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/offroad/wifiManager.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

class WifiUI;

class WifiItem : public QWidget {
  Q_OBJECT
public:
  explicit WifiItem(QWidget* parent = nullptr);
  void update(WifiUI *wifi_ui, const Network& n);

signals:
  void connectToNetwork(const Network &n);
  void forgotNetwork(const Network &n);

protected:
  ElidedLabel* ssidLabel;
  QPushButton* connecting;
  QPushButton* forgetBtn;
  QLabel* iconLabel;
  QLabel* strengthLabel;
  Network network;
};

class WifiUI : public QWidget {
  Q_OBJECT

public:
  explicit WifiUI(QWidget *parent = 0, WifiManager* wifi = 0);

private:
  QWidget *list_container = nullptr;
  ListWidget *wifi_list = nullptr;
  std::vector<WifiItem*> wifi_items;
  WifiManager *wifi = nullptr;
  QLabel *scanning = nullptr;
  QVBoxLayout* main_layout;
  QPixmap lock;
  QPixmap checkmark;
  QPixmap circled_slash;
  QVector<QPixmap> strengths;

signals:
  void connectToNetwork(const Network &n);

public slots:
  void refresh();

  friend class WifiItem;
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0, WifiManager* wifi = 0);

private:
  void showEvent(QShowEvent* event) override {
    ipLabel->setText(wifi->getIP4Address());
  }
  LabelControl* ipLabel;
  ToggleControl* tetheringToggle;
  WifiManager* wifi = nullptr;
  Params params;

signals:
  void backPress();

public slots:
  void toggleTethering(bool enabled);
  void refresh();
};

class Networking : public QFrame {
  Q_OBJECT

public:
  explicit Networking(QWidget* parent = 0, bool show_advanced = true);
  WifiManager* wifi = nullptr;

private:
  QStackedLayout* main_layout = nullptr;
  QWidget* wifiScreen = nullptr;
  AdvancedNetworking* an = nullptr;

  WifiUI* wifiWidget;

protected:
  void showEvent(QShowEvent* event) override;
  void hideEvent(QHideEvent* event) override;

public slots:
  void refresh();

private slots:
  void connectToNetwork(const Network &n);
  void wrongPassword(const QString &ssid);
};
