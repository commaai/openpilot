#pragma once

#include <vector>

#include "selfdrive/ui/qt/network/wifi_manager.h"
#include "selfdrive/ui/qt/prime_state.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

class WifiItem : public QWidget {
  Q_OBJECT
public:
  explicit WifiItem(const QString &connecting_text, const QString &forget_text, QWidget* parent = nullptr);
  void setItem(const Network& n, const QPixmap &icon, bool show_forget_btn, const QPixmap &strength);

signals:
  // Cannot pass Network by reference. it may change after the signal is sent.
  void connectToNetwork(const Network n);
  void forgotNetwork(const Network n);

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
  WifiItem *getItem(int n);

  WifiManager *wifi = nullptr;
  QLabel *scanningLabel = nullptr;
  QPixmap lock;
  QPixmap checkmark;
  QPixmap circled_slash;
  QVector<QPixmap> strengths;
  ListWidget *wifi_list_widget = nullptr;
  std::vector<WifiItem*> wifi_items;

signals:
  void connectToNetwork(const Network n);

public slots:
  void refresh();
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0, WifiManager* wifi = 0);
  void setGsmVisible(bool visible);

private:
  LabelControl* ipLabel;
  ToggleControl* tetheringToggle;
  ToggleControl* roamingToggle;
  ButtonControl* editApnButton;
  ButtonControl* hiddenNetworkButton;
  ToggleControl* cellularMeteredToggle;
  MultiButtonControl* wifiMeteredToggle;
  WifiManager* wifi = nullptr;
  Params params;

signals:
  void backPress();
  void requestWifiScreen();

public slots:
  void toggleTethering(bool enabled);
  void refresh();
};

class Networking : public QFrame {
  Q_OBJECT

public:
  explicit Networking(QWidget* parent = 0, bool show_advanced = true);
  void setPrimeType(PrimeState::Type type);
  WifiManager* wifi = nullptr;

private:
  QStackedLayout* main_layout = nullptr;
  QWidget* wifiScreen = nullptr;
  AdvancedNetworking* an = nullptr;
  WifiUI* wifiWidget;

  void showEvent(QShowEvent* event) override;
  void hideEvent(QHideEvent* event) override;

public slots:
  void refresh();

private slots:
  void connectToNetwork(const Network n);
  void wrongPassword(const QString &ssid);
};
