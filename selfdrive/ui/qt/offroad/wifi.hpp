#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QTimer>

#include "wifiManager.hpp"
#include "widgets/input_field.hpp"


class WifiUI : public QWidget {
  Q_OBJECT

public:
  int page;
  explicit WifiUI(QWidget *parent = 0, int page_length = 5);

private:
  WifiManager *wifi = nullptr;
  const int networks_per_page;

  QStackedWidget *swidget;
  QVBoxLayout *vlayout;
  QWidget *wifi_widget;

  InputField *input_field;
  QEventLoop loop;
  QTimer *timer;
  QString text;
  QButtonGroup *connectButtons;
  bool tetheringEnabled;
  QLabel *ipv4;

  void connectToNetwork(Network n);
  QString getStringFromUser();

private slots:
  void handleButton(QAbstractButton* m_button);
  void toggleTethering(int enable);
  void refresh();
  void receiveText(QString text);
  void wrongPassword(QString ssid);

  void prevPage();
  void nextPage();

signals:
  void openKeyboard();
  void closeKeyboard();
};
