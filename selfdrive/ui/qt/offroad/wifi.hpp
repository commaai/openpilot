#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QTimer>

#include "wifiManager.hpp"
#include "input_field.hpp"


class WifiUI : public QWidget {
  Q_OBJECT

private:
  WifiManager* wifi;
  int page;
  const int networks_per_page = 10;

  QStackedWidget* swidget;
  QVBoxLayout* vlayout;
  QWidget * wifi_widget;

  InputField *a;
  QEventLoop loop;
  QTimer * timer;
  QString text;
  QButtonGroup *connectButtons;

  QString getStringFromUser();

public:
  explicit WifiUI(QWidget *parent = 0);

private slots:
  void handleButton(QAbstractButton* m_button);
  void refresh();
  void receiveText(QString text);
  void prevPage();
  void nextPage();
};
