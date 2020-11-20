#pragma once
#include "wifiManager.hpp"
#include "input_field.hpp"
#include <QWidget>
#include <QtDBus>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>


class CustomConnectButton : public QPushButton{

public:
  explicit CustomConnectButton(QString text, int iid);
  int id;
};

class WifiUI : public QWidget {
  Q_OBJECT

private:
  WifiManager* wifi;

  QStackedWidget* swidget;
  QVBoxLayout* vlayout;
  QWidget * wifi_widget;

  InputField *a;
  QEventLoop loop;
  QTimer * timer;
  QString text;
  QString getStringFromUser();

public:
  explicit WifiUI(QWidget *parent = 0);

private slots:
  void handleButton(QAbstractButton* m_button);
  void refresh();
  void receiveText(QString text);
signals:
  void gotText();
};
