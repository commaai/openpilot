#pragma once


#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>
#include <QStackedLayout>
#include <QPushButton>

#include "wifi.hpp"
#include "widgets/offroad_alerts.hpp"

class ParamsToggle : public QFrame {
  Q_OBJECT

public:
  explicit ParamsToggle(QString param, QString title, QString description, QString icon, QWidget *parent = 0);

private:
  QCheckBox *checkbox;
  QString param;

public slots:
  void checkboxClicked(int state);
};

class SettingsWindow : public QWidget {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);
  void refreshParams();

signals:
  void closeSettings();

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  OffroadAlert *alerts_widget;
  std::map<QString, QWidget *> panels;
  QStackedLayout *panel_layout;


public slots:
  void setActivePanel();
  void closeAlerts();
  void openAlerts();
  void closeSidebar();
  void openSidebar();
};
