#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QPushButton>
#include <QButtonGroup>
#include <QStackedLayout>

class SettingsWindow : public QFrame {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

signals:
  void closeSettings();

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedLayout *panel_layout;
  QFrame* panel_frame;
};
