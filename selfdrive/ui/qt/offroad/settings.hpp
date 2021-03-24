#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QScrollArea>
#include <QStackedWidget>

#include "selfdrive/ui/qt/widgets/controls.hpp"

// ********** settings window + top-level panels **********

class DevicePanel : public QWidget {
  Q_OBJECT
public:
  explicit DevicePanel(QWidget* parent = nullptr);
};

class DeveloperPanel : public QFrame {
  Q_OBJECT
public:
  explicit DeveloperPanel(QWidget* parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;
  QList<LabelControl *> labels;
};

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
  QStackedWidget *panel_widget;
  QScrollArea *panel_frame;
};
