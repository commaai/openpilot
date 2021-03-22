#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QStackedLayout>

#include "selfdrive/ui/qt/widgets/toggle.hpp"

// *** settings widgets ***

class ParamsToggle : public QFrame {
  Q_OBJECT

public:
  explicit ParamsToggle(QString param, QString title, QString description,
                        QString icon, QWidget *parent = 0);
  Toggle *toggle;

private:
  QString param;

public slots:
  void checkboxClicked(int state);
};

class DeveloperPanel : public QFrame {
  Q_OBJECT
public:
  explicit DeveloperPanel(QWidget* parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;
  QList<QLabel *> labels;
};

// *** settings window ***

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
