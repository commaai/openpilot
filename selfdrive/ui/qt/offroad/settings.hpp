#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QPushButton>
#include <QButtonGroup>
#include <QStackedLayout>
#include <QLabel>

#include "selfdrive/ui/qt/widgets/toggle.hpp"

// *** settings widgets ***
class ParamsToggle : public QFrame {
  Q_OBJECT

public:
  explicit ParamsToggle(QString param, QString title, QString description,
                        QString icon, QWidget *parent = 0);
protected:
  void mousePressEvent(QMouseEvent *event) override {
    pressed = true;
  }
  void mouseMoveEvent(QMouseEvent *event) override {
    if (pressed) dragging = true;
  }
  void mouseReleaseEvent(QMouseEvent *event) override {
    if (!dragging) {
      desc_label->setVisible(!desc_label->isVisible());
    }
    pressed = dragging = false;
  }

  QString param;
  QLabel *desc_label;
  bool pressed = false, dragging = false;

public slots:
  void checkboxClicked(int state);
};

// *** settings window ***

class SettingsWindow : public QFrame {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

signals:
  void closeSettings();
  void sidebarPressed();

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  std::map<QString, QWidget *> panels;
  QButtonGroup *nav_btns;
  QStackedLayout *panel_layout;
  QFrame* panel_frame;

public slots:
  void setActivePanel();
};
