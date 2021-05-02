#pragma once

#include <QtWidgets>

#include "ui/ui.h"


// ***** onroad widgets *****

class OnroadAlerts : public QFrame {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor bg;
  QLabel *title, *msg;
  QVBoxLayout *layout;

public slots:
  void update(const UIState &s);
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0) : QOpenGLWidget(parent) {};
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;

private:
  double prev_draw_t = 0;

public slots:
  void update(const UIState &s);
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);

private:
  OnroadAlerts *alerts;
  NvgWindow *nvg;
  QStackedLayout *layout;

signals:
  void update(const UIState &s);
};
