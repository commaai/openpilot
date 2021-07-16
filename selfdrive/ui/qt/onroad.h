#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QStackedLayout>
#include <QWidget>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/ui.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

// ***** onroad widgets *****

class OnroadAlerts : public QWidget {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0) {};

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor bg;
  Alert alert;

  void updateAlert(Alert a);

public slots:
  void updateState(const UIState &s);
  void offroadTransition(bool offroad);
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0);
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;

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
  QWidget *map = nullptr;

private:
  OnroadAlerts *alerts;
  NvgWindow *nvg;
  QStackedLayout *main_layout;
  QHBoxLayout* split;

signals:
  void update(const UIState &s);
  void offroadTransitionSignal(bool offroad);

private slots:
  void offroadTransition(bool offroad);
};
