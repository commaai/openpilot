#pragma once

#include <QLabel>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>
#include <QWidget>

#include "sound.hpp"
#include "ui/ui.hpp"
#include "common/util.h"
#include "widgets/offroad_alerts.hpp"

// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget* parent = 0);
  ~GLWindow();

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  double prev_draw_t = 0;

public slots:
  void update(const UIState &s);
};

// offroad home screen
class OffroadHome : public QWidget {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

private:
  QTimer* timer;

  QLabel* date;
  QStackedLayout* center_layout;
  OffroadAlert* alerts_widget;
  QPushButton* alert_notification;

public slots:
  void closeAlerts();
  void openAlerts();
  void refresh();
};

class HomeWindow : public QWidget {
  Q_OBJECT

public:
  explicit HomeWindow(QWidget* parent = 0);
  GLWindow* glWindow;

signals:
  void openSettings();
  void closeSettings();
  void offroadTransition(bool offroad);

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  OffroadHome* home;
  QStackedLayout* layout;
};
