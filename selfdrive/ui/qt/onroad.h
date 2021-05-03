#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QTimer>

#include "ui/ui.h"

// container window for the NVG UI
class OnroadWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit OnroadWindow(QWidget* parent = 0) : QOpenGLWidget(parent) {};
  ~OnroadWindow();

protected:
  void paintGL() override;
  void initializeGL() override;

private:
  double prev_draw_t = 0;

public slots:
  void update(const UIState &s);
};
