#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QTimer>

#include "ui/ui.hpp"

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
  bool enabled;
  double prev_draw_t = 0;

public slots:
  void setEnabled(bool on);
  void update(const UIState &s);
};
