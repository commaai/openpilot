#pragma once

#include <memory>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "selfdrive/ui/ui.h"

class DriverViewWindow : public QOpenGLWidget, protected QOpenGLFunctions {
Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit DriverViewWindow(QWidget* parent = 0);
  ~DriverViewWindow();

protected:
  void showEvent(QShowEvent *event);
  void hideEvent(QHideEvent *event);
  void paintGL() override;
  void initializeGL() override;

protected slots:
  void onTimeout();

private:
  SubMaster sm;
  std::unique_ptr<UIVision> vision;
  QTimer* timer;
  QImage face_img;
  bool is_rhd;
};