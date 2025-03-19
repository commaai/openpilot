#pragma once

#include <vector>
#include <QPainter>
#include <QWidget>

#include "selfdrive/ui/ui.h"

class DriverMonitorRenderer : public QWidget {
  Q_OBJECT

public:
  explicit DriverMonitorRenderer(QWidget *parent = nullptr);
  void updateState(const UIState &s);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  void draw(QPainter &painter, const QRect &surface_rect);
  float driver_pose_vals[3] = {};
  float driver_pose_diff[3] = {};
  float driver_pose_sins[3] = {};
  float driver_pose_coss[3] = {};
  bool is_visible = false;
  bool is_active = false;
  bool is_rhd = false;
  float dm_fade_state = 1.0;
  QPixmap dm_img;
  std::vector<vec3> face_kpts_draw;
};
