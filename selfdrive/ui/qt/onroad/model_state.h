#pragma once

#include <QPolygonF>
#include <QTransform>

#include "selfdrive/ui/ui.h"

class ModelState : public QObject {
  Q_OBJECT

public:
  ModelState(QObject *parent = nullptr) : QObject(parent) {}
  void setTransform(int w, int h, const QTransform &transform, const mat3 &calib, const mat3 &intrinsic);
  void draw(QPainter &p, const SubMaster &sm);

protected:
  void drawLaneLines(QPainter &painter, const SubMaster &sm);
  void drawLeadVehicles(QPainter &painter, const cereal::RadarState::Reader &radar_state);
  void drawLeadVehicleIcon(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data, const QPointF &vd);
  void updateModelData(const cereal::ModelDataV2::Reader &model, const cereal::RadarState::LeadData::Reader &lead_one);
  void updateLeadVehicles(const cereal::RadarState::Reader &radar_state, const cereal::XYZTData::Reader &line);
  bool mapToFrame(float in_x, float in_y, float in_z, QPointF *out);
  void updateLineData(const cereal::XYZTData::Reader &line, float y_off, float z_off, QPolygonF *pvd,
                      int max_idx, bool allow_invert = true);

  float lane_line_probs[4] = {};
  float road_edge_stds[2] = {};
  QPolygonF track_vertices;
  QPolygonF lane_line_vertices[4];
  QPolygonF road_edge_vertices[2];

  QPointF lead_vertices[2];

  int fb_w = 0, fb_h = 0;
  mat3 calibration = DEFAULT_CALIBRATION;
  mat3 intrinsic_matrix = FCAM_INTRINSIC_MATRIX;
  QTransform car_space_transform;
  const float clip_margin = 500.0f;
  QRectF clip_region;
  bool longitudinal_control = false;
};
