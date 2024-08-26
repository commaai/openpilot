#pragma once

#include <QPainter>
#include <QPolygonF>

#include "selfdrive/ui/ui.h"

class ModelRenderer {
public:
  ModelRenderer() {}
  void setTransform(const Eigen::Matrix3f &transform) { car_space_transform = transform; }
  void updateState(const UIState &s) {}
  void draw(QPainter &painter, const QRect &surface_rect);

private:
  QPointF mapToScreen(float in_x, float in_y, float in_z);
  void mapLineToPolygon(const cereal::XYZTData::Reader &line,
                      float y_off, float z_off, QPolygonF *pvd, int max_idx, bool allow_invert = true);
  void drawLaneLines(QPainter &painter, const cereal::ModelDataV2::Reader &model, float max_dist);
  void drawPath(QPainter &painter, const cereal::ModelDataV2::Reader &model, float max_dist, int surface_height);
  void drawLead(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data,
                const cereal::XYZTData::Reader &model_position, const QRect &surface_rect);
  uint8_t calculateLeadAlpha(float d_rel, float v_rel);

  bool longitudinal_control = false;
  bool experimental_model = false;
  Eigen::Matrix3f car_space_transform = Eigen::Matrix3f::Zero();
  QRectF clip_region;
};
