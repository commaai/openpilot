#include "selfdrive/ui/qt/onroad/model_state.h"

#include <QPainter>
#include <algorithm>
#include <cmath>

const float MIN_DRAW_DISTANCE = 10.0;
const float MAX_DRAW_DISTANCE = 100.0;

inline QColor redColor(int alpha = 255) { return QColor(201, 34, 49, alpha); }

static int getPathLengthIndex(const cereal::XYZTData::Reader &line, const float path_height) {
  const auto line_x = line.getX();
  int max_idx = 0;
  for (int i = 1; i < line_x.size() && line_x[i] <= path_height; ++i) {
    max_idx = i;
  }
  return max_idx;
}

void ModelState::draw(QPainter &painter, const SubMaster &sm) {
  painter.save();

  auto model = sm["modelV2"].getModelV2();
  auto radar_state = sm["radarState"].getRadarState();

  updateModelData(model, radar_state.getLeadOne());
  drawLaneLines(painter, sm);

  if (sm.updated("carParams")) {
    longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }

  if (longitudinal_control) {
    updateLeadVehicles(radar_state, model.getPosition());
    drawLeadVehicles(painter, radar_state);
  }

  painter.restore();
}

void ModelState::updateModelData(const cereal::ModelDataV2::Reader &model, const cereal::RadarState::LeadData::Reader &lead_one) {
  auto model_position = model.getPosition();
  float max_distance = std::clamp(*(model_position.getX().end() - 1),
                                  MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE);

  // update lane lines
  const auto lines = model.getLaneLines();
  const auto probs = model.getLaneLineProbs();
  int max_idx = getPathLengthIndex(lines[0], max_distance);
  for (int i = 0; i < std::size(lane_line_vertices); i++) {
    lane_line_probs[i] = probs[i];
    updateLineData(lines[i], 0.025 * lane_line_probs[i], 0, &lane_line_vertices[i], max_idx);
  }

  // update road edges
  const auto edges = model.getRoadEdges();
  const auto stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(road_edge_vertices); i++) {
    road_edge_stds[i] = stds[i];
    updateLineData(edges[i], 0.025, 0, &road_edge_vertices[i], max_idx);
  }

  // update path
  if (lead_one.getStatus()) {
    const float lead_d = lead_one.getDRel() * 2.;
    max_distance = std::clamp((float)(lead_d - fmin(lead_d * 0.35, 10.)), 0.0f, max_distance);
  }
  max_idx = getPathLengthIndex(model_position, max_distance);
  updateLineData(model_position, 0.9, 1.22, &track_vertices, max_idx, false);
}

void ModelState::updateLeadVehicles(const cereal::RadarState::Reader &radar_state, const cereal::XYZTData::Reader &line) {
  for (int i = 0; i < 2; ++i) {
    auto lead_data = (i == 0) ? radar_state.getLeadOne() : radar_state.getLeadTwo();
    if (lead_data.getStatus()) {
      float z = line.getZ()[getPathLengthIndex(line, lead_data.getDRel())];
      mapToFrame(lead_data.getDRel(), -lead_data.getYRel(), z + 1.22, &lead_vertices[i]);
    }
  }
}

void ModelState::drawLaneLines(QPainter &painter, const SubMaster &sm) {
  // lanelines
  for (int i = 0; i < std::size(lane_line_vertices); ++i) {
    painter.setBrush(QColor::fromRgbF(1.0, 1.0, 1.0, std::clamp<float>(lane_line_probs[i], 0.0, 0.7)));
    painter.drawPolygon(lane_line_vertices[i]);
  }

  // road edges
  for (int i = 0; i < std::size(road_edge_vertices); ++i) {
    painter.setBrush(QColor::fromRgbF(1.0, 0, 0, std::clamp<float>(1.0 - road_edge_stds[i], 0.0, 1.0)));
    painter.drawPolygon(road_edge_vertices[i]);
  }

  // paint path
  QLinearGradient bg(0, fb_h, 0, 0);
  if (sm["controlsState"].getControlsState().getExperimentalMode()) {
    // The first half of track_vertices are the points for the right side of the path
    const auto &acceleration = sm["modelV2"].getModelV2().getAcceleration().getX();
    const int max_len = std::min<int>(track_vertices.length() / 2, acceleration.size());

    for (int i = 0; i < max_len; ++i) {
      // Some points are out of frame
      int track_idx = (track_vertices.length() / 2) - i;  // flip idx to start from top
      if (track_vertices[track_idx].y() < 0 || track_vertices[track_idx].y() > fb_h) continue;

      // Flip so 0 is bottom of frame
      float lin_grad_point = (fb_h - track_vertices[track_idx].y()) / fb_h;

      // speed up: 120, slow down: 0
      float path_hue = fmax(fmin(60 + acceleration[i] * 35, 120), 0);
      // FIXME: painter.drawPolygon can be slow if hue is not rounded
      path_hue = int(path_hue * 100 + 0.5) / 100;

      float saturation = fmin(fabs(acceleration[i] * 1.5), 1);
      float lightness = util::map_val(saturation, 0.0f, 1.0f, 0.95f, 0.62f);  // lighter when grey
      float alpha = util::map_val(lin_grad_point, 0.75f / 2.f, 0.75f, 0.4f, 0.0f);  // matches previous alpha fade
      bg.setColorAt(lin_grad_point, QColor::fromHslF(path_hue / 360., saturation, lightness, alpha));

      // Skip a point, unless next is last
      i += (i + 2) < max_len ? 1 : 0;
    }

  } else {
    bg.setColorAt(0.0, QColor::fromHslF(148 / 360., 0.94, 0.51, 0.4));
    bg.setColorAt(0.5, QColor::fromHslF(112 / 360., 1.0, 0.68, 0.35));
    bg.setColorAt(1.0, QColor::fromHslF(112 / 360., 1.0, 0.68, 0.0));
  }

  painter.setBrush(bg);
  painter.drawPolygon(track_vertices);
}

void ModelState::drawLeadVehicles(QPainter &painter, const cereal::RadarState::Reader &radar_state) {
  auto lead_one = radar_state.getLeadOne();
  auto lead_two = radar_state.getLeadTwo();
  if (lead_one.getStatus()) {
    drawLeadVehicleIcon(painter, lead_one, lead_vertices[0]);
  }
  if (lead_two.getStatus() && (std::abs(lead_one.getDRel() - lead_two.getDRel()) > 3.0)) {
    drawLeadVehicleIcon(painter, lead_two, lead_vertices[1]);
  }
}

void ModelState::drawLeadVehicleIcon(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data, const QPointF &vd) {
  const float speedBuff = 10.;
  const float leadBuff = 40.;
  const float d_rel = lead_data.getDRel();
  const float v_rel = lead_data.getVRel();

  float fillAlpha = 0;
  if (d_rel < leadBuff) {
    fillAlpha = 255 * (1.0 - (d_rel / leadBuff));
    if (v_rel < 0) {
      fillAlpha += 255 * (-1 * (v_rel / speedBuff));
    }
    fillAlpha = (int)(fmin(fillAlpha, 255));
  }

  float sz = std::clamp((25 * 30) / (d_rel / 3 + 30), 15.0f, 30.0f) * 2.35;
  float x = std::clamp((float)vd.x(), 0.f, fb_w - sz / 2);
  float y = std::fmin(fb_h - sz * .6, (float)vd.y());

  float g_xo = sz / 5;
  float g_yo = sz / 10;

  QPointF glow[] = {{x + (sz * 1.35) + g_xo, y + sz + g_yo}, {x, y - g_yo}, {x - (sz * 1.35) - g_xo, y + sz + g_yo}};

  painter.setBrush(QColor(218, 202, 37, 255));
  painter.drawPolygon(glow, std::size(glow));

  // chevron
  QPointF chevron[] = {{x + (sz * 1.25), y + sz}, {x, y}, {x - (sz * 1.25), y + sz}};
  painter.setBrush(redColor(fillAlpha));
  painter.drawPolygon(chevron, std::size(chevron));
}

void ModelState::setTransform(int w, int h, const QTransform &transform, const mat3 &calib, const mat3 &intrinsic) {
  fb_w = w;
  fb_h = h;
  clip_region = QRectF{-clip_margin, -clip_margin, fb_w + 2 * clip_margin, fb_h + 2 * clip_margin};
  car_space_transform = transform;
  calibration = calib;
  intrinsic_matrix = intrinsic;
}

void ModelState::updateLineData(const cereal::XYZTData::Reader &line, float y_off, float z_off,
                                  QPolygonF *pvd, int max_idx, bool allow_invert) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  QPointF left, right;
  pvd->clear();
  for (int i = 0; i <= max_idx; i++) {
    // highly negative x positions  are drawn above the frame and cause flickering, clip to zy plane of camera
    if (line_x[i] < 0) continue;

    bool l = mapToFrame(line_x[i], line_y[i] - y_off, line_z[i] + z_off, &left);
    bool r = mapToFrame(line_x[i], line_y[i] + y_off, line_z[i] + z_off, &right);
    if (l && r) {
      // For wider lines the drawn polygon will "invert" when going over a hill and cause artifacts
      if (!allow_invert && pvd->size() && left.y() > pvd->back().y()) {
        continue;
      }
      pvd->push_back(left);
      pvd->push_front(right);
    }
  }
}

// Projects a point in car to space to the corresponding point in full frame image space.
bool ModelState::mapToFrame(float in_x, float in_y, float in_z, QPointF *out) {
  const vec3 pt = (vec3){{in_x, in_y, in_z}};
  const vec3 Ep = matvecmul3(calibration, pt);
  const vec3 KEp = matvecmul3(intrinsic_matrix, Ep);

  QPointF point = car_space_transform.map(QPointF{KEp.v[0] / KEp.v[2], KEp.v[1] / KEp.v[2]});
  if (clip_region.contains(point)) {
    *out = point;
    return true;
  }
  return false;
}
