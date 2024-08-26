#include "selfdrive/ui/qt/onroad/model.h"

constexpr float MIN_DRAW_DISTANCE = 10.0f;
constexpr float MAX_DRAW_DISTANCE = 100.0f;
constexpr int CLIP_MARGIN = 500;

static int get_path_length_idx(const cereal::XYZTData::Reader &line, const float path_height) {
  const auto &line_x = line.getX();
  int max_idx = 0;
  for (int i = 1; i < line_x.size() && line_x[i] <= path_height; ++i) {
    max_idx = i;
  }
  return max_idx;
}

void ModelRenderer::draw(QPainter &painter, const QRect &surface_rect) {
  auto &sm = *(uiState()->sm);

  if (sm.updated("carParams")) {
    longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }

  // Check if data is up-to-date
  if (!(sm.alive("liveCalibration") && sm.alive("modelV2"))) {
    return;
  }

  clip_region = surface_rect.adjusted(-CLIP_MARGIN, -CLIP_MARGIN, CLIP_MARGIN, CLIP_MARGIN);
  experimental_model = sm["selfdriveState"].getSelfdriveState().getExperimentalMode();

  painter.save();

  const auto &model = sm["modelV2"].getModelV2();
  const auto &model_position = model.getPosition();
  float max_dist = std::clamp(*(model_position.getX().end() - 1), MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE);

  drawLaneLines(painter, model, max_dist);

  // Adjust max distance
  const auto &radar_state = sm["radarState"].getRadarState();
  const auto &lead_one = radar_state.getLeadOne();
  if (lead_one.getStatus()) {
    float lead_distance = lead_one.getDRel() * 2.0f;
    max_dist = std::clamp(lead_distance - std::min(lead_distance * 0.35f, 10.0f), 0.0f, max_dist);
  }

  drawPath(painter, model, max_dist, surface_rect.height());

  if (longitudinal_control && sm.alive("radarState")) {
    if (lead_one.getStatus()) {
      drawLead(painter, lead_one, model_position, surface_rect);
    }

    const auto &lead_two = radar_state.getLeadTwo();
    if (lead_two.getStatus() && std::abs(lead_one.getDRel() - lead_two.getDRel()) > 3.0f) {
      drawLead(painter, lead_two, model_position, surface_rect);
    }
  }

  painter.restore();
}

void ModelRenderer::drawLaneLines(QPainter &painter, const cereal::ModelDataV2::Reader &model, float max_dist) {
  painter.setPen(Qt::NoPen);

  QPolygonF polygon;
  const auto &lines = model.getLaneLines();
  const auto &probs = model.getLaneLineProbs();
  const int max_idx = get_path_length_idx(lines[0], max_dist);

  for (int i = 0; i < lines.size(); ++i) {
    mapLineToPolygon(lines[i], 0.025f * probs[i], 0.0f, &polygon, max_idx);
    painter.setBrush(QColor::fromRgbF(1.0, 1.0, 1.0, std::clamp(probs[i], 0.0f, 0.7f)));
    painter.drawPolygon(polygon);
  }

  const auto &edges = model.getRoadEdges();
  const auto &stds = model.getRoadEdgeStds();
  for (int i = 0; i < stds.size(); ++i) {
    mapLineToPolygon(edges[i], 0.025f, 0.0f, &polygon, max_idx);
    painter.setBrush(QColor::fromRgbF(1.0, 0.0, 0.0, std::clamp(1.0f - stds[i], 0.0f, 1.0f)));
    painter.drawPolygon(polygon);
  }
}

void ModelRenderer::drawPath(QPainter &painter, const cereal::ModelDataV2::Reader &model, float max_dist, int surface_height) {
  QPolygonF path_polygon;
  const auto &position = model.getPosition();
  const int max_idx = get_path_length_idx(position, max_dist);
  mapLineToPolygon(position, 0.9f, 1.22f, &path_polygon, max_idx, false);

  QLinearGradient bg(0, surface_height, 0, 0);
  if (experimental_model) {
    // The first half of path_polygon are the points for the right side of the path
    const auto &acceleration = model.getAcceleration().getX();
    const int max_len = std::min<int>(path_polygon.length() / 2, acceleration.size());

    for (int i = 0; i < max_len; ++i) {
      // Some points are out of frame
      const int track_idx = max_len - i - 1;  // flip idx to start from bottom right
      const float y_pos = path_polygon[track_idx].y();

      if (y_pos < 0 || y_pos > surface_height) continue;

      // Flip so 0 is bottom of frame
      const float lin_grad_point = (surface_height - y_pos) / surface_height;

      // speed up: 120, slow down: 0
      float path_hue = std::clamp(60 + acceleration[i] * 35, 0.0f, 120.f);
      // FIXME: painter.drawPolygon can be slow if hue is not rounded
      path_hue = int(path_hue * 100 + 0.5) / 100;

      float saturation = std::min(std::abs(acceleration[i] * 1.5f), 1.0f);
      float lightness = util::map_val(saturation, 0.0f, 1.0f, 0.95f, 0.62f);        // lighter when grey
      float alpha = util::map_val(lin_grad_point, 0.75f / 2.f, 0.75f, 0.4f, 0.0f);  // matches previous alpha fade
      bg.setColorAt(lin_grad_point, QColor::fromHslF(path_hue / 360.0f, saturation, lightness, alpha));

      // Skip a point, unless next is last
      i += (i + 2) < max_len ? 1 : 0;
    }
  } else {
    bg.setColorAt(0.0, QColor::fromHslF(148 / 360.0, 0.94, 0.51, 0.4));
    bg.setColorAt(0.5, QColor::fromHslF(112 / 360.0, 1.0, 0.68, 0.35));
    bg.setColorAt(1.0, QColor::fromHslF(112 / 360.0, 1.0, 0.68, 0.0));
  }

  painter.setBrush(bg);
  painter.drawPolygon(path_polygon);
}

void ModelRenderer::drawLead(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data,
                             const cereal::XYZTData::Reader &model_position, const QRect &surface_rect) {
  const float d_rel = lead_data.getDRel();
  const int z_index = get_path_length_idx(model_position, d_rel);
  const QPointF lead_pos = mapToScreen(d_rel, -lead_data.getYRel(), model_position.getZ()[z_index] + 1.22f);

  const float sz = std::clamp((25 * 30) / (d_rel / 3 + 30), 15.0f, 30.0f) * 2.35f;
  const float x = std::clamp<float>(lead_pos.x(), 0.f, surface_rect.width() - sz / 2.0f);
  const float y = std::min<float>(lead_pos.y(), surface_rect.height() - sz * 0.6f);

  const float g_xo = sz / 5.0f;
  const float g_yo = sz / 10.0f;

  QPointF glow[] = {{x + (sz * 1.35) + g_xo, y + sz + g_yo}, {x, y - g_yo}, {x - (sz * 1.35) - g_xo, y + sz + g_yo}};
  painter.setBrush(QColor(218, 202, 37, 255));
  painter.drawPolygon(glow, std::size(glow));

  QPointF chevron[] = {{x + (sz * 1.25), y + sz}, {x, y}, {x - (sz * 1.25), y + sz}};
  const float alpha = calculateLeadAlpha(d_rel, lead_data.getVRel());
  painter.setBrush(QColor(201, 34, 49, alpha));
  painter.drawPolygon(chevron, std::size(chevron));
}

uint8_t ModelRenderer::calculateLeadAlpha(float d_rel, float v_rel) {
  const float speed_threshold = 10.0f;
  const float lead_threshold = 40.0f;
  float alpha = 0.0f;
  if (d_rel < lead_threshold) {
    alpha = 255 * (1.0f - (d_rel / lead_threshold));
    if (v_rel < 0) {
      alpha += 255 * (-v_rel / speed_threshold);
    }
  }
  return std::clamp<uint8_t>(alpha, 0, 255);
}

QPointF ModelRenderer::mapToScreen(float in_x, float in_y, float in_z) {
  Eigen::Vector3f input(in_x, in_y, in_z);
  auto pt = car_space_transform * input;
  return {pt.x() / pt.z(), pt.y() / pt.z()};
}

void ModelRenderer::mapLineToPolygon(const cereal::XYZTData::Reader &line, float y_off, float z_off,
                                     QPolygonF *pvd, int max_idx, bool allow_invert) {
  const auto &x_coords = line.getX(), &y_coords = line.getY(), &z_coords = line.getZ();
  pvd->clear();

  for (int i = 0; i <= max_idx; ++i) {
    float x = x_coords[i], y = y_coords[i], z = z_coords[i];
    // highly negative x positions  are drawn above the frame and cause flickering, clip to zy plane of camera
    if (x < 0) continue;

    const QPointF left = mapToScreen(x, y - y_off, z + z_off);
    const QPointF right = mapToScreen(x, y + y_off, z + z_off);
    if (clip_region.contains(left) && clip_region.contains(right)) {
      // For wider lines the drawn polygon will "invert" when going over a hill and cause artifacts
      if (!allow_invert && !pvd->empty() && left.y() > pvd->back().y()) {
        continue;
      }
      pvd->push_back(left);
      pvd->push_front(right);
    }
  }
}
