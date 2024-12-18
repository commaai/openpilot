#include "selfdrive/ui/qt/onroad/driver_monitoring.h"
#include <algorithm>
#include <cmath>

#include "selfdrive/ui/qt/onroad/buttons.h"
#include "selfdrive/ui/qt/util.h"

// Default 3D coordinates for face keypoints
static constexpr vec3 DEFAULT_FACE_KPTS_3D[] = {
  {-5.98, -51.20, 8.00}, {-17.64, -49.14, 8.00}, {-23.81, -46.40, 8.00}, {-29.98, -40.91, 8.00}, {-32.04, -37.49, 8.00},
  {-34.10, -32.00, 8.00}, {-36.16, -21.03, 8.00}, {-36.16, 6.40, 8.00}, {-35.47, 10.51, 8.00}, {-32.73, 19.43, 8.00},
  {-29.30, 26.29, 8.00}, {-24.50, 33.83, 8.00}, {-19.01, 41.37, 8.00}, {-14.21, 46.17, 8.00}, {-12.16, 47.54, 8.00},
  {-4.61, 49.60, 8.00}, {4.99, 49.60, 8.00}, {12.53, 47.54, 8.00}, {14.59, 46.17, 8.00}, {19.39, 41.37, 8.00},
  {24.87, 33.83, 8.00}, {29.67, 26.29, 8.00}, {33.10, 19.43, 8.00}, {35.84, 10.51, 8.00}, {36.53, 6.40, 8.00},
  {36.53, -21.03, 8.00}, {34.47, -32.00, 8.00}, {32.42, -37.49, 8.00}, {30.36, -40.91, 8.00}, {24.19, -46.40, 8.00},
  {18.02, -49.14, 8.00}, {6.36, -51.20, 8.00}, {-5.98, -51.20, 8.00},
};

// Colors used for drawing based on monitoring state
static const QColor DMON_ENGAGED_COLOR = QColor::fromRgbF(0.1, 0.945, 0.26);
static const QColor DMON_DISENGAGED_COLOR = QColor::fromRgbF(0.545, 0.545, 0.545);

DriverMonitorRenderer::DriverMonitorRenderer() : face_kpts_draw(std::size(DEFAULT_FACE_KPTS_3D)) {
  dm_img = loadPixmap("../assets/img_driver_face.png", {img_size + 5, img_size + 5});
}

void DriverMonitorRenderer::updateState(const UIState &s) {
  auto &sm = *(s.sm);
  is_visible = sm["selfdriveState"].getSelfdriveState().getAlertSize() == cereal::SelfdriveState::AlertSize::NONE &&
               sm.rcv_frame("driverStateV2") > s.scene.started_frame;
  if (!is_visible) return;

  auto dm_state = sm["driverMonitoringState"].getDriverMonitoringState();
  is_active = dm_state.getIsActiveMode();
  is_rhd = dm_state.getIsRHD();
  dm_fade_state = std::clamp(dm_fade_state + 0.2f * (0.5f - is_active), 0.0f, 1.0f);

  const auto &driverstate = sm["driverStateV2"].getDriverStateV2();
  const auto driver_orient = is_rhd ? driverstate.getRightDriverData().getFaceOrientation() : driverstate.getLeftDriverData().getFaceOrientation();

  for (int i = 0; i < 3; ++i) {
    float v_this = (i == 0 ? (driver_orient[i] < 0 ? 0.7 : 0.9) : 0.4) * driver_orient[i];
    driver_pose_diff[i] = std::abs(driver_pose_vals[i] - v_this);
    driver_pose_vals[i] = 0.8f * v_this + (1 - 0.8) * driver_pose_vals[i];
    driver_pose_sins[i] = std::sin(driver_pose_vals[i] * (1.0f - dm_fade_state));
    driver_pose_coss[i] = std::cos(driver_pose_vals[i] * (1.0f - dm_fade_state));
  }

  auto [sin_y, sin_x, sin_z] = driver_pose_sins;
  auto [cos_y, cos_x, cos_z] = driver_pose_coss;

  // Rotation matrix for transforming face keypoints based on driver's head orientation
  const mat3 r_xyz = {{
    cos_x * cos_z, cos_x * sin_z, -sin_x,
    -sin_y * sin_x * cos_z - cos_y * sin_z, -sin_y * sin_x * sin_z + cos_y * cos_z, -sin_y * cos_x,
    cos_y * sin_x * cos_z - sin_y * sin_z, cos_y * sin_x * sin_z + sin_y * cos_z, cos_y * cos_x,
  }};

  // Transform vertices
  for (int i = 0; i < face_kpts_draw.size(); ++i) {
    vec3 kpt = matvecmul3(r_xyz, DEFAULT_FACE_KPTS_3D[i]);
    face_kpts_draw[i] = {{kpt.v[0], kpt.v[1], kpt.v[2] * (1.0f - dm_fade_state) + 8 * dm_fade_state}};
  }
}

void DriverMonitorRenderer::draw(QPainter &painter, const QRect &surface_rect) {
  if (!is_visible) return;

  painter.save();

  int offset = UI_BORDER_SIZE + btn_size / 2;
  float x = is_rhd ? surface_rect.width() - offset : offset;
  float y = surface_rect.height() - offset;
  float opacity = is_active ? 0.65f : 0.2f;

  drawIcon(painter, QPoint(x, y), dm_img, QColor(0, 0, 0, 70), opacity);

  QPointF keypoints[std::size(DEFAULT_FACE_KPTS_3D)];
  for (int i = 0; i < std::size(keypoints); ++i) {
    const auto &v = face_kpts_draw[i].v;
    float kp = (v[2] - 8) / 120.0f + 1.0f;
    keypoints[i] = QPointF(v[0] * kp + x, v[1] * kp + y);
  }

  painter.setPen(QPen(QColor::fromRgbF(1.0, 1.0, 1.0, opacity), 5.2, Qt::SolidLine, Qt::RoundCap));
  painter.drawPolyline(keypoints, std::size(keypoints));

  // tracking arcs
  const int arc_l = 133;
  const float arc_t_default = 6.7f;
  const float arc_t_extend = 12.0f;
  QColor arc_color = uiState()->engaged() ? DMON_ENGAGED_COLOR : DMON_DISENGAGED_COLOR;
  arc_color.setAlphaF(0.4 * (1.0f - dm_fade_state));

  float delta_x = -driver_pose_sins[1] * arc_l / 2.0f;
  float delta_y = -driver_pose_sins[0] * arc_l / 2.0f;

  // Draw horizontal tracking arc
  painter.setPen(QPen(arc_color, arc_t_default + arc_t_extend * std::min(1.0, driver_pose_diff[1] * 5.0), Qt::SolidLine, Qt::RoundCap));
  painter.drawArc(QRectF(std::min(x + delta_x, x), y - arc_l / 2, std::abs(delta_x), arc_l), (driver_pose_sins[1] > 0 ? 90 : -90) * 16, 180 * 16);

  // Draw vertical tracking arc
  painter.setPen(QPen(arc_color, arc_t_default + arc_t_extend * std::min(1.0, driver_pose_diff[0] * 5.0), Qt::SolidLine, Qt::RoundCap));
  painter.drawArc(QRectF(x - arc_l / 2, std::min(y + delta_y, y), arc_l, std::abs(delta_y)), (driver_pose_sins[0] > 0 ? 0 : 180) * 16, 180 * 16);

  painter.restore();
}
