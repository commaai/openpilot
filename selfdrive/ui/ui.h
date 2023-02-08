#pragma once

#include <memory>
#include <string>
#include <optional>

#include <QObject>
#include <QTimer>
#include <QColor>
#include <QFuture>
#include <QPolygonF>
#include <QTransform>

#include "cereal/messaging/messaging.h"
#include "common/modeldata.h"
#include "common/params.h"
#include "common/timing.h"

const int bdr_s = 30;
const int header_h = 420;
const int footer_h = 280;

const int UI_FREQ = 20; // Hz
typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

const mat3 DEFAULT_CALIBRATION = {{ 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 }};

const int FACE_KPTS_SIZE = 31;
const vec3 default_face_kpts_3d[FACE_KPTS_SIZE] = {
  {-7.74, -51.07, 8.00}, {-18.64, -48.16, 8.00}, {-26.64, -43.80, 8.00}, {-33.18, -37.26, 8.00}, {-36.08, -29.99, 8.00},
  {-36.81, -25.63, 8.00}, {-36.81, -9.64, 8.00}, {-36.08, -3.82, 8.00}, {-33.90, 4.17, 8.00}, {-33.90, 20.89, 8.00},
  {-33.18, 23.07, 8.00}, {-31.72, 25.98, 8.00}, {-22.27, 36.88, 8.00}, {-12.83, 47.06, 8.00}, {-7.74, 49.24, 8.00},
  {8.25, 49.24, 8.00}, {13.34, 47.06, 8.00}, {22.79, 36.88, 8.00}, {32.24, 25.98, 8.00}, {33.69, 23.07, 8.00},
  {34.42, 20.89, 8.00}, {34.42, 4.17, 8.00}, {36.60, -3.82, 8.00}, {37.33, -9.64, 8.00}, {37.33, -25.63, 8.00},
  {36.60, -29.99, 8.00}, {33.69, -37.26, 8.00}, {27.15, -43.80, 8.00}, {19.16, -48.16, 8.00}, {8.25, -51.07, 8.00},
  {-7.74, -51.07, 8.00},
};

struct Alert {
  QString text1;
  QString text2;
  QString type;
  cereal::ControlsState::AlertSize size;
  AudibleAlert sound;

  bool equal(const Alert &a2) {
    return text1 == a2.text1 && text2 == a2.text2 && type == a2.type && sound == a2.sound;
  }

  static Alert get(const SubMaster &sm, uint64_t started_frame) {
    const cereal::ControlsState::Reader &cs = sm["controlsState"].getControlsState();
    if (sm.updated("controlsState")) {
      return {cs.getAlertText1().cStr(), cs.getAlertText2().cStr(),
              cs.getAlertType().cStr(), cs.getAlertSize(),
              cs.getAlertSound()};
    } else if ((sm.frame - started_frame) > 5 * UI_FREQ) {
      const int CONTROLS_TIMEOUT = 5;
      const int controls_missing = (nanos_since_boot() - sm.rcv_time("controlsState")) / 1e9;

      // Handle controls timeout
      if (sm.rcv_frame("controlsState") < started_frame) {
        // car is started, but controlsState hasn't been seen at all
        return {"openpilot Unavailable", "Waiting for controls to start",
                "controlsWaiting", cereal::ControlsState::AlertSize::MID,
                AudibleAlert::NONE};
      } else if (controls_missing > CONTROLS_TIMEOUT && !Hardware::PC()) {
        // car is started, but controls is lagging or died
        if (cs.getEnabled() && (controls_missing - CONTROLS_TIMEOUT) < 10) {
          return {"TAKE CONTROL IMMEDIATELY", "Controls Unresponsive",
                  "controlsUnresponsive", cereal::ControlsState::AlertSize::FULL,
                  AudibleAlert::WARNING_IMMEDIATE};
        } else {
          return {"Controls Unresponsive", "Reboot Device",
                  "controlsUnresponsivePermanent", cereal::ControlsState::AlertSize::MID,
                  AudibleAlert::NONE};
        }
      }
    }
    return {};
  }
};

typedef enum UIStatus {
  STATUS_DISENGAGED,
  STATUS_OVERRIDE,
  STATUS_ENGAGED,
  STATUS_WARNING,
  STATUS_ALERT,
} UIStatus;

const QColor bg_colors [] = {
  [STATUS_DISENGAGED] = QColor(0x17, 0x33, 0x49, 0xc8),
  [STATUS_OVERRIDE] = QColor(0x91, 0x9b, 0x95, 0xf1),
  [STATUS_ENGAGED] = QColor(0x17, 0x86, 0x44, 0xf1),
  [STATUS_WARNING] = QColor(0xDA, 0x6F, 0x25, 0xf1),
  [STATUS_ALERT] = QColor(0xC9, 0x22, 0x31, 0xf1),
};

typedef struct UIScene {
  bool calibration_valid = false;
  bool calibration_wide_valid  = false;
  bool wide_cam = true;
  mat3 view_from_calib = DEFAULT_CALIBRATION;
  mat3 view_from_wide_calib = DEFAULT_CALIBRATION;
  cereal::PandaState::PandaType pandaType;

  // modelV2
  float lane_line_probs[4];
  float road_edge_stds[2];
  QPolygonF track_vertices;
  QPolygonF lane_line_vertices[4];
  QPolygonF road_edge_vertices[2];

  // lead
  QPointF lead_vertices[2];

  // driverStateV2
  float driver_pose_vals[3];
  float driver_pose_diff[3];
  float driver_pose_sins[3];
  float driver_pose_coss[3];
  vec3 face_kpts_draw[FACE_KPTS_SIZE];

  float light_sensor;
  bool started, ignition, is_metric, map_on_left, longitudinal_control;
  uint64_t started_frame;
} UIScene;

class UIState : public QObject {
  Q_OBJECT

public:
  UIState(QObject* parent = 0);
  void updateStatus();
  inline bool worldObjectsVisible() const {
    return sm->rcv_frame("liveCalibration") > scene.started_frame;
  };
  inline bool engaged() const {
    return scene.started && (*sm)["controlsState"].getControlsState().getEnabled();
  };

  int fb_w = 0, fb_h = 0;

  std::unique_ptr<SubMaster> sm;

  UIStatus status;
  UIScene scene = {};

  bool awake;
  int prime_type;
  QString language;

  QTransform car_space_transform;
  bool wide_cam_only;

signals:
  void uiUpdate(const UIState &s);
  void offroadTransition(bool offroad);
  void primeTypeChanged(int prime_type);

private slots:
  void update();

private:
  QTimer *timer;
  bool started_prev = false;
  int prime_type_prev = -1;
};

UIState *uiState();

// device management class
class Device : public QObject {
  Q_OBJECT

public:
  Device(QObject *parent = 0);

private:
  bool awake = false;
  int interactive_timeout = 0;
  bool ignition_on = false;
  int last_brightness = 0;
  FirstOrderFilter brightness_filter;
  QFuture<void> brightness_future;

  void updateBrightness(const UIState &s);
  void updateWakefulness(const UIState &s);
  bool motionTriggered(const UIState &s);
  void setAwake(bool on);

signals:
  void displayPowerChanged(bool on);
  void interactiveTimout();

public slots:
  void resetInteractiveTimout();
  void update(const UIState &s);
};

void ui_update_params(UIState *s);
int get_path_length_idx(const cereal::ModelDataV2::XYZTData::Reader &line, const float path_height);
void update_model(UIState *s, const cereal::ModelDataV2::Reader &model);
void update_dmonitoring(UIState *s, const cereal::DriverStateV2::Reader &driverstate, float dm_fade_state);
void update_leads(UIState *s, const cereal::RadarState::Reader &radar_state, const cereal::ModelDataV2::XYZTData::Reader &line);
void update_line_data(const UIState *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                      float y_off, float z_off, QPolygonF *pvd, int max_idx, bool allow_invert);
