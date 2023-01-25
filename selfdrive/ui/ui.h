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

const vec3 default_face_kpts_3d[68] = {
  {-75.34, -28.73, -72.74},
  {-73.31, -8.93, -70.39},
  {-69.41, 8.99, -68.19},
  {-65.71, 25.10, -63.58},
  {-60.01, 42.16, -53.63},
  {-49.42, 55.24, -36.42},
  {-36.47, 62.42, -15.67},
  {-20.46, 67.31, 4.18},
  {2.24, 70.02, 11.84},
  {24.57, 66.82, 3.56},
  {39.83, 61.92, -16.65},
  {51.69, 54.98, -37.33},
  {61.04, 41.87, -54.24},
  {65.78, 24.96, -64.03},
  {68.68, 8.91, -68.76},
  {71.64, -8.87, -71.06},
  {73.20, -28.67, -73.26},
  {-59.29, -49.38, -10.69},
  {-50.88, -56.11, 1.37},
  {-40.06, -58.21, 9.98},
  {-29.69, -57.45, 15.30},
  {-20.28, -54.92, 17.81},
  {17.83, -55.03, 17.56},
  {27.26, -57.53, 14.93},
  {37.58, -58.47, 9.53},
  {48.53, -56.51, 0.88},
  {57.14, -49.35, -11.19},
  {-0.71, -35.56, 20.64},
  {-0.50, -24.11, 29.98},
  {-0.16, -13.01, 39.51},
  {0.04, -3.29, 42.01},
  {-12.83, 5.64, 19.74},
  {-7.32, 6.64, 24.27},
  {0.11, 8.01, 26.43},
  {7.36, 6.59, 24.18},
  {12.74, 5.50, 19.46},
  {-45.10, -34.88, -3.60},
  {-38.64, -38.62, 4.89},
  {-29.28, -38.72, 5.22},
  {-20.55, -34.83, 2.94},
  {-28.43, -32.58, 4.97},
  {-38.13, -32.10, 2.46},
  {18.38, -34.80, 2.46},
  {27.04, -38.75, 4.65},
  {36.62, -38.48, 3.85},
  {43.26, -34.68, -3.90},
  {36.23, -32.12, 2.22},
  {26.43, -32.47, 4.62},
  {-27.77, 27.36, 8.71},
  {-17.96, 22.67, 19.53},
  {-5.79, 19.25, 25.83},
  {0.53, 20.47, 26.39},
  {6.58, 19.25, 25.76},
  {18.69, 22.54, 19.23},
  {28.37, 26.86, 8.16},
  {18.35, 28.52, 19.94},
  {9.99, 30.18, 25.32},
  {1.57, 30.38, 26.57},
  {-7.05, 30.28, 25.39},
  {-15.89, 28.82, 20.19},
  {-25.12, 26.47, 9.42},
  {-7.78, 25.68, 20.81},
  {0.68, 25.70, 22.58},
  {8.87, 25.70, 20.75},
  {26.67, 26.22, 8.67},
  {9.10, 22.46, 22.98},
  {1.32, 22.60, 24.31},
  {-6.61, 22.68, 23.34},
};

const int face_end_idxs[8]= {16, 21, 26, 30, 35, 41, 47, 67};

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
  float driver_pose_pitch;
  float driver_pose_yaw;
  float driver_pose_roll;
  QLineF face_kpt_segments[60];
  float face_kpt_segments_d[61];

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
void update_leads(UIState *s, const cereal::RadarState::Reader &radar_state, const cereal::ModelDataV2::XYZTData::Reader &line);
void update_line_data(const UIState *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                      float y_off, float z_off, QPolygonF *pvd, int max_idx, bool allow_invert);
