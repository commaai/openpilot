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
  {-67.81, -20.85, -15.47},
  {-65.98, -3.04, -13.35},
  {-62.47, 13.09, -11.37},
  {-59.14, 27.59, -7.22},
  {-54.01, 42.94, 1.73},
  {-44.48, 54.71, 17.22},
  {-32.82, 61.18, 35.89},
  {-18.42, 65.58, 53.77},
  {2.02, 68.02, 60.66},
  {22.11, 65.14, 53.20},
  {35.84, 60.73, 35.01},
  {46.52, 54.48, 16.40},
  {54.94, 42.68, 1.18},
  {59.20, 27.47, -7.62},
  {61.81, 13.02, -11.88},
  {64.48, -2.98, -13.95},
  {65.88, -20.80, -15.94},
  {-53.36, -39.44, 40.38},
  {-45.79, -45.50, 51.24},
  {-36.05, -47.39, 58.98},
  {-26.72, -46.71, 63.77},
  {-18.26, -44.43, 66.03},
  {16.04, -44.53, 65.81},
  {24.53, -46.78, 63.43},
  {33.82, -47.62, 58.57},
  {43.68, -45.86, 50.79},
  {51.43, -39.42, 39.93},
  {-0.63, -27.00, 68.58},
  {-0.45, -16.70, 76.98},
  {-0.15, -6.71, 85.56},
  {0.03, 2.04, 87.81},
  {-11.55, 10.08, 67.77},
  {-6.59, 10.98, 71.84},
  {0.10, 12.20, 73.78},
  {6.63, 10.93, 71.77},
  {11.47, 9.95, 67.51},
  {-40.59, -26.39, 46.76},
  {-34.78, -29.75, 54.40},
  {-26.35, -29.84, 54.70},
  {-18.49, -26.35, 52.64},
  {-25.58, -24.33, 54.47},
  {-34.32, -23.89, 52.21},
  {16.54, -26.32, 52.22},
  {24.34, -29.87, 54.18},
  {32.96, -29.63, 53.46},
  {38.94, -26.21, 46.49},
  {32.60, -23.91, 52.00},
  {23.79, -24.23, 54.16},
  {-24.99, 29.63, 57.84},
  {-16.17, 25.40, 67.58},
  {-5.21, 22.32, 73.25},
  {0.48, 23.42, 73.75},
  {5.92, 22.32, 73.18},
  {16.83, 25.28, 67.31},
  {25.53, 29.17, 57.34},
  {16.52, 30.67, 67.94},
  {8.99, 32.17, 72.79},
  {1.41, 32.34, 73.91},
  {-6.34, 32.25, 72.85},
  {-14.31, 30.94, 68.17},
  {-22.61, 28.83, 58.48},
  {-7.00, 28.11, 68.73},
  {0.61, 28.13, 70.32},
  {7.98, 28.13, 68.68},
  {24.00, 28.60, 57.80},
  {8.19, 25.22, 70.68},
  {1.19, 25.34, 71.88},
  {-5.95, 25.41, 71.00},
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
  QPointF face_kpts_draw[68];
  float face_kpts_draw_d[68];

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
