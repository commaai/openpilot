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
  {-67.81, -25.85, -65.47},
  {-65.98, -8.04, -63.35},
  {-62.47, 8.09, -61.37},
  {-59.14, 22.59, -57.22},
  {-54.01, 37.94, -48.27},
  {-44.48, 49.71, -32.78},
  {-32.82, 56.18, -14.11},
  {-18.42, 60.58, 3.77},
  {2.02, 63.02, 10.66},
  {22.11, 60.14, 3.20},
  {35.84, 55.73, -14.99},
  {46.52, 49.48, -33.60},
  {54.94, 37.68, -48.82},
  {59.20, 22.47, -57.62},
  {61.81, 8.02, -61.88},
  {64.48, -7.98, -63.95},
  {65.88, -25.80, -65.94},
  {-53.36, -44.44, -9.62},
  {-45.79, -50.50, 1.24},
  {-36.05, -52.39, 8.98},
  {-26.72, -51.71, 13.77},
  {-18.26, -49.43, 16.03},
  {16.04, -49.53, 15.81},
  {24.53, -51.78, 13.43},
  {33.82, -52.62, 8.57},
  {43.68, -50.86, 0.79},
  {51.43, -44.42, -10.07},
  {-0.63, -32.00, 18.58},
  {-0.45, -21.70, 26.98},
  {-0.15, -11.71, 35.56},
  {0.03, -2.96, 37.81},
  {-11.55, 5.08, 17.77},
  {-6.59, 5.98, 21.84},
  {0.10, 7.20, 23.78},
  {6.63, 5.93, 21.77},
  {11.47, 4.95, 17.51},
  {-40.59, -31.39, -3.24},
  {-34.78, -34.75, 4.40},
  {-26.35, -34.84, 4.70},
  {-18.49, -31.35, 2.64},
  {-25.58, -29.33, 4.47},
  {-34.32, -28.89, 2.21},
  {16.54, -31.32, 2.22},
  {24.34, -34.87, 4.18},
  {32.96, -34.63, 3.46},
  {38.94, -31.21, -3.51},
  {32.60, -28.91, 2.00},
  {23.79, -29.23, 4.16},
  {-24.99, 24.63, 7.84},
  {-16.17, 20.40, 17.58},
  {-5.21, 17.32, 23.25},
  {0.48, 18.42, 23.75},
  {5.92, 17.32, 23.18},
  {16.83, 20.28, 17.31},
  {25.53, 24.17, 7.34},
  {16.52, 25.67, 17.94},
  {8.99, 27.17, 22.79},
  {1.41, 27.34, 23.91},
  {-6.34, 27.25, 22.85},
  {-14.31, 25.94, 18.17},
  {-22.61, 23.83, 8.48},
  {-7.00, 23.11, 18.73},
  {0.61, 23.13, 20.32},
  {7.98, 23.13, 18.68},
  {24.00, 23.60, 7.80},
  {8.19, 20.22, 20.68},
  {1.19, 20.34, 21.88},
  {-5.95, 20.41, 21.00},
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
