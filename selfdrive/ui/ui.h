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

const int UI_FREQ = 20;   // Hz
typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

const mat3 DEFAULT_CALIBRATION = {{ 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 }};

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
  [STATUS_DISENGAGED] =  QColor(0x17, 0x33, 0x49, 0xc8),
  [STATUS_OVERRIDE] = QColor(0x91, 0x9b, 0x95, 0xf1),
  [STATUS_ENGAGED] = QColor(0x17, 0x86, 0x44, 0xf1),
  [STATUS_WARNING] = QColor(0xDA, 0x6F, 0x25, 0xf1),
  [STATUS_ALERT] = QColor(0xC9, 0x22, 0x31, 0xf1),
};

typedef struct UIScene {
  bool calibration_valid = false;
  mat3 view_from_calib = DEFAULT_CALIBRATION;
  cereal::PandaState::PandaType pandaType;

  // modelV2
  float lane_line_probs[4];
  float road_edge_stds[2];
  QPolygonF track_vertices;
  QPolygonF lane_line_vertices[4];
  QPolygonF road_edge_vertices[2];

  // lead
  QPointF lead_vertices[2];

  float light_sensor, accel_sensor, gyro_sensor;
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
  int prime_type = 0;

  QTransform car_space_transform;
  bool wide_camera;

signals:
  void uiUpdate(const UIState &s);
  void offroadTransition(bool offroad);

private slots:
  void update();

private:
  QTimer *timer;
  bool started_prev = false;
};

UIState *uiState();

// device management class

class Device : public QObject {
  Q_OBJECT

public:
  Device(QObject *parent = 0);

private:
  // auto brightness
  const float accel_samples = 5*UI_FREQ;

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
