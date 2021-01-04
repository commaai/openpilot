#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <poll.h>
#include <sys/mman.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "common/visionimg.h"
#include "common/utilpp.h"
#include "ui.hpp"
#include "paint.hpp"

int write_param_float(float param, const char* param_name, bool persistent_param) {
  char s[16];
  int size = snprintf(s, sizeof(s), "%f", param);
  return Params(persistent_param).write_db_value(param_name, s, size < sizeof(s) ? size : sizeof(s));
}

void ui_init(UIState *s) {
  s->sm = new SubMaster({"modelV2", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal", "frame",
                         "health", "carParams", "ubloxGnss", "driverState", "dMonitoringState", "sensorEvents"});

  s->scene.started = false;
  s->scene.status = STATUS_OFFROAD;
  s->scene.satelliteCount = -1;

  s->fb = framebuffer_init("ui", 0, true, &s->fb_w, &s->fb_h);
  assert(s->fb);

  ui_nvg_init(s);
}

static void ui_init_vision(UIState *s) {
  // Invisible until we receive a calibration message.
  s->world_objects_visible = false;

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    if (s->khr[i] != 0) {
      visionimg_destroy_gl(s->khr[i], s->priv_hnds[i]);
      glDeleteTextures(1, &s->frame_texs[i]);
    }

    VisionImg img = {
      .fd = s->stream.bufs[i].fd,
      .format = VISIONIMG_FORMAT_RGB24,
      .width = s->stream.bufs_info.width,
      .height = s->stream.bufs_info.height,
      .stride = s->stream.bufs_info.stride,
      .bpp = 3,
      .size = s->stream.bufs_info.buf_len,
    };
#ifndef QCOM
    s->priv_hnds[i] = s->stream.bufs[i].addr;
#endif
    s->frame_texs[i] = visionimg_to_gl(&img, &s->khr[i], &s->priv_hnds[i]);

    glBindTexture(GL_TEXTURE_2D, s->frame_texs[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
  }
  assert(glGetError() == GL_NO_ERROR);
}

void ui_update_vision(UIState *s) {

  if (!s->vision_connected && s->scene.started) {
    const VisionStreamType type = s->scene.frontview ? VISION_STREAM_RGB_FRONT : VISION_STREAM_RGB_BACK;
    int err = visionstream_init(&s->stream, type, true, nullptr);
    if (err == 0) {
      ui_init_vision(s);
      s->vision_connected = true;
    }
  }

  if (s->vision_connected) {
    if (!s->scene.started) goto destroy;

    // poll for a new frame
    struct pollfd fds[1] = {{
      .fd = s->stream.ipc_fd,
      .events = POLLOUT,
    }};
    int ret = poll(fds, 1, 100);
    if (ret > 0) {
      if (!visionstream_get(&s->stream, nullptr)) goto destroy;
    }
  }

  return;

destroy:
  visionstream_destroy(&s->stream);
  s->vision_connected = false;
}

static void update_lead(const cereal::RadarState::LeadData::Reader &lead, UIScene::LeadData &lead_data) {
  lead_data = {.status = lead.getStatus(),
               .d_rel = lead.getDRel(),
               .v_rel = lead.getVRel(),
               .y_rel = lead.getYRel()};
}

void update_sockets(UIState *s) {

  UIScene &scene = s->scene;
  SubMaster &sm = *(s->sm);

  if (sm.update(0) == 0){
    return;
  }

  if (scene.started && sm.updated("controlsState")) {
    auto event = sm["controlsState"];
    const auto cs = event.getControlsState();
    scene.v_cruise = cs.getVCruise();
    scene.v_ego = cs.getVEgo();
    scene.decel_for_model = cs.getDecelForModel();
    scene.controls_enabled = cs.getEnabled();
    scene.engageable = cs.getEngageable();

    // TODO: the alert stuff shouldn't be handled here
    auto alert_sound = cs.getAlertSound();
    if (scene.alert_type.compare(cs.getAlertType()) != 0) {
      if (alert_sound == AudibleAlert::NONE) {
        s->sound->stop();
      } else {
        s->sound->play(alert_sound);
      }
    }
    scene.alert_text1 = cs.getAlertText1();
    scene.alert_text2 = cs.getAlertText2();
    scene.alert_size = cs.getAlertSize();
    scene.alert_type = cs.getAlertType();
    auto alertStatus = cs.getAlertStatus();
    if (alertStatus == cereal::ControlsState::AlertStatus::USER_PROMPT) {
      scene.status = STATUS_WARNING;
    } else if (alertStatus == cereal::ControlsState::AlertStatus::CRITICAL) {
      scene.status = STATUS_ALERT;
    } else {
      scene.status = cs.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
    }

    float alert_blinkingrate = cs.getAlertBlinkingRate();
    if (alert_blinkingrate > 0.) {
      if (scene.alert_blinked) {
        if (scene.alert_blinking_alpha > 0.0 && scene.alert_blinking_alpha < 1.0) {
          scene.alert_blinking_alpha += (0.05*alert_blinkingrate);
        } else {
          scene.alert_blinked = false;
        }
      } else {
        if (scene.alert_blinking_alpha > 0.25) {
          scene.alert_blinking_alpha -= (0.05*alert_blinkingrate);
        } else {
          scene.alert_blinking_alpha += 0.25;
          scene.alert_blinked = true;
        }
      }
    }
  }
  if (sm.updated("radarState")) {
    const auto radar_state = sm["radarState"].getRadarState();
    update_lead(radar_state.getLeadOne(), scene.lead[0]);
    update_lead(radar_state.getLeadTwo(), scene.lead[1]);
  }
  if (sm.updated("liveCalibration")) {
    s->world_objects_visible = true;
    auto extrinsicl = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
    for (int i = 0; i < 3 * 4; i++) {
      scene.extrinsic_matrix.v[i] = extrinsicl[i];
    }
  }
  if (sm.updated("modelV2")) {
    scene.model = sm["modelV2"].getModelV2();
    scene.max_distance = fmin(scene.model.getPosition().getX()[TRAJECTORY_SIZE - 1], MAX_DRAW_DISTANCE);
    for (int ll_idx = 0; ll_idx < 4; ll_idx++) {
      if (scene.model.getLaneLineProbs().size() > ll_idx) {
        scene.lane_line_probs[ll_idx] = scene.model.getLaneLineProbs()[ll_idx];
      } else {
        scene.lane_line_probs[ll_idx] = 0.0;
      }
    }

    for (int re_idx = 0; re_idx < 2; re_idx++) {
      if (scene.model.getRoadEdgeStds().size() > re_idx) {
        scene.road_edge_stds[re_idx] = scene.model.getRoadEdgeStds()[re_idx];
      } else {
        scene.road_edge_stds[re_idx] = 1.0;
      }
    }
  }
  if (sm.updated("uiLayoutState")) {
    auto data = sm["uiLayoutState"].getUiLayoutState();
    s->active_app = data.getActiveApp();
    s->sidebar_collapsed = data.getSidebarCollapsed();
  }
  if (sm.updated("thermal")) {
    const auto thermal = sm["thermal"].getThermal();
    scene.network_type = thermal.getNetworkType();
    scene.network_strength = thermal.getNetworkStrength();
    scene.battery_status = thermal.getBatteryStatus();
    scene.battery_percent = thermal.getBatteryPercent();
    scene.ambient = thermal.getAmbient();
    scene.thermal_status = thermal.getThermalStatus();
  }
  if (sm.updated("ubloxGnss")) {
    auto data = sm["ubloxGnss"].getUbloxGnss();
    if (data.which() == cereal::UbloxGnss::MEASUREMENT_REPORT) {
      scene.satelliteCount = data.getMeasurementReport().getNumMeas();
    }
  }
  if (sm.updated("health")) {
    auto health = sm["health"].getHealth();
    scene.hwType = health.getHwType();
    scene.ignition = health.getIgnitionLine() || health.getIgnitionCan();
  } else if ((s->sm->frame - s->sm->rcv_frame("health")) > 5*UI_FREQ) {
    scene.hwType = cereal::HealthData::HwType::UNKNOWN;
  }
  if (sm.updated("carParams")) {
    scene.longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }
  if (sm.updated("driverState")) {
    const auto fact_position = sm["driverState"].getDriverState().getFacePosition();
    scene.face_position[0] = fact_position[0];
    scene.face_position[1] = fact_position[1];
  }
  if (sm.updated("dMonitoringState")) {
    scene.dmonitoring_state = sm["dMonitoringState"].getDMonitoringState();
    scene.is_rhd = scene.dmonitoring_state.getIsRHD();
    scene.frontview = scene.dmonitoring_state.getIsPreview();
    scene.face_detected = scene.dmonitoring_state.getFaceDetected();
  } else if (scene.frontview && (sm.frame - sm.rcv_frame("dMonitoringState")) > UI_FREQ/2) {
    scene.frontview = false;
  }
  if (sm.updated("sensorEvents")) {
    for (auto sensor : sm["sensorEvents"].getSensorEvents()) {
      if (sensor.which() == cereal::SensorEventData::LIGHT) {
        scene.light_sensor = sensor.getLight();
      } else if (!scene.started && sensor.which() == cereal::SensorEventData::ACCELERATION) {
        scene.accel_sensor = sensor.getAcceleration().getV()[2];
      } else if (!scene.started && sensor.which() == cereal::SensorEventData::GYRO_UNCALIBRATED) {
        scene.gyro_sensor = sensor.getGyroUncalibrated().getV()[1];
      }
    }
  }

  scene.started =  sm["thermal"].getThermal().getStarted() || scene.frontview;
}

static void ui_read_params(UIState *s) {
  const uint64_t frame = s->sm->frame;
  UIScene &scene = s->scene;

  if (frame % (5*UI_FREQ) == 0) {
    read_param(&scene.is_metric, "IsMetric");
  } else if (frame % (6*UI_FREQ) == 0) {
    scene.athenaStatus = NET_DISCONNECTED;
    uint64_t last_ping = 0;
    if (read_param(&last_ping, "LastAthenaPingTime") == 0) {
      scene.athenaStatus = nanos_since_boot() - last_ping < 70e9 ? NET_CONNECTED : NET_ERROR;
    }
  }
}

void ui_update(UIState *s) {
  ui_read_params(s);
  update_sockets(s);
  ui_update_vision(s);

  UIScene &scene = s->scene;
  // Handle onroad/offroad transition
  if (!scene.started && scene.status != STATUS_OFFROAD) {
    scene.status = STATUS_OFFROAD;
    s->active_app = cereal::UiLayoutState::App::HOME;
    s->sidebar_collapsed = false;
    s->sound->stop();
  } else if (scene.started && scene.status == STATUS_OFFROAD) {
    scene.status = STATUS_DISENGAGED;
    s->started_frame = s->sm->frame;

    s->active_app = cereal::UiLayoutState::App::NONE;
    s->sidebar_collapsed = true;
    scene.alert_blinked = false;
    scene.alert_blinking_alpha = 1.0;
    scene.alert_size = cereal::ControlsState::AlertSize::NONE;
  }

  // Handle controls/fcamera timeout
  if (scene.started && !scene.frontview && ((s->sm)->frame - s->started_frame) > 10*UI_FREQ) {
    if ((s->sm)->rcv_frame("controlsState") < s->started_frame) {
      // car is started, but controlsState hasn't been seen at all
      scene.alert_text1 = "openpilot Unavailable";
      scene.alert_text2 = "Waiting for controls to start";
      scene.alert_size = cereal::ControlsState::AlertSize::MID;
    } else if (((s->sm)->frame - (s->sm)->rcv_frame("controlsState")) > 5*UI_FREQ) {
      // car is started, but controls is lagging or died
      if (scene.alert_text2 != "Controls Unresponsive" &&
          scene.alert_text1 != "Camera Malfunction") {
        s->sound->play(AudibleAlert::CHIME_WARNING_REPEAT);
        LOGE("Controls unresponsive");
      }

      scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      scene.alert_text2 = "Controls Unresponsive";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      scene.status = STATUS_ALERT;
    }

    const uint64_t frame_pkt = (s->sm)->rcv_frame("frame");
    const uint64_t frame_delayed = (s->sm)->frame - frame_pkt;
    const uint64_t since_started = (s->sm)->frame - s->started_frame;
    if ((frame_pkt > s->started_frame || since_started > 15*UI_FREQ) && frame_delayed > 5*UI_FREQ) {
      // controls is fine, but rear camera is lagging or died
      scene.alert_text1 = "Camera Malfunction";
      scene.alert_text2 = "Contact Support";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      scene.status = STATUS_DISENGAGED;
      s->sound->stop();
    }
  }
}
