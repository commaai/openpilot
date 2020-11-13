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

void ui_update(UIState *s) {
  s->state_thread.getScene(&s->scene);
  ui_update_vision(s);
}


// UIStateThread

UIStateThread::UIStateThread(UIState *s) : ui_state(s),
      sm({"modelV2", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal", "frame",
          "health", "carParams", "ubloxGnss", "driverState", "dMonitoringState", "sensorEvents"}) {
  thread = std::thread(&UIStateThread::threadMain, this);
}

UIStateThread::~UIStateThread() {
  run = false;
  thread.join();
}

// Projects a point in car to space to the corresponding point in full frame
// image space.
bool UIStateThread::car_space_to_full_frame(float in_x, float in_y, float in_z, float *out_x, float *out_y, float margin) {
  const vec4 car_space_projective = (vec4){{in_x, in_y, in_z, 1.}};
  // We'll call the car space point p.
  // First project into normalized image coordinates with the extrinsics matrix.
  const vec4 Ep4 = matvecmul(extrinsic_matrix, car_space_projective);

  // The last entry is zero because of how we store E (to use matvecmul).
  const vec3 Ep = {{Ep4.v[0], Ep4.v[1], Ep4.v[2]}};
  const vec3 KEp = matvecmul3(intrinsic_matrix, Ep);

  // Project.
  *out_x = KEp.v[0] / KEp.v[2];
  *out_y = KEp.v[1] / KEp.v[2];

  return *out_x >= -margin && *out_x <= ui_state->fb_w + margin && *out_y >= -margin && *out_y <= ui_state->fb_h + margin;
}

template <class T>
static void update_line_data(UIStateThread *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                             float y_off, float z_off, T *pvd, float max_distance) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  int max_idx = -1;
  vertex_data *v = &pvd->v[0];
  const float margin = 500.0f;
  for (int i = 0; ((i < TRAJECTORY_SIZE) and (line_x[i] < fmax(MIN_DRAW_DISTANCE, max_distance))); i++) {
    v += s->car_space_to_full_frame(line_x[i], -line_y[i] - y_off, -line_z[i] + z_off, &v->x, &v->y, margin);
    max_idx = i;
  }
  for (int i = max_idx; i >= 0; i--) {
    v += s->car_space_to_full_frame(line_x[i], -line_y[i] + y_off, -line_z[i] + z_off, &v->x, &v->y, margin);
  }
  pvd->cnt = v - pvd->v;
  assert(pvd->cnt < std::size(pvd->v));
}

void UIStateThread::update_model(const cereal::ModelDataV2::Reader &model) {
  const float max_distance = fmin(model.getPosition().getX()[TRAJECTORY_SIZE - 1], MAX_DRAW_DISTANCE);
  // update lane lines
  const auto lane_lines = model.getLaneLines();
  const auto lane_line_probs = model.getLaneLineProbs();
  for (int i = 0; i < std::size(scene.lane_line_vertices); i++) {
    scene.lane_line_probs[i] = lane_line_probs[i];
    update_line_data(this, lane_lines[i], 0.025 * scene.lane_line_probs[i], 1.22, &scene.lane_line_vertices[i], max_distance);
  }

  // update road edges
  const auto road_edges = model.getRoadEdges();
  const auto road_edge_stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(scene.road_edge_vertices); i++) {
    scene.road_edge_stds[i] = road_edge_stds[i];
    update_line_data(this, road_edges[i], 0.025, 1.22, &scene.road_edge_vertices[i], max_distance);
  }

  // update path
  const float lead_d = scene.lead[0].status ? scene.lead[0].d_rel * 2. : MAX_DRAW_DISTANCE;
  float path_length = (lead_d > 0.) ? lead_d - fmin(lead_d * 0.35, 10.) : MAX_DRAW_DISTANCE;
  path_length = fmin(path_length, max_distance);
  update_line_data(this, model.getPosition(), 0.5, 0, &scene.track_vertices, path_length);
}

static void update_lead(UIStateThread *s, const cereal::RadarState::LeadData::Reader &lead,
                        const cereal::ModelDataV2::XYZTData::Reader &line, UIScene::LeadData &lead_data) {
  lead_data = {.status = lead.getStatus(),
               .d_rel = lead.getDRel(),
               .v_rel = lead.getVRel(),
               .y_rel = lead.getYRel()};

  if (float z = 0.; lead_data.status) {
    const float path_length = fmin(lead_data.d_rel, MAX_DRAW_DISTANCE);
    const auto line_x = line.getX(), line_z = line.getZ();
    for (int i = 0; i < TRAJECTORY_SIZE && line_x[i] < path_length; ++i) {
      z = line_z[i];
    }
    s->car_space_to_full_frame(lead_data.d_rel, lead_data.y_rel, z, &lead_data.vd.x, &lead_data.vd.y, 500.0f);
  }
}

void UIStateThread::updateSockets() {
  if (sm.update(0) == 0) return;

  if (scene.started && sm.updated("controlsState")) {
    const auto cs = sm["controlsState"].getControlsState();
    scene.v_cruise = cs.getVCruise();
    scene.v_ego = cs.getVEgo();
    scene.decel_for_model = cs.getDecelForModel();
    scene.controls_enabled = cs.getEnabled();
    scene.engageable = cs.getEngageable();

    // TODO: the alert stuff shouldn't be handled here
    auto alert_sound = cs.getAlertSound();
    if (scene.alert_type.compare(cs.getAlertType()) != 0) {
      if (alert_sound == AudibleAlert::NONE) {
        ui_state->sound->stop();
      } else {
        ui_state->sound->play(alert_sound);
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
  }
   if (sm.updated("radarState")) {
    const auto radar_state = sm["radarState"].getRadarState();
    const auto line = sm["modelV2"].getModelV2().getPosition();
    update_lead(this, radar_state.getLeadOne(), line, scene.lead[0]);
    update_lead(this, radar_state.getLeadTwo(), line, scene.lead[1]);
  }
  if (sm.updated("liveCalibration")) {
    ui_state->world_objects_visible = true;
    const auto extrinsicl = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
    for (int i = 0; i < 3 * 4; ++i) {
      extrinsic_matrix.v[i] = extrinsicl[i];
    }
  }
  if (sm.updated("modelV2")) {
    update_model(sm["modelV2"].getModelV2());
  }
  if (sm.updated("uiLayoutState")) {
    auto data = sm["uiLayoutState"].getUiLayoutState();
    ui_state->active_app = data.getActiveApp();
    ui_state->sidebar_collapsed = data.getSidebarCollapsed();
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
  } else if ((sm.frame - sm.rcv_frame("health")) > 5*UI_FREQ) {
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
    const auto state = sm["dMonitoringState"].getDMonitoringState();
    scene.is_rhd = state.getIsRHD();
    scene.frontview = state.getIsPreview();
    scene.face_detected = state.getFaceDetected();
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

void UIStateThread::threadMain() {
  while (run) {
    update();

    std::unique_lock<std::mutex> lk(lock);
    state_type = UIStateSync::kReady;
    cv.notify_one();
    cv.wait(lk, [&] { return state_type == UIStateSync::kFetch || !run; });
  }
}

void UIStateThread::getScene(UIScene *scene) {
  { // copy state to UI thread
    std::unique_lock<std::mutex> lk(lock);
    cv.wait(lk, [&] { return state_type == UIStateSync::kReady; });
    *scene = this->scene;
    state_type = UIStateSync::kFetch;
  }
  cv.notify_one();
}

void UIStateThread::update_params() {
  const uint64_t frame = sm.frame;

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

void UIStateThread::update() {
  update_params();
  updateSockets();
  
  // Handle onroad/offroad transition
  if (!scene.started && scene.status != STATUS_OFFROAD) {
    scene.status = STATUS_OFFROAD;
    ui_state->active_app = cereal::UiLayoutState::App::HOME;
    ui_state->sidebar_collapsed = false;
    ui_state->sound->stop();
  } else if (scene.started && scene.status == STATUS_OFFROAD) {
    scene.status = STATUS_DISENGAGED;
    started_frame = sm.frame;

    ui_state->active_app = cereal::UiLayoutState::App::NONE;
    ui_state->sidebar_collapsed = true;
    scene.alert_size = cereal::ControlsState::AlertSize::NONE;
  }

  // Handle controls/fcamera timeout
  if (scene.started && !scene.frontview && (sm.frame - started_frame) > 10*UI_FREQ) {
    if (sm.rcv_frame("controlsState") < started_frame) {
      // car is started, but controlsState hasn't been seen at all
      scene.alert_text1 = "openpilot Unavailable";
      scene.alert_text2 = "Waiting for controls to start";
      scene.alert_size = cereal::ControlsState::AlertSize::MID;
    } else if ((sm.frame - sm.rcv_frame("controlsState")) > 5*UI_FREQ) {
      // car is started, but controls is lagging or died
      if (scene.alert_text2 != "Controls Unresponsive" &&
          scene.alert_text1 != "Camera Malfunction") {
        ui_state->sound->play(AudibleAlert::CHIME_WARNING_REPEAT);
        LOGE("Controls unresponsive");
      }

      scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      scene.alert_text2 = "Controls Unresponsive";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      scene.status = STATUS_ALERT;
    }

    const uint64_t frame_pkt = sm.rcv_frame("frame");
    const uint64_t frame_delayed = sm.frame - frame_pkt;
    const uint64_t since_started = sm.frame - started_frame;
    if ((frame_pkt > started_frame || since_started > 15*UI_FREQ) && frame_delayed > 5*UI_FREQ) {
      // controls is fine, but rear camera is lagging or died
      scene.alert_text1 = "Camera Malfunction";
      scene.alert_text2 = "Contact Support";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      scene.status = STATUS_DISENGAGED;
      ui_state->sound->stop();
   }
  }
}
