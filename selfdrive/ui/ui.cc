#include <iostream>
#include <stdio.h>
#include <cmath>
#include <unistd.h>
#include <assert.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "common/visionimg.h"
#include "ui.hpp"
#include "paint.hpp"


int write_param_float(float param, const char* param_name, bool persistent_param) {
  char s[16];
  int size = snprintf(s, sizeof(s), "%f", param);
  return Params(persistent_param).write_db_value(param_name, s, size < sizeof(s) ? size : sizeof(s));
}

// Projects a point in car to space to the corresponding point in full frame
// image space.
static bool calib_frame_to_full_frame(const UIState *s, float in_x, float in_y, float in_z, vertex_data *out) {
  const float margin = 500.0f;
  const vec3 pt = (vec3){{in_x, in_y, in_z}};
  const vec3 Ep = matvecmul3(s->scene.view_from_calib, pt);
  const vec3 KEp = matvecmul3(fcam_intrinsic_matrix, Ep);

  // Project.
  float x = KEp.v[0] / KEp.v[2];
  float y = KEp.v[1] / KEp.v[2];

  nvgTransformPoint(&out->x, &out->y, s->car_space_transform, x, y);
  return out->x >= -margin && out->x <= s->fb_w + margin && out->y >= -margin && out->y <= s->fb_h + margin;
}

static void ui_init_vision(UIState *s) {
  // Invisible until we receive a calibration message.
  s->scene.world_objects_visible = false;

  for (int i = 0; i < s->vipc_client->num_buffers; i++) {
    s->texture[i].reset(new EGLImageTexture(&s->vipc_client->buffers[i]));

    glBindTexture(GL_TEXTURE_2D, s->texture[i]->frame_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
  }
  assert(glGetError() == GL_NO_ERROR);
}


void ui_init(UIState *s) {
  s->sm = new SubMaster({"modelV2", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "deviceState", "liveLocationKalman",
                         "pandaState", "carParams", "driverState", "driverMonitoringState", "sensorEvents", "carState", "ubloxGnss"});

  s->scene.started = false;
  s->status = STATUS_OFFROAD;

  s->fb = std::make_unique<FrameBuffer>("ui", 0, true, &s->fb_w, &s->fb_h);

  ui_nvg_init(s);

  s->last_frame = nullptr;
  s->vipc_client_rear = new VisionIpcClient("camerad", VISION_STREAM_RGB_BACK, true);
  s->vipc_client_front = new VisionIpcClient("camerad", VISION_STREAM_RGB_FRONT, true);
  s->vipc_client = s->vipc_client_rear;
}

static int get_path_length_idx(const cereal::ModelDataV2::XYZTData::Reader &line, const float path_height) {
  const auto line_x = line.getX();
  int max_idx = 0;
  for (int i = 0; i < TRAJECTORY_SIZE && line_x[i] < path_height; ++i) {
    max_idx = i;
  }
  return max_idx;
}

static void update_lead(UIState *s, const cereal::RadarState::Reader &radar_state,
                        const cereal::ModelDataV2::XYZTData::Reader &line, int idx) {
  auto &lead_data = s->scene.lead_data[idx];
  lead_data = (idx == 0) ? radar_state.getLeadOne() : radar_state.getLeadTwo();
  if (lead_data.getStatus()) {
    const int path_idx = get_path_length_idx(line, lead_data.getDRel());
    // negative because radarState uses left positive convention
    calib_frame_to_full_frame(s, lead_data.getDRel(), -lead_data.getYRel(), line.getZ()[path_idx] + 1.22, &s->scene.lead_vertices[idx]);
  }
}

static void update_line_data(const UIState *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                             float y_off, float z_off, line_vertices_data *pvd, float max_distance) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  int max_idx = -1;
  vertex_data *v = &pvd->v[0];
  for (int i = 0; ((i < TRAJECTORY_SIZE) and (line_x[i] < fmax(MIN_DRAW_DISTANCE, max_distance))); i++) {
    v += calib_frame_to_full_frame(s, line_x[i], line_y[i] - y_off, line_z[i] + z_off, v);
    max_idx = i;
  }
  for (int i = max_idx; i >= 0; i--) {
    v += calib_frame_to_full_frame(s, line_x[i], line_y[i] + y_off, line_z[i] + z_off, v);
  }
  pvd->cnt = v - pvd->v;
  assert(pvd->cnt < std::size(pvd->v));
}

static void update_model(UIState *s, const cereal::ModelDataV2::Reader &model) {
  UIScene &scene = s->scene;
  const float max_distance = fmin(model.getPosition().getX()[TRAJECTORY_SIZE - 1], MAX_DRAW_DISTANCE);
  // update lane lines
  const auto lane_lines = model.getLaneLines();
  const auto lane_line_probs = model.getLaneLineProbs();
  for (int i = 0; i < std::size(scene.lane_line_vertices); i++) {
    scene.lane_line_probs[i] = lane_line_probs[i];
    update_line_data(s, lane_lines[i], 0.025 * scene.lane_line_probs[i], 0, &scene.lane_line_vertices[i], max_distance);
  }

  // update road edges
  const auto road_edges = model.getRoadEdges();
  const auto road_edge_stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(scene.road_edge_vertices); i++) {
    scene.road_edge_stds[i] = road_edge_stds[i];
    update_line_data(s, road_edges[i], 0.025, 0, &scene.road_edge_vertices[i], max_distance);
  }

  // update path
  const float lead_d = scene.lead_data[0].getStatus() ? scene.lead_data[0].getDRel() * 2. : MAX_DRAW_DISTANCE;
  float path_length = (lead_d > 0.) ? lead_d - fmin(lead_d * 0.35, 10.) : MAX_DRAW_DISTANCE;
  path_length = fmin(path_length, max_distance);
  update_line_data(s, model.getPosition(), 0.5, 1.22, &scene.track_vertices, path_length);
}

static void update_sockets(UIState *s) {
  SubMaster &sm = *(s->sm);
  if (sm.update(0) == 0) return;

  UIScene &scene = s->scene;
  if (scene.started && sm.updated("controlsState")) {
    scene.controls_state = sm["controlsState"].getControlsState();
  }
  if (sm.updated("carState")) {
    scene.car_state = sm["carState"].getCarState();
  }
  if (sm.updated("radarState")) {
    auto radar_state = sm["radarState"].getRadarState();
    const auto line = sm["modelV2"].getModelV2().getPosition();
    update_lead(s, radar_state, line, 0);
    update_lead(s, radar_state, line, 1);
  }
  if (sm.updated("liveCalibration")) {
    scene.world_objects_visible = true;
    auto rpy_list = sm["liveCalibration"].getLiveCalibration().getRpyCalib();
    Eigen::Vector3d rpy;
    rpy << rpy_list[0], rpy_list[1], rpy_list[2];
    Eigen::Matrix3d device_from_calib = euler2rot(rpy);
    Eigen::Matrix3d view_from_device;
    view_from_device << 0,1,0,
                        0,0,1,
                        1,0,0;
    Eigen::Matrix3d view_from_calib = view_from_device * device_from_calib;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        scene.view_from_calib.v[i*3 + j] = view_from_calib(i,j);
      }
    }
  }
  if (sm.updated("modelV2")) {
    update_model(s, sm["modelV2"].getModelV2());
  }
  if (sm.updated("uiLayoutState")) {
    auto data = sm["uiLayoutState"].getUiLayoutState();
    s->active_app = data.getActiveApp();
    s->sidebar_collapsed = data.getSidebarCollapsed();
  }
  if (sm.updated("deviceState")) {
    scene.deviceState = sm["deviceState"].getDeviceState();
  }
  if (sm.updated("pandaState")) {
    auto pandaState = sm["pandaState"].getPandaState();
    scene.pandaType = pandaState.getPandaType();
    scene.ignition = pandaState.getIgnitionLine() || pandaState.getIgnitionCan();
  } else if ((s->sm->frame - s->sm->rcv_frame("pandaState")) > 5*UI_FREQ) {
    scene.pandaType = cereal::PandaState::PandaType::UNKNOWN;
  }
  if (sm.updated("ubloxGnss")) {
    auto data = sm["ubloxGnss"].getUbloxGnss();
    if (data.which() == cereal::UbloxGnss::MEASUREMENT_REPORT) {
      scene.satelliteCount = data.getMeasurementReport().getNumMeas();
    }
  }
  if (sm.updated("liveLocationKalman")) {
    scene.gpsOK = sm["liveLocationKalman"].getLiveLocationKalman().getGpsOK();
  }
  if (sm.updated("carParams")) {
    scene.longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }
  if (sm.updated("driverState")) {
    scene.driver_state = sm["driverState"].getDriverState();
  }
  if (sm.updated("driverMonitoringState")) {
    scene.dmonitoring_state = sm["driverMonitoringState"].getDriverMonitoringState();
    if(!scene.driver_view && !scene.ignition) {
      read_param(&scene.driver_view, "IsDriverViewEnabled");
    }
  } else if ((sm.frame - sm.rcv_frame("driverMonitoringState")) > UI_FREQ/2) {
    scene.driver_view = false;
  }
  if (sm.updated("sensorEvents")) {
    for (auto sensor : sm["sensorEvents"].getSensorEvents()) {
      if (sensor.which() == cereal::SensorEventData::LIGHT) {
        scene.light_sensor = sensor.getLight();
      } else if (!scene.started && sensor.which() == cereal::SensorEventData::ACCELERATION) {
        auto accel = sensor.getAcceleration().getV();
        if (accel.totalSize().wordCount){ // TODO: sometimes empty lists are received. Figure out why
          scene.accel_sensor = accel[2];
        }
      } else if (!scene.started && sensor.which() == cereal::SensorEventData::GYRO_UNCALIBRATED) {
        auto gyro = sensor.getGyroUncalibrated().getV();
        if (gyro.totalSize().wordCount){
          scene.gyro_sensor = gyro[1];
        }
      }
    }
  }
  scene.started = scene.deviceState.getStarted() || scene.driver_view;
}

static void update_alert(UIState *s) {
  UIScene &scene = s->scene;
  if (s->sm->updated("controlsState")) {
    auto alert_sound = scene.controls_state.getAlertSound();
    if (scene.alert_type.compare(scene.controls_state.getAlertType()) != 0) {
      if (alert_sound == AudibleAlert::NONE) {
        s->sound->stop();
      } else {
        s->sound->play(alert_sound);
      }
    }
    scene.alert_text1 = scene.controls_state.getAlertText1();
    scene.alert_text2 = scene.controls_state.getAlertText2();
    scene.alert_size = scene.controls_state.getAlertSize();
    scene.alert_type = scene.controls_state.getAlertType();
    scene.alert_blinking_rate = scene.controls_state.getAlertBlinkingRate();
  }

  // Handle controls timeout
  if (scene.deviceState.getStarted() && (s->sm->frame - scene.started_frame) > 10 * UI_FREQ) {
    const uint64_t cs_frame = s->sm->rcv_frame("controlsState");
    if (cs_frame < scene.started_frame) {
      // car is started, but controlsState hasn't been seen at all
      scene.alert_text1 = "openpilot Unavailable";
      scene.alert_text2 = "Waiting for controls to start";
      scene.alert_size = cereal::ControlsState::AlertSize::MID;
    } else if ((s->sm->frame - cs_frame) > 5 * UI_FREQ) {
      // car is started, but controls is lagging or died
      if (scene.alert_text2 != "Controls Unresponsive") {
        s->sound->play(AudibleAlert::CHIME_WARNING_REPEAT);
        LOGE("Controls unresponsive");
      }

      scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      scene.alert_text2 = "Controls Unresponsive";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      s->status = STATUS_ALERT;
    }
  }
}

static void update_params(UIState *s) {
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

static void update_vision(UIState *s) {
  if (!s->vipc_client->connected && s->scene.started) {
    if (s->vipc_client->connect(false)){
      ui_init_vision(s);
    }
  }

  if (s->vipc_client->connected){
    VisionBuf * buf = s->vipc_client->recv();
    if (buf != nullptr){
      s->last_frame = buf;
    } else {
#if defined(QCOM) || defined(QCOM2)
      LOGE("visionIPC receive timeout");
#endif
    }
  }
}

static void update_status(UIState *s) {
  if (s->scene.started && s->sm->updated("controlsState")) {
    auto alert_status = s->scene.controls_state.getAlertStatus();
    if (alert_status == cereal::ControlsState::AlertStatus::USER_PROMPT) {
      s->status = STATUS_WARNING;
    } else if (alert_status == cereal::ControlsState::AlertStatus::CRITICAL) {
      s->status = STATUS_ALERT;
    } else {
      s->status = s->scene.controls_state.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
    }
  }

  // Handle onroad/offroad transition
  static bool started_prev = false;
  if (s->scene.started != started_prev) {
    if (s->scene.started) {
      s->status = STATUS_DISENGAGED;
      s->scene.started_frame = s->sm->frame;

      read_param(&s->scene.is_rhd, "IsRHD");
      s->active_app = cereal::UiLayoutState::App::NONE;
      s->sidebar_collapsed = true;
      s->scene.alert_size = cereal::ControlsState::AlertSize::NONE;
      s->vipc_client = s->scene.driver_view ? s->vipc_client_front : s->vipc_client_rear;
    } else {
      s->status = STATUS_OFFROAD;
      s->active_app = cereal::UiLayoutState::App::HOME;
      s->sidebar_collapsed = false;
      s->sound->stop();
      s->vipc_client->connected = false;
    }
  }
  started_prev = s->scene.started;
}

void ui_update(UIState *s) {
  update_params(s);
  update_sockets(s);
  update_status(s);
  update_alert(s);
  update_vision(s);
}
