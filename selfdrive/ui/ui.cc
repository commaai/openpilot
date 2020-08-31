#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <czmq.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "common/visionimg.h"
#include "common/utilpp.h"
#include "ui.hpp"
#include "paint.hpp"

extern volatile sig_atomic_t do_exit;

int write_param_float(float param, const char* param_name, bool persistent_param) {
  char s[16];
  int size = snprintf(s, sizeof(s), "%f", param);
  return write_db_value(param_name, s, MIN(size, sizeof(s)), persistent_param);
}

void ui_init(UIState *s) {
  s->sm = new SubMaster({"model", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal",
                         "health", "carParams", "ubloxGnss", "driverState", "dMonitoringState"
  });

  s->started = false;
  s->scene.satelliteCount = -1;
  read_param(&s->is_metric, "IsMetric");
  read_param(&s->longitudinal_control, "LongitudinalControl");

  s->fb = framebuffer_init("ui", 0, true, &s->fb_w, &s->fb_h);
  assert(s->fb);

  ui_nvg_init(s);
}

static void ui_init_vision(UIState *s) {
  if (!s->scene.frontview) {
    s->scene.frontview = getenv("FRONTVIEW") != NULL;
  }
  s->scene.fullview = getenv("FULLVIEW") != NULL;
  s->scene.world_objects_visible = false;  // Invisible until we receive a calibration message.

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

  s->scene.uilayout_sidebarcollapsed = true;
  s->alert_blinking_alpha = 1.0;
  s->alert_blinked = false;

  // TODO: shouldn't be here
  // Drain sockets
  s->sm->drain();
}

void ui_update_vision(UIState *s) {
  if (!s->vision_connected && s->started) {
    const VisionStreamType type = s->scene.frontview ? VISION_STREAM_RGB_FRONT : VISION_STREAM_RGB_BACK;
    int err = visionstream_init(&s->stream, type, true, nullptr);
    if (err == 0) {
      ui_init_vision(s);
      s->vision_connected = true;
    }
  }

  if (s->vision_connected && (!s->started || !visionstream_get(&s->stream, nullptr))) {
    visionstream_destroy(&s->stream);
    s->vision_connected = false;
  }
}


void update_status(UIState *s, int status) {
  if (s->status != status) {
    s->status = status;
  }
}

static inline void fill_path_points(const cereal::ModelData::PathData::Reader &path, float *points) {
  const capnp::List<float>::Reader &poly = path.getPoly();
  for (int i = 0; i < path.getValidLen(); i++) {
    points[i] = poly[0] * (i * i * i) + poly[1] * (i * i) + poly[2] * i + poly[3];
  }
}

void update_sockets(UIState *s) {

  // poll sockets
  if (s->sm->update(0) == 0){
    return;
  }

  UIScene &scene = s->scene;
  SubMaster &sm = *(s->sm);

  if (sm.updated("controlsState")) {
    auto event = sm["controlsState"];
    scene.controls_state = event.getControlsState();

    auto alert_sound = scene.controls_state.getAlertSound();
    if (scene.alert_type.compare(scene.controls_state.getAlertType()) != 0) {
      if (alert_sound == AudibleAlert::NONE) {
        s->sound.stop();
      } else {
        s->sound.play(alert_sound);
      }
    }
    scene.alert_text1 = scene.controls_state.getAlertText1();
    scene.alert_text2 = scene.controls_state.getAlertText2();
    scene.alert_size = scene.controls_state.getAlertSize();
    scene.alert_type = scene.controls_state.getAlertType();
    auto alertStatus = scene.controls_state.getAlertStatus();
    if (alertStatus == cereal::ControlsState::AlertStatus::USER_PROMPT) {
      update_status(s, STATUS_WARNING);
    } else if (alertStatus == cereal::ControlsState::AlertStatus::CRITICAL) {
      update_status(s, STATUS_ALERT);
    } else{
      update_status(s, scene.controls_state.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED);
    }

    float alert_blinkingrate = scene.controls_state.getAlertBlinkingRate();
    if (alert_blinkingrate > 0.) {
      if (s->alert_blinked) {
        if (s->alert_blinking_alpha > 0.0 && s->alert_blinking_alpha < 1.0) {
          s->alert_blinking_alpha += (0.05*alert_blinkingrate);
        } else {
          s->alert_blinked = false;
        }
      } else {
        if (s->alert_blinking_alpha > 0.25) {
          s->alert_blinking_alpha -= (0.05*alert_blinkingrate);
        } else {
          s->alert_blinking_alpha += 0.25;
          s->alert_blinked = true;
        }
      }
    }
  }
  if (sm.updated("radarState")) {
    auto data = sm["radarState"].getRadarState();
    scene.lead_data[0] = data.getLeadOne();
    scene.lead_data[1] = data.getLeadTwo();
  }
  if (sm.updated("liveCalibration")) {
    scene.world_objects_visible = true;
    auto extrinsicl = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
    for (int i = 0; i < 3 * 4; i++) {
      scene.extrinsic_matrix.v[i] = extrinsicl[i];
    }
  }
  if (sm.updated("model")) {
    scene.model = sm["model"].getModel();
    fill_path_points(scene.model.getPath(), scene.path_points);
    fill_path_points(scene.model.getLeftLane(), scene.left_lane_points);
    fill_path_points(scene.model.getRightLane(), scene.right_lane_points);
  }
  if (sm.updated("uiLayoutState")) {
    auto data = sm["uiLayoutState"].getUiLayoutState();
    s->active_app = data.getActiveApp();
    scene.uilayout_sidebarcollapsed = data.getSidebarCollapsed();
  }
  if (sm.updated("thermal")) {
    scene.thermal = sm["thermal"].getThermal();
  }
  if (sm.updated("ubloxGnss")) {
    auto data = sm["ubloxGnss"].getUbloxGnss();
    if (data.which() == cereal::UbloxGnss::MEASUREMENT_REPORT) {
      scene.satelliteCount = data.getMeasurementReport().getNumMeas();
    }
  }
  if (sm.updated("health")) {
    scene.hwType = sm["health"].getHealth().getHwType();
  } else if ((s->sm->frame - s->sm->rcv_frame("health")) > 5*UI_FREQ) {
    // manage hardware disconnect
    scene.hwType = cereal::HealthData::HwType::UNKNOWN;
  }

  if (sm.updated("carParams")) {
    s->longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }
  if (sm.updated("driverState")) {
    scene.driver_state = sm["driverState"].getDriverState();
  }
  if (sm.updated("dMonitoringState")) {
    scene.dmonitoring_state = sm["dMonitoringState"].getDMonitoringState();
    scene.is_rhd = scene.dmonitoring_state.getIsRHD();
    scene.frontview = scene.dmonitoring_state.getIsPreview();
  } else if ((sm.frame - sm.rcv_frame("dMonitoringState")) > 1*UI_FREQ) {
    scene.frontview = false;
  }

  s->started = scene.thermal.getStarted() || scene.frontview;
}

void ui_update_sizes(UIState *s){
  // resize vision for collapsing sidebar
  const bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  s->scene.ui_viz_rx = hasSidebar ? box_x : (box_x - sbr_w + (bdr_s * 2));
  s->scene.ui_viz_rw = hasSidebar ? box_w : (box_w + sbr_w - (bdr_s * 2));
  s->scene.ui_viz_ro = hasSidebar ? -(sbr_w - 6 * bdr_s) : 0;
}

void ui_update(UIState *s) {

  SubMaster &sm = *(s->sm);
  bool started_prev = s->started;
  update_sockets(s);
  if (s->started && !started_prev) {
    s->started_frame = sm.frame;
  }

  ui_update_sizes(s);
  ui_update_vision(s);

  // Handle onroad/offroad transition
  if (!s->started) {
    if (s->status != STATUS_STOPPED) {
      update_status(s, STATUS_STOPPED);
      s->active_app = cereal::UiLayoutState::App::HOME;
    }
  } else if (s->status == STATUS_STOPPED) {
    update_status(s, STATUS_DISENGAGED);
    s->active_app = cereal::UiLayoutState::App::NONE;
  }

  // Handle controls timeout
  bool controls_timeout = (sm.frame - sm.rcv_frame("controlsState")) > 5*UI_FREQ;
  if (s->started && !s->scene.frontview && controls_timeout) {
    if (sm.rcv_frame("controlsState") < s->started_frame) {
      // car is started, but controlsState hasn't been seen at all
      s->scene.alert_text1 = "openpilot Unavailable";
      s->scene.alert_text2 = "Waiting for controls to start";
      s->scene.alert_size = cereal::ControlsState::AlertSize::MID;
    } else {
      // car is started, but controls is lagging or died
      LOGE("Controls unresponsive");

      if (s->scene.alert_text2 != "Controls Unresponsive") {
        s->sound.play(AudibleAlert::CHIME_WARNING_REPEAT);
      }

      s->scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      s->scene.alert_text2 = "Controls Unresponsive";
      s->scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      update_status(s, STATUS_ALERT);
    }
    ui_draw_vision_alert(s, s->scene.alert_size, s->status, s->scene.alert_text1.c_str(), s->scene.alert_text2.c_str());
  }

  // Sample params
  if (s->sm->frame % (5*UI_FREQ) == 0) {
    read_param(&s->is_metric, "IsMetric");
  } else if (s->sm->frame % (6*UI_FREQ) == 0) {
    int param_read = read_param(&s->last_athena_ping, "LastAthenaPingTime");
    if (param_read != 0) { // Failed to read param
      s->scene.athenaStatus = NET_DISCONNECTED;
    } else if (nanos_since_boot() - s->last_athena_ping < 70e9) {
      s->scene.athenaStatus = NET_CONNECTED;
    } else {
      s->scene.athenaStatus = NET_ERROR;
    }
  }

}

