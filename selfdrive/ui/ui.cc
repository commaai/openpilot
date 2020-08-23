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
  pthread_mutex_init(&s->lock, NULL);
  s->sm = new SubMaster({"model", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal",
                         "health", "carParams", "ubloxGnss", "driverState", "dMonitoringState"
  });

  s->scene.satelliteCount = -1;
  s->started = false;
  s->vision_seen = false;

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

  read_param(&s->speed_lim_off, "SpeedLimitOffset");
  read_param(&s->is_metric, "IsMetric");
  read_param(&s->longitudinal_control, "LongitudinalControl");
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

void handle_message(UIState *s, SubMaster &sm) {
  UIScene &scene = s->scene;
  if (s->started && sm.updated("controlsState")) {
    auto event = sm["controlsState"];
    scene.controls_state = event.getControlsState();
    s->controls_seen = true;

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
  // else if (which == cereal::Event::LIVE_MPC) {
  //   auto data = event.getLiveMpc();
  //   auto x_list = data.getX();
  //   auto y_list = data.getY();
  //   for (int i = 0; i < 50; i++){
  //     scene.mpc_x[i] = x_list[i];
  //     scene.mpc_y[i] = y_list[i];
  //   }
  //   s->livempc_or_radarstate_changed = true;
  // }
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
  // Handle onroad/offroad transition
  if (!s->started) {
    if (s->status != STATUS_STOPPED) {
      update_status(s, STATUS_STOPPED);
      s->vision_seen = false;
      s->controls_seen = false;
      s->active_app = cereal::UiLayoutState::App::HOME;
    }
  } else if (s->status == STATUS_STOPPED) {
    update_status(s, STATUS_DISENGAGED);
    s->active_app = cereal::UiLayoutState::App::NONE;
  }
}

void check_messages(UIState *s) {
  if (s->sm->update(0) > 0){
    handle_message(s, *(s->sm));
  }
}

void ui_update_sizes(UIState *s){
  // resize vision for collapsing sidebar
  const bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  s->scene.ui_viz_rx = hasSidebar ? box_x : (box_x - sbr_w + (bdr_s * 2));
  s->scene.ui_viz_rw = hasSidebar ? box_w : (box_w + sbr_w - (bdr_s * 2));
  s->scene.ui_viz_ro = hasSidebar ? -(sbr_w - 6 * bdr_s) : 0;
}

void ui_update(UIState *s) {
  if (s->vision_connect_firstrun) {
    // cant run this in connector thread because opengl.
    // do this here for now in lieu of a run_on_main_thread event

    for (int i=0; i<UI_BUF_COUNT; i++) {
      if(s->khr[i] != 0) {
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
    s->scene.ui_viz_rx = (box_x-sbr_w+bdr_s*2);
    s->scene.ui_viz_rw = (box_w+sbr_w-(bdr_s*2));
    s->scene.ui_viz_ro = 0;

    s->vision_connect_firstrun = false;

    s->alert_blinking_alpha = 1.0;
    s->alert_blinked = false;
  }
  
  VIPCBuf *buf = visionstream_get(&s->stream, nullptr);
  if (!buf) {
    visionstream_destroy(&s->stream);
    s->vision_connected = false;
  }
}

void* vision_connect_thread(void *args) {
  set_thread_name("vision_connect");

  UIState *s = (UIState *)args;
  while (!do_exit) {
    usleep(100000);
    pthread_mutex_lock(&s->lock);
    if (s->vision_connected) {
      if (!s->started) {
        visionstream_destroy(&s->stream);
        s->vision_connected = false;
      }
    } else {
      if (s->started) {
        const VisionStreamType type = s->scene.frontview ? VISION_STREAM_RGB_FRONT : VISION_STREAM_RGB_BACK;
        int err = visionstream_init(&s->stream, type, true, nullptr);
        if (err == 0) {
          ui_init_vision(s);

          s->vision_connected = true;
          s->vision_seen = true;
          s->vision_connect_firstrun = true;

          // Drain sockets
          s->sm->drain();
        } else {
          LOGW("visionstream connect failed");
        }
      }
    }
    pthread_mutex_unlock(&s->lock);
  }
  return NULL;
}
