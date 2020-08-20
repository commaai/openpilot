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
                         "health", "ubloxGnss", "driverState", "dMonitoringState"
#ifdef SHOW_SPEEDLIMIT
                                    , "liveMapData"
#endif
  });
  s->pm = new PubMaster({"offroadLayout"});

  s->ipc_fd = -1;
  s->scene.satelliteCount = -1;
  s->started = false;
  s->vision_seen = false;

  s->fb = framebuffer_init("ui", 0, true, &s->fb_w, &s->fb_h);
  assert(s->fb);

  ui_nvg_init(s);
}

static void ui_init_vision(UIState *s, const VisionStreamBufs back_bufs,
                           int num_back_fds, const int *back_fds,
                           const VisionStreamBufs front_bufs, int num_front_fds,
                           const int *front_fds) {
  assert(num_back_fds == UI_BUF_COUNT);
  assert(num_front_fds == UI_BUF_COUNT);

  vipc_bufs_load(s->bufs, &back_bufs, num_back_fds, back_fds);
  vipc_bufs_load(s->front_bufs, &front_bufs, num_front_fds, front_fds);

  s->cur_vision_idx = -1;
  s->cur_vision_front_idx = -1;

  s->scene.frontview = getenv("FRONTVIEW") != NULL;
  s->scene.fullview = getenv("FULLVIEW") != NULL;
  s->scene.world_objects_visible = false;  // Invisible until we receive a calibration message.

  s->rgb_width = back_bufs.width;
  s->rgb_height = back_bufs.height;
  s->rgb_stride = back_bufs.stride;
  s->rgb_buf_len = back_bufs.buf_len;

  s->rgb_front_width = front_bufs.width;
  s->rgb_front_height = front_bufs.height;
  s->rgb_front_stride = front_bufs.stride;
  s->rgb_front_buf_len = front_bufs.buf_len;

  read_param(&s->speed_lim_off, "SpeedLimitOffset");
  read_param(&s->is_metric, "IsMetric");
  read_param(&s->longitudinal_control, "LongitudinalControl");
  read_param(&s->limit_set_speed, "LimitSetSpeed");

  // Set offsets so params don't get read at the same time
  s->longitudinal_control_timeout = UI_FREQ / 3;
  s->is_metric_timeout = UI_FREQ / 2;
  s->limit_set_speed_timeout = UI_FREQ;
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
    s->controls_timeout = 1 * UI_FREQ;
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
#ifdef SHOW_SPEEDLIMIT
  if (sm.updated("liveMapData")) {
    scene.map_valid = sm["liveMapData"].getLiveMapData().getMapValid();
  }
#endif
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
    s->hardware_timeout = 5*UI_FREQ; // 5 seconds
  }
  if (sm.updated("driverState")) {
    scene.driver_state = sm["driverState"].getDriverState();
  }
  if (sm.updated("dMonitoringState")) {
    scene.dmonitoring_state = sm["dMonitoringState"].getDMonitoringState();
    scene.is_rhd = scene.dmonitoring_state.getIsRHD();
    scene.frontview = scene.dmonitoring_state.getIsPreview();
  }

  // timeout on frontview
  if ((sm.frame - sm.rcv_frame("dMonitoringState")) > 1*UI_FREQ) {
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

      #ifndef QCOM
      // disconnect from visionipc on PC
      close(s->ipc_fd);
      s->ipc_fd = -1;
      #endif
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
  int err;

  if (s->vision_connect_firstrun) {
    // cant run this in connector thread because opengl.
    // do this here for now in lieu of a run_on_main_thread event

    for (int i=0; i<UI_BUF_COUNT; i++) {
      if(s->khr[i] != 0) {
        visionimg_destroy_gl(s->khr[i], s->priv_hnds[i]);
        glDeleteTextures(1, &s->frame_texs[i]);
      }

      VisionImg img = {
        .fd = s->bufs[i].fd,
        .format = VISIONIMG_FORMAT_RGB24,
        .width = s->rgb_width,
        .height = s->rgb_height,
        .stride = s->rgb_stride,
        .bpp = 3,
        .size = s->rgb_buf_len,
      };
      #ifndef QCOM
        s->priv_hnds[i] = s->bufs[i].addr;
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

    for (int i=0; i<UI_BUF_COUNT; i++) {
      if(s->khr_front[i] != 0) {
        visionimg_destroy_gl(s->khr_front[i], s->priv_hnds_front[i]);
        glDeleteTextures(1, &s->frame_front_texs[i]);
      }

      VisionImg img = {
        .fd = s->front_bufs[i].fd,
        .format = VISIONIMG_FORMAT_RGB24,
        .width = s->rgb_front_width,
        .height = s->rgb_front_height,
        .stride = s->rgb_front_stride,
        .bpp = 3,
        .size = s->rgb_front_buf_len,
      };
      #ifndef QCOM
        s->priv_hnds_front[i] = s->bufs[i].addr;
      #endif
      s->frame_front_texs[i] = visionimg_to_gl(&img, &s->khr_front[i], &s->priv_hnds_front[i]);

      glBindTexture(GL_TEXTURE_2D, s->frame_front_texs[i]);
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

  zmq_pollitem_t polls[1] = {{0}};
  // Take an rgb image from visiond if there is one
  assert(s->ipc_fd >= 0);
  while(true) {
    if (s->ipc_fd < 0) {
      // TODO: rethink this, for now it should only trigger on PC
      LOGW("vision disconnected by other thread");
      s->vision_connected = false;
      return;
    }
    polls[0].fd = s->ipc_fd;
    polls[0].events = ZMQ_POLLIN;
    #ifdef UI_60FPS
      // uses more CPU in both UI and surfaceflinger
      // 16% / 21%
      int ret = zmq_poll(polls, 1, 1);
    #else
      // 9% / 13% = a 14% savings
      int ret = zmq_poll(polls, 1, 1000);
    #endif
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;

      LOGE("poll failed (%d - %d)", ret, errno);
      close(s->ipc_fd);
      s->ipc_fd = -1;
      s->vision_connected = false;
      return;
    } else if (ret == 0) {
      break;
    }
    // vision ipc event
    VisionPacket rp;
    err = vipc_recv(s->ipc_fd, &rp);
    if (err <= 0) {
      LOGW("vision disconnected");
      close(s->ipc_fd);
      s->ipc_fd = -1;
      s->vision_connected = false;
      return;
    }
    if (rp.type == VIPC_STREAM_ACQUIRE) {
      bool front = rp.d.stream_acq.type == VISION_STREAM_RGB_FRONT;
      int idx = rp.d.stream_acq.idx;

      int release_idx;
      if (front) {
        release_idx = s->cur_vision_front_idx;
      } else {
        release_idx = s->cur_vision_idx;
      }
      if (release_idx >= 0) {
        VisionPacket rep = {
          .type = VIPC_STREAM_RELEASE,
          .d = { .stream_rel = {
            .type = rp.d.stream_acq.type,
            .idx = release_idx,
          }},
        };
        vipc_send(s->ipc_fd, &rep);
      }

      if (front) {
        assert(idx < UI_BUF_COUNT);
        s->cur_vision_front_idx = idx;
      } else {
        assert(idx < UI_BUF_COUNT);
        s->cur_vision_idx = idx;
        // printf("v %d\n", ((uint8_t*)s->bufs[idx].addr)[0]);
      }
    } else {
      assert(false);
    }
    break;
  }
}

static int vision_subscribe(int fd, VisionPacket *rp, VisionStreamType type) {
  int err;
  LOGW("vision_subscribe type:%d", type);

  VisionPacket p1 = {
    .type = VIPC_STREAM_SUBSCRIBE,
    .d = { .stream_sub = { .type = type, .tbuffer = true, }, },
  };
  err = vipc_send(fd, &p1);
  if (err < 0) {
    close(fd);
    return 0;
  }

  do {
    err = vipc_recv(fd, rp);
    if (err <= 0) {
      close(fd);
      return 0;
    }

    // release what we aren't ready for yet
    if (rp->type == VIPC_STREAM_ACQUIRE) {
      VisionPacket rep = {
        .type = VIPC_STREAM_RELEASE,
        .d = { .stream_rel = {
          .type = rp->d.stream_acq.type,
          .idx = rp->d.stream_acq.idx,
        }},
      };
      vipc_send(fd, &rep);
    }
  } while (rp->type != VIPC_STREAM_BUFS || rp->d.stream_bufs.type != type);

  return 1;
}

void* vision_connect_thread(void *args) {
  set_thread_name("vision_connect");

  UIState *s = (UIState*)args;
  while (!do_exit) {
    usleep(100000);
    pthread_mutex_lock(&s->lock);
    bool connected = s->vision_connected;
    pthread_mutex_unlock(&s->lock);
    if (connected) continue;

    int fd = vipc_connect();
    if (fd < 0) continue;

    VisionPacket back_rp, front_rp;
    if (!vision_subscribe(fd, &back_rp, VISION_STREAM_RGB_BACK)) continue;
    if (!vision_subscribe(fd, &front_rp, VISION_STREAM_RGB_FRONT)) continue;

    pthread_mutex_lock(&s->lock);
    assert(!s->vision_connected);
    s->ipc_fd = fd;

    ui_init_vision(s,
                   back_rp.d.stream_bufs, back_rp.num_fds, back_rp.fds,
                   front_rp.d.stream_bufs, front_rp.num_fds, front_rp.fds);

    s->vision_connected = true;
    s->vision_seen = true;
    s->vision_connect_firstrun = true;

    // Drain sockets
    s->sm->drain();

    pthread_mutex_unlock(&s->lock);
  }
  return NULL;
}
