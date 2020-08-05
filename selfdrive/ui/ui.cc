#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <string>
#include <sstream>
#include <sys/resource.h>
#include <czmq.h>
#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "common/touch.h"
#include "common/params.h"
#include "common/utilpp.h"
#include "ui.hpp"

static void ui_set_brightness(UIState *s, int brightness) {
  static int last_brightness = -1;
  if (last_brightness != brightness && (s->awake || brightness == 0)) {
    if (set_brightness(brightness)) {
      last_brightness = brightness;
    }
  }
}

int event_processing_enabled = -1;
static void enable_event_processing(bool yes) {
  if (event_processing_enabled != 1 && yes) {
    system("service call window 18 i32 1");  // enable event processing
    event_processing_enabled = 1;
  } else if (event_processing_enabled != 0 && !yes) {
    system("service call window 18 i32 0");  // disable event processing
    event_processing_enabled = 0;
  }
}

static void set_awake(UIState *s, bool awake) {
#ifdef QCOM
  if (awake) {
    // 30 second timeout
    s->awake_timeout = 30*UI_FREQ;
  }
  if (s->awake != awake) {
    s->awake = awake;

    // TODO: replace command_awake and command_sleep with direct calls to android
    if (awake) {
      LOGW("awake normal");
      framebuffer_set_power(s->vision.fb, HWC_POWER_MODE_NORMAL);
      enable_event_processing(true);
    } else {
      LOGW("awake off");
      ui_set_brightness(s, 0);
      framebuffer_set_power(s->vision.fb, HWC_POWER_MODE_OFF);
      enable_event_processing(false);
    }
  }
#else
  // computer UI doesn't sleep
  s->awake = true;
#endif
}

static void update_offroad_layout_state(UIState *s) {
#ifdef QCOM
  static int timeout = 0;
  static bool prev_collapsed = false;
  static cereal::UiLayoutState::App prev_app = cereal::UiLayoutState::App::NONE;
  if (timeout > 0) {
    timeout--;
  }
  if (prev_collapsed != s->scene.uilayout_sidebarcollapsed || prev_app != s->active_app || timeout == 0) {
    capnp::MallocMessageBuilder msg;
    auto event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(nanos_since_boot());
    auto layout = event.initUiLayoutState();
    layout.setActiveApp(s->active_app);
    layout.setSidebarCollapsed(s->scene.uilayout_sidebarcollapsed);
    s->pm->send("offroadLayout", msg);
    LOGD("setting active app to %d with sidebar %d", (int)s->active_app, s->scene.uilayout_sidebarcollapsed);
    prev_collapsed = s->scene.uilayout_sidebarcollapsed;
    prev_app = s->active_app;
    timeout = 2 * UI_FREQ;
  }
#endif
}

static void handle_sidebar_touch(UIState *s, int touch_x, int touch_y) {
  if (!s->scene.uilayout_sidebarcollapsed && touch_x <= sbr_w) {
    if (touch_x >= settings_btn_x && touch_x < (settings_btn_x + settings_btn_w)
      && touch_y >= settings_btn_y && touch_y < (settings_btn_y + settings_btn_h)) {
      s->active_app = cereal::UiLayoutState::App::SETTINGS;
    }
    else if (touch_x >= home_btn_x && touch_x < (home_btn_x + home_btn_w)
      && touch_y >= home_btn_y && touch_y < (home_btn_y + home_btn_h)) {
      if (s->started) {
        s->active_app = cereal::UiLayoutState::App::NONE;
        s->scene.uilayout_sidebarcollapsed = true;
      } else {
        s->active_app = cereal::UiLayoutState::App::HOME;
      }
    }
  }
}

static void handle_vision_touch(UIState *s, int touch_x, int touch_y) {
  if (s->started && (touch_x >= s->scene.ui_viz_rx - bdr_s)
    && (s->active_app != cereal::UiLayoutState::App::SETTINGS)) {
      s->scene.uilayout_sidebarcollapsed = !s->scene.uilayout_sidebarcollapsed;
  }
}
volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

template <class T>
static int read_param(T* param, const char *param_name, bool persistent_param = false){
  T param_orig = *param;
  char *value;
  size_t sz;

  int result = read_db_value(param_name, &value, &sz, persistent_param);
  if (result == 0){
    std::string s = std::string(value, sz); // value is not null terminated
    free(value);

    // Parse result
    std::istringstream iss(s);
    iss >> *param;

    // Restore original value if parsing failed
    if (iss.fail()) {
      *param = param_orig;
      result = -1;
    }
  }
  return result;
}

template <class T>
static int read_param_timeout(T* param, const char* param_name, int* timeout, bool persistent_param = false) {
  int result = -1;
  if (*timeout > 0){
    (*timeout)--;
  } else {
    *timeout = 2 * UI_FREQ; // 0.5Hz
    result = read_param(param, param_name, persistent_param);
  }
  return result;
}

static int write_param_float(float param, const char* param_name, bool persistent_param = false) {
  char s[16];
  int size = snprintf(s, sizeof(s), "%f", param);
  return write_db_value(param_name, s, MIN(size, sizeof(s)), persistent_param);
}

static void ui_init(UIState *s) {

  s->sm = new SubMaster({"model", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal",
                         "health", "ubloxGnss", "dMonitoringState"
#ifdef SHOW_SPEEDLIMIT
                                    , "liveMapData"
#endif
  });
  s->pm = new PubMaster({"offroadLayout"});

  s->scene.satelliteCount = -1;
  s->started = false;

  // init display
  s->vision.init();

  s->scene.uilayout_sidebarcollapsed = false;
  s->scene.ui_viz_rx = box_x;
  s->scene.ui_viz_rw = box_w;
  s->scene.ui_viz_ro = 0;

  s->alert_blinking_alpha = 1.0;
  s->alert_blinked = false;

  set_awake(s, true);

  ui_nvg_init(s);

  read_param(&s->speed_lim_off, "SpeedLimitOffset");
  read_param(&s->is_metric, "IsMetric");
  read_param(&s->longitudinal_control, "LongitudinalControl");
  read_param(&s->limit_set_speed, "LimitSetSpeed");

  // Set offsets so params don't get read at the same time
  s->longitudinal_control_timeout = UI_FREQ / 3;
  s->is_metric_timeout = UI_FREQ / 2;
  s->limit_set_speed_timeout = UI_FREQ;
}

static void read_path(PathData& p, const cereal::ModelData::PathData::Reader &pathp) {
  p = {};

  p.prob = pathp.getProb();
  p.std = pathp.getStd();

  auto polyp = pathp.getPoly();
  for (int i = 0; i < POLYFIT_DEGREE; i++) {
    p.poly[i] = polyp[i];
  }

  // Compute points locations
  for (int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    p.points[i] = p.poly[0] * (i*i*i) + p.poly[1] * (i*i)+ p.poly[2] * i + p.poly[3];
  }

  p.validLen = pathp.getValidLen();
}

static void read_model(ModelData &d, const cereal::ModelData::Reader &model) {
  d = {};
  read_path(d.path, model.getPath());
  read_path(d.left_lane, model.getLeftLane());
  read_path(d.right_lane, model.getRightLane());
  auto leadd = model.getLead();
  d.lead = (LeadData){
      .dist = leadd.getDist(), .prob = leadd.getProb(), .std = leadd.getStd(),
  };
}

static void update_status(UIState *s, int status) {
  if (s->status != status) {
    s->status = status;
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
    read_model(scene.model, sm["model"].getModel());
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
  bool prev_frontview = scene.frontview;
  if (sm.updated("dMonitoringState")) {
    scene.dmonitoring_state = sm["dMonitoringState"].getDMonitoringState();
    scene.frontview = scene.dmonitoring_state.getIsPreview();
   
  }

  // timeout on frontview
  if ((sm.frame - sm.rcv_frame("dMonitoringState")) > 1*UI_FREQ) {
    scene.frontview = false;
  }

  if (prev_frontview != scene.frontview) {
    s->scene.uilayout_sidebarcollapsed = scene.frontview;
    s->active_app = scene.frontview ? cereal::UiLayoutState::App::NONE : cereal::UiLayoutState::App::SETTINGS;
  }

  s->started = scene.thermal.getStarted();
  // Handle onroad/offroad transition
  if (!s->started) {
    if (s->status != STATUS_STOPPED) {
      update_status(s, STATUS_STOPPED);
      s->controls_seen = false;
      s->active_app = cereal::UiLayoutState::App::HOME;

      #ifndef QCOM
      // disconnect from visionipc on PC
      close(s->vision.ipc_fd);
      s->vision.ipc_fd = -1;
      #endif
    } 
  } else if (s->status == STATUS_STOPPED) {
    update_status(s, STATUS_DISENGAGED);
    s->active_app = cereal::UiLayoutState::App::NONE;
  }
  
}

static void check_messages(UIState *s) {
  if (s->sm->update(0) > 0){
    handle_message(s, *(s->sm));
  }
}


#ifdef QCOM

#include <cutils/properties.h>
#include <hardware/sensors.h>
#include <utils/Timers.h>

static void* light_sensor_thread(void *args) {
  int err;
  set_thread_name("light_sensor");

  UIState *s = (UIState*)args;
  s->light_sensor = 0.0;

  struct sensors_poll_device_t* device;
  struct sensors_module_t* module;

  hw_get_module(SENSORS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);
  sensors_open(&module->common, &device);

  // need to do this
  struct sensor_t const* list;
  module->get_sensors_list(module, &list);

  int SENSOR_LIGHT = 7;

  err = device->activate(device, SENSOR_LIGHT, 0);
  if (err != 0) goto fail;
  err = device->activate(device, SENSOR_LIGHT, 1);
  if (err != 0) goto fail;

  device->setDelay(device, SENSOR_LIGHT, ms2ns(100));

  while (!do_exit) {
    static const size_t numEvents = 1;
    sensors_event_t buffer[numEvents];

    int n = device->poll(device, buffer, numEvents);
    if (n < 0) {
      LOG_100("light_sensor_poll failed: %d", n);
    }
    if (n > 0) {
      s->light_sensor = buffer[0].light;
    }
  }
  sensors_close(device);
  return NULL;

fail:
  LOGE("LIGHT SENSOR IS MISSING");
  s->light_sensor = 255;
  return NULL;
}

#endif

int main(int argc, char* argv[]) {
  setpriority(PRIO_PROCESS, 0, -14);

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);

  UIState uistate = {};
  UIState *s = &uistate;
  ui_init(s);

  enable_event_processing(true);

#ifdef QCOM
  pthread_t light_sensor_thread_handle;
  int err = pthread_create(&light_sensor_thread_handle, NULL,
                       light_sensor_thread, s);
  assert(err == 0);
#endif

  TouchState touch = {0};
  touch_init(&touch);
  s->touch_fd = touch.fd;

  // light sensor scaling params
  const bool LEON = util::read_file("/proc/cmdline").find("letv") != std::string::npos;

  float brightness_b, brightness_m;
  int result = read_param(&brightness_b, "BRIGHTNESS_B", true);
  result += read_param(&brightness_m, "BRIGHTNESS_M", true);

  if(result != 0){
    brightness_b = LEON ? 10.0 : 5.0;
    brightness_m = LEON ? 2.6 : 1.3;
    write_param_float(brightness_b, "BRIGHTNESS_B", true);
    write_param_float(brightness_m, "BRIGHTNESS_M", true);
  }

  float smooth_brightness = brightness_b;

  const int MIN_VOLUME = LEON ? 12 : 9;
  const int MAX_VOLUME = LEON ? 15 : 12;
  assert(s->sound.init(MIN_VOLUME));

  int draws = 0;

  while (!do_exit) {
    bool should_swap = false;
    if (!s->started) {
      // Delay a while to avoid 9% cpu usage while car is not started and user is keeping touching on the screen.
      // Don't hold the lock while sleeping, so that vision_connect_thread have chances to get the lock.
      usleep(30 * 1000);
    }
    double u1 = millis_since_boot();

    // light sensor is only exposed on EONs
    float clipped_brightness = (s->light_sensor*brightness_m) + brightness_b;
    if (clipped_brightness > 512) clipped_brightness = 512;
    smooth_brightness = clipped_brightness * 0.01 + smooth_brightness * 0.99;
    if (smooth_brightness > 255) smooth_brightness = 255;
    ui_set_brightness(s, (int)smooth_brightness);

    // resize vision for collapsing sidebar
    const bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
    s->scene.ui_viz_rx = hasSidebar ? box_x : (box_x - sbr_w + (bdr_s * 2));
    s->scene.ui_viz_rw = hasSidebar ? box_w : (box_w + sbr_w - (bdr_s * 2));
    s->scene.ui_viz_ro = hasSidebar ? -(sbr_w - 6 * bdr_s) : 0;

    // poll for touch events
    int touch_x = -1, touch_y = -1;
    int touched = touch_poll(&touch, &touch_x, &touch_y, 0);
    if (touched == 1) {
      set_awake(s, true);
      handle_sidebar_touch(s, touch_x, touch_y);
      handle_vision_touch(s, touch_x, touch_y);
    }

    if (!s->started) {
      // always process events offroad
      check_messages(s);

      if (s->started) {
        s->controls_timeout = 5 * UI_FREQ;
      }
    } else {
      // Car started, fetch a new rgb image from ipc
      set_awake(s, true);
      s->vision.ui_update();

      check_messages(s);

      // Visiond process is just stopped, force a redraw to make screen blank again.
      if (!s->started) {
        s->scene.uilayout_sidebarcollapsed = false;
        ui_draw(s);
        glFinish();
        should_swap = true;
      }
    }

    // manage wakefulness
    if (s->awake_timeout > 0) {
      s->awake_timeout--;
    } else if (!s->scene.frontview){
      set_awake(s, false);
    }

    // manage hardware disconnect
    if (s->hardware_timeout > 0) {
      s->hardware_timeout--;
    } else {
      s->scene.hwType = cereal::HealthData::HwType::UNKNOWN;
    }
    update_offroad_layout_state(s);
    // Don't waste resources on drawing in case screen is off
    if (s->awake && !s->scene.frontview) {
      ui_draw(s);
      glFinish();
      should_swap = true;
    }

    s->sound.setVolume(fmin(MAX_VOLUME, MIN_VOLUME + s->scene.controls_state.getVEgo() / 5)); // up one notch every 5 m/s

    if (s->controls_timeout > 0) {
      s->controls_timeout--;
    } else if (s->started && !s->scene.frontview) {
      if (!s->controls_seen) {
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

    read_param_timeout(&s->is_metric, "IsMetric", &s->is_metric_timeout);
    read_param_timeout(&s->longitudinal_control, "LongitudinalControl", &s->longitudinal_control_timeout);
    read_param_timeout(&s->limit_set_speed, "LimitSetSpeed", &s->limit_set_speed_timeout);
    read_param_timeout(&s->speed_lim_off, "SpeedLimitOffset", &s->limit_set_speed_timeout);
    int param_read = read_param_timeout(&s->last_athena_ping, "LastAthenaPingTime", &s->last_athena_ping_timeout);
    if (param_read != -1) { // Param was updated this loop
      if (param_read != 0) { // Failed to read param
        s->scene.athenaStatus = NET_DISCONNECTED;
      } else if (nanos_since_boot() - s->last_athena_ping < 70e9) {
        s->scene.athenaStatus = NET_CONNECTED;
      } else {
        s->scene.athenaStatus = NET_ERROR;
      }
    }
    
    // the bg thread needs to be scheduled, so the main thread needs time without the lock
    // safe to do this outside the lock?
    if (should_swap) {
      double u2 = millis_since_boot();
      if (u2-u1 > 66) {
        // warn on sub 15fps
        LOGW("slow frame(%d) time: %.2f", draws, u2-u1);
      }
      draws++;
      s->vision.swap();
    }
  }

  set_awake(s, true);

#ifdef QCOM
  // join light_sensor_thread?
#endif

  delete s->sm;
  delete s->pm;
  return 0;
}
