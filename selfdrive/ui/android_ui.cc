#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <czmq.h>

#include "common/util.h"
#include "common/utilpp.h"
#include "common/params.h"
#include "common/touch.h"
#include "common/timing.h"
#include "common/swaglog.h"

#include "ui.hpp"
#include "paint.hpp"

// Includes for light sensor
#include <cutils/properties.h>
#include <hardware/sensors.h>
#include <utils/Timers.h>

volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}


static void* light_sensor_thread(void *args) {
  set_thread_name("light_sensor");

  int err;
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
  if (awake) {
    // 30 second timeout
    s->awake_timeout = 30*UI_FREQ;
  }
  if (s->awake != awake) {
    s->awake = awake;

    // TODO: replace command_awake and command_sleep with direct calls to android
    if (awake) {
      LOGW("awake normal");
      framebuffer_set_power(s->fb, HWC_POWER_MODE_NORMAL);
      enable_event_processing(true);
    } else {
      LOGW("awake off");
      ui_set_brightness(s, 0);
      framebuffer_set_power(s->fb, HWC_POWER_MODE_OFF);
      enable_event_processing(false);
    }
  }
}

static void handle_vision_touch(UIState *s, int touch_x, int touch_y) {
  if (s->started && (touch_x >= s->scene.ui_viz_rx - bdr_s)
      && (s->active_app != cereal::UiLayoutState::App::SETTINGS)) {
    if (!s->scene.frontview) {
      s->scene.uilayout_sidebarcollapsed = !s->scene.uilayout_sidebarcollapsed;
    } else {
      write_db_value("IsDriverViewEnabled", "0", 1);
    }
  }
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

static void update_offroad_layout_state(UIState *s) {
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
}

int main(int argc, char* argv[]) {
  int err;
  setpriority(PRIO_PROCESS, 0, -14);

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);

  UIState uistate = {};
  UIState *s = &uistate;
  ui_init(s);
  set_awake(s, true);

  enable_event_processing(true);

  pthread_t connect_thread_handle;
  err = pthread_create(&connect_thread_handle, NULL,
                       vision_connect_thread, s);
  assert(err == 0);

  pthread_t light_sensor_thread_handle;
  err = pthread_create(&light_sensor_thread_handle, NULL,
                       light_sensor_thread, s);
  assert(err == 0);

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
    pthread_mutex_lock(&s->lock);
    double u1 = millis_since_boot();

    // light sensor is only exposed on EONs
    float clipped_brightness = (s->light_sensor*brightness_m) + brightness_b;
    if (clipped_brightness > 512) clipped_brightness = 512;
    smooth_brightness = clipped_brightness * 0.01 + smooth_brightness * 0.99;
    if (smooth_brightness > 255) smooth_brightness = 255;
    ui_set_brightness(s, (int)smooth_brightness);
    ui_update_sizes(s);

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
      set_awake(s, true);
      // Car started, fetch a new rgb image from ipc
      if (s->vision_connected){
        ui_update(s);
      }

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
    } else {
      set_awake(s, false);
    }

    // manage hardware disconnect
    if (s->hardware_timeout > 0) {
      s->hardware_timeout--;
    } else {
      s->scene.hwType = cereal::HealthData::HwType::UNKNOWN;
    }

    // Don't waste resources on drawing in case screen is off
    if (s->awake) {
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
    update_offroad_layout_state(s);

    pthread_mutex_unlock(&s->lock);

    // the bg thread needs to be scheduled, so the main thread needs time without the lock
    // safe to do this outside the lock?
    if (should_swap) {
      double u2 = millis_since_boot();
      if (u2-u1 > 66) {
        // warn on sub 15fps
        LOGW("slow frame(%d) time: %.2f", draws, u2-u1);
      }
      draws++;
      framebuffer_swap(s->fb);
    }
  }

  set_awake(s, true);

  // wake up bg thread to exit
  pthread_mutex_lock(&s->lock);
  pthread_mutex_unlock(&s->lock);

  // join light_sensor_thread?
  err = pthread_join(connect_thread_handle, NULL);
  assert(err == 0);
  delete s->sm;
  delete s->pm;
  return 0;
}
