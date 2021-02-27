#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <iostream>

#include <algorithm>

#include "common/util.h"
#include "common/params.h"
#include "common/touch.h"
#include "common/swaglog.h"

#include "ui.hpp"
#include "paint.hpp"
#include "android/sl_sound.hpp"

ExitHandler do_exit;
static void ui_set_brightness(UIState *s, int brightness) {
  static int last_brightness = -1;
  if (last_brightness != brightness && (s->awake || brightness == 0)) {
    if (set_brightness(brightness)) {
      last_brightness = brightness;
    }
  }
}

static void send_ls(UIState *s, int status) {
  MessageBuilder msg;
  auto lsStatus = msg.initEvent().initLaneSpeedButton();
  lsStatus.setStatus(status);
  s->pm->send("laneSpeedButton", msg);
}

static void send_df(UIState *s, int status) {
  MessageBuilder msg;
  auto dfStatus = msg.initEvent().initDynamicFollowButton();
  dfStatus.setStatus(status);
  s->pm->send("dynamicFollowButton", msg);
}

static void send_ml(UIState *s, bool enabled) {
  MessageBuilder msg;
  auto mlStatus = msg.initEvent().initModelLongButton();
  mlStatus.setEnabled(enabled);
  s->pm->send("modelLongButton", msg);
}

static bool handle_ls_touch(UIState *s, int touch_x, int touch_y) {
  //lsButton manager
  int padding = 40;
  int btn_x_1 = 1660 - 200;
  int btn_x_2 = 1660 - 50;
  if ((btn_x_1 - padding <= touch_x) && (touch_x <= btn_x_2 + padding) && (855 - padding <= touch_y)) {
    s->scene.lsButtonStatus++;
    if (s->scene.lsButtonStatus > 2) { s->scene.lsButtonStatus = 0; }
    send_ls(s, s->scene.lsButtonStatus);
    printf("ls button: %d\n", s->scene.lsButtonStatus);
    return true;
  }
  return false;
}

static bool handle_df_touch(UIState *s, int touch_x, int touch_y) {
  //dfButton manager
  int padding = 40;
  if ((1660 - padding <= touch_x) && (855 - padding <= touch_y)) {
    s->scene.dfButtonStatus++;
    if (s->scene.dfButtonStatus > 3) { s->scene.dfButtonStatus = 0; }
    send_df(s, s->scene.dfButtonStatus);
    printf("df button: %d\n", s->scene.dfButtonStatus);
    return true;
  }
  return false;
}

static bool handle_ml_touch(UIState *s, int touch_x, int touch_y) {
  //mlButton manager
  int padding = 40;
  int btn_w = 500;
  int btn_h = 138;
  int xs[2] = {1920 / 2 - btn_w / 2, 1920 / 2 + btn_w / 2};
  int y_top = 915 - btn_h / 2;
  if (xs[0] <= touch_x + padding && touch_x - padding <= xs[1] && y_top - padding <= touch_y) {
    s->scene.mlButtonEnabled = !s->scene.mlButtonEnabled;
    send_ml(s, s->scene.mlButtonEnabled);
    printf("ml button: %d\n", s->scene.mlButtonEnabled);
    return true;
  }
  return false;
}

static bool handle_SA_touched(UIState *s, int touch_x, int touch_y) {
  if (s->active_app == cereal::UiLayoutState::App::NONE) {  // if onroad (not settings or home)
    if ((s->awake && s->status != STATUS_OFFROAD) || s->ui_debug) {  // if car started or debug mode
      if (handle_df_touch(s, touch_x, touch_y) || handle_ls_touch(s, touch_x, touch_y) || handle_ml_touch(s, touch_x, touch_y)) {
        s->sidebar_collapsed = true;  // collapse sidebar when tapping any SA button
        return true;  // only allow one button to be pressed at a time
      }
    }
  }
  return false;
}

static void handle_display_state(UIState *s, bool user_input) {
  static int awake_timeout = 0;

  constexpr float accel_samples = 5*UI_FREQ;
  static float accel_prev = 0., gyro_prev = 0.;

  bool should_wake = s->started || s->ignition || user_input;
  if (!should_wake) {
    // tap detection while display is off
    bool accel_trigger = abs(s->accel_sensor - accel_prev) > 0.2;
    bool gyro_trigger = abs(s->gyro_sensor - gyro_prev) > 0.15;
    should_wake = accel_trigger && gyro_trigger;
    gyro_prev = s->gyro_sensor;
    accel_prev = (accel_prev * (accel_samples - 1) + s->accel_sensor) / accel_samples;
  }

  // determine desired state
  if (should_wake) {
    awake_timeout = 30*UI_FREQ;
  } else if (awake_timeout > 0){
    --awake_timeout;
    should_wake = true;
  }

  // handle state transition
  if (s->awake != should_wake) {
    s->awake = should_wake;
    int display_mode = s->awake ? HWC_POWER_MODE_NORMAL : HWC_POWER_MODE_OFF;
    LOGW("setting display mode %d", display_mode);
    s->fb->set_power(display_mode);

    if (s->awake) {
      system("service call window 18 i32 1");
    }
  }
}

static void handle_vision_touch(UIState *s, int touch_x, int touch_y) {
  if (s->started && (touch_x >= s->viz_rect.x - bdr_s)
      && (s->active_app != cereal::UiLayoutState::App::SETTINGS)) {
    if (!s->scene.frontview) {
      s->sidebar_collapsed = !s->sidebar_collapsed;
    } else {
      Params().write_db_value("IsDriverViewEnabled", "0", 1);
    }
  }
}

static void handle_sidebar_touch(UIState *s, int touch_x, int touch_y) {
  if (!s->sidebar_collapsed && touch_x <= sbr_w) {
    if (settings_btn.ptInRect(touch_x, touch_y)) {
      s->active_app = cereal::UiLayoutState::App::SETTINGS;
    } else if (home_btn.ptInRect(touch_x, touch_y)) {
      if (s->started) {
        s->active_app = cereal::UiLayoutState::App::NONE;
        s->sidebar_collapsed = true;
      } else {
        s->active_app = cereal::UiLayoutState::App::HOME;
      }
    }
  }
}

static void update_offroad_layout_state(UIState *s, PubMaster *pm) {
  static int timeout = 0;
  static bool prev_collapsed = false;
  static cereal::UiLayoutState::App prev_app = cereal::UiLayoutState::App::NONE;
  if (timeout > 0) {
    timeout--;
  }
  if (prev_collapsed != s->sidebar_collapsed || prev_app != s->active_app || timeout == 0) {
    MessageBuilder msg;
    auto layout = msg.initEvent().initUiLayoutState();
    layout.setActiveApp(s->active_app);
    layout.setSidebarCollapsed(s->sidebar_collapsed);
    pm->send("offroadLayout", msg);
    LOGD("setting active app to %d with sidebar %d", (int)s->active_app, s->sidebar_collapsed);
    prev_collapsed = s->sidebar_collapsed;
    prev_app = s->active_app;
    timeout = 2 * UI_FREQ;
  }
}

int main(int argc, char* argv[]) {
  setpriority(PRIO_PROCESS, 0, -14);
  SLSound sound;

  UIState uistate = {};
  UIState *s = &uistate;
  ui_init(s);
  sa_init(s, true);
  s->sound = &sound;

  TouchState touch = {0};
  touch_init(&touch);
  handle_display_state(s, true);

  PubMaster *pm = new PubMaster({"offroadLayout"});

  // light sensor scaling and volume params
  const bool LEON = util::read_file("/proc/cmdline").find("letv") != std::string::npos;

  float brightness_b = 0, brightness_m = 0;
  int result = read_param(&brightness_b, "BRIGHTNESS_B", true);
  result += read_param(&brightness_m, "BRIGHTNESS_M", true);
  if (result != 0) {
    brightness_b = LEON ? 10.0 : 5.0;
    brightness_m = LEON ? 2.6 : 1.3;
    write_param_float(brightness_b, "BRIGHTNESS_B", true);
    write_param_float(brightness_m, "BRIGHTNESS_M", true);
  }
  float smooth_brightness = brightness_b;

  const int MIN_VOLUME = LEON ? 12 : 9;
  const int MAX_VOLUME = LEON ? 15 : 12;
  s->sound->setVolume(MIN_VOLUME);

  bool last_started = s->started;
  while (!do_exit) {
    if (!s->started) {
      util::sleep_for(50);
    }

    if (s->started && !last_started) {
      sa_init(s, false);  // reset ml button and regrab params
    }
    last_started = s->started;

    double u1 = millis_since_boot();

    ui_update(s);

    // poll for touch events
    int touch_x = -1, touch_y = -1;
    int touched = touch_poll(&touch, &touch_x, &touch_y, 0);
    if (touched == 1) {
      if (s->ui_debug) { printf("touched x: %d, y: %d\n", touch_x, touch_y); }
      handle_sidebar_touch(s, touch_x, touch_y);
      if (!handle_SA_touched(s, touch_x, touch_y)) {  // if SA button not touched
        handle_vision_touch(s, touch_x, touch_y);
      }
    }

    // Don't waste resources on drawing in case screen is off
    handle_display_state(s, touched == 1);
    if (!s->awake) {
      continue;
    }

    // up one notch every 5 m/s
    s->sound->setVolume(fmin(MAX_VOLUME, MIN_VOLUME + s->scene.car_state.getVEgo() / 5));

    // set brightness
    float clipped_brightness = fmin(512, (s->light_sensor*brightness_m) + brightness_b);
    smooth_brightness = fmin(255, clipped_brightness * 0.01 + smooth_brightness * 0.99);
    ui_set_brightness(s, (int)smooth_brightness);

    update_offroad_layout_state(s, pm);

    ui_draw(s);
    double u2 = millis_since_boot();
    if (!s->scene.frontview && (u2-u1 > 66)) {
      // warn on sub 15fps
      LOGW("slow frame(%llu) time: %.2f", (s->sm)->frame, u2-u1);
    }
    s->fb->swap();
  }

  handle_display_state(s, true);
  delete s->sm;
  delete pm;
  return 0;
}
