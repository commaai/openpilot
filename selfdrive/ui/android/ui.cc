#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>

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

static void handle_display_state(UIState *s, bool user_input) {
  static int awake_timeout = 0;

  constexpr float accel_samples = 5*UI_FREQ;
  static float accel_prev = 0., gyro_prev = 0.;

  bool should_wake = s->scene.started || s->scene.ignition || user_input;
  if (!should_wake) {
    // tap detection while display is off
    bool accel_trigger = abs(s->scene.accel_sensor - accel_prev) > 0.2;
    bool gyro_trigger = abs(s->scene.gyro_sensor - gyro_prev) > 0.15;
    should_wake = accel_trigger && gyro_trigger;
    gyro_prev = s->scene.gyro_sensor;
    accel_prev = (accel_prev * (accel_samples - 1) + s->scene.accel_sensor) / accel_samples;
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
  if (s->scene.started && (touch_x >= s->viz_rect.x - bdr_s)
      && (s->active_app != cereal::UiLayoutState::App::SETTINGS)) {
    if (!s->scene.driver_view) {
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
      if (s->scene.started) {
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
  s->sound = &sound;

  TouchState touch = {0};
  touch_init(&touch);
  handle_display_state(s, true);

  PubMaster *pm = new PubMaster({"offroadLayout"});

  // light sensor scaling and volume params
  float brightness_b = 0, brightness_m = 0;
  int result = read_param(&brightness_b, "BRIGHTNESS_B", true);
  result += read_param(&brightness_m, "BRIGHTNESS_M", true);
  if (result != 0) {
    brightness_b = 10.0;
    brightness_m = 2.6;
    write_param_float(brightness_b, "BRIGHTNESS_B", true);
    write_param_float(brightness_m, "BRIGHTNESS_M", true);
  }
  float smooth_brightness = brightness_b;

  const int MIN_VOLUME = 12;
  const int MAX_VOLUME = 15;
  s->sound->setVolume(MIN_VOLUME);

  while (!do_exit) {
    if (!s->scene.started) {
      util::sleep_for(50);
    }
    double u1 = millis_since_boot();

    ui_update(s);

    // poll for touch events
    int touch_x = -1, touch_y = -1;
    int touched = touch_poll(&touch, &touch_x, &touch_y, 0);
    if (touched == 1) {
      handle_sidebar_touch(s, touch_x, touch_y);
      handle_vision_touch(s, touch_x, touch_y);
    }

    // Don't waste resources on drawing in case screen is off
    handle_display_state(s, touched == 1);
    if (!s->awake) {
      continue;
    }

    // up one notch every 5 m/s
    s->sound->setVolume(fmin(MAX_VOLUME, MIN_VOLUME + s->scene.car_state.getVEgo() / 5));

    // set brightness
    float clipped_brightness = fmin(512, (s->scene.light_sensor*brightness_m) + brightness_b);
    smooth_brightness = fmin(255, clipped_brightness * 0.01 + smooth_brightness * 0.99);
    ui_set_brightness(s, (int)smooth_brightness);

    update_offroad_layout_state(s, pm);

    ui_draw(s);
    double u2 = millis_since_boot();
    if (!s->scene.driver_view && (u2-u1 > 66)) {
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
