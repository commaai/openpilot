#include "ui.hpp"

static void ui_draw_sidebar_background(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  int sbr_x = hasSidebar ? 0 : -(sbr_w) + bdr_s * 2;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, sbr_x, 0, sbr_w, vwp_h);
  nvgFillColor(s->vg, nvgRGBAf(0,0,0,0.33));
  nvgFill(s->vg);
}

static void ui_draw_sidebar_settings_button(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int settings_btn_h = 117;
  const int settings_btn_w = 200;
  const int settings_btn_x = hasSidebar ? 50 : -(settings_btn_w);
  const int settings_btn_y = 35;

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, settings_btn_x, settings_btn_y,
    settings_btn_w, settings_btn_h, 0, s->img_button_settings, 1.0f);
  nvgRect(s->vg, settings_btn_x, settings_btn_y, settings_btn_w, settings_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_home_button(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int home_btn_h = 180;
  const int home_btn_w = 180;
  const int home_btn_x = hasSidebar ? 60 : -(home_btn_w);
  const int home_btn_y = vwp_h - home_btn_h - 40;

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, home_btn_x, home_btn_y,
    home_btn_w, home_btn_h, 0, s->img_button_home, 1.0f);
  nvgRect(s->vg, home_btn_x, home_btn_y, home_btn_w, home_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_battery_icon(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int battery_img_h = 36;
  const int battery_img_w = 76;
  const int battery_img_x = hasSidebar ? 160 : -(battery_img_w + 160);
  const int battery_img_y = 255;

  int battery_img = strcmp(s->scene.batteryStatus, "Charging") == 0 ?
    s->img_battery_0_charging : s->img_battery_0;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, battery_img_x + 6, battery_img_y + 5,
    ((battery_img_w - 19) * (s->scene.batteryPercent * 0.01)), battery_img_h - 11);
  nvgFillColor(s->vg, nvgRGBAf(255,255,255,1.0));
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, battery_img_x, battery_img_y,
    battery_img_w, battery_img_h, 0, battery_img, 1.0f);
  nvgRect(s->vg, battery_img_x, battery_img_y, battery_img_w, battery_img_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_network_type(UIState *s) {
  const UIScene *scene = &s->scene;
  const int network_x = 50;
  const int network_y = 273;
  const int network_w = 100;
  const int network_h = 100;
  char network_type_str[32];

  if (s->scene.networkType == NETWORKTYPE_NONE) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "--");
  } else if (s->scene.networkType == NETWORKTYPE_WIFI) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "WiFi");
  } else if (s->scene.networkType == NETWORKTYPE_CELL2G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "2G");
  } else if (s->scene.networkType == NETWORKTYPE_CELL3G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "3G");
  } else if (s->scene.networkType == NETWORKTYPE_CELL4G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "LTE");
  } else if (s->scene.networkType == NETWORKTYPE_CELL5G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "5G");
  }

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgFontSize(s->vg, 48);
  nvgFontFace(s->vg, "sans-regular");
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, network_x, network_y, network_w, network_type_str, NULL);
}

void ui_draw_sidebar(UIState *s) {
  ui_draw_sidebar_background(s);
  ui_draw_sidebar_settings_button(s);
  ui_draw_sidebar_home_button(s);
  ui_draw_sidebar_battery_icon(s);
  ui_draw_sidebar_network_type(s);
}
