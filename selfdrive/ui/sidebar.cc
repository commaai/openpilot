#include <stdio.h>
#include <string.h>
#include <math.h>
#include <map>
#include "common/util.h"
#include "paint.hpp"
#include "sidebar.hpp"

static void draw_background(UIState *s) {
#ifdef QCOM
  const NVGcolor color = COLOR_BLACK_ALPHA(85);
#else
  const NVGcolor color = nvgRGBA(0x39, 0x39, 0x39, 0xff);
#endif
  ui_fill_rect(s->vg, {0, 0, sbr_w, s->fb_h}, color);
}

static void draw_settings_button(UIState *s) {
  const float alpha = s->active_app == cereal::UiLayoutState::App::SETTINGS ? 1.0f : 0.65f;
  ui_draw_image(s, settings_btn, "button_settings", alpha);
}

static void draw_home_button(UIState *s) {
  const float alpha = s->active_app == cereal::UiLayoutState::App::HOME ? 1.0f : 0.65f;
  ui_draw_image(s, home_btn, "button_home", alpha);
}

static void draw_network_strength(UIState *s) {
  static std::map<cereal::ThermalData::NetworkStrength, int> network_strength_map = {
      {cereal::ThermalData::NetworkStrength::UNKNOWN, 1},
      {cereal::ThermalData::NetworkStrength::POOR, 2},
      {cereal::ThermalData::NetworkStrength::MODERATE, 3},
      {cereal::ThermalData::NetworkStrength::GOOD, 4},
      {cereal::ThermalData::NetworkStrength::GREAT, 5}};
  const int img_idx = s->scene.thermal.getNetworkType() == cereal::ThermalData::NetworkType::NONE ? 0 : network_strength_map[s->scene.thermal.getNetworkStrength()];
  ui_draw_image(s, {58, 196, 176, 27}, util::string_format("network_%d", img_idx).c_str(), 1.0f);
}

static void draw_battery_icon(UIState *s) {
  const char *battery_img = s->scene.thermal.getBatteryStatus() == "Charging" ? "battery_charging" : "battery";
  const Rect rect = {160, 255, 76, 36};
  ui_fill_rect(s->vg, {rect.x + 6, rect.y + 5,
              int((rect.w - 19) * s->scene.thermal.getBatteryPercent() * 0.01), rect.h - 11}, COLOR_WHITE);
  ui_draw_image(s, rect, battery_img, 1.0f);
}

static void draw_network_type(UIState *s) {
  static std::map<cereal::ThermalData::NetworkType, const char *> network_type_map = {
      {cereal::ThermalData::NetworkType::NONE, "--"},
      {cereal::ThermalData::NetworkType::WIFI, "WiFi"},
      {cereal::ThermalData::NetworkType::CELL2_G, "2G"},
      {cereal::ThermalData::NetworkType::CELL3_G, "3G"},
      {cereal::ThermalData::NetworkType::CELL4_G, "4G"},
      {cereal::ThermalData::NetworkType::CELL5_G, "5G"}};
  const int network_x = 50;
  const int network_y = 273;
  const int network_w = 100;
  const char *network_type = network_type_map[s->scene.thermal.getNetworkType()];
  nvgFillColor(s->vg, COLOR_WHITE);
  nvgFontSize(s->vg, 48);
  nvgFontFace(s->vg, "sans-regular");
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, network_x, network_y, network_w, network_type ? network_type : "--", NULL);
}

static void draw_metric(UIState *s, const char *label_str, const char *value_str, const int severity, const int y_offset, const char *message_str) {
  NVGcolor status_color;

  if (severity == 0) {
    status_color = COLOR_WHITE;
  } else if (severity == 1) {
    status_color = COLOR_YELLOW;
  } else if (severity > 1) {
    status_color = COLOR_RED;
  }

  const Rect rect = {30, 338 + y_offset, 240, message_str ? strchr(message_str, '\n') ? 124 : 100 : 148};
  ui_draw_rect(s->vg, rect, severity > 0 ? COLOR_WHITE : COLOR_WHITE_ALPHA(85), 2, 20.);

  nvgBeginPath(s->vg);
  nvgRoundedRectVarying(s->vg, rect.x + 6, rect.y + 6, 18, rect.h - 12, 25, 0, 0, 25);
  nvgFillColor(s->vg, status_color);
  nvgFill(s->vg);

  if (!message_str) {
    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 78);
    nvgFontFace(s->vg, "sans-bold");
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, rect.x + 50, rect.y + 50, rect.w - 60, value_str, NULL);

    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 48);
    nvgFontFace(s->vg, "sans-regular");
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, rect.x + 50, rect.y + 50 + 66, rect.w - 60, label_str, NULL);
  } else {
    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 48);
    nvgFontFace(s->vg, "sans-bold");
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, rect.x + 35, rect.y + (strchr(message_str, '\n') ? 40 : 50), rect.w - 50, message_str, NULL);
  }
}

static void draw_temp_metric(UIState *s) {
  static std::map<cereal::ThermalData::ThermalStatus, const int> temp_severity_map = {
      {cereal::ThermalData::ThermalStatus::GREEN, 0},
      {cereal::ThermalData::ThermalStatus::YELLOW, 1},
      {cereal::ThermalData::ThermalStatus::RED, 2},
      {cereal::ThermalData::ThermalStatus::DANGER, 3}};
  std::string temp_val = std::to_string((int)s->scene.thermal.getAmbient()) + "°C";
  draw_metric(s, "TEMP", temp_val.c_str(), temp_severity_map[s->scene.thermal.getThermalStatus()], 0, NULL);
}

static void draw_panda_metric(UIState *s) {
  const int panda_y_offset = 32 + 148;

  int panda_severity = 0;
  std::string panda_message = "VEHICLE\nONLINE";
  if (s->scene.hwType == cereal::HealthData::HwType::UNKNOWN) {
    panda_severity = 2;
    panda_message = "NO\nVEHICLE";
  } else if (s->started) {
    if (s->scene.satelliteCount < 6) {
      panda_severity = 1;
      panda_message = "VEHICLE\nNO GPS";
    } else {
      panda_severity = 0;
      panda_message = "VEHICLE\nGOOD GPS";
    }
  }
  draw_metric(s, NULL, NULL, panda_severity, panda_y_offset, panda_message.c_str());
}

static void draw_connectivity(UIState *s) {
  static std::map<NetStatus, std::pair<const char *, int>> connectivity_map = {
      {NET_ERROR, {"CONNECT\nERROR", 2}},
      {NET_CONNECTED, {"CONNECT\nONLINE", 0}},
      {NET_DISCONNECTED, {"CONNECT\nOFFLINE", 1}},
  };
  auto net_params = connectivity_map[s->scene.athenaStatus];
  draw_metric(s, NULL, NULL, net_params.second, 180 + 158, net_params.first);
}

void ui_draw_sidebar(UIState *s) {
  if (s->scene.sidebar_collapsed) {
    return;
  }
  draw_background(s);
  draw_settings_button(s);
  draw_home_button(s);
  draw_network_strength(s);
  draw_battery_icon(s);
  draw_network_type(s);
  draw_temp_metric(s);
  draw_panda_metric(s);
  draw_connectivity(s);
}
