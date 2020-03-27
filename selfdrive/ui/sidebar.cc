#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ui.hpp"

static void ui_draw_sidebar_background(UIState *s, bool hasSidebar) {
  int sbr_x = hasSidebar ? 0 : -(sbr_w) + bdr_s * 2;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, sbr_x, 0, sbr_w, vwp_h);
  nvgFillColor(s->vg, COLOR_BLACK_ALPHA(85));
  nvgFill(s->vg);
}

static void ui_draw_sidebar_settings_button(UIState *s, bool hasSidebar) {
  bool settingsActive = s->active_app == cereal_UiLayoutState_App_settings;
  const int settings_btn_xr = hasSidebar ? settings_btn_x : -(sbr_w);

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, settings_btn_xr, settings_btn_y,
    settings_btn_w, settings_btn_h, 0, s->img_button_settings, settingsActive ? 1.0f : 0.65f);
  nvgRect(s->vg, settings_btn_xr, settings_btn_y, settings_btn_w, settings_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_home_button(UIState *s, bool hasSidebar) {
  bool homeActive = s->active_app == cereal_UiLayoutState_App_home;
  const int home_btn_xr = hasSidebar ? home_btn_x : -(sbr_w);

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, home_btn_xr, home_btn_y,
    home_btn_w, home_btn_h, 0, s->img_button_home, homeActive ? 1.0f : 0.65f);
  nvgRect(s->vg, home_btn_xr, home_btn_y, home_btn_w, home_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_network_strength(UIState *s, bool hasSidebar) {
  const int network_img_h = 27;
  const int network_img_w = 176;
  const int network_img_x = hasSidebar ? 58 : -(sbr_w);
  const int network_img_y = 196;
  const int network_img = s->scene.networkType == cereal_ThermalData_NetworkType_none ?
                          s->img_network[0] : s->img_network[s->scene.networkStrength + 1];

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, network_img_x, network_img_y,
    network_img_w, network_img_h, 0, network_img, 1.0f);
  nvgRect(s->vg, network_img_x, network_img_y, network_img_w, network_img_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_battery_icon(UIState *s, bool hasSidebar) {
  const int battery_img_h = 36;
  const int battery_img_w = 76;
  const int battery_img_x = hasSidebar ? 160 : -(sbr_w);
  const int battery_img_y = 255;

  int battery_img = strcmp(s->scene.batteryStatus, "Charging") == 0 ?
    s->img_battery_charging : s->img_battery;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, battery_img_x + 6, battery_img_y + 5,
    ((battery_img_w - 19) * (s->scene.batteryPercent * 0.01)), battery_img_h - 11);
  nvgFillColor(s->vg, COLOR_WHITE);
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, battery_img_x, battery_img_y,
    battery_img_w, battery_img_h, 0, battery_img, 1.0f);
  nvgRect(s->vg, battery_img_x, battery_img_y, battery_img_w, battery_img_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_network_type(UIState *s, bool hasSidebar) {
  const int network_x = hasSidebar ? 50 : -(sbr_w);
  const int network_y = 273;
  const int network_w = 100;
  const int network_h = 100;
  const char *network_types[6] = {"--", "WiFi", "2G", "3G", "4G", "5G"};
  char network_type_str[32];

  if (s->scene.networkType <= 5) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", network_types[s->scene.networkType]);
  }

  nvgFillColor(s->vg, COLOR_WHITE);
  nvgFontSize(s->vg, 48);
  nvgFontFaceId(s->vg, s->font_sans_regular);
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, network_x, network_y, network_w, network_type_str, NULL);
}

static void ui_draw_sidebar_metric(UIState *s, const char* label_str, const char* value_str, const int severity, const int y_offset, const char* message_str, bool hasSidebar) {
  const int metric_x = hasSidebar ? 30 : -(sbr_w);
  const int metric_y = 338 + y_offset;
  const int metric_w = 240;
  const int metric_h = message_str ? strchr(message_str, '\n') ? 124 : 100 : 148;

  NVGcolor status_color;

  if (severity == 0) {
    status_color = COLOR_WHITE;
  } else if (severity == 1) {
    status_color = COLOR_YELLOW;
  } else if (severity > 1) {
    status_color = COLOR_RED;
  }

  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, metric_x, metric_y, metric_w, metric_h, 20);
  nvgStrokeColor(s->vg, severity > 0 ? COLOR_WHITE : COLOR_WHITE_ALPHA(85));
  nvgStrokeWidth(s->vg, 2);
  nvgStroke(s->vg);

  nvgBeginPath(s->vg);
  nvgRoundedRectVarying(s->vg, metric_x + 6, metric_y + 6, 18, metric_h - 12, 25, 0, 0, 25);
  nvgFillColor(s->vg, status_color);
  nvgFill(s->vg);

  if (!message_str) {
    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 78);
    nvgFontFaceId(s->vg, s->font_sans_bold);
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, metric_x + 50, metric_y + 50, metric_w - 60, value_str, NULL);

    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 48);
    nvgFontFaceId(s->vg, s->font_sans_regular);
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, metric_x + 50, metric_y + 50 + 66, metric_w - 60, label_str, NULL);
  } else {
    nvgFillColor(s->vg, COLOR_WHITE);
    nvgFontSize(s->vg, 48);
    nvgFontFaceId(s->vg, s->font_sans_bold);
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, metric_x + 35, metric_y + (strchr(message_str, '\n') ? 40 : 50), metric_w - 50, message_str, NULL);
  }
}

static void ui_draw_sidebar_temp_metric(UIState *s, bool hasSidebar) {
  int temp_severity;
  char temp_label_str[32];
  char temp_value_str[32];
  char temp_value_unit[32];
  const int temp_y_offset = 0;

  if (s->scene.thermalStatus == cereal_ThermalData_ThermalStatus_green) {
    temp_severity = 0;
  } else if (s->scene.thermalStatus == cereal_ThermalData_ThermalStatus_yellow) {
    temp_severity = 1;
  } else if (s->scene.thermalStatus == cereal_ThermalData_ThermalStatus_red) {
    temp_severity = 2;
  } else if (s->scene.thermalStatus == cereal_ThermalData_ThermalStatus_danger) {
    temp_severity = 3;
  }

  snprintf(temp_value_str, sizeof(temp_value_str), "%d", s->scene.paTemp);
  snprintf(temp_value_unit, sizeof(temp_value_unit), "%s", "°C");
  snprintf(temp_label_str, sizeof(temp_label_str), "%s", "TEMP");
  strcat(temp_value_str, temp_value_unit);

  ui_draw_sidebar_metric(s, temp_label_str, temp_value_str, temp_severity, temp_y_offset, NULL, hasSidebar);
}

static void ui_draw_sidebar_panda_metric(UIState *s, bool hasSidebar) {
  int panda_severity;
  char panda_message_str[32];
  const int panda_y_offset = 32 + 148;

  if (s->scene.hwType == cereal_HealthData_HwType_unknown) {
    panda_severity = 2;
    snprintf(panda_message_str, sizeof(panda_message_str), "%s", "NO\nPANDA");
  } else if (s->scene.hwType == cereal_HealthData_HwType_whitePanda) {
    panda_severity = 0;
    snprintf(panda_message_str, sizeof(panda_message_str), "%s", "PANDA\nACTIVE");
  } else if (
      (s->scene.hwType == cereal_HealthData_HwType_greyPanda) ||
      (s->scene.hwType == cereal_HealthData_HwType_blackPanda) ||
      (s->scene.hwType == cereal_HealthData_HwType_uno)) {
      if (s->scene.satelliteCount == -1) {
        panda_severity = 0;
        snprintf(panda_message_str, sizeof(panda_message_str), "%s", "PANDA\nACTIVE");
      } else if (s->scene.satelliteCount < 6) {
        panda_severity = 1;
        snprintf(panda_message_str, sizeof(panda_message_str), "%s", "PANDA\nNO GPS");
      } else if (s->scene.satelliteCount >= 6) {
        panda_severity = 0;
        snprintf(panda_message_str, sizeof(panda_message_str), "%s", "PANDA\nGOOD GPS");
      }
  }

  ui_draw_sidebar_metric(s, NULL, NULL, panda_severity, panda_y_offset, panda_message_str, hasSidebar);
}

void ui_draw_sidebar(UIState *s) {
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  ui_draw_sidebar_background(s, hasSidebar);
  ui_draw_sidebar_settings_button(s, hasSidebar);
  ui_draw_sidebar_home_button(s, hasSidebar);
  ui_draw_sidebar_network_strength(s, hasSidebar);
  ui_draw_sidebar_battery_icon(s, hasSidebar);
  ui_draw_sidebar_network_type(s, hasSidebar);
  ui_draw_sidebar_temp_metric(s, hasSidebar);
  ui_draw_sidebar_panda_metric(s, hasSidebar);
}
