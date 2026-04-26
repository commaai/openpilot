#include "tools/jotpluggler/style.h"

#include "implot.h"
#include "tools/jotpluggler/internal.h"

namespace {

AppTheme current_theme = AppTheme::Light;

struct ThemePair {
  ImVec4 light;
  ImVec4 dark;
};

const ThemePair APP_COLORS[static_cast<size_t>(AppColor::Count)] = {
  /* Clear */             {color_rgb(227, 229, 233),        color_rgb(18, 21, 25)},
  /* FpsBg */             {color_rgb(248, 249, 251, 0.92f), color_rgb(28, 33, 39, 0.94f)},
  /* FpsBorder */         {color_rgb(182, 188, 196, 0.95f), color_rgb(74, 84, 96, 0.95f)},
  /* FpsText */           {color_rgb(57, 62, 69),           color_rgb(226, 231, 237)},
  /* PlotBg */            {color_rgb(255, 255, 255),        color_rgb(17, 20, 24)},
  /* PlotBorder */        {color_rgb(186, 190, 196),        color_rgb(67, 76, 87)},
  /* PlotLegendBg */      {color_rgb(248, 249, 251, 0.92f), color_rgb(26, 31, 37, 0.94f)},
  /* PlotLegendBorder */  {color_rgb(168, 175, 184),        color_rgb(78, 89, 102)},
  /* PlotText */          {color_rgb(57, 62, 69),           color_rgb(224, 230, 237)},
  /* PlotMutedText */     {color_rgb(95, 103, 112),         color_rgb(157, 168, 181)},
  /* PlotGrid */          {color_rgb(188, 196, 206),        color_rgb(64, 73, 84)},
  /* PlotAxisBg */        {color_rgb(255, 255, 255, 0.0f),  color_rgb(17, 20, 24, 0.0f)},
  /* PlotAxisBgHovered */ {color_rgb(214, 220, 228, 0.45f), color_rgb(75, 91, 108, 0.38f)},
  /* PlotAxisBgActive */  {color_rgb(199, 209, 222, 0.55f), color_rgb(92, 110, 130, 0.48f)},
  /* PlotSelection */     {color_rgb(252, 211, 77, 0.28f),  color_rgb(250, 204, 21, 0.30f)},
  /* PlotCrosshairs */    {color_rgb(120, 128, 138, 0.70f), color_rgb(174, 185, 197, 0.72f)},
  /* PlotCursor */        {color_rgb(108, 118, 128, 0.70f), color_rgb(180, 190, 202, 0.74f)},
};

struct ImGuiThemeColor {
  ImGuiCol color;
  ImVec4 light;
  ImVec4 dark;
};

const ImGuiThemeColor IMGUI_COLORS[] = {
  {ImGuiCol_WindowBg, color_rgb(250, 250, 251), color_rgb(20, 24, 29)},
  {ImGuiCol_ChildBg, color_rgb(255, 255, 255), color_rgb(18, 22, 27)},
  {ImGuiCol_Border, color_rgb(194, 198, 204), color_rgb(58, 66, 75)},
  {ImGuiCol_TitleBg, color_rgb(252, 252, 253), color_rgb(26, 30, 36)},
  {ImGuiCol_TitleBgActive, color_rgb(252, 252, 253), color_rgb(31, 36, 43)},
  {ImGuiCol_TitleBgCollapsed, color_rgb(252, 252, 253), color_rgb(24, 28, 33)},
  {ImGuiCol_Text, color_rgb(74, 80, 88), color_rgb(221, 227, 234)},
  {ImGuiCol_TextDisabled, color_rgb(108, 118, 128), color_rgb(134, 145, 158)},
  {ImGuiCol_Button, color_rgb(255, 255, 255), color_rgb(33, 39, 47)},
  {ImGuiCol_ButtonHovered, color_rgb(246, 248, 250), color_rgb(44, 52, 62)},
  {ImGuiCol_ButtonActive, color_rgb(238, 240, 244), color_rgb(55, 65, 77)},
  {ImGuiCol_FrameBg, color_rgb(255, 255, 255), color_rgb(27, 32, 39)},
  {ImGuiCol_FrameBgHovered, color_rgb(248, 249, 251), color_rgb(37, 45, 54)},
  {ImGuiCol_FrameBgActive, color_rgb(241, 244, 248), color_rgb(47, 57, 68)},
  {ImGuiCol_Header, color_rgb(243, 245, 248), color_rgb(35, 42, 50)},
  {ImGuiCol_HeaderHovered, color_rgb(237, 240, 244), color_rgb(45, 54, 65)},
  {ImGuiCol_HeaderActive, color_rgb(232, 236, 240), color_rgb(55, 66, 79)},
  {ImGuiCol_PopupBg, color_rgb(248, 249, 251), color_rgb(24, 29, 35)},
  {ImGuiCol_MenuBarBg, color_rgb(232, 236, 241), color_rgb(24, 29, 35)},
  {ImGuiCol_Separator, color_rgb(194, 198, 204), color_rgb(62, 71, 82)},
  {ImGuiCol_DockingEmptyBg, color_rgb(244, 246, 248), color_rgb(18, 21, 25)},
  {ImGuiCol_DockingPreview, color_rgb(69, 115, 184, 0.22f), color_rgb(99, 148, 220, 0.28f)},
};

}  // namespace

bool app_dark_mode() {
  return current_theme == AppTheme::Dark;
}

ImVec4 app_color(AppColor color) {
  const ThemePair &pair = APP_COLORS[static_cast<size_t>(color)];
  return app_dark_mode() ? pair.dark : pair.light;
}

void apply_app_style(AppTheme theme) {
  current_theme = theme;
  if (app_dark_mode()) {
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();
  } else {
    ImGui::StyleColorsLight();
    ImPlot::StyleColorsLight();
  }

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 0.0f;
  style.ChildRounding = 0.0f;
  style.PopupRounding = 0.0f;
  style.FrameRounding = 2.0f;
  style.ScrollbarRounding = 2.0f;
  style.GrabRounding = 2.0f;
  style.TabRounding = 0.0f;
  style.WindowBorderSize = 1.0f;
  style.ChildBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.WindowPadding = ImVec2(8.0f, 7.0f);
  style.FramePadding = ImVec2(6.0f, 3.0f);
  style.ItemSpacing = ImVec2(8.0f, 5.0f);
  style.ItemInnerSpacing = ImVec2(6.0f, 3.0f);
  for (const ImGuiThemeColor &color : IMGUI_COLORS) {
    style.Colors[color.color] = app_dark_mode() ? color.dark : color.light;
  }

  ImPlotStyle &plot_style = ImPlot::GetStyle();
  plot_style.PlotBorderSize = 1.0f;
  plot_style.MinorAlpha = app_dark_mode() ? 0.50f : 0.65f;
  plot_style.LegendPadding = ImVec2(6.0f, 5.0f);
  plot_style.LegendInnerPadding = ImVec2(6.0f, 3.0f);
  plot_style.LegendSpacing = ImVec2(7.0f, 2.0f);
  plot_style.PlotPadding = ImVec2(4.0f, 8.0f);
  plot_style.FitPadding = ImVec2(0.02f, static_cast<float>(PLOT_Y_PADDING_FRACTION));
}
