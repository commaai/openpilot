#include "tools/jotpluggler/style.h"

#include "implot.h"

namespace {

AppTheme current_theme = AppTheme::Light;

ImVec4 rgb(int r, int g, int b, float a = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a);
}

struct ThemeColor {
  AppColor color;
  ImVec4 light;
  ImVec4 dark;
};

const ThemeColor APP_COLORS[] = {
  {AppColor::Clear, rgb(227, 229, 233), rgb(18, 21, 25)},
  {AppColor::FpsBg, rgb(248, 249, 251, 0.92f), rgb(28, 33, 39, 0.94f)},
  {AppColor::FpsBorder, rgb(182, 188, 196, 0.95f), rgb(74, 84, 96, 0.95f)},
  {AppColor::FpsText, rgb(57, 62, 69), rgb(226, 231, 237)},
  {AppColor::PlotBg, rgb(255, 255, 255), rgb(17, 20, 24)},
  {AppColor::PlotBorder, rgb(186, 190, 196), rgb(67, 76, 87)},
  {AppColor::PlotLegendBg, rgb(248, 249, 251, 0.92f), rgb(26, 31, 37, 0.94f)},
  {AppColor::PlotLegendBorder, rgb(168, 175, 184), rgb(78, 89, 102)},
  {AppColor::PlotText, rgb(57, 62, 69), rgb(224, 230, 237)},
  {AppColor::PlotMutedText, rgb(95, 103, 112), rgb(157, 168, 181)},
  {AppColor::PlotGrid, rgb(188, 196, 206), rgb(64, 73, 84)},
  {AppColor::PlotAxisBg, rgb(255, 255, 255, 0.0f), rgb(17, 20, 24, 0.0f)},
  {AppColor::PlotAxisBgHovered, rgb(214, 220, 228, 0.45f), rgb(75, 91, 108, 0.38f)},
  {AppColor::PlotAxisBgActive, rgb(199, 209, 222, 0.55f), rgb(92, 110, 130, 0.48f)},
  {AppColor::PlotSelection, rgb(252, 211, 77, 0.28f), rgb(250, 204, 21, 0.30f)},
  {AppColor::PlotCrosshairs, rgb(120, 128, 138, 0.70f), rgb(174, 185, 197, 0.72f)},
  {AppColor::PlotCursor, rgb(108, 118, 128, 0.70f), rgb(180, 190, 202, 0.74f)},
};

struct ImGuiThemeColor {
  ImGuiCol color;
  ImVec4 light;
  ImVec4 dark;
};

const ImGuiThemeColor IMGUI_COLORS[] = {
  {ImGuiCol_WindowBg, rgb(250, 250, 251), rgb(20, 24, 29)},
  {ImGuiCol_ChildBg, rgb(255, 255, 255), rgb(18, 22, 27)},
  {ImGuiCol_Border, rgb(194, 198, 204), rgb(58, 66, 75)},
  {ImGuiCol_TitleBg, rgb(252, 252, 253), rgb(26, 30, 36)},
  {ImGuiCol_TitleBgActive, rgb(252, 252, 253), rgb(31, 36, 43)},
  {ImGuiCol_TitleBgCollapsed, rgb(252, 252, 253), rgb(24, 28, 33)},
  {ImGuiCol_Text, rgb(74, 80, 88), rgb(221, 227, 234)},
  {ImGuiCol_TextDisabled, rgb(108, 118, 128), rgb(134, 145, 158)},
  {ImGuiCol_Button, rgb(255, 255, 255), rgb(33, 39, 47)},
  {ImGuiCol_ButtonHovered, rgb(246, 248, 250), rgb(44, 52, 62)},
  {ImGuiCol_ButtonActive, rgb(238, 240, 244), rgb(55, 65, 77)},
  {ImGuiCol_FrameBg, rgb(255, 255, 255), rgb(27, 32, 39)},
  {ImGuiCol_FrameBgHovered, rgb(248, 249, 251), rgb(37, 45, 54)},
  {ImGuiCol_FrameBgActive, rgb(241, 244, 248), rgb(47, 57, 68)},
  {ImGuiCol_Header, rgb(243, 245, 248), rgb(35, 42, 50)},
  {ImGuiCol_HeaderHovered, rgb(237, 240, 244), rgb(45, 54, 65)},
  {ImGuiCol_HeaderActive, rgb(232, 236, 240), rgb(55, 66, 79)},
  {ImGuiCol_PopupBg, rgb(248, 249, 251), rgb(24, 29, 35)},
  {ImGuiCol_MenuBarBg, rgb(232, 236, 241), rgb(24, 29, 35)},
  {ImGuiCol_Separator, rgb(194, 198, 204), rgb(62, 71, 82)},
  {ImGuiCol_DockingEmptyBg, rgb(244, 246, 248), rgb(18, 21, 25)},
  {ImGuiCol_DockingPreview, rgb(69, 115, 184, 0.22f), rgb(99, 148, 220, 0.28f)},
};

}  // namespace

bool app_dark_mode() {
  return current_theme == AppTheme::Dark;
}

ImVec4 app_color(AppColor color) {
  for (const ThemeColor &theme_color : APP_COLORS) {
    if (theme_color.color == color) {
      return app_dark_mode() ? theme_color.dark : theme_color.light;
    }
  }
  return ImVec4(1.0f, 0.0f, 1.0f, 1.0f);
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
  plot_style.FitPadding = ImVec2(0.02f, 0.05f);
}
