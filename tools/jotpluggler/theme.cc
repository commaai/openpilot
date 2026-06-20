#include "tools/jotpluggler/theme.h"
#include "tools/jotpluggler/util.h"

#include <cstddef>

namespace {

struct ColorDef {
  ImGuiCol idx;
  int r;
  int g;
  int b;
};

constexpr ColorDef kLightImGuiColors[] = {
  {ImGuiCol_WindowBg, 250, 250, 251},  {ImGuiCol_ChildBg, 255, 255, 255},
  {ImGuiCol_Border, 194, 198, 204},    {ImGuiCol_TitleBg, 252, 252, 253},
  {ImGuiCol_TitleBgActive, 252, 252, 253}, {ImGuiCol_TitleBgCollapsed, 252, 252, 253},
  {ImGuiCol_Text, 74, 80, 88},         {ImGuiCol_TextDisabled, 108, 118, 128},
  {ImGuiCol_Button, 255, 255, 255},    {ImGuiCol_ButtonHovered, 246, 248, 250},
  {ImGuiCol_ButtonActive, 238, 240, 244}, {ImGuiCol_FrameBg, 255, 255, 255},
  {ImGuiCol_FrameBgHovered, 248, 249, 251}, {ImGuiCol_FrameBgActive, 241, 244, 248},
  {ImGuiCol_Header, 243, 245, 248},    {ImGuiCol_HeaderHovered, 237, 240, 244},
  {ImGuiCol_HeaderActive, 232, 236, 240}, {ImGuiCol_PopupBg, 248, 249, 251},
  {ImGuiCol_MenuBarBg, 232, 236, 241}, {ImGuiCol_Separator, 194, 198, 204},
  {ImGuiCol_ScrollbarBg, 240, 242, 245}, {ImGuiCol_ScrollbarGrab, 202, 207, 214},
  {ImGuiCol_ScrollbarGrabHovered, 180, 186, 194}, {ImGuiCol_ScrollbarGrabActive, 164, 171, 180},
  {ImGuiCol_Tab, 219, 224, 230},       {ImGuiCol_TabHovered, 232, 236, 241},
  {ImGuiCol_TabSelected, 250, 251, 253}, {ImGuiCol_TabSelectedOverline, 92, 109, 136},
  {ImGuiCol_TabDimmed, 213, 219, 226}, {ImGuiCol_TabDimmedSelected, 244, 247, 249},
  {ImGuiCol_TabDimmedSelectedOverline, 92, 109, 136}, {ImGuiCol_DockingEmptyBg, 244, 246, 248},
};

constexpr ColorDef kDarkImGuiColors[] = {
  {ImGuiCol_WindowBg, 18, 20, 23},     {ImGuiCol_ChildBg, 24, 26, 30},
  {ImGuiCol_Border, 55, 60, 68},       {ImGuiCol_TitleBg, 20, 22, 26},
  {ImGuiCol_TitleBgActive, 22, 24, 28},{ImGuiCol_TitleBgCollapsed, 20, 22, 26},
  {ImGuiCol_Text, 210, 214, 220},      {ImGuiCol_TextDisabled, 120, 128, 138},
  {ImGuiCol_Button, 34, 38, 44},       {ImGuiCol_ButtonHovered, 44, 49, 57},
  {ImGuiCol_ButtonActive, 54, 60, 70}, {ImGuiCol_FrameBg, 28, 31, 36},
  {ImGuiCol_FrameBgHovered, 35, 39, 46},{ImGuiCol_FrameBgActive, 42, 47, 55},
  {ImGuiCol_Header, 36, 41, 49},       {ImGuiCol_HeaderHovered, 44, 50, 60},
  {ImGuiCol_HeaderActive, 52, 59, 70}, {ImGuiCol_PopupBg, 24, 27, 32},
  {ImGuiCol_MenuBarBg, 22, 25, 30},    {ImGuiCol_Separator, 55, 60, 68},
  {ImGuiCol_ScrollbarBg, 20, 23, 27},  {ImGuiCol_ScrollbarGrab, 60, 66, 76},
  {ImGuiCol_ScrollbarGrabHovered, 76, 83, 95},{ImGuiCol_ScrollbarGrabActive, 92, 100, 114},
  {ImGuiCol_Tab, 28, 32, 38},          {ImGuiCol_TabHovered, 40, 46, 54},
  {ImGuiCol_TabSelected, 34, 39, 47},  {ImGuiCol_TabSelectedOverline, 92, 139, 214},
  {ImGuiCol_TabDimmed, 24, 27, 33},    {ImGuiCol_TabDimmedSelected, 30, 35, 42},
  {ImGuiCol_TabDimmedSelectedOverline, 70, 100, 160},{ImGuiCol_DockingEmptyBg, 14, 16, 19},
};

void apply_imgui_colors(const ColorDef *colors, size_t count, int preview_r, int preview_g, int preview_b) {
  ImGuiStyle &style = ImGui::GetStyle();
  for (size_t i = 0; i < count; ++i) {
    style.Colors[colors[i].idx] = color_rgb(colors[i].r, colors[i].g, colors[i].b);
  }
  style.Colors[ImGuiCol_DockingPreview] = color_rgb(preview_r, preview_g, preview_b, 0.22f);
}

}  // namespace

void apply_dark_theme() {
  ImGui::StyleColorsDark();
  ImPlot::StyleColorsDark();
  apply_imgui_colors(kDarkImGuiColors, std::size(kDarkImGuiColors), 92, 139, 214);
}

void apply_light_theme() {
  ImGui::StyleColorsLight();
  ImPlot::StyleColorsLight();
  apply_imgui_colors(kLightImGuiColors, std::size(kLightImGuiColors), 69, 115, 184);
}

void push_sidebar_style(bool dark_mode) {
  if (dark_mode) {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(26, 29, 33));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(55, 60, 68));
  } else {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(238, 240, 244));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(190, 197, 205));
  }
}

void push_pane_style(bool dark_mode) {
  if (dark_mode) {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(24, 26, 30));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(55, 60, 68));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, color_rgb(22, 24, 28));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, color_rgb(22, 24, 28));
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, color_rgb(22, 24, 28));
  } else {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(250, 250, 251));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(194, 198, 204));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, color_rgb(252, 252, 253));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, color_rgb(252, 252, 253));
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, color_rgb(252, 252, 253));
  }
}

void push_workspace_style(bool dark_mode) {
  if (dark_mode) {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(14, 16, 19));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(50, 55, 63));
  } else {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(244, 246, 248));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(186, 191, 198));
  }
}

void push_new_tab_button_style(bool dark_mode) {
  if (dark_mode) {
    ImGui::PushStyleColor(ImGuiCol_Tab, color_rgb(28, 32, 38));
    ImGui::PushStyleColor(ImGuiCol_TabHovered, color_rgb(40, 46, 54));
    ImGui::PushStyleColor(ImGuiCol_TabSelected, color_rgb(34, 39, 47));
  } else {
    ImGui::PushStyleColor(ImGuiCol_Tab, color_rgb(210, 217, 225));
    ImGui::PushStyleColor(ImGuiCol_TabHovered, color_rgb(224, 230, 237));
    ImGui::PushStyleColor(ImGuiCol_TabSelected, color_rgb(242, 245, 248));
  }
}

ImU32 new_tab_icon_color(bool dark_mode) {
  return ImGui::GetColorU32(dark_mode ? color_rgb(200, 207, 215) : color_rgb(72, 79, 88));
}

void push_plot_style(bool dark_mode) {
  if (dark_mode) {
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, color_rgb(18, 18, 22));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, color_rgb(52, 58, 70));
    ImPlot::PushStyleColor(ImPlotCol_LegendBg, color_rgb(28, 30, 36, 0.92f));
    ImPlot::PushStyleColor(ImPlotCol_LegendBorder, color_rgb(64, 72, 84));
    ImPlot::PushStyleColor(ImPlotCol_LegendText, color_rgb(230, 232, 235));
    ImPlot::PushStyleColor(ImPlotCol_TitleText, color_rgb(240, 242, 245));
    ImPlot::PushStyleColor(ImPlotCol_InlayText, color_rgb(160, 167, 176));
    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, color_rgb(58, 64, 76));
    ImPlot::PushStyleColor(ImPlotCol_AxisText, color_rgb(170, 176, 184));
    ImPlot::PushStyleColor(ImPlotCol_AxisBg, color_rgb(0, 0, 0, 0.0f));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgHovered, color_rgb(80, 88, 102, 0.35f));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgActive, color_rgb(96, 106, 122, 0.45f));
    ImPlot::PushStyleColor(ImPlotCol_Selection, color_rgb(255, 196, 64, 0.22f));
    ImPlot::PushStyleColor(ImPlotCol_Crosshairs, color_rgb(190, 196, 204, 0.55f));
  } else {
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, color_rgb(255, 255, 255));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, color_rgb(186, 190, 196));
    ImPlot::PushStyleColor(ImPlotCol_LegendBg, color_rgb(248, 249, 251, 0.92f));
    ImPlot::PushStyleColor(ImPlotCol_LegendBorder, color_rgb(168, 175, 184));
    ImPlot::PushStyleColor(ImPlotCol_LegendText, color_rgb(57, 62, 69));
    ImPlot::PushStyleColor(ImPlotCol_TitleText, color_rgb(57, 62, 69));
    ImPlot::PushStyleColor(ImPlotCol_InlayText, color_rgb(95, 103, 112));
    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, color_rgb(188, 196, 206));
    ImPlot::PushStyleColor(ImPlotCol_AxisText, color_rgb(95, 103, 112));
    ImPlot::PushStyleColor(ImPlotCol_AxisBg, color_rgb(255, 255, 255, 0.0f));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgHovered, color_rgb(214, 220, 228, 0.45f));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgActive, color_rgb(199, 209, 222, 0.55f));
    ImPlot::PushStyleColor(ImPlotCol_Selection, color_rgb(252, 211, 77, 0.28f));
    ImPlot::PushStyleColor(ImPlotCol_Crosshairs, color_rgb(120, 128, 138, 0.70f));
  }
  ImPlot::PushStyleVar(ImPlotStyleVar_LegendPadding, ImVec2(56.0f, 10.0f));
}
