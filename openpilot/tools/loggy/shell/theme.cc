#include "tools/loggy/shell/theme.h"

#include <cmath>

#include "implot.h"

namespace loggy {
namespace {

ImFont *g_ui_font = nullptr;
ImFont *g_ui_bold_font = nullptr;
ImFont *g_mono_font = nullptr;

constexpr float UI_FONT_SIZE = 15.0f;
constexpr float BOLD_FONT_SIZE = 15.5f;
constexpr float MONO_FONT_SIZE = 14.5f;

struct ColorDef {
  ImGuiCol idx;
  int r;
  int g;
  int b;
};

constexpr ColorDef COLORS[] = {
  {ImGuiCol_WindowBg, 53, 53, 53},       {ImGuiCol_ChildBg, 60, 63, 65},
  {ImGuiCol_Border, 85, 85, 85},         {ImGuiCol_TitleBg, 43, 43, 43},
  {ImGuiCol_TitleBgActive, 43, 43, 43},  {ImGuiCol_TitleBgCollapsed, 43, 43, 43},
  {ImGuiCol_Text, 187, 187, 187},        {ImGuiCol_TextDisabled, 119, 119, 119},
  {ImGuiCol_Button, 60, 63, 65},         {ImGuiCol_ButtonHovered, 71, 75, 77},
  {ImGuiCol_ButtonActive, 79, 84, 86},   {ImGuiCol_FrameBg, 60, 63, 65},
  {ImGuiCol_FrameBgHovered, 69, 73, 75}, {ImGuiCol_FrameBgActive, 76, 80, 82},
  {ImGuiCol_Header, 47, 101, 202},       {ImGuiCol_HeaderHovered, 57, 111, 212},
  {ImGuiCol_HeaderActive, 42, 91, 182},  {ImGuiCol_PopupBg, 60, 63, 65},
  {ImGuiCol_MenuBarBg, 53, 53, 53},      {ImGuiCol_Separator, 85, 85, 85},
  {ImGuiCol_ScrollbarBg, 53, 53, 53},    {ImGuiCol_ScrollbarGrab, 92, 96, 98},
  {ImGuiCol_ScrollbarGrabHovered, 108, 112, 114},
  {ImGuiCol_ScrollbarGrabActive, 120, 124, 126},
  {ImGuiCol_Tab, 49, 52, 54},            {ImGuiCol_TabHovered, 66, 70, 72},
  {ImGuiCol_TabSelected, 60, 63, 65},    {ImGuiCol_TabSelectedOverline, 47, 101, 202},
  {ImGuiCol_TabDimmed, 45, 47, 49},      {ImGuiCol_TabDimmedSelected, 55, 58, 60},
  {ImGuiCol_TabDimmedSelectedOverline, 47, 101, 202},
  {ImGuiCol_DockingEmptyBg, 47, 47, 47}, {ImGuiCol_CheckMark, 187, 187, 187},
  {ImGuiCol_SliderGrab, 47, 101, 202},   {ImGuiCol_SliderGrabActive, 57, 111, 212},
};

void icon_add_font(float size, bool merge = false, const ImFont *base_font = nullptr) {
  const std::filesystem::path ttf = BOOTSTRAP_ICONS_TTF;
  ImGuiIO &io = ImGui::GetIO();
  ImFontConfig config;
  config.MergeMode = merge;
  config.GlyphMinAdvanceX = size;
  if (base_font != nullptr) {
    ImFontBaked *baked = const_cast<ImFont *>(base_font)->GetFontBaked(size);
    const float base_center = baked != nullptr ? (baked->Ascent + baked->Descent) * 0.5f : size * 0.5f;
    config.GlyphOffset.y = std::round(size * 0.5f - base_center);
  }
  static const ImWchar ranges[] = {0xF000, 0xF8FF, 0};
  io.Fonts->AddFontFromFileTTF(ttf.c_str(), size, &config, ranges);
}

}  // namespace

const std::filesystem::path &repo_root() {
  static const std::filesystem::path root = LOGGY_REPO_ROOT;
  return root;
}

ImVec4 color_rgb(int r, int g, int b, float alpha) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

void load_fonts() {
  ImGuiIO &io = ImGui::GetIO();
  const std::filesystem::path fonts_dir = repo_root() / "openpilot" / "selfdrive" / "assets" / "fonts";
  ImFontConfig font_cfg;
  font_cfg.OversampleH = 2;
  font_cfg.OversampleV = 2;
  font_cfg.RasterizerDensity = 1.0f;

  icon_add_font(UI_FONT_SIZE);
  const auto add_font_with_icons = [&](const std::filesystem::path &path, float size) -> ImFont * {
    ImFont *font = io.Fonts->AddFontFromFileTTF(path.c_str(), size, &font_cfg);
    if (font != nullptr) {
      icon_add_font(size, true, font);
    }
    return font;
  };

  if (ImFont *font = add_font_with_icons(fonts_dir / "Inter-Regular.ttf", UI_FONT_SIZE); font != nullptr) {
    g_ui_font = font;
    io.FontDefault = font;
  }
  g_ui_bold_font = add_font_with_icons(fonts_dir / "Inter-SemiBold.ttf", BOLD_FONT_SIZE);
  g_mono_font = add_font_with_icons(fonts_dir / "JetBrainsMono-Medium.ttf", MONO_FONT_SIZE);
  if (g_ui_bold_font == nullptr) g_ui_bold_font = g_ui_font;
  if (g_mono_font == nullptr) g_mono_font = g_ui_font;
}

void apply_theme() {
  ImGui::StyleColorsDark();
  ImPlot::StyleColorsDark();

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

  for (const auto &c : COLORS) {
    style.Colors[c.idx] = color_rgb(c.r, c.g, c.b);
  }
  style.Colors[ImGuiCol_DockingPreview] = color_rgb(47, 101, 202, 0.22f);

  ImPlotStyle &plot_style = ImPlot::GetStyle();
  plot_style.PlotBorderSize = 1.0f;
  plot_style.MinorAlpha = 0.65f;
  plot_style.LegendPadding = ImVec2(6.0f, 5.0f);
  plot_style.LegendInnerPadding = ImVec2(6.0f, 3.0f);
  plot_style.LegendSpacing = ImVec2(7.0f, 2.0f);
  plot_style.PlotPadding = ImVec2(4.0f, 8.0f);
  plot_style.FitPadding = ImVec2(0.02f, 0.05f);
  plot_style.Colors[ImPlotCol_FrameBg] = color_rgb(60, 63, 65);
  plot_style.Colors[ImPlotCol_PlotBg] = color_rgb(60, 63, 65);

  ImPlot::MapInputDefault();
  ImPlotInputMap &input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Right;
  input_map.PanMod = ImGuiMod_None;
  input_map.Select = ImGuiMouseButton_Left;
  input_map.SelectCancel = ImGuiMouseButton_Right;
  input_map.SelectMod = ImGuiMod_None;
}

ImVec4 clear_color() {
  return color_rgb(43, 43, 43);
}

void push_bold_font() {
  if (g_ui_bold_font != nullptr) ImGui::PushFont(g_ui_bold_font);
}

void pop_bold_font() {
  if (g_ui_bold_font != nullptr) ImGui::PopFont();
}

void push_mono_font() {
  if (g_mono_font != nullptr) ImGui::PushFont(g_mono_font);
}

void pop_mono_font() {
  if (g_mono_font != nullptr) ImGui::PopFont();
}

}  // namespace loggy
