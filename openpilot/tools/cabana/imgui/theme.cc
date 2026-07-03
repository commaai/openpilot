#include "tools/cabana/imgui/app.h"

#include <cmath>

#include "implot.h"

namespace fs = std::filesystem;

namespace {

ImFont *g_ui_font = nullptr;
ImFont *g_ui_bold_font = nullptr;
ImFont *g_mono_font = nullptr;
Theme g_theme = Theme::Light;

constexpr float UI_FONT_SIZE = 15.0f;
constexpr float BOLD_FONT_SIZE = 15.5f;
constexpr float MONO_FONT_SIZE = 14.5f;

inline ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

void icon_add_font(float size, bool merge = false, const ImFont *base_font = nullptr) {
  const fs::path ttf = BOOTSTRAP_ICONS_TTF;
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

struct ColorDef {
  ImGuiCol idx;
  int r, g, b;
};

constexpr ColorDef LIGHT_COLORS[] = {
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

// "Darcula" like dark theme, matching tools/cabana/utils/util.cc setTheme()
constexpr ColorDef DARK_COLORS[] = {
  {ImGuiCol_WindowBg, 53, 53, 53},     {ImGuiCol_ChildBg, 60, 63, 65},
  {ImGuiCol_Border, 85, 85, 85},       {ImGuiCol_TitleBg, 43, 43, 43},
  {ImGuiCol_TitleBgActive, 43, 43, 43}, {ImGuiCol_TitleBgCollapsed, 43, 43, 43},
  {ImGuiCol_Text, 187, 187, 187},      {ImGuiCol_TextDisabled, 119, 119, 119},
  {ImGuiCol_Button, 60, 63, 65},       {ImGuiCol_ButtonHovered, 71, 75, 77},
  {ImGuiCol_ButtonActive, 79, 84, 86}, {ImGuiCol_FrameBg, 60, 63, 65},
  {ImGuiCol_FrameBgHovered, 69, 73, 75}, {ImGuiCol_FrameBgActive, 76, 80, 82},
  {ImGuiCol_Header, 47, 101, 202},     {ImGuiCol_HeaderHovered, 57, 111, 212},
  {ImGuiCol_HeaderActive, 42, 91, 182}, {ImGuiCol_PopupBg, 60, 63, 65},
  {ImGuiCol_MenuBarBg, 60, 63, 65},    {ImGuiCol_Separator, 85, 85, 85},
  {ImGuiCol_ScrollbarBg, 53, 53, 53},  {ImGuiCol_ScrollbarGrab, 92, 96, 98},
  {ImGuiCol_ScrollbarGrabHovered, 108, 112, 114}, {ImGuiCol_ScrollbarGrabActive, 120, 124, 126},
  {ImGuiCol_Tab, 49, 52, 54},          {ImGuiCol_TabHovered, 66, 70, 72},
  {ImGuiCol_TabSelected, 60, 63, 65},  {ImGuiCol_TabSelectedOverline, 47, 101, 202},
  {ImGuiCol_TabDimmed, 45, 47, 49},    {ImGuiCol_TabDimmedSelected, 55, 58, 60},
  {ImGuiCol_TabDimmedSelectedOverline, 47, 101, 202}, {ImGuiCol_DockingEmptyBg, 47, 47, 47},
  {ImGuiCol_CheckMark, 187, 187, 187}, {ImGuiCol_SliderGrab, 130, 135, 140},
  {ImGuiCol_SliderGrabActive, 150, 155, 160},
};

}  // namespace

const fs::path &repo_root() {
  static const fs::path root = CABANA_REPO_ROOT;
  return root;
}

void load_fonts() {
  ImGuiIO &io = ImGui::GetIO();
  const fs::path fonts_dir = repo_root() / "openpilot" / "selfdrive" / "assets" / "fonts";
  ImFontConfig font_cfg;
  font_cfg.OversampleH = 2;
  font_cfg.OversampleV = 2;
  font_cfg.RasterizerDensity = 1.0f;
  icon_add_font(UI_FONT_SIZE);
  const auto add_font_with_icons = [&](const fs::path &path, float size) -> ImFont * {
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

void apply_theme(Theme theme) {
  g_theme = theme;

  ImGuiStyle &style = ImGui::GetStyle();
  if (theme == Theme::Dark) {
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();
  } else {
    ImGui::StyleColorsLight();
    ImPlot::StyleColorsLight();
  }

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

  if (theme == Theme::Dark) {
    for (const auto &c : DARK_COLORS) style.Colors[c.idx] = color_rgb(c.r, c.g, c.b);
    style.Colors[ImGuiCol_DockingPreview] = color_rgb(47, 101, 202, 0.22f);
  } else {
    for (const auto &c : LIGHT_COLORS) style.Colors[c.idx] = color_rgb(c.r, c.g, c.b);
    style.Colors[ImGuiCol_DockingPreview] = color_rgb(69, 115, 184, 0.22f);
  }

  ImPlotStyle &plot_style = ImPlot::GetStyle();
  plot_style.PlotBorderSize = 1.0f;
  plot_style.MinorAlpha = 0.65f;
  plot_style.LegendPadding = ImVec2(6.0f, 5.0f);
  plot_style.LegendInnerPadding = ImVec2(6.0f, 3.0f);
  plot_style.LegendSpacing = ImVec2(7.0f, 2.0f);
  plot_style.PlotPadding = ImVec2(4.0f, 8.0f);
  plot_style.FitPadding = ImVec2(0.02f, 0.05f);

  ImPlot::MapInputDefault();
  ImPlotInputMap &input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Right;
  input_map.PanMod = ImGuiMod_None;
  input_map.Select = ImGuiMouseButton_Left;
  input_map.SelectCancel = ImGuiMouseButton_Right;
  input_map.SelectMod = ImGuiMod_None;
}

ImVec4 theme_clear_color() {
  return g_theme == Theme::Dark ? color_rgb(43, 43, 43) : color_rgb(227, 229, 233);
}

void push_bold_font(float size) {
  ImGui::PushFont(g_ui_bold_font, size > 0.0f ? size : BOLD_FONT_SIZE);
}

void pop_bold_font() {
  ImGui::PopFont();
}

void push_mono_font(float size) {
  ImGui::PushFont(g_mono_font, size > 0.0f ? size : MONO_FONT_SIZE);
}

void pop_mono_font() {
  ImGui::PopFont();
}
