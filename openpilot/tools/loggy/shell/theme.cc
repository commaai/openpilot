#include "tools/loggy/shell/theme.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <vector>

#include "implot.h"

#if __has_include("misc/cpp/imgui_stdlib.h")
#include "misc/cpp/imgui_stdlib.h"
#endif

namespace loggy {
namespace {

ImFont *g_ui_font = nullptr;
ImFont *g_ui_bold_font = nullptr;
ImFont *g_mono_font = nullptr;
ThemeKind g_current_theme = ThemeKind::Light;

constexpr float UI_FONT_SIZE = 15.0f;
constexpr float BOLD_FONT_SIZE = 15.5f;
constexpr float MONO_FONT_SIZE = 14.5f;

ImVec4 rgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

// Pure data. Sampled/measured against the reference tools (Qt cabana's Darcula skin; jotpluggler
// screenshots pixel-picked for Light — see tools/loggy/REVIEW.md item 8). Ports translate at the
// boundary: numbers only, no logic.
const Theme kDarculaTheme = {
  .window_bg = rgb(53, 53, 53),
  .child_bg = rgb(60, 63, 65),
  .border = rgb(85, 85, 85),
  .title_bg = rgb(43, 43, 43),
  .title_bg_active = rgb(43, 43, 43),
  .title_bg_collapsed = rgb(43, 43, 43),
  .button = rgb(60, 63, 65),
  .button_hovered = rgb(71, 75, 77),
  .button_active = rgb(79, 84, 86),
  .frame_bg = rgb(60, 63, 65),
  .frame_bg_hovered = rgb(69, 73, 75),
  .frame_bg_active = rgb(76, 80, 82),
  .popup_bg = rgb(60, 63, 65),
  .menu_bar_bg = rgb(53, 53, 53),
  .separator = rgb(85, 85, 85),
  .scrollbar_bg = rgb(53, 53, 53),
  .scrollbar_grab = rgb(92, 96, 98),
  .scrollbar_grab_hovered = rgb(108, 112, 114),
  .scrollbar_grab_active = rgb(120, 124, 126),
  .tab = rgb(49, 52, 54),
  .tab_hovered = rgb(66, 70, 72),
  .tab_selected = rgb(60, 63, 65),
  .tab_dimmed = rgb(45, 47, 49),
  .tab_dimmed_selected = rgb(55, 58, 60),
  .docking_empty_bg = rgb(47, 47, 47),
  .docking_preview = rgb(47, 101, 202, 0.22f),
  .chrome_border = rgb(92, 96, 98),

  .text = rgb(187, 187, 187),
  .text_muted = rgb(119, 119, 119),

  .accent = rgb(47, 101, 202),
  .accent_active = rgb(57, 111, 212),
  .check_mark = rgb(187, 187, 187),
  .accent_soft = rgb(47, 101, 202),
  .accent_soft_hovered = rgb(57, 111, 212),
  .accent_soft_active = rgb(42, 91, 182),

  .table_header_bg = rgb(49, 52, 54),
  .table_border_strong = rgb(85, 85, 85),
  .table_border_light = rgb(70, 72, 74),
  .table_row_bg_alt = rgb(255, 255, 255, 0.03f),

  .plot_bg = rgb(47, 49, 51),
  .plot_legend_bg = rgb(53, 53, 53, 0.55f),
  .plot_legend_border = rgb(92, 96, 98, 0.85f),
  .plot_grid = rgb(92, 96, 98, 0.70f),
  .plot_crosshair = rgb(187, 187, 187, 0.70f),
  .plot_tracker_line = rgb(220, 220, 220, 0.72f),
  .plot_drop_target_fill = rgb(47, 101, 202, 0.16f),
  .plot_drop_target_border = rgb(75, 135, 230, 0.90f),
  .plot_selection = rgb(47, 101, 202, 0.25f),
  .plot_series_palette = {
    rgb(89, 168, 250),
    rgb(84, 199, 140),
    rgb(250, 179, 77),
    rgb(217, 122, 199),
    rgb(240, 97, 97),
  },

  .binary_idle_cell = rgb(68, 71, 73),
  .binary_suppressed_cell = rgb(45, 48, 50),
  .binary_heat_accent = rgb(47, 101, 202),
  .binary_drag_selection = rgb(82, 141, 255, 0.42f),

  .transport_bg = rgb(47, 47, 47),
  .transport_border = rgb(85, 85, 85),
  .transport_tracker = rgb(235, 235, 235),

  .hud_bg = rgb(43, 43, 43, 0.92f),
  .hud_border = rgb(100, 100, 100, 0.95f),
  .hud_text = rgb(220, 220, 220),

  .sparkline_bg = rgb(48, 51, 53),
  .sparkline_border = rgb(82, 86, 88),
  .sparkline_line = rgb(116, 178, 255),

  .camera_video_border = rgb(116, 178, 255, 0.75f),
  .camera_overlay_bg = rgb(20, 22, 24, 0.60f),
  .camera_overlay_border = rgb(84, 87, 89, 0.85f),

  .map_bg = rgb(43, 45, 47),
  .map_grid = rgb(72, 77, 80, 0.76f),
  .map_water_fill = rgb(54, 93, 117, 0.42f),
  .map_water_outline = rgb(88, 139, 168, 0.70f),
  .map_water_line = rgb(96, 151, 181, 0.74f),
  .map_marker = rgb(238, 188, 82),
  .map_marker_outline = rgb(30, 32, 34),
  .map_road_motorway_casing = rgb(92, 91, 86, 0.78f),
  .map_road_motorway_fill = rgb(213, 199, 158, 0.92f),
  .map_road_primary_casing = rgb(80, 84, 86, 0.74f),
  .map_road_primary_fill = rgb(202, 207, 202, 0.88f),
  .map_road_secondary_casing = rgb(74, 80, 82, 0.68f),
  .map_road_secondary_fill = rgb(184, 193, 190, 0.82f),
  .map_road_local_casing = rgb(68, 74, 77, 0.62f),
  .map_road_local_fill = rgb(151, 162, 163, 0.72f),
};

// Light is the default theme now (item 9a) and is sampled directly from the jotpluggler
// reference screenshots (tools/loggy/REVIEW.md item 8): menu/chrome ~ rgb(232,236,241), content
// panels near-white, borders ~ rgb(190,198,204), and — notably — selected rows are barely tinted
// rather than a saturated blue, so accent_soft stays close to child_bg instead of copying
// Darcula's vivid Header blue.
const Theme kLightTheme = {
  .window_bg = rgb(233, 236, 240),
  .child_bg = rgb(250, 250, 251),
  .border = rgb(190, 195, 201),
  .title_bg = rgb(226, 229, 233),
  .title_bg_active = rgb(214, 220, 227),
  .title_bg_collapsed = rgb(230, 233, 236),
  .button = rgb(234, 236, 239),
  .button_hovered = rgb(222, 228, 234),
  .button_active = rgb(206, 215, 225),
  .frame_bg = rgb(252, 253, 254),
  .frame_bg_hovered = rgb(234, 238, 242),
  .frame_bg_active = rgb(222, 228, 235),
  .popup_bg = rgb(250, 250, 251),
  .menu_bar_bg = rgb(232, 235, 238),
  .separator = rgb(190, 195, 201),
  .scrollbar_bg = rgb(231, 234, 237),
  .scrollbar_grab = rgb(184, 193, 202),
  .scrollbar_grab_hovered = rgb(160, 171, 183),
  .scrollbar_grab_active = rgb(139, 152, 166),
  .tab = rgb(224, 228, 232),
  .tab_hovered = rgb(208, 219, 230),
  .tab_selected = rgb(250, 251, 253),
  .tab_dimmed = rgb(217, 221, 225),
  .tab_dimmed_selected = rgb(235, 238, 241),
  .docking_empty_bg = rgb(236, 238, 240),
  .docking_preview = rgb(74, 132, 214, 0.22f),
  .chrome_border = rgb(194, 198, 204),

  .text = rgb(43, 47, 51),
  .text_muted = rgb(111, 119, 128),

  .accent = rgb(74, 132, 214),
  .accent_active = rgb(58, 116, 196),
  .check_mark = rgb(58, 116, 196),
  .accent_soft = rgb(222, 227, 233),
  .accent_soft_hovered = rgb(208, 216, 225),
  .accent_soft_active = rgb(195, 206, 218),

  .table_header_bg = rgb(224, 228, 232),
  .table_border_strong = rgb(190, 195, 201),
  .table_border_light = rgb(208, 212, 216),
  .table_row_bg_alt = rgb(0, 0, 0, 0.03f),

  .plot_bg = rgb(248, 249, 250),
  .plot_legend_bg = rgb(250, 250, 251, 0.55f),
  .plot_legend_border = rgb(194, 198, 204, 0.85f),
  .plot_grid = rgb(206, 211, 216, 0.55f),
  .plot_crosshair = rgb(60, 64, 68, 0.45f),
  .plot_tracker_line = rgb(60, 64, 68, 0.55f),
  .plot_drop_target_fill = rgb(74, 132, 214, 0.14f),
  .plot_drop_target_border = rgb(58, 116, 196, 0.85f),
  .plot_selection = rgb(74, 132, 214, 0.22f),
  .plot_series_palette = {
    rgb(43, 105, 199),
    rgb(41, 143, 97),
    rgb(199, 128, 25),
    rgb(163, 66, 148),
    rgb(191, 59, 59),
  },

  .binary_idle_cell = rgb(234, 237, 240),
  .binary_suppressed_cell = rgb(215, 218, 222),
  .binary_heat_accent = rgb(74, 132, 214),
  .binary_drag_selection = rgb(58, 116, 196, 0.38f),

  .transport_bg = rgb(222, 225, 229),
  .transport_border = rgb(190, 195, 201),
  .transport_tracker = rgb(60, 64, 68),

  .hud_bg = rgb(255, 255, 255, 0.85f),
  .hud_border = rgb(181, 188, 196, 0.90f),
  .hud_text = rgb(43, 47, 51),

  .sparkline_bg = rgb(234, 237, 240),
  .sparkline_border = rgb(194, 198, 204),
  .sparkline_line = rgb(54, 120, 196),

  .camera_video_border = rgb(58, 116, 196, 0.65f),
  .camera_overlay_bg = rgb(255, 255, 255, 0.75f),
  .camera_overlay_border = rgb(181, 188, 196, 0.90f),

  // Carto-light: near-white canvas, warm motorways, white minor roads — the usual light map
  // conventions, kept muted so the GPS trace and car marker stay the loudest things on it.
  .map_bg = rgb(240, 243, 245),
  .map_grid = rgb(207, 212, 216, 0.76f),
  .map_water_fill = rgb(168, 206, 229, 0.55f),
  .map_water_outline = rgb(122, 170, 203, 0.75f),
  .map_water_line = rgb(112, 163, 198, 0.78f),
  .map_marker = rgb(238, 188, 82),
  .map_marker_outline = rgb(30, 32, 34),
  .map_road_motorway_casing = rgb(224, 167, 90, 0.88f),
  .map_road_motorway_fill = rgb(252, 214, 164, 0.95f),
  .map_road_primary_casing = rgb(196, 172, 118, 0.82f),
  .map_road_primary_fill = rgb(252, 239, 197, 0.92f),
  .map_road_secondary_casing = rgb(172, 178, 184, 0.80f),
  .map_road_secondary_fill = rgb(255, 255, 255, 0.95f),
  .map_road_local_casing = rgb(191, 196, 201, 0.70f),
  .map_road_local_fill = rgb(255, 255, 255, 0.85f),
};

const std::filesystem::path &repo_root() {
  static const std::filesystem::path root = LOGGY_REPO_ROOT;
  return root;
}

void icon_add_font(float size, float density, bool merge = false, const ImFont *base_font = nullptr) {
  const std::filesystem::path ttf = BOOTSTRAP_ICONS_TTF;
  ImGuiIO &io = ImGui::GetIO();
  ImFontConfig config;
  config.MergeMode = merge;
  config.GlyphMinAdvanceX = size;
  config.RasterizerDensity = density;
  if (base_font != nullptr) {
    ImFontBaked *baked = const_cast<ImFont *>(base_font)->GetFontBaked(size);
    const float base_center = baked != nullptr ? (baked->Ascent + baked->Descent) * 0.5f : size * 0.5f;
    config.GlyphOffset.y = std::round(size * 0.5f - base_center);
  }
  static const ImWchar ranges[] = {0xF000, 0xF8FF, 0};
  io.Fonts->AddFontFromFileTTF(ttf.c_str(), size, &config, ranges);
}

}  // namespace

const Theme &theme() {
  return g_current_theme == ThemeKind::Light ? kLightTheme : kDarculaTheme;
}

void load_fonts(float density) {
  ImGuiIO &io = ImGui::GetIO();
  const std::filesystem::path fonts_dir = repo_root() / "openpilot" / "selfdrive" / "assets" / "fonts";
  ImFontConfig font_cfg;
  font_cfg.OversampleH = 2;
  font_cfg.OversampleV = 2;
  font_cfg.RasterizerDensity = density;

  icon_add_font(UI_FONT_SIZE, density);
  const auto add_font_with_icons = [&](const std::filesystem::path &path, float size) -> ImFont * {
    ImFont *font = io.Fonts->AddFontFromFileTTF(path.c_str(), size, &font_cfg);
    if (font != nullptr) {
      icon_add_font(size, density, true, font);
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

ThemeKind theme_from_name(std::string_view name) {
  if (name == "darcula") return ThemeKind::Darcula;
  return ThemeKind::Light;
}

const char *theme_name(ThemeKind kind) {
  switch (kind) {
    case ThemeKind::Darcula: return "darcula";
    case ThemeKind::Light:
    default: return "light";
  }
}

const char *theme_label(ThemeKind kind) {
  switch (kind) {
    case ThemeKind::Darcula: return "Darcula";
    case ThemeKind::Light:
    default: return "Light";
  }
}

void apply_theme(ThemeKind kind) {
  g_current_theme = kind;
  if (kind == ThemeKind::Light) {
    ImGui::StyleColorsLight();
    ImPlot::StyleColorsLight();
  } else {
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();
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

  // Single source of truth: every ImGuiCol_* below is read straight from the Theme struct, never
  // duplicated as a second literal.
  const Theme &t = theme();
  style.Colors[ImGuiCol_WindowBg] = t.window_bg;
  style.Colors[ImGuiCol_ChildBg] = t.child_bg;
  style.Colors[ImGuiCol_Border] = t.border;
  style.Colors[ImGuiCol_TitleBg] = t.title_bg;
  style.Colors[ImGuiCol_TitleBgActive] = t.title_bg_active;
  style.Colors[ImGuiCol_TitleBgCollapsed] = t.title_bg_collapsed;
  style.Colors[ImGuiCol_Text] = t.text;
  style.Colors[ImGuiCol_TextDisabled] = t.text_muted;
  style.Colors[ImGuiCol_Button] = t.button;
  style.Colors[ImGuiCol_ButtonHovered] = t.button_hovered;
  style.Colors[ImGuiCol_ButtonActive] = t.button_active;
  style.Colors[ImGuiCol_FrameBg] = t.frame_bg;
  style.Colors[ImGuiCol_FrameBgHovered] = t.frame_bg_hovered;
  style.Colors[ImGuiCol_FrameBgActive] = t.frame_bg_active;
  style.Colors[ImGuiCol_Header] = t.accent_soft;
  style.Colors[ImGuiCol_HeaderHovered] = t.accent_soft_hovered;
  style.Colors[ImGuiCol_HeaderActive] = t.accent_soft_active;
  style.Colors[ImGuiCol_PopupBg] = t.popup_bg;
  style.Colors[ImGuiCol_MenuBarBg] = t.menu_bar_bg;
  style.Colors[ImGuiCol_Separator] = t.separator;
  style.Colors[ImGuiCol_ScrollbarBg] = t.scrollbar_bg;
  style.Colors[ImGuiCol_ScrollbarGrab] = t.scrollbar_grab;
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = t.scrollbar_grab_hovered;
  style.Colors[ImGuiCol_ScrollbarGrabActive] = t.scrollbar_grab_active;
  style.Colors[ImGuiCol_Tab] = t.tab;
  style.Colors[ImGuiCol_TabHovered] = t.tab_hovered;
  style.Colors[ImGuiCol_TabSelected] = t.tab_selected;
  style.Colors[ImGuiCol_TabSelectedOverline] = t.accent;
  style.Colors[ImGuiCol_TabDimmed] = t.tab_dimmed;
  style.Colors[ImGuiCol_TabDimmedSelected] = t.tab_dimmed_selected;
  style.Colors[ImGuiCol_TabDimmedSelectedOverline] = t.accent;
  style.Colors[ImGuiCol_DockingEmptyBg] = t.docking_empty_bg;
  style.Colors[ImGuiCol_DockingPreview] = t.docking_preview;
  style.Colors[ImGuiCol_CheckMark] = t.check_mark;
  style.Colors[ImGuiCol_SliderGrab] = t.accent;
  style.Colors[ImGuiCol_SliderGrabActive] = t.accent_active;
  style.Colors[ImGuiCol_TableHeaderBg] = t.table_header_bg;
  style.Colors[ImGuiCol_TableBorderStrong] = t.table_border_strong;
  style.Colors[ImGuiCol_TableBorderLight] = t.table_border_light;
  style.Colors[ImGuiCol_TableRowBgAlt] = t.table_row_bg_alt;

  ImPlotStyle &plot_style = ImPlot::GetStyle();
  plot_style.PlotBorderSize = 1.0f;
  plot_style.MinorAlpha = 0.65f;
  plot_style.LegendPadding = ImVec2(4.0f, 3.0f);
  plot_style.LegendInnerPadding = ImVec2(4.0f, 2.0f);
  plot_style.LegendSpacing = ImVec2(5.0f, 1.0f);
  plot_style.PlotPadding = ImVec2(4.0f, 8.0f);
  plot_style.FitPadding = ImVec2(0.02f, 0.05f);
  plot_style.Colors[ImPlotCol_FrameBg] = t.frame_bg;
  plot_style.Colors[ImPlotCol_PlotBg] = t.plot_bg;
  plot_style.Colors[ImPlotCol_PlotBorder] = t.chrome_border;
  plot_style.Colors[ImPlotCol_LegendBg] = t.plot_legend_bg;
  plot_style.Colors[ImPlotCol_LegendBorder] = t.plot_legend_border;
  plot_style.Colors[ImPlotCol_AxisGrid] = t.plot_grid;
  plot_style.Colors[ImPlotCol_Crosshairs] = t.plot_crosshair;
  plot_style.Colors[ImPlotCol_Selection] = t.plot_selection;

  // jotpluggler's plot input map: left-drag pans, wheel zooms, right-drag box-zooms — which
  // leaves a plain right CLICK free to open the pane context menu (plot.cc opens it manually;
  // ImPlot holds the button as active, so BeginPopupContextWindow alone never fires there).
  ImPlot::MapInputDefault();
  ImPlotInputMap &input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Left;
  input_map.PanMod = ImGuiMod_None;
  input_map.Select = ImGuiMouseButton_Right;
  input_map.SelectCancel = ImGuiMouseButton_Left;
  input_map.SelectMod = ImGuiMod_None;
}

ImVec4 clear_color() {
  return theme().window_bg;
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

#if !__has_include("misc/cpp/imgui_stdlib.h")
struct InputTextCallbackData {
  std::string *text = nullptr;
};

int StringResizeCallback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag != ImGuiInputTextFlags_CallbackResize || data->UserData == nullptr) return 0;
  auto *callback = reinterpret_cast<InputTextCallbackData *>(data->UserData);
  if (callback->text == nullptr) return 0;
  callback->text->resize(data->BufTextLen);
  data->Buf = callback->text->data();
  return 0;
}

bool input_text_with_hint(const char *label, const char *hint, std::string *text, ImGuiInputTextFlags flags) {
  if (text == nullptr) return false;
  InputTextCallbackData callback{.text = text};
  const ImGuiInputTextFlags edit_flags = flags | ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputTextWithHint(label, hint, text->data(), text->capacity() + 1,
                                  edit_flags, StringResizeCallback, &callback);
}

bool input_text_multiline(const char *label, std::string *text, ImVec2 size) {
  if (text == nullptr) return false;
  InputTextCallbackData callback{.text = text};
  return ImGui::InputTextMultiline(label, text->data(), text->capacity() + 1, size,
                                   ImGuiInputTextFlags_CallbackResize, StringResizeCallback, &callback);
}
#else
bool input_text_with_hint(const char *label, const char *hint, std::string *text, ImGuiInputTextFlags flags) {
  if (text == nullptr) return false;
  return ImGui::InputTextWithHint(label, hint, text, flags);
}

bool input_text_multiline(const char *label, std::string *text, ImVec2 size) {
  if (text == nullptr) return false;
  return ImGui::InputTextMultiline(label, text, size);
}
#endif


}  // namespace loggy
