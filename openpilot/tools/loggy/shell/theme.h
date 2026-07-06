#pragma once

#include <array>
#include <string>
#include <string_view>

#include "imgui.h"

namespace loggy {

enum class ThemeKind {
  Darcula,
  Light,
};

// Every color loggy draws, as plain data (no framework). Two const instances (kLightTheme,
// kDarculaTheme, in theme.cc) are the single source of truth: apply_theme() copies these into
// ImGuiStyle/ImPlotStyle, and panes read theme() directly for anything a draw-list paints by
// hand. Alpha lives in the token itself (e.g. hud_bg is already translucent) — call sites should
// not multiply in their own opacity, except where a value is genuinely computed per-frame (the
// binary pane's change-heat fade uses binary_heat_accent as a base hue and varies only alpha).
struct Theme {
  // Window / chrome — mirrors ImGuiCol_*; apply_theme() assigns each straight into style.Colors.
  ImVec4 window_bg;
  ImVec4 child_bg;
  ImVec4 border;
  ImVec4 title_bg;
  ImVec4 title_bg_active;
  ImVec4 title_bg_collapsed;
  ImVec4 button;
  ImVec4 button_hovered;
  ImVec4 button_active;
  ImVec4 frame_bg;
  ImVec4 frame_bg_hovered;
  ImVec4 frame_bg_active;
  ImVec4 popup_bg;
  ImVec4 menu_bar_bg;
  ImVec4 separator;
  ImVec4 scrollbar_bg;
  ImVec4 scrollbar_grab;
  ImVec4 scrollbar_grab_hovered;
  ImVec4 scrollbar_grab_active;
  ImVec4 tab;
  ImVec4 tab_hovered;
  ImVec4 tab_selected;
  ImVec4 tab_dimmed;
  ImVec4 tab_dimmed_selected;
  ImVec4 docking_empty_bg;
  ImVec4 docking_preview;
  // A muted stroke shared by chart/grid/video draw-list borders (binary grid, plot border,
  // camera canvas) — deliberately a touch lighter than `border` so charts read as one family.
  ImVec4 chrome_border;

  // Text hierarchy
  ImVec4 text;
  ImVec4 text_muted;

  // Accents: `accent`/`accent_active` is the one saturated hue (sliders, tab overline, docking
  // preview base, the binary pane's change-heat hue); `check_mark` is separate because the two
  // reference themes disagree on whether a checkmark matches text or the accent hue;
  // `accent_soft*` is the much quieter selection-row family (ImGuiCol_Header) — the reference
  // tools barely tint a selected row, so this stays close to child_bg rather than a loud
  // highlight.
  ImVec4 accent;
  ImVec4 accent_active;
  ImVec4 check_mark;
  ImVec4 accent_soft;
  ImVec4 accent_soft_hovered;
  ImVec4 accent_soft_active;

  // Plot pane (the theme-constant ImPlotCol_* below are assigned once in apply_theme(), not
  // pushed per frame)
  ImVec4 plot_bg;
  ImVec4 plot_legend_bg;
  ImVec4 plot_legend_border;
  ImVec4 plot_grid;
  ImVec4 plot_crosshair;
  ImVec4 plot_tracker_line;
  ImVec4 plot_drop_target_fill;
  ImVec4 plot_drop_target_border;
  std::array<ImVec4, 5> plot_series_palette;

  // Binary grid
  ImVec4 binary_idle_cell;
  ImVec4 binary_suppressed_cell;
  ImVec4 binary_heat_accent;  // base hue; heat_color() in binary.cc varies only alpha
  ImVec4 binary_drag_selection;

  // Transport / scrubber
  ImVec4 transport_bg;
  ImVec4 transport_border;
  ImVec4 transport_tracker;

  // Frame-time HUD chip
  ImVec4 hud_bg;
  ImVec4 hud_border;
  ImVec4 hud_text;

  // Browser sparkline
  ImVec4 sparkline_bg;
  ImVec4 sparkline_border;
  ImVec4 sparkline_line;

  // Camera pane (canvas/letterbox fill reuses child_bg directly — see camera.cc)
  ImVec4 camera_video_border;
  ImVec4 camera_overlay_bg;
  ImVec4 camera_overlay_border;

  // Map pane carto palette (canvas border reuses chrome_border). Road pairs follow the usual
  // carto convention: a darker casing stroke under a lighter fill.
  ImVec4 map_bg;
  ImVec4 map_grid;
  ImVec4 map_water_fill;
  ImVec4 map_water_outline;
  ImVec4 map_water_line;
  ImVec4 map_marker;
  ImVec4 map_marker_outline;
  ImVec4 map_road_motorway_casing;
  ImVec4 map_road_motorway_fill;
  ImVec4 map_road_primary_casing;
  ImVec4 map_road_primary_fill;
  ImVec4 map_road_secondary_casing;
  ImVec4 map_road_secondary_fill;
  ImVec4 map_road_local_casing;
  ImVec4 map_road_local_fill;
};

const Theme &theme();

// density: window content scale (e.g. 2.0 on macOS retina) — the atlas rasterizes at device
// pixels while layout metrics stay logical, so text is crisp instead of blurry. 1.0 is a no-op.
void load_fonts(float density = 1.0f);
ThemeKind theme_from_name(std::string_view name);
const char *theme_name(ThemeKind kind);
const char *theme_label(ThemeKind kind);
void apply_theme(ThemeKind kind = ThemeKind::Light);
ImVec4 clear_color();
void push_bold_font();
void pop_bold_font();
void push_mono_font();
void pop_mono_font();
bool input_text_with_hint(const char *label, const char *hint, std::string *text, ImGuiInputTextFlags flags = 0);
bool input_text_multiline(const char *label, std::string *text, ImVec2 size);

}  // namespace loggy
