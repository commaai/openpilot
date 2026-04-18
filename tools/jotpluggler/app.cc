#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/camera.h"
#include "tools/jotpluggler/common.h"
#include "tools/jotpluggler/internal.h"
#include "tools/jotpluggler/map.h"
#include "system/hardware/hw.h"
#include "imgui_impl_glfw.h"

#include "imgui_internal.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <unistd.h>

#include "third_party/json11/json11.hpp"

namespace fs = std::filesystem;

constexpr const char *UNTITLED_PANE_TITLE = "...";
ImFont *g_ui_font = nullptr;
ImFont *g_ui_bold_font = nullptr;
ImFont *g_mono_font = nullptr;

std::string layout_name_from_arg(const std::string &layout_arg) {
  const fs::path raw(layout_arg);
  if (raw.extension() == ".xml" || raw.extension() == ".json") {
    return raw.stem().string();
  }
  if (raw.filename() != raw) {
    return raw.filename().replace_extension("").string();
  }
  fs::path stem_path = raw;
  return stem_path.replace_extension("").string();
}

fs::path layouts_dir() {
  return repo_root() / "tools" / "jotpluggler" / "layouts";
}

std::string sanitize_layout_stem(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  bool last_was_dash = false;
  for (const char raw : name) {
    const unsigned char c = static_cast<unsigned char>(raw);
    if (std::isalnum(c) != 0) {
      out.push_back(static_cast<char>(std::tolower(c)));
      last_was_dash = false;
    } else if (raw == '-' || raw == '_') {
      out.push_back(raw);
      last_was_dash = false;
    } else if (!last_was_dash && !out.empty()) {
      out.push_back('-');
      last_was_dash = true;
    }
  }
  while (!out.empty() && out.back() == '-') {
    out.pop_back();
  }
  return out.empty() ? "untitled" : out;
}

fs::path autosave_dir() {
  return layouts_dir() / ".jotpluggler_autosave";
}

fs::path resolve_layout_path(const std::string &layout_arg) {
  const fs::path direct(layout_arg);
  if (fs::exists(direct)) {
    if (direct.extension() == ".json") return fs::absolute(direct);
    const fs::path sibling_json = direct.parent_path() / (direct.stem().string() + ".json");
    if (direct.extension() == ".xml" && fs::exists(sibling_json)) {
      return fs::absolute(sibling_json);
    }
  }
  const fs::path candidate = layouts_dir() / (layout_name_from_arg(layout_arg) + ".json");
  if (!fs::exists(candidate)) throw std::runtime_error("Unknown layout: " + layout_arg);
  return candidate;
}

fs::path autosave_path_for_layout(const fs::path &layout_path) {
  const std::string stem = layout_path.empty() ? "untitled" : layout_path.stem().string();
  return autosave_dir() / (sanitize_layout_stem(stem) + ".json");
}

std::vector<std::string> available_layout_names() {
  std::vector<std::string> names;
  const fs::path root = layouts_dir();
  if (!fs::exists(root) || !fs::is_directory(root)) {
    return names;
  }
  for (const auto &entry : fs::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    names.push_back(entry.path().stem().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

void refresh_replaced_layout_ui(AppSession *session, UiState *state, bool mark_docks) {
  state->tabs.clear();
  cancel_rename_tab(state);
  sync_ui_state(state, session->layout);
  sync_layout_buffers(state, *session);
  if (mark_docks) {
    mark_all_docks_dirty(state);
  }
}

void start_new_layout(AppSession *session, UiState *state, const std::string &status_text) {
  session->layout = make_empty_layout();
  session->layout_path.clear();
  session->autosave_path.clear();
  state->undo.reset(session->layout);
  state->layout_dirty = false;
  state->status_text = status_text;
  refresh_replaced_layout_ui(session, state, true);
  reset_shared_range(state, *session);
}

bool is_decoded_can_series_path(std::string_view path) {
  const std::string value(path);
  return util::starts_with(value, "/can/") || util::starts_with(value, "/sendcan/");
}

bool apply_route_can_decode_update(AppSession *session, UiState *state);

void rebuild_series_lookup_preserving_formats(AppSession *session,
                                              std::string_view updated_prefix,
                                              bool refresh_updated_formats_only) {
  const std::string prefix(updated_prefix);
  if (!updated_prefix.empty()) {
    for (auto it = session->route_data.series_formats.begin(); it != session->route_data.series_formats.end();) {
      if (util::starts_with(it->first, prefix)) {
        it = session->route_data.series_formats.erase(it);
      } else {
        ++it;
      }
    }
  }
  session->series_by_path.clear();
  session->series_by_path.reserve(session->route_data.series.size());
  for (RouteSeries &series : session->route_data.series) {
    session->series_by_path.emplace(series.path, &series);
    if (refresh_updated_formats_only) {
      if (!updated_prefix.empty() && util::starts_with(series.path, prefix)) {
        const bool enum_like = session->route_data.enum_info.find(series.path) != session->route_data.enum_info.end();
        session->route_data.series_formats[series.path] = compute_series_format(series.values, enum_like);
      }
    } else {
      const bool enum_like = session->route_data.enum_info.find(series.path) != session->route_data.enum_info.end();
      session->route_data.series_formats[series.path] = compute_series_format(series.values, enum_like);
    }
  }
}

bool apply_route_can_decode_update(AppSession *session, UiState *state) {
  const std::string active_dbc_name = !session->dbc_override.empty() ? session->dbc_override : session->route_data.dbc_name;
  if (!active_dbc_name.empty() && !load_dbc_by_name(active_dbc_name).has_value()) {
    state->error_text = "DBC not found: " + active_dbc_name;
    state->open_error_popup = true;
    return false;
  }
  std::unordered_map<std::string, EnumInfo> can_enum_info;
  std::vector<RouteSeries> can_series = decode_can_messages(session->route_data.can_messages, active_dbc_name, &can_enum_info);

  std::vector<RouteSeries> updated_series;
  updated_series.reserve(session->route_data.series.size() + can_series.size());
  for (RouteSeries &series : session->route_data.series) {
    if (!is_decoded_can_series_path(series.path)) {
      updated_series.push_back(std::move(series));
    }
  }
  for (RouteSeries &series : can_series) {
    updated_series.push_back(std::move(series));
  }
  std::sort(updated_series.begin(), updated_series.end(), [](const RouteSeries &a, const RouteSeries &b) {
    return a.path < b.path;
  });

  std::unordered_map<std::string, EnumInfo> updated_enum_info;
  updated_enum_info.reserve(session->route_data.enum_info.size() + can_enum_info.size());
  for (auto &[path, info] : session->route_data.enum_info) {
    if (!is_decoded_can_series_path(path)) {
      updated_enum_info.emplace(path, std::move(info));
    }
  }
  for (auto &[path, info] : can_enum_info) {
    updated_enum_info[path] = std::move(info);
  }

  session->route_data.series = std::move(updated_series);
  session->route_data.enum_info = std::move(updated_enum_info);
  session->route_data.paths.clear();
  session->route_data.paths.reserve(session->route_data.series.size());
  for (const RouteSeries &series : session->route_data.series) {
    session->route_data.paths.push_back(series.path);
  }
  std::sort(session->route_data.paths.begin(), session->route_data.paths.end());
  session->route_data.roots = collect_route_roots_for_paths(session->route_data.paths);

  rebuild_route_index(session);
  rebuild_browser_nodes(session, state);
  refresh_all_custom_curves(session, state);
  sync_camera_feeds(session);
  return true;
}

void apply_dbc_override_change(AppSession *session, UiState *state, const std::string &dbc_override) {
  session->dbc_override = dbc_override;
  if (session->data_mode == SessionDataMode::Stream) {
    start_stream_session(session, state, session->stream_source, session->stream_buffer_seconds, false);
  } else if (!session->route_name.empty()) {
    const bool ok = apply_route_can_decode_update(session, state);
    if (ok) {
      state->status_text = dbc_override.empty() ? "DBC auto-detect enabled" : "DBC set to " + dbc_override;
    } else {
      state->status_text = "Failed to apply DBC";
    }
  } else if (dbc_override.empty()) {
    state->status_text = "DBC auto-detect enabled";
  } else {
    state->status_text = "DBC set to " + dbc_override;
  }
}

void configure_style() {
  ImGui::StyleColorsLight();
  ImPlot::StyleColorsLight();

  ImGuiIO &io = ImGui::GetIO();
  g_ui_font = nullptr;
  g_ui_bold_font = nullptr;
  g_mono_font = nullptr;
  const fs::path fonts_dir = repo_root() / "selfdrive" / "assets" / "fonts";
  ImFontConfig font_cfg;
  font_cfg.OversampleH = 2;
  font_cfg.OversampleV = 2;
  font_cfg.RasterizerDensity = 1.0f;
  icon_add_font(16.0f);
  const auto add_font_with_icons = [&](const fs::path &path, float size) -> ImFont * {
    ImFont *font = io.Fonts->AddFontFromFileTTF(path.c_str(), size, &font_cfg);
    if (font != nullptr) {
      icon_add_font(size, true, font);
    }
    return font;
  };
  if (ImFont *font = add_font_with_icons(fonts_dir / "Inter-Regular.ttf", 16.0f); font != nullptr) {
    g_ui_font = font;
    io.FontDefault = font;
  }
  g_ui_bold_font = add_font_with_icons(fonts_dir / "Inter-SemiBold.ttf", 16.75f);
  if (g_ui_font == nullptr) {
    if (ImFont *font = add_font_with_icons(fonts_dir / "JetBrainsMono-Medium.ttf", 15.75f); font != nullptr) {
      g_mono_font = font;
      io.FontDefault = font;
    }
  }
  if (g_mono_font == nullptr) {
    g_mono_font = add_font_with_icons(fonts_dir / "JetBrainsMono-Medium.ttf", 15.75f);
  }
  if (g_ui_bold_font == nullptr) {
    g_ui_bold_font = g_ui_font;
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
  struct ColorDef { ImGuiCol idx; int r, g, b; };
  constexpr ColorDef COLORS[] = {
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
  for (const auto &c : COLORS) { style.Colors[c.idx] = color_rgb(c.r, c.g, c.b); }
  style.Colors[ImGuiCol_DockingPreview] = color_rgb(69, 115, 184, 0.22f);

  ImPlotStyle &plot_style = ImPlot::GetStyle();
  plot_style.PlotBorderSize = 1.0f;
  plot_style.MinorAlpha = 0.65f;
  plot_style.LegendPadding = ImVec2(6.0f, 5.0f);
  plot_style.LegendInnerPadding = ImVec2(6.0f, 3.0f);
  plot_style.LegendSpacing = ImVec2(7.0f, 2.0f);
  plot_style.PlotPadding = ImVec2(4.0f, 8.0f);
  plot_style.FitPadding = ImVec2(0.02f, 0.4f);

  ImPlot::MapInputDefault();
  ImPlotInputMap &input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Right;
  input_map.PanMod = ImGuiMod_None;
  input_map.Select = ImGuiMouseButton_Left;
  input_map.SelectCancel = ImGuiMouseButton_Right;
  input_map.SelectMod = ImGuiMod_None;
}

void app_push_mono_font() {
  if (g_mono_font != nullptr) {
    ImGui::PushFont(g_mono_font);
  }
}

void app_pop_mono_font() {
  if (g_mono_font != nullptr) {
    ImGui::PopFont();
  }
}

void app_push_bold_font() {
  if (g_ui_bold_font != nullptr) {
    ImGui::PushFont(g_ui_bold_font);
  }
}

void app_pop_bold_font() {
  if (g_ui_bold_font != nullptr) {
    ImGui::PopFont();
  }
}

UiMetrics compute_ui_metrics(const ImVec2 &size, float top_offset, float sidebar_width) {
  UiMetrics ui;
  ui.width = size.x;
  ui.height = size.y;
  ui.top_offset = top_offset;
  ui.sidebar_width = sidebar_width <= 0.0f
    ? 0.0f
    : std::clamp(sidebar_width, SIDEBAR_MIN_WIDTH, std::min(SIDEBAR_MAX_WIDTH, size.x * 0.6f));
  ui.content_x = ui.sidebar_width;
  ui.content_y = top_offset;
  ui.content_w = std::max(1.0f, size.x - ui.content_x);
  ui.content_h = std::max(1.0f, size.y - ui.content_y - STATUS_BAR_HEIGHT);
  ui.status_bar_y = std::max(0.0f, size.y - STATUS_BAR_HEIGHT);
  return ui;
}

void sync_ui_state(UiState *state, const SketchLayout &layout) {
  const bool initializing = state->tabs.empty();
  state->tabs.resize(layout.tabs.size());
  if (layout.tabs.empty()) {
    state->active_tab_index = 0;
    state->requested_tab_index = -1;
    return;
  }
  if (initializing) {
    state->active_tab_index = std::clamp(layout.current_tab_index, 0, static_cast<int>(layout.tabs.size()) - 1);
    state->requested_tab_index = state->active_tab_index;
  }
  state->active_tab_index = std::clamp(state->active_tab_index, 0, static_cast<int>(layout.tabs.size()) - 1);
  for (size_t i = 0; i < layout.tabs.size(); ++i) {
    if (state->tabs[i].runtime_id == 0) {
      state->tabs[i].runtime_id = state->next_tab_runtime_id++;
    }
    const int pane_count = static_cast<int>(layout.tabs[i].panes.size());
    state->tabs[i].map_panes.resize(static_cast<size_t>(std::max(0, pane_count)));
    state->tabs[i].camera_panes.resize(static_cast<size_t>(std::max(0, pane_count)));
    state->tabs[i].active_pane_index = pane_count <= 0
      ? 0
      : std::clamp(state->tabs[i].active_pane_index, 0, pane_count - 1);
  }
}

void resize_tab_pane_state(TabUiState *tab_state, size_t pane_count) {
  if (tab_state == nullptr) return;
  tab_state->map_panes.resize(pane_count);
  tab_state->camera_panes.resize(pane_count);
}

void sync_route_buffers(UiState *state, const AppSession &session) {
  state->route_buffer = session.route_name;
  state->data_dir_buffer = session.data_dir;
}

void sync_stream_buffers(UiState *state, const AppSession &session) {
  state->stream_address_buffer = session.stream_source.address;
  state->stream_source_kind = session.stream_source.kind;
  state->stream_buffer_seconds = session.stream_buffer_seconds;
}

fs::path default_layout_save_path(const AppSession &session) {
  return session.layout_path.empty() ? layouts_dir() / "new-layout.json" : session.layout_path;
}

void sync_layout_buffers(UiState *state, const AppSession &session) {
  state->load_layout_buffer = session.layout_path.empty() ? std::string() : session.layout_path.string();
  state->save_layout_buffer = default_layout_save_path(session).string();
}

const WorkspaceTab *app_active_tab(const SketchLayout &layout, const UiState &state) {
  if (layout.tabs.empty()) return nullptr;
  const int index = std::clamp(state.active_tab_index, 0, static_cast<int>(layout.tabs.size()) - 1);
  return &layout.tabs[static_cast<size_t>(index)];
}

WorkspaceTab *app_active_tab(SketchLayout *layout, const UiState &state) {
  if (layout->tabs.empty()) return nullptr;
  const int index = std::clamp(state.active_tab_index, 0, static_cast<int>(layout->tabs.size()) - 1);
  return &layout->tabs[static_cast<size_t>(index)];
}

TabUiState *app_active_tab_state(UiState *state) {
  if (state->tabs.empty()) return nullptr;
  const int index = std::clamp(state->active_tab_index, 0, static_cast<int>(state->tabs.size()) - 1);
  return &state->tabs[static_cast<size_t>(index)];
}

std::string pane_window_name(int tab_runtime_id, int pane_index, const Pane &pane) {
  const char *title = pane.title.empty() ? UNTITLED_PANE_TITLE : pane.title.c_str();
  return util::string_format("%s###tab%d_pane%d", title, tab_runtime_id, pane_index);
}

std::string tab_item_label(const WorkspaceTab &tab, int tab_runtime_id) {
  return util::string_format("%s##workspace_tab_%d", tab.tab_name.c_str(), tab_runtime_id);
}

void request_tab_selection(UiState *state, int tab_index) {
  state->active_tab_index = tab_index;
  state->requested_tab_index = tab_index;
}

void begin_rename_tab(const SketchLayout &layout, UiState *state, int tab_index) {
  if (tab_index < 0 || tab_index >= static_cast<int>(layout.tabs.size())) {
    return;
  }
  state->rename_tab_buffer = layout.tabs[static_cast<size_t>(tab_index)].tab_name;
  state->rename_tab_index = tab_index;
  state->focus_rename_tab_input = true;
  request_tab_selection(state, tab_index);
}

void cancel_rename_tab(UiState *state) {
  state->rename_tab_index = -1;
  state->focus_rename_tab_input = false;
}

ImGuiID dockspace_id_for_tab(int tab_runtime_id) {
  return ImHashStr(util::string_format("jotpluggler_dockspace_%d", tab_runtime_id).c_str());
}

bool curve_has_local_samples(const Curve &curve) {
  return curve.xs.size() > 1 && curve.xs.size() == curve.ys.size();
}

void mark_all_docks_dirty(UiState *state) {
  for (TabUiState &tab_state : state->tabs) {
    tab_state.dock_needs_build = true;
  }
}

void mark_tab_dock_dirty(UiState *state, int tab_index) {
  if (tab_index >= 0 && tab_index < static_cast<int>(state->tabs.size())) {
    state->tabs[static_cast<size_t>(tab_index)].dock_needs_build = true;
  }
}

void normalize_split_node(WorkspaceNode *node) {
  if (node->is_pane) {
    return;
  }
  for (WorkspaceNode &child : node->children) {
    normalize_split_node(&child);
  }
  if (node->children.empty()) {
    return;
  }
  if (node->children.size() == 1) {
    *node = node->children.front();
    return;
  }
  if (node->sizes.size() != node->children.size()) {
    node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
    return;
  }
  float total = 0.0f;
  for (float &size : node->sizes) {
    size = std::max(size, 0.0f);
    total += size;
  }
  if (total <= 0.0f) {
    node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
    return;
  }
  for (float &size : node->sizes) {
    size /= total;
  }
}

void decrement_pane_indices(WorkspaceNode *node, int removed_index) {
  if (node->is_pane) {
    if (node->pane_index > removed_index) {
      node->pane_index -= 1;
    }
    return;
  }
  for (WorkspaceNode &child : node->children) {
    decrement_pane_indices(&child, removed_index);
  }
}

bool remove_pane_node(WorkspaceNode *node, int pane_index) {
  if (node->is_pane) return node->pane_index == pane_index;

  for (size_t i = 0; i < node->children.size();) {
    if (remove_pane_node(&node->children[i], pane_index)) {
      node->children.erase(node->children.begin() + static_cast<std::ptrdiff_t>(i));
      if (i < node->sizes.size()) {
        node->sizes.erase(node->sizes.begin() + static_cast<std::ptrdiff_t>(i));
      }
    } else {
      ++i;
    }
  }

  normalize_split_node(node);
  return !node->is_pane && node->children.empty();
}

bool split_pane_node(WorkspaceNode *node, int target_pane_index, SplitOrientation orientation,
                     bool new_before, int new_pane_index) {
  if (node->is_pane) {
    if (node->pane_index != target_pane_index) return false;
    WorkspaceNode existing_pane;
    existing_pane.is_pane = true;
    existing_pane.pane_index = target_pane_index;

    WorkspaceNode new_pane;
    new_pane.is_pane = true;
    new_pane.pane_index = new_pane_index;

    node->is_pane = false;
    node->pane_index = -1;
    node->orientation = orientation;
    node->sizes = {0.5f, 0.5f};
    node->children.clear();
    if (new_before) {
      node->children.push_back(std::move(new_pane));
      node->children.push_back(std::move(existing_pane));
    } else {
      node->children.push_back(std::move(existing_pane));
      node->children.push_back(std::move(new_pane));
    }
    return true;
  }

  if (node->orientation == orientation) {
    for (size_t i = 0; i < node->children.size(); ++i) {
      WorkspaceNode &child = node->children[i];
      if (!child.is_pane || child.pane_index != target_pane_index) {
        continue;
      }

      WorkspaceNode new_pane;
      new_pane.is_pane = true;
      new_pane.pane_index = new_pane_index;

      const auto insert_it = node->children.begin() + static_cast<std::ptrdiff_t>(new_before ? i : i + 1);
      node->children.insert(insert_it, std::move(new_pane));
      node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
      return true;
    }
  }

  for (WorkspaceNode &child : node->children) {
    if (split_pane_node(&child, target_pane_index, orientation, new_before, new_pane_index)) return true;
  }
  return false;
}

Pane make_empty_pane(const std::string &title = UNTITLED_PANE_TITLE) {
  Pane pane;
  pane.title = title;
  return pane;
}

WorkspaceTab make_empty_tab(const std::string &tab_name) {
  WorkspaceTab tab;
  tab.tab_name = tab_name;
  tab.panes.push_back(make_empty_pane());
  tab.root.is_pane = true;
  tab.root.pane_index = 0;
  return tab;
}

SketchLayout make_empty_layout() {
  SketchLayout layout;
  layout.tabs.push_back(make_empty_tab("tab1"));
  layout.current_tab_index = 0;
  layout.roots.push_back("layout");
  return layout;
}

bool tab_name_exists(const SketchLayout &layout, const std::string &name) {
  return std::any_of(layout.tabs.begin(), layout.tabs.end(), [&](const WorkspaceTab &tab) {
    return tab.tab_name == name;
  });
}

std::string next_tab_name(const SketchLayout &layout, const std::string &base_name) {
  if (base_name == "tab" || base_name == "tab1") {
    int max_suffix = 0;
    for (const WorkspaceTab &tab : layout.tabs) {
      if (tab.tab_name.size() > 3 && util::starts_with(tab.tab_name, "tab")) {
        const std::string suffix = tab.tab_name.substr(3);
        if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
          max_suffix = std::max(max_suffix, std::stoi(suffix));
        }
      }
    }
    return "tab" + std::to_string(std::max(1, max_suffix + 1));
  }
  std::string base = base_name.empty() ? "tab" : base_name;
  if (!tab_name_exists(layout, base)) return base;
  for (int i = 2; i < 1000; ++i) {
    const std::string candidate = base + " " + std::to_string(i);
    if (!tab_name_exists(layout, candidate)) return candidate;
  }
  return base + " copy";
}

void clear_layout_autosave(const AppSession &session) {
  if (!session.autosave_path.empty() && fs::exists(session.autosave_path)) {
    fs::remove(session.autosave_path);
  }
}

bool autosave_layout(AppSession *session, UiState *state) {
  try {
    if (session->autosave_path.empty()) {
      session->autosave_path = autosave_path_for_layout(session->layout_path);
    }
    session->layout.current_tab_index = state->active_tab_index;
    save_layout_json(session->layout, session->autosave_path);
    state->layout_dirty = true;
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to save layout draft";
    return false;
  }
}

bool mark_layout_dirty(AppSession *session, UiState *state) {
  return autosave_layout(session, state);
}

bool active_tab_has_map_pane(const SketchLayout &layout) {
  if (layout.tabs.empty()) {
    return false;
  }
  const int tab_index = std::clamp(layout.current_tab_index, 0, static_cast<int>(layout.tabs.size()) - 1);
  const WorkspaceTab &tab = layout.tabs[static_cast<size_t>(tab_index)];
  return std::any_of(tab.panes.begin(), tab.panes.end(), [](const Pane &pane) {
    return pane_kind_is_special(pane.kind);
  });
}

void draw_browser_special_item(const char *item_id, const char *label) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImVec2 row_size(std::max(1.0f, ImGui::GetContentRegionAvail().x), ImGui::GetFrameHeight());
  ImGui::PushID(item_id);
  ImGui::InvisibleButton("##special_data_row", row_size);
  const bool hovered = ImGui::IsItemHovered();
  const bool held = ImGui::IsItemActive();
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  if (hovered) {
    const ImU32 bg = ImGui::GetColorU32(held ? ImGuiCol_HeaderActive : ImGuiCol_HeaderHovered);
    draw_list->AddRectFilled(rect.Min, rect.Max, bg, 0.0f);
  }
  ImGui::RenderTextEllipsis(draw_list,
                            ImVec2(rect.Min.x + style.FramePadding.x, rect.Min.y + style.FramePadding.y),
                            ImVec2(rect.Max.x - style.FramePadding.x, rect.Max.y),
                            rect.Max.x - style.FramePadding.x,
                            label,
                            nullptr,
                            nullptr);
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload("JOTP_SPECIAL_ITEM", item_id, std::strlen(item_id) + 1);
    ImGui::TextUnformatted(label);
    ImGui::EndDragDropSource();
  }
  ImGui::PopID();
}

std::array<uint8_t, 3> app_next_curve_color(const Pane &pane) {
  static constexpr std::array<std::array<uint8_t, 3>, 10> PALETTE = {{
    {35, 107, 180},
    {220, 82, 52},
    {67, 160, 71},
    {243, 156, 18},
    {123, 97, 255},
    {0, 150, 136},
    {214, 48, 49},
    {52, 73, 94},
    {197, 90, 17},
    {96, 125, 139},
  }};
  return PALETTE[pane.curves.size() % PALETTE.size()];
}

void draw_sidebar(AppSession *session, const UiMetrics &ui, UiState *state, bool show_camera_feed) {
  ImGui::SetNextWindowPos(ImVec2(0.0f, ui.top_offset));
  ImGui::SetNextWindowSize(ImVec2(ui.sidebar_width, std::max(1.0f, ui.height - ui.top_offset)));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(238, 240, 244));
  ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(190, 197, 205));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin("##sidebar", nullptr, flags)) {
    const RouteLoadSnapshot load = session->route_loader ? session->route_loader->snapshot() : RouteLoadSnapshot{};
    const bool show_load_progress = session->route_loader && (load.active || load.total_segments > 0);
    const bool streaming = session->data_mode == SessionDataMode::Stream;
    CameraFeedView *sidebar_camera = session->pane_camera_feeds[static_cast<size_t>(sidebar_preview_camera_view(*session))].get();
    if (show_camera_feed && sidebar_camera != nullptr) {
      sidebar_camera->draw(ImGui::GetContentRegionAvail().x, load.active);
    } else if (streaming) {
      ImGui::SeparatorText("Camera");
      ImGui::TextDisabled("Camera not available during live stream.");
      ImGui::Spacing();
    }

    ImGui::SeparatorText(streaming ? "Stream" : "Route");
    if (streaming) {
      const StreamPollSnapshot stream = session->stream_poller ? session->stream_poller->snapshot() : StreamPollSnapshot{};
      const bool paused = stream.paused || session->stream_paused;
      const bool live = stream.connected && !paused;
      const ImVec4 status_color = live ? color_rgb(38, 135, 67) : (paused ? color_rgb(168, 119, 34) : color_rgb(155, 63, 63));
      ImGui::TextColored(status_color, "%s %s", live ? "●" : "○", stream.source_label.c_str());
      ImGui::TextDisabled("%s%s", stream_source_kind_label(stream.source_kind), paused ? "  paused" : "");
      const double span = session->route_data.has_time_range ? (session->route_data.x_max - session->route_data.x_min) : 0.0;
      const float fill = stream.buffer_seconds <= 0.0
        ? 0.0f
        : std::clamp(static_cast<float>(span / stream.buffer_seconds), 0.0f, 1.0f);
      ImGui::ProgressBar(fill, ImVec2(-FLT_MIN, 0.0f), nullptr);
      ImGui::TextDisabled("%.0fs buffer | %zu series", session->stream_buffer_seconds, session->route_data.series.size());
      const char *button_label = paused ? "Resume" : "Pause";
      if (ImGui::Button(button_label, ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), 0.0f))) {
        if (paused) {
          start_stream_session(session, state, session->stream_source, session->stream_buffer_seconds, true);
        } else {
          stop_stream_session(session, state);
          state->status_text = "Paused stream " + stream_source_target_label(session->stream_source);
        }
      }
    } else if (session->route_name.empty()) {
      ImGui::TextDisabled("No route loaded");
    }
    if (!session->route_data.car_fingerprint.empty()) {
      ImGui::TextWrapped("Car: %s", session->route_data.car_fingerprint.c_str());
    }
    const std::vector<std::string> dbc_names = available_dbc_names();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##dbc_combo", dbc_combo_label(*session).c_str())) {
      const bool auto_selected = session->dbc_override.empty();
      if (ImGui::Selectable("Auto", auto_selected)) {
        apply_dbc_override_change(session, state, {});
      }
      if (auto_selected) {
        ImGui::SetItemDefaultFocus();
      }
      ImGui::Separator();
      for (const std::string &dbc_name : dbc_names) {
        const bool selected = session->dbc_override == dbc_name;
        if (ImGui::Selectable(dbc_name.c_str(), selected) && !selected) {
          apply_dbc_override_change(session, state, dbc_name);
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SeparatorText("Layout");
    ImGui::SetNextItemWidth(-FLT_MIN);
    const std::string layout_combo_label = [&] {
      const std::string base = session->layout_path.empty() ? std::string("untitled") : session->layout_path.stem().string();
      return state->layout_dirty ? base + " *" : base;
    }();
    if (ImGui::BeginCombo("##layout_combo", layout_combo_label.c_str())) {
      if (ImGui::Selectable("New Layout")) {
        start_new_layout(session, state);
      }
      ImGui::Separator();
      const std::vector<std::string> layouts = available_layout_names();
      const std::string current_layout = session->layout_path.empty() ? std::string("untitled") : session->layout_path.stem().string();
      for (const std::string &layout_name : layouts) {
        const bool selected = layout_name == current_layout;
        if (ImGui::Selectable(layout_name.c_str(), selected) && !selected) {
          reload_layout(session, state, layout_name);
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    const float layout_button_gap = ImGui::GetStyle().ItemSpacing.x;
    const float layout_row_width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    const float layout_button_width = std::max(1.0f, (layout_row_width - 2.0f * layout_button_gap) / 3.0f);
    if (ImGui::Button("New", ImVec2(layout_button_width, 0.0f))) {
      start_new_layout(session, state);
    }
    ImGui::SameLine(0.0f, layout_button_gap);
    if (ImGui::Button("Save", ImVec2(layout_button_width, 0.0f))) {
      state->request_save_layout = true;
    }
    ImGui::SameLine(0.0f, layout_button_gap);
    ImGui::BeginDisabled(!state->layout_dirty);
    if (ImGui::Button("Reset", ImVec2(layout_button_width, 0.0f))) {
      state->request_reset_layout = true;
    }
    ImGui::EndDisabled();
    ImGui::Spacing();

    ImGui::SeparatorText("Data Sources");
    ImGui::SetNextItemWidth(-FLT_MIN);
    input_text_with_hint_string("##browser_filter", "Search...", &state->browser_filter);
    const float footer_height = ImGui::GetFrameHeightWithSpacing()
                              + ImGui::GetTextLineHeightWithSpacing()
                              + 16.0f
                              + (show_load_progress ? (ImGui::GetFrameHeightWithSpacing() + 12.0f) : 0.0f);
    const float browser_height = std::max(1.0f, ImGui::GetContentRegionAvail().y - footer_height);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 3.0f));
    if (ImGui::BeginChild("##timeseries_browser", ImVec2(0.0f, browser_height), true)) {
      const std::string filter = lowercase_copy(state->browser_filter);
      std::vector<std::string> visible_paths;
      for (const BrowserNode &node : session->browser_nodes) {
        collect_visible_leaf_paths(node, filter, &visible_paths);
      }
      for (const SpecialItemSpec &spec : kSpecialItemSpecs) {
        draw_browser_special_item(spec.id, spec.label);
      }
      ImGui::Dummy(ImVec2(0.0f, 2.0f));
      ImGui::Separator();
      ImGui::Dummy(ImVec2(0.0f, 2.0f));
      for (const BrowserNode &node : session->browser_nodes) {
        draw_browser_node(session, node, state, filter, visible_paths);
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);

    ImGui::SeparatorText("Custom Series");
    if (ImGui::Button("Create...", ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), 0.0f))) {
      open_custom_series_editor(state, state->selected_browser_path);
    }
    if (show_load_progress) {
      const float total = static_cast<float>(std::max<size_t>(1, load.total_segments));
      const bool finalizing = load.active
                           && load.total_segments > 0
                           && load.segments_downloaded >= load.total_segments
                           && load.segments_parsed >= load.total_segments;
      const float progress = load.total_segments == 0
        ? 0.0f
        : (finalizing
            ? 0.99f
            : std::clamp(static_cast<float>(load.segments_downloaded + load.segments_parsed) / (2.0f * total), 0.0f, 0.99f));
      ImGui::Dummy(ImVec2(0.0f, 8.0f));
      ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0.0f), finalizing ? "Finalizing..." : nullptr);
    }
  }
  ImGui::End();
  ImGui::PopStyleColor(2);
}

std::string app_curve_display_name(const Curve &curve) {
  if (!curve.label.empty()) return curve.label;
  if (!curve.name.empty()) return curve.name;
  return "curve";
}

Curve make_curve_for_path(const Pane &pane, const std::string &path) {
  Curve curve;
  curve.name = path;
  curve.label = path;
  curve.color = app_next_curve_color(pane);
  return curve;
}

bool add_curve_to_pane(WorkspaceTab *tab, int pane_index, Curve curve) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return false;
  }
  Pane &pane = tab->panes[static_cast<size_t>(pane_index)];
  if (pane.kind != PaneKind::Plot) {
    pane.kind = PaneKind::Plot;
    if (is_default_special_title(pane.title)) {
      pane.title = UNTITLED_PANE_TITLE;
    }
  }
  for (Curve &existing : pane.curves) {
    const bool same_named_curve = !curve.name.empty() && existing.name == curve.name;
    const bool same_unnamed_curve = curve.name.empty() && existing.name.empty() && existing.label == curve.label;
    if (same_named_curve || same_unnamed_curve) {
      existing.visible = true;
      return false;
    }
  }
  pane.curves.push_back(std::move(curve));
  return true;
}

bool add_path_curve_to_pane(AppSession *session, UiState *state, int pane_index, const std::string &path) {
  if (app_find_route_series(*session, path) == nullptr) {
    state->status_text = "Path not found in route";
    return false;
  }
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  if (tab == nullptr || pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    state->status_text = "No active pane";
    return false;
  }
  const SketchLayout before_layout = session->layout;
  const bool inserted = add_curve_to_pane(tab, pane_index, make_curve_for_path(tab->panes[static_cast<size_t>(pane_index)], path));
  bool autosave_ok = true;
  if (inserted) {
    state->undo.push(before_layout);
    autosave_ok = mark_layout_dirty(session, state);
  }
  if (autosave_ok) {
    state->status_text = inserted ? "Added " + path : "Curve already present";
  }
  return true;
}

int add_path_curves_to_pane(AppSession *session, UiState *state, int pane_index, const std::vector<std::string> &paths) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  if (tab == nullptr || pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    state->status_text = "No active pane";
    return 0;
  }

  int inserted_count = 0;
  int duplicate_count = 0;
  const SketchLayout before_layout = session->layout;
  for (const std::string &path : paths) {
    if (app_find_route_series(*session, path) == nullptr) continue;
    if (add_curve_to_pane(tab, pane_index, make_curve_for_path(tab->panes[static_cast<size_t>(pane_index)], path))) {
      ++inserted_count;
    } else {
      ++duplicate_count;
    }
  }

  if (inserted_count > 0) {
    state->undo.push(before_layout);
    if (mark_layout_dirty(session, state)) {
      state->status_text = inserted_count == 1
        ? "Added " + paths.front()
        : "Added " + std::to_string(inserted_count) + " curves";
    }
    return inserted_count;
  }

  if (duplicate_count > 0) {
    state->status_text = duplicate_count == 1 ? "Curve already present" : "Curves already present";
  } else {
    state->status_text = "No matching series found";
  }
  return 0;
}

bool app_add_curve_to_active_pane(AppSession *session, UiState *state, const std::string &path) {
  const TabUiState *tab_state = app_active_tab_state(state);
  if (tab_state == nullptr) {
    state->status_text = "No active pane";
    return false;
  }
  return add_path_curve_to_pane(session, state, tab_state->active_pane_index, path);
}

bool apply_special_item_to_pane(WorkspaceTab *tab, TabUiState *tab_state, int pane_index, std::string_view item_id) {
  if (tab == nullptr || tab_state == nullptr) return false;
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) return false;
  const SpecialItemSpec *spec = special_item_spec(item_id);
  if (spec == nullptr) return false;
  Pane &pane = tab->panes[static_cast<size_t>(pane_index)];
  if (!((pane.kind == PaneKind::Plot && pane.curves.empty()) || pane_kind_is_special(pane.kind))) {
    return false;
  }
  if (pane.kind == spec->kind && (spec->kind != PaneKind::Camera || pane.camera_view == spec->camera_view)) {
    tab_state->active_pane_index = pane_index;
    return false;
  }
  const PaneKind previous_kind = pane.kind;
  pane.kind = spec->kind;
  pane.camera_view = spec->camera_view;
  if (spec->kind == PaneKind::Map) {
    if (pane.title == UNTITLED_PANE_TITLE || previous_kind != PaneKind::Plot) {
      pane.title = spec->label;
    }
  } else {
    pane.title = spec->label;
    resize_tab_pane_state(tab_state, tab->panes.size());
    tab_state->camera_panes[static_cast<size_t>(pane_index)].fit_to_pane = true;
  }
  tab_state->active_pane_index = pane_index;
  return true;
}

bool split_pane(WorkspaceTab *tab, int pane_index, PaneDropZone zone, std::optional<Curve> curve = std::nullopt) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return false;
  }
  if (zone == PaneDropZone::Center) return false;

  const int new_pane_index = static_cast<int>(tab->panes.size());
  Pane new_pane = make_empty_pane();
  if (curve.has_value()) {
    new_pane.curves.push_back(*curve);
  }
  tab->panes.push_back(std::move(new_pane));

  const bool vertical = zone == PaneDropZone::Top || zone == PaneDropZone::Bottom;
  const bool new_before = zone == PaneDropZone::Left || zone == PaneDropZone::Top;
  return split_pane_node(&tab->root, pane_index,
    vertical ? SplitOrientation::Vertical : SplitOrientation::Horizontal,
    new_before, new_pane_index);
}

bool close_pane(WorkspaceTab *tab, int pane_index) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return false;
  }
  if (tab->panes.size() <= 1) {
    tab->panes[static_cast<size_t>(pane_index)] = make_empty_pane();
    return true;
  }
  if (remove_pane_node(&tab->root, pane_index)) return false;
  tab->panes.erase(tab->panes.begin() + static_cast<std::ptrdiff_t>(pane_index));
  decrement_pane_indices(&tab->root, pane_index);
  normalize_split_node(&tab->root);
  return true;
}

void clear_pane(WorkspaceTab *tab, int pane_index) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return;
  }
  Pane &pane = tab->panes[static_cast<size_t>(pane_index)];
  pane.curves.clear();
  pane.title = UNTITLED_PANE_TITLE;
}

void create_runtime_tab(SketchLayout *layout, UiState *state) {
  const std::string tab_name = next_tab_name(*layout, "tab1");
  layout->tabs.push_back(make_empty_tab(tab_name));
  state->tabs.push_back(TabUiState{.dock_needs_build = true, .active_pane_index = 0, .runtime_id = state->next_tab_runtime_id++});
  request_tab_selection(state, static_cast<int>(layout->tabs.size()) - 1);
  state->status_text = "Created " + tab_name;
}

void duplicate_runtime_tab(SketchLayout *layout, UiState *state) {
  if (layout->tabs.empty()) {
    return;
  }
  const int source_index = std::clamp(state->active_tab_index, 0, static_cast<int>(layout->tabs.size()) - 1);
  WorkspaceTab copy = layout->tabs[static_cast<size_t>(source_index)];
  copy.tab_name = next_tab_name(*layout, copy.tab_name + " copy");
  layout->tabs.push_back(std::move(copy));
  const int active_pane_index = source_index < static_cast<int>(state->tabs.size()) ? state->tabs[static_cast<size_t>(source_index)].active_pane_index : 0;
  state->tabs.push_back(TabUiState{.dock_needs_build = true, .active_pane_index = active_pane_index, .runtime_id = state->next_tab_runtime_id++});
  request_tab_selection(state, static_cast<int>(layout->tabs.size()) - 1);
  state->status_text = "Duplicated tab";
}

void close_runtime_tab(SketchLayout *layout, UiState *state) {
  if (layout->tabs.empty()) {
    return;
  }
  const int tab_index = std::clamp(state->active_tab_index, 0, static_cast<int>(layout->tabs.size()) - 1);
  if (layout->tabs.size() == 1) {
    layout->tabs[0] = make_empty_tab(layout->tabs[0].tab_name.empty() ? "tab1" : layout->tabs[0].tab_name);
    if (state->tabs.empty()) {
      state->tabs.push_back(TabUiState{.dock_needs_build = true, .active_pane_index = 0});
    } else {
      state->tabs.resize(1);
      state->tabs[0] = TabUiState{
        .dock_needs_build = true,
        .active_pane_index = 0,
        .runtime_id = state->tabs[0].runtime_id == 0 ? state->next_tab_runtime_id++ : state->tabs[0].runtime_id,
      };
    }
    state->active_tab_index = 0;
    state->requested_tab_index = 0;
    layout->current_tab_index = 0;
    cancel_rename_tab(state);
    state->status_text = "Closed tab";
    return;
  }
  layout->tabs.erase(layout->tabs.begin() + static_cast<std::ptrdiff_t>(tab_index));
  if (tab_index < static_cast<int>(state->tabs.size())) {
    state->tabs.erase(state->tabs.begin() + static_cast<std::ptrdiff_t>(tab_index));
  }
  if (state->active_tab_index >= static_cast<int>(layout->tabs.size())) {
    state->active_tab_index = static_cast<int>(layout->tabs.size()) - 1;
  }
  sync_ui_state(state, *layout);
  state->requested_tab_index = state->active_tab_index;
  state->status_text = "Closed tab";
}

void rename_runtime_tab(SketchLayout *layout, UiState *state) {
  if (state->rename_tab_index < 0 || state->rename_tab_index >= static_cast<int>(layout->tabs.size())) {
    return;
  }
  layout->tabs[static_cast<size_t>(state->rename_tab_index)].tab_name = state->rename_tab_buffer;
  state->status_text = "Renamed tab";
  layout->current_tab_index = state->rename_tab_index;
  cancel_rename_tab(state);
}

void draw_inline_tab_editor(AppSession *session, UiState *state, const ImRect &tab_rect) {
  const int rename_tab_index = state->rename_tab_index;
  if (rename_tab_index < 0 || rename_tab_index >= static_cast<int>(session->layout.tabs.size())) {
    return;
  }

  const float width = std::max(48.0f, tab_rect.Max.x - tab_rect.Min.x - 10.0f);
  const ImVec2 pos = ImVec2(tab_rect.Min.x + 5.0f, tab_rect.Min.y + 2.0f);
  ImGui::SetCursorScreenPos(pos);
  ImGui::PushItemWidth(width);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
  if (state->focus_rename_tab_input) {
    ImGui::SetKeyboardFocusHere();
    state->focus_rename_tab_input = false;
  }
  const bool submitted = input_text_string("##rename_tab_inline",
                                           &state->rename_tab_buffer,
                                           ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);
  const bool active = ImGui::IsItemActive();
  const bool escape = active && ImGui::IsKeyPressed(ImGuiKey_Escape);
  const bool deactivated = ImGui::IsItemDeactivated();
  ImGui::PopStyleVar();
  ImGui::PopItemWidth();

  if (escape) {
    cancel_rename_tab(state);
  } else if (submitted || deactivated) {
    const SketchLayout before_layout = session->layout;
    rename_runtime_tab(&session->layout, state);
    state->undo.push(before_layout);
    mark_layout_dirty(session, state);
  }
}


std::optional<PaneDropAction> draw_pane_drop_target(int tab_index, int pane_index, const Pane &target_pane) {
  if (ImGui::GetDragDropPayload() == nullptr) return std::nullopt;

  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
  ImRect content_rect(ImVec2(window_pos.x + content_min.x, window_pos.y + content_min.y),
                      ImVec2(window_pos.x + content_max.x, window_pos.y + content_max.y));
  content_rect.Expand(ImVec2(-6.0f, -6.0f));
  if (content_rect.GetWidth() < 60.0f || content_rect.GetHeight() < 60.0f) {
    return std::nullopt;
  }

  const float edge_w = std::min(90.0f, content_rect.GetWidth() * 0.24f);
  const float edge_h = std::min(72.0f, content_rect.GetHeight() * 0.24f);
  struct ZoneRect {
    PaneDropZone zone;
    ImRect rect;
  };
  const std::array<ZoneRect, 5> zones = {{
    {PaneDropZone::Left, ImRect(content_rect.Min, ImVec2(content_rect.Min.x + edge_w, content_rect.Max.y))},
    {PaneDropZone::Right, ImRect(ImVec2(content_rect.Max.x - edge_w, content_rect.Min.y), content_rect.Max)},
    {PaneDropZone::Top, ImRect(content_rect.Min, ImVec2(content_rect.Max.x, content_rect.Min.y + edge_h))},
    {PaneDropZone::Bottom, ImRect(ImVec2(content_rect.Min.x, content_rect.Max.y - edge_h), content_rect.Max)},
    {PaneDropZone::Center, ImRect(ImVec2(content_rect.Min.x + edge_w, content_rect.Min.y + edge_h),
                                  ImVec2(content_rect.Max.x - edge_w, content_rect.Max.y - edge_h))},
  }};

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  for (const ZoneRect &zone : zones) {
    if (zone.rect.GetWidth() <= 0.0f || zone.rect.GetHeight() <= 0.0f) {
      continue;
    }

    ImGui::PushID(static_cast<int>(zone.zone) * 1000 + pane_index + tab_index * 100);
    ImGui::SetCursorScreenPos(zone.rect.Min);
    ImGui::InvisibleButton("##drop_zone", zone.rect.GetSize());
    if (ImGui::BeginDragDropTarget()) {
      auto try_accept = [&](const char *type) -> const ImGuiPayload * {
        const ImGuiPayload *p = ImGui::AcceptDragDropPayload(type, ImGuiDragDropFlags_AcceptBeforeDelivery);
        if (p && p->Preview) {
          draw_list->AddRectFilled(zone.rect.Min, zone.rect.Max, IM_COL32(70, 130, 220, 55));
          draw_list->AddRect(zone.rect.Min, zone.rect.Max, IM_COL32(45, 95, 175, 220), 0.0f, 0, 2.0f);
        }
        return p;
      };
      auto deliver = [&](PaneDropAction action) -> std::optional<PaneDropAction> {
        action.zone = zone.zone;
        action.target_pane_index = pane_index;
        ImGui::EndDragDropTarget();
        ImGui::PopID();
        return action;
      };
      if (const ImGuiPayload *p = try_accept("JOTP_BROWSER_PATHS"); p && p->Delivery) {
        if (zone.zone != PaneDropZone::Center || target_pane.kind == PaneKind::Plot) {
          PaneDropAction action;
          action.from_browser = true;
          action.browser_paths = decode_browser_drag_payload(static_cast<const char *>(p->Data));
          return deliver(std::move(action));
        }
      }
      if (zone.zone != PaneDropZone::Center || (target_pane.kind == PaneKind::Plot && target_pane.curves.empty()) || pane_kind_is_special(target_pane.kind)) {
        if (const ImGuiPayload *p = try_accept("JOTP_SPECIAL_ITEM"); p && p->Delivery) {
          PaneDropAction action;
          action.special_item_id = static_cast<const char *>(p->Data);
          return deliver(std::move(action));
        }
      }
      if (const ImGuiPayload *p = try_accept("JOTP_PANE_CURVE"); p && p->Delivery) {
        if (zone.zone != PaneDropZone::Center || target_pane.kind == PaneKind::Plot) {
          PaneDropAction action;
          action.curve_ref = *static_cast<const PaneCurveDragPayload *>(p->Data);
          return deliver(std::move(action));
        }
      }
      ImGui::EndDragDropTarget();
    }
    ImGui::PopID();
  }
  return std::nullopt;
}

bool commit_tab_layout_change(AppSession *session,
                              UiState *state,
                              WorkspaceTab *tab,
                              TabUiState *tab_state,
                              const SketchLayout &before_layout,
                              std::string_view status_text,
                              bool dock_changed) {
  if (dock_changed) {
    mark_tab_dock_dirty(state, state->active_tab_index);
  }
  resize_tab_pane_state(tab_state, tab->panes.size());
  state->undo.push(before_layout);
  if (mark_layout_dirty(session, state)) {
    state->status_text = std::string(status_text);
  }
  return true;
}

bool apply_pane_menu_action(AppSession *session, UiState *state, int pane_index,
                            const PaneMenuAction &action) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  TabUiState *tab_state = app_active_tab_state(state);
  if (tab == nullptr || tab_state == nullptr) return false;

  const int original_pane_count = static_cast<int>(tab->panes.size());
  const SketchLayout before_layout = session->layout;
  bool dock_changed = false;
  bool layout_changed = false;
  std::string_view success_status = "Workspace updated";
  switch (action.kind) {
    case PaneMenuActionKind::OpenAxisLimits:
      tab_state->active_pane_index = pane_index;
      open_axis_limits_editor(*session, state, pane_index);
      state->status_text = "Axis limits editor opened";
      return true;
    case PaneMenuActionKind::OpenCustomSeries:
      tab_state->active_pane_index = pane_index;
      open_custom_series_editor(state, preferred_custom_series_source(tab->panes[static_cast<size_t>(pane_index)]));
      state->status_text = "Custom series editor opened";
      return true;
    case PaneMenuActionKind::SplitLeft:
    case PaneMenuActionKind::SplitRight:
    case PaneMenuActionKind::SplitTop:
    case PaneMenuActionKind::SplitBottom: {
      constexpr PaneDropZone kZones[] = {PaneDropZone::Left, PaneDropZone::Right, PaneDropZone::Top, PaneDropZone::Bottom};
      const auto zone = kZones[static_cast<int>(action.kind) - static_cast<int>(PaneMenuActionKind::SplitLeft)];
      if (split_pane(tab, pane_index, zone)) {
        tab_state->active_pane_index = static_cast<int>(tab->panes.size()) - 1;
        dock_changed = true;
        layout_changed = true;
      }
      break;
    }
    case PaneMenuActionKind::ResetView:
      reset_shared_range(state, *session);
      state->follow_latest = session->data_mode == SessionDataMode::Stream;
      state->suppress_range_side_effects = true;
      clamp_shared_range(state, *session);
      persist_shared_range_to_tab(tab, *state);
      clear_pane_vertical_limits(&tab->panes[static_cast<size_t>(pane_index)]);
      layout_changed = true;
      success_status = "Plot view reset";
      break;
    case PaneMenuActionKind::ResetHorizontal:
      reset_shared_range(state, *session);
      state->follow_latest = session->data_mode == SessionDataMode::Stream;
      state->suppress_range_side_effects = true;
      clamp_shared_range(state, *session);
      persist_shared_range_to_tab(tab, *state);
      layout_changed = true;
      success_status = "Horizontal zoom reset";
      break;
    case PaneMenuActionKind::ResetVertical:
      clear_pane_vertical_limits(&tab->panes[static_cast<size_t>(pane_index)]);
      layout_changed = true;
      success_status = "Vertical zoom reset";
      break;
    case PaneMenuActionKind::Clear:
      clear_pane(tab, pane_index);
      tab_state->active_pane_index = pane_index;
      layout_changed = true;
      break;
    case PaneMenuActionKind::Close:
      if (close_pane(tab, pane_index)) {
        tab_state->active_pane_index = std::clamp(pane_index, 0, static_cast<int>(tab->panes.size()) - 1);
        layout_changed = true;
        dock_changed = static_cast<int>(tab->panes.size()) != original_pane_count;
      }
      break;
    case PaneMenuActionKind::None:
      return false;
  }

  if (!layout_changed) {
    return false;
  }
  return commit_tab_layout_change(session, state, tab, tab_state, before_layout, success_status, dock_changed);
}

bool apply_pane_drop_action(AppSession *session, UiState *state, const PaneDropAction &action) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  TabUiState *tab_state = app_active_tab_state(state);
  if (tab == nullptr || tab_state == nullptr) return false;

  if (!action.special_item_id.empty()) {
    const SpecialItemSpec *spec = special_item_spec(action.special_item_id);
    if (spec == nullptr) {
      return false;
    }
    if (action.zone == PaneDropZone::Center) {
      if (action.target_pane_index < 0 || action.target_pane_index >= static_cast<int>(tab->panes.size())) {
        return false;
      }
      if (!((tab->panes[static_cast<size_t>(action.target_pane_index)].kind == PaneKind::Plot
             && tab->panes[static_cast<size_t>(action.target_pane_index)].curves.empty())
             || pane_kind_is_special(tab->panes[static_cast<size_t>(action.target_pane_index)].kind))) {
        state->status_text = std::string(special_item_label(action.special_item_id)) + " can only replace another special pane or use an empty pane";
        return false;
      }
      const SketchLayout before_layout = session->layout;
      const bool changed = apply_special_item_to_pane(tab, tab_state, action.target_pane_index, spec->id);
      if (!changed) {
        state->status_text = std::string(special_item_label(action.special_item_id)) + " already shown in pane";
        return false;
      }
      return commit_tab_layout_change(session, state, tab, tab_state, before_layout,
                                      std::string(special_item_label(action.special_item_id)) + " added to pane",
                                      false);
    }
    const SketchLayout before_layout = session->layout;
    if (split_pane(tab, action.target_pane_index, action.zone)) {
      tab_state->active_pane_index = static_cast<int>(tab->panes.size()) - 1;
      const bool changed = apply_special_item_to_pane(tab, tab_state, tab_state->active_pane_index, spec->id);
      if (!changed) {
        return false;
      }
      return commit_tab_layout_change(session, state, tab, tab_state, before_layout,
                                      "Split pane and added " + std::string(special_item_label(action.special_item_id)),
                                      true);
    }
    return false;
  }

  if (action.from_browser) {
    if (action.browser_paths.empty()) return false;
    if (action.zone == PaneDropZone::Center) {
      const int inserted_count = add_path_curves_to_pane(session, state, action.target_pane_index, action.browser_paths);
      if (inserted_count > 0) {
        tab_state->active_pane_index = action.target_pane_index;
      }
      return inserted_count > 0;
    }
    const SketchLayout before_layout = session->layout;
    if (split_pane(tab, action.target_pane_index, action.zone)) {
      tab_state->active_pane_index = static_cast<int>(tab->panes.size()) - 1;
      int inserted_count = 0;
      for (const std::string &path : action.browser_paths) {
        if (app_find_route_series(*session, path) == nullptr) continue;
        if (add_curve_to_pane(tab, tab_state->active_pane_index,
                              make_curve_for_path(tab->panes[static_cast<size_t>(tab_state->active_pane_index)], path))) {
          ++inserted_count;
        }
      }
      if (inserted_count > 0) {
        return commit_tab_layout_change(session, state, tab, tab_state, before_layout,
                                        inserted_count == 1
                                          ? "Split pane and added " + action.browser_paths.front()
                                          : "Split pane and added " + std::to_string(inserted_count) + " curves",
                                        true);
      }
      return false;
    }
    return false;
  }

  if (action.curve_ref.tab_index < 0
      || action.curve_ref.tab_index >= static_cast<int>(session->layout.tabs.size())) {
    return false;
  }
  WorkspaceTab &source_tab = session->layout.tabs[static_cast<size_t>(action.curve_ref.tab_index)];
  if (action.curve_ref.pane_index < 0
      || action.curve_ref.pane_index >= static_cast<int>(source_tab.panes.size())) {
    return false;
  }
  const Pane &source_pane = source_tab.panes[static_cast<size_t>(action.curve_ref.pane_index)];
  if (action.curve_ref.curve_index < 0
      || action.curve_ref.curve_index >= static_cast<int>(source_pane.curves.size())) {
    return false;
  }
  const Curve curve = source_pane.curves[static_cast<size_t>(action.curve_ref.curve_index)];

  if (action.zone == PaneDropZone::Center) {
    const SketchLayout before_layout = session->layout;
    const bool inserted = add_curve_to_pane(tab, action.target_pane_index, curve);
    tab_state->active_pane_index = action.target_pane_index;
    if (inserted) {
      state->undo.push(before_layout);
      if (mark_layout_dirty(session, state)) {
        state->status_text = "Added " + app_curve_display_name(curve);
      }
    } else {
      state->status_text = "Curve already present";
    }
    return true;
  }
  const SketchLayout before_layout = session->layout;
  if (split_pane(tab, action.target_pane_index, action.zone, curve)) {
    tab_state->active_pane_index = static_cast<int>(tab->panes.size()) - 1;
    return commit_tab_layout_change(session, state, tab, tab_state, before_layout,
                                    "Split pane and added " + app_curve_display_name(curve),
                                    true);
  }
  return false;
}

ImGuiDir dock_direction(SplitOrientation orientation) {
  return orientation == SplitOrientation::Horizontal ? ImGuiDir_Left : ImGuiDir_Up;
}

void build_dock_tree(const WorkspaceNode &node, const WorkspaceTab &tab, int tab_runtime_id, ImGuiID dock_id) {
  if (node.is_pane) {
    if (node.pane_index >= 0 && node.pane_index < static_cast<int>(tab.panes.size())) {
      ImGui::DockBuilderDockWindow(
        pane_window_name(tab_runtime_id, node.pane_index, tab.panes[static_cast<size_t>(node.pane_index)]).c_str(),
        dock_id);
      if (ImGuiDockNode *dock_node = ImGui::DockBuilderGetNode(dock_id); dock_node != nullptr) {
        dock_node->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar |
                                 ImGuiDockNodeFlags_NoWindowMenuButton |
                                 ImGuiDockNodeFlags_NoCloseButton;
      }
    }
    return;
  }
  if (node.children.empty()) {
    return;
  }
  if (node.children.size() == 1) {
    build_dock_tree(node.children.front(), tab, tab_runtime_id, dock_id);
    return;
  }

  float remaining = 1.0f;
  ImGuiID current = dock_id;
  for (size_t i = 0; i + 1 < node.children.size(); ++i) {
    const float child_size = i < node.sizes.size() ? node.sizes[i] : 0.0f;
    const float ratio = remaining <= 0.0f ? 0.5f : std::clamp(child_size / remaining, 0.05f, 0.95f);
    ImGuiID child_id = 0;
    ImGuiID remainder_id = 0;
    ImGui::DockBuilderSplitNode(current, dock_direction(node.orientation), ratio, &child_id, &remainder_id);
    build_dock_tree(node.children[i], tab, tab_runtime_id, child_id);
    current = remainder_id;
    remaining = std::max(0.0f, remaining - child_size);
  }
  build_dock_tree(node.children.back(), tab, tab_runtime_id, current);
}

void ensure_dockspace(const WorkspaceTab &tab, TabUiState *tab_state, ImVec2 dockspace_size) {
  if (dockspace_size.x <= 0.0f || dockspace_size.y <= 0.0f || tab_state == nullptr) {
    return;
  }
  const bool size_changed = std::abs(tab_state->last_dockspace_size.x - dockspace_size.x) > 1.0f
                         || std::abs(tab_state->last_dockspace_size.y - dockspace_size.y) > 1.0f;
  if (!tab_state->dock_needs_build && !size_changed) {
    return;
  }

  const ImGuiID dockspace_id = dockspace_id_for_tab(tab_state->runtime_id);
  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace | ImGuiDockNodeFlags_AutoHideTabBar);
  ImGui::DockBuilderSetNodeSize(dockspace_id, dockspace_size);
  build_dock_tree(tab.root, tab, tab_state->runtime_id, dockspace_id);
  ImGui::DockBuilderFinish(dockspace_id);
  tab_state->dock_needs_build = false;
  tab_state->last_dockspace_size = dockspace_size;
}

void draw_pane_windows(AppSession *session, UiState *state) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  TabUiState *tab_state = app_active_tab_state(state);
  if (tab == nullptr || tab_state == nullptr) {
    return;
  }

  std::optional<std::pair<int, PaneMenuAction>> pending_menu_action;
  std::optional<int> pending_close_pane;
  std::optional<PaneDropAction> pending_drop_action;

  for (size_t i = 0; i < tab->panes.size(); ++i) {
    Pane &pane = tab->panes[i];
    std::optional<PaneMenuAction> menu_action;
    std::optional<PaneDropAction> drop_action;
    bool close_pane_requested = false;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(250, 250, 251));
    ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(194, 198, 204));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, color_rgb(252, 252, 253));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, color_rgb(252, 252, 253));
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, color_rgb(252, 252, 253));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
    const std::string window_name = pane_window_name(tab_state->runtime_id, static_cast<int>(i), pane);
    const bool opened = ImGui::Begin(window_name.c_str(), nullptr, flags);
    if (opened) {
      if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
          || (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) && ImGui::IsMouseClicked(0))) {
        tab_state->active_pane_index = static_cast<int>(i);
      }
      if (pane.kind == PaneKind::Map) {
        draw_map_pane(session, state, &pane, static_cast<int>(i));
      } else if (pane.kind == PaneKind::Camera) {
        draw_camera_pane(session, state, tab_state, static_cast<int>(i), pane);
      } else {
        draw_plot(*session, &pane, state);
      }
      draw_pane_frame_overlay();
      close_pane_requested = draw_pane_close_button_overlay();
      menu_action = draw_pane_context_menu(*tab, static_cast<int>(i));
      drop_action = draw_pane_drop_target(state->active_tab_index, static_cast<int>(i), pane);
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);
    if (!pending_menu_action.has_value() && menu_action.has_value()) {
      pending_menu_action = std::make_pair(static_cast<int>(i), *menu_action);
    }
    if (!pending_menu_action.has_value() && !pending_close_pane.has_value() && close_pane_requested) {
      pending_close_pane = static_cast<int>(i);
    }
    if (!pending_menu_action.has_value() && !pending_close_pane.has_value()
        && !pending_drop_action.has_value() && drop_action.has_value()) {
      pending_drop_action = *drop_action;
    }
  }

  if (pending_menu_action.has_value()) {
    apply_pane_menu_action(session, state, pending_menu_action->first, pending_menu_action->second);
    return;
  }
  if (pending_close_pane.has_value()) {
    PaneMenuAction action;
    action.kind = PaneMenuActionKind::Close;
    action.pane_index = *pending_close_pane;
    apply_pane_menu_action(session, state, *pending_close_pane, action);
    return;
  }
  if (pending_drop_action.has_value()) {
    apply_pane_drop_action(session, state, *pending_drop_action);
  }
}

void draw_workspace(AppSession *session, const UiMetrics &ui, UiState *state) {
  state->custom_series.selected = false;
  state->logs.selected = false;
  ImGui::SetNextWindowPos(ImVec2(ui.content_x, ui.content_y));
  ImGui::SetNextWindowSize(ImVec2(ui.content_w, ui.content_h));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(244, 246, 248));
  ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(186, 191, 198));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoScrollWithMouse;
  if (ImGui::Begin("##workspace_host", nullptr, flags)) {
    const int selection_request = state->requested_tab_index;
    std::optional<ImRect> rename_tab_rect;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(8.0f, 4.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    if (ImGui::BeginTabBar("##layout_tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
      enum class TabActionKind {
        None,
        New,
        Rename,
        Duplicate,
        Close,
      };
      TabActionKind pending_action = TabActionKind::None;
      int pending_tab_index = -1;
      bool custom_series_tab_open = state->custom_series.open;
      bool suppress_aux_tabs_this_frame = state->request_close_tab && session->layout.tabs.size() == 1;
      for (size_t i = 0; i < session->layout.tabs.size(); ++i) {
        const WorkspaceTab &tab = session->layout.tabs[i];
        const TabUiState &tab_ui = state->tabs[i];
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (static_cast<int>(i) == selection_request) {
          tab_flags |= ImGuiTabItemFlags_SetSelected;
        }
        bool tab_open = true;
        const bool opened = ImGui::BeginTabItem(tab_item_label(tab, tab_ui.runtime_id).c_str(), &tab_open, tab_flags);
        if (state->rename_tab_index == static_cast<int>(i)) {
          rename_tab_rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          pending_action = TabActionKind::Rename;
          pending_tab_index = static_cast<int>(i);
        }
        if (!tab_open) {
          pending_action = TabActionKind::Close;
          pending_tab_index = static_cast<int>(i);
          if (session->layout.tabs.size() == 1) {
            suppress_aux_tabs_this_frame = true;
          }
        }
        if (ImGui::BeginPopupContextItem()) {
          if (ImGui::MenuItem("New Tab")) {
            pending_action = TabActionKind::New;
          }
          if (ImGui::MenuItem("Rename Tab...")) {
            pending_action = TabActionKind::Rename;
            pending_tab_index = static_cast<int>(i);
          }
          if (ImGui::MenuItem("Duplicate Tab")) {
            pending_action = TabActionKind::Duplicate;
            pending_tab_index = static_cast<int>(i);
          }
          if (ImGui::MenuItem("Close Tab")) {
            pending_action = TabActionKind::Close;
            pending_tab_index = static_cast<int>(i);
          }
          ImGui::EndPopup();
        }
        if (opened) {
          state->active_tab_index = static_cast<int>(i);
          session->layout.current_tab_index = state->active_tab_index;
          if (i < state->tabs.size()) {
            ensure_dockspace(tab, &state->tabs[i], ImGui::GetContentRegionAvail());
          }
          ImGui::DockSpace(dockspace_id_for_tab(tab_ui.runtime_id),
                           ImVec2(0.0f, 0.0f),
                           ImGuiDockNodeFlags_AutoHideTabBar |
                             ImGuiDockNodeFlags_NoWindowMenuButton |
                             ImGuiDockNodeFlags_NoCloseButton);
          ImGui::EndTabItem();
        }
      }
      if (!suppress_aux_tabs_this_frame) {
        ImGuiTabItemFlags logs_flags = ImGuiTabItemFlags_None;
        if (state->logs.request_select) {
          logs_flags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem("Logs##workspace_logs", nullptr, logs_flags)) {
          state->logs.request_select = false;
          state->logs.selected = true;
          draw_logs_tab(session, state);
          ImGui::EndTabItem();
        }
        if (custom_series_tab_open) {
          ImGuiTabItemFlags custom_flags = ImGuiTabItemFlags_None;
          if (state->custom_series.request_select) {
            custom_flags |= ImGuiTabItemFlags_SetSelected;
          }
          if (ImGui::BeginTabItem("Custom Series##workspace_custom_series", &custom_series_tab_open, custom_flags)) {
            state->custom_series.request_select = false;
            state->custom_series.selected = true;
            draw_custom_series_editor(session, state);
            ImGui::EndTabItem();
          }
        }
      }
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 5.0f));
      ImGui::PushStyleColor(ImGuiCol_Tab, color_rgb(210, 217, 225));
      ImGui::PushStyleColor(ImGuiCol_TabHovered, color_rgb(224, 230, 237));
      ImGui::PushStyleColor(ImGuiCol_TabSelected, color_rgb(242, 245, 248));
      if (ImGui::TabItemButton("   ##new_tab_button", ImGuiTabItemFlags_Trailing)) {
        pending_action = TabActionKind::New;
      }
      {
        const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        const ImU32 color = ImGui::GetColorU32(color_rgb(72, 79, 88));
        const ImVec2 center((rect.Min.x + rect.Max.x) * 0.5f, (rect.Min.y + rect.Max.y) * 0.5f);
        constexpr float half_extent = 6.25f;
        constexpr float thickness = 2.0f;
        draw_list->AddLine(ImVec2(center.x - half_extent, center.y),
                           ImVec2(center.x + half_extent, center.y),
                           color,
                           thickness);
        draw_list->AddLine(ImVec2(center.x, center.y - half_extent),
                           ImVec2(center.x, center.y + half_extent),
                           color,
                           thickness);
      }
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("New Tab");
        ImGui::EndTooltip();
        ImGui::PopStyleVar();
      }
      ImGui::PopStyleColor(3);
      ImGui::PopStyleVar();
      ImGui::EndTabBar();

      if (!custom_series_tab_open) {
        state->custom_series.open = false;
        state->custom_series.request_select = false;
      }

      if (rename_tab_rect.has_value()) {
        draw_inline_tab_editor(session, state, *rename_tab_rect);
      }

      if (state->request_new_tab || pending_action == TabActionKind::New) {
        const SketchLayout before_layout = session->layout;
        create_runtime_tab(&session->layout, state);
        state->undo.push(before_layout);
        mark_layout_dirty(session, state);
        state->request_new_tab = false;
      } else if (pending_action == TabActionKind::Rename) {
        begin_rename_tab(session->layout, state, pending_tab_index);
      } else if (state->request_duplicate_tab || pending_action == TabActionKind::Duplicate) {
        if (pending_tab_index >= 0) {
          request_tab_selection(state, pending_tab_index);
        }
        const SketchLayout before_layout = session->layout;
        duplicate_runtime_tab(&session->layout, state);
        state->undo.push(before_layout);
        mark_layout_dirty(session, state);
        state->request_duplicate_tab = false;
      } else if (state->request_close_tab || pending_action == TabActionKind::Close) {
        if (pending_tab_index >= 0) {
          request_tab_selection(state, pending_tab_index);
        }
        const SketchLayout before_layout = session->layout;
        close_runtime_tab(&session->layout, state);
        state->undo.push(before_layout);
        mark_layout_dirty(session, state);
        state->request_close_tab = false;
      }
      if (state->requested_tab_index == selection_request) {
        state->requested_tab_index = -1;
      }
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
  }
  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(2);
}

int run(const Options &options) {
  try {
  const fs::path layout_path = options.layout.empty() ? fs::path() : resolve_layout_path(options.layout);
  AppSession session = {
    .layout_path = layout_path,
    .autosave_path = layout_path.empty() ? fs::path() : autosave_path_for_layout(layout_path),
    .route_name = options.route_name,
    .data_dir = options.data_dir,
    .dbc_override = {},
    .stream_source = StreamSourceConfig{.kind = is_local_stream_address(options.stream_address)
                                                 ? StreamSourceKind::CerealLocal
                                                 : StreamSourceKind::CerealRemote,
                                        .address = options.stream_address},
    .stream_buffer_seconds = options.stream_buffer_seconds,
    .data_mode = options.stream ? SessionDataMode::Stream : SessionDataMode::Route,
    .route_id = options.stream ? RouteIdentifier{} : parse_route_identifier(options.route_name),
    .layout = options.layout.empty() ? make_empty_layout() : load_sketch_layout(layout_path),
  };
  UiState ui_state;
  if (!layout_path.empty() && !session.autosave_path.empty() && fs::exists(session.autosave_path)) {
    session.layout = load_sketch_layout(session.autosave_path);
    ui_state.layout_dirty = true;
  }
  ui_state.undo.reset(session.layout);
  sync_ui_state(&ui_state, session.layout);
  sync_route_buffers(&ui_state, session);
  sync_stream_buffers(&ui_state, session);
  sync_layout_buffers(&ui_state, session);

  session.async_route_loading = session.data_mode == SessionDataMode::Route
    && options.show && options.output_path.empty() && !options.sync_load;
  if (session.data_mode == SessionDataMode::Route && !session.async_route_loading) {
    TerminalRouteProgress route_progress(::isatty(STDERR_FILENO) != 0);
    rebuild_session_route_data(&session, &ui_state, [&](const RouteLoadProgress &update) {
      route_progress.update(update);
    });
    route_progress.finish();
  }

  GlfwRuntime glfw_runtime(options);
  ImGuiRuntime imgui_runtime(glfw_runtime.window());
  configure_style();
  session.map_data = std::make_unique<MapDataManager>();
  for (std::unique_ptr<CameraFeedView> &feed : session.pane_camera_feeds) {
    feed = std::make_unique<CameraFeedView>();
  }
  sync_camera_feeds(&session);

  if (session.async_route_loading) {
    session.route_loader = std::make_unique<AsyncRouteLoader>(::isatty(STDERR_FILENO) != 0);
    start_async_route_load(&session, &ui_state);
  } else if (session.data_mode == SessionDataMode::Stream) {
    session.stream_poller = std::make_unique<StreamPoller>();
    start_stream_session(&session, &ui_state, session.stream_source, session.stream_buffer_seconds);
  }

  const bool should_capture = !options.output_path.empty();
  const fs::path output_path = should_capture ? fs::path(options.output_path) : fs::path();
  const bool capture_has_map = should_capture && active_tab_has_map_pane(session.layout);
  if (options.show) {
    bool captured = false;
    const auto capture_ready_at = std::chrono::steady_clock::now() + (capture_has_map ? std::chrono::milliseconds(1800)
                                                                                      : std::chrono::milliseconds(0));
    while (!glfwWindowShouldClose(glfw_runtime.window())) {
      const bool capture_ready = std::chrono::steady_clock::now() >= capture_ready_at;
      const fs::path *capture_path = (!captured && should_capture && capture_ready) ? &output_path : nullptr;
      render_frame(glfw_runtime.window(), &session, &ui_state, capture_path);
      captured = captured || capture_path != nullptr;
    }
  } else {
    render_frame(glfw_runtime.window(), &session, &ui_state, nullptr);
    if (should_capture) {
      for (int i = 0; i < 3; ++i) {
        render_frame(glfw_runtime.window(), &session, &ui_state, nullptr);
      }
      if (capture_has_map) {
        for (int i = 0; i < 18; ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          render_frame(glfw_runtime.window(), &session, &ui_state, nullptr);
        }
      }
      render_frame(glfw_runtime.window(), &session, &ui_state, &output_path);
    }
  }
  if (session.stream_poller) {
    session.stream_poller->stop();
  }
  session.map_data.reset();
  for (std::unique_ptr<CameraFeedView> &feed : session.pane_camera_feeds) {
    feed.reset();
  }
  return 0;
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n";
    return 1;
  }
}
