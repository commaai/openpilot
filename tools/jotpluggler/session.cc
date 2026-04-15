#include "tools/jotpluggler/internal.h"

#include "imgui_internal.h"

#include <array>
#include <cmath>
#include <cstdlib>

namespace fs = std::filesystem;

const RouteSeries *app_find_route_series(const AppSession &session, const std::string &path) {
  auto it = session.series_by_path.find(path);
  return it == session.series_by_path.end() ? nullptr : it->second;
}

void sync_camera_feeds(AppSession *session) {
  for (size_t i = 0; i < kCameraViewSpecs.size(); ++i) {
    if (session->pane_camera_feeds[i]) {
      session->pane_camera_feeds[i]->setCameraIndex(session->route_data.*(kCameraViewSpecs[i].route_member), kCameraViewSpecs[i].view);
    }
  }
}

void apply_route_data(AppSession *session, UiState *state, RouteData route_data) {
  if (!route_data.route_id.empty()) {
    session->route_id = route_data.route_id;
  } else if (session->route_name.empty() && session->data_mode == SessionDataMode::Route) {
    session->route_id = {};
  }
  session->route_data = std::move(route_data);
  rebuild_route_index(session);
  rebuild_browser_nodes(session, state);
  state->browser_nodes_dirty = false;
  refresh_all_custom_curves(session, state);
  sync_camera_feeds(session);
  state->has_shared_range = false;
  state->has_tracker_time = false;
  reset_shared_range(state, *session);
}

bool restore_undo_layout(AppSession *session, UiState *state, const SketchLayout &layout, const char *status_text) {
  session->layout = layout;
  cancel_rename_tab(state);
  state->custom_series.request_select = false;
  state->active_tab_index = std::clamp(layout.current_tab_index, 0, std::max(0, static_cast<int>(layout.tabs.size()) - 1));
  state->requested_tab_index = state->active_tab_index;
  sync_ui_state(state, session->layout);
  mark_all_docks_dirty(state);
  const bool autosave_ok = autosave_layout(session, state);
  if (autosave_ok) {
    state->status_text = status_text;
  }
  return autosave_ok;
}

bool apply_undo(AppSession *session, UiState *state) {
  if (!state->undo.can_undo()) {
    return false;
  }
  return restore_undo_layout(session, state, state->undo.undo(), "Undo");
}

bool apply_redo(AppSession *session, UiState *state) {
  if (!state->undo.can_redo()) {
    return false;
  }
  return restore_undo_layout(session, state, state->undo.redo(), "Redo");
}

std::optional<std::pair<double, double>> tab_default_x_range(const WorkspaceTab &tab) {
  bool found = false;
  double min_value = 0.0;
  double max_value = 1.0;
  for (const Pane &pane : tab.panes) {
    if (!pane.range.valid || pane.range.right <= pane.range.left) continue;
    if (!found) {
      min_value = pane.range.left;
      max_value = pane.range.right;
      found = true;
    } else {
      min_value = std::min(min_value, pane.range.left);
      max_value = std::max(max_value, pane.range.right);
    }
  }
  if (!found) return std::nullopt;
  return std::make_pair(min_value, max_value);
}

bool infer_stream_follow_state(const UiState &state, const AppSession &session) {
  if (session.data_mode != SessionDataMode::Stream || !state.has_shared_range || !session.route_data.has_time_range) {
    return false;
  }
  const double target_span = std::max(MIN_HORIZONTAL_ZOOM_SECONDS, session.stream_buffer_seconds);
  const double current_span = std::max(0.0, state.x_view_max - state.x_view_min);
  const double edge_epsilon = std::max(0.05, target_span * 0.02);
  return std::abs(state.x_view_max - state.route_x_max) <= edge_epsilon
      && std::abs(current_span - target_span) <= edge_epsilon;
}

void ensure_shared_range(UiState *state, const AppSession &session) {
  if (session.route_data.has_time_range) {
    state->route_x_min = session.route_data.x_min;
    state->route_x_max = session.route_data.x_max;
  } else {
    state->route_x_min = 0.0;
    state->route_x_max = 1.0;
  }

  if (state->has_shared_range) {
    return;
  }

  if (session.data_mode == SessionDataMode::Stream) {
    const double target_span = std::max(MIN_HORIZONTAL_ZOOM_SECONDS, session.stream_buffer_seconds);
    if (session.route_data.has_time_range) {
      state->x_view_max = state->route_x_max;
      state->x_view_min = state->x_view_max - target_span;
    } else {
      state->x_view_min = 0.0;
      state->x_view_max = target_span;
    }
    if (state->x_view_max <= state->x_view_min) {
      state->x_view_max = state->x_view_min + 1.0;
    }
    state->has_shared_range = true;
    if (!state->has_tracker_time || state->tracker_time < state->route_x_min || state->tracker_time > state->route_x_max) {
      state->tracker_time = state->route_x_max;
      state->has_tracker_time = session.route_data.has_time_range;
    }
    return;
  }

  if (const WorkspaceTab *tab = app_active_tab(session.layout, *state); tab != nullptr) {
    if (std::optional<std::pair<double, double>> tab_range = tab_default_x_range(*tab); tab_range.has_value()) {
      state->x_view_min = tab_range->first;
      state->x_view_max = tab_range->second;
      state->has_shared_range = true;
      if (!state->has_tracker_time || state->tracker_time < state->route_x_min || state->tracker_time > state->route_x_max) {
        state->tracker_time = state->route_x_min;
        state->has_tracker_time = true;
      }
      return;
    }
  }

  state->x_view_min = state->route_x_min;
  state->x_view_max = state->route_x_max;
  if (state->x_view_max <= state->x_view_min) {
    state->x_view_max = state->x_view_min + 1.0;
  }
  state->has_shared_range = true;
  if (!state->has_tracker_time || state->tracker_time < state->route_x_min || state->tracker_time > state->route_x_max) {
    state->tracker_time = state->route_x_min;
    state->has_tracker_time = true;
  }
}

void clamp_shared_range(UiState *state, const AppSession &session) {
  if (!state->has_shared_range) {
    return;
  }
  const double min_span = MIN_HORIZONTAL_ZOOM_SECONDS;
  double span = state->x_view_max - state->x_view_min;
  if (span < min_span) {
    const double center = 0.5 * (state->x_view_min + state->x_view_max);
    span = min_span;
    state->x_view_min = center - span * 0.5;
    state->x_view_max = center + span * 0.5;
  }
  if (session.data_mode == SessionDataMode::Stream) {
    if (session.route_data.has_time_range && state->x_view_max > state->route_x_max) {
      state->x_view_min -= state->x_view_max - state->route_x_max;
      state->x_view_max = state->route_x_max;
    }
    if (state->x_view_max <= state->x_view_min) {
      state->x_view_max = state->x_view_min + min_span;
    }
    if (state->has_tracker_time && session.route_data.has_time_range) {
      state->tracker_time = std::clamp(state->tracker_time, state->route_x_min, state->route_x_max);
    }
    if (session.route_data.has_time_range) {
      state->follow_latest = infer_stream_follow_state(*state, session);
    }
    return;
  }
  if (state->route_x_max > state->route_x_min) {
    if (state->x_view_min < state->route_x_min) {
      state->x_view_max += state->route_x_min - state->x_view_min;
      state->x_view_min = state->route_x_min;
    }
    if (state->x_view_max > state->route_x_max) {
      state->x_view_min -= state->x_view_max - state->route_x_max;
      state->x_view_max = state->route_x_max;
    }
    if (state->x_view_min < state->route_x_min) {
      state->x_view_min = state->route_x_min;
    }
    if (state->x_view_max <= state->x_view_min) {
      state->x_view_max = std::min(state->route_x_max, state->x_view_min + min_span);
    }
  }
  if (state->has_tracker_time) {
    state->tracker_time = std::clamp(state->tracker_time, state->route_x_min, state->route_x_max);
  }
}

void reset_shared_range(UiState *state, const AppSession &session) {
  state->has_shared_range = false;
  ensure_shared_range(state, session);
  clamp_shared_range(state, session);
}

void update_follow_range(UiState *state, const AppSession &session) {
  if (!state->follow_latest || !state->has_shared_range) {
    return;
  }
  const double span = session.data_mode == SessionDataMode::Stream
    ? std::max(MIN_HORIZONTAL_ZOOM_SECONDS, session.stream_buffer_seconds)
    : std::max(MIN_HORIZONTAL_ZOOM_SECONDS, state->x_view_max - state->x_view_min);
  const double route_span = state->route_x_max - state->route_x_min;
  if (route_span <= 0.0) {
    return;
  }
  state->x_view_max = state->route_x_max;
  state->x_view_min = state->x_view_max - span;
  clamp_shared_range(state, session);
}

void advance_playback(UiState *state, const AppSession &session) {
  if (!state->playback_playing || !state->has_shared_range || state->route_x_max <= state->route_x_min) {
    return;
  }

  const double delta = std::max(0.0, static_cast<double>(ImGui::GetIO().DeltaTime)) * state->playback_rate;
  const double view_span = std::max(MIN_HORIZONTAL_ZOOM_SECONDS, state->x_view_max - state->x_view_min);
  const double loop_min = state->playback_loop
    ? std::clamp(state->x_view_min, state->route_x_min, state->route_x_max)
    : state->route_x_min;
  const double loop_max = state->playback_loop
    ? std::clamp(state->x_view_max, state->route_x_min, state->route_x_max)
    : state->route_x_max;

  state->tracker_time += delta;
  if (state->tracker_time >= loop_max) {
    if (state->playback_loop) {
      state->tracker_time = loop_min;
    } else {
      state->tracker_time = state->route_x_max;
      state->playback_playing = false;
    }
  }

  if (!state->playback_loop) {
    constexpr double kScrollStartFraction = 0.70;
    const double scroll_anchor = state->x_view_min + view_span * kScrollStartFraction;
    if (state->tracker_time > scroll_anchor && state->x_view_max < state->route_x_max) {
      state->x_view_min = state->tracker_time - view_span * kScrollStartFraction;
      state->x_view_max = state->x_view_min + view_span;
      clamp_shared_range(state, session);
    } else if (state->tracker_time < state->x_view_min || state->tracker_time > state->x_view_max) {
      state->x_view_min = state->tracker_time - view_span * 0.5;
      state->x_view_max = state->x_view_min + view_span;
      clamp_shared_range(state, session);
    }
  }
}

void step_tracker(UiState *state, double direction) {
  if (!state->has_shared_range) {
    return;
  }
  state->tracker_time += direction * std::max(0.001, state->playback_step);
  state->tracker_time = std::clamp(state->tracker_time, state->route_x_min, state->route_x_max);
}

const char *log_selector_name(LogSelector selector) {
  static constexpr const char *kLabels[] = {"a", "r", "q"};
  const size_t index = static_cast<size_t>(selector);
  return index < std::size(kLabels) ? kLabels[index] : kLabels[0];
}

const char *log_selector_description(LogSelector selector) {
  static constexpr const char *kLabels[] = {
    "any of rlog or qlog",
    "rlog only",
    "qlog only",
  };
  const size_t index = static_cast<size_t>(selector);
  return index < std::size(kLabels) ? kLabels[index] : kLabels[0];
}

std::string shorten_route_part(std::string_view text, size_t keep) {
  if (text.size() <= keep) {
    return std::string(text);
  }
  return std::string(text.substr(0, keep));
}

bool parse_slice_spec(std::string_view text, int *begin, int *end) {
  const auto parse_nonnegative = [](std::string_view value, int *out) {
    if (value.empty()) return false;
    char *end_ptr = nullptr;
    const long parsed = std::strtol(std::string(value).c_str(), &end_ptr, 10);
    if (end_ptr == nullptr || *end_ptr != '\0' || parsed < 0) {
      return false;
    }
    *out = static_cast<int>(parsed);
    return true;
  };
  const std::string trimmed = util::strip(std::string(text));
  if (trimmed.empty()) {
    return false;
  }
  const size_t colon = trimmed.find(':');
  int parsed_begin = 0;
  if (!parse_nonnegative(trimmed.substr(0, colon), &parsed_begin)) {
    return false;
  }
  int parsed_end = parsed_begin;
  if (colon != std::string::npos) {
    const std::string end_text = trimmed.substr(colon + 1);
    if (end_text.empty()) {
      parsed_end = -1;
    } else if (!parse_nonnegative(end_text, &parsed_end) || parsed_end < parsed_begin) {
      return false;
    }
  }
  *begin = parsed_begin;
  *end = parsed_end;
  return true;
}

std::string format_duration_short(double seconds) {
  const double clamped = std::max(0.0, seconds);
  const int total_ms = static_cast<int>(std::round(clamped * 1000.0));
  const int minutes = total_ms / 60000;
  const int rem_ms = total_ms % 60000;
  const int secs = rem_ms / 1000;
  const int millis = rem_ms % 1000;
  return util::string_format("%d:%02d.%03d", minutes, secs, millis);
}

bool apply_route_identifier(AppSession *session, UiState *state, const RouteIdentifier &route_id, const char *status_text) {
  if (route_id.empty()) {
    return false;
  }
  if (!reload_session(session, state, route_id.full_spec(), session->data_dir)) {
    return false;
  }
  state->status_text = status_text;
  return true;
}

bool apply_route_slice_change(AppSession *session, UiState *state, std::string_view slice_text) {
  int begin = 0;
  int end = 0;
  if (!parse_slice_spec(slice_text, &begin, &end)) {
    state->error_text = "Slice must be N, N:, or N:M.";
    state->open_error_popup = true;
    return false;
  }
  RouteIdentifier next = session->route_id;
  next.slice_begin = begin;
  next.slice_end = end;
  next.slice_explicit = true;
  return apply_route_identifier(session, state, next, "Updated route slice");
}

bool apply_route_selector_change(AppSession *session, UiState *state, LogSelector selector) {
  RouteIdentifier next = session->route_id;
  next.selector = selector;
  next.selector_explicit = true;
  return apply_route_identifier(session, state, next, "Updated log selector");
}

ImU32 route_chip_part_color(int part_index, bool explicit_part) {
  constexpr std::array<std::array<int, 3>, 4> BASE = {{
    {70, 96, 126},   // dongle
    {100, 86, 148},  // log id
    {72, 112, 86},   // slice
    {156, 104, 38},  // selector
  }};
  const std::array<int, 3> &base = BASE[static_cast<size_t>(std::clamp(part_index, 0, 3))];
  if (explicit_part) {
    return ImGui::GetColorU32(color_rgb(base[0], base[1], base[2]));
  }
  const int gray = 144;
  return ImGui::GetColorU32(color_rgb((base[0] + gray) / 2, (base[1] + gray) / 2, (base[2] + gray) / 2));
}

bool draw_route_chip_text_button(const char *id,
                                 std::string_view text,
                                 ImVec2 pos,
                                 ImU32 color,
                                 ImDrawList *draw_list,
                                 const char *tooltip = nullptr) {
  const ImVec2 size = ImGui::CalcTextSize(text.data(), text.data() + text.size());
  ImGui::SetCursorScreenPos(pos);
  ImGui::InvisibleButton(id, size);
  const bool hovered = ImGui::IsItemHovered();
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    draw_list->AddRectFilled(ImVec2(pos.x - 5.0f, pos.y - 1.0f),
                             ImVec2(pos.x + size.x + 5.0f, pos.y + size.y + 2.0f),
                             ImGui::GetColorU32(color_rgb(225, 231, 239, 0.95f)), 0.0f);
  }
  draw_list->AddText(pos, color, text.data(), text.data() + text.size());
  if (tooltip != nullptr && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted(tooltip);
    ImGui::EndTooltip();
  }
  return ImGui::IsItemClicked(ImGuiMouseButton_Left);
}

void draw_route_copy_feedback(UiState *state, ImDrawList *draw_list, ImVec2 chip_max) {
  if (state->route_copy_feedback_text.empty()) {
    return;
  }
  const double now = ImGui::GetTime();
  if (now >= state->route_copy_feedback_until) {
    state->route_copy_feedback_text.clear();
    state->route_copy_feedback_until = 0.0;
    return;
  }

  const float alpha = static_cast<float>(std::clamp((state->route_copy_feedback_until - now) / 1.1, 0.0, 1.0));
  const ImVec2 text_size = ImGui::CalcTextSize(state->route_copy_feedback_text.c_str());
  const ImVec2 pad(9.0f, 5.0f);
  const ImVec2 bubble_min(chip_max.x - text_size.x - pad.x * 2.0f, chip_max.y + 7.0f);
  const ImVec2 bubble_max(chip_max.x, bubble_min.y + text_size.y + pad.y * 2.0f);
  draw_list->AddRectFilled(bubble_min, bubble_max,
                           ImGui::GetColorU32(color_rgb(46, 125, 80, 0.96f * alpha)), 7.0f);
  draw_list->AddRect(bubble_min, bubble_max,
                     ImGui::GetColorU32(color_rgb(35, 96, 61, 0.9f * alpha)), 7.0f, 0, 1.0f);
  draw_list->AddText(ImVec2(std::floor(bubble_min.x + pad.x), std::floor(bubble_min.y + pad.y)),
                     ImGui::GetColorU32(color_rgb(247, 251, 248, alpha)),
                     state->route_copy_feedback_text.c_str());
}

void draw_route_info_popup(AppSession *session, UiState *state, ImVec2 anchor) {
  if (session->route_id.empty()) {
    return;
  }
  ImGui::SetNextWindowPos(anchor, ImGuiCond_Appearing);
  ImGui::SetNextWindowSizeConstraints(ImVec2(300.0f, 0.0f), ImVec2(420.0f, FLT_MAX));
  if (!ImGui::BeginPopup("##route_info_popup",
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings)) {
    return;
  }

  ImGui::TextUnformatted("Route Info");
  ImGui::Separator();
  app_push_mono_font();
  ImGui::TextUnformatted(session->route_id.canonical().c_str());
  app_pop_mono_font();

  const char *copy_icon = icon::CLIPBOARD;
  const char *link_icon = icon::BOX_ARROW_UP_RIGHT;
  const std::string useradmin_label = std::string("Useradmin ") + link_icon;
  const std::string connect_label = std::string("comma connect ") + link_icon;
  if (ImGui::Button(copy_icon, ImVec2(34.0f, 26.0f))) {
    ImGui::SetClipboardText(session->route_id.canonical().c_str());
    state->status_text = "Copied route to clipboard";
    state->route_copy_feedback_text = "Copied";
    state->route_copy_feedback_until = ImGui::GetTime() + 1.1;
  }
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted("Copy route");
    ImGui::EndTooltip();
  }
  ImGui::SameLine();
  if (ImGui::Button(useradmin_label.c_str(), ImVec2(132.0f, 26.0f))) {
    open_external_url(route_useradmin_url(session->route_id));
    state->status_text = "Opened useradmin";
  }
  ImGui::SameLine();
  if (ImGui::Button(connect_label.c_str(), ImVec2(156.0f, 26.0f))) {
    open_external_url(route_connect_url(session->route_id));
    state->status_text = "Opened comma connect";
  }

  ImGui::Spacing();
  const int loaded_begin = session->route_id.available_begin;
  const int loaded_end = session->route_id.available_end;
  const int loaded_count = loaded_end >= loaded_begin ? (loaded_end - loaded_begin + 1) : 0;
  ImGui::Text("Duration   %s", format_duration_short(session->route_data.x_max - session->route_data.x_min).c_str());
  ImGui::Text("Segments   %s (%d)", session->route_id.display_slice().c_str(), loaded_count);
  ImGui::Text("Selector   %s", log_selector_description(session->route_id.selector));
  if (!session->route_data.car_fingerprint.empty()) {
    ImGui::TextWrapped("Car        %s", session->route_data.car_fingerprint.c_str());
  }
  if (!session->route_data.dbc_name.empty()) {
    ImGui::TextWrapped("DBC        %s", session->route_data.dbc_name.c_str());
  }

  ImGui::EndPopup();
}

void draw_route_id_chip(AppSession *session, UiState *state) {
  if (session->data_mode != SessionDataMode::Route || session->route_id.empty()) {
    return;
  }

  ImGuiWindow *window = ImGui::GetCurrentWindow();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const RouteIdentifier &route_id = session->route_id;
  app_push_bold_font();
  const std::string dongle_text = shorten_route_part(route_id.dongle_id, 8);
  const std::string log_text = shorten_route_part(route_id.log_id, 16);
  const std::string slice_text = route_id.display_slice();
  const std::string selector_text(1, route_id.selector_char());
  const std::string sep_text = " / ";

  const ImVec2 dongle_size = ImGui::CalcTextSize(dongle_text.c_str());
  const ImVec2 log_size = ImGui::CalcTextSize(log_text.c_str());
  const ImVec2 slice_size = state->editing_route_slice
    ? ImVec2(68.0f, ImGui::GetFrameHeight())
    : ImGui::CalcTextSize(slice_text.c_str());
  const ImVec2 selector_size = ImGui::CalcTextSize(selector_text.c_str());
  const ImVec2 sep_size = ImGui::CalcTextSize(sep_text.c_str());
  constexpr float chip_pad_x = 12.0f;
  constexpr float info_size = 18.0f;
  const float chip_h = 28.0f;
  const float chip_w = chip_pad_x * 2.0f + dongle_size.x + sep_size.x + log_size.x + sep_size.x
                     + slice_size.x + sep_size.x + selector_size.x + 10.0f + info_size;
  const float menu_right = window->Pos.x + window->Size.x - 8.0f;
  const float cursor_x = ImGui::GetCursorScreenPos().x + 4.0f;
  const float chip_x = std::clamp(cursor_x, window->Pos.x + 48.0f, std::max(window->Pos.x + 48.0f, menu_right - chip_w));
  const float chip_y = std::floor(window->Pos.y + std::max(0.0f, (window->Size.y - chip_h) * 0.5f));
  const ImVec2 chip_min(chip_x, chip_y);
  const ImVec2 chip_max(chip_x + chip_w, chip_y + chip_h);
  const float text_y = std::floor(chip_y + std::max(0.0f, (chip_h - ImGui::GetTextLineHeight()) * 0.5f));
  const ImU32 chip_bg = ImGui::GetColorU32(color_rgb(247, 249, 252));
  const ImU32 chip_border = ImGui::GetColorU32(color_rgb(184, 191, 200));
  const ImU32 sep = ImGui::GetColorU32(color_rgb(162, 170, 178));
  draw_list->AddRectFilled(chip_min, chip_max, chip_bg, 0.0f);
  draw_list->AddRect(chip_min, chip_max, chip_border, 0.0f, 0, 1.0f);

  float x = chip_x + chip_pad_x;
  const bool dongle_click = draw_route_chip_text_button(
    "##route_dongle", dongle_text, ImVec2(x, text_y), route_chip_part_color(0, true), draw_list,
    "Device identifier");
  x += dongle_size.x;
  draw_list->AddText(ImVec2(x, text_y), sep, sep_text.c_str());
  x += sep_size.x;
  const bool log_click = draw_route_chip_text_button(
    "##route_log", log_text, ImVec2(x, text_y), route_chip_part_color(1, true), draw_list,
    "Route identifier");
  x += log_size.x;
  draw_list->AddText(ImVec2(x, text_y), sep, sep_text.c_str());
  x += sep_size.x;

  if (state->editing_route_slice) {
    ImGui::SetCursorScreenPos(ImVec2(x - 4.0f, chip_y + 1.0f));
    ImGui::SetNextItemWidth(76.0f);
    if (state->focus_route_slice_input) {
      ImGui::SetKeyboardFocusHere();
      state->focus_route_slice_input = false;
    }
    const bool applied = input_text_string("##route_slice_edit", &state->route_slice_buffer,
                                           ImGuiInputTextFlags_EnterReturnsTrue);
    const bool deactivated = ImGui::IsItemDeactivated();
    const bool clicked_elsewhere = ImGui::IsMouseClicked(ImGuiMouseButton_Left)
                                && !ImGui::IsItemHovered()
                                && !ImGui::IsItemActive();
    if (applied) {
      if (apply_route_slice_change(session, state, state->route_slice_buffer)) {
        state->editing_route_slice = false;
      }
    } else if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      state->editing_route_slice = false;
    } else if (deactivated || clicked_elsewhere) {
      const std::string trimmed = util::strip(state->route_slice_buffer);
      if (trimmed != route_id.display_slice()) {
        int begin = 0;
        int end = 0;
        if (parse_slice_spec(trimmed, &begin, &end)) {
          apply_route_slice_change(session, state, trimmed);
        } else {
          state->status_text = "Canceled route slice edit";
        }
      }
      state->editing_route_slice = false;
    }
    x += slice_size.x;
  } else {
    const bool slice_click = draw_route_chip_text_button(
      "##route_slice", slice_text, ImVec2(x, text_y),
      route_chip_part_color(2, route_id.slice_explicit), draw_list,
      "Segment range");
    if (slice_click) {
      state->editing_route_slice = true;
      state->focus_route_slice_input = true;
      state->route_slice_buffer = route_id.display_slice();
    }
    x += slice_size.x;
  }

  draw_list->AddText(ImVec2(x, text_y), sep, sep_text.c_str());
  x += sep_size.x;
  const bool selector_click = draw_route_chip_text_button(
    "##route_selector", selector_text, ImVec2(x, text_y),
    route_chip_part_color(3, route_id.selector_explicit), draw_list,
    "Log selector");
  if (selector_click) {
    ImGui::OpenPopup("##route_selector_popup");
  }
  x += selector_size.x + 10.0f;

  const ImVec2 info_center(x + info_size * 0.5f, chip_y + chip_h * 0.5f);
  ImGui::SetCursorScreenPos(ImVec2(x, chip_y + (chip_h - info_size) * 0.5f));
  ImGui::InvisibleButton("##route_info_button", ImVec2(info_size, info_size));
  const bool info_hovered = ImGui::IsItemHovered();
  if (info_hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  }
  draw_list->AddCircleFilled(info_center, info_size * 0.5f,
                             ImGui::GetColorU32(info_hovered ? color_rgb(220, 229, 240) : color_rgb(239, 243, 248)));
  draw_list->AddCircle(info_center, info_size * 0.5f, chip_border, 20, 1.0f);
  const char *info_text = icon::INFO_CIRCLE;
  const ImVec2 info_text_size = ImGui::CalcTextSize(info_text);
  draw_list->AddText(ImVec2(std::floor(info_center.x - info_text_size.x * 0.5f),
                            std::floor(info_center.y - info_text_size.y * 0.5f)),
                     route_chip_part_color(0, true), info_text);
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted("Route details");
    ImGui::EndTooltip();
  }
  if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
    ImGui::OpenPopup("##route_info_popup");
  }

  app_pop_bold_font();

  if (dongle_click || log_click) {
    ImGui::SetClipboardText(route_id.canonical().c_str());
    state->status_text = "Copied route to clipboard";
    state->route_copy_feedback_text = "Copied";
    state->route_copy_feedback_until = ImGui::GetTime() + 1.1;
  }

  ImGui::SetNextWindowPos(ImVec2(chip_max.x - 60.0f, chip_max.y + 4.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopup("##route_selector_popup")) {
    for (LogSelector selector : {LogSelector::Auto, LogSelector::RLog, LogSelector::QLog}) {
      const bool selected = route_id.selector == selector;
      const std::string label = std::string(log_selector_name(selector)) + "  " + log_selector_description(selector);
      if (ImGui::Selectable(label.c_str(), selected) && !selected) {
        apply_route_selector_change(session, state, selector);
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndPopup();
  }

  draw_route_copy_feedback(state, draw_list, chip_max);
  draw_route_info_popup(session, state, ImVec2(std::max(window->Pos.x + 16.0f, chip_max.x - 360.0f), chip_max.y + 6.0f));
}

std::string format_cache_bytes(uint64_t bytes) {
  if (bytes >= (1ULL << 30)) {
    return util::string_format("%.1f GiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
  } else if (bytes >= (1ULL << 20)) {
    return util::string_format("%.1f MiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 20));
  } else if (bytes >= (1ULL << 10)) {
    return util::string_format("%.1f KiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 10));
  }
  return util::string_format("%llu B", static_cast<unsigned long long>(bytes));
}

MapCacheStats directory_cache_stats(const fs::path &root) {
  MapCacheStats stats;
  std::error_code ec;
  if (!fs::exists(root, ec)) {
    return stats;
  }
  fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
  for (const fs::directory_entry &entry : it) {
    if (ec) {
      ec.clear();
      continue;
    }
    const fs::file_status status = entry.symlink_status(ec);
    if (ec || !fs::is_regular_file(status)) {
      ec.clear();
      continue;
    }
    const uintmax_t size = entry.file_size(ec);
    if (!ec) {
      stats.bytes += static_cast<uint64_t>(size);
      ++stats.files;
    } else {
      ec.clear();
    }
  }
  return stats;
}

float draw_main_menu_bar(AppSession *session, UiState *state) {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(7.0f, 5.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(9.0f, 6.0f));
  float height = ImGui::GetFrameHeight();
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Undo", "Ctrl+Z", false, state->undo.can_undo())) {
        apply_undo(session, state);
      }
      if (ImGui::MenuItem("Redo", "Ctrl+Shift+Z", false, state->undo.can_redo())) {
        apply_redo(session, state);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Open Route...")) {
        state->open_open_route = true;
      }
      if (ImGui::MenuItem("Stream...")) {
        state->open_stream = true;
      }
      if (ImGui::MenuItem("Find Signal...", "Ctrl+F")) {
        state->open_find_signal = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("New Layout")) {
        start_new_layout(session, state);
      }
      if (ImGui::MenuItem("Load Layout...")) {
        state->open_load_layout = true;
      }
      if (ImGui::MenuItem("Save Layout")) {
        state->request_save_layout = true;
      }
      if (ImGui::MenuItem("Save Layout As...")) {
        state->open_save_layout = true;
      }
      if (ImGui::MenuItem("Reset Layout")) {
        state->request_reset_layout = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Show DEPRECATED Fields", nullptr, state->show_deprecated_fields)) {
        state->show_deprecated_fields = !state->show_deprecated_fields;
        rebuild_browser_nodes(session, state);
      }
      if (ImGui::MenuItem("Show FPS", nullptr, state->show_fps_overlay)) {
        state->show_fps_overlay = !state->show_fps_overlay;
      }
      if (ImGui::MenuItem("Preferences...")) {
        state->open_preferences = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Reset Plot View")) {
        reset_shared_range(state, *session);
        state->follow_latest = session->data_mode == SessionDataMode::Stream;
        clamp_shared_range(state, *session);
        state->suppress_range_side_effects = true;
        state->status_text = "Plot view reset";
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Close")) {
        state->request_close = true;
      }
      ImGui::EndMenu();
    }
    ImGui::SameLine(0.0f, 8.0f);
    draw_route_id_chip(session, state);
    height = ImGui::GetWindowSize().y;
    ImGui::EndMainMenuBar();
  }
  ImGui::PopStyleVar(2);
  return height;
}
