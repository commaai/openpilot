#include "tools/jotpluggler/internal.h"

std::string dbc_combo_label(const AppSession &session) {
  if (!session.dbc_override.empty()) return session.dbc_override;
  if (!session.route_data.dbc_name.empty()) return "Auto: " + session.route_data.dbc_name;
  return "Auto";
}

float timeline_time_to_x(double time_value, double route_min, double route_max, float x_min, float x_max) {
  const double span = route_max - route_min;
  if (span <= 0.0) {
    return x_min;
  }
  const double ratio = (time_value - route_min) / span;
  return x_min + static_cast<float>(ratio * static_cast<double>(x_max - x_min));
}

double timeline_x_to_time(float x, double route_min, double route_max, float x_min, float x_max) {
  const float width = std::max(1.0f, x_max - x_min);
  const float clamped_x = std::clamp(x, x_min, x_max);
  const double ratio = static_cast<double>((clamped_x - x_min) / width);
  return route_min + ratio * (route_max - route_min);
}

void reset_timeline_view(UiState *state, const AppSession &session) {
  state->follow_latest = session.data_mode == SessionDataMode::Stream;
  reset_shared_range(state, session);
}

void draw_timeline_bar_contents(const AppSession &session, UiState *state, float width) {
  if (!session.route_data.has_time_range) {
    ImGui::Dummy(ImVec2(width, TIMELINE_BAR_HEIGHT));
    return;
  }

  const ImVec2 cursor = ImGui::GetCursorScreenPos();
  const ImVec2 size(width, TIMELINE_BAR_HEIGHT);
  const ImVec2 bar_min(cursor.x + 1.0f, cursor.y + 1.0f);
  const ImVec2 bar_max(cursor.x + size.x - 1.0f, cursor.y + size.y - 1.0f);
  const double route_min = state->route_x_min;
  const double route_max = state->route_x_max;
  const float vp_left = timeline_time_to_x(std::clamp(state->x_view_min, route_min, route_max),
                                           route_min, route_max, bar_min.x, bar_max.x);
  const float vp_right = timeline_time_to_x(std::clamp(state->x_view_max, route_min, route_max),
                                            route_min, route_max, bar_min.x, bar_max.x);

  ImGui::InvisibleButton("##timeline_button", size);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  const bool double_clicked = hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  ImDrawList *draw_list = ImGui::GetWindowDrawList();

  draw_list->AddRectFilled(bar_min, bar_max, timeline_entry_color(TimelineEntry::Type::None, 0.2f));
  if (session.route_data.timeline.empty()) {
    draw_list->AddRectFilled(ImVec2(vp_left, bar_min.y), ImVec2(vp_right, bar_max.y),
                             timeline_entry_color(TimelineEntry::Type::None, 1.0f));
  } else {
    for (const TimelineEntry &entry : session.route_data.timeline) {
      float x0 = timeline_time_to_x(entry.start_time, route_min, route_max, bar_min.x, bar_max.x);
      float x1 = timeline_time_to_x(entry.end_time, route_min, route_max, bar_min.x, bar_max.x);
      x1 = std::max(x1, x0 + 1.0f);
      draw_list->AddRectFilled(ImVec2(x0, bar_min.y), ImVec2(x1, bar_max.y),
                               timeline_entry_color(entry.type, 0.25f));
    }
    for (const TimelineEntry &entry : session.route_data.timeline) {
      float x0 = std::max(timeline_time_to_x(entry.start_time, route_min, route_max, bar_min.x, bar_max.x), vp_left);
      float x1 = std::min(std::max(timeline_time_to_x(entry.end_time, route_min, route_max, bar_min.x, bar_max.x), x0 + 1.0f), vp_right);
      if (x1 <= x0) {
        continue;
      }
      draw_list->AddRectFilled(ImVec2(x0, bar_min.y), ImVec2(x1, bar_max.y),
                               timeline_entry_color(entry.type, 1.0f));
    }
  }

  draw_list->AddLine(ImVec2(vp_left, bar_min.y), ImVec2(vp_left, bar_max.y), IM_COL32(60, 70, 80, 200), 1.0f);
  draw_list->AddLine(ImVec2(vp_right, bar_min.y), ImVec2(vp_right, bar_max.y), IM_COL32(60, 70, 80, 200), 1.0f);
  if (state->has_tracker_time) {
    const float cx = timeline_time_to_x(std::clamp(state->tracker_time, route_min, route_max),
                                        route_min, route_max, bar_min.x, bar_max.x);
    draw_list->AddLine(ImVec2(cx, bar_min.y), ImVec2(cx, bar_max.y), IM_COL32(220, 60, 50, 255), 1.5f);
  }
  draw_list->AddRect(bar_min, bar_max, IM_COL32(170, 178, 186, 255), 0.0f, 0, 1.0f);

  const float edge_grab = 4.0f;
  const float mouse_x = ImGui::GetIO().MousePos.x;
  const double mouse_time = timeline_x_to_time(mouse_x, route_min, route_max, bar_min.x, bar_max.x);
  if (double_clicked) {
    reset_timeline_view(state, session);
  } else if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    state->timeline_drag_anchor_time = mouse_time;
    state->timeline_drag_anchor_x_min = state->x_view_min;
    state->timeline_drag_anchor_x_max = state->x_view_max;
    if (std::abs(mouse_x - vp_left) <= edge_grab) {
      state->timeline_drag_mode = TimelineDragMode::ResizeLeft;
    } else if (std::abs(mouse_x - vp_right) <= edge_grab) {
      state->timeline_drag_mode = TimelineDragMode::ResizeRight;
    } else if (mouse_x >= vp_left && mouse_x <= vp_right) {
      state->timeline_drag_mode = TimelineDragMode::PanViewport;
    } else {
      state->timeline_drag_mode = TimelineDragMode::ScrubCursor;
      state->tracker_time = std::clamp(mouse_time, route_min, route_max);
      state->has_tracker_time = true;
    }
  }

  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    state->timeline_drag_mode = TimelineDragMode::None;
  } else if (active || state->timeline_drag_mode != TimelineDragMode::None) {
    switch (state->timeline_drag_mode) {
      case TimelineDragMode::ScrubCursor:
        state->tracker_time = std::clamp(mouse_time, route_min, route_max);
        state->has_tracker_time = true;
        break;
      case TimelineDragMode::PanViewport: {
        const double delta = mouse_time - state->timeline_drag_anchor_time;
        state->x_view_min = state->timeline_drag_anchor_x_min + delta;
        state->x_view_max = state->timeline_drag_anchor_x_max + delta;
        clamp_shared_range(state, session);
        break;
      }
      case TimelineDragMode::ResizeLeft:
        state->x_view_min = std::min(mouse_time, state->x_view_max - MIN_HORIZONTAL_ZOOM_SECONDS);
        clamp_shared_range(state, session);
        break;
      case TimelineDragMode::ResizeRight:
        state->x_view_max = std::max(mouse_time, state->x_view_min + MIN_HORIZONTAL_ZOOM_SECONDS);
        clamp_shared_range(state, session);
        break;
      case TimelineDragMode::None:
        break;
    }
  }

  if (hovered) {
    if (std::abs(mouse_x - vp_left) <= edge_grab || std::abs(mouse_x - vp_right) <= edge_grab) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    } else if (mouse_x >= vp_left && mouse_x <= vp_right) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    ImGui::BeginTooltip();
    ImGui::Text("t=%.1fs — %s", mouse_time, timeline_entry_label(timeline_type_at_time(session.route_data.timeline, mouse_time)));
    ImGui::EndTooltip();
  }
}

void draw_status_bar(const AppSession &session, const UiMetrics &ui, UiState *state) {
  ImGui::SetNextWindowPos(ImVec2(ui.content_x, ui.status_bar_y));
  ImGui::SetNextWindowSize(ImVec2(ui.content_w, STATUS_BAR_HEIGHT));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, color_rgb(247, 248, 250));
  ImGui::PushStyleColor(ImGuiCol_Border, color_rgb(188, 193, 199));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin("##status_bar", nullptr, flags)) {
    draw_timeline_bar_contents(session, state, ui.content_w);
    const float row_y = TIMELINE_BAR_HEIGHT + 8.0f;
    ImGui::SetCursorPos(ImVec2(8.0f, row_y));
    ImGui::BeginDisabled(!session.route_data.has_time_range);
    ImGui::Checkbox("Loop", &state->playback_loop);
    ImGui::SameLine(0.0f, 10.0f);
    if (ImGui::Button(state->playback_playing ? "Pause" : "Play", ImVec2(56.0f, 0.0f))) {
      state->playback_playing = !state->playback_playing;
    }
    ImGui::SameLine(0.0f, 10.0f);
    if (ImGui::Button("Reset View", ImVec2(86.0f, 0.0f))) {
      reset_timeline_view(state, session);
    }
    const float controls_end_x = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x;
    ImGui::EndDisabled();

    const char *status_text = state->status_text.empty() ? "Ready" : state->status_text.c_str();
    const float status_x = controls_end_x + 16.0f;
    ImGui::SetCursorPos(ImVec2(status_x, row_y + 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(102, 110, 118));
    ImGui::TextUnformatted(status_text);
    ImGui::PopStyleColor();

  }
  ImGui::End();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar();
}

void draw_sidebar_resizer(const UiMetrics &ui, UiState *state) {
  constexpr float kHandleWidth = 14.0f;
  ImGui::SetNextWindowPos(ImVec2(ui.sidebar_width - kHandleWidth * 0.5f, ui.top_offset));
  ImGui::SetNextWindowSize(ImVec2(kHandleWidth, std::max(1.0f, ui.height - ui.top_offset)));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoBackground;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  if (ImGui::Begin("##sidebar_resizer", nullptr, flags)) {
    ImGui::InvisibleButton("##sidebar_resizer_button", ImVec2(kHandleWidth, std::max(1.0f, ui.height - ui.top_offset)));
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    if (ImGui::IsItemActive()) {
      const float max_width = std::min(SIDEBAR_MAX_WIDTH, ui.width * 0.6f);
      state->sidebar_width = std::clamp(ImGui::GetIO().MousePos.x, SIDEBAR_MIN_WIDTH, max_width);
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetWindowPos();
    draw_list->AddLine(ImVec2(origin.x + kHandleWidth * 0.5f, origin.y),
                       ImVec2(origin.x + kHandleWidth * 0.5f, origin.y + std::max(1.0f, ui.height - ui.top_offset)),
                       IM_COL32(194, 198, 204, 255));
  }
  ImGui::End();
  ImGui::PopStyleVar();
}
