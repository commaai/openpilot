#include "tools/loggy/shell/route_controls.h"

#include "tools/loggy/shell/remote_routes.h"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace loggy {
namespace {

std::string shell_quote(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

void open_external_url(std::string_view url) {
  if (url.empty()) return;
#ifdef __APPLE__
  const std::string command = "open " + shell_quote(url) + " >/dev/null 2>&1 &";
#else
  const std::string command = "xdg-open " + shell_quote(url) + " >/dev/null 2>&1 &";
#endif
  const int ignored = std::system(command.c_str());
  (void)ignored;
}

}  // namespace

RouteSelection current_route_selection(const Session &session) {
  RouteSelection selection = session.ingest_status().selection;
  if (selection.timestamp.empty()) selection = parse_route_selection(session.config.route_name);
  return selection;
}

void sync_route_popup_fields(const Session &session, RouteUiState &state) {
  const SessionConfig &config = session.config;
  std::snprintf(state.route_name_buffer.data(), state.route_name_buffer.size(), "%s", config.route_name.c_str());
  const RouteSelection selection = current_route_selection(session);
  const std::string slice = selection.timestamp.empty() ? std::string("0:") : route_selection_display_slice(selection);
  std::snprintf(state.route_slice_buffer.data(), state.route_slice_buffer.size(), "%s", slice.c_str());
  state.route_selector_index = std::clamp(static_cast<int>(selection.selector), 0, 2);
}

void request_route_popup(const Session &session, RouteUiState &state) {
  sync_route_popup_fields(session, state);
  state.open_popup = true;
}

bool restart_route_from_popup(Session &session, RouteUiState &state, std::string route_name, std::string status_prefix) {
  std::string error;
  if (!session.restart_route(std::move(route_name), error)) {
    state.status = error.empty() ? "Route restart failed" : error;
    return false;
  }
  state.status = std::move(status_prefix);
  sync_route_popup_fields(session, state);
  return true;
}

void draw_route_popup(Session &session, RouteUiState &state, bool close_requested) {
  if (state.open_popup) {
    ImGui::OpenPopup("Route");
    state.open_popup = false;
  }

  if (!ImGui::BeginPopupModal("Route", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;

  const RouteIngestStatus ingest = session.ingest_status();
  const RouteSelection selection = current_route_selection(session);
  const std::string full_spec = route_selection_full_spec(selection);
  const std::string useradmin_url = route_useradmin_url(selection);
  const std::string connect_url = route_connect_url(selection);

  ImGui::TextDisabled("%s  |  %zu/%zu segments  %.2fs first  %.2fs total",
                      route_ingest_state_label(ingest.state),
                      ingest.segments_loaded,
                      ingest.segments_resolved,
                      ingest.first_segment_seconds,
                      ingest.total_seconds);
  if (!ingest.error.empty()) ImGui::TextDisabled("%s", ingest.error.c_str());
  if (!session.car_fingerprint.empty()) {
    ImGui::TextDisabled("Car: %s", session.car_fingerprint.c_str());
  }
  if (!session.auto_dbc_name.empty() || !session.active_dbc_name.empty()) {
    std::string dbc_line = session.active_dbc_name.empty() ? "--" : session.active_dbc_name;
    if (!session.manual_dbc_name.empty()) dbc_line += " override";
    if (!session.auto_dbc_name.empty()) dbc_line += " (auto " + session.auto_dbc_name + ")";
    ImGui::TextDisabled("DBC: %s", dbc_line.c_str());
  }
  if (!session.dbc_status.empty()) ImGui::TextDisabled("%s", session.dbc_status.c_str());

  ImGui::SetNextItemWidth(420.0f);
  ImGui::InputText("Route", state.route_name_buffer.data(), state.route_name_buffer.size());
  if (ImGui::Button("Open")) {
    restart_route_from_popup(session, state, state.route_name_buffer.data(), "Opened route");
  }
  ImGui::SameLine();
  if (ImGui::Button("Browse...")) {
    open_remote_route_browser();
  }
  ImGui::SameLine();
  if (ImGui::Button("Copy")) {
    ImGui::SetClipboardText(full_spec.empty() ? session.config.route_name.c_str() : full_spec.c_str());
    state.status = "Copied route";
  }
  ImGui::SameLine();
  if (ImGui::Button("Copy Onebox")) {
    const std::string onebox = selection.dongle_id.empty() || selection.timestamp.empty()
                             ? std::string()
                             : selection.dongle_id + "|" + selection.timestamp;
    ImGui::SetClipboardText(onebox.empty() ? session.config.route_name.c_str() : onebox.c_str());
    state.status = "Copied onebox";
  }

  if (!full_spec.empty()) {
    ImGui::TextDisabled("Spec: %s", full_spec.c_str());
  }

  ImGui::Separator();
  ImGui::SetNextItemWidth(120.0f);
  ImGui::InputText("Slice", state.route_slice_buffer.data(), state.route_slice_buffer.size());
  ImGui::SameLine();
  if (ImGui::Button("Apply Slice")) {
    const std::optional<loggy::RouteSliceSpec> slice = parse_route_slice_spec(state.route_slice_buffer.data());
    if (!slice.has_value()) {
      state.status = "Slice must be N, N:, or N:M";
    } else {
      RouteSelection next = selection;
      next.begin_segment = slice->first;
      next.end_segment = slice->second;
      next.slice_explicit = true;
      restart_route_from_popup(session, state, route_selection_full_spec(next), "Updated route slice");
    }
  }

  const char *selector_labels[] = {"auto", "rlog", "qlog"};
  ImGui::SetNextItemWidth(120.0f);
  if (ImGui::Combo("Selector", &state.route_selector_index, selector_labels, 3)) {
    RouteSelection next = selection;
    next.selector = static_cast<LogSelector>(std::clamp(state.route_selector_index, 0, 2));
    next.selector_explicit = true;
    restart_route_from_popup(session, state, route_selection_full_spec(next), "Updated log selector");
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", log_selector_description(static_cast<LogSelector>(
                               std::clamp(state.route_selector_index, 0, 2))));

  ImGui::Separator();
  if (useradmin_url.empty()) ImGui::BeginDisabled();
  if (ImGui::Button("Useradmin")) {
    open_external_url(useradmin_url);
    state.status = "Opened useradmin";
  }
  if (useradmin_url.empty()) ImGui::EndDisabled();
  ImGui::SameLine();
  if (connect_url.empty()) ImGui::BeginDisabled();
  if (ImGui::Button("Connect")) {
    open_external_url(connect_url);
    state.status = "Opened connect";
  }
  if (connect_url.empty()) ImGui::EndDisabled();

  if (!state.status.empty()) {
    ImGui::TextDisabled("%s", state.status.c_str());
  }

  ImGui::Separator();
  if (ImGui::Button("Close") || close_requested) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
}

}  // namespace loggy
