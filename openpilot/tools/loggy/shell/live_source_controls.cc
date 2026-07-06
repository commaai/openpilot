#include "tools/loggy/shell/live_source_controls.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>

namespace loggy {
namespace {

LiveSourceKind live_source_kind_from_index(int index) {
  switch (index) {
    case 1: return LiveSourceKind::CerealRemote;
    case 2: return LiveSourceKind::DeviceBridge;
    case 3: return LiveSourceKind::PandaUsb;
    case 4: return LiveSourceKind::SocketCan;
    case 0:
    default: return LiveSourceKind::CerealLocal;
  }
}

const char *live_source_input_label(LiveSourceKind kind) {
  switch (kind) {
    case LiveSourceKind::PandaUsb: return "Serial";
    case LiveSourceKind::SocketCan: return "Device";
    default: return "Address";
  }
}

const char *live_source_field_text(const SessionConfig &config) {
  if (config.stream_source_kind == LiveSourceKind::CerealLocal && config.stream_address.empty()) return "127.0.0.1";
  return config.stream_address.c_str();
}

template <size_t N>
int speed_index(const std::array<uint16_t, N> &speeds, uint16_t speed) {
  const auto it = std::find(speeds.begin(), speeds.end(), speed);
  return it == speeds.end() ? 0 : static_cast<int>(std::distance(speeds.begin(), it));
}

void sync_panda_source_fields(std::array<int, kPandaBusCount> &panda_can_speed_index,
                              std::array<int, kPandaBusCount> &panda_data_speed_index,
                              std::array<bool, kPandaBusCount> &panda_can_fd,
                              const std::array<PandaBusConfig, kPandaBusCount> &buses) {
  for (size_t bus = 0; bus < buses.size(); ++bus) {
    const PandaBusConfig config = normalize_live_panda_bus_config(buses[bus]);
    panda_can_speed_index[bus] = speed_index(kPandaCanSpeedsKbps, config.can_speed_kbps);
    panda_data_speed_index[bus] = speed_index(kPandaDataSpeedsKbps, config.data_speed_kbps);
    panda_can_fd[bus] = config.can_fd;
  }
}

std::array<PandaBusConfig, kPandaBusCount> panda_source_fields(
  const std::array<int, kPandaBusCount> &panda_can_speed_index,
  const std::array<int, kPandaBusCount> &panda_data_speed_index,
  const std::array<bool, kPandaBusCount> &panda_can_fd) {
  std::array<PandaBusConfig, kPandaBusCount> out{};
  for (size_t bus = 0; bus < out.size(); ++bus) {
    out[bus].can_speed_kbps = kPandaCanSpeedsKbps[static_cast<size_t>(
      std::clamp(panda_can_speed_index[bus], 0, static_cast<int>(kPandaCanSpeedsKbps.size() - 1)))];
    out[bus].data_speed_kbps = kPandaDataSpeedsKbps[static_cast<size_t>(
      std::clamp(panda_data_speed_index[bus], 0, static_cast<int>(kPandaDataSpeedsKbps.size() - 1)))];
    out[bus].can_fd = panda_can_fd[bus];
  }
  return out;
}

void speed_text(char *buffer, size_t size, uint16_t speed_kbps) {
  std::snprintf(buffer, size, "%u", static_cast<unsigned int>(speed_kbps));
}

template <size_t N>
bool draw_speed_combo(const char *label, int &index, const std::array<uint16_t, N> &speeds) {
  if (speeds.empty()) return false;
  index = std::clamp(index, 0, static_cast<int>(speeds.size() - 1));
  char preview[16];
  speed_text(preview, sizeof(preview), speeds[static_cast<size_t>(index)]);
  bool changed = false;
  ImGui::SetNextItemWidth(86.0f);
  if (ImGui::BeginCombo(label, preview)) {
    for (size_t i = 0; i < speeds.size(); ++i) {
      char item[16];
      speed_text(item, sizeof(item), speeds[i]);
      const bool selected = index == static_cast<int>(i);
      if (ImGui::Selectable(item, selected)) {
        index = static_cast<int>(i);
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  return changed;
}

void draw_panda_bus_config_controls(std::array<int, kPandaBusCount> &panda_can_speed_index,
                                    std::array<int, kPandaBusCount> &panda_data_speed_index,
                                    std::array<bool, kPandaBusCount> &panda_can_fd) {
  ImGui::SeparatorText("Panda buses");
  for (size_t bus = 0; bus < kPandaBusCount; ++bus) {
    ImGui::PushID(static_cast<int>(bus));
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Bus %zu", bus);
    ImGui::SameLine(70.0f);
    draw_speed_combo("CAN##speed", panda_can_speed_index[bus], kPandaCanSpeedsKbps);
    ImGui::SameLine();
    ImGui::Checkbox("FD", &panda_can_fd[bus]);
    ImGui::SameLine();
    if (!panda_can_fd[bus]) ImGui::BeginDisabled();
    draw_speed_combo("Data##speed", panda_data_speed_index[bus], kPandaDataSpeedsKbps);
    if (!panda_can_fd[bus]) ImGui::EndDisabled();
    ImGui::PopID();
  }
}

}  // namespace

int live_source_kind_to_index(LiveSourceKind kind) {
  switch (kind) {
    case LiveSourceKind::CerealRemote: return 1;
    case LiveSourceKind::DeviceBridge: return 2;
    case LiveSourceKind::PandaUsb: return 3;
    case LiveSourceKind::SocketCan: return 4;
    case LiveSourceKind::CerealLocal:
    default: return 0;
  }
}

void sync_live_source_fields(const Session &session, LiveSourceUiState &state) {
  const SessionConfig &config = session.config;
  std::snprintf(state.address_buffer.data(), state.address_buffer.size(), "%s", live_source_field_text(config));
  state.source_kind_index = live_source_kind_to_index(config.stream_source_kind);
  sync_panda_source_fields(state.panda_can_speed_index, state.panda_data_speed_index, state.panda_can_fd,
                           config.stream_panda_buses);
  state.buffer_seconds = std::max(1.0, config.stream_buffer_seconds);
}

void request_live_source_popup(const Session &session, LiveSourceUiState &state) {
  sync_live_source_fields(session, state);
  state.open_popup = true;
}

void draw_live_source_popup(Session &session, LiveSourceUiState &state) {
  if (state.open_popup) {
    ImGui::OpenPopup("Live Source");
    state.open_popup = false;
  }

  if (!ImGui::BeginPopupModal("Live Source", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
  const bool close_requested = ImGui::IsKeyPressed(ImGuiKey_Escape);

  const char *source_labels[] = {"Local MSGQ", "Remote ZMQ", "Device Bridge", "Panda USB", "SocketCAN"};
  ImGui::SetNextItemWidth(160.0f);
  ImGui::Combo("Source", &state.source_kind_index, source_labels, 5);
  const LiveSourceKind source_kind = live_source_kind_from_index(state.source_kind_index);

  if (source_kind == LiveSourceKind::CerealLocal) ImGui::BeginDisabled();
  ImGui::SetNextItemWidth(260.0f);
  ImGui::InputText(live_source_input_label(source_kind), state.address_buffer.data(), state.address_buffer.size());
  if (source_kind == LiveSourceKind::CerealLocal) ImGui::EndDisabled();
  if (source_kind == LiveSourceKind::CerealLocal) {
    ImGui::SameLine();
    ImGui::TextDisabled("127.0.0.1");
  } else if (source_kind == LiveSourceKind::PandaUsb) {
    ImGui::SameLine();
    ImGui::TextDisabled(live_panda_available() ? "Panda detected" : "No Panda detected");
  } else if (source_kind == LiveSourceKind::SocketCan && !live_socketcan_available()) {
    ImGui::SameLine();
    ImGui::TextDisabled("SocketCAN unavailable");
  }
  if (source_kind == LiveSourceKind::PandaUsb) {
    draw_panda_bus_config_controls(state.panda_can_speed_index, state.panda_data_speed_index, state.panda_can_fd);
  }

  ImGui::SetNextItemWidth(120.0f);
  ImGui::InputDouble("Buffer seconds", &state.buffer_seconds, 1.0, 5.0, "%.0f");
  state.buffer_seconds = std::max(1.0, state.buffer_seconds);

  const bool route_mode = !session.config.route_name.empty();
  if (route_mode) {
    ImGui::TextDisabled("Live source is unavailable while a route is open.");
  }
  if (route_mode) ImGui::BeginDisabled();
  if (ImGui::Button(session.config.stream ? "Reconnect" : "Open")) {
    std::string error;
    const std::string target = source_kind == LiveSourceKind::CerealLocal ? "127.0.0.1" : state.address_buffer.data();
    LiveSourceConfig source;
    source.kind = source_kind;
    source.address = target;
    source.panda_buses =
      panda_source_fields(state.panda_can_speed_index, state.panda_data_speed_index, state.panda_can_fd);
    source.buffer_seconds = state.buffer_seconds;
    if (session.restart_live(std::move(source), error)) {
      sync_live_source_fields(session, state);
      state.status = "Live source opened";
      ImGui::CloseCurrentPopup();
    } else {
      state.status = error;
    }
  }
  if (route_mode) ImGui::EndDisabled();

  if (session.config.stream) {
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
      session.stop_live();
      state.status = "Live source stopped";
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel") || close_requested) {
    ImGui::CloseCurrentPopup();
  }

  if (!state.status.empty()) {
    ImGui::TextDisabled("%s", state.status.c_str());
  }

  ImGui::EndPopup();
}

}  // namespace loggy
