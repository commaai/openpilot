#include "tools/jotpluggler/app_internal.h"
#include "tools/jotpluggler/app_socketcan.h"
#include "tools/cabana/panda.h"
#include "system/hardware/hw.h"

#include <cstdio>
#include <fstream>
#include <unistd.h>

namespace {

enum class ModalAction {
  None,
  Primary,
  Secondary,
};

struct FindSignalMatch {
  const std::string *path = nullptr;
  int score = 0;
};

template <size_t N>
void copy_string_to_buffer(const std::string &value, std::array<char, N> *buffer) {
  std::snprintf(buffer->data(), buffer->size(), "%s", value.c_str());
}

constexpr int kPandaCanSpeeds[] = {10, 20, 50, 100, 125, 250, 500, 1000};
constexpr int kPandaDataSpeeds[] = {10, 20, 50, 100, 125, 250, 500, 1000, 2000, 5000};

std::vector<std::string> list_panda_serials() {
  try {
    return Panda::list();
  } catch (...) {
    return {};
  }
}

std::string stream_source_target_label(const StreamSourceConfig &source) {
  switch (source.kind) {
    case StreamSourceKind::CerealRemote:
      return source.address.empty() ? std::string("127.0.0.1") : source.address;
    case StreamSourceKind::Panda:
      return source.panda.serial.empty() ? std::string("auto") : source.panda.serial;
    case StreamSourceKind::SocketCan:
      return source.socketcan.device.empty() ? std::string("can0") : source.socketcan.device;
    case StreamSourceKind::CerealLocal:
    default:
      return "127.0.0.1";
  }
}

StreamSourceConfig stream_source_config_from_ui(const UiState &state) {
  StreamSourceConfig source;
  source.kind = state.stream_source_kind;
  source.address = trim_copy(state.stream_address_buffer.data());
  source.panda.serial = trim_copy(state.panda_serial_buffer.data());
  source.socketcan.device = trim_copy(state.socketcan_device_buffer.data());
  for (size_t i = 0; i < source.panda.buses.size(); ++i) {
    source.panda.buses[i].can_speed_kbps = state.panda_can_speed_kbps[i];
    source.panda.buses[i].data_speed_kbps = state.panda_data_speed_kbps[i];
    source.panda.buses[i].can_fd = state.panda_can_fd[i];
  }
  if (source.kind == StreamSourceKind::CerealLocal) {
    source.address = "127.0.0.1";
  } else if (source.kind == StreamSourceKind::SocketCan && source.socketcan.device.empty()) {
    source.socketcan.device = "can0";
  }
  return source;
}

void open_queued_popup(bool &flag, const char *name) {
  if (flag) {
    ImGui::OpenPopup(name);
    flag = false;
  }
}

ModalAction draw_modal_action_row(const char *primary_label,
                                  const char *secondary_label = "Cancel",
                                  float width = 120.0f) {
  if (ImGui::Button(primary_label, ImVec2(width, 0.0f))) {
    return ModalAction::Primary;
  }
  ImGui::SameLine();
  if (ImGui::Button(secondary_label, ImVec2(width, 0.0f))) {
    return ModalAction::Secondary;
  }
  return ModalAction::None;
}

std::vector<FindSignalMatch> find_signal_matches(const AppSession &session, std::string_view query) {
  std::vector<FindSignalMatch> matches;
  if (query.empty()) {
    return matches;
  }
  const std::string needle = lowercase(query);
  for (const std::string &path : session.route_data.paths) {
    const std::string hay = lowercase(path);
    const size_t pos = hay.find(needle);
    if (pos == std::string::npos) {
      continue;
    }
    const size_t slash = path.find_last_of('/');
    const std::string_view label = slash == std::string::npos ? std::string_view(path) : std::string_view(path).substr(slash + 1);
    int score = static_cast<int>(pos * 8 + path.size());
    if (lowercase(label) == needle) score -= 60;
    if (hay.rfind(needle, 0) == 0) score -= 30;
    matches.push_back({.path = &path, .score = score});
  }
  std::sort(matches.begin(), matches.end(), [](const FindSignalMatch &a, const FindSignalMatch &b) {
    return std::tie(a.score, *a.path) < std::tie(b.score, *b.path);
  });
  if (matches.size() > 200) {
    matches.resize(200);
  }
  return matches;
}

bool open_find_signal_result(AppSession *session, UiState *state, const std::string &path) {
  for (const CabanaMessageSummary &message : session->cabana_messages) {
    const auto signal_it = std::find_if(message.signals.begin(), message.signals.end(), [&](const CabanaSignalSummary &signal) {
      return signal.path == path;
    });
    if (signal_it != message.signals.end()) {
      state->view_mode = AppViewMode::Cabana;
      state->cabana.selected_message_root = message.root_path;
      state->cabana.selected_signal_path = path;
      state->cabana.has_bit_selection = false;
      state->cabana.similar_bit_matches.clear();
      state->status_text = "Opened signal in Cabana";
      return true;
    }
  }

  state->view_mode = AppViewMode::Plot;
  state->selected_browser_paths = {path};
  state->selected_browser_path = path;
  state->browser_selection_anchor = path;
  state->status_text = "Selected signal " + path;
  return true;
}

void draw_open_route_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Open Route", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Load a route into the current layout.");
  ImGui::Separator();
  ImGui::InputText("Route", state->route_buffer.data(), state->route_buffer.size());
  ImGui::InputText("Data Dir", state->data_dir_buffer.data(), state->data_dir_buffer.size());
  ImGui::Spacing();
  switch (draw_modal_action_row("Load")) {
    case ModalAction::Primary:
      reload_session(session, state, std::string(state->route_buffer.data()), std::string(state->data_dir_buffer.data()));
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::Secondary:
      sync_route_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_stream_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Live Stream", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  static std::vector<std::string> panda_serials;
  static std::vector<std::string> socketcan_devices;

  ImGui::TextUnformatted("Connect to a live source.");
  ImGui::Separator();
  if (ImGui::RadioButton("Local (MSGQ)", state->stream_source_kind == StreamSourceKind::CerealLocal)) {
    state->stream_source_kind = StreamSourceKind::CerealLocal;
  }
  if (ImGui::RadioButton("Remote (ZMQ)", state->stream_source_kind == StreamSourceKind::CerealRemote)) {
    state->stream_source_kind = StreamSourceKind::CerealRemote;
  }
  if (ImGui::RadioButton("Panda", state->stream_source_kind == StreamSourceKind::Panda)) {
    state->stream_source_kind = StreamSourceKind::Panda;
    if (panda_serials.empty()) panda_serials = list_panda_serials();
  }
#ifdef __linux__
  if (ImGui::RadioButton("SocketCAN", state->stream_source_kind == StreamSourceKind::SocketCan)) {
    state->stream_source_kind = StreamSourceKind::SocketCan;
    if (socketcan_devices.empty()) socketcan_devices = list_socketcan_devices();
  }
#else
  ImGui::BeginDisabled(true);
  ImGui::RadioButton("SocketCAN", false);
  ImGui::EndDisabled();
#endif

  if (state->stream_source_kind == StreamSourceKind::CerealRemote) {
    ImGui::InputText("Address", state->stream_address_buffer.data(), state->stream_address_buffer.size());
  } else if (state->stream_source_kind == StreamSourceKind::Panda) {
    if (ImGui::Button("Refresh Pandas")) {
      panda_serials = list_panda_serials();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu found", panda_serials.size());
    if (ImGui::BeginCombo("Serial", state->panda_serial_buffer.data()[0] == '\0' ? "auto" : state->panda_serial_buffer.data())) {
      const bool auto_selected = state->panda_serial_buffer.data()[0] == '\0';
      if (ImGui::Selectable("auto", auto_selected)) {
        state->panda_serial_buffer[0] = '\0';
      }
      for (const std::string &serial : panda_serials) {
        const bool selected = serial == state->panda_serial_buffer.data();
        if (ImGui::Selectable(serial.c_str(), selected)) {
          copy_string_to_buffer(serial, &state->panda_serial_buffer);
        }
      }
      ImGui::EndCombo();
    }
    for (int bus = 0; bus < 3; ++bus) {
      ImGui::PushID(bus);
      ImGui::SeparatorText((std::string("Bus ") + std::to_string(bus)).c_str());
      if (ImGui::BeginCombo("CAN Speed", (std::to_string(state->panda_can_speed_kbps[bus]) + " kbps").c_str())) {
        for (const int speed : kPandaCanSpeeds) {
          const bool selected = speed == state->panda_can_speed_kbps[bus];
          if (ImGui::Selectable((std::to_string(speed) + " kbps").c_str(), selected)) {
            state->panda_can_speed_kbps[bus] = speed;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::Checkbox("CAN-FD", &state->panda_can_fd[bus]);
      ImGui::BeginDisabled(!state->panda_can_fd[bus]);
      if (ImGui::BeginCombo("Data Speed", (std::to_string(state->panda_data_speed_kbps[bus]) + " kbps").c_str())) {
        for (const int speed : kPandaDataSpeeds) {
          const bool selected = speed == state->panda_data_speed_kbps[bus];
          if (ImGui::Selectable((std::to_string(speed) + " kbps").c_str(), selected)) {
            state->panda_data_speed_kbps[bus] = speed;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::EndDisabled();
      ImGui::PopID();
    }
  } else if (state->stream_source_kind == StreamSourceKind::SocketCan) {
#ifdef __linux__
    if (ImGui::Button("Refresh Devices")) {
      socketcan_devices = list_socketcan_devices();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu found", socketcan_devices.size());
    if (ImGui::BeginCombo("Device", state->socketcan_device_buffer.data()[0] == '\0' ? "can0" : state->socketcan_device_buffer.data())) {
      for (const std::string &device : socketcan_devices) {
        const bool selected = device == state->socketcan_device_buffer.data();
        if (ImGui::Selectable(device.c_str(), selected)) {
          copy_string_to_buffer(device, &state->socketcan_device_buffer);
        }
      }
      ImGui::EndCombo();
    }
#else
    ImGui::TextDisabled("SocketCAN is only available on Linux.");
#endif
  }
  ImGui::InputDouble("Buffer (seconds)", &state->stream_buffer_seconds, 0.0, 0.0, "%.0f");
  ImGui::Spacing();
  switch (draw_modal_action_row("Connect")) {
    case ModalAction::Primary: {
      const StreamSourceConfig source = stream_source_config_from_ui(*state);
      if (start_stream_session(session, state, source, state->stream_buffer_seconds, false)) {
        ImGui::CloseCurrentPopup();
      }
      break;
    }
    case ModalAction::Secondary:
      sync_stream_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_load_layout_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Load Layout", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Load a JotPlugger JSON layout.");
  ImGui::Separator();
  ImGui::InputText("Layout", state->load_layout_buffer.data(), state->load_layout_buffer.size());
  ImGui::Spacing();
  switch (draw_modal_action_row("Load")) {
    case ModalAction::Primary:
      if (reload_layout(session, state, std::string(state->load_layout_buffer.data()))) {
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      sync_layout_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_save_layout_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Save Layout", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Save the current workspace as a JotPlugger JSON layout.");
  ImGui::Separator();
  ImGui::InputText("Layout", state->save_layout_buffer.data(), state->save_layout_buffer.size());
  ImGui::Spacing();
  switch (draw_modal_action_row("Save")) {
    case ModalAction::Primary:
      if (save_layout(session, state, std::string(state->save_layout_buffer.data()))) {
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      sync_layout_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_preferences_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Preferences", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  if (session->map_data) {
    const MapCacheStats map_cache = session->map_data->cacheStats();
    const MapCacheStats download_cache = directory_cache_stats(Path::download_cache_root());
    ImGui::TextUnformatted("Map");
    ImGui::Separator();
    ImGui::Text("Map cache: %s in %zu file%s",
                format_cache_bytes(map_cache.bytes).c_str(),
                map_cache.files,
                map_cache.files == 1 ? "" : "s");
    if (ImGui::Button("Clear Map Cache", ImVec2(120.0f, 0.0f))) {
      session->map_data->clearCache();
      state->status_text = "Cleared map cache";
    }
    ImGui::Spacing();
    ImGui::TextUnformatted("comma Download Cache");
    ImGui::Separator();
    ImGui::Text("Download cache: %s in %zu file%s",
                format_cache_bytes(download_cache.bytes).c_str(),
                download_cache.files,
                download_cache.files == 1 ? "" : "s");
    ImGui::TextDisabled("%s", Path::download_cache_root().c_str());
    ImGui::Spacing();
  }
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_find_signal_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Find Signal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Search decoded signals across the loaded route.");
  ImGui::Separator();
  ImGui::SetNextItemWidth(560.0f);
  ImGui::InputTextWithHint("##find_signal_query", "Search signal path or name...", state->find_signal_buffer.data(), state->find_signal_buffer.size());
  if (ImGui::IsWindowAppearing()) {
    ImGui::SetKeyboardFocusHere(-1);
  }
  const std::vector<FindSignalMatch> matches = find_signal_matches(*session, state->find_signal_buffer.data());
  ImGui::Spacing();
  ImGui::TextDisabled("%zu match%s", matches.size(), matches.size() == 1 ? "" : "es");
  if (ImGui::BeginChild("##find_signal_results", ImVec2(760.0f, 360.0f), true)) {
    for (const FindSignalMatch &match : matches) {
      const std::string &path = *match.path;
      const size_t slash = path.find_last_of('/');
      const std::string_view label = slash == std::string::npos ? std::string_view(path) : std::string_view(path).substr(slash + 1);
      if (ImGui::Selectable((std::string(label) + "##" + path).c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
        if (open_find_signal_result(session, state, path)) {
          ImGui::CloseCurrentPopup();
        }
      }
      ImGui::SameLine(280.0f);
      ImGui::TextDisabled("%s", path.c_str());
    }
  }
  ImGui::EndChild();
  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

const fs::path &repo_root() {
  static const fs::path root = []() {
#ifdef JOTP_REPO_ROOT
    return fs::path(JOTP_REPO_ROOT);
#else
    return fs::current_path();
#endif
  }();
  return root;
}

std::string default_dbc_template() {
  return "VERSION \"\"\n\nNS_ :\nBS_:\nBU_: XXX\n";
}

std::string active_dbc_name(const AppSession &session) {
  return !session.dbc_override.empty() ? session.dbc_override : session.route_data.dbc_name;
}

fs::path generated_dbc_dir() {
  return repo_root() / "tools" / "jotpluggler" / "generated_dbcs";
}

fs::path resolve_dbc_editor_source(const std::string &dbc_name) {
  for (const fs::path &candidate : {
         repo_root() / "opendbc" / "dbc" / (dbc_name + ".dbc"),
         generated_dbc_dir() / (dbc_name + ".dbc"),
       }) {
    if (fs::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

std::string read_text_file(const fs::path &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void load_dbc_editor_state(const AppSession &session, UiState *state) {
  DbcEditorState &editor = state->dbc_editor;
  const std::string dbc_name = active_dbc_name(session);
  editor.source_name = dbc_name.empty() ? "untitled" : dbc_name;
  editor.source_path.clear();
  if (dbc_name.empty()) {
    editor.save_name = "custom_can";
    editor.text = default_dbc_template();
  } else {
    const fs::path path = resolve_dbc_editor_source(dbc_name);
    editor.source_path = path.string();
    editor.text = path.empty() ? default_dbc_template() : read_text_file(path);
    editor.save_name = path.string().find("/generated_dbcs/") != std::string::npos ? dbc_name : dbc_name + "_edited";
  }
  editor.loaded = true;
}

bool ensure_dbc_editor_loaded(const AppSession &session, UiState *state) {
  if (!state->dbc_editor.loaded) {
    try {
      load_dbc_editor_state(session, state);
    } catch (const std::exception &err) {
      state->error_text = err.what();
      state->open_error_popup = true;
      return false;
    }
  }
  return true;
}

std::string multiplex_indicator_for_signal(const CabanaSignalEditorState &signal) {
  if (signal.type == static_cast<int>(dbc::Signal::Type::Multiplexor)) {
    return "M ";
  }
  if (signal.type == static_cast<int>(dbc::Signal::Type::Multiplexed)) {
    return "m" + std::to_string(signal.multiplex_value) + " ";
  }
  return {};
}

std::string build_signal_definition_line(const CabanaSignalEditorState &signal) {
  return " SG_ " + signal.signal_name + " " + multiplex_indicator_for_signal(signal) + ": "
       + std::to_string(signal.start_bit) + "|" + std::to_string(signal.size) + "@"
       + std::string(1, signal.is_little_endian ? '1' : '0')
       + std::string(1, signal.is_signed ? '-' : '+')
       + " (" + util::string_format("%.15g", signal.factor) + "," + util::string_format("%.15g", signal.offset) + ")"
       + " [" + util::string_format("%.15g", signal.min) + "|" + util::string_format("%.15g", signal.max) + "]"
       + " \"" + signal.unit + "\" " + (signal.receiver_name.empty() ? "XXX" : signal.receiver_name);
}

bool replace_signal_line(std::string *text,
                         uint32_t address,
                         const std::string &signal_name,
                         const std::string &replacement) {
  std::istringstream in(*text);
  std::string line;
  std::string out;
  bool in_message = false;
  bool replaced = false;
  while (std::getline(in, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.rfind("BO_ ", 0) == 0) {
      char *end = nullptr;
      const long parsed = std::strtol(trimmed.c_str() + 4, &end, 10);
      in_message = end != nullptr && parsed == static_cast<long>(address);
    } else if (in_message && trimmed.rfind("SG_ ", 0) == 0) {
      const size_t name_start = 4;
      const size_t name_end = trimmed.find(' ', name_start);
      const std::string current_name = name_end == std::string::npos ? trimmed.substr(name_start) : trimmed.substr(name_start, name_end - name_start);
      if (current_name == signal_name) {
        out += replacement + "\n";
        replaced = true;
        continue;
      }
    }
    out += line + "\n";
  }
  if (replaced) {
    *text = std::move(out);
  }
  return replaced;
}

bool insert_signal_line(std::string *text,
                        uint32_t address,
                        const std::string &line_to_insert) {
  std::istringstream in(*text);
  std::string line;
  std::string out;
  bool in_message = false;
  bool inserted = false;
  while (std::getline(in, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.rfind("BO_ ", 0) == 0) {
      if (in_message && !inserted) {
        out += line_to_insert + "\n";
        inserted = true;
      }
      char *end = nullptr;
      const long parsed = std::strtol(trimmed.c_str() + 4, &end, 10);
      in_message = end != nullptr && parsed == static_cast<long>(address);
      out += line + "\n";
      continue;
    }
    if (in_message && !inserted) {
      const bool starts_new_top_level = !trimmed.empty() && trimmed.rfind("SG_ ", 0) != 0;
      if (starts_new_top_level) {
        out += line_to_insert + "\n";
        inserted = true;
      }
    }
    out += line + "\n";
  }
  if (in_message && !inserted) {
    out += line_to_insert + "\n";
    inserted = true;
  }
  if (inserted) {
    *text = std::move(out);
  }
  return inserted;
}

bool remove_signal_line(std::string *text,
                        uint32_t address,
                        const std::string &signal_name) {
  std::istringstream in(*text);
  std::string line;
  std::string out;
  bool in_message = false;
  bool removed = false;
  while (std::getline(in, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.rfind("BO_ ", 0) == 0) {
      char *end = nullptr;
      const long parsed = std::strtol(trimmed.c_str() + 4, &end, 10);
      in_message = end != nullptr && parsed == static_cast<long>(address);
    } else if (in_message && trimmed.rfind("SG_ ", 0) == 0) {
      const size_t name_start = 4;
      const size_t name_end = trimmed.find(' ', name_start);
      const std::string current_name = name_end == std::string::npos ? trimmed.substr(name_start) : trimmed.substr(name_start, name_end - name_start);
      if (current_name == signal_name) {
        removed = true;
        continue;
      }
    }
    out += line + "\n";
  }
  if (removed) {
    *text = std::move(out);
  }
  return removed;
}

bool save_dbc_editor_contents(AppSession *session, UiState *state) {
  DbcEditorState &editor = state->dbc_editor;
  editor.save_name = trim_copy(editor.save_name);
  if (editor.save_name.empty()) {
    state->error_text = "DBC name cannot be empty";
    state->open_error_popup = true;
    return false;
  }
  if (!editor.source_path.empty()
      && editor.source_path.find("/opendbc/dbc/") != std::string::npos
      && editor.save_name == editor.source_name) {
    state->error_text = "Save edited opendbc files under a new name";
    state->open_error_popup = true;
    return false;
  }
  try {
    dbc::Database::fromContent(editor.text, editor.save_name + ".dbc");
    fs::create_directories(generated_dbc_dir());
    const fs::path output = generated_dbc_dir() / (editor.save_name + ".dbc");
    std::ofstream out(output);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open " + output.string());
    }
    out << editor.text;
    if (!out.good()) {
      throw std::runtime_error("Failed while writing " + output.string());
    }
    apply_dbc_override_change(session, state, editor.save_name);
    editor.source_name = editor.save_name;
    editor.source_path = output.string();
    editor.loaded = false;
    state->status_text = "Saved DBC " + editor.save_name;
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    return false;
  }
}

bool apply_cabana_signal_edit_impl(AppSession *session, UiState *state) {
  if (!ensure_dbc_editor_loaded(*session, state)) {
    return false;
  }
  CabanaSignalEditorState &signal = state->cabana_signal_editor;
  if (trim_copy(signal.signal_name).empty()) {
    state->error_text = "Signal name cannot be empty";
    state->open_error_popup = true;
    return false;
  }
  signal.signal_name = trim_copy(signal.signal_name);
  signal.receiver_name = trim_copy(signal.receiver_name);
  if (signal.size <= 0) {
    state->error_text = "Signal size must be positive";
    state->open_error_popup = true;
    return false;
  }
  if (signal.creating) {
    if (!insert_signal_line(&state->dbc_editor.text,
                            signal.message_address,
                            build_signal_definition_line(signal))) {
      state->error_text = "Failed to locate message in DBC text";
      state->open_error_popup = true;
      return false;
    }
  } else {
    if (!replace_signal_line(&state->dbc_editor.text,
                             signal.message_address,
                             signal.original_signal_name,
                             build_signal_definition_line(signal))) {
      state->error_text = "Failed to locate signal in DBC text";
      state->open_error_popup = true;
      return false;
    }
  }
  if (save_dbc_editor_contents(session, state)) {
    const std::string old_path = signal.creating
      ? std::string()
      : "/" + signal.service + "/" + std::to_string(signal.bus) + "/" + signal.message_name + "/" + signal.original_signal_name;
    const std::string new_path = "/" + signal.service + "/" + std::to_string(signal.bus) + "/" + signal.message_name + "/" + signal.signal_name;
    state->cabana_signal_editor.open = false;
    state->cabana_signal_editor.loaded = false;
    state->cabana.selected_message_root = signal.message_root;
    state->cabana.selected_signal_path = new_path;
    bool replaced_chart = false;
    if (!old_path.empty()) {
      for (std::string &path : state->cabana.chart_signal_paths) {
        if (path == old_path) {
          path = new_path;
          replaced_chart = true;
        }
      }
    }
    if (!signal.creating && !replaced_chart) {
      state->cabana.chart_signal_paths.erase(
        std::remove(state->cabana.chart_signal_paths.begin(), state->cabana.chart_signal_paths.end(), new_path),
        state->cabana.chart_signal_paths.end());
    }
    state->status_text = std::string(signal.creating ? "Created signal " : "Updated signal ") + signal.signal_name;
    return true;
  }
  return false;
}

bool apply_cabana_signal_delete_impl(AppSession *session, UiState *state) {
  if (!ensure_dbc_editor_loaded(*session, state)) {
    return false;
  }
  CabanaSignalEditorState &signal = state->cabana_signal_editor;
  if (signal.creating || trim_copy(signal.original_signal_name).empty()) {
    state->error_text = "No existing signal selected for deletion";
    state->open_error_popup = true;
    return false;
  }
  if (!remove_signal_line(&state->dbc_editor.text, signal.message_address, signal.original_signal_name)) {
    state->error_text = "Failed to locate signal in DBC text";
    state->open_error_popup = true;
    return false;
  }
  if (save_dbc_editor_contents(session, state)) {
    const std::string old_path = "/" + signal.service + "/" + std::to_string(signal.bus) + "/" + signal.message_name + "/" + signal.original_signal_name;
    state->cabana_signal_editor.open = false;
    state->cabana_signal_editor.loaded = false;
    state->cabana.selected_message_root = signal.message_root;
    if (state->cabana.selected_signal_path == old_path) {
      state->cabana.selected_signal_path.clear();
    }
    state->cabana.chart_signal_paths.erase(
      std::remove(state->cabana.chart_signal_paths.begin(), state->cabana.chart_signal_paths.end(), old_path),
      state->cabana.chart_signal_paths.end());
    state->status_text = "Deleted signal " + signal.original_signal_name;
    return true;
  }
  return false;
}

void draw_dbc_editor_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("DBC Editor", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  DbcEditorState &editor = state->dbc_editor;
  if (!ensure_dbc_editor_loaded(*session, state)) {
    ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return;
  }
  ImGui::TextUnformatted("Edit DBC text and save it into generated_dbcs.");
  ImGui::Separator();
  ImGui::SetNextItemWidth(260.0f);
  input_text_string("DBC Name", &editor.save_name, ImGuiInputTextFlags_AutoSelectAll);
  if (!editor.source_path.empty()) {
    ImGui::TextDisabled("%s", editor.source_path.c_str());
  } else {
    ImGui::TextDisabled("New in-memory DBC");
  }
  ImGui::Spacing();
  input_text_multiline_string("##dbc_editor_text", &editor.text, ImVec2(920.0f, 520.0f), ImGuiInputTextFlags_AllowTabInput);
  ImGui::Spacing();
  if (ImGui::Button("Apply + Save", ImVec2(140.0f, 0.0f))) {
    if (save_dbc_editor_contents(session, state)) {
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload Source", ImVec2(140.0f, 0.0f))) {
    editor.loaded = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    editor.loaded = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_cabana_signal_editor_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Edit CAN Signal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  CabanaSignalEditorState &signal = state->cabana_signal_editor;
  ImGui::TextUnformatted(signal.creating
                           ? "Create a decoded signal from the selected bit and save it into generated_dbcs."
                           : "Edit the selected decoded signal and save it into generated_dbcs.");
  ImGui::Separator();
  input_text_string("Name", &signal.signal_name, ImGuiInputTextFlags_AutoSelectAll);
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputInt("Start Bit", &signal.start_bit);
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputInt("Size", &signal.size);
  ImGui::Checkbox("Little Endian", &signal.is_little_endian);
  ImGui::Checkbox("Signed", &signal.is_signed);
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputDouble("Factor", &signal.factor, 0.0, 0.0, "%.6g");
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputDouble("Offset", &signal.offset, 0.0, 0.0, "%.6g");
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputDouble("Min", &signal.min, 0.0, 0.0, "%.6g");
  ImGui::SetNextItemWidth(140.0f);
  ImGui::InputDouble("Max", &signal.max, 0.0, 0.0, "%.6g");
  input_text_string("Unit", &signal.unit);
  input_text_string("Receiver", &signal.receiver_name);
  ImGui::Spacing();
  if (ImGui::Button("Apply + Save", ImVec2(140.0f, 0.0f))) {
    if (apply_cabana_signal_edit_impl(session, state)) {
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Open Raw DBC Editor", ImVec2(170.0f, 0.0f))) {
    state->dbc_editor.open = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    state->cabana_signal_editor.loaded = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_axis_limits_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Edit Axis Limits", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  const WorkspaceTab *tab = app_active_tab(session->layout, *state);
  const bool valid_pane = tab != nullptr
    && state->axis_limits.pane_index >= 0
    && state->axis_limits.pane_index < static_cast<int>(tab->panes.size());
  if (!valid_pane) {
    ImGui::TextWrapped("The selected pane is no longer available.");
    ImGui::Spacing();
    if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
      state->axis_limits.pane_index = -1;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    return;
  }

  ImGui::TextUnformatted("X range applies to the active tab. Y limits apply to the selected pane.");
  ImGui::Separator();
  ImGui::TextUnformatted("Horizontal");
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("X Min", &state->axis_limits.x_min, 0.0, 0.0, "%.3f");
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("X Max", &state->axis_limits.x_max, 0.0, 0.0, "%.3f");
  ImGui::Spacing();
  ImGui::TextUnformatted("Vertical");
  ImGui::Checkbox("Use Y Min", &state->axis_limits.y_min_enabled);
  ImGui::BeginDisabled(!state->axis_limits.y_min_enabled);
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("Y Min", &state->axis_limits.y_min, 0.0, 0.0, "%.6g");
  ImGui::EndDisabled();
  ImGui::Checkbox("Use Y Max", &state->axis_limits.y_max_enabled);
  ImGui::BeginDisabled(!state->axis_limits.y_max_enabled);
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("Y Max", &state->axis_limits.y_max, 0.0, 0.0, "%.6g");
  ImGui::EndDisabled();
  ImGui::Spacing();
  switch (draw_modal_action_row("Apply")) {
    case ModalAction::Primary:
      if (apply_axis_limits_editor(session, state)) {
        state->axis_limits.pane_index = -1;
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      state->axis_limits.pane_index = -1;
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_error_popup(UiState *state) {
  if (state->open_error_popup) {
    ImGui::OpenPopup("Error");
    state->open_error_popup = false;
  }
  if (!ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextWrapped("%s", state->error_text.c_str());
  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

}  // namespace

bool apply_cabana_signal_edit(AppSession *session, UiState *state) {
  return apply_cabana_signal_edit_impl(session, state);
}

bool apply_cabana_signal_delete(AppSession *session, UiState *state) {
  return apply_cabana_signal_delete_impl(session, state);
}

bool reset_layout(AppSession *session, UiState *state) {
  try {
    if (session->layout_path.empty()) {
      start_new_layout(session, state, "Reset layout");
      return true;
    }
    clear_layout_autosave(*session);
    session->layout = load_sketch_layout(session->layout_path);
    state->layout_dirty = false;
    session->autosave_path = autosave_path_for_layout(session->layout_path);
    state->undo.reset(session->layout);
    refresh_replaced_layout_ui(session, state, false);
    reset_shared_range(state, *session);
    state->status_text = "Reset layout";
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to reset layout";
    return false;
  }
}

bool reload_layout(AppSession *session, UiState *state, const std::string &layout_arg) {
  try {
    const bool preserve_shared_range = session->route_data.has_time_range && state->has_shared_range;
    const double preserved_x_min = state->x_view_min;
    const double preserved_x_max = state->x_view_max;
    const fs::path layout_path = resolve_layout_path(layout_arg);
    session->autosave_path = autosave_path_for_layout(layout_path);
    const bool load_draft = fs::exists(session->autosave_path);
    session->layout = load_sketch_layout(load_draft ? session->autosave_path : layout_path);
    session->layout_path = layout_path;
    state->layout_dirty = load_draft;
    state->undo.reset(session->layout);
    refresh_replaced_layout_ui(session, state, true);
    if (preserve_shared_range) {
      state->has_shared_range = true;
      state->x_view_min = preserved_x_min;
      state->x_view_max = preserved_x_max;
      clamp_shared_range(state, *session);
    } else {
      reset_shared_range(state, *session);
    }
    state->status_text = std::string(load_draft ? "Loaded layout draft " : "Loaded layout ")
      + layout_path.filename().string();
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to load layout";
    return false;
  }
}

bool save_layout(AppSession *session, UiState *state, const std::string &layout_path) {
  try {
    if (layout_path.empty()) throw std::runtime_error("Layout path is empty");
    session->layout.current_tab_index = state->active_tab_index;
    const fs::path previous_autosave = session->autosave_path;
    const fs::path output = fs::absolute(fs::path(layout_path));
    save_layout_json(session->layout, output);
    session->layout_path = output;
    session->autosave_path = autosave_path_for_layout(output);
    if (!previous_autosave.empty() && previous_autosave != session->autosave_path && fs::exists(previous_autosave)) {
      fs::remove(previous_autosave);
    }
    clear_layout_autosave(*session);
    state->layout_dirty = false;
    sync_layout_buffers(state, *session);
    state->status_text = "Saved layout " + output.filename().string();
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to save layout";
    return false;
  }
}

void rebuild_session_route_data(AppSession *session, UiState *state,
                                const RouteLoadProgressCallback &progress) {
  apply_route_data(session, state, load_route_data(session->route_name, session->data_dir, session->dbc_override, progress));
}

void stop_stream_session(AppSession *session, UiState *state, bool preserve_data) {
  if (preserve_data && session->stream_poller && session->data_mode == SessionDataMode::Stream) {
    session->stream_poller->setPaused(true);
  } else if (session->stream_poller) {
    session->stream_poller->stop();
  }
  session->stream_paused = preserve_data && session->data_mode == SessionDataMode::Stream;
  if (!preserve_data) {
    session->stream_time_offset.reset();
    apply_route_data(session, state, RouteData{});
  }
  sync_stream_buffers(state, *session);
}

bool start_stream_session(AppSession *session,
                          UiState *state,
                          const StreamSourceConfig &source,
                          double buffer_seconds,
                          bool preserve_existing_data) {
  try {
    if (session->route_loader) {
      session->route_loader.reset();
    }
    session->data_mode = SessionDataMode::Stream;
    session->route_id = {};
    session->route_name.clear();
    session->data_dir.clear();
    session->stream_source = source;
    if (session->stream_source.kind == StreamSourceKind::CerealLocal) {
      session->stream_source.address = "127.0.0.1";
    }
    session->stream_buffer_seconds = std::max(1.0, buffer_seconds);
    session->next_stream_custom_refresh_time = 0.0;
    session->stream_paused = false;
    if (preserve_existing_data && session->stream_poller) {
      StreamPollSnapshot snapshot = session->stream_poller->snapshot();
      if (snapshot.active) {
        session->stream_poller->setPaused(false);
        sync_route_buffers(state, *session);
        sync_stream_buffers(state, *session);
        state->follow_latest = true;
        state->playback_playing = false;
        state->status_text = "Resumed stream " + stream_source_target_label(session->stream_source);
        return true;
      }
    }
    if (!preserve_existing_data) {
      session->stream_time_offset.reset();
      apply_route_data(session, state, RouteData{});
    }
    if (!session->stream_poller) {
      session->stream_poller = std::make_unique<StreamPoller>();
    }
    session->stream_poller->start(session->stream_source,
                                  session->stream_buffer_seconds,
                                  session->dbc_override,
                                  session->stream_time_offset);
    sync_route_buffers(state, *session);
    sync_stream_buffers(state, *session);
    state->follow_latest = true;
    state->playback_playing = false;
    state->status_text = preserve_existing_data ? "Resumed stream " + stream_source_target_label(session->stream_source)
                                                : "Streaming from " + stream_source_target_label(session->stream_source);
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to start stream";
    return false;
  }
}

void start_async_route_load(AppSession *session, UiState *state) {
  if (!session->route_loader) {
    return;
  }
  apply_route_data(session, state, RouteData{});
  session->route_loader->start(session->route_name, session->data_dir, session->dbc_override);
  state->status_text = session->route_name.empty() ? "Ready" : "Loading route " + session->route_name;
}

void poll_async_route_load(AppSession *session, UiState *state) {
  if (!session->route_loader) {
    return;
  }
  RouteData loaded_route;
  std::string error_text;
  if (!session->route_loader->consume(&loaded_route, &error_text)) {
    return;
  }
  if (!error_text.empty()) {
    state->error_text = error_text;
    state->open_error_popup = true;
    state->status_text = "Failed to load route";
    return;
  }
  apply_route_data(session, state, std::move(loaded_route));
  state->status_text = session->route_name.empty() ? "Ready" : "Loaded route " + session->route_name;
}

bool reload_session(AppSession *session, UiState *state, const std::string &route_name, const std::string &data_dir) {
  try {
    stop_stream_session(session, state, false);
    session->data_mode = SessionDataMode::Route;
    session->route_name = route_name;
    session->route_id = parse_route_identifier(route_name);
    session->data_dir = data_dir;
    if (session->async_route_loading) {
      if (!session->route_loader) {
        session->route_loader = std::make_unique<AsyncRouteLoader>(::isatty(STDERR_FILENO) != 0);
      }
      start_async_route_load(session, state);
    } else {
      rebuild_session_route_data(session, state);
      state->status_text = "Loaded route " + route_name;
    }
    sync_route_buffers(state, *session);
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to load route";
    return false;
  }
}

void draw_popups(AppSession *session, UiState *state) {
  open_queued_popup(state->open_open_route, "Open Route");
  if (state->open_stream) {
    sync_stream_buffers(state, *session);
  }
  open_queued_popup(state->open_stream, "Live Stream");
  if (state->open_load_layout || state->open_save_layout) {
    sync_layout_buffers(state, *session);
  }
  open_queued_popup(state->open_load_layout, "Load Layout");
  open_queued_popup(state->open_save_layout, "Save Layout");
  open_queued_popup(state->open_preferences, "Preferences");
  open_queued_popup(state->dbc_editor.open, "DBC Editor");
  open_queued_popup(state->cabana_signal_editor.open, "Edit CAN Signal");
  open_queued_popup(state->open_find_signal, "Find Signal");
  open_queued_popup(state->axis_limits.open, "Edit Axis Limits");

  draw_open_route_popup(session, state);
  draw_stream_popup(session, state);
  draw_load_layout_popup(session, state);
  draw_save_layout_popup(session, state);
  draw_preferences_popup(session, state);
  draw_dbc_editor_popup(session, state);
  draw_cabana_signal_editor_popup(session, state);
  draw_find_signal_popup(session, state);
  draw_axis_limits_popup(session, state);
  draw_error_popup(state);
}
