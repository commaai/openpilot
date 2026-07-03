// ImGui port of tools/cabana/streamselector.{h,cc} (StreamSelector) plus the
// per-stream Open*Widget forms that used to live in streams/*.cc before the
// de-Qt pass (recovered from git history, see MIGRATION.md Phase 6). Parity
// target: tools/cabana/streamselector.cc (tabbed dialog) and
// tools/cabana/mainwin.cc's MainWindow::openStream/startStream/selectAndOpenStream
// (runtime stream-swap lifecycle), both frozen Qt references.
#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "common/util.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/imgui/dbc_menus.h"
#include "tools/cabana/imgui/file_dialog.h"
#include "tools/cabana/panda.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif
#include "tools/replay/route.h"

namespace fs = std::filesystem;

namespace {

enum class Tab { Replay, Panda, SocketCan, Device };

struct SelectorState {
  bool need_open = false;
  bool active = false;
  Tab tab = Tab::Replay;
  bool has_socketcan = false;

  // dbc file field (shared across tabs, mirrors StreamSelector::dbc_file)
  char dbc_buf[1024] = {};

  // replay tab (OpenReplayWidget)
  char route_buf[1024] = {};
  bool cam_road = true;
  bool cam_driver = false;
  bool cam_wide = false;

  // panda tab (OpenPandaWidget)
  std::vector<std::string> panda_serials;
  int panda_serial_idx = -1;
  bool panda_has_serial = false;
  bool panda_has_fd = false;
  PandaStreamConfig panda_config;

  // socketcan tab (OpenSocketCanWidget)
  std::vector<std::string> can_devices;
  int can_device_idx = -1;

  // device tab (OpenDeviceWidget): 0 = MSGQ, 1 = ZMQ (Qt default: zmq checked)
  int device_mode = 1;
  char ip_buf[64] = {};

  std::string error;
};
SelectorState g_sel;

// Target written by open_remote_route_browser() on confirm (routes_dialog.cc,
// another workstream's file); synced into route_buf once populated.
std::string g_remote_route_out;

struct WarningState {
  bool need_open = false;
  bool active = false;
  std::string message;
};
WarningState g_warn;

void show_warning(const std::string &msg) {
  g_warn.message = msg;
  g_warn.need_open = true;
  g_warn.active = true;
}

void draw_warning_popup() {
  constexpr const char *kPopupId = "Warning##stream_selector_warning";
  if (g_warn.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_warn.need_open = false;
  }
  if (!g_warn.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSizeConstraints(ImVec2(280.0f, 0.0f), ImVec2(520.0f, FLT_MAX));
  if (ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", g_warn.message.c_str());
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      ImGui::CloseCurrentPopup();
      g_warn.active = false;
    }
    ImGui::EndPopup();
  }
}

// -- "Local route..." directory picker ---------------------------------------
// file_dialog.h's shared browser only supports Open (existing regular file)
// and Save (no directory concept) modes -- neither can return a directory the
// way Qt's QFileDialog::getExistingDirectory() does, and file_dialog.{h,cc}
// isn't owned by this package. This is a small, self-contained directory
// browser instead of extending someone else's file. See report for details.

struct DirPickerState {
  bool need_open = false;
  bool active = false;
  fs::path current_dir;
  std::vector<std::string> subdirs;
  bool refresh_needed = true;
};
DirPickerState g_dirpick;

void dirpick_refresh() {
  g_dirpick.subdirs.clear();
  std::error_code ec;
  fs::directory_iterator it(g_dirpick.current_dir, fs::directory_options::skip_permission_denied, ec);
  if (!ec) {
    for (const auto &de : it) {
      std::error_code ec2;
      if (!de.is_directory(ec2)) continue;
      std::string name = de.path().filename().string();
      if (name.empty() || name[0] == '.') continue;
      g_dirpick.subdirs.push_back(name);
    }
    std::sort(g_dirpick.subdirs.begin(), g_dirpick.subdirs.end());
  }
  g_dirpick.refresh_needed = false;
}

void dirpick_open(const std::string &start_dir) {
  fs::path dir = start_dir.empty() ? fs::current_path() : fs::path(start_dir);
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) dir = fs::current_path();
  g_dirpick.current_dir = dir;
  g_dirpick.refresh_needed = true;
  g_dirpick.need_open = true;
  g_dirpick.active = true;
}

void draw_dir_picker() {
  constexpr const char *kPopupId = "Open Local Route##stream_selector_dirpick";
  if (g_dirpick.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_dirpick.need_open = false;
  }
  if (!g_dirpick.active) return;

  ImGui::SetNextWindowSize(ImVec2(560.0f, 420.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  ImGui::TextUnformatted(g_dirpick.current_dir.string().c_str());
  ImGui::Separator();

  if (g_dirpick.refresh_needed) dirpick_refresh();

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float bottom_h = ImGui::GetFrameHeightWithSpacing();
  if (ImGui::BeginChild("##ss_dirlist", ImVec2(0.0f, avail.y - bottom_h), ImGuiChildFlags_Borders)) {
    const bool at_root = !g_dirpick.current_dir.has_relative_path() && g_dirpick.current_dir == g_dirpick.current_dir.root_path();
    if (!at_root) {
      if (ImGui::Selectable("..", false, ImGuiSelectableFlags_AllowDoubleClick) && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        g_dirpick.current_dir = g_dirpick.current_dir.parent_path();
        g_dirpick.refresh_needed = true;
      }
    }
    for (const auto &name : g_dirpick.subdirs) {
      if (ImGui::Selectable(("[dir]  " + name).c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) &&
          ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        g_dirpick.current_dir /= name;
        g_dirpick.refresh_needed = true;
        break;
      }
    }
  }
  ImGui::EndChild();

  if (ImGui::Button("Select Folder", ImVec2(140.0f, 0.0f))) {
    std::snprintf(g_sel.route_buf, sizeof(g_sel.route_buf), "%s", g_dirpick.current_dir.string().c_str());
    settings.last_route_dir = g_dirpick.current_dir.parent_path().string();
    g_dirpick.active = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    g_dirpick.active = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

// -- hand-rolled tab button ---------------------------------------------------
// Native ImGui::BeginTabBar()/BeginTabItem() doesn't render tab-item caption
// text on the frame a tab is (re)created in this ImGui build -- see
// detail_panel.cc's draw_tab_button()/video_panel.cc's draw_camera_tab_button()
// for the original repro. Same workaround here.
bool draw_selector_tab_button(const char *label, bool selected) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(selected ? ImGuiCol_TabSelected : ImGuiCol_Tab));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_TabHovered));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_TabSelected));
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
  const bool clicked = ImGui::Button(label);
  ImGui::PopStyleColor(4);
  return clicked;
}

// -- panda tab helpers (mirrors OpenPandaWidget::refreshSerials/buildConfigForm) --

// Probes hw_type via a throwaway Panda connection, mirroring buildConfigForm()'s
// `Panda panda(serial.toStdString())` -- resize(3) (not assign) preserves any
// bus speeds the user already dialed in when switching between serials, same
// as the Qt reference.
void panda_rebuild_config() {
  g_sel.panda_has_fd = false;
  g_sel.panda_has_serial = false;
  if (g_sel.panda_serial_idx < 0 || g_sel.panda_serial_idx >= (int)g_sel.panda_serials.size()) {
    g_sel.panda_config.serial.clear();
    return;
  }
  const std::string &serial = g_sel.panda_serials[g_sel.panda_serial_idx];
  try {
    Panda probe(serial);
    g_sel.panda_has_fd = (probe.hw_type == cereal::PandaState::PandaType::RED_PANDA ||
                          probe.hw_type == cereal::PandaState::PandaType::RED_PANDA_V2);
    g_sel.panda_config.serial = serial;
    g_sel.panda_config.bus_config.resize(3);
    g_sel.panda_has_serial = true;
  } catch (const std::exception &) {
    g_sel.panda_config.serial.clear();
  }
}

void panda_refresh_serials() {
  g_sel.panda_serials = Panda::list();
  g_sel.panda_serial_idx = g_sel.panda_serials.empty() ? -1 : 0;
  panda_rebuild_config();
}

#ifdef __linux__
void socketcan_refresh_devices() {
  g_sel.can_devices.clear();
  std::error_code ec;
  fs::directory_iterator it("/sys/class/net", ec);
  if (!ec) {
    for (const auto &entry : it) {
      std::error_code ec2;
      if (!entry.is_directory(ec2)) continue;
      const std::string iface = entry.path().filename().string();
      std::ifstream f(entry.path() / "type");
      int type = -1;
      if (f) f >> type;
      if (type == 280) g_sel.can_devices.push_back(iface);  // ARPHRD_CAN
    }
  }
  std::sort(g_sel.can_devices.begin(), g_sel.can_devices.end());
  g_sel.can_device_idx = g_sel.can_devices.empty() ? -1 : 0;
}
#endif

// -- tab content ---------------------------------------------------------------

void draw_replay_tab() {
  if (!g_remote_route_out.empty()) {
    std::snprintf(g_sel.route_buf, sizeof(g_sel.route_buf), "%s", g_remote_route_out.c_str());
    g_remote_route_out.clear();
  }

  constexpr float kBtnW = 130.0f;
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Route");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(std::max(60.0f, ImGui::GetContentRegionAvail().x - 2.0f * (kBtnW + ImGui::GetStyle().ItemSpacing.x)));
  ImGui::InputTextWithHint("##ss_route", "Enter route name or browse for local/remote route", g_sel.route_buf, sizeof(g_sel.route_buf));
  ImGui::SameLine();
  if (ImGui::Button("Remote route...", ImVec2(kBtnW, 0.0f))) {
    g_remote_route_out.clear();
    open_remote_route_browser(&g_remote_route_out);
  }
  ImGui::SameLine();
  if (ImGui::Button("Local route...", ImVec2(kBtnW, 0.0f))) {
    dirpick_open(settings.last_route_dir);
  }

  ImGui::Spacing();
  ImGui::Checkbox("Road camera", &g_sel.cam_road);
  ImGui::SameLine();
  ImGui::Checkbox("Driver camera", &g_sel.cam_driver);
  ImGui::SameLine();
  ImGui::Checkbox("Wide road camera", &g_sel.cam_wide);

  draw_dir_picker();
}

void draw_panda_tab() {
  // mirrors OpenPandaWidget's early return when a PandaStream is already
  // connected -- Panda hardware can't be opened twice concurrently.
  if (dynamic_cast<PandaStream *>(can) != nullptr) {
    ImGui::TextWrapped("Already connected to %s.", can->routeName().c_str());
    ImGui::TextWrapped("Close the current connection via [File menu -> Close Stream] before connecting to another Panda.");
    return;
  }

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Serial");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(240.0f);
  const char *preview = (g_sel.panda_serial_idx >= 0) ? g_sel.panda_serials[g_sel.panda_serial_idx].c_str() : "";
  if (ImGui::BeginCombo("##ss_panda_serial", preview)) {
    for (int i = 0; i < (int)g_sel.panda_serials.size(); ++i) {
      const bool selected = (i == g_sel.panda_serial_idx);
      if (ImGui::Selectable(g_sel.panda_serials[i].c_str(), selected) && !selected) {
        g_sel.panda_serial_idx = i;
        panda_rebuild_config();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) panda_refresh_serials();

  if (!g_sel.panda_has_serial) {
    ImGui::TextUnformatted("No panda found");
    return;
  }

  for (int i = 0; i < (int)g_sel.panda_config.bus_config.size(); ++i) {
    ImGui::PushID(i);
    BusConfig &bus = g_sel.panda_config.bus_config[i];

    ImGui::AlignTextToFramePadding();
    ImGui::Text("Bus %d:", i);
    ImGui::SameLine();
    ImGui::TextUnformatted("CAN Speed (kbps):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::BeginCombo("##can_speed", std::to_string(bus.can_speed_kbps).c_str())) {
      for (uint32_t s : speeds) {
        const bool selected = ((int)s == bus.can_speed_kbps);
        if (ImGui::Selectable(std::to_string(s).c_str(), selected)) bus.can_speed_kbps = (int)s;
      }
      ImGui::EndCombo();
    }

    if (g_sel.panda_has_fd) {
      ImGui::SameLine();
      ImGui::Checkbox("CAN-FD", &bus.can_fd);
      ImGui::SameLine();
      ImGui::TextUnformatted("Data Speed (kbps):");
      ImGui::SameLine();
      ImGui::BeginDisabled(!bus.can_fd);
      ImGui::SetNextItemWidth(90.0f);
      if (ImGui::BeginCombo("##data_speed", std::to_string(bus.data_speed_kbps).c_str())) {
        for (uint32_t s : data_speeds) {
          const bool selected = ((int)s == bus.data_speed_kbps);
          if (ImGui::Selectable(std::to_string(s).c_str(), selected)) bus.data_speed_kbps = (int)s;
        }
        ImGui::EndCombo();
      }
      ImGui::EndDisabled();
    }
    ImGui::PopID();
  }
}

void draw_socketcan_tab() {
#ifdef __linux__
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(240.0f);
  const char *preview = (g_sel.can_device_idx >= 0) ? g_sel.can_devices[g_sel.can_device_idx].c_str() : "";
  if (ImGui::BeginCombo("##ss_can_device", preview)) {
    for (int i = 0; i < (int)g_sel.can_devices.size(); ++i) {
      const bool selected = (i == g_sel.can_device_idx);
      if (ImGui::Selectable(g_sel.can_devices[i].c_str(), selected)) g_sel.can_device_idx = i;
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("Refresh##can")) socketcan_refresh_devices();
  if (g_sel.can_devices.empty()) {
    ImGui::TextUnformatted("No SocketCAN devices found");
  }
#endif
}

void draw_device_tab() {
  ImGui::RadioButton("MSGQ", &g_sel.device_mode, 0);
  ImGui::SameLine();
  ImGui::RadioButton("ZMQ", &g_sel.device_mode, 1);
  ImGui::SameLine();
  ImGui::BeginDisabled(g_sel.device_mode != 1);
  ImGui::SetNextItemWidth(220.0f);
  ImGui::InputTextWithHint("##ss_ip", "Enter device Ip Address", g_sel.ip_buf, sizeof(g_sel.ip_buf));
  ImGui::EndDisabled();
}

void draw_dbc_row() {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("dbc File");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(std::max(60.0f, ImGui::GetContentRegionAvail().x - 110.0f));
  ImGui::InputTextWithHint("##ss_dbc_file", "Choose a dbc file to open", g_sel.dbc_buf, sizeof(g_sel.dbc_buf),
                            ImGuiInputTextFlags_ReadOnly);
  ImGui::SameLine();
  if (ImGui::Button("Browse...", ImVec2(100.0f, 0.0f))) {
    file_dialog_open(FileDialogMode::Open, "Open File", settings.last_dir, ".dbc", "", [](const std::string &path) {
      std::snprintf(g_sel.dbc_buf, sizeof(g_sel.dbc_buf), "%s", path.c_str());
      settings.last_dir = fs::path(path).parent_path().string();
    });
  }
}

// -- build-new-stream per tab (mirrors each Open*Widget::open()) -------------

std::unique_ptr<AbstractStream> open_replay() {
  std::string route = g_sel.route_buf;
  std::string data_dir;
  if (auto idx = route.rfind('/'); idx != std::string::npos && util::file_exists(route)) {
    data_dir = route.substr(0, idx + 1);
    route = route.substr(idx + 1);
  }

  if (Route::parseRoute(route).str.empty()) {
    g_sel.error = "Invalid route format: '" + route + "'";
    return nullptr;
  }

  uint32_t flags = REPLAY_FLAG_NONE;
  if (g_sel.cam_driver) flags |= REPLAY_FLAG_DCAM;
  if (g_sel.cam_wide) flags |= REPLAY_FLAG_ECAM;
  if (flags == REPLAY_FLAG_NONE && !g_sel.cam_road) flags = REPLAY_FLAG_NO_VIPC;

  auto stream = std::make_unique<ReplayStream>();
  if (!stream->loadRoute(route, data_dir, flags)) {
    // detailed cause already printed to stderr by loadRoute()
    g_sel.error = "Failed to load route: '" + route + "'";
    return nullptr;
  }
  return stream;
}

std::unique_ptr<AbstractStream> open_panda() {
  try {
    return std::make_unique<PandaStream>(g_sel.panda_config);
  } catch (const std::exception &e) {
    g_sel.error = std::string("Failed to connect to panda: '") + e.what() + "'";
    return nullptr;
  }
}

std::unique_ptr<AbstractStream> open_socketcan() {
#ifdef __linux__
  try {
    SocketCanStreamConfig cfg;
    cfg.device = (g_sel.can_device_idx >= 0) ? g_sel.can_devices[g_sel.can_device_idx] : std::string();
    return std::make_unique<SocketCanStream>(cfg);
  } catch (const std::exception &e) {
    g_sel.error = std::string("Failed to connect to SocketCAN device: '") + e.what() + "'";
    return nullptr;
  }
#else
  g_sel.error = "SocketCAN is only available on Linux";
  return nullptr;
#endif
}

std::unique_ptr<AbstractStream> open_device() {
  const std::string ip = g_sel.ip_buf[0] != '\0' ? std::string(g_sel.ip_buf) : std::string("127.0.0.1");
  const bool msgq = (g_sel.device_mode == 0);
  return std::make_unique<DeviceStream>(msgq ? std::string() : ip);
}

bool tab_can_open() {
  // mirrors AbstractOpenStreamWidget::enableOpenButton wiring: every tab can
  // attempt to open except Panda while already connected to one.
  if (g_sel.tab == Tab::Panda) return dynamic_cast<PandaStream *>(can) == nullptr;
  return true;
}

// -- swap lifecycle (mirrors MainWindow::openStream/startStream) -------------
//
// Teardown order: the caller only reaches here once a *new* stream has
// already been fully constructed (loadRoute()'d / connect()'d) -- exactly
// like Qt's widget->open() running before StreamSelector::accept(). So a
// failure to build never touches app.stream; the old stream (if any) keeps
// running and the dialog stays open. On success: destroy the old stream
// first (frees its thread/socket/USB resources), then install + start the
// new one, then load the dbc field (if set) -- Qt's loadFile(dbc_file) is a
// no-op for an empty filename, so the previously-loaded DBC survives
// otherwise, same here. Selection/tab state is cleared since it's keyed to
// data that belonged to the old stream.
void swap_stream(AppState &app, std::unique_ptr<AbstractStream> new_stream) {
  app.stream.reset();
  app.stream = std::move(new_stream);
  can = app.stream.get();
  app.stream->start();

  const std::string dbc_path = g_sel.dbc_buf;
  if (!dbc_path.empty()) {
    std::string error;
    if (dbc()->open(SOURCE_ALL, dbc_path, &error)) {
      dbc_menus_note_recent_file(dbc_path);
    } else {
      fprintf(stderr, "[cabana] Failed to load DBC file %s: %s\n", dbc_path.c_str(), error.c_str());
    }
  }

  app.selected_msg_id.reset();
  app.open_msg_tabs.clear();

  // Don't overwrite an already-loaded DBC; ensure at least one (possibly
  // empty) file is open, same as Qt's startStream() fallback.
  dbc_menus_ensure_dbc_open();
}

void try_open(AppState &app) {
  g_sel.error.clear();
  std::unique_ptr<AbstractStream> new_stream;
  switch (g_sel.tab) {
    case Tab::Replay: new_stream = open_replay(); break;
    case Tab::Panda: new_stream = open_panda(); break;
    case Tab::SocketCan: new_stream = open_socketcan(); break;
    case Tab::Device: new_stream = open_device(); break;
  }

  if (!new_stream) {
    show_warning(g_sel.error.empty() ? "Failed to open stream." : g_sel.error);
    return;
  }

  swap_stream(app, std::move(new_stream));
  g_sel.active = false;
  ImGui::CloseCurrentPopup();
}

}  // namespace

void draw_stream_selector(AppState &app) {
  draw_warning_popup();

  constexpr const char *kPopupId = "Open stream##stream_selector";
  if (g_sel.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_sel.need_open = false;
  }
  if (!g_sel.active) return;

  ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  if (draw_selector_tab_button("Replay", g_sel.tab == Tab::Replay)) g_sel.tab = Tab::Replay;
  ImGui::SameLine();
  if (draw_selector_tab_button("Panda", g_sel.tab == Tab::Panda)) g_sel.tab = Tab::Panda;
  if (g_sel.has_socketcan) {
    ImGui::SameLine();
    if (draw_selector_tab_button("SocketCAN", g_sel.tab == Tab::SocketCan)) g_sel.tab = Tab::SocketCan;
  }
  ImGui::SameLine();
  if (draw_selector_tab_button("Device", g_sel.tab == Tab::Device)) g_sel.tab = Tab::Device;

  ImGui::Separator();
  const float footer_h = ImGui::GetFrameHeightWithSpacing() * 2.0f + ImGui::GetStyle().ItemSpacing.y;
  if (ImGui::BeginChild("##ss_tab_content", ImVec2(0.0f, std::max(0.0f, ImGui::GetContentRegionAvail().y - footer_h)))) {
    switch (g_sel.tab) {
      case Tab::Replay: draw_replay_tab(); break;
      case Tab::Panda: draw_panda_tab(); break;
      case Tab::SocketCan: draw_socketcan_tab(); break;
      case Tab::Device: draw_device_tab(); break;
    }
  }
  ImGui::EndChild();

  ImGui::Separator();
  draw_dbc_row();
  ImGui::Separator();

  const bool can_open = tab_can_open();
  ImGui::BeginDisabled(!can_open);
  if (ImGui::Button("Open", ImVec2(100.0f, 0.0f))) {
    try_open(app);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    g_sel.active = false;
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndPopup();
}

void open_stream_selector() {
  g_sel = SelectorState{};
  g_sel.need_open = true;
  g_sel.active = true;
#ifdef __linux__
  g_sel.has_socketcan = SocketCanStream::available();
  if (g_sel.has_socketcan) socketcan_refresh_devices();
#endif
  panda_refresh_serials();
}
