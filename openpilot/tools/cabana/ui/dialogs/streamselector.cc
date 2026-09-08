#include "tools/cabana/ui/dialogs/streamselector.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/util.h"

void OpenReplayWidget::draw() {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Route");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-250.0f);
  inputText("##route", &route_, "Enter a route name or browse for a local or remote route");
  ImGui::SameLine();
  if (ImGui::Button("Remote Route...")) {
    routes_dialog_.open(utils::guarded(alive_, [this](bool accepted, const std::string &route) {
      if (accepted) route_ = route;
    }));
  }
  ImGui::SameLine();
  if (ImGui::Button("Local Route...")) {
    FileDialog::getExistingDirectory("Open Local Route", settings.last_route_dir, utils::guarded(alive_, [this](const std::string &dir) {
      if (!dir.empty()) {
        route_ = dir;
        settings.last_route_dir = std::filesystem::absolute(dir).parent_path().string();
      }
    }));
  }
  checkBox("Road camera", &cameras_[0]);
  ImGui::SameLine();
  checkBox("Driver camera", &cameras_[1]);
  ImGui::SameLine();
  checkBox("Wide road camera", &cameras_[2]);
}

void OpenReplayWidget::drawPopups() {
  routes_dialog_.draw();
}

std::unique_ptr<AbstractStream> OpenReplayWidget::open() {
  std::string route = route_;
  std::string data_dir;
  if (auto idx = route.rfind('/'); idx != std::string::npos && util::file_exists(route)) {
    data_dir = route.substr(0, idx + 1);
    route = route.substr(idx + 1);
  }

  bool is_valid_format = Route::parseRoute(route).str.size() > 0;
  if (!is_valid_format) {
    MessageBox::warning("Warning", "Invalid route format: '" + route + "'");
  } else {
    auto replay_stream = std::make_unique<ReplayStream>();
    Connection err = replay_stream->error.connect([](const std::string &msg) {
      MessageBox::warning("Error", msg);
    });
    uint32_t flags = REPLAY_FLAG_NONE;
    if (cameras_[1]) flags |= REPLAY_FLAG_CABIN_CAMERA;
    if (cameras_[2]) flags |= REPLAY_FLAG_WIDE_ROAD;
    if (flags == REPLAY_FLAG_NONE && !cameras_[0]) flags = REPLAY_FLAG_NO_VIPC;

    if (replay_stream->loadRoute(route, data_dir, flags)) {
      return replay_stream;
    }
  }
  return nullptr;
}

namespace {
const uint32_t speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U};
const uint32_t data_speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U, 2000U, 5000U};
}

OpenPandaWidget::OpenPandaWidget() {
  if (can && dynamic_cast<PandaStream *>(can) != nullptr) {
    already_connected_ = true;
    return;
  }
  refreshSerials();
  buildConfigForm();
}

void OpenPandaWidget::refreshSerials() {
  serials_ = Panda::list();
  serial_index_ = 0;
}

void OpenPandaWidget::buildConfigForm() {
  std::string serial = serial_index_ < static_cast<int>(serials_.size()) ? serials_[serial_index_] : "";
  has_fd_ = false;
  has_panda_ = !serial.empty();
  if (has_panda_) {
    try {
      Panda panda(serial);
      has_fd_ = (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA) || (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA_V2);
    } catch (const std::exception &e) {
      fprintf(stderr, "failed to open panda %s\n", serial.c_str());
      has_panda_ = false;
    }
  }

  if (has_panda_) {
    config.serial = serial;
    config.bus_config.resize(3);
    can_speed_index_.assign(3, 0);
    data_speed_index_.assign(3, 0);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < static_cast<int>(std::size(speeds)); j++) {
        if (speeds[j] == config.bus_config[i].can_speed_kbps) can_speed_index_[i] = j;
      }
      for (int j = 0; j < static_cast<int>(std::size(data_speeds)); j++) {
        if (data_speeds[j] == config.bus_config[i].data_speed_kbps) data_speed_index_[i] = j;
      }
    }
  } else {
    config.serial = "";
  }
}

void OpenPandaWidget::draw() {
  if (already_connected_) {
    ImGui::Text("Already connected to %s.", can->routeName().c_str());
    ImGui::TextUnformatted("Select File > Close Stream before connecting to another panda.");
    return;
  }
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Serial");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-100.0f);
  if (comboBox("##serial", &serial_index_, serials_)) buildConfigForm();
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) {
    refreshSerials();
    buildConfigForm();
  }

  if (!has_panda_) {
    ImGui::TextUnformatted("No panda found");
    return;
  }
  for (int i = 0; i < static_cast<int>(config.bus_config.size()); i++) {
    ImGui::PushID(i);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Bus %d:", i);
    ImGui::SameLine();
    ImGui::TextUnformatted("CAN Speed (kbps):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (comboBox("##can_speed", &can_speed_index_[i], speeds, (int)std::size(speeds))) {
      config.bus_config[i].can_speed_kbps = speeds[can_speed_index_[i]];
    }
    if (has_fd_) {
      ImGui::SameLine();
      checkBox("CAN-FD", &config.bus_config[i].can_fd);
      ImGui::SameLine();
      ImGui::TextUnformatted("Data Speed (kbps):");
      ImGui::SameLine();
      ImGui::BeginDisabled(!config.bus_config[i].can_fd);
      ImGui::SetNextItemWidth(90.0f);
      if (comboBox("##data_speed", &data_speed_index_[i], data_speeds, (int)std::size(data_speeds))) {
        config.bus_config[i].data_speed_kbps = data_speeds[data_speed_index_[i]];
      }
      ImGui::EndDisabled();
    }
    ImGui::PopID();
  }
}

std::unique_ptr<AbstractStream> OpenPandaWidget::open() {
  try {
    return std::make_unique<PandaStream>(config);
  } catch (std::exception &e) {
    MessageBox::warning("Warning", std::string("Failed to connect to panda: '") + e.what() + "'");
    return nullptr;
  }
}

void OpenDeviceWidget::draw() {
  ImGui::RadioButton("MSGQ", &mode_, 0);
  ImGui::RadioButton("ZMQ", &mode_, 1);
  // the radio buttons are the label column, the ip address is the field column
  const float label_width = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x +
                            std::max(ImGui::CalcTextSize("MSGQ").x, ImGui::CalcTextSize("ZMQ").x) +
                            ImGui::GetStyle().ItemInnerSpacing.x;
  ImGui::SameLine(label_width);
  ImGui::BeginDisabled(mode_ != 1);
  ImGui::SetNextItemWidth(-1.0f);
  validatedText("##ip", &ip_address_, validateIpAddress, "Enter device IP address", ipValidator);
  ImGui::EndDisabled();
}

std::unique_ptr<AbstractStream> OpenDeviceWidget::open() {
  std::string ip = ip_address_.empty() ? "127.0.0.1" : ip_address_;
  bool msgq = mode_ == 0;
  return std::make_unique<DeviceStream>(msgq ? "" : ip);
}

#ifdef __linux__

OpenSocketCanWidget::OpenSocketCanWidget() {
  refreshDevices();
}

void OpenSocketCanWidget::refreshDevices() {
  devices_.clear();
  // type 280 = ARPHRD_CAN
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator("/sys/class/net", ec)) {
    std::ifstream type_file(entry.path() / "type");
    int type = 0;
    if (type_file >> type && type == 280) {
      devices_.push_back(entry.path().filename().string());
    }
  }
  device_index_ = 0;
  config.device = devices_.empty() ? "" : devices_[0];
}

void OpenSocketCanWidget::draw() {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(300.0f);
  if (comboBox("##device", &device_index_, devices_)) config.device = devices_[device_index_];
  ImGui::SameLine();
  if (ImGui::Button("Refresh", ImVec2(100.0f, 0.0f))) refreshDevices();
}

std::unique_ptr<AbstractStream> OpenSocketCanWidget::open() {
  try {
    return std::make_unique<SocketCanStream>(config);
  } catch (std::exception &e) {
    MessageBox::warning("Warning", std::string("Failed to connect to SocketCAN device: '") + e.what() + "'");
    return nullptr;
  }
}
#endif

void StreamSelector::open(Callback on_done) {
  on_done_ = std::move(on_done);
  open_ = true;
  popup_.reset();
  first_frame_ = true;
  dbc_file_.clear();
  widgets_.clear();
  widgets_.push_back(std::make_unique<OpenReplayWidget>());
  widgets_.push_back(std::make_unique<OpenPandaWidget>());
#ifdef __linux__
  if (SocketCanStream::available()) {
    widgets_.push_back(std::make_unique<OpenSocketCanWidget>());
  }
#endif
  widgets_.push_back(std::make_unique<OpenDeviceWidget>());
}

void StreamSelector::draw() {
  if (!open_) return;
  if (!beginDialog("Open Stream", &popup_, ImVec2(768.0f, 0.0f))) return;

  AbstractOpenStreamWidget *current = nullptr;
  const ImVec4 pane = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, pane);
  ImGui::PushStyleColor(ImGuiCol_TabSelected, pane);
  if (ImGui::BeginTabBar("streams")) {
    for (auto &w : widgets_) {
      // a fresh dialog every time, so the first tab is always the current one
      ImGuiTabItemFlags tab_flags = (first_frame_ && w == widgets_.front()) ? ImGuiTabItemFlags_SetSelected : 0;
      if (ImGui::BeginTabItem(w->title(), nullptr, tab_flags)) {
        current = w.get();
        ImGui::BeginChild("tab", ImVec2(0, 130.0f), ImGuiChildFlags_Borders);
        w->draw();
        ImGui::EndChild();
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
  ImGui::PopStyleColor(2);
  first_frame_ = false;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("DBC File");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-90.0f);
  inputText("##dbc", &dbc_file_, "Choose a DBC file to open", ImGuiInputTextFlags_ReadOnly);
  ImGui::SameLine();
  if (ImGui::Button("Browse...")) {
    FileDialog::getOpenFileName("Open File", settings.last_dir, ".dbc", [this](const std::string &fn) {
      if (!fn.empty()) {
        dbc_file_ = fn;
        settings.last_dir = std::filesystem::absolute(fn).parent_path().string();
      }
    });
  }
  ImGui::Separator();

  bool accepted = false, rejected = false;
  std::unique_ptr<AbstractStream> stream;
  bool open_clicked = false;
  dialogButtons("Open", &open_clicked, &rejected, current != nullptr && current->openEnabled());
  if (open_clicked) {
    if (stream = current->open(); stream) accepted = true;
  }

  // nested so they stack on this modal
  if (current) current->drawPopups();
  FileDialog::draw();
  MessageBox::draw();

  if (accepted || rejected) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (accepted || rejected) {
    open_ = false;
    widgets_.clear();
    auto on_done = std::move(on_done_);
    if (on_done) on_done(std::move(stream), dbc_file_);
  }
}
