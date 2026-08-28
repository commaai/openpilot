#include "tools/cabana/ui/dialogs/streamselector.h"

#include <filesystem>
#include <fstream>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

// OpenReplayWidget

OpenReplayWidget::OpenReplayWidget() {}

void OpenReplayWidget::draw() {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Route");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-250.0f);
  inputText("##route", &route_, "Enter route name or browse for local/remote route");
  ImGui::SameLine();
  if (ImGui::Button("Remote route...")) {
    routes_dialog_.open([this, alive = std::weak_ptr<bool>(alive_)](bool accepted, const std::string &route) {
      if (!alive.expired() && accepted) route_ = route;
    });
  }
  ImGui::SameLine();
  if (ImGui::Button("Local route...")) {
    FileDialog::getExistingDirectory("Open Local Route", settings.last_route_dir, [this, alive = std::weak_ptr<bool>(alive_)](const std::string &dir) {
      if (!alive.expired() && !dir.empty()) {
        route_ = dir;
        settings.last_route_dir = std::filesystem::absolute(dir).parent_path().string();
      }
    });
  }
  ImGui::Checkbox("Road camera", &cameras_[0]);
  ImGui::SameLine();
  ImGui::Checkbox("Driver camera", &cameras_[1]);
  ImGui::SameLine();
  ImGui::Checkbox("Wide road camera", &cameras_[2]);
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

// OpenPandaWidget

static const uint32_t speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U};
static const uint32_t data_speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U, 2000U, 5000U};

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
        if (data_speeds[j] == config.bus_config[i].can_speed_kbps) can_speed_index_[i] = j;
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
    ImGui::TextUnformatted("Close the current connection via [File menu -> Close Stream] before connecting to another Panda.");
    return;
  }
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Serial");
  ImGui::SameLine();
  std::string items;
  for (const auto &s : serials_) items += s + '\0';
  ImGui::SetNextItemWidth(-100.0f);
  if (ImGui::Combo("##serial", &serial_index_, items.c_str())) buildConfigForm();
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
    std::string speed_items;
    for (uint32_t s : speeds) speed_items += std::to_string(s) + '\0';
    if (ImGui::Combo("##can_speed", &can_speed_index_[i], speed_items.c_str())) {
      config.bus_config[i].can_speed_kbps = speeds[can_speed_index_[i]];
    }
    if (has_fd_) {
      ImGui::SameLine();
      ImGui::Checkbox("CAN-FD", &config.bus_config[i].can_fd);
      ImGui::SameLine();
      ImGui::TextUnformatted("Data Speed (kbps):");
      ImGui::SameLine();
      ImGui::BeginDisabled(!config.bus_config[i].can_fd);
      ImGui::SetNextItemWidth(90.0f);
      std::string data_items;
      for (uint32_t s : data_speeds) data_items += std::to_string(s) + '\0';
      if (ImGui::Combo("##data_speed", &data_speed_index_[i], data_items.c_str())) {
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

// OpenDeviceWidget

namespace {

struct IpInputContext {
  std::string *text;
  const std::string *last_valid;
};

// QRegExpValidator equivalent: only digits and dots get in, and an edit that makes the address Invalid is refused
int ipInputCallback(ImGuiInputTextCallbackData *data) {
  auto *ctx = static_cast<IpInputContext *>(data->UserData);
  if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
    const ImWchar c = data->EventChar;
    return ((c >= '0' && c <= '9') || c == '.') ? 0 : 1;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackEdit) {
    if (validateIpAddress(std::string(data->Buf, data->BufTextLen)) == ValidState::Invalid) {
      data->DeleteChars(0, data->BufTextLen);
      data->InsertChars(0, ctx->last_valid->c_str());
    }
  } else if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    ctx->text->resize(data->BufTextLen);
    data->Buf = ctx->text->data();
  }
  return 0;
}

}  // namespace

void OpenDeviceWidget::draw() {
  ImGui::RadioButton("MSGQ", &mode_, 0);
  ImGui::RadioButton("ZMQ", &mode_, 1);
  ImGui::SameLine();
  ImGui::BeginDisabled(mode_ != 1);
  ImGui::SetNextItemWidth(-1.0f);
  const std::string prev = ip_address_;
  IpInputContext ctx{&ip_address_, &prev};
  ImGui::InputTextWithHint("##ip", "Enter device Ip Address", ip_address_.data(), ip_address_.capacity() + 1,
                           ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackEdit | ImGuiInputTextFlags_CallbackResize,
                           ipInputCallback, &ctx);
  ImGui::EndDisabled();
}

std::unique_ptr<AbstractStream> OpenDeviceWidget::open() {
  std::string ip = ip_address_.empty() ? "127.0.0.1" : ip_address_;
  bool msgq = mode_ == 0;
  return std::make_unique<DeviceStream>(msgq ? "" : ip);
}

#ifdef __linux__
// OpenSocketCanWidget

OpenSocketCanWidget::OpenSocketCanWidget() {
  refreshDevices();
}

void OpenSocketCanWidget::refreshDevices() {
  devices_.clear();
  // Scan /sys/class/net/ for CAN interfaces (type 280 = ARPHRD_CAN)
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
  std::string items;
  for (const auto &d : devices_) items += d + '\0';
  ImGui::SetNextItemWidth(300.0f);
  if (ImGui::Combo("##device", &device_index_, items.c_str())) config.device = devices_[device_index_];
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

// StreamSelector

void StreamSelector::open(Callback on_done) {
  on_done_ = std::move(on_done);
  open_ = true;
  show_ = false;
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
  if (!show_) {
    ImGui::OpenPopup("Open stream");
    show_ = true;
  }
  ImGui::SetNextWindowSize(ImVec2(640.0f, 0.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal("Open stream", nullptr, ImGuiWindowFlags_NoResize)) return;

  AbstractOpenStreamWidget *current = nullptr;
  if (ImGui::BeginTabBar("streams")) {
    for (auto &w : widgets_) {
      if (ImGui::BeginTabItem(w->title())) {
        current = w.get();
        ImGui::BeginChild("tab", ImVec2(0, 130.0f));
        w->draw();
        ImGui::EndChild();
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("dbc File");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-90.0f);
  inputText("##dbc", &dbc_file_, "Choose a dbc file to open", ImGuiInputTextFlags_ReadOnly);
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
  ImGui::BeginDisabled(current == nullptr || !current->openEnabled());
  if (ImGui::Button("Open", ImVec2(80.0f, 0.0f))) {
    if (stream = current->open(); stream) accepted = true;
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f))) rejected = true;
  if (ImGui::GetTopMostPopupModal() == ImGui::GetCurrentWindow() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) rejected = true;

  // nested so they stack on this modal
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
