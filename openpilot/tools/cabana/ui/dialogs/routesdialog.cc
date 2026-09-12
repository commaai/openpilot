#include "tools/cabana/ui/dialogs/routesdialog.h"

#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/util.h"

namespace {
const char *PERIOD_NAMES[] = {"Last week", "Last 2 weeks", "Last month", "Last 6 months", "Preserved"};
const int PERIOD_DAYS[] = {7, 14, 30, 180, -1};
}  // namespace

void RoutesDialog::open(std::function<void(bool, const std::string &)> on_done) {
  on_done_ = std::move(on_done);
  open_ = true;
  popup_.reset();
  s_ = State{};
  alive_ = std::make_shared<bool>(true);

  routes::fetchDevices([this, alive = std::weak_ptr<bool>(alive_)](std::vector<routes::DeviceInfo> devices, bool success, int error_code) {
    utils::runOnMainThread(utils::guarded(alive.lock(), [this, devices = std::move(devices), success, error_code]() {
      setDeviceList(devices, success, error_code);
    }));
  });
}

void RoutesDialog::setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code) {
  if (success) {
    s_.devices.clear();
    for (const auto &device : devices) s_.devices.push_back(device.dongle_id);
    s_.devices_loaded = true;
    s_.device_index = 0;
    fetchRoutes();
  } else {
    // the box shows on top of the dialog, which is rejected once the box is dismissed
    MessageBox::warning("Error", error_code == 401 ? "Unauthorized. Authenticate with openpilot/tools/lib/auth.py" : "Network error", "",
                        utils::guarded(alive_, [this]() { finish(false); }));
  }
}

void RoutesDialog::fetchRoutes() {
  if (!s_.devices_loaded || s_.devices.empty()) return;

  s_.routes.clear();
  s_.route_index = -1;
  s_.empty_text = "Loading...";

  const int request_id = ++s_.fetch_id;
  auto on_routes = [this, alive = std::weak_ptr<bool>(alive_), request_id](std::vector<routes::RouteInfo> list, bool success, int) {
    utils::runOnMainThread(utils::guarded(alive.lock(), [this, list = std::move(list), success, request_id]() {
      if (s_.fetch_id == request_id) setRouteList(list, success);
    }));
  };
  routes::fetchRoutes(s_.devices[s_.device_index], PERIOD_DAYS[s_.period_index], std::move(on_routes));
}

void RoutesDialog::setRouteList(const std::vector<routes::RouteInfo> &list, bool success) {
  if (success) {
    for (const auto &route : list) {
      const int mins = static_cast<int>((route.end_ms - route.start_ms) / 60000);
      s_.routes.push_back({routes::formatUnixMs(route.start_ms) + "    " + std::to_string(mins) + " min", route.name});
    }
    if (!s_.routes.empty()) s_.route_index = 0;
  } else {
    MessageBox::warning("Error", "Failed to fetch routes. Check your network connection.", "",
                        utils::guarded(alive_, [this]() { finish(false); }));
  }
  s_.empty_text = "No items";
}

void RoutesDialog::finish(bool accepted) {
  alive_.reset();
  open_ = false;
  auto on_done = std::move(on_done_);
  if (on_done) on_done(accepted, accepted && s_.route_index >= 0 ? s_.routes[s_.route_index].name : "");
}

void RoutesDialog::draw() {
  if (!open_) return;
  if (!beginDialog("Remote Routes", &popup_, ImVec2(480.0f, 420.0f))) return;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (s_.devices_loaded) {
    if (comboBox("##device", &s_.device_index, s_.devices)) fetchRoutes();
  } else {
    int idx = 0;
    ImGui::BeginDisabled();
    comboBox("##device", &idx, {"Loading..."});
    ImGui::EndDisabled();
  }
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::Combo("##period", &s_.period_index, PERIOD_NAMES, IM_ARRAYSIZE(PERIOD_NAMES))) fetchRoutes();

  bool accepted = false, rejected = false;
  const float footer = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
  ImGui::BeginChild("routes", ImVec2(0, -footer), ImGuiChildFlags_Borders);
  if (s_.routes.empty()) {
    const ImVec2 size = ImGui::CalcTextSize(s_.empty_text.c_str());
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::SetCursorPos(ImVec2((avail.x - size.x) * 0.5f, (avail.y - size.y) * 0.5f));
    ImGui::TextUnformatted(s_.empty_text.c_str());
  }
  for (int i = 0; i < static_cast<int>(s_.routes.size()); ++i) {
    ImGui::PushID(i);
    if (ImGui::Selectable(s_.routes[i].label.c_str(), s_.route_index == i, ImGuiSelectableFlags_AllowDoubleClick)) {
      s_.route_index = i;
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) accepted = true;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();

  dialogButtons("OK", &accepted, &rejected);
  MessageBox::draw();
  if (accepted || rejected || !open_) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (accepted || rejected) finish(accepted);
}
