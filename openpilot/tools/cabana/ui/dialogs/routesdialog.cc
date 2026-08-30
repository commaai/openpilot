#include "tools/cabana/ui/dialogs/routesdialog.h"

#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

namespace {
const char *PERIOD_NAMES[] = {"Last week", "Last 2 weeks", "Last month", "Last 6 months", "Preserved"};
const int PERIOD_DAYS[] = {7, 14, 30, 180, -1};
}  // namespace

void RoutesDialog::open(std::function<void(bool, const std::string &)> on_done) {
  on_done_ = std::move(on_done);
  open_ = true;
  show_ = false;
  devices_loaded_ = false;
  devices_.clear();
  device_index_ = 0;
  period_index_ = 0;
  routes_.clear();
  route_index_ = -1;
  empty_text_ = "No items";
  alive_ = std::make_shared<bool>(true);

  routes::fetchDevices([this, alive = std::weak_ptr<bool>(alive_)](std::vector<routes::DeviceInfo> devices, bool success, int error_code) {
    utils::runOnMainThread([this, alive, devices = std::move(devices), success, error_code]() {
      if (!alive.expired()) setDeviceList(devices, success, error_code);
    });
  });
}

void RoutesDialog::setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code) {
  if (success) {
    devices_.clear();
    for (const auto &device : devices) devices_.push_back(device.dongle_id);
    devices_loaded_ = true;
    device_index_ = 0;
    fetchRoutes();
  } else {
    // the box shows on top of the dialog, which is rejected once the box is dismissed
    MessageBox::warning("Error", error_code == 401 ? "Unauthorized. Authenticate with openpilot/tools/lib/auth.py" : "Network error", "",
                        [this, alive = std::weak_ptr<bool>(alive_)]() {
                          if (!alive.expired()) finish(false);
                        });
  }
}

void RoutesDialog::fetchRoutes() {
  if (!devices_loaded_ || devices_.empty()) return;

  routes_.clear();
  route_index_ = -1;
  empty_text_ = "Loading...";

  int request_id = ++fetch_id_;
  auto on_routes = [this, alive = std::weak_ptr<bool>(alive_), request_id](std::vector<routes::RouteInfo> list, bool success, int) {
    utils::runOnMainThread([this, alive, list = std::move(list), success, request_id]() {
      if (!alive.expired() && fetch_id_ == request_id) setRouteList(list, success);
    });
  };
  routes::fetchRoutes(devices_[device_index_], PERIOD_DAYS[period_index_], std::move(on_routes));
}

void RoutesDialog::setRouteList(const std::vector<routes::RouteInfo> &list, bool success) {
  if (success) {
    for (const auto &route : list) {
      const int mins = static_cast<int>((route.end_ms - route.start_ms) / 60000);
      routes_.push_back({routes::formatUnixMs(route.start_ms) + "    " + std::to_string(mins) + "min", route.name});
    }
    if (!routes_.empty()) route_index_ = 0;
  } else {
    MessageBox::warning("Error", "Failed to fetch routes. Check your network connection.", "",
                        [this, alive = std::weak_ptr<bool>(alive_)]() {
                          if (!alive.expired()) finish(false);
                        });
  }
  empty_text_ = "No items";
}

void RoutesDialog::finish(bool accepted) {
  alive_.reset();
  open_ = false;
  auto on_done = std::move(on_done_);
  if (on_done) on_done(accepted, accepted && route_index_ >= 0 ? routes_[route_index_].name : "");
}

void RoutesDialog::draw() {
  if (!open_) return;
  if (!beginDialog("Remote routes", &show_, ImVec2(480.0f, 420.0f))) return;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (!devices_loaded_) {
    int idx = 0;
    ImGui::Combo("##device", &idx, "Loading...\0", 1);
  } else {
    std::string items;
    for (const auto &d : devices_) items += d + '\0';
    if (ImGui::Combo("##device", &device_index_, items.c_str())) fetchRoutes();
  }
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::Combo("##period", &period_index_, PERIOD_NAMES, IM_ARRAYSIZE(PERIOD_NAMES))) fetchRoutes();

  bool accepted = false, rejected = false;
  const float footer = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
  ImGui::BeginChild("routes", ImVec2(0, -footer), ImGuiChildFlags_Borders);
  if (routes_.empty()) {
    const ImVec2 size = ImGui::CalcTextSize(empty_text_.c_str());
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::SetCursorPos(ImVec2((avail.x - size.x) * 0.5f, (avail.y - size.y) * 0.5f));
    ImGui::TextUnformatted(empty_text_.c_str());
  }
  for (int i = 0; i < static_cast<int>(routes_.size()); ++i) {
    ImGui::PushID(i);
    if (ImGui::Selectable(routes_[i].label.c_str(), route_index_ == i, ImGuiSelectableFlags_AllowDoubleClick)) {
      route_index_ = i;
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) accepted = true;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();

  dialogButtons("OK", &accepted, &rejected);
  if (dialogEscapePressed()) rejected = true;
  MessageBox::draw();
  if (accepted || rejected || !open_) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (accepted || rejected) finish(accepted);
}
