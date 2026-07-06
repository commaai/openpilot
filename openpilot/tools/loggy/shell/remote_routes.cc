#include "tools/loggy/shell/remote_routes.h"

#include "tools/loggy/backend/route.h"
#include "tools/replay/py_downloader.h"

#include "json11/json11.hpp"
#include "imgui.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace loggy {
namespace {

bool modal_escape_pressed() {
  return ImGui::IsKeyPressed(ImGuiKey_Escape);
}

enum class RemoteRouteCompletionKind {
  Devices,
  Routes,
};

struct RemoteRouteCompletion {
  RemoteRouteCompletionKind kind = RemoteRouteCompletionKind::Devices;
  std::string result;
  int fetch_id = 0;
  bool preserved = false;
};

class UiCompletionQueue {
public:
  void push(RemoteRouteCompletion completion) {
    std::lock_guard lock(mutex_);
    pending_.push_back(std::move(completion));
  }
  std::vector<RemoteRouteCompletion> drain() {
    std::vector<RemoteRouteCompletion> ready;
    {
      std::lock_guard lock(mutex_);
      ready.swap(pending_);
    }
    return ready;
  }
private:
  std::mutex mutex_;
  std::vector<RemoteRouteCompletion> pending_;
};

UiCompletionQueue g_route_browser_queue;

enum class FetchState : uint8_t { Idle, Loading, Loaded, Error };

struct RemoteRouteBrowserState {
  bool active = false;
  bool open_requested = false;
  FetchState device_state = FetchState::Idle;
  std::string device_error;
  std::vector<std::string> devices;
  int selected_device = -1;
  int selected_period = 0;
  FetchState route_state = FetchState::Idle;
  std::string route_error;
  std::vector<RouteBrowserEntry> routes_;
  int selected_route = -1;
  std::atomic<int> fetch_id{0};
};

RemoteRouteBrowserState g_route_browser;

std::string route_api_error(const std::string &result, bool device_list) {
  if (result.empty()) return device_list ? "Failed to fetch devices." : "Failed to fetch routes.";
  std::string err;
  const json11::Json doc = json11::Json::parse(result, err);
  if (!err.empty()) return device_list ? "Failed to parse devices." : "Failed to parse routes.";
  if (doc.is_object() && doc["error"].is_string()) {
    const std::string code = doc["error"].string_value();
    if (code == "unauthorized") return "Unauthorized. Authenticate with openpilot/tools/lib/auth.py.";
    return device_list ? "Failed to fetch devices." : "Failed to fetch routes.";
  }
  return {};
}

void fetch_remote_routes();

void apply_remote_devices_result(const std::string &result) {
  const std::string api_error = route_api_error(result, true);
  if (!api_error.empty()) {
    g_route_browser.device_state = FetchState::Error;
    g_route_browser.device_error = api_error;
    return;
  }
  std::string err;
  const json11::Json doc = json11::Json::parse(result, err);
  g_route_browser.devices.clear();
  if (err.empty() && doc.is_array()) {
    for (const json11::Json &device : doc.array_items()) {
      std::string dongle_id = device["dongle_id"].string_value();
      if (!dongle_id.empty()) g_route_browser.devices.push_back(std::move(dongle_id));
    }
  }
  g_route_browser.device_state = FetchState::Loaded;
  g_route_browser.selected_device = g_route_browser.devices.empty() ? -1 : 0;
  if (g_route_browser.selected_device >= 0) fetch_remote_routes();
}

void fetch_remote_devices() {
  g_route_browser.device_state = FetchState::Loading;
  g_route_browser.device_error.clear();
  g_route_browser.devices.clear();
  g_route_browser.selected_device = -1;
  g_route_browser.routes_.clear();
  g_route_browser.selected_route = -1;
  g_route_browser.route_state = FetchState::Idle;
  ++g_route_browser.fetch_id;
  std::thread([]() {
    std::string result = PyDownloader::getDevices();
    RemoteRouteCompletion completion;
    completion.kind = RemoteRouteCompletionKind::Devices;
    completion.result = std::move(result);
    g_route_browser_queue.push(std::move(completion));
  }).detach();
}

void apply_remote_routes_result(const std::string &result, int fetch_id, bool preserved) {
  if (fetch_id != g_route_browser.fetch_id.load()) return;
  const RouteBrowserParseResult parsed_routes = parse_route_browser_routes(result, preserved);
  g_route_browser.routes_ = std::move(parsed_routes.first);
  if (!parsed_routes.second.empty()) {
    g_route_browser.route_state = FetchState::Error;
    g_route_browser.route_error = std::move(parsed_routes.second);
    return;
  }
  g_route_browser.route_state = FetchState::Loaded;
  g_route_browser.selected_route = g_route_browser.routes_.empty() ? -1 : 0;
}

void fetch_remote_routes() {
  if (g_route_browser.selected_device < 0 ||
      g_route_browser.selected_device >= static_cast<int>(g_route_browser.devices.size())) {
    return;
  }
  g_route_browser.routes_.clear();
  g_route_browser.selected_route = -1;
  g_route_browser.route_state = FetchState::Loading;
  g_route_browser.route_error.clear();
  const auto &periods = route_browser_periods();
  const int period_index = std::clamp(g_route_browser.selected_period, 0, static_cast<int>(periods.size()) - 1);
  const int days = periods[static_cast<size_t>(period_index)].days;
  const bool preserved = days < 0;
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  if (!preserved) {
    const auto now = std::chrono::system_clock::now();
    end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    start_ms = end_ms - static_cast<int64_t>(days) * 24LL * 60LL * 60LL * 1000LL;
  }
  const std::string dongle_id = g_route_browser.devices[static_cast<size_t>(g_route_browser.selected_device)];
  const int fetch_id = ++g_route_browser.fetch_id;
  std::thread([dongle_id, start_ms, end_ms, preserved, fetch_id]() {
    std::string result = PyDownloader::getDeviceRoutes(dongle_id, start_ms, end_ms, preserved);
    RemoteRouteCompletion completion;
    completion.kind = RemoteRouteCompletionKind::Routes;
    completion.result = std::move(result);
    completion.fetch_id = fetch_id;
    completion.preserved = preserved;
    g_route_browser_queue.push(std::move(completion));
  }).detach();
}

void drain_remote_route_completions() {
  for (RemoteRouteCompletion &completion : g_route_browser_queue.drain()) {
    switch (completion.kind) {
      case RemoteRouteCompletionKind::Devices:
        apply_remote_devices_result(completion.result);
        break;
      case RemoteRouteCompletionKind::Routes:
        apply_remote_routes_result(completion.result, completion.fetch_id, completion.preserved);
        break;
    }
  }
}

}  // namespace

void open_remote_route_browser() {
  g_route_browser.active = true;
  g_route_browser.open_requested = true;
  g_route_browser.selected_period = 0;
  fetch_remote_devices();
}

void draw_remote_route_browser(const RemoteRouteBrowserActions &actions) {
  drain_remote_route_completions();
  constexpr const char *kPopupId = "Remote Routes";
  if (g_route_browser.open_requested) {
    ImGui::OpenPopup(kPopupId);
    g_route_browser.open_requested = false;
  }
  if (!g_route_browser.active) return;
  ImGui::SetNextWindowSize(ImVec2(620.0f, 520.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
  const bool close_requested = modal_escape_pressed();
  bool open_selected = false;
  constexpr float kLabelWidth = 78.0f;
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine(kLabelWidth);
  ImGui::SetNextItemWidth(-82.0f);
  const bool devices_loading = g_route_browser.device_state == FetchState::Loading;
  const bool devices_error = g_route_browser.device_state == FetchState::Error;
  const char *device_preview = devices_loading ? "Loading..." :
                               devices_error ? "(error)" :
                               (g_route_browser.selected_device >= 0
                                  ? g_route_browser.devices[static_cast<size_t>(g_route_browser.selected_device)].c_str()
                                  : "(no devices)");
  ImGui::BeginDisabled(devices_loading || devices_error || g_route_browser.devices.empty());
  if (ImGui::BeginCombo("##remote_route_device", device_preview)) {
    for (int i = 0; i < static_cast<int>(g_route_browser.devices.size()); ++i) {
      const bool selected = i == g_route_browser.selected_device;
      if (ImGui::Selectable(g_route_browser.devices[static_cast<size_t>(i)].c_str(), selected) &&
          i != g_route_browser.selected_device) {
        g_route_browser.selected_device = i;
        fetch_remote_routes();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) fetch_remote_devices();
  if (devices_error) ImGui::TextDisabled("%s", g_route_browser.device_error.c_str());
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Period");
  ImGui::SameLine(kLabelWidth);
  ImGui::SetNextItemWidth(180.0f);
  const auto &periods = route_browser_periods();
  const int period_index = std::clamp(g_route_browser.selected_period, 0, static_cast<int>(periods.size()) - 1);
  if (ImGui::BeginCombo("##remote_route_period", periods[static_cast<size_t>(period_index)].label)) {
    for (int i = 0; i < static_cast<int>(periods.size()); ++i) {
      const bool selected = i == g_route_browser.selected_period;
      if (ImGui::Selectable(periods[static_cast<size_t>(i)].label, selected) &&
          i != g_route_browser.selected_period) {
        g_route_browser.selected_period = i;
        fetch_remote_routes();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::Spacing();
  const float bottom_height = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
  const ImVec2 list_size(0.0f, std::max(120.0f, ImGui::GetContentRegionAvail().y - bottom_height));
  if (ImGui::BeginChild("##remote_route_list", list_size, ImGuiChildFlags_Borders)) {
    if (g_route_browser.route_state == FetchState::Loading) {
      ImGui::TextDisabled("Loading...");
    } else if (g_route_browser.route_state == FetchState::Error) {
      ImGui::TextDisabled("%s", g_route_browser.route_error.c_str());
    } else if (g_route_browser.routes_.empty()) {
      ImGui::TextDisabled("No routes");
    } else {
      for (int i = 0; i < static_cast<int>(g_route_browser.routes_.size()); ++i) {
        const RouteBrowserEntry &route = g_route_browser.routes_[static_cast<size_t>(i)];
        const bool selected = i == g_route_browser.selected_route;
        if (ImGui::Selectable((route.label + "##" + route.fullname).c_str(), selected,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
          g_route_browser.selected_route = i;
          if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_selected = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", route.fullname.c_str());
      }
    }
  }
  ImGui::EndChild();
  const bool has_route = g_route_browser.selected_route >= 0 &&
                         g_route_browser.selected_route < static_cast<int>(g_route_browser.routes_.size());
  if (!has_route) ImGui::BeginDisabled();
  if (ImGui::Button("Open", ImVec2(100.0f, 0.0f))) open_selected = true;
  if (!has_route) ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || close_requested) {
    g_route_browser.active = false;
    ImGui::CloseCurrentPopup();
  }
  if (open_selected && has_route) {
    const std::string route = g_route_browser.routes_[static_cast<size_t>(g_route_browser.selected_route)].fullname;
    if (actions.open_route != nullptr) actions.open_route(actions.ctx, route);
    g_route_browser.active = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

}  // namespace loggy
