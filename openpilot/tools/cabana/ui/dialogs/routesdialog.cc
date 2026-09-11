#include "tools/cabana/ui/dialogs/routesdialog.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <thread>

#include "json11/json11.hpp"
#include "tools/replay/py_downloader.h"
#include "tools/cabana/ui/theme.h"

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

  fetchDevices();
}

void RoutesDialog::fetchDevices() {
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
  } else if (error_code == 401) {
    s_.login = true;
  } else {
    // the box shows on top of the dialog, which is rejected once the box is dismissed
    MessageBox::warning("Error", "Network error", "",
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
  if (auth_abort_) *auth_abort_ = true;
  auth_abort_.reset();
  alive_.reset();
  open_ = false;
  auto on_done = std::move(on_done_);
  if (on_done) on_done(accepted, accepted && s_.route_index >= 0 ? s_.routes[s_.route_index].name : "");
}

void RoutesDialog::draw() {
  if (!open_) return;
  if (!popup_.begin("Remote Routes")) return;
  setNextDialogWindow(ImVec2(480.0f, 480.0f));
  if (s_.login) {
    ImGui::SetNextWindowSizeConstraints(ImVec2(480, 0), ImVec2(480, FLT_MAX));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  }
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoSavedSettings | (s_.login ? ImGuiWindowFlags_AlwaysAutoResize : 0);
  if (!ImGui::BeginPopupModal("Remote Routes", nullptr, flags)) return;

  if (s_.login) {
    drawLogin();
    ImGui::EndPopup();
    return;
  }

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

void RoutesDialog::signIn(const std::string &provider) {
  s_.provider = provider == "google" ? "Google" : provider == "apple" ? "Apple" : "GitHub";
  s_.auth_error.clear();
  auth_abort_ = std::make_shared<std::atomic<bool>>(false);
  std::thread([this, alive = std::weak_ptr<bool>(alive_), abort = auth_abort_, provider]() {
    const std::string result = PyDownloader::authenticate(provider, abort.get());
    utils::runOnMainThread(utils::guarded(alive.lock(), [this, abort, result]() {
      if (*abort) return;
      auth_abort_.reset();
      std::string error;
      auto status = json11::Json::parse(result, error);
      if (status["success"].bool_value()) {
        s_.login = false;
        fetchDevices();
      } else {
        s_.auth_error = status["error"].string_value();
        if (s_.auth_error.empty()) s_.auth_error = "Could not start sign-in. Please try again.";
      }
    }));
  }).detach();
}

void RoutesDialog::drawLogin() {
  const auto &p = palette();
  ImGui::Spacing();
  ImGui::Indent(16);
  ImGui::PushFont(boldFont(), 28.0f);
  ImGui::TextUnformatted(auth_abort_ ? "Finish signing in" : "Open your routes in Cabana");
  ImGui::PopFont();
  ImGui::Spacing();
  ImGui::PushTextWrapPos(ImGui::GetWindowWidth() - 28);
  if (auth_abort_) {
    ImGui::TextWrapped("Sign in with %s in your browser, then return to Cabana to choose a device and route.", s_.provider.c_str());
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::BeginChild("auth_status", ImVec2(-16, 76), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
    const char *status = "Waiting for browser sign-in";
    const char *timeout = "This request expires after 3 minutes.";
    const float spinner_size = ImGui::GetFontSize();
    const float status_width = spinner_size + 8 + ImGui::CalcTextSize(status).x;
    ImGui::SetCursorPos(ImVec2((ImGui::GetWindowWidth() - status_width) * 0.5f, 16));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float angle = std::fmod(ImGui::GetTime() * 4.0, 2.0 * IM_PI);
    auto *draw_list = ImGui::GetWindowDrawList();
    draw_list->PathArcTo(ImVec2(pos.x + spinner_size * 0.5f, pos.y + spinner_size * 0.5f),
                        spinner_size * 0.35f, angle, angle + IM_PI * 1.5f, 24);
    draw_list->PathStroke(ImGui::GetColorU32(p.accent), 0, 2.0f);
    ImGui::Dummy(ImVec2(spinner_size, spinner_size));
    ImGui::SameLine(0, 8);
    ImGui::TextColored(p.accent, "%s", status);
    ImGui::SetCursorPos(ImVec2((ImGui::GetWindowWidth() - ImGui::CalcTextSize(timeout).x) * 0.5f, 40));
    ImGui::TextDisabled("%s", timeout);
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8);
    if (ImGui::Button("Choose another method", ImVec2(-16, 44))) {
      *auth_abort_ = true;
      auth_abort_.reset();
    }
    ImGui::PopStyleVar();
  } else {
    ImGui::TextWrapped("Use your comma account to browse recorded drives and open a route for analysis.");
    ImGui::Dummy(ImVec2(0, 12));
    const char *providers[] = {"Google", "Apple", "GitHub"};
    const char *methods[] = {"google", "apple", "github"};
    const char *icons[] = {"\xef\x8f\xb0", "\xef\x99\x9b", "\xef\x8f\xad"};
    const float icon_width = ImGui::GetFontSize() * 1.5f;
    const float gap = ImGui::GetFontSize() * 0.75f;
    float text_width = 0;
    for (const char *provider : providers) {
      text_width = std::max(text_width, ImGui::CalcTextSize((std::string("Sign in with ") + provider).c_str()).x);
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8);
    for (int i = 0; i < 3; ++i) {
      const std::string label = std::string("Sign in with ") + providers[i];
      if (ImGui::Button((std::string("##auth_") + methods[i]).c_str(), ImVec2(-16, 44))) signIn(methods[i]);
      const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
      // Center a shared two-column block, keeping every logo and label aligned.
      const float left = min.x + std::max(ImGui::GetStyle().FramePadding.x, (max.x - min.x - icon_width - gap - text_width) * 0.5f);
      const float top = min.y + (max.y - min.y - ImGui::GetFontSize()) * 0.5f;
      auto *draw_list = ImGui::GetWindowDrawList();
      const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
      draw_list->AddText(ImVec2(left + (icon_width - ImGui::CalcTextSize(icons[i]).x) * 0.5f, top), color, icons[i]);
      draw_list->AddText(ImVec2(left + icon_width + gap, top), color, label.c_str());
      ImGui::Spacing();
    }
    ImGui::PopStyleVar();
    if (!s_.auth_error.empty()) {
      ImGui::TextWrapped("%s", s_.auth_error.c_str());
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, p.text_disabled);
      const char *hint = "Use the comma account paired with your device.";
      const float width = ImGui::GetContentRegionAvail().x - 16;
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (width - ImGui::CalcTextSize(hint).x) * 0.5f));
      ImGui::TextWrapped("%s", hint);
      ImGui::PopStyleColor();
    }
  }
  ImGui::PopTextWrapPos();
  ImGui::Unindent(16);
  ImGui::Dummy(ImVec2(0, 12));
  ImGui::Separator();
  ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 92);
  bool rejected = ImGui::Button("Cancel", ImVec2(80, 0)) || dialogEscapePressed();
  if (rejected) {
    ImGui::CloseCurrentPopup();
    finish(false);
  }
}
