// ImGui port of tools/cabana/streams/routes.{h,cc} (RoutesDialog) and
// tools/cabana/utils/export.cc's utils::exportToCSV, both frozen Qt
// references. Wired from app.cc's menu bar ("Export to CSV...") and from
// stream_selector.cc's "Remote route..." button (another workstream's file;
// contract is the out_route pointer documented in app.h).
//
// Data source: Qt's RoutesDialog fetches devices/routes via
// tools/replay/py_downloader.h's PyDownloader::getDevices() /
// PyDownloader::getDeviceRoutes() -- a subprocess call into
// openpilot.tools.lib.file_downloader that shells out to the comma API using
// ~/.comma/auth.json for auth. That helper is already Qt-free (plain
// std::string in/out) and already linked into _cabana_imgui via replay_lib
// (jotpluggler's sketch_layout.cc uses the same module, PyDownloader::
// getRouteFiles(), to fetch a route's segment file list). So this file
// reuses PyDownloader unchanged instead of inventing a new transport; the
// only real ports are the JSON parsing (QJsonDocument -> json11, matching
// the convention dbc_menus.cc/tools/replay/route.cc already use) and the
// Qt signal/QMetaObject::invokeMethod wiring, which becomes a small
// mutex-protected callback queue drained once per frame from
// draw_route_tools() -- worker threads never touch ImGui or AppState.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"

#include "json11/json11.hpp"

#include "tools/cabana/imgui/file_dialog.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/replay/py_downloader.h"

namespace {

const ImVec4 kErrorColor(0.85f, 0.3f, 0.25f, 1.0f);
const ImVec4 kSuccessColor(0.20f, 0.6f, 0.25f, 1.0f);

// -----------------------------------------------------------------------
// A tiny async-result queue: background threads never touch ImGui/AppState.
// They compute a result and push a completion closure; draw_route_tools()
// drains and runs those closures on the UI thread once per frame. This
// replaces Qt's QMetaObject::invokeMethod(qApp, ..., Qt::QueuedConnection).
// -----------------------------------------------------------------------
class AsyncQueue {
public:
  void push(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(mutex_);
    pending_.push_back(std::move(fn));
  }
  void drain() {
    std::vector<std::function<void()>> ready;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      ready.swap(pending_);
    }
    for (auto &fn : ready) fn();
  }

private:
  std::mutex mutex_;
  std::vector<std::function<void()>> pending_;
};

AsyncQueue g_async;

// =========================================================================
// Remote route browser (RoutesDialog parity)
// =========================================================================

// Mirrors RoutesDialog's period_selector_ items exactly (label + "days"
// userdata; -1 means the /preserved endpoint).
struct PeriodOption { const char *label; int days; };
constexpr PeriodOption kPeriods[] = {
  {"Last week", 7},
  {"Last 2 weeks", 14},
  {"Last month", 30},
  {"Last 6 months", 180},
  {"Preserved", -1},
};

enum class FetchState { Idle, Loading, Loaded, Error };

struct RouteItem {
  std::string label;     // "<date>    <n>min" (RoutesDialog::parseRouteList)
  std::string fullname;  // route()'s Qt::UserRole payload; written to *out_route
};

struct RouteBrowserState {
  bool active = false;
  bool need_open_popup = false;
  std::string *out_route = nullptr;

  FetchState device_state = FetchState::Idle;
  std::string device_error;
  std::vector<std::string> devices;
  int selected_device = -1;

  int selected_period = 0;

  FetchState route_state = FetchState::Idle;
  std::string route_error;
  std::vector<RouteItem> routes;
  int selected_route = -1;

  // Mirrors RoutesDialog::fetch_id_: a monotonic request id so a route
  // response for a since-superseded device/period selection is dropped.
  std::atomic<int> fetch_id{0};
};

RouteBrowserState g_routes;

// Parses "YYYY-MM-DDTHH:MM:SS[.fff][Z]" (Qt::ISODateWithMs, used for the
// /preserved endpoint's start_time/end_time) as UTC seconds-since-epoch.
double parse_iso8601_epoch(const std::string &s) {
  std::tm tm{};
  std::istringstream iss(s);
  iss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
  if (iss.fail()) return 0.0;
#if defined(_WIN32)
  return static_cast<double>(_mkgmtime(&tm));
#else
  return static_cast<double>(timegm(&tm));
#endif
}

// Mirrors RoutesDialog::parseRouteList's item text:
// QString("%1    %2min").arg(from.toString()).arg(from.secsTo(to) / 60)
std::string format_route_label(double from_epoch_sec, double to_epoch_sec) {
  const std::time_t from_t = static_cast<std::time_t>(from_epoch_sec);
  std::tm tm_local{};
  localtime_r(&from_t, &tm_local);
  char date_buf[64] = {};
  std::strftime(date_buf, sizeof(date_buf), "%a %b %e %H:%M:%S %Y", &tm_local);
  const int minutes = static_cast<int>(std::max(0.0, to_epoch_sec - from_epoch_sec) / 60.0);
  char out[128];
  std::snprintf(out, sizeof(out), "%s    %dmin", date_buf, minutes);
  return out;
}

struct ApiResult { bool ok; std::string error; };

// Mirrors routes.cc's checkApiResponse(): PyDownloader/file_downloader.py
// reports failures as a JSON object {"error": "..."}; "unauthorized" gets
// its own message (auth.json missing/expired), anything else is a generic
// network/API error.
ApiResult check_api_response(const std::string &result) {
  if (result.empty()) return {false, "Network error"};
  std::string err;
  const json11::Json doc = json11::Json::parse(result, err);
  if (!err.empty()) return {false, "Network error"};
  if (doc.is_object() && doc["error"].is_string()) {
    const std::string code = doc["error"].string_value();
    return {false, code == "unauthorized" ? "Unauthorized. Authenticate with openpilot/tools/lib/auth.py" : "Network error"};
  }
  return {true, ""};
}

void fetch_routes();

void apply_devices_result(const std::string &result) {
  const ApiResult resp = check_api_response(result);
  if (!resp.ok) {
    g_routes.device_state = FetchState::Error;
    g_routes.device_error = resp.error;
    return;
  }

  std::string err;
  const json11::Json doc = json11::Json::parse(result, err);
  g_routes.devices.clear();
  if (err.empty() && doc.is_array()) {
    for (const auto &device : doc.array_items()) {
      std::string dongle_id = device["dongle_id"].string_value();
      if (!dongle_id.empty()) g_routes.devices.push_back(std::move(dongle_id));
    }
  }
  g_routes.device_state = FetchState::Loaded;
  g_routes.selected_device = g_routes.devices.empty() ? -1 : 0;
  if (g_routes.selected_device >= 0) fetch_routes();
}

void fetch_devices() {
  g_routes.device_state = FetchState::Loading;
  g_routes.device_error.clear();
  std::thread([]() {
    std::string result = PyDownloader::getDevices();
    g_async.push([result = std::move(result)]() { apply_devices_result(result); });
  }).detach();
}

void apply_routes_result(const std::string &result, int request_id, bool preserved) {
  if (request_id != g_routes.fetch_id.load()) return;  // superseded by a newer device/period pick

  const ApiResult resp = check_api_response(result);
  if (!resp.ok) {
    g_routes.route_state = FetchState::Error;
    // Matches RoutesDialog::parseRouteList's failure message (generic,
    // regardless of error_code -- Qt only special-cases 401 for the device list).
    g_routes.route_error = "Failed to fetch routes. Check your network connection.";
    return;
  }

  std::string err;
  const json11::Json doc = json11::Json::parse(result, err);
  g_routes.routes.clear();
  if (err.empty() && doc.is_array()) {
    for (const auto &route : doc.array_items()) {
      double from_sec = 0.0, to_sec = 0.0;
      if (preserved) {
        from_sec = parse_iso8601_epoch(route["start_time"].string_value());
        to_sec = parse_iso8601_epoch(route["end_time"].string_value());
      } else {
        from_sec = route["start_time_utc_millis"].number_value() / 1000.0;
        to_sec = route["end_time_utc_millis"].number_value() / 1000.0;
      }
      std::string fullname = route["fullname"].string_value();
      if (fullname.empty()) continue;
      g_routes.routes.push_back({format_route_label(from_sec, to_sec), std::move(fullname)});
    }
  }
  g_routes.route_state = FetchState::Loaded;
  g_routes.selected_route = g_routes.routes.empty() ? -1 : 0;
}

// Mirrors RoutesDialog::fetchRoutes(): triggered on device/period change.
void fetch_routes() {
  if (g_routes.selected_device < 0 || g_routes.selected_device >= static_cast<int>(g_routes.devices.size())) {
    return;
  }
  g_routes.routes.clear();
  g_routes.selected_route = -1;
  g_routes.route_state = FetchState::Loading;
  g_routes.route_error.clear();

  const std::string dongle_id = g_routes.devices[static_cast<size_t>(g_routes.selected_device)];
  const int period_idx = std::clamp(g_routes.selected_period, 0, static_cast<int>(std::size(kPeriods)) - 1);
  const int days = kPeriods[period_idx].days;
  const bool preserved = (days == -1);

  int64_t start_ms = 0, end_ms = 0;
  if (!preserved) {
    const auto now = std::chrono::system_clock::now();
    end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    start_ms = end_ms - static_cast<int64_t>(days) * 24 * 3600 * 1000;
  }

  const int request_id = ++g_routes.fetch_id;
  std::thread([dongle_id, start_ms, end_ms, preserved, request_id]() {
    std::string result = PyDownloader::getDeviceRoutes(dongle_id, start_ms, end_ms, preserved);
    g_async.push([result = std::move(result), request_id, preserved]() {
      apply_routes_result(result, request_id, preserved);
    });
  }).detach();
}

void draw_remote_route_browser() {
  constexpr const char *kPopupId = "Remote routes##routes_dialog";
  if (g_routes.need_open_popup) {
    ImGui::OpenPopup(kPopupId);
    g_routes.need_open_popup = false;
  }
  if (!g_routes.active) return;

  ImGui::SetNextWindowSize(ImVec2(560.0f, 480.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  bool accept_now = false;
  bool close_now = false;
  constexpr float kLabelW = 90.0f;

  // -- device combo (RoutesDialog's device_list_) --------------------------
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Device");
  ImGui::SameLine(kLabelW);
  ImGui::SetNextItemWidth(-1.0f);
  const bool device_loading = (g_routes.device_state == FetchState::Loading);
  const bool device_error = (g_routes.device_state == FetchState::Error);
  const char *device_preview =
      device_loading ? "Loading..." :
      device_error   ? "(error)" :
      (g_routes.selected_device >= 0 ? g_routes.devices[static_cast<size_t>(g_routes.selected_device)].c_str() : "(no devices)");
  ImGui::BeginDisabled(device_loading || device_error || g_routes.devices.empty());
  if (ImGui::BeginCombo("##routes_device", device_preview)) {
    for (int i = 0; i < static_cast<int>(g_routes.devices.size()); ++i) {
      const bool selected = (i == g_routes.selected_device);
      if (ImGui::Selectable(g_routes.devices[static_cast<size_t>(i)].c_str(), selected) && i != g_routes.selected_device) {
        g_routes.selected_device = i;
        fetch_routes();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (device_error) {
    ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
    ImGui::TextWrapped("%s", g_routes.device_error.c_str());
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::SmallButton("Retry##device_retry")) fetch_devices();
  }

  // -- period combo (RoutesDialog's period_selector_) -----------------------
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Period");
  ImGui::SameLine(kLabelW);
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::BeginCombo("##routes_period", kPeriods[g_routes.selected_period].label)) {
    for (int i = 0; i < static_cast<int>(std::size(kPeriods)); ++i) {
      const bool selected = (i == g_routes.selected_period);
      if (ImGui::Selectable(kPeriods[i].label, selected) && i != g_routes.selected_period) {
        g_routes.selected_period = i;
        fetch_routes();
      }
    }
    ImGui::EndCombo();
  }

  // -- route list (RoutesDialog's RouteListWidget) --------------------------
  ImGui::Spacing();
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float bottom_h = ImGui::GetFrameHeightWithSpacing();
  if (ImGui::BeginChild("##routes_list", ImVec2(0.0f, avail.y - bottom_h), ImGuiChildFlags_Borders)) {
    if (g_routes.route_state == FetchState::Loading) {
      ImGui::TextDisabled("Loading...");
    } else if (g_routes.route_state == FetchState::Error) {
      ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
      ImGui::TextWrapped("%s", g_routes.route_error.c_str());
      ImGui::PopStyleColor();
    } else if (g_routes.routes.empty()) {
      ImGui::TextDisabled("No items");
    } else {
      for (int i = 0; i < static_cast<int>(g_routes.routes.size()); ++i) {
        const bool selected = (i == g_routes.selected_route);
        if (ImGui::Selectable(g_routes.routes[static_cast<size_t>(i)].label.c_str(), selected,
                               ImGuiSelectableFlags_AllowDoubleClick)) {
          g_routes.selected_route = i;
          if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) accept_now = true;
        }
      }
    }
  }
  ImGui::EndChild();

  // -- OK / Cancel (RoutesDialog's QDialogButtonBox) -------------------------
  if (ImGui::Button("OK", ImVec2(100.0f, 0.0f))) accept_now = true;
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) close_now = true;

  if (accept_now) {
    const bool has_selection = g_routes.selected_route >= 0 && g_routes.selected_route < static_cast<int>(g_routes.routes.size());
    if (has_selection && g_routes.out_route != nullptr) {
      *g_routes.out_route = g_routes.routes[static_cast<size_t>(g_routes.selected_route)].fullname;
    }
    close_now = true;
  }
  if (close_now) {
    g_routes.active = false;
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndPopup();
}

// =========================================================================
// CSV export (utils::exportToCSV parity)
// =========================================================================

struct ExportState {
  bool active = false;
  bool need_open_popup = false;
  std::string path;
  size_t total_rows = 0;
  std::atomic<size_t> rows_written{0};
  std::atomic<bool> done{false};
  std::atomic<bool> cancel_requested{false};
  bool success = false;      // only valid once done.load() is true
  std::string error;         // only valid once done.load() is true
  double elapsed_sec = 0.0;  // only valid once done.load() is true
};

ExportState g_export;

// Runs on a worker thread. Only touches the pre-copied `events` snapshot and
// primitive/atomic state -- never AbstractStream/`can` (which the UI thread
// may be mutating via stream->update() concurrently) and never ImGui.
void export_csv_worker(std::vector<const CanEvent *> events, uint64_t begin_mono_time, std::string path) {
  const auto t0 = std::chrono::steady_clock::now();
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  bool ok = file.is_open();
  std::string error = ok ? std::string() : ("Failed to open " + path + " for writing");

  if (ok) {
    static constexpr char kHexDigits[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(1 << 20);
    out += "time,addr,bus,data\n";  // utils::exportToCSV's header, verbatim

    size_t written = 0;
    char head[64];
    for (const CanEvent *e : events) {
      if (g_export.cancel_requested.load(std::memory_order_relaxed)) break;

      // AbstractStream::toSeconds(), evaluated with signed arithmetic against
      // a snapshot of beginMonoTime() instead of calling into the live
      // stream object from this thread.
      const double sec = std::max(0.0, static_cast<double>(static_cast<int64_t>(e->mono_time) -
                                                             static_cast<int64_t>(begin_mono_time)) / 1e9);
      const int n = std::snprintf(head, sizeof(head), "%.3f,0x%x,%d,0x", sec, static_cast<unsigned>(e->address),
                                   static_cast<int>(e->src));
      out.append(head, static_cast<size_t>(n));
      for (int b = 0; b < e->size; ++b) {
        const uint8_t v = e->dat[b];
        out += kHexDigits[v >> 4];
        out += kHexDigits[v & 0x0F];
      }
      out += '\n';
      ++written;

      if (out.size() >= (1 << 20)) {
        file.write(out.data(), static_cast<std::streamsize>(out.size()));
        out.clear();
        g_export.rows_written.store(written, std::memory_order_relaxed);
      }
    }
    if (!out.empty()) file.write(out.data(), static_cast<std::streamsize>(out.size()));
    file.flush();
    g_export.rows_written.store(written, std::memory_order_relaxed);

    ok = !file.fail();
    if (!ok) {
      error = "Write error while exporting " + path;
    } else if (g_export.cancel_requested.load(std::memory_order_relaxed)) {
      ok = false;
      error = "Cancelled";
    }
  }
  file.close();

  // Plain members written here happen-before the UI thread's read of them
  // because both sides order around the `done` atomic (release here,
  // acquire/seq_cst load in draw_export_progress()).
  g_export.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  g_export.error = std::move(error);
  g_export.success = ok;
  g_export.done.store(true, std::memory_order_release);
}

void start_csv_export(const std::string &path) {
  // Snapshot the event list on the UI thread: all_events_ can be mutated by
  // AbstractStream::mergeEvents() (called from stream->update(), UI thread
  // only) while the export runs, so the worker gets its own copy of the
  // pointer vector up front. CanEvent contents are immutable after
  // AbstractStream::newEvent() constructs them (arena-allocated, never
  // freed/moved), so copying the pointers is enough -- no need to copy the
  // underlying event bytes.
  std::vector<const CanEvent *> events(can->allEvents().begin(), can->allEvents().end());
  const uint64_t begin_mono_time = can->beginMonoTime();

  g_export.path = path;
  g_export.total_rows = events.size();
  g_export.rows_written.store(0, std::memory_order_relaxed);
  g_export.done.store(false, std::memory_order_relaxed);
  g_export.cancel_requested.store(false, std::memory_order_relaxed);
  g_export.success = false;
  g_export.error.clear();
  g_export.elapsed_sec = 0.0;
  g_export.active = true;
  g_export.need_open_popup = true;

  std::thread([events = std::move(events), begin_mono_time, path]() mutable {
    export_csv_worker(std::move(events), begin_mono_time, path);
  }).detach();
}

void draw_export_progress() {
  constexpr const char *kPopupId = "Export to CSV##export_csv_progress";
  if (g_export.need_open_popup) {
    ImGui::OpenPopup(kPopupId);
    g_export.need_open_popup = false;
  }
  if (!g_export.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) return;

  ImGui::TextWrapped("%s", g_export.path.c_str());
  ImGui::Spacing();

  const bool done = g_export.done.load(std::memory_order_acquire);
  const size_t written = g_export.rows_written.load(std::memory_order_relaxed);
  if (!done) {
    const float frac = g_export.total_rows > 0 ? static_cast<float>(written) / static_cast<float>(g_export.total_rows) : 0.0f;
    ImGui::ProgressBar(frac, ImVec2(320.0f, 0.0f));
    ImGui::Text("%zu / %zu events", written, g_export.total_rows);
    if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f))) {
      g_export.cancel_requested.store(true, std::memory_order_relaxed);
    }
  } else {
    if (g_export.success) {
      ImGui::PushStyleColor(ImGuiCol_Text, kSuccessColor);
      ImGui::Text("Export complete: %zu rows in %.2fs", written, g_export.elapsed_sec);
      ImGui::PopStyleColor();
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
      ImGui::TextWrapped("Export failed: %s", g_export.error.c_str());
      ImGui::PopStyleColor();
    }
    if (ImGui::Button("Close", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      g_export.active = false;
      ImGui::CloseCurrentPopup();
    }
  }

  ImGui::EndPopup();
}

}  // namespace

void draw_route_tools(AppState &app) {
  (void)app;
  g_async.drain();
  draw_remote_route_browser();
  draw_export_progress();
}

void open_remote_route_browser(std::string *out_route) {
  g_routes.out_route = out_route;
  g_routes.active = true;
  g_routes.need_open_popup = true;

  g_routes.device_state = FetchState::Loading;
  g_routes.device_error.clear();
  g_routes.devices.clear();
  g_routes.selected_device = -1;
  g_routes.selected_period = 0;

  g_routes.route_state = FetchState::Idle;
  g_routes.route_error.clear();
  g_routes.routes.clear();
  g_routes.selected_route = -1;
  ++g_routes.fetch_id;  // invalidate any in-flight route fetch from a previous open

  fetch_devices();
}

void open_export_csv() {
  // has_stream(AppState&) (app.h) is what gates the "Export to CSV..." menu
  // item in app.cc, but this free function -- mirroring mainwin.cc's
  // exportToCSV(), likewise only reachable from that menu item -- isn't
  // handed an AppState. `can` is the same pointer has_stream() dereferences
  // (app.cc: `can = app.stream.get();`), so re-check the same condition
  // directly against it as a defensive no-op guard.
  if (can == nullptr || dynamic_cast<DummyStream *>(can) != nullptr) return;

  // Mirrors MainWindow::exportToCSV()'s default path:
  // QString("%1/%2.csv").arg(settings.last_dir).arg(routeName())
  const std::string default_name = can->routeName() + ".csv";
  file_dialog_open(FileDialogMode::Save, "Export stream to CSV file", settings.last_dir, ".csv", default_name,
                    [](const std::string &path) { start_csv_export(path); });
}
