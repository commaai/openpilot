#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <optional>
#include <string_view>
#include <mutex>
#include <string>
#include <utility>
#include <thread>
#include <vector>

#include "tools/loggy/backend/extract.h"
#include "tools/loggy/backend/ingest.h"

namespace loggy {

enum class LogSelector : uint8_t {
  Auto,
  RLog,
  QLog,
};

struct RouteBrowserPeriod {
  const char *label;
  int days;
};

struct RouteBrowserEntry {
  std::string label;
  std::string fullname;
};

using RouteBrowserParseResult = std::pair<std::vector<RouteBrowserEntry>, std::string>;

struct RouteSelection {
  std::string dongle_id;
  std::string timestamp;
  int begin_segment = 0;
  int end_segment = -1;
  bool slice_explicit = false;
  LogSelector selector = LogSelector::Auto;
  bool selector_explicit = false;
  std::string canonical_name;
};

enum class LogOrigin : uint8_t {
  Log,
  OperatingSystem,
  Alert,
};

struct LogEntry {
  double mono_time = 0.0;
  double boot_time = 0.0;
  double wall_time = 0.0;
  uint8_t level = 20;
  std::string source;
  std::string func;
  std::string message;
  std::string context;
  LogOrigin origin = LogOrigin::Log;
};

// Canonical log/timeline extraction — shared by route.cc's batch ingest and live.cc's
// streaming ingest, so it lives here rather than duplicated per call site.
void append_timeline_point(std::vector<TimelineSpan> &spans, double mono_time, TimelineSpanKind kind);
TimelineSpanKind timeline_kind_for_selfdrive(cereal::SelfdriveState::AlertStatus status, bool enabled);
void append_log_event(cereal::Event::Which which, const cereal::Event::Reader &event,
                     double time_offset, std::vector<LogEntry> &logs, std::string &last_alert_key);

// Repo-root resolution shared by live/session/computed ingest paths.
std::filesystem::path repo_root_path();

using RouteSliceSpec = std::pair<int, int>;

enum class RouteIngestState : uint8_t {
  Idle,
  Resolving,
  Loading,
  Completed,
  Failed,
  Canceled,
};

struct RouteIngestConfig {
  std::string route_name;
  std::string data_dir;
  LogSelector selector = LogSelector::Auto;
  int max_segments = -1;
  size_t worker_count = 2;
  bool local_cache = true;
};

struct RouteIngestStatus {
  RouteIngestState state = RouteIngestState::Idle;
  std::string route_name;
  RouteSelection selection;
  std::string car_fingerprint;
  std::string error;
  TimeRange route_range;
  size_t segments_resolved = 0;
  size_t segments_loaded = 0;
  size_t segments_failed = 0;
  size_t batches_published = 0;
  double first_segment_seconds = 0.0;
  double total_seconds = 0.0;
};

const std::array<RouteBrowserPeriod, 5> &route_browser_periods();
std::string route_browser_device_routes_url(const std::string &dongle_id,
                                       int64_t start_ms = 0,
                                       int64_t end_ms = 0,
                                       bool preserved = false);
std::string route_browser_route_files_url(const std::string &route_name);
std::string route_browser_route_label(double from_epoch_sec, double to_epoch_sec);
RouteBrowserParseResult parse_route_browser_routes(const std::string &json_text,
                                               bool preserved);
std::optional<RouteSelection> route_selection_from_text(std::string_view route_name);

RouteSelection parse_route_selection(std::string route_name);
const char *log_selector_name(LogSelector selector);
const char *log_selector_description(LogSelector selector);
char log_selector_char(LogSelector selector);
std::string route_selection_display_slice(const RouteSelection &selection);
std::string route_selection_full_spec(const RouteSelection &selection);
std::string route_useradmin_url(const RouteSelection &selection);
std::string route_connect_url(const RouteSelection &selection);
std::optional<RouteSliceSpec> parse_route_slice_spec(std::string_view text);
const char *route_ingest_state_label(RouteIngestState state);

class RouteIngestor {
public:
  explicit RouteIngestor(SegmentScheduler *scheduler = nullptr);
  ~RouteIngestor();

  void set_scheduler(SegmentScheduler *scheduler);
  void start(RouteIngestConfig config);
  void stop();
  RouteIngestStatus status() const;
  std::vector<TimelineSpan> drain_timeline_spans();
  std::vector<LogEntry> drain_log_entries();

private:
  void run(RouteIngestConfig config);
  void update_status(const RouteIngestStatus &status);
  void mutate_status(void (*fn)(RouteIngestStatus *));
  void stage_timeline_spans(std::vector<TimelineSpan> spans);
  void stage_log_entries(std::vector<LogEntry> logs);

  SegmentScheduler *scheduler_ = nullptr;
  std::thread thread_;
  std::atomic<bool> abort_{false};
  mutable std::mutex status_mutex_;
  RouteIngestStatus status_;
  mutable std::mutex timeline_mutex_;
  std::vector<TimelineSpan> staged_timeline_spans_;
  mutable std::mutex logs_mutex_;
  std::vector<LogEntry> staged_logs_;
};

}  // namespace loggy
