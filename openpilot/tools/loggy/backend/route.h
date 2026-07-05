#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
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

struct SegmentLogs {
  std::string rlog;
  std::string qlog;
  std::string road_cam;
  std::string driver_cam;
  std::string wide_road_cam;
  std::string qcamera;
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

struct RouteResolveConfig {
  std::string route_name;
  std::string data_dir;
  LogSelector selector = LogSelector::Auto;
  int max_segments = -1;
};

struct RouteResolveResult {
  RouteSelection selection;
  std::vector<RouteSegment> segments;
  TimeRange route_range;
};

struct SegmentLoadOptions {
  SegmentExtractOptions extract;
  bool local_cache = true;
};

struct SegmentLoadResult {
  StoreBatch batch;
  std::vector<TimelineSpan> timeline_spans;
  std::vector<LogEntry> logs;
  size_t event_count = 0;
  size_t appended_event_count = 0;
  size_t series_count = 0;
  size_t can_message_count = 0;
  size_t timeline_span_count = 0;
  size_t log_count = 0;
  uint64_t compressed_bytes = 0;
  uint64_t decompressed_bytes = 0;
  double download_seconds = 0.0;
  double decompress_seconds = 0.0;
  double parse_seconds = 0.0;
  double extract_seconds = 0.0;
};

enum class RouteIngestState : uint8_t {
  Idle,
  Resolving,
  Loading,
  Completed,
  Failed,
  Canceled,
};

struct RouteIngestConfig {
  RouteResolveConfig resolve;
  size_t worker_count = 2;
  bool local_cache = true;
};

struct RouteIngestStatus {
  RouteIngestState state = RouteIngestState::Idle;
  std::string route_name;
  std::string error;
  TimeRange route_range;
  size_t segments_resolved = 0;
  size_t segments_loaded = 0;
  size_t segments_failed = 0;
  size_t batches_published = 0;
  double first_segment_seconds = 0.0;
  double total_seconds = 0.0;
};

RouteSelection parseRouteSelection(std::string route_name);
RouteResolveResult resolveRouteSegments(const RouteResolveConfig &config);
SegmentLoadResult loadRouteSegment(const SegmentWorkItem &work,
                                   const SegmentLoadOptions &options,
                                   std::atomic<bool> *abort = nullptr);
const char *routeIngestStateLabel(RouteIngestState state);

class RouteIngestor {
public:
  explicit RouteIngestor(SegmentScheduler *scheduler = nullptr);
  ~RouteIngestor();

  void setScheduler(SegmentScheduler *scheduler);
  void start(RouteIngestConfig config);
  void stop();
  RouteIngestStatus status() const;
  std::vector<TimelineSpan> drainTimelineSpans();
  std::vector<LogEntry> drainLogEntries();

private:
  void run(RouteIngestConfig config);
  void updateStatus(const RouteIngestStatus &status);
  void mutateStatus(void (*fn)(RouteIngestStatus *));
  void stageTimelineSpans(std::vector<TimelineSpan> spans);
  void stageLogEntries(std::vector<LogEntry> logs);

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
