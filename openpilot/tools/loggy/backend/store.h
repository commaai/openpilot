#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/shell/transport.h"

namespace loggy {

bool intersects(TimeRange a, TimeRange b);
TimeRange intersection(TimeRange a, TimeRange b);
double distance(TimeRange a, TimeRange b);
double distance_to_point(TimeRange range, double point);
std::vector<TimeRange> normalize_ranges(std::vector<TimeRange> ranges);

struct CoverageInfo {
  TimeRange requested;
  std::vector<TimeRange> ranges;
  double covered_seconds = 0.0;
  bool complete = false;
};

struct SeriesPoint {
  double t = 0.0;
  double value = 0.0;
};

struct SeriesChunk {
  std::string path;
  TimeRange range;
  std::vector<SeriesPoint> points;
  int segment = -1;
};

struct SeriesView {
  std::string path;
  TimeRange requested;
  CoverageInfo coverage;
  std::vector<SeriesPoint> points;
  size_t total_points = 0;
  bool decimated = false;
};

struct CanEvent {
  double mono_time = 0.0;
  uint16_t bus_time = 0;
  std::vector<uint8_t> data;
};

struct CanEventChunk {
  MessageId id;
  TimeRange range;
  std::vector<CanEvent> events;
  int segment = -1;
};

struct CanEventView {
  MessageId id;
  TimeRange requested;
  CoverageInfo coverage;
  std::vector<CanEvent> events;
};

struct CanSummaryView {
  MessageId id;
  TimeRange requested;
  CoverageInfo coverage;
  size_t count = 0;
  double first_time = 0.0;
  double last_time = 0.0;
  std::vector<uint8_t> latest_data;
};

struct StoreBatch {
  int segment = -1;
  std::vector<std::string> replace_series_paths;
  std::vector<TimeRange> coverage;
  std::vector<SeriesChunk> series;
  std::vector<CanEventChunk> can_events;
};

struct DrainResult {
  size_t batches = 0;
  size_t series_chunks = 0;
  size_t series_points = 0;
  size_t can_chunks = 0;
  size_t can_events = 0;
  std::vector<std::string> touched_series_paths;
};

struct StoreTrimResult {
  double cutoff_time = 0.0;
  size_t series_paths_removed = 0;
  size_t series_chunks_removed = 0;
  size_t series_points_removed = 0;
  size_t can_messages_removed = 0;
  size_t can_events_removed = 0;
  size_t coverage_ranges_removed = 0;
  std::vector<std::string> touched_series_paths;
};

class Store {
public:
  // Producers may call stage() from worker threads. The UI thread calls
  // begin_frame() once per frame; only that drain makes data visible to panes.
  void stage(StoreBatch batch);
  DrainResult begin_frame();
  StoreTrimResult trim_before(double cutoff_time);
  void clear();

  SeriesView series(std::string_view path, double t0, double t1, size_t max_points) const;
  SeriesView series_full(std::string_view path, TimeRange range) const;
  CanEventView can_events(const MessageId &id, TimeRange range) const;
  CanSummaryView can_event_summary(const MessageId &id, TimeRange range, bool with_data = true) const;
  // Per-byte time of the most recent value change within `range` (-inf where none), computed
  // in place — can_events() copies every event and is too slow to call per row per frame.
  std::vector<double> byte_change_times(const MessageId &id, TimeRange range, size_t byte_count) const;

  size_t staged_batch_count() const;
  uint64_t generation() const { return generation_; }
  size_t series_path_count() const { return series_.size(); }
  size_t can_message_count() const { return can_events_.size(); }
  std::vector<std::string> series_paths() const;
  std::vector<std::string> series_paths_matching(std::string_view filter, size_t limit) const;
  std::vector<MessageId> can_message_ids() const;

private:
  struct SeriesState {
    std::vector<SeriesChunk> chunks;
  };

  struct CanState {
    // Sorted-by-start chunk list, merged lazily at query time: the drain APPENDS a chunk
    // instead of re-copying the id's whole history (mergeEvents was a measured 20 ms+ UI-thread
    // hitch per drained sub-batch late in a route load). Chunks come one per (id, segment), so
    // queries walk ~segment-count chunks with a binary search each.
    std::vector<CanEventChunk> chunks;
    std::vector<TimeRange> coverage;
  };

  mutable std::mutex staged_mutex_;
  std::vector<StoreBatch> staged_batches_;
  // Budget leftovers from the previous begin_frame, drained first (FIFO) — pushing them back
  // onto the FRONT of staged_batches_ shifted thousands of queued batches per frame mid-load.
  std::vector<StoreBatch> carryover_batches_;

  std::unordered_map<std::string, SeriesState> series_;
  std::unordered_map<MessageId, CanState> can_events_;
  std::vector<TimeRange> coverage_;
  uint64_t generation_ = 0;
};

}  // namespace loggy
