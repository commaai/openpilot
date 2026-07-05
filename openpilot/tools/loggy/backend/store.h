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
double distanceToPoint(TimeRange range, double point);
std::vector<TimeRange> normalizeRanges(std::vector<TimeRange> ranges);

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
};

class Store {
public:
  // Producers may call stage() from worker threads. The UI thread calls
  // beginFrame() once per frame; only that drain makes data visible to panes.
  void stage(StoreBatch batch);
  DrainResult beginFrame();
  void clear();

  SeriesView series(std::string_view path, double t0, double t1, size_t max_points) const;
  CanEventView canEvents(const MessageId &id, TimeRange range) const;
  CanSummaryView canEventSummary(const MessageId &id, TimeRange range) const;

  size_t stagedBatchCount() const;
  size_t seriesPathCount() const { return series_.size(); }
  size_t canMessageCount() const { return can_events_.size(); }
  std::vector<std::string> seriesPaths() const;
  std::vector<MessageId> canMessageIds() const;

private:
  struct SeriesState {
    std::vector<SeriesChunk> chunks;
  };

  struct CanState {
    std::vector<CanEvent> events;
    std::vector<TimeRange> coverage;
  };

  mutable std::mutex staged_mutex_;
  std::vector<StoreBatch> staged_batches_;

  std::unordered_map<std::string, SeriesState> series_;
  std::unordered_map<MessageId, CanState> can_events_;
  std::vector<TimeRange> coverage_;
};

}  // namespace loggy
