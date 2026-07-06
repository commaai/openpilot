#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "openpilot/cereal/gen/cpp/log.capnp.h"
#include "tools/loggy/backend/store.h"

class Event;

namespace loggy {

enum class CanServiceKind : uint8_t {
  Can,
  Sendcan,
};

struct RouteSeries {
  std::string path;
  std::vector<double> times;
  std::vector<double> values;
};

struct SeriesMetadata {
  std::vector<std::string> enum_names;
  bool deprecated = false;
};

struct SchemaIndex {
  size_t fixed_series_count = 0;
  std::vector<std::string> fixed_paths;
};

struct SegmentExtractOptions {
  int segment = -1;
  TimeRange coverage;
  bool include_raw_can_events = true;
  bool include_can_scalar_fields = false;
  bool include_decoded_can_series = false;
  std::optional<double> time_offset;

  double time_offset_seconds() const { return time_offset.value_or(0.0); }
};

struct SegmentExtractResult {
  StoreBatch batch;
  std::unordered_map<std::string, SeriesMetadata> metadata;
  size_t events_seen = 0;
  size_t events_appended = 0;
};

class SeriesAccumulator {
public:
  explicit SeriesAccumulator(int segment = -1, std::vector<std::string> fixed_paths = {});

  RouteSeries *fixed_slot(size_t slot);
  RouteSeries *ensure_series(const std::string &path);
  RouteSeries *ensure_list_scalar_series(const std::string &base_path, size_t index);

  void append_fixed_scalar(size_t slot, double tm, double value);
  void append_scalar(const std::string &path, double tm, double value);
  void append_can_frame(CanServiceKind service, uint8_t bus, uint32_t address,
                      uint16_t bus_time, const uint8_t *data, size_t size, double tm);

  void capture_enum_info(const std::string &path, std::initializer_list<std::string_view> names);
  // Marks a path as deprecated in metadata; the field is still extracted like any other (the
  // browser's Deprecated toggle filters display, not extraction — see generate_event_extractors.py).
  void mark_deprecated(const std::string &path);

  const std::unordered_map<std::string, SeriesMetadata> &metadata() const { return metadata_; }

  SegmentExtractResult finish(TimeRange coverage = {});

  std::vector<RouteSeries> fixed_series;
  std::vector<RouteSeries> dynamic_series;

private:
  size_t ensure_dynamic_slot(const std::string &path);
  MessageId can_message_id(CanServiceKind service, uint8_t bus, uint32_t address) const;

  struct CanAccum {
    MessageId id;
    std::vector<CanEvent> events;
  };

  int segment_ = -1;
  std::unordered_map<std::string, size_t> dynamic_slots_;
  std::unordered_map<std::string, std::vector<size_t>> list_scalar_slots_;
  std::unordered_map<MessageId, size_t> can_slots_;
  std::vector<CanAccum> can_events_;
  std::unordered_map<std::string, SeriesMetadata> metadata_;
};

const SchemaIndex &event_schema_index();
SeriesAccumulator make_series_accumulator(const SchemaIndex &schema, int segment = -1);

bool append_event_reader(cereal::Event::Which which,
                       const cereal::Event::Reader &event,
                       const SegmentExtractOptions &options,
                       SeriesAccumulator *series);
bool append_event_data(cereal::Event::Which which,
                     int32_t eidx_segnum,
                     kj::ArrayPtr<const capnp::word> data,
                     const SegmentExtractOptions &options,
                     SeriesAccumulator *series);
void append_events_fast_range(const std::vector<::Event> &events,
                           size_t begin,
                           size_t end,
                           const SegmentExtractOptions &options,
                           SeriesAccumulator *series);
SegmentExtractResult extract_segment_series(const std::vector<::Event> &events,
                                          SegmentExtractOptions options = {});

}  // namespace loggy
