#include "tools/loggy/backend/extract.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <kj/exception.h>

#include "tools/replay/logreader.h"

namespace loggy {
namespace {

constexpr size_t INVALID_DYNAMIC_SLOT = static_cast<size_t>(-1);

TimeRange pointRange(const RouteSeries &series) {
  if (series.times.empty()) return {};
  auto [lo, hi] = std::minmax_element(series.times.begin(), series.times.end());
  return {*lo, *hi};
}

TimeRange eventRange(const std::vector<CanEvent> &events) {
  if (events.empty()) return {};
  auto [lo, hi] = std::minmax_element(events.begin(), events.end(), [](const auto &a, const auto &b) {
    return a.mono_time < b.mono_time;
  });
  return {lo->mono_time, hi->mono_time};
}

bool seriesIsSorted(const RouteSeries &series) {
  for (size_t i = 1; i < series.times.size(); ++i) {
    if (series.times[i] < series.times[i - 1]) return false;
  }
  return true;
}

void sortSeriesByTime(RouteSeries *series) {
  if (series->times.size() <= 1 || seriesIsSorted(*series)) return;

  std::vector<size_t> order(series->times.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return series->times[a] < series->times[b];
  });

  std::vector<double> sorted_times(series->times.size());
  std::vector<double> sorted_values(series->values.size());
  for (size_t i = 0; i < order.size(); ++i) {
    sorted_times[i] = series->times[order[i]];
    sorted_values[i] = series->values[order[i]];
  }
  series->times = std::move(sorted_times);
  series->values = std::move(sorted_values);
}

SeriesChunk makeChunk(RouteSeries &&series, int segment) {
  sortSeriesByTime(&series);
  SeriesChunk chunk;
  chunk.path = std::move(series.path);
  chunk.range = pointRange(series);
  chunk.segment = segment;
  const size_t count = std::min(series.times.size(), series.values.size());
  chunk.points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    chunk.points.push_back({series.times[i], series.values[i]});
  }
  return chunk;
}

void append_fixed_scalar_point(RouteSeries *series, double tm, double value) {
  series->times.push_back(tm);
  series->values.push_back(value);
}

void append_dynamic_scalar_point(const std::string &path, double tm, double value, SeriesAccumulator *series) {
  series->append_scalar(path, tm, value);
}

RouteSeries *ensure_list_scalar_series(const std::string &base_path, size_t index, SeriesAccumulator *series) {
  return series->ensure_list_scalar_series(base_path, index);
}

void capture_static_enum_info(const std::string &path,
                              std::initializer_list<std::string_view> names,
                              SeriesAccumulator *series) {
  series->capture_enum_info(path, names);
}

void capture_deprecated_series(const std::string &path, SeriesAccumulator *series) {
  series->mark_deprecated(path);
}

void append_can_frame(CanServiceKind service,
                      uint8_t bus,
                      uint32_t address,
                      uint16_t bus_time,
                      capnp::Data::Reader dat,
                      double tm,
                      SeriesAccumulator *series) {
  series->append_can_frame(service, bus, address, bus_time, dat.begin(), dat.size(), tm);
}

template <typename Fn>
bool with_parseable_event(kj::ArrayPtr<const capnp::word> data, Fn &&fn) {
  try {
    capnp::FlatArrayMessageReader event_reader(data);
    fn(event_reader.getRoot<cereal::Event>());
    return true;
  } catch (const kj::Exception &) {
    return false;
  }
}

#include "tools/loggy/backend/generated_event_extractors.h"

}  // namespace

SeriesAccumulator::SeriesAccumulator(int segment, std::vector<std::string> fixed_paths) : segment_(segment) {
  fixed_series.reserve(fixed_paths.size());
  for (std::string &path : fixed_paths) {
    fixed_series.push_back(RouteSeries{.path = std::move(path)});
  }
}

RouteSeries *SeriesAccumulator::fixed_slot(size_t slot) {
  if (slot >= fixed_series.size()) {
    throw std::out_of_range("fixed series slot out of range");
  }
  return &fixed_series[slot];
}

RouteSeries *SeriesAccumulator::ensure_series(const std::string &path) {
  return &dynamic_series[ensure_dynamic_slot(path)];
}

RouteSeries *SeriesAccumulator::ensure_list_scalar_series(const std::string &base_path, size_t index) {
  auto [it, _] = list_scalar_slots_.try_emplace(base_path);
  std::vector<size_t> &slots = it->second;
  if (slots.size() <= index) {
    slots.resize(index + 1, INVALID_DYNAMIC_SLOT);
  }
  if (slots[index] == INVALID_DYNAMIC_SLOT) {
    slots[index] = ensure_dynamic_slot(base_path + "/" + std::to_string(index));
  }
  return &dynamic_series[slots[index]];
}

void SeriesAccumulator::append_fixed_scalar(size_t slot, double tm, double value) {
  append_fixed_scalar_point(fixed_slot(slot), tm, value);
}

void SeriesAccumulator::append_scalar(const std::string &path, double tm, double value) {
  RouteSeries *series = ensure_series(path);
  if (series->path.empty()) series->path = path;
  append_fixed_scalar_point(series, tm, value);
}

void SeriesAccumulator::append_can_frame(CanServiceKind service, uint8_t bus, uint32_t address,
                                       uint16_t bus_time, const uint8_t *data, size_t size, double tm) {
  const MessageId id = can_message_id(service, bus, address);
  auto [it, inserted] = can_slots_.try_emplace(id, can_events_.size());
  if (inserted) {
    can_events_.push_back(CanAccum{.id = id});
  }

  CanEvent event;
  event.mono_time = tm;
  event.bus_time = bus_time;
  if (data != nullptr && size > 0) {
    event.data.assign(data, data + size);
  }
  can_events_[it->second].events.push_back(std::move(event));
}

void SeriesAccumulator::capture_enum_info(const std::string &path, std::initializer_list<std::string_view> names) {
  if (names.size() == 0) return;
  SeriesMetadata &metadata = metadata_[path];
  if (!metadata.enum_names.empty()) return;
  metadata.enum_names.reserve(names.size());
  for (std::string_view name : names) {
    metadata.enum_names.emplace_back(name);
  }
}

void SeriesAccumulator::mark_deprecated(const std::string &path) {
  metadata_[path].deprecated = true;
}

SegmentExtractResult SeriesAccumulator::finish(TimeRange coverage) {
  SegmentExtractResult result;
  result.metadata = std::move(metadata_);
  result.batch.segment = segment_;
  if (coverage.valid()) {
    result.batch.coverage.push_back(coverage);
  }

  result.batch.series.reserve(fixed_series.size() + dynamic_series.size());
  for (RouteSeries &series : fixed_series) {
    if (series.times.empty()) continue;
    SeriesChunk chunk = makeChunk(std::move(series), segment_);
    if (!coverage.valid() && chunk.range.valid()) result.batch.coverage.push_back(chunk.range);
    result.batch.series.push_back(std::move(chunk));
  }
  for (RouteSeries &series : dynamic_series) {
    if (series.times.empty()) continue;
    SeriesChunk chunk = makeChunk(std::move(series), segment_);
    if (!coverage.valid() && chunk.range.valid()) result.batch.coverage.push_back(chunk.range);
    result.batch.series.push_back(std::move(chunk));
  }

  result.batch.can_events.reserve(can_events_.size());
  for (CanAccum &accum : can_events_) {
    if (accum.events.empty()) continue;
    CanEventChunk chunk;
    chunk.id = accum.id;
    chunk.range = eventRange(accum.events);
    chunk.events = std::move(accum.events);
    chunk.segment = segment_;
    if (!coverage.valid() && chunk.range.valid()) result.batch.coverage.push_back(chunk.range);
    result.batch.can_events.push_back(std::move(chunk));
  }

  result.batch.coverage = normalize_ranges(std::move(result.batch.coverage));
  return result;
}

size_t SeriesAccumulator::ensure_dynamic_slot(const std::string &path) {
  auto [it, inserted] = dynamic_slots_.try_emplace(path, dynamic_series.size());
  if (inserted) {
    dynamic_series.push_back(RouteSeries{.path = it->first});
  }
  return it->second;
}

MessageId SeriesAccumulator::can_message_id(CanServiceKind service, uint8_t bus, uint32_t address) const {
  uint8_t source = bus;
  if (service == CanServiceKind::Sendcan) {
    source = static_cast<uint8_t>(bus | 0x80);
  }
  return {.source = source, .address = address};
}

const SchemaIndex &event_schema_index() {
  static const SchemaIndex index = [] {
    SchemaIndex out;
    out.fixed_paths = static_event_fixed_paths();
    out.fixed_series_count = out.fixed_paths.size();
    return out;
  }();
  return index;
}

SeriesAccumulator make_series_accumulator(const SchemaIndex &schema, int segment) {
  return SeriesAccumulator(segment, schema.fixed_paths);
}

bool append_event_reader(cereal::Event::Which which,
                       const cereal::Event::Reader &event,
                       const SegmentExtractOptions &options,
                       SeriesAccumulator *series) {
  return append_event_static_reader(which, event, options, series);
}

bool append_event_data(cereal::Event::Which which,
                     int32_t eidx_segnum,
                     kj::ArrayPtr<const capnp::word> data,
                     const SegmentExtractOptions &options,
                     SeriesAccumulator *series) {
  if (eidx_segnum != -1) return false;
  bool appended = false;
  with_parseable_event(data, [&](const cereal::Event::Reader &event) {
    appended = append_event_reader(which, event, options, series);
  });
  return appended;
}

void append_events_fast_range(const std::vector<::Event> &events,
                           size_t begin,
                           size_t end,
                           const SegmentExtractOptions &options,
                           SeriesAccumulator *series) {
  const size_t bounded_end = std::min(end, events.size());
  for (size_t i = begin; i < bounded_end; ++i) {
    const ::Event &event_record = events[i];
    append_event_data(event_record.which, event_record.eidx_segnum, event_record.data, options, series);
  }
}

SegmentExtractResult extract_segment_series(const std::vector<::Event> &events, SegmentExtractOptions options) {
  SeriesAccumulator series = make_series_accumulator(event_schema_index(), options.segment);
  size_t appended = 0;
  for (const ::Event &event_record : events) {
    if (append_event_data(event_record.which, event_record.eidx_segnum, event_record.data, options, &series)) {
      ++appended;
    }
  }

  SegmentExtractResult result = series.finish(options.coverage);
  result.events_seen = events.size();
  result.events_appended = appended;
  return result;
}

}  // namespace loggy
