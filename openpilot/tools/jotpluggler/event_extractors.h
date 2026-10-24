#pragma once

#include "tools/jotpluggler/app.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Shared between sketch_layout.cc and the generated event extractors, which
// are split across several translation units so they compile in parallel.

struct SeriesAccumulator {
  explicit SeriesAccumulator(size_t fixed_count = 0) : fixed_series(fixed_count) {}

  std::vector<RouteSeries> fixed_series;
  std::vector<RouteSeries> dynamic_series;
  std::vector<CanMessageData> can_messages;
  std::unordered_map<std::string, size_t> dynamic_slots;
  std::unordered_map<std::string, std::vector<size_t>> list_scalar_slots;
  std::unordered_map<CanMessageId, size_t, CanMessageIdHash> can_message_slots;
  std::unordered_map<std::string, EnumInfo> enum_info;
};

// defined in sketch_layout.cc
void append_fixed_scalar_point(RouteSeries *series, double tm, double value);
void append_dynamic_scalar_point(const std::string &path, double tm, double value, SeriesAccumulator *series);
RouteSeries *ensure_dynamic_series(const std::string &path, SeriesAccumulator *series);
void append_text_point(RouteSeries *series, double tm, capnp::Text::Reader text, SeriesAccumulator *accumulator);
RouteSeries *ensure_list_scalar_series(const std::string &base_path, size_t index, SeriesAccumulator *series);
void append_can_frame(CanServiceKind service,
                      uint8_t bus,
                      uint32_t address,
                      uint16_t bus_time,
                      capnp::Data::Reader dat,
                      double tm,
                      SeriesAccumulator *series);
void decode_can_frame(const dbc::Database *can_dbc,
                      const std::string &service_name,
                      uint8_t bus,
                      uint32_t address,
                      const uint8_t *raw,
                      size_t data_size,
                      double tm,
                      SeriesAccumulator *series);

// defined in generated/event_extractors.cc
const std::vector<std::string> &static_event_fixed_paths();
void capture_static_enum_info(const std::string &path, std::initializer_list<std::string_view> names, SeriesAccumulator *series);
__attribute__((noinline)) void append_fixed_enum_point(size_t series_slot, size_t enum_index, double tm, double value, SeriesAccumulator *series);
bool append_event_static_reader(cereal::Event::Which which, const cereal::Event::Reader &event, const dbc::Database *can_dbc, bool skip_raw_can, double time_offset, SeriesAccumulator *series);
