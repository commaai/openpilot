#include "tools/loggy/backend/scan.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace loggy {

uint8_t bit_value_at(const std::vector<uint8_t> &data, int byte_idx, int bit_idx) {
  if (byte_idx < 0 || bit_idx < 0 || bit_idx > 7 || byte_idx >= static_cast<int>(data.size())) return 0;
  return static_cast<uint8_t>((data[static_cast<size_t>(byte_idx)] >> (7 - bit_idx)) & 1U);
}

std::vector<FindBitsRow> scan_find_bits_events(const std::vector<FindBitsEvent> &events, const FindBitsParams &params) {
  struct Accum {
    uint32_t total = 0;
    uint32_t mismatches = 0;
  };
  std::unordered_map<uint64_t, Accum> accum;
  auto key_for = [](uint32_t address, uint32_t byte_idx, uint32_t bit_idx) {
    return (static_cast<uint64_t>(address) << 16) | (static_cast<uint64_t>(byte_idx) << 8) | bit_idx;
  };

  for (const FindBitsEvent &event : events) {
    for (size_t byte_idx = 0; byte_idx < event.data.size(); ++byte_idx) {
      for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        const uint8_t target_value = bit_value_at(event.data, static_cast<int>(byte_idx), bit_idx);
        const bool matched = params.equal ? target_value == event.source_value : target_value != event.source_value;
        Accum &row = accum[key_for(event.id.address, static_cast<uint32_t>(byte_idx), static_cast<uint32_t>(bit_idx))];
        ++row.total;
        if (!matched) ++row.mismatches;
      }
    }
  }

  std::vector<FindBitsRow> rows;
  rows.reserve(accum.size());
  for (const auto &[key, value] : accum) {
    if (static_cast<int>(value.total) <= params.min_msgs) continue;
    FindBitsRow row;
    row.address = static_cast<uint32_t>(key >> 16);
    row.byte_idx = static_cast<uint32_t>((key >> 8) & 0xFF);
    row.bit_idx = static_cast<uint32_t>(key & 0xFF);
    row.total = value.total;
    row.mismatches = value.mismatches;
    row.percent = value.total == 0 ? 0.0f : 100.0f * static_cast<float>(value.mismatches) / static_cast<float>(value.total);
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end(), [](const FindBitsRow &a, const FindBitsRow &b) {
    return std::tie(a.percent, a.mismatches, a.address, a.byte_idx, a.bit_idx) <
           std::tie(b.percent, b.mismatches, b.address, b.byte_idx, b.bit_idx);
  });
  if (rows.size() > params.max_rows) rows.resize(params.max_rows);
  return rows;
}

FindBitsJob make_find_bits_job(const Store &store, const FindBitsParams &params) {
  FindBitsJob job;
  job.store = &store;
  job.params = params;
  job.ids = store.can_message_ids();
  job.ids.erase(std::remove_if(job.ids.begin(), job.ids.end(), [&](const MessageId &id) {
    return id.source != params.find_bus ||
           (id.source == params.source_bus && id.address == params.source_address);
  }), job.ids.end());
  std::sort(job.ids.begin(), job.ids.end());
  job.done = false;
  return job;
}

bool step_find_bits_job(FindBitsJob &job, size_t max_messages) {
  if (job.done || job.store == nullptr) return true;
  const MessageId source_id{.source = job.params.source_bus, .address = job.params.source_address};
  const CanEventView source = job.store->can_events(source_id, job.params.range);
  if (source.events.empty()) {
    job.done = true;
    return true;
  }

  size_t visited = 0;
  while (visited < max_messages && job.id_index < job.ids.size()) {
    const MessageId id = job.ids[job.id_index++];
    ++visited;
    const CanEventView target = job.store->can_events(id, job.params.range);
    if (target.events.empty()) continue;

    // One comparison per candidate event, paired with the source sample in effect at that
    // time (nearest source sample <= event time). Pairing the other way — walking source
    // samples and snapshotting "the last known target data" — replays the same target frame
    // for every intervening source sample, so a row's total can exceed the candidate's own
    // event count (what the binary view shows for that message): a stat nobody can reproduce
    // by hand-counting frames.
    size_t source_idx = 0;
    for (const CanEvent &target_event : target.events) {
      while (source_idx + 1 < source.events.size() && source.events[source_idx + 1].mono_time <= target_event.mono_time) {
        ++source_idx;
      }
      if (source.events[source_idx].mono_time > target_event.mono_time) continue;
      job.events.push_back({
        .mono_time = target_event.mono_time,
        .source_value = bit_value_at(source.events[source_idx].data, job.params.byte_idx, job.params.bit_idx),
        .id = id,
        .data = target_event.data,
      });
    }
  }
  job.done = job.id_index >= job.ids.size();
  if (job.done) job.rows = scan_find_bits_events(job.events, job.params);
  return job.done;
}

namespace {

bool vector_contains_int(const std::vector<int> &values, int value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

bool vector_contains_address(const std::vector<uint32_t> &values, uint32_t value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

uint32_t event_msg_size(const CanEventView &view) {
  size_t size = 0;
  for (const CanEvent &event : view.events) size = std::max(size, event.data.size());
  return static_cast<uint32_t>(std::min<size_t>(size, CAN_MAX_DATA_BYTES));
}

void advance_find_signal_cursor(FindSignalJob &job, int msg_bits) {
  ++job.start_bit;
  if (job.start_bit + job.size <= msg_bits) return;
  job.start_bit = 0;
  ++job.size;
  if (job.size <= job.params.max_size) return;
  ++job.id_index;
  job.size = job.params.min_size;
  job.start_bit = 0;
}

Signal find_signal_candidate(int start_bit, int size, const FindSignalParams &params) {
  Signal signal;
  signal.start_bit = start_bit;
  signal.size = size;
  signal.is_little_endian = params.little_endian;
  signal.is_signed = params.is_signed;
  signal.factor = params.factor;
  signal.offset = params.offset;
  signal.min = 0.0;
  signal.max = std::pow(2.0, static_cast<double>(size)) - 1.0;
  signal.receiver_name = DEFAULT_NODE_NAME;
  signal.update();
  return signal;
}

}  // namespace

bool find_signal_compare_value(double value, FindSignalCompare compare, double target) {
  if (!std::isfinite(value)) return false;
  switch (compare) {
    case FindSignalCompare::Equal: return std::abs(value - target) <= 1.0e-9;
    case FindSignalCompare::NotEqual: return std::abs(value - target) > 1.0e-9;
    case FindSignalCompare::Greater: return value > target;
    case FindSignalCompare::GreaterEqual: return value >= target;
    case FindSignalCompare::Less: return value < target;
    case FindSignalCompare::LessEqual: return value <= target;
    case FindSignalCompare::Any:
    default: return true;
  }
}

FindSignalJob make_find_signal_job(const Store &store, const FindSignalParams &params) {
  FindSignalJob job;
  job.store = &store;
  job.params = params;
  job.params.min_size = std::clamp(job.params.min_size, 1, CAN_MAX_DATA_BYTES * 8);
  job.params.max_size = std::clamp(std::max(job.params.max_size, job.params.min_size),
                                   job.params.min_size, CAN_MAX_DATA_BYTES * 8);
  job.size = job.params.min_size;
  job.done = false;
  job.ids = store.can_message_ids();
  job.ids.erase(std::remove_if(job.ids.begin(), job.ids.end(), [&](const MessageId &id) {
    return !vector_contains_int(job.params.buses, id.source) ||
           !vector_contains_address(job.params.addresses, id.address);
  }), job.ids.end());
  std::sort(job.ids.begin(), job.ids.end());
  if (job.ids.empty()) job.done = true;
  return job;
}

bool step_find_signal_job(FindSignalJob &job, size_t max_candidates) {
  if (job.done || job.store == nullptr) return true;
  size_t visited = 0;
  while (visited < max_candidates && job.id_index < job.ids.size() && job.results.size() < job.params.max_results) {
    const MessageId id = job.ids[job.id_index];
    const CanEventView view = job.store->can_events(id, job.params.range);
    const uint32_t msg_size = event_msg_size(view);
    const int msg_bits = static_cast<int>(msg_size * 8);
    if (view.events.empty() || msg_bits <= 0 || job.size > msg_bits) {
      ++job.id_index;
      job.size = job.params.min_size;
      job.start_bit = 0;
      continue;
    }

    if (job.start_bit + job.size > msg_bits) {
      advance_find_signal_cursor(job, msg_bits);
      continue;
    }

    ++visited;
    Signal signal = find_signal_candidate(job.start_bit, job.size, job.params);
    FindSignalResult result;
    result.id = id;
    result.sig = signal;
    result.msg_size = msg_size;
    for (const CanEvent &event : view.events) {
      if (event.data.size() < msg_size) continue;
      double value = 0.0;
      if (!signal.get_value(event.data.data(), event.data.size(), &value)) continue;
      if (!find_signal_compare_value(value, job.params.compare, job.params.target_value)) continue;
      if (result.matches.empty()) result.mono_time = event.mono_time;
      result.matches.push_back({event.mono_time, value});
    }
    if (!result.matches.empty()) job.results.push_back(std::move(result));
    advance_find_signal_cursor(job, msg_bits);
  }
  job.done = job.id_index >= job.ids.size() || job.results.size() >= job.params.max_results;
  return job.done;
}

}  // namespace loggy
