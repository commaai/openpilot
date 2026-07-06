#include "tools/loggy/backend/csv.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace loggy {
namespace {

struct StreamCsvRow {
  MessageId id;
  CanEvent event;
};

bool parse_u32_value(std::string_view text, int base, uint32_t *out) {
  if (text.empty() || out == nullptr) return false;
  std::string copy(text);
  char *end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(copy.c_str(), &end, base);
  if (end != copy.c_str() + copy.size() || errno == ERANGE || value > std::numeric_limits<uint32_t>::max()) return false;
  *out = static_cast<uint32_t>(value);
  return true;
}

bool parse_message_id_text(std::string_view text, MessageId *out) {
  if (out == nullptr || text.empty()) return false;
  const size_t colon = text.find(':');
  if (colon == std::string_view::npos) {
    uint32_t address = 0;
    if (!parse_u32_value(text, 16, &address)) return false;
    *out = MessageId{.source = 0, .address = address};
    return true;
  }

  uint32_t source = 0;
  uint32_t address = 0;
  if (!parse_u32_value(text.substr(0, colon), 10, &source)) return false;
  const std::string_view addr_text = text.substr(colon + 1);
  if (!parse_u32_value(addr_text, 16, &address)) return false;
  *out = MessageId{.source = static_cast<uint8_t>(std::min<uint32_t>(source, 255)), .address = address};
  return true;
}

bool stream_row_less(const StreamCsvRow &a, const StreamCsvRow &b) {
  if (a.event.mono_time != b.event.mono_time) return a.event.mono_time < b.event.mono_time;
  if (a.event.bus_time != b.event.bus_time) return a.event.bus_time < b.event.bus_time;
  return a.id < b.id;
}

std::string csv_time(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << value;
  return out.str();
}

std::string address_hex(uint32_t address) {
  std::ostringstream out;
  out << "0x" << std::uppercase << std::hex << address;
  return out.str();
}

std::string hex_bytes(const std::vector<uint8_t> &bytes) {
  std::ostringstream out;
  out << std::uppercase << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) out << ' ';
    out << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return out.str();
}

std::string csv_escape(std::string_view text) {
  const bool needs_quotes = text.find_first_of(",\"\r\n") != std::string_view::npos;
  if (!needs_quotes) return std::string(text);

  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (const char c : text) {
    if (c == '"') out.push_back('"');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

void append_cell(std::ostringstream *out, std::string_view value, bool last = false) {
  *out << csv_escape(value);
  *out << (last ? '\n' : ',');
}

std::string decoded_values(const Msg *msg, const std::vector<uint8_t> &data) {
  if (msg == nullptr || data.empty()) return {};
  std::string out;
  for (const Signal *sig : msg->signals()) {
    if (sig == nullptr) continue;
    double value = 0.0;
    if (!sig->get_value(data.data(), data.size(), &value)) continue;
    if (!out.empty()) out += "; ";
    out += sig->name + "=" + sig->format_value(value);
  }
  return out;
}

void append_event_row(std::ostringstream *out, const MessageId &id, const CanEvent &event, const Msg *msg = nullptr) {
  append_cell(out, csv_time(event.mono_time));
  append_cell(out, std::to_string(event.bus_time));
  append_cell(out, std::to_string(id.source));
  append_cell(out, address_hex(id.address));
  append_cell(out, std::to_string(event.data.size()));
  append_cell(out, hex_bytes(event.data));
  append_cell(out, decoded_values(msg, event.data), true);
}

std::vector<StreamCsvRow> collect_stream_rows(const Store &store, TimeRange range) {
  std::vector<StreamCsvRow> rows;
  for (const MessageId &id : store.can_message_ids()) {
    const CanEventView view = store.can_events(id, range);
    rows.reserve(rows.size() + view.events.size());
    for (const CanEvent &event : view.events) rows.push_back({.id = id, .event = event});
  }
  std::sort(rows.begin(), rows.end(), stream_row_less);
  return rows;
}

std::string history_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool history_text_matches_filter(std::string_view text, std::string_view filter) {
  return filter.empty() || history_lower_text(text).find(history_lower_text(filter)) != std::string::npos;
}

bool history_compare_values(double lhs, std::string_view op, double rhs) {
  if (op == ">") return lhs > rhs;
  if (op == "=") return lhs == rhs;
  if (op == "!=") return lhs != rhs;
  if (op == "<") return lhs < rhs;
  if (op == ">=") return lhs >= rhs;
  if (op == "<=") return lhs <= rhs;
  return true;
}

std::string history_hex_bytes(const std::vector<uint8_t> &bytes) {
  std::string out;
  out.reserve(bytes.size() * 3);
  for (size_t i = 0; i < bytes.size(); ++i) {
    char buf[4];
    std::snprintf(buf, sizeof(buf), "%02X", bytes[i]);
    if (!out.empty()) out.push_back(' ');
    out += buf;
  }
  return out;
}

std::string history_decoded_values(const Msg *msg, const std::vector<uint8_t> &data, size_t max_values = 4) {
  if (msg == nullptr || data.empty()) return {};
  std::string out;
  size_t count = 0;
  for (const Signal *sig : msg->signals()) {
    if (sig == nullptr) continue;
    double value = 0.0;
    if (!sig->get_value(data.data(), data.size(), &value)) continue;
    if (!out.empty()) out += ", ";
    out += sig->name + "=" + sig->format_value(value);
    if (++count >= max_values) break;
  }
  return out;
}

bool history_matches_compare(const Msg *msg, const HistoryLogParams &params, const std::vector<uint8_t> &data) {
  if (!params.compare_enabled || params.compare_signal.empty()) return true;
  if (msg == nullptr || data.empty()) return false;
  const Signal *sig = msg->sig(params.compare_signal);
  if (sig == nullptr) return false;
  double value = 0.0;
  if (!sig->get_value(data.data(), data.size(), &value)) return false;
  return history_compare_values(value, params.compare_op, params.compare_value);
}

}  // namespace

MessageId parse_message_id_state(std::string_view state_json, std::optional<MessageId> selection, MessageId fallback) {
  if (selection.has_value()) return *selection;

  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  MessageId id = fallback;
  if (!err.empty()) {
    parse_message_id_text(state_json, &id);
    return id;
  }
  if (state.is_string()) {
    parse_message_id_text(state.string_value(), &id);
    return id;
  }
  if (!state.is_object()) return id;

  const json11::Json &text_id = state["id"].is_string() ? state["id"]
                            : state["message_id"].is_string() ? state["message_id"]
                            : state["selected_id"];
  if (text_id.is_string() && parse_message_id_text(text_id.string_value(), &id)) return id;

  const json11::Json &source_json = state["source"].is_number() ? state["source"] : state["bus"];
  const json11::Json &addr_json = state["address"].is_number() || state["address"].is_string() ? state["address"] : state["addr"];
  if (source_json.is_number()) id.source = static_cast<uint8_t>(std::clamp(source_json.int_value(), 0, 255));
  if (addr_json.is_number()) {
    id.address = static_cast<uint32_t>(std::max(0, addr_json.int_value()));
  } else if (addr_json.is_string()) {
    uint32_t address = id.address;
    if (parse_u32_value(addr_json.string_value(), 16, &address)) id.address = address;
  }
  return id;
}

MessageId initial_message_id_for_store(const Store &store, std::string_view state_json,
                                      std::optional<MessageId> selection) {
  if (selection.has_value()) return *selection;
  const std::vector<MessageId> ids = store.can_message_ids();
  const MessageId fallback = ids.empty() ? kDefaultLoggyMessageId : ids.front();
  return parse_message_id_state(state_json, std::nullopt, fallback);
}

MessageSummary summarize_message_events(const Store &store, const MessageId &id, TimeRange range, bool with_data) {
  MessageSummary summary;
  summary.id = id;
  const CanSummaryView view = store.can_event_summary(id, range, with_data);
  summary.coverage = view.coverage;
  summary.count = view.count;
  if (view.count == 0) return summary;

  summary.first_time = view.first_time;
  summary.last_time = view.last_time;
  summary.latest_data = view.latest_data;
  if (summary.count > 1 && summary.last_time > summary.first_time) {
    summary.frequency_hz = static_cast<double>(summary.count - 1) / (summary.last_time - summary.first_time);
  }
  return summary;
}

std::string can_message_csv(const Store &store, const MessageId &id, TimeRange range, const Msg *msg) {
  std::ostringstream out;
  out << "mono_time,bus_time,bus,address,length,hex,decoded\n";
  const CanEventView view = store.can_events(id, range);
  for (const CanEvent &event : view.events) append_event_row(&out, id, event, msg);
  return out.str();
}

std::string can_stream_csv(const Store &store, TimeRange range) {
  std::ostringstream out;
  out << "mono_time,bus_time,bus,address,length,hex,decoded\n";
  for (const StreamCsvRow &row : collect_stream_rows(store, range)) append_event_row(&out, row.id, row.event);
  return out.str();
}

std::string can_signal_csv(const Store &store, const MessageId &id, TimeRange range, const Signal &signal) {
  std::ostringstream out;
  out << "mono_time,bus_time,bus,address,signal,value,unit,hex\n";
  const CanEventView view = store.can_events(id, range);
  for (const CanEvent &event : view.events) {
    double value = 0.0;
    if (!signal.get_value(event.data.data(), event.data.size(), &value)) continue;
    append_cell(&out, csv_time(event.mono_time));
    append_cell(&out, std::to_string(event.bus_time));
    append_cell(&out, std::to_string(id.source));
    append_cell(&out, address_hex(id.address));
    append_cell(&out, signal.name);
    // Same decode path as decoded_values()/history_decoded_values(): a value matching a
    // val_desc entry (e.g. an out-of-range sentinel) shows its label, not the raw number.
    append_cell(&out, signal.format_value(value, /*with_unit=*/false));
    append_cell(&out, signal.unit);
    append_cell(&out, hex_bytes(event.data), true);
  }
  return out.str();
}

std::string series_csv(const Store &store, std::string_view path, TimeRange range) {
  std::ostringstream out;
  out << "time,path,value\n";
  const SeriesView view = store.series_full(path, range);
  for (const SeriesPoint &point : view.points) {
    append_cell(&out, csv_time(point.t));
    append_cell(&out, view.path);
    append_cell(&out, double_to_string(point.value), true);
  }
  return out.str();
}

float byte_change_alpha(double last_change, double tracker) {
  if (!std::isfinite(last_change)) return 0.0f;
  const double age = std::max(0.0, tracker - last_change);
  return static_cast<float>(std::clamp(1.0 - age / kByteChangeFadeSeconds, 0.0, 1.0));
}

std::optional<BinaryGrid> build_binary_grid(const Store &store, const MessageId &id, TimeRange range) {
  // O(log n): `range` is {route_start, tracker}, which grows for the life of the route, so this
  // must not copy every matching event the way can_events() would.
  const CanSummaryView view = store.can_event_summary(id, range, /*with_data=*/true);
  if (view.count == 0) return std::nullopt;

  BinaryGrid grid;
  grid.id = id;
  grid.event_count = view.count;
  grid.first_time = view.first_time;
  grid.last_time = view.last_time;
  grid.latest_data = view.latest_data;
  grid.rows.resize(grid.latest_data.size());

  for (size_t byte_index = 0; byte_index < grid.latest_data.size(); ++byte_index) {
    const uint8_t byte = grid.latest_data[byte_index];
    for (int bit = 0; bit < 8; ++bit) {
      BinaryBitCell &cell = grid.rows[byte_index][static_cast<size_t>(bit)];
      cell.valid = true;
      cell.value = static_cast<uint8_t>((byte >> (7 - bit)) & 1U);
    }
  }

  grid.byte_last_change = store.byte_change_times(
      id, TimeRange{range.end - kByteChangeFadeSeconds, range.end}, grid.latest_data.size());
  return grid;
}

bool history_valid_compare_op(std::string_view op) {
  return op == ">" || op == "=" || op == "!=" || op == "<" || op == ">=" || op == "<=";
}

HistoryLogPage prepare_history_log_page(const Store &store, const MessageId &id, TimeRange range,
                                       const HistoryLogParams &params, const Msg *msg) {
  const CanEventView view = store.can_events(id, range);
  std::vector<HistoryLogRow> matches;
  matches.reserve(std::min(view.events.size(), params.max_rows));
  bool truncated = false;
  size_t event_index = 0;
  for (auto it = view.events.rbegin(); it != view.events.rend(); ++it) {
    const CanEvent &event = *it;
    ++event_index;
    if (!history_matches_compare(msg, params, event.data)) continue;

    HistoryLogRow row;
    row.mono_time = event.mono_time;
    row.bus_time = event.bus_time;
    row.byte_count = event.data.size();
    row.data_hex = history_hex_bytes(event.data);
    row.decoded = history_decoded_values(msg, event.data);
    const std::string searchable = row.data_hex + " " + row.decoded;
    if (!history_text_matches_filter(searchable, params.filter)) continue;
    matches.push_back(std::move(row));
    if (matches.size() >= params.max_rows) {
      truncated = event_index < view.events.size();
      break;
    }
  }

  HistoryLogPage page;
  page.total_rows = matches.size();
  page.page_size = std::clamp(params.page_size, static_cast<size_t>(1), static_cast<size_t>(5000));
  page.page_count = std::max(static_cast<size_t>(1), (matches.size() + page.page_size - 1) / page.page_size);
  page.page_index = std::min(params.page_index, page.page_count - 1);
  page.truncated = truncated;

  const size_t start_ = std::min(matches.size(), page.page_index * page.page_size);
  const size_t end = std::min(matches.size(), start_ + page.page_size);
  page.rows.assign(matches.begin() + static_cast<std::ptrdiff_t>(start_), matches.begin() + static_cast<std::ptrdiff_t>(end));
  return page;
}

bool write_csv_file(const std::filesystem::path &path, std::string_view csv, std::string &error) {
  if (path.empty()) {
    error = "empty export path";
    return false;
  }

  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      error = "failed to create export directory: " + ec.message();
      return false;
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    error = "failed to open export file";
    return false;
  }
  out.write(csv.data(), static_cast<std::streamsize>(csv.size()));
  if (!out.good()) {
    error = "failed to write export file";
    return false;
  }
  error.clear();
  return true;
}

}  // namespace loggy
