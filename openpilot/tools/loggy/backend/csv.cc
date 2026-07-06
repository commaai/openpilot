#include "tools/loggy/backend/csv.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cerrno>
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

MessageSummary summarize_message_events(const Store &store, const MessageId &id, TimeRange range) {
  MessageSummary summary;
  summary.id = id;
  const CanSummaryView view = store.can_event_summary(id, range);
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
    append_cell(&out, double_to_string(value));
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
