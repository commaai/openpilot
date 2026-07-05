#include "tools/loggy/backend/export.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace loggy {
namespace {

struct StreamCsvRow {
  MessageId id;
  CanEvent event;
};

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

void append_cell(std::ostringstream *out, std::string_view value, bool last = false) {
  *out << csv_escape(value);
  *out << (last ? '\n' : ',');
}

std::string decoded_values(const Msg *msg, const std::vector<uint8_t> &data) {
  if (msg == nullptr || data.empty()) return {};
  std::string out;
  for (const Signal *sig : msg->getSignals()) {
    if (sig == nullptr) continue;
    double value = 0.0;
    if (!sig->getValue(data.data(), data.size(), &value)) continue;
    if (!out.empty()) out += "; ";
    out += sig->name + "=" + sig->formatValue(value);
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
  for (const MessageId &id : store.canMessageIds()) {
    const CanEventView view = store.canEvents(id, range);
    rows.reserve(rows.size() + view.events.size());
    for (const CanEvent &event : view.events) rows.push_back({.id = id, .event = event});
  }
  std::sort(rows.begin(), rows.end(), stream_row_less);
  return rows;
}

}  // namespace

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

std::string can_message_csv(const Store &store, const MessageId &id, TimeRange range, const Msg *msg) {
  std::ostringstream out;
  out << "mono_time,bus_time,bus,address,length,hex,decoded\n";
  const CanEventView view = store.canEvents(id, range);
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
  const CanEventView view = store.canEvents(id, range);
  for (const CanEvent &event : view.events) {
    double value = 0.0;
    if (!signal.getValue(event.data.data(), event.data.size(), &value)) continue;
    append_cell(&out, csv_time(event.mono_time));
    append_cell(&out, std::to_string(event.bus_time));
    append_cell(&out, std::to_string(id.source));
    append_cell(&out, address_hex(id.address));
    append_cell(&out, signal.name);
    append_cell(&out, doubleToString(value));
    append_cell(&out, signal.unit);
    append_cell(&out, hex_bytes(event.data), true);
  }
  return out.str();
}

bool write_csv_file(const std::filesystem::path &path, std::string_view csv, std::string *error) {
  if (path.empty()) {
    if (error != nullptr) *error = "empty export path";
    return false;
  }

  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      if (error != nullptr) *error = "failed to create export directory: " + ec.message();
      return false;
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    if (error != nullptr) *error = "failed to open export file";
    return false;
  }
  out.write(csv.data(), static_cast<std::streamsize>(csv.size()));
  if (!out.good()) {
    if (error != nullptr) *error = "failed to write export file";
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace loggy
