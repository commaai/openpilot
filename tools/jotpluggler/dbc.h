#pragma once

#include "common/util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dbc {


struct ValueDescriptionEntry {
  double value = 0.0;
  std::string text;
};

struct Signal {
  enum class Type {
    Normal = 0,
    Multiplexed,
    Multiplexor,
  };

  Type type = Type::Normal;
  std::string name;
  int start_bit = 0;
  int msb = 0;
  int lsb = 0;
  int size = 0;
  double factor = 1.0;
  double offset = 0.0;
  double min = 0.0;
  double max = 0.0;
  bool is_signed = false;
  bool is_little_endian = false;
  std::string unit;
  std::string comment;
  std::string receiver_name;
  int multiplex_value = 0;
  int multiplexor_index = -1;
  std::vector<ValueDescriptionEntry> value_descriptions;
};

struct Message {
  uint32_t address = 0;
  std::string name;
  uint32_t size = 0;
  std::string comment;
  std::string transmitter;
  std::vector<Signal> signals;
  int multiplexor_index = -1;

  const std::vector<Signal> &getSignals() const { return signals; }
};

class Database {
public:
  Database() = default;
  explicit Database(const std::filesystem::path &path);
  static Database fromContent(const std::string &content, const std::string &filename = "<dbc>");

  const Message *message(uint32_t address) const;
  const std::unordered_map<uint32_t, Message> &messages() const { return messages_; }
  std::vector<std::string> enumNames(const Signal &signal) const;

private:
  void parse(const std::string &content, const std::string &filename);
  void parseBo(const std::string &line, int line_number, Message **current_message);
  void parseSg(const std::string &line, int line_number, Message *current_message);
  void parseVal(const std::string &line, int line_number);
  void parseCmBo(const std::string &line, int line_number);
  void parseCmSg(const std::string &line, int line_number);
  void finalize();

  std::string filename_;
  std::unordered_map<uint32_t, Message> messages_;
};

void updateMsbLsb(Signal *signal);
double rawSignalValue(const Signal &signal, const uint8_t *data, size_t data_size);
std::optional<double> signalValue(const Signal &signal, const Message &message, const uint8_t *data, size_t data_size);

namespace {

std::string unescape_dbc_string(std::string text) {
  size_t pos = 0;
  while ((pos = text.find("\\\"", pos)) != std::string::npos) {
    text.replace(pos, 2, "\"");
    ++pos;
  }
  return text;
}

int flip_bit_pos(int start_bit) {
  return 8 * (start_bit / 8) + 7 - start_bit % 8;
}

std::string read_multiline_statement(std::istream &stream, std::string statement, int *line_number) {
  static const std::regex statement_end(R"(\"\s*;\s*$)");
  while (true) {
    const std::string trimmed = util::strip(statement);
    if (std::regex_search(trimmed, statement_end)) {
      return trimmed;
    }

    std::string next_line;
    if (!std::getline(stream, next_line)) {
      return trimmed;
    }
    statement += "\n";
    statement += next_line;
    ++(*line_number);
  }
}

}  // namespace

inline void updateMsbLsb(Signal *signal) {
  if (signal->is_little_endian) {
    signal->lsb = signal->start_bit;
    signal->msb = signal->start_bit + signal->size - 1;
  } else {
    signal->lsb = flip_bit_pos(flip_bit_pos(signal->start_bit) + signal->size - 1);
    signal->msb = signal->start_bit;
  }
}

inline double rawSignalValue(const Signal &signal, const uint8_t *data, size_t data_size) {
  const int msb_byte = signal.msb / 8;
  if (msb_byte >= static_cast<int>(data_size)) return 0.0;

  const int lsb_byte = signal.lsb / 8;
  uint64_t val = 0;
  if (msb_byte == lsb_byte) {
    val = (data[msb_byte] >> (signal.lsb & 7)) & ((1ULL << signal.size) - 1);
  } else {
    int bits = signal.size;
    int i = msb_byte;
    const int step = signal.is_little_endian ? -1 : 1;
    while (i >= 0 && i < static_cast<int>(data_size) && bits > 0) {
      const int msb = (i == msb_byte) ? signal.msb & 7 : 7;
      const int lsb = (i == lsb_byte) ? signal.lsb & 7 : 0;
      const int nbits = msb - lsb + 1;
      val = (val << nbits) | ((data[i] >> lsb) & ((1ULL << nbits) - 1));
      bits -= nbits;
      i += step;
    }
  }

  if (signal.is_signed && (val & (1ULL << (signal.size - 1)))) {
    val |= ~((1ULL << signal.size) - 1);
  }

  return static_cast<int64_t>(val) * signal.factor + signal.offset;
}

[[noreturn]] inline void parse_error(const std::string &filename, int line_number, const std::string &message, const std::string &line) {
  std::ostringstream out;
  out << "[" << filename << ":" << line_number << "] " << message << ": " << line;
  throw std::runtime_error(out.str());
}

inline Database::Database(const std::filesystem::path &path) {
  const std::string content = util::read_file(path.string());
  if (content.empty() && !std::filesystem::exists(path)) {
    throw std::runtime_error("Failed to open DBC " + path.string());
  }
  parse(content, path.filename().string());
}

inline Database Database::fromContent(const std::string &content, const std::string &filename) {
  Database db;
  db.parse(content, filename);
  return db;
}

inline const Message *Database::message(uint32_t address) const {
  auto it = messages_.find(address);
  return it == messages_.end() ? nullptr : &it->second;
}

inline std::vector<std::string> Database::enumNames(const Signal &signal) const {
  if (signal.value_descriptions.empty()) return {};
  int max_index = -1;
  for (const auto &entry : signal.value_descriptions) {
    const double rounded = std::round(entry.value);
    if (std::abs(entry.value - rounded) > 1e-6 || rounded < 0.0 || rounded > 512.0) return {};
    max_index = std::max(max_index, static_cast<int>(rounded));
  }
  if (max_index < 0) return {};
  std::vector<std::string> names(static_cast<size_t>(max_index + 1));
  for (const auto &entry : signal.value_descriptions) {
    names[static_cast<size_t>(std::llround(entry.value))] = entry.text;
  }
  return names;
}

inline void Database::parse(const std::string &content, const std::string &filename) {
  filename_ = filename;
  messages_.clear();
  std::istringstream stream(content);
  std::string raw_line;
  Message *current_message = nullptr;
  int line_number = 0;
  while (std::getline(stream, raw_line)) {
    ++line_number;
    std::string line = util::strip(raw_line);
    if (line.empty()) continue;
    if (util::starts_with(line, "BO_ ")) {
      parseBo(line, line_number, &current_message);
    } else if (util::starts_with(line, "SG_ ")) {
      if (current_message == nullptr) {
        parse_error(filename, line_number, "Signal without current message", line);
      }
      parseSg(line, line_number, current_message);
    } else if (util::starts_with(line, "VAL_ ")) {
      parseVal(line, line_number);
    } else if (util::starts_with(line, "CM_ BO_")) {
      parseCmBo(read_multiline_statement(stream, raw_line, &line_number), line_number);
    } else if (util::starts_with(line, "CM_ SG_")) {
      parseCmSg(read_multiline_statement(stream, raw_line, &line_number), line_number);
    }
  }
  finalize();
}

inline void Database::parseBo(const std::string &line, int line_number, Message **current_message) {
  static const std::regex pattern(R"(^BO_\s+(\w+)\s+(\w+)\s*:\s*(\w+)\s+(\w+)\s*$)");
  std::smatch match;
  if (!std::regex_match(line, match, pattern)) {
    parse_error("<dbc>", line_number, "Invalid BO_ line format", line);
  }
  uint32_t address = static_cast<uint32_t>(std::stoul(match[1].str(), nullptr, 0));
  if (messages_.find(address) != messages_.end()) {
    parse_error(filename_, line_number, "Duplicate message address", line);
  }
  Message &message = messages_[address];
  message.address = address;
  message.name = match[2].str();
  message.size = static_cast<uint32_t>(std::stoul(match[3].str(), nullptr, 0));
  message.transmitter = match[4].str();
  message.signals.clear();
  message.multiplexor_index = -1;
  *current_message = &message;
}

inline void Database::parseSg(const std::string &line, int line_number, Message *current_message) {
  static const std::regex multiplex_pattern(R"(^SG_\s+(\w+)\s+(\w+)\s*:\s*(\d+)\|(\d+)@(\d)([+-])\s+\(([0-9.+\-eE]+),([0-9.+\-eE]+)\)\s+\[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\]\s+\"(.*)\"\s+(.*)$)");
  static const std::regex normal_pattern(R"(^SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@(\d)([+-])\s+\(([0-9.+\-eE]+),([0-9.+\-eE]+)\)\s+\[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\]\s+\"(.*)\"\s+(.*)$)");

  std::smatch match;
  Signal signal;
  int offset = 0;
  if (std::regex_match(line, match, normal_pattern)) {
    offset = 0;
  } else if (std::regex_match(line, match, multiplex_pattern)) {
    offset = 1;
    const std::string indicator = match[2].str();
    if (indicator == "M") {
      if (std::any_of(current_message->signals.begin(), current_message->signals.end(), [](const Signal &existing) {
            return existing.type == Signal::Type::Multiplexor;
          })) {
        parse_error(filename_, line_number, "Multiple multiplexor", line);
      }
      signal.type = Signal::Type::Multiplexor;
    } else if (!indicator.empty() && indicator.front() == 'm') {
      signal.type = Signal::Type::Multiplexed;
      signal.multiplex_value = std::stoi(indicator.substr(1));
    } else {
      parse_error("<dbc>", line_number, "Invalid multiplex indicator", line);
    }
  } else {
    parse_error("<dbc>", line_number, "Invalid SG_ line format", line);
  }

  signal.name = match[1].str();
  if (std::any_of(current_message->signals.begin(), current_message->signals.end(), [&](const Signal &existing) {
        return existing.name == signal.name;
      })) {
    parse_error(filename_, line_number, "Duplicate signal name", line);
  }
  signal.start_bit = std::stoi(match[2 + offset].str());
  signal.size = std::stoi(match[3 + offset].str());
  signal.is_little_endian = match[4 + offset].str() == "1";
  signal.is_signed = match[5 + offset].str() == "-";
  signal.factor = std::stod(match[6 + offset].str());
  signal.offset = std::stod(match[7 + offset].str());
  signal.min = std::stod(match[8 + offset].str());
  signal.max = std::stod(match[9 + offset].str());
  signal.unit = match[10 + offset].str();
  signal.receiver_name = util::strip(match[11 + offset].str());
  updateMsbLsb(&signal);
  current_message->signals.push_back(std::move(signal));
}

inline void Database::parseVal(const std::string &line, int line_number) {
  static const std::regex prefix(R"(^VAL_\s+(\w+)\s+(\w+)\s+(.*);$)");
  std::smatch match;
  if (!std::regex_match(line, match, prefix)) {
    parse_error("<dbc>", line_number, "Invalid VAL_ line format", line);
  }

  const uint32_t address = static_cast<uint32_t>(std::stoul(match[1].str(), nullptr, 0));
  auto msg_it = messages_.find(address);
  if (msg_it == messages_.end()) {
    return;
  }
  auto sig_it = std::find_if(msg_it->second.signals.begin(), msg_it->second.signals.end(), [&](const Signal &signal) {
    return signal.name == match[2].str();
  });
  if (sig_it == msg_it->second.signals.end()) {
    return;
  }

  static const std::regex entry_pattern(R"(([+-]?\d+(?:\.\d+)?)\s+\"((?:[^\"\\]|\\.)*)\")");
  const std::string defs = match[3].str();
  for (std::sregex_iterator it(defs.begin(), defs.end(), entry_pattern), end; it != end; ++it) {
    sig_it->value_descriptions.push_back(ValueDescriptionEntry{
      .value = std::stod((*it)[1].str()),
      .text = (*it)[2].str(),
    });
  }
}

inline void Database::parseCmBo(const std::string &line, int line_number) {
  static const std::regex pattern(R"(^CM_\s+BO_\s*(\w+)\s*\"((?:[^\"\\]|\\.|[\r\n])*)\"\s*;\s*$)");
  std::smatch match;
  if (!std::regex_match(line, match, pattern)) {
    parse_error(filename_, line_number, "Invalid message comment format", line);
  }
  const uint32_t address = static_cast<uint32_t>(std::stoul(match[1].str(), nullptr, 0));
  auto it = messages_.find(address);
  if (it != messages_.end()) {
    it->second.comment = unescape_dbc_string(match[2].str());
  }
}

inline void Database::parseCmSg(const std::string &line, int line_number) {
  static const std::regex pattern(R"(^CM_\s+SG_\s*(\w+)\s*(\w+)\s*\"((?:[^\"\\]|\\.|[\r\n])*)\"\s*;\s*$)");
  std::smatch match;
  if (!std::regex_match(line, match, pattern)) {
    parse_error(filename_, line_number, "Invalid signal comment format", line);
  }

  const uint32_t address = static_cast<uint32_t>(std::stoul(match[1].str(), nullptr, 0));
  auto msg_it = messages_.find(address);
  if (msg_it == messages_.end()) return;

  auto sig_it = std::find_if(msg_it->second.signals.begin(), msg_it->second.signals.end(), [&](const Signal &signal) {
    return signal.name == match[2].str();
  });
  if (sig_it != msg_it->second.signals.end()) {
    sig_it->comment = unescape_dbc_string(match[3].str());
  }
}

inline void Database::finalize() {
  for (auto &[_, message] : messages_) {
    std::sort(message.signals.begin(), message.signals.end(), [](const Signal &left, const Signal &right) {
      return std::tie(right.type, left.multiplex_value, left.start_bit, left.name)
           < std::tie(left.type, right.multiplex_value, right.start_bit, right.name);
    });
    message.multiplexor_index = -1;
    for (size_t i = 0; i < message.signals.size(); ++i) {
      if (message.signals[i].type == Signal::Type::Multiplexor) {
        message.multiplexor_index = static_cast<int>(i);
        break;
      }
    }
    for (Signal &signal : message.signals) {
      signal.multiplexor_index = signal.type == Signal::Type::Multiplexed ? message.multiplexor_index : -1;
      if (signal.type == Signal::Type::Multiplexed && signal.multiplexor_index < 0) {
        signal.type = Signal::Type::Normal;
        signal.multiplex_value = 0;
      }
    }
  }
}

inline std::optional<double> signalValue(const Signal &signal, const Message &message, const uint8_t *data, size_t data_size) {
  if (signal.multiplexor_index >= 0) {
    const Signal &multiplexor = message.signals[static_cast<size_t>(signal.multiplexor_index)];
    const double mux_value = rawSignalValue(multiplexor, data, data_size);
    if (std::llround(mux_value) != signal.multiplex_value) return std::nullopt;
  }
  return rawSignalValue(signal, data, data_size);
}

}  // namespace dbc
