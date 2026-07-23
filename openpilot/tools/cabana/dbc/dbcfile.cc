#include "tools/cabana/dbc/dbcfile.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string &value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

bool startsWith(const std::string &value, const char *prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string unescapeComment(std::string value) {
  for (size_t pos = 0; (pos = value.find("\\\"", pos)) != std::string::npos; ++pos) {
    value.replace(pos, 2, "\"");
  }
  return trim(value);
}

bool commentComplete(const std::string &line) {
  bool escaped = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\\' && !escaped) {
      escaped = true;
      continue;
    }
    if (line[i] == '"' && !escaped) {
      size_t next = line.find_first_not_of(" \t\r\n", i + 1);
      if (next != std::string::npos && line[next] == ';') return true;
    }
    escaped = false;
  }
  return false;
}

}  // namespace

DBCFile::DBCFile(const std::string &dbc_file_name) {
  std::ifstream file(dbc_file_name, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open file.");
  filename = dbc_file_name;
  name_ = std::filesystem::path(dbc_file_name).stem().string();
  parse(std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()));
}

DBCFile::DBCFile(const std::string &name, const std::string &content) : name_(name) {
  parse(content);
}

bool DBCFile::save() {
  assert(!filename.empty());
  return writeContents(filename);
}

bool DBCFile::saveAs(const std::string &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::writeContents(const std::string &fn) {
  std::ofstream file(fn, std::ios::binary | std::ios::trunc);
  if (!file) return false;
  file << generateDBC();
  return file.good();
}

void DBCFile::updateMsg(const MessageId &id, const std::string &name, uint32_t size,
                        const std::string &node, const std::string &comment) {
  auto &m = msgs[id.address];
  m.address = id.address;
  m.name = name;
  m.size = size;
  m.transmitter = node.empty() ? DEFAULT_NODE_NAME : node;
  m.comment = comment;
}

cabana::Msg *DBCFile::msg(uint32_t address) {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

cabana::Msg *DBCFile::msg(const std::string &name) {
  auto it = std::find_if(msgs.begin(), msgs.end(), [&name](auto &m) { return m.second.name == name; });
  return it != msgs.end() ? &it->second : nullptr;
}

cabana::Signal *DBCFile::signal(uint32_t address, const std::string &name) {
  auto m = msg(address);
  return m ? m->sig(name) : nullptr;
}

void DBCFile::parse(const std::string &content) {
  msgs.clear();
  header.clear();
  std::istringstream input(content);
  std::string raw_line;
  cabana::Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  int line_num = 0;
  bool seen_first = false;

  while (std::getline(input, raw_line)) {
    ++line_num;
    const size_t first_nonspace = raw_line.find_first_not_of(" \t\r");
    std::string line = first_nonspace == std::string::npos ? std::string() : raw_line.substr(first_nonspace);
    const int statement_line = line_num;
    if ((startsWith(line, "CM_ BO_") || startsWith(line, "CM_ SG_ ")) && !commentComplete(line)) {
      std::string continuation;
      while (std::getline(input, continuation)) {
        ++line_num;
        line += "\n" + continuation;
        if (commentComplete(line)) break;
      }
    }

    bool seen = true;
    try {
      if (startsWith(line, "BO_ ")) {
        multiplexor_cnt = 0;
        current_msg = parseBO(line);
      } else if (startsWith(line, "SG_ ")) {
        parseSG(line, current_msg, multiplexor_cnt);
      } else if (startsWith(line, "VAL_ ")) {
        parseVAL(line);
      } else if (startsWith(line, "CM_ BO_")) {
        parseCM_BO(line);
      } else if (startsWith(line, "CM_ SG_ ")) {
        parseCM_SG(line);
      } else {
        seen = false;
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("[" + filename + ":" + std::to_string(statement_line) + "]" + e.what() + ": " + line);
    }
    if (seen) seen_first = true;
    else if (!seen_first) header += raw_line + "\n";
  }
  for (auto &[_, message] : msgs) message.update();
}

cabana::Msg *DBCFile::parseBO(const std::string &line) {
  static const std::regex pattern(R"(^BO_ ([[:alnum:]_]+) ([[:alnum:]_]+) *: ([[:alnum:]_]+) ([[:alnum:]_]+))");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) throw std::runtime_error("Invalid BO_ line format");
  const uint32_t address = std::stoul(match[1].str());
  if (msgs.count(address)) throw std::runtime_error("Duplicate message address: " + std::to_string(address));
  auto &message = msgs[address];
  message.address = address;
  message.name = match[2].str();
  message.size = std::stoul(match[3].str());
  message.transmitter = trim(match[4].str());
  return &message;
}

void DBCFile::parseSG(const std::string &line, cabana::Msg *current_msg, int &multiplexor_cnt) {
  static const std::regex pattern(R"dbc(^SG_ ([[:alnum:]_]+)(?: +([[:alnum:]_]+))? *: ([0-9]+)\|([0-9]+)@([0-9]+)([+-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] "(.*)" (.*))dbc");
  if (!current_msg) throw std::runtime_error("No Message");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) throw std::runtime_error("Invalid SG_ line format");
  if (current_msg->sig(match[1].str())) throw std::runtime_error("Duplicate signal name");

  cabana::Signal signal{};
  const std::string indicator = match[2].str();
  if (!indicator.empty()) {
    if (indicator == "M") {
      if (++multiplexor_cnt >= 2) throw std::runtime_error("Multiple multiplexor");
      signal.type = cabana::Signal::Type::Multiplexor;
    } else {
      signal.type = cabana::Signal::Type::Multiplexed;
      signal.multiplex_value = indicator.size() > 1 ? std::stoi(indicator.substr(1)) : 0;
    }
  }
  signal.name = match[1].str();
  signal.start_bit = std::stoi(match[3].str());
  signal.size = std::stoi(match[4].str());
  signal.is_little_endian = match[5].str() == "1";
  signal.is_signed = match[6].str() == "-";
  signal.factor = std::stod(match[7].str());
  signal.offset = std::stod(match[8].str());
  signal.min = std::stod(match[9].str());
  signal.max = std::stod(match[10].str());
  signal.unit = match[11].str();
  signal.receiver_name = trim(match[12].str());
  current_msg->sigs.push_back(new cabana::Signal(signal));
}

void DBCFile::parseCM_BO(const std::string &line) {
  std::istringstream prefix(line.substr(7));
  uint32_t address = 0;
  prefix >> address;
  const size_t first_quote = line.find('"');
  const size_t last_quote = line.rfind('"');
  if (!prefix || first_quote == std::string::npos || last_quote <= first_quote) {
    throw std::runtime_error("Invalid message comment format");
  }
  if (auto message = msg(address)) message->comment = unescapeComment(line.substr(first_quote + 1, last_quote - first_quote - 1));
}

void DBCFile::parseCM_SG(const std::string &line) {
  std::istringstream prefix(line.substr(7));
  uint32_t address = 0;
  std::string name;
  prefix >> address >> name;
  const size_t first_quote = line.find('"');
  const size_t last_quote = line.rfind('"');
  if (!prefix || name.empty() || first_quote == std::string::npos || last_quote <= first_quote) {
    throw std::runtime_error("Invalid CM_ SG_ line format");
  }
  if (auto sig = signal(address, name)) sig->comment = unescapeComment(line.substr(first_quote + 1, last_quote - first_quote - 1));
}

void DBCFile::parseVAL(const std::string &line) {
  static const std::regex header_pattern(R"(^VAL_ ([[:alnum:]_]+) ([[:alnum:]_]+) (.*))");
  static const std::regex entry_pattern(R"dbc(([+-]?[0-9]+(?:\.[0-9]+)?)\s+"([^"]*)")dbc");
  std::smatch match;
  if (!std::regex_search(line, match, header_pattern)) throw std::runtime_error("invalid VAL_ line format");
  if (auto sig = signal(std::stoul(match[1].str()), match[2].str())) {
    const std::string entries = match[3].str();
    for (std::sregex_iterator it(entries.begin(), entries.end(), entry_pattern), end; it != end; ++it) {
      sig->val_desc.emplace_back(std::stod((*it)[1].str()), trim((*it)[2].str()));
    }
  }
}

std::string DBCFile::generateDBC() {
  std::string dbc_string, comment, val_desc;
  for (const auto &[address, m] : msgs) {
    const std::string &transmitter = m.transmitter.empty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += "BO_ " + std::to_string(address) + " " + m.name + ": " + std::to_string(m.size) + " " + transmitter + "\n";
    if (!m.comment.empty()) {
      std::string escaped = m.comment;
      for (size_t pos = 0; (pos = escaped.find('"', pos)) != std::string::npos; pos += 2) escaped.replace(pos, 1, "\\\"");
      comment += "CM_ BO_ " + std::to_string(address) + " \"" + escaped + "\";\n";
    }
    for (auto sig : m.getSignals()) {
      std::string mux;
      if (sig->type == cabana::Signal::Type::Multiplexor) mux = "M ";
      else if (sig->type == cabana::Signal::Type::Multiplexed) mux = "m" + std::to_string(sig->multiplex_value) + " ";
      const std::string &receiver = sig->receiver_name.empty() ? DEFAULT_NODE_NAME : sig->receiver_name;
      dbc_string += " SG_ " + sig->name + " " + mux + ": " + std::to_string(sig->start_bit) + "|" + std::to_string(sig->size) + "@" +
                    (sig->is_little_endian ? "1" : "0") + (sig->is_signed ? "-" : "+") +
                    " (" + doubleToString(sig->factor) + "," + doubleToString(sig->offset) + ")" +
                    " [" + doubleToString(sig->min) + "|" + doubleToString(sig->max) + "] \"" + sig->unit + "\" " + receiver + "\n";
      if (!sig->comment.empty()) {
        std::string escaped = sig->comment;
        for (size_t pos = 0; (pos = escaped.find('"', pos)) != std::string::npos; pos += 2) escaped.replace(pos, 1, "\\\"");
        comment += "CM_ SG_ " + std::to_string(address) + " " + sig->name + " \"" + escaped + "\";\n";
      }
      if (!sig->val_desc.empty()) {
        std::string text;
        for (const auto &[value, description] : sig->val_desc) {
          if (!text.empty()) text += " ";
          text += doubleToString(value) + " \"" + description + "\"";
        }
        val_desc += "VAL_ " + std::to_string(address) + " " + sig->name + " " + text + ";\n";
      }
    }
    dbc_string += "\n";
  }
  return header + dbc_string + comment + val_desc;
}
