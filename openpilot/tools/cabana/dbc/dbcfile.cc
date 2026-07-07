#include "tools/cabana/dbc/dbcfile.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "common/util.h"

namespace {

std::string dbcBaseName(const std::string &path) {
  size_t slash = path.find_last_of('/');
  std::string fname = slash == std::string::npos ? path : path.substr(slash + 1);
  size_t dot = fname.find('.');
  return dot == std::string::npos ? fname : fname.substr(0, dot);
}

std::string unescapeQuotes(std::string s) {
  size_t pos = 0;
  while ((pos = s.find("\\\"", pos)) != std::string::npos) {
    s.replace(pos, 2, "\"");
    pos += 1;
  }
  return s;
}

std::vector<std::string> splitOnChar(const std::string &s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == sep) {
      parts.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return parts;
}

// Qt's QString::toUInt/toInt/toDouble never throw: they return 0 on any parse
// failure (empty string, non-numeric, overflow). std::stoul/stoi/stod throw
// in those cases, so wrap them to keep parsing behavior identical.
uint32_t toUInt(const std::string &s) {
  try {
    return static_cast<uint32_t>(std::stoul(s));
  } catch (const std::exception &) {
    return 0;
  }
}

int toInt(const std::string &s) {
  try {
    return std::stoi(s);
  } catch (const std::exception &) {
    return 0;
  }
}

double toDouble(const std::string &s) {
  try {
    return std::stod(s);
  } catch (const std::exception &) {
    return 0.0;
  }
}

}  // namespace

DBCFile::DBCFile(const std::string &dbc_file_name) {
  std::ifstream file(dbc_file_name, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file.");
  }
  name_ = dbcBaseName(dbc_file_name);
  filename = dbc_file_name;
  std::ostringstream ss;
  ss << file.rdbuf();
  parse(ss.str());
}

DBCFile::DBCFile(const std::string &name, const std::string &content) : name_(name), filename("") {
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
  std::ofstream file(fn, std::ios::binary);
  if (!file.is_open()) return false;
  std::string content = generateDBC();
  file.write(content.c_str(), content.size());
  return file.good();
}

void DBCFile::updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
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
  return it != msgs.end() ? &(it->second) : nullptr;
}

cabana::Signal *DBCFile::signal(uint32_t address, const std::string &name) {
  auto m = msg(address);
  return m ? (cabana::Signal *)m->sig(name) : nullptr;
}

void DBCFile::parse(const std::string &content) {
  msgs.clear();

  int line_num = 0;
  cabana::Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  bool seen_first = false;
  std::istringstream stream(content);
  std::string raw_line;
  size_t offset = 0;

  while (std::getline(stream, raw_line)) {
    ++line_num;
    size_t line_offset = offset;
    offset += raw_line.size() + 1;
    std::string line = util::strip(raw_line);

    bool seen = true;
    try {
      if (util::starts_with(line, "BO_ ")) {
        multiplexor_cnt = 0;
        current_msg = parseBO(line);
      } else if (util::starts_with(line, "SG_ ")) {
        parseSG(line, current_msg, multiplexor_cnt);
      } else if (util::starts_with(line, "VAL_ ")) {
        parseVAL(line);
      } else if (util::starts_with(line, "CM_ BO_")) {
        parseCM_BO(line, content, line_offset);
      } else if (util::starts_with(line, "CM_ SG_ ")) {
        parseCM_SG(line, content, line_offset);
      } else {
        seen = false;
      }
    } catch (std::exception &e) {
      throw std::runtime_error("[" + filename + ":" + std::to_string(line_num) + "]" + e.what() + ": " + line);
    }

    if (seen) {
      seen_first = true;
    } else if (!seen_first) {
      header += raw_line + "\n";
    }
  }

  for (auto &[_, m] : msgs) {
    m.update();
  }
}

cabana::Msg *DBCFile::parseBO(const std::string &line) {
  static const std::regex bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");

  std::smatch match;
  if (!std::regex_search(line, match, bo_regexp))
    throw std::runtime_error("Invalid BO_ line format");

  uint32_t address = toUInt(match[1].str());
  if (msgs.count(address) > 0)
    throw std::runtime_error("Duplicate message address: " + std::to_string(address));

  // Create a new message object
  cabana::Msg *msg = &msgs[address];
  msg->address = address;
  msg->name = match[2].str();
  msg->size = toUInt(match[3].str());
  msg->transmitter = util::strip(match[4].str());
  return msg;
}

void DBCFile::parseCM_BO(const std::string &line, const std::string &content, size_t line_offset) {
  static const std::regex msg_comment_regexp(R"(^CM_ BO_ *(\w+) *\"((?:[^"\\]|\\.)*)\"\s*;)");

  std::smatch match;
  bool matched;
  if (util::ends_with(line, "\";")) {
    matched = std::regex_search(line, match, msg_comment_regexp);
  } else {
    matched = std::regex_search(content.cbegin() + line_offset, content.cend(), match, msg_comment_regexp);
  }
  if (!matched)
    throw std::runtime_error("Invalid message comment format");

  if (auto m = msg(toUInt(match[1].str())))
    m->comment = unescapeQuotes(util::strip(match[2].str()));
}

void DBCFile::parseSG(const std::string &line, cabana::Msg *current_msg, int &multiplexor_cnt) {
  static const std::regex sg_regexp(R"(^SG_ (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static const std::regex sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");

  if (!current_msg)
    throw std::runtime_error("No Message");

  int offset = 0;
  std::smatch match;
  bool matched = std::regex_search(line, match, sg_regexp);
  if (!matched) {
    matched = std::regex_search(line, match, sgm_regexp);
    offset = 1;
  }
  if (!matched)
    throw std::runtime_error("Invalid SG_ line format");

  std::string name = match[1].str();
  if (current_msg->sig(name) != nullptr)
    throw std::runtime_error("Duplicate signal name");

  cabana::Signal s{};
  if (offset == 1) {
    std::string indicator = match[2].str();
    if (indicator == "M") {
      ++multiplexor_cnt;
      // Only one signal within a single message can be the multiplexer switch.
      if (multiplexor_cnt >= 2)
        throw std::runtime_error("Multiple multiplexor");

      s.type = cabana::Signal::Type::Multiplexor;
    } else {
      s.type = cabana::Signal::Type::Multiplexed;
      s.multiplex_value = toInt(indicator.substr(1));
    }
  }
  s.name = name;
  s.start_bit = toInt(match[offset + 2].str());
  s.size = toInt(match[offset + 3].str());
  s.is_little_endian = toInt(match[offset + 4].str()) == 1;
  s.is_signed = match[offset + 5].str() == "-";
  s.factor = toDouble(match[offset + 6].str());
  s.offset = toDouble(match[offset + 7].str());
  s.min = toDouble(match[8 + offset].str());
  s.max = toDouble(match[9 + offset].str());
  s.unit = match[10 + offset].str();
  s.receiver_name = util::strip(match[11 + offset].str());
  current_msg->sigs.push_back(new cabana::Signal(s));
}

void DBCFile::parseCM_SG(const std::string &line, const std::string &content, size_t line_offset) {
  static const std::regex sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"((?:[^"\\]|\\.)*)\"\s*;)");

  std::smatch match;
  bool matched;
  if (util::ends_with(line, "\";")) {
    matched = std::regex_search(line, match, sg_comment_regexp);
  } else {
    matched = std::regex_search(content.cbegin() + line_offset, content.cend(), match, sg_comment_regexp);
  }
  if (!matched)
    throw std::runtime_error("Invalid CM_ SG_ line format");

  if (auto s = signal(toUInt(match[1].str()), match[2].str())) {
    s->comment = unescapeQuotes(util::strip(match[3].str()));
  }
}

void DBCFile::parseVAL(const std::string &line) {
  static const std::regex val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

  std::smatch match;
  if (!std::regex_search(line, match, val_regexp))
    throw std::runtime_error("invalid VAL_ line format");

  if (auto s = signal(toUInt(match[1].str()), match[2].str())) {
    std::vector<std::string> desc_list = splitOnChar(util::strip(match[3].str()), '"');
    for (size_t i = 0; i < desc_list.size(); i += 2) {
      auto val = util::strip(desc_list[i]);
      if (!val.empty() && (i + 1) < desc_list.size()) {
        auto desc = util::strip(desc_list[i + 1]);
        s->val_desc.push_back({toDouble(val), desc});
      }
    }
  }
}

std::string DBCFile::generateDBC() {
  std::string dbc_string, comment, val_desc;
  for (const auto &[address, m] : msgs) {
    const std::string &transmitter = m.transmitter.empty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += "BO_ " + std::to_string(address) + " " + m.name + ": " + std::to_string(m.size) + " " + transmitter + "\n";
    if (!m.comment.empty()) {
      std::string escaped_comment = m.comment;
      // Replace " with \"
      for (size_t pos = 0; (pos = escaped_comment.find('"', pos)) != std::string::npos; pos += 2)
        escaped_comment.replace(pos, 1, "\\\"");
      comment += "CM_ BO_ " + std::to_string(address) + " \"" + escaped_comment + "\";\n";
    }
    for (auto sig : m.getSignals()) {
      std::string multiplexer_indicator;
      if (sig->type == cabana::Signal::Type::Multiplexor) {
        multiplexer_indicator = "M ";
      } else if (sig->type == cabana::Signal::Type::Multiplexed) {
        multiplexer_indicator = "m" + std::to_string(sig->multiplex_value) + " ";
      }
      const std::string &recv = sig->receiver_name.empty() ? DEFAULT_NODE_NAME : sig->receiver_name;
      dbc_string += " SG_ " + sig->name + " " + multiplexer_indicator + ": " +
                    std::to_string(sig->start_bit) + "|" + std::to_string(sig->size) + "@" +
                    std::string(1, sig->is_little_endian ? '1' : '0') +
                    std::string(1, sig->is_signed ? '-' : '+') +
                    " (" + doubleToString(sig->factor) + "," + doubleToString(sig->offset) + ")" +
                    " [" + doubleToString(sig->min) + "|" + doubleToString(sig->max) + "]" +
                    " \"" + sig->unit + "\" " + recv + "\n";
      if (!sig->comment.empty()) {
        std::string escaped_comment = sig->comment;
        for (size_t pos = 0; (pos = escaped_comment.find('"', pos)) != std::string::npos; pos += 2)
          escaped_comment.replace(pos, 1, "\\\"");
        comment += "CM_ SG_ " + std::to_string(address) + " " + sig->name + " \"" + escaped_comment + "\";\n";
      }
      if (!sig->val_desc.empty()) {
        std::string text;
        for (auto &[val, desc] : sig->val_desc) {
          if (!text.empty()) text += " ";
          char val_buf[64];
          snprintf(val_buf, sizeof(val_buf), "%g", val);
          text += std::string(val_buf) + " \"" + desc + "\"";
        }
        val_desc += "VAL_ " + std::to_string(address) + " " + sig->name + " " + text + ";\n";
      }
    }
    dbc_string += "\n";
  }
  return header + dbc_string + comment + val_desc;
}
