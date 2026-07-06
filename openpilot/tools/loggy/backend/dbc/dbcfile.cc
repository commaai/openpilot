#include "tools/loggy/backend/dbc/dbcfile.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

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
  size_t start_ = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == sep) {
      parts.push_back(s.substr(start_, i - start_));
      start_ = i + 1;
    }
  }
  return parts;
}

bool startsWith(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &s, const char *suffix) {
  const std::string suffix_str(suffix);
  return s.size() >= suffix_str.size() &&
         s.compare(s.size() - suffix_str.size(), suffix_str.size(), suffix_str) == 0;
}

std::string strip(const std::string &str) {
  auto should_trim = [](unsigned char ch) {
    return std::isspace(ch) || ch == '\0';
  };

  size_t start_ = 0;
  while (start_ < str.size() && should_trim(static_cast<unsigned char>(str[start_]))) {
    ++start_;
  }
  if (start_ == str.size()) return "";

  size_t end = str.size() - 1;
  while (end > 0 && should_trim(static_cast<unsigned char>(str[end]))) {
    --end;
  }
  return str.substr(start_, end - start_ + 1);
}

// Finds the end of a quoted, possibly-escaped string. `pos` is the index of
// the first character after the opening quote. Mirrors what the regex group
// `(?:[^"\\]|\\.)*` (followed by a literal `"`) used to match: any run of
// characters where a `"` only terminates the string if it isn't preceded by
// an (unescaped) backslash. Returns the index of the closing quote, or
// std::string::npos if the string is unterminated.
//
// This exists so we can scan quoted text with a plain linear loop instead of
// a regex. libstdc++'s std::regex executor (_M_dfs) recurses once per
// backtracking step of a repeated group like the one above, so matching it
// against a long string (a large VAL_ enum table, or a long CM_ comment) can
// overflow the stack even though the match itself is unambiguous -- see the
// std::regex "catastrophic stack depth" issue. A hand-rolled scan has no
// such recursion.
size_t findClosingQuote(const std::string &s, size_t pos) {
  for (; pos < s.size(); ++pos) {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      ++pos;  // escaped character (e.g. \" or \\); doesn't terminate the string
    } else if (s[pos] == '"') {
      return pos;
    }
  }
  return std::string::npos;
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

namespace loggy {

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
  return write_contents(filename);
}

bool DBCFile::save_as(const std::string &new_filename) {
  if (!write_contents(new_filename)) return false;
  filename = new_filename;
  return true;
}

bool DBCFile::write_contents(const std::string &fn) {
  std::ofstream file(fn, std::ios::binary);
  if (!file.is_open()) return false;
  std::string content = generate_dbc();
  file.write(content.c_str(), content.size());
  return file.good();
}

void DBCFile::update_msg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
  auto &m = msgs[id.address];
  m.address = id.address;
  m.name = name;
  m.size = size;
  m.transmitter = node.empty() ? DEFAULT_NODE_NAME : node;
  m.comment = comment;
  m.update();
}

Msg *DBCFile::msg(uint32_t address) {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

Msg *DBCFile::msg(const std::string &name) {
  auto it = std::find_if(msgs.begin(), msgs.end(), [&name](auto &m) { return m.second.name == name; });
  return it != msgs.end() ? &(it->second) : nullptr;
}

Signal *DBCFile::signal(uint32_t address, const std::string &name) {
  auto m = msg(address);
  return m ? (Signal *)m->sig(name) : nullptr;
}

void DBCFile::parse(const std::string &content) {
  msgs.clear();

  int line_num = 0;
  Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  bool seen_first = false;
  std::istringstream stream(content);
  std::string raw_line;
  size_t offset = 0;

  while (std::getline(stream, raw_line)) {
    ++line_num;
    size_t line_offset = offset;
    offset += raw_line.size() + 1;
    std::string line = strip(raw_line);

    bool seen = true;
    try {
      if (startsWith(line, "BO_ ")) {
        multiplexor_cnt = 0;
        current_msg = parse_bo(line);
      } else if (startsWith(line, "SG_ ")) {
        parse_sg(line, current_msg, multiplexor_cnt);
      } else if (startsWith(line, "VAL_ ")) {
        parse_val(line);
      } else if (startsWith(line, "CM_ BO_")) {
        parse_cm_bo(line, content, line_offset);
      } else if (startsWith(line, "CM_ SG_ ")) {
        parse_cm_sg(line, content, line_offset);
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

Msg *DBCFile::parse_bo(const std::string &line) {
  static const std::regex bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");

  std::smatch match;
  if (!std::regex_search(line, match, bo_regexp))
    throw std::runtime_error("Invalid BO_ line format");

  uint32_t address = toUInt(match[1].str());
  if (msgs.count(address) > 0)
    throw std::runtime_error("Duplicate message address: " + std::to_string(address));

  // Create a new message object
  Msg *msg = &msgs[address];
  msg->address = address;
  msg->name = match[2].str();
  msg->size = toUInt(match[3].str());
  msg->transmitter = strip(match[4].str());
  return msg;
}

void DBCFile::parse_cm_bo(const std::string &line, const std::string &content, size_t line_offset) {
  // Only the short, fixed-shape prefix (up to and including the opening
  // quote of the comment text) is matched with a regex. The comment text
  // itself -- which can be arbitrarily long -- is scanned with
  // findClosingQuote() instead of a regex, to avoid unbounded std::regex
  // backtracking recursion on long comments.
  static const std::regex msg_comment_prefix_regexp(R"(^CM_ BO_ *(\w+) *\")");

  bool single_line = endsWith(line, "\";");
  const std::string &text = single_line ? line : content;
  size_t base = single_line ? 0 : line_offset;

  std::smatch match;
  if (!std::regex_search(text.cbegin() + base, text.cend(), match, msg_comment_prefix_regexp))
    throw std::runtime_error("Invalid message comment format");

  size_t text_start = base + match.position(0) + match.length(0);
  size_t quote_end = findClosingQuote(text, text_start);
  if (quote_end == std::string::npos)
    throw std::runtime_error("Invalid message comment format");

  size_t p = quote_end + 1;
  while (p < text.size() && std::isspace((unsigned char)text[p])) ++p;
  if (p >= text.size() || text[p] != ';')
    throw std::runtime_error("Invalid message comment format");

  if (auto m = msg(toUInt(match[1].str())))
    m->comment = unescapeQuotes(strip(text.substr(text_start, quote_end - text_start)));
}

void DBCFile::parse_sg(const std::string &line, Msg *current_msg, int &multiplexor_cnt) {
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

  Signal s{};
  if (offset == 1) {
    std::string indicator = match[2].str();
    if (indicator == "M") {
      ++multiplexor_cnt;
      // Only one signal within a single message can be the multiplexer switch.
      if (multiplexor_cnt >= 2)
        throw std::runtime_error("Multiple multiplexor");

      s.type = Signal::Type::Multiplexor;
    } else {
      s.type = Signal::Type::Multiplexed;
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
  s.receiver_name = strip(match[11 + offset].str());
  current_msg->sigs.push_back(new Signal(s));
}

void DBCFile::parse_cm_sg(const std::string &line, const std::string &content, size_t line_offset) {
  // See parse_cm_bo() for why the comment text is scanned by hand rather than
  // matched with a regex.
  static const std::regex sg_comment_prefix_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\")");

  bool single_line = endsWith(line, "\";");
  const std::string &text = single_line ? line : content;
  size_t base = single_line ? 0 : line_offset;

  std::smatch match;
  if (!std::regex_search(text.cbegin() + base, text.cend(), match, sg_comment_prefix_regexp))
    throw std::runtime_error("Invalid CM_ SG_ line format");

  size_t text_start = base + match.position(0) + match.length(0);
  size_t quote_end = findClosingQuote(text, text_start);
  if (quote_end == std::string::npos)
    throw std::runtime_error("Invalid CM_ SG_ line format");

  size_t p = quote_end + 1;
  while (p < text.size() && std::isspace((unsigned char)text[p])) ++p;
  if (p >= text.size() || text[p] != ';')
    throw std::runtime_error("Invalid CM_ SG_ line format");

  if (auto s = signal(toUInt(match[1].str()), match[2].str())) {
    s->comment = unescapeQuotes(strip(text.substr(text_start, quote_end - text_start)));
  }
}

void DBCFile::parse_val(const std::string &line) {
  // Only the fixed-shape "VAL_ <addr> <name> " prefix is matched with a
  // regex. The entry list ("<value> \"<description>\" ..." repeated,
  // sometimes thousands of times in vendor/diagnostic DBCs) used to be
  // captured whole by one regex and then split on '"'. libstdc++'s
  // std::regex executor recurses once per backtracking step of a repeated
  // group, so that single capture could overflow the stack on a long entry
  // list even though nothing about the match was ambiguous. Instead, find
  // the same span with plain string scans (linear, no recursion), then feed
  // it through the exact same splitOnChar()-based parsing as before.
  static const std::regex val_prefix_regexp(R"(^VAL_ (\w+) (\w+) )");

  std::smatch match;
  if (!std::regex_search(line, match, val_prefix_regexp))
    throw std::runtime_error("invalid VAL_ line format");

  size_t prefix_end = match.position(0) + match.length(0);

  // Original group 3 was `\s*[-+]?[0-9]+\s+\".+?\"[^;]*`: it required the
  // text up to the first quote to look like a number, and the first
  // description to be non-empty. Just require *some* non-blank text there --
  // this is at least as permissive as the old regex and every existing entry
  // list (including all of opendbc) satisfies it either way.
  size_t first_quote = line.find('"', prefix_end);
  if (first_quote == std::string::npos ||
      strip(line.substr(prefix_end, first_quote - prefix_end)).empty())
    throw std::runtime_error("invalid VAL_ line format");

  size_t first_desc_end = line.find('"', first_quote + 1);
  if (first_desc_end == std::string::npos)
    throw std::runtime_error("invalid VAL_ line format");

  // Matches `[^;]*`: the entry list runs up to (not including) the first ';'
  // found after the first description closes, or to the end of the line.
  size_t semi = line.find(';', first_desc_end + 1);
  size_t entries_end = (semi == std::string::npos) ? line.size() : semi;

  if (auto s = signal(toUInt(match[1].str()), match[2].str())) {
    std::string entries_text = line.substr(prefix_end, entries_end - prefix_end);
    std::vector<std::string> desc_list = splitOnChar(strip(entries_text), '"');
    for (size_t i = 0; i < desc_list.size(); i += 2) {
      auto val = strip(desc_list[i]);
      if (!val.empty() && (i + 1) < desc_list.size()) {
        auto desc = strip(desc_list[i + 1]);
        s->val_desc.push_back({toDouble(val), desc});
      }
    }
  }
}

std::string DBCFile::generate_dbc() {
  // Legacy Cabana writer behavior: BA_ attributes, BO_TX_BU_ declarations,
  // and signal-less BO_ messages are not re-emitted. Keep this behavior
  // byte-compatible for the first lift; fix it only in a separate patch with
  // explicit migration tests.
  std::string dbc_string, comment, val_desc;
  for (const auto &[address, m] : msgs) {
    if (m.signals().empty()) continue;
    const std::string &transmitter = m.transmitter.empty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += "BO_ " + std::to_string(address) + " " + m.name + ": " + std::to_string(m.size) + " " + transmitter + "\n";
    if (!m.comment.empty()) {
      std::string escaped_comment = m.comment;
      // Replace " with \"
      for (size_t pos = 0; (pos = escaped_comment.find('"', pos)) != std::string::npos; pos += 2)
        escaped_comment.replace(pos, 1, "\\\"");
      comment += "CM_ BO_ " + std::to_string(address) + " \"" + escaped_comment + "\";\n";
    }
    for (auto sig : m.signals()) {
      std::string multiplexer_indicator;
      if (sig->type == Signal::Type::Multiplexor) {
        multiplexer_indicator = "M ";
      } else if (sig->type == Signal::Type::Multiplexed) {
        multiplexer_indicator = "m" + std::to_string(sig->multiplex_value) + " ";
      }
      const std::string &recv = sig->receiver_name.empty() ? DEFAULT_NODE_NAME : sig->receiver_name;
      dbc_string += " SG_ " + sig->name + " " + multiplexer_indicator + ": " +
                    std::to_string(sig->start_bit) + "|" + std::to_string(sig->size) + "@" +
                    std::string(1, sig->is_little_endian ? '1' : '0') +
                    std::string(1, sig->is_signed ? '-' : '+') +
                    " (" + double_to_string(sig->factor) + "," + double_to_string(sig->offset) + ")" +
                    " [" + double_to_string(sig->min) + "|" + double_to_string(sig->max) + "]" +
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

}  // namespace loggy
