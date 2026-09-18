#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cabana { class Signal; }

namespace utils {

std::string formatSeconds(double sec, bool include_milliseconds = false, bool absolute_time = false);
std::string signalToolTip(const cabana::Signal *sig);

inline std::string trimmed(const std::string &s) {
  const char *ws = " \t\n\r\f\v";
  size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

inline bool containsCI(const std::string &s, const std::string &txt) {
  auto it = std::search(s.begin(), s.end(), txt.begin(), txt.end(),
                        [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); });
  return it != s.end();
}

// drop the tags of a rich text tooltip for the plain text imgui tooltip
inline std::string stripHtml(const std::string &s) {
  std::string out;
  bool in_tag = false;
  for (char c : s) {
    if (c == '<') in_tag = true;
    else if (c == '>') in_tag = false;
    else if (!in_tag) out += c;
  }
  return trimmed(out);
}

inline std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos; (pos = s.find(sep, start)) != std::string::npos; start = pos + 1) {
    parts.push_back(s.substr(start, pos - start));
  }
  parts.push_back(s.substr(start));
  return parts;
}

inline std::string toString(double v) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

// 0 when the text is not a valid number
inline double toDouble(const std::string &s) {
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  return (end != s.c_str() && *end == '\0') ? v : 0.0;
}

// 0 when the text is not fully consumed
inline int toInt(const std::string &s) {
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  return (end != s.c_str() && *end == '\0') ? (int)v : 0;
}

// 0 when the text is not fully consumed
inline unsigned long toULong(const std::string &s, int base = 10) {
  char *end = nullptr;
  unsigned long v = std::strtoul(s.c_str(), &end, base);
  return (end != s.c_str() && *end == '\0') ? v : 0;
}

// "00".."FF"
inline const char *hexByte(uint8_t value) {
  static const auto table = [] {
    std::array<char[3], 256> t;
    for (int i = 0; i < 256; ++i) snprintf(t[i], sizeof(t[i]), "%02X", i);
    return t;
  }();
  return table[value];
}

inline std::string toHex(const std::vector<uint8_t> &dat, char separator = '\0') {
  static const char digits[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(dat.size() * (separator ? 3 : 2));
  for (size_t i = 0; i < dat.size(); ++i) {
    if (separator && i) hex += separator;
    hex += digits[dat[i] >> 4];
    hex += digits[dat[i] & 0xf];
  }
  return hex;
}

inline std::string toHexString(int value) {
  char buf[16] = {};
  snprintf(buf, sizeof(buf), "0x%02X", value);
  return buf;
}

}  // namespace utils
