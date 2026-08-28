#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cabana { class Signal; }

namespace utils {

std::string formatSeconds(double sec, bool include_milliseconds = false, bool absolute_time = false);
std::string signalToolTip(const cabana::Signal *sig);

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
