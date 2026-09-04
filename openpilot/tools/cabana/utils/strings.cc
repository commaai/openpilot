#include "tools/cabana/utils/strings.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <sstream>

#include "tools/cabana/dbc/dbc.h"

namespace utils {

std::string formatSeconds(double sec, bool include_milliseconds, bool absolute_time) {
  char out[80] = {};
  if (absolute_time) {
    const auto ms_total = static_cast<int64_t>(std::llround(sec * 1000.0));
    const std::time_t secs = static_cast<std::time_t>(ms_total / 1000);
    int millis = static_cast<int>(ms_total % 1000);
    if (millis < 0) millis = -millis;
    std::tm tm{};
    localtime_r(&secs, &tm);
    char buf[64] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    if (!include_milliseconds) return buf;
    snprintf(out, sizeof(out), "%s.%03d", buf, millis);
    return out;
  }

  // Relative duration (not wall-clock).
  const bool show_hours = sec > 60 * 60;
  int total_ms = static_cast<int>(std::llround(std::max(0.0, sec) * 1000.0));
  const int hours = total_ms / (3600 * 1000);
  const int minutes = (total_ms / (60 * 1000)) % 60;
  const int seconds = (total_ms / 1000) % 60;
  const int millis = total_ms % 1000;
  if (show_hours && include_milliseconds) {
    snprintf(out, sizeof(out), "%02d:%02d:%02d.%03d", hours, minutes, seconds, millis);
  } else if (show_hours) {
    snprintf(out, sizeof(out), "%02d:%02d:%02d", hours, minutes, seconds);
  } else if (include_milliseconds) {
    snprintf(out, sizeof(out), "%02d:%02d.%03d", minutes, seconds, millis);
  } else {
    snprintf(out, sizeof(out), "%02d:%02d", minutes, seconds);
  }
  return out;
}

std::string signalToolTip(const cabana::Signal *sig) {
  std::ostringstream s;
  s << "\n    " << sig->name << "<br /><span font-size:small\">\n"
    << "    Start Bit: " << sig->start_bit << " Size: " << sig->size << "<br />\n"
    << "    MSB: " << sig->msb << " LSB: " << sig->lsb << "<br />\n"
    << "    Little Endian: " << (sig->is_little_endian ? "Y" : "N")
    << " Signed: " << (sig->is_signed ? "Y" : "N") << "</span>\n  ";
  return s.str();
}

}  // namespace utils
