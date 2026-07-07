#include "tools/cabana/utils/util_core.h"

#include <cstdio>
#include <string>

int num_decimals(double num) {
  // Mirrors QString::number(num) (format 'g', precision 6): count every
  // character after the first '.', including any exponent suffix.
  char buf[64];
  snprintf(buf, sizeof(buf), "%.6g", num);
  std::string str(buf);
  auto dot_pos = str.find('.');
  return dot_pos == std::string::npos ? 0 : static_cast<int>(str.size() - dot_pos - 1);
}
