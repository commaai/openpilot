#pragma once

#include <string_view>

inline bool is_tss3_exact_extended_response(std::string_view data) {
  constexpr std::string_view expected("\x06\x50\x03\x00\x32\x01\xf4\x00", 8);
  return data == expected;
}
