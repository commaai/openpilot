#pragma once

#include <string_view>

inline bool is_tss3_exact_extended_response(std::string_view data) {
  constexpr std::string_view expected("\x06\x50\x03\x00\x32\x01\xf4\x00", 8);
  return data == expected;
}

inline bool is_tss3_programming_pending_response(std::string_view data) {
  constexpr std::string_view expected("\x03\x7f\x10\x78\x00\x00\x00\x00", 8);
  return data == expected;
}
