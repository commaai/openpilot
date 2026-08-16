#pragma once
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <tuple>

constexpr int INVALID_SOURCE = 0xff;

struct MessageId {
  uint8_t source = 0;
  uint32_t address = 0;
  std::string toString() const { char b[64]; snprintf(b, sizeof(b), "%u:%X", source, address); return b; }
  static MessageId fromString(const std::string &s) {
    const auto p = s.find(':');
    if (p == std::string::npos) return {};
    return {.source = static_cast<uint8_t>(std::stoul(s.substr(0, p))), .address = static_cast<uint32_t>(std::stoul(s.substr(p + 1), nullptr, 16))};
  }
  bool operator==(const MessageId &o) const { return source == o.source && address == o.address; }
  bool operator!=(const MessageId &o) const { return !(*this == o); }
  bool operator<(const MessageId &o) const { return std::tie(source, address) < std::tie(o.source, o.address); }
  bool operator>(const MessageId &o) const { return o < *this; }
};

template <> struct std::hash<MessageId> {
  size_t operator()(const MessageId &id) const noexcept { return std::hash<uint8_t>{}(id.source) ^ (std::hash<uint32_t>{}(id.address) << 1); }
};
