#pragma once
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <tuple>

constexpr int INVALID_SOURCE = 0xff;

struct MessageId {
  uint8_t source = 0;
  uint32_t address = 0;
  std::string toString() const { char b[64]; snprintf(b, sizeof(b), "%u:%X", source, address); return b; }
  // strict "bus:HEX" parser
  static std::optional<MessageId> parse(const std::string &s) {
    const auto colon = s.find(':');
    if (colon == std::string::npos) return std::nullopt;
    const char *begin = s.data(), *end = begin + s.size();
    uint32_t source = 0, address = 0;
    auto bus = std::from_chars(begin, begin + colon, source);
    auto addr = std::from_chars(begin + colon + 1, end, address, 16);
    if (bus.ec != std::errc() || bus.ptr != begin + colon || source > 255 || addr.ec != std::errc() || addr.ptr != end) return std::nullopt;
    return MessageId{static_cast<uint8_t>(source), address};
  }
  static MessageId fromString(const std::string &s) { return parse(s).value_or(MessageId{}); }
  bool operator==(const MessageId &o) const { return source == o.source && address == o.address; }
  bool operator!=(const MessageId &o) const { return !(*this == o); }
  bool operator<(const MessageId &o) const { return std::tie(source, address) < std::tie(o.source, o.address); }
  bool operator>(const MessageId &o) const { return o < *this; }
};

template <> struct std::hash<MessageId> {
  size_t operator()(const MessageId &id) const noexcept { return std::hash<uint8_t>{}(id.source) ^ (std::hash<uint32_t>{}(id.address) << 1); }
};
