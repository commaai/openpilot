#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/backend/store.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr MessageId kDefaultLoggyMessageId{.source = 0, .address = 0x123};

struct MessageSummary {
  MessageId id;
  size_t count = 0;
  double first_time = 0.0;
  double last_time = 0.0;
  double frequency_hz = 0.0;
  std::vector<uint8_t> latest_data;
  CoverageInfo coverage;
};

MessageId parse_message_id_state(std::string_view state_json,
                                std::optional<MessageId> selection = std::nullopt,
                                MessageId fallback = kDefaultLoggyMessageId);
MessageId initial_message_id_for_store(const Store &store, std::string_view state_json,
                                      std::optional<MessageId> selection = std::nullopt);
MessageSummary summarize_message_events(const Store &store, const MessageId &id, TimeRange range);

std::string can_message_csv(const Store &store, const MessageId &id, TimeRange range, const Msg *msg = nullptr);
std::string can_stream_csv(const Store &store, TimeRange range);
std::string can_signal_csv(const Store &store, const MessageId &id, TimeRange range, const Signal &signal);
std::string series_csv(const Store &store, std::string_view path, TimeRange range);
bool write_csv_file(const std::filesystem::path &path, std::string_view csv, std::string &error);

}  // namespace loggy
