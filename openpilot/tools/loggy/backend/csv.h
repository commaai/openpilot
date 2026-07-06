#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/backend/store.h"

#include <array>
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

struct BinaryBitCell {
  bool valid = false;
  uint8_t value = 0;
  uint32_t flip_count = 0;
};

struct BinaryGrid {
  MessageId id;
  double first_time = 0.0;
  double last_time = 0.0;
  size_t event_count = 0;
  std::vector<uint8_t> latest_data;
  std::vector<std::array<BinaryBitCell, 8>> rows;
  uint32_t max_flip_count = 0;
};

// History log page: newest-first CAN history for one message, filtered/compared/paginated.
struct HistoryLogParams {
  std::string filter;
  std::string compare_signal;
  std::string compare_op = ">";
  double compare_value = 0.0;
  bool compare_enabled = false;
  size_t max_rows = 1000;
  size_t page_size = 250;
  size_t page_index = 0;
};

struct HistoryLogRow {
  double mono_time = 0.0;
  uint16_t bus_time = 0;
  size_t byte_count = 0;
  std::string data_hex;
  std::string decoded;
};

struct HistoryLogPage {
  std::vector<HistoryLogRow> rows;
  size_t total_rows = 0;
  size_t page_index = 0;
  size_t page_size = 250;
  size_t page_count = 1;
  bool truncated = false;
};

MessageId parse_message_id_state(std::string_view state_json,
                                std::optional<MessageId> selection = std::nullopt,
                                MessageId fallback = kDefaultLoggyMessageId);
MessageId initial_message_id_for_store(const Store &store, std::string_view state_json,
                                      std::optional<MessageId> selection = std::nullopt);
MessageSummary summarize_message_events(const Store &store, const MessageId &id, TimeRange range);
std::optional<BinaryGrid> build_binary_grid(const Store &store, const MessageId &id, TimeRange range);
bool history_valid_compare_op(std::string_view op);
HistoryLogPage prepare_history_log_page(const Store &store, const MessageId &id, TimeRange range,
                                       const HistoryLogParams &params, const Msg *msg = nullptr);

std::string can_message_csv(const Store &store, const MessageId &id, TimeRange range, const Msg *msg = nullptr);
std::string can_stream_csv(const Store &store, TimeRange range);
std::string can_signal_csv(const Store &store, const MessageId &id, TimeRange range, const Signal &signal);
std::string series_csv(const Store &store, std::string_view path, TimeRange range);
bool write_csv_file(const std::filesystem::path &path, std::string_view csv, std::string &error);

}  // namespace loggy
