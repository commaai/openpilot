#pragma once

#include "tools/loggy/backend/store.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace loggy {

std::string csv_escape(std::string_view text);
std::string can_message_csv(const Store &store, const MessageId &id, TimeRange range, const Msg *msg = nullptr);
std::string can_stream_csv(const Store &store, TimeRange range);
std::string can_signal_csv(const Store &store, const MessageId &id, TimeRange range, const Signal &signal);
bool write_csv_file(const std::filesystem::path &path, std::string_view csv, std::string *error = nullptr);

}  // namespace loggy
