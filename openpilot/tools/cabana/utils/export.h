#pragma once

#include <optional>
#include <string>

#include "tools/cabana/dbc/dbcmanager.h"

namespace utils {
void exportToCSV(const std::string &file_name, std::optional<MessageId> msg_id = std::nullopt);
void exportSignalsToCSV(const std::string &file_name, const MessageId &msg_id);
}  // namespace utils
