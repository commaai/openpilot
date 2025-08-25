#pragma once

#include <optional>

#include "tools/cabana/dbc/dbcmanager.h"

namespace utils {
void exportToCSV(const QString &file_name, std::optional<MessageId> msg_id = std::nullopt);
void exportSignalsToCSV(const QString &file_name, const MessageId &msg_id);
}  // namespace utils
