#pragma once

#include <atomic>
#include <ostream>
#include <string>
#include <vector>

bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort = nullptr);
bool readBZ2File(const std::string_view file, std::ostream &stream);
