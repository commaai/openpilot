#pragma once

#include <atomic>
#include <ostream>
#include <string>

void precise_nano_sleep(long sleep_ns);
bool readBZ2File(const std::string_view file, std::ostream &stream);
void enableHttpLogging(bool enable);
bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort = nullptr);
