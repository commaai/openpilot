#pragma once

#include <atomic>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

void precise_nano_sleep(long sleep_ns);
bool readBZ2File(const std::string_view file, std::ostream &stream);
bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort = nullptr,
                           std::function<void(void *, size_t, size_t)> func = nullptr, void *param = nullptr);
