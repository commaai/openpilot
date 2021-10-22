#pragma once

#include <atomic>
#include <ostream>
#include <string>

void precise_nano_sleep(long sleep_ns);
std::string decompressBZ2(const std::string &in);
void enableHttpLogging(bool enable);
size_t getRemoteFileSize(const std::string &url);
bool httpMultiPartDownload(const std::string &url, std::ostream &stream, int parts, size_t content_length, std::atomic<bool> *abort = nullptr);
