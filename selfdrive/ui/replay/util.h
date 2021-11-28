#pragma once

#include <atomic>
#include <ostream>
#include <string>

std::string sha256(const std::string &str);
void precise_nano_sleep(long sleep_ns);
std::string decompressBZ2(const std::string &in);
void enableHttpLogging(bool enable);
std::string getUrlWithoutQuery(const std::string &url);
size_t getRemoteFileSize(const std::string &url);
std::string httpGet(const std::string &url, int parts, size_t content_length, std::atomic<bool> *abort = nullptr);
bool httpDownload(const std::string &url, const std::string &file, int parts, size_t content_length, std::atomic<bool> *abort = nullptr);
