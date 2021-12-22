#pragma once

#include <atomic>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define rInfo(msg) std::cout << msg << std::endl
#define rDebug(msg) std::cout << "\033[38;5;248m" << msg << "\033[00m" << std::endl
#define rWarning(msg) std::cout << "\033[38;5;227m" << msg << "\033[00m" << std::endl
#define rError(msg) std::cout << "\033[38;5;196m" << msg << "\033[00m" << std::endl;

std::string sha256(const std::string &str);
void precise_nano_sleep(long sleep_ns);
std::string decompressBZ2(const std::string &in);
std::string decompressBZ2(const std::byte *in, size_t in_size);
void enableHttpLogging(bool enable);
std::string getUrlWithoutQuery(const std::string &url);
size_t getRemoteFileSize(const std::string &url);
std::string httpGet(const std::string &url, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);
bool httpDownload(const std::string &url, const std::string &file, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);

template <typename T>
std::string join_vector(const std::vector<T> &v, const char *delim = ", ") {
  return v.empty() ? "" : std::accumulate(v.begin() + 1, v.end(), std::to_string(v[0]), [delim](const std::string &a, auto b) {
    return a + delim + std::to_string(b);
  });
}
