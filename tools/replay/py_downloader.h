#pragma once

#include <atomic>
#include <functional>
#include <string>

typedef std::function<void(uint64_t cur, uint64_t total, bool success)> DownloadProgressHandler;
void installDownloadProgressHandler(DownloadProgressHandler handler);

// Download handler writes result path into *out. Empty string = failure.
typedef void (*DownloadHandlerFn)(const std::string &url, bool use_cache, std::string *out);
void installDownloadHandler(DownloadHandlerFn handler);

namespace PyDownloader {

// Downloads url to local cache, returns local file path. Reports progress via installDownloadProgressHandler.
std::string download(const std::string &url, bool use_cache = true, std::atomic<bool> *abort = nullptr);

}  // namespace PyDownloader
