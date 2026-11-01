#pragma once

#include <atomic>
#include <functional>
#include <string>

typedef std::function<void(uint64_t cur, uint64_t total, bool success)> DownloadProgressHandler;
void installDownloadProgressHandler(DownloadProgressHandler handler);

namespace PyTools {

// Downloads url to local cache, returns local file path. Reports progress via installDownloadProgressHandler.
std::string download(const std::string &url, bool use_cache = true, std::atomic<bool> *abort = nullptr);

// Decompresses a local log file and returns the temporary output path.
std::string decompress(const std::string &path, std::atomic<bool> *abort = nullptr);

// Returns JSON string of route files (same format as /v1/route/.../files API)
std::string getRouteFiles(const std::string &route);

}  // namespace PyTools
