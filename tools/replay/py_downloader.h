#pragma once

#include <atomic>
#include <functional>
#include <string>

typedef std::function<void(uint64_t cur, uint64_t total, bool success)> DownloadProgressHandler;
void installDownloadProgressHandler(DownloadProgressHandler handler);

namespace PyDownloader {

// Downloads url to local cache, returns local file path. Reports progress via installDownloadProgressHandler.
std::string download(const std::string &url, bool use_cache = true, std::atomic<bool> *abort = nullptr);

// Returns JSON string of route files (same format as /v1/route/.../files API)
std::string getRouteFiles(const std::string &route);

// Returns JSON string of user's devices
std::string getDevices();

// Returns JSON string of device routes
std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms = 0, int64_t end_ms = 0, bool preserved = false);

}  // namespace PyDownloader
