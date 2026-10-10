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

// Resolves API/internal files into a JSON manifest keyed by segment number. End is inclusive.
std::string resolveRouteFiles(const std::string &route, int begin, int end, const std::string &selector);

// Browser sign-in; abort closes the local callback server. Returns a JSON status.
std::string authenticate(const std::string &provider, std::atomic<bool> *abort);

// Returns JSON string of user's devices
std::string getDevices();

// Returns JSON string of device routes
std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms = 0, int64_t end_ms = 0, bool preserved = false);

// Migrates a log segment through the Python migration CLI. Returns the migrated bytes (empty on failure).
std::string migrateLog(const std::string &log_path);

}  // namespace PyTools
