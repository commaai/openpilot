#include "tools/cabana/routes.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <thread>

#include "json11/json11.hpp"
#include "tools/replay/py_downloader.h"

namespace routes {

std::pair<bool, int> checkApiResponse(const std::string &result) {
  if (result.empty()) return {false, 500};
  std::string err;
  auto doc = json11::Json::parse(result, err);
  if (!err.empty()) return {false, 500};
  if (doc.is_object() && doc["error"].is_string()) {
    return {false, doc["error"].string_value() == "unauthorized" ? 401 : 500};
  }
  return {true, 0};
}

int64_t nowUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t parseIsoToUnixMs(const std::string &s) {
  std::string bytes = s;
  if (!bytes.empty() && (bytes.back() == 'Z' || bytes.back() == 'z')) bytes.pop_back();
  int millis = 0;
  auto dot = bytes.find('.');
  if (dot != std::string::npos) {
    std::string frac = bytes.substr(dot + 1);
    bytes = bytes.substr(0, dot);
    while (frac.size() < 3) frac.push_back('0');
    millis = std::atoi(frac.substr(0, 3).c_str());
  }
  std::tm tm{};
  const char *ret = strptime(bytes.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
  if (!ret) ret = strptime(bytes.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
  if (!ret) return 0;
  time_t secs = timegm(&tm);
  if (secs == static_cast<time_t>(-1)) return 0;
  return static_cast<int64_t>(secs) * 1000 + millis;
}

std::string formatUnixMs(int64_t ms) {
  time_t secs = static_cast<time_t>(ms / 1000);
  std::tm tm{};
  localtime_r(&secs, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

std::vector<DeviceInfo> parseDevices(const std::string &json) {
  std::vector<DeviceInfo> devices;
  std::string err;
  auto doc = json11::Json::parse(json, err);
  if (err.empty() && doc.is_array()) {
    for (const auto &device : doc.array_items()) {
      devices.push_back({device["dongle_id"].string_value()});
    }
  }
  return devices;
}

std::vector<RouteInfo> parseRoutes(const std::string &json, bool preserved) {
  std::vector<RouteInfo> items;
  std::string err;
  auto doc = json11::Json::parse(json, err);
  if (err.empty() && doc.is_array()) {
    for (const auto &route : doc.array_items()) {
      RouteInfo info;
      info.name = route["fullname"].string_value();
      if (preserved) {
        info.start_ms = parseIsoToUnixMs(route["start_time"].string_value());
        info.end_ms = parseIsoToUnixMs(route["end_time"].string_value());
      } else {
        info.start_ms = static_cast<int64_t>(route["start_time_utc_millis"].number_value());
        info.end_ms = static_cast<int64_t>(route["end_time_utc_millis"].number_value());
      }
      items.push_back(std::move(info));
    }
  }
  return items;
}

void fetchDevices(DevicesCallback callback) {
  std::thread([callback = std::move(callback)]() {
    std::string result = PyDownloader::getDevices();
    auto [success, error_code] = checkApiResponse(result);
    callback(success ? parseDevices(result) : std::vector<DeviceInfo>{}, success, error_code);
  }).detach();
}

void fetchRoutes(const std::string &dongle_id, int period_days, RoutesCallback callback) {
  const bool preserved = period_days == -1;
  int64_t start_ms = 0, end_ms = 0;
  if (!preserved) {
    end_ms = nowUnixMs();
    start_ms = end_ms - static_cast<int64_t>(period_days) * 24LL * 60LL * 60LL * 1000LL;
  }

  std::thread([dongle_id, start_ms, end_ms, preserved, callback = std::move(callback)]() {
    std::string result = PyDownloader::getDeviceRoutes(dongle_id, start_ms, end_ms, preserved);
    auto [success, error_code] = checkApiResponse(result);
    callback(success ? parseRoutes(result, preserved) : std::vector<RouteInfo>{}, success, error_code);
  }).detach();
}

}  // namespace routes
