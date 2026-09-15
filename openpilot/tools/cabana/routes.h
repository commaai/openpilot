#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace routes {

struct DeviceInfo {
  std::string dongle_id;
};

struct RouteInfo {
  std::string name;
  int64_t start_ms = 0;
  int64_t end_ms = 0;
};

using DevicesCallback = std::function<void(std::vector<DeviceInfo> devices, bool success, int error_code)>;
using RoutesCallback = std::function<void(std::vector<RouteInfo> routes, bool success, int error_code)>;

// Parse a PyDownloader JSON response into (success, error_code).
std::pair<bool, int> checkApiResponse(const std::string &result);

int64_t nowUnixMs();
// Parse ISO-8601 (with optional fractional seconds / Z) to unix ms. Returns 0 on failure.
int64_t parseIsoToUnixMs(const std::string &s);
// Local time, "%Y-%m-%d %H:%M:%S".
std::string formatUnixMs(int64_t ms);

std::vector<DeviceInfo> parseDevices(const std::string &json);
// preserved routes report ISO-8601 timestamps instead of unix millis
std::vector<RouteInfo> parseRoutes(const std::string &json, bool preserved);

// Both fetch on a detached thread and invoke the callback from that thread.
void fetchDevices(DevicesCallback callback);
// period_days of -1 requests preserved routes
void fetchRoutes(const std::string &dongle_id, int period_days, RoutesCallback callback);

}  // namespace routes
