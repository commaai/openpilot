#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/routes.h"

// "Remote routes" browser. on_done gets accepted=true with the selected route name ("" if none), accepted=false on cancel.
class RoutesDialog {
public:
  void open(std::function<void(bool accepted, const std::string &route)> on_done);
  void draw();

private:
  void setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code);
  void setRouteList(const std::vector<routes::RouteInfo> &list, bool success);
  void fetchRoutes();
  void finish(bool accepted);

  struct RouteItem {
    std::string label;
    std::string name;
  };

  bool open_ = false;
  bool show_ = false;
  bool devices_loaded_ = false;
  std::vector<std::string> devices_;
  int device_index_ = 0;
  int period_index_ = 0;
  std::vector<RouteItem> routes_;
  int route_index_ = -1;
  std::string empty_text_ = "No items";
  std::function<void(bool, const std::string &)> on_done_;
  std::atomic<int> fetch_id_{0};
  // expires on destruction; guards main-thread callbacks from detached worker threads
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};
