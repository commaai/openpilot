#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/routes.h"
#include "tools/cabana/ui/util.h"

// "Remote routes" browser. on_done gets accepted=true with the selected route name ("" if none), accepted=false on cancel.
class RoutesDialog {
public:
  ~RoutesDialog() { if (auth_abort_) *auth_abort_ = true; }
  void open(std::function<void(bool accepted, const std::string &route)> on_done);
  void draw();

private:
  void fetchDevices();
  void signIn(const std::string &provider);
  void drawLogin();
  void setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code);
  void setRouteList(const std::vector<routes::RouteInfo> &list, bool success);
  void fetchRoutes();
  void finish(bool accepted);

  struct RouteItem {
    std::string label;
    std::string name;
  };

  struct State {
    bool login = false;
    std::string provider;
    std::string auth_error;
    bool devices_loaded = false;
    std::vector<std::string> devices;
    int device_index = 0;
    int period_index = 0;
    std::vector<RouteItem> routes;
    int route_index = -1;
    std::string empty_text = "No items";
    int fetch_id = 0;  // the reply of an older request is dropped
  };

  std::shared_ptr<std::atomic<bool>> auth_abort_;
  bool open_ = false;
  PopupOwner popup_;
  State s_;
  std::function<void(bool, const std::string &)> on_done_;
  // created by open() and reset by finish(); guards main-thread callbacks from detached worker threads
  std::shared_ptr<bool> alive_;
};
