#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <QComboBox>
#include <QDialog>

#include "tools/cabana/routes.h"

class RouteListWidget;

class RoutesDialog : public QDialog {
  Q_OBJECT
public:
  RoutesDialog(QWidget *parent);
  std::string route();

protected:
  void setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code);
  void setRouteList(const std::vector<routes::RouteInfo> &list, bool success);
  void fetchRoutes();

  QComboBox *device_list_;
  QComboBox *period_selector_;
  RouteListWidget *route_list_;
  std::atomic<int> fetch_id_{0};
  // expires on destruction; guards main-thread callbacks from detached worker threads
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};
