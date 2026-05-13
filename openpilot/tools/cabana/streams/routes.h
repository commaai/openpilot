#pragma once

#include <atomic>
#include <QComboBox>
#include <QDialog>

class RouteListWidget;

class RoutesDialog : public QDialog {
  Q_OBJECT
public:
  RoutesDialog(QWidget *parent);
  QString route();

protected:
  void parseDeviceList(const QString &json, bool success, int error_code);
  void parseRouteList(const QString &json, bool success, int error_code);
  void fetchRoutes();

  QComboBox *device_list_;
  QComboBox *period_selector_;
  RouteListWidget *route_list_;
  std::atomic<int> fetch_id_{0};
};
