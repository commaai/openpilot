#pragma once

#include <QComboBox>
#include <QDialog>

#include "selfdrive/ui/qt/api.h"

class RouteListWidget;

class RoutesDialog : public QDialog {
  Q_OBJECT
public:
  RoutesDialog(QWidget *parent);
  QString route() const { return route_; }

protected:
  void accept() override;
  void parseDeviceList(const QString &json, bool success, QNetworkReply::NetworkError err);
  void parseRouteList(const QString &json, bool success, QNetworkReply::NetworkError err);
  void fetchRoutes();

  QComboBox *device_list_;
  QComboBox *period_selector_;
  RouteListWidget *route_list_;
  QString route_;
};
