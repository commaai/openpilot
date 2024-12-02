#pragma once

#include <QComboBox>
#include <QDialog>

#include "selfdrive/ui/qt/api.h"

class RouteListWidget;
class OneShotHttpRequest;

class RoutesDialog : public QDialog {
  Q_OBJECT
public:
  RoutesDialog(QWidget *parent);
  QString route();

protected:
  void parseDeviceList(const QString &json, bool success, QNetworkReply::NetworkError err);
  void parseRouteList(const QString &json, bool success, QNetworkReply::NetworkError err);
  void fetchRoutes();

  QComboBox *device_list_;
  QComboBox *period_selector_;
  RouteListWidget *route_list_;
  OneShotHttpRequest *route_requester_;
};
