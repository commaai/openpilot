#pragma once

#include <QComboBox>
#include <QDialog>
#include <QTabWidget>
#include <QVBoxLayout>

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
  bool isPreservedTabSelected();
  RouteListWidget* currentRoutesList();

  QTabWidget *routes_type_selector_;
  QComboBox *device_list_;
  QComboBox *period_selector_;
  RouteListWidget *preserved_route_list_;
  RouteListWidget *route_list_;
  OneShotHttpRequest *route_requester_;
};
