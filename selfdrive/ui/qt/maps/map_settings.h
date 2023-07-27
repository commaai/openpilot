#pragma once

#include <QFrame>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/controls.h"

const QString NAV_TYPE_FAVORITE = "favorite";
const QString NAV_TYPE_RECENT = "recent";

const QString NAV_FAVORITE_LABEL_HOME = "home";
const QString NAV_FAVORITE_LABEL_WORK = "work";

class DestinationWidget;

class NavigationRequest : public QObject {
  Q_OBJECT

public:
  static NavigationRequest *instance();
  QJsonArray currentLocations() const { return locations; };

signals:
  void locationsUpdated(const QJsonArray &locations);
  void nextDestinationUpdated();

private:
  NavigationRequest(QObject *parent);
  void parseLocationsResponse(const QString &response, bool success);

  Params params;
  QString prev_response;
  QJsonArray locations;
};

class MapSettings : public QFrame {
  Q_OBJECT
public:
  explicit MapSettings(bool closeable = false, QWidget *parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void updateLocations(const QJsonArray &locations);
  void updateCurrentRoute();

private:
  void mousePressEvent(QMouseEvent *ev) override;
  void showEvent(QShowEvent *event) override;
  void refresh();

  Params params;
  QJsonArray current_locations;
  QJsonObject current_destination;
  QVBoxLayout *destinations_layout;
  DestinationWidget *current_widget;
  DestinationWidget *home_widget;
  DestinationWidget *work_widget;
  std::vector<DestinationWidget *> widgets;

signals:
  void closeSettings();
};

class DestinationWidget : public QPushButton {
  Q_OBJECT
public:
  explicit DestinationWidget(QWidget *parent = nullptr);
  void set(const QJsonObject &location, bool current = false);
  void unset(const QString &label, bool current = false);

signals:
  void actionClicked();
  void navigateTo(const QJsonObject &destination);

private:
  struct NavIcons {
    QPixmap home, work, favorite, recent, directions;
  };

  static NavIcons icons() {
    static NavIcons nav_icons {
      loadPixmap("../assets/navigation/icon_home.svg", {48, 48}),
      loadPixmap("../assets/navigation/icon_work.svg", {48, 48}),
      loadPixmap("../assets/navigation/icon_favorite.svg", {48, 48}),
      loadPixmap("../assets/navigation/icon_recent.svg", {48, 48}),
      loadPixmap("../assets/navigation/icon_directions.svg", {48, 48}),
    };
    return nav_icons;
  }

private:
  QLabel *icon, *title, *subtitle;
  QPushButton *action;
  QJsonObject dest;
};
