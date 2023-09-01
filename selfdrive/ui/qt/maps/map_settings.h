#pragma once

#include <future>
#include <vector>

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

class NavManager : public QObject {
  Q_OBJECT

public:
  static NavManager *instance();
  QJsonArray currentLocations() const { return locations; }
  QJsonObject currentDestination() const { return current_dest; }
  void setCurrentDestination(const QJsonObject &loc);
  qint64 getLastActivity(const QJsonObject &loc) const;

signals:
  void updated();

private:
  NavManager(QObject *parent);
  void parseLocationsResponse(const QString &response, bool success);
  void sortLocations();

  Params params;
  QString prev_response;
  QJsonArray locations;
  QJsonObject current_dest;
  std::future<void> write_param_future;
};

class MapSettings : public QFrame {
  Q_OBJECT
public:
  explicit MapSettings(bool closeable = false, QWidget *parent = nullptr);
  void navigateTo(const QJsonObject &place);

private:
  void showEvent(QShowEvent *event) override;
  void refresh();

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
