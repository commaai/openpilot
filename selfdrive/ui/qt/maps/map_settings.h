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

class NavDestination {
public:
  explicit NavDestination(const QJsonObject &place)
      : type(place["save_type"].toString()), label(place["label"].toString()), name(place["place_name"].toString()),
        details(place["place_details"].toString()) {}

  bool isFavorite() const { return type == NAV_TYPE_FAVORITE; }
  bool isRecent() const { return type == NAV_TYPE_RECENT; }

  friend bool operator==(const NavDestination& lhs, const NavDestination& rhs) {
    return lhs.type == rhs.type &&
           lhs.label == rhs.label &&
           lhs.name == rhs.name &&
           lhs.details == rhs.details;
  }

  QJsonObject toJson() const {
    QJsonObject obj;
    obj["save_type"] = type;
    obj["label"] = label;
    obj["place_name"] = name;
    obj["place_details"] = details;
    return obj;
  }

public:
  const QString type, label, name, details;
};

class MapSettings : public QFrame {
  Q_OBJECT
public:
  class DestinationWidget;

  explicit MapSettings(bool closeable = false, QWidget *parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void parseResponse(const QString &response, bool success);
  void updateCurrentRoute();

private:
  void showEvent(QShowEvent *event) override;
  void refresh();

  Params params;
  QString cur_destinations;
  QVBoxLayout *destinations_layout;
  NavDestination *current_destination;
  DestinationWidget *current_widget;

signals:
  void closeSettings();
};

class MapSettings::DestinationWidget : public ClickableWidget {
  Q_OBJECT
public:
  explicit DestinationWidget(QWidget *parent = nullptr);

  void set(NavDestination *, bool current = false);
  void unset(const QString &label, bool current = false);

private:
  struct NavIcons {
    QPixmap home, work, favorite, recent, directions;
  };

  static NavIcons icons() {
    static NavIcons nav_icons {
      loadPixmap("../assets/navigation/icon_home.svg", {96, 96}),
      loadPixmap("../assets/navigation/icon_work.svg", {96, 96}),
      loadPixmap("../assets/navigation/icon_favorite.svg", {96, 96}),
      loadPixmap("../assets/navigation/icon_recent.svg", {96, 96}),
      loadPixmap("../assets/navigation/icon_directions.svg", {96, 96}),
    };
    return nav_icons;
  }

private:
  QLabel *icon, *title, *subtitle, *action;
};
