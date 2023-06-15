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
      : type(place["type"].toString()), label(place["label"].toString()),
        name(place["name"].toString()), details(place["details"].toString()) {}

  bool isFavorite() const { return type == NAV_TYPE_FAVORITE; }
  bool isRecent() const { return type == NAV_TYPE_RECENT; }

  bool operator==(NavDestination *other) const {
    return type == other->type && label == other->label &&
           name == other->name && details == other->details;
  }
  bool operator!=(NavDestination *other) const { return !(*this == other); }

  QJsonObject toJson() const {
    QJsonObject obj;
    obj["type"] = type;
    obj["label"] = label;
    obj["name"] = name;
    obj["details"] = details;
    return obj;
  }

public:
  const QString type, label, name, details;
};

class MapSettings : public QFrame {
  Q_OBJECT
public:
  class DestinationWidget;

  explicit MapSettings(QWidget *parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void parseResponse(const QString &response, bool success);
  void updateCurrentRoute();
  void clear();

private:
  void showEvent(QShowEvent *event) override;
  void refresh();

  Params params;
  QString prev_destinations, cur_destinations;
  QVBoxLayout *destinations_layout;
  QWidget *current_container;
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
  void clear(const QString &label);

private:
  struct NavIcons {
    QPixmap home, work, favorite, recent;
  };

  static NavIcons icons() {
    static NavIcons nav_icons {
      loadPixmap("../assets/navigation/icon_home.svg", {72, 72}),
      loadPixmap("../assets/navigation/icon_work.svg", {72, 72}),
      loadPixmap("../assets/navigation/icon_favorite.svg", {72, 72}),
      loadPixmap("../assets/navigation/icon_recent.svg", {72, 72}),
    };
    return nav_icons;
  }

private:
  QLabel *icon, *title, *subtitle, *action;
};
