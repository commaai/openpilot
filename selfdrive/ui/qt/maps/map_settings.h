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
      : type_(place["save_type"].toString()), label_(place["label"].toString()),
        name_(place["place_name"].toString()), details_(place["place_details"].toString()) {
    // if details starts with `name, ` remove it
    if (details_.startsWith(name_ + ", ")) {
      details_ = details_.mid(name_.length() + 2);
    }
  }

  // getters
  QString type() const { return type_; }
  QString label() const { return label_; }
  QString name() const { return name_; }
  QString details() const { return details_; }

  bool isFavorite() const { return type_ == NAV_TYPE_FAVORITE; }
  bool isRecent() const { return type_ == NAV_TYPE_RECENT; }

  friend bool operator==(const NavDestination &lhs, const NavDestination &rhs) {
    return lhs.type_ == rhs.type_ &&
           lhs.label_ == rhs.label_ &&
           lhs.name_ == rhs.name_ &&
           lhs.details_ == rhs.details_;
  }

  QJsonObject toJson() const {
    QJsonObject obj;
    obj["save_type"] = type_;
    obj["label"] = label_;
    obj["place_name"] = name_;
    obj["place_details"] = details_;
    return obj;
  }

private:
  QString type_, label_, name_, details_;
};

class DestinationWidget : public ClickableWidget {
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

class MapSettings : public QFrame {
  Q_OBJECT
public:
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
