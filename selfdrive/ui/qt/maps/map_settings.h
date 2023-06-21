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

class NavDestination;
class DestinationWidget;

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

  QPixmap close_icon;

signals:
  void closeSettings();
};

class NavDestination {
public:
  explicit NavDestination(const QJsonObject &place)
      : type_(place["save_type"].toString()), label_(place["label"].toString()),
        name_(place["place_name"].toString()), details_(place["place_details"].toString()),
        latitude_(place["latitude"].toDouble()), longitude_(place["longitude"].toDouble()) {
    // if details starts with `name, ` remove it
    if (details_.startsWith(name_ + ", ")) {
      details_ = details_.mid(name_.length() + 2);
    }
  }

  QString type() const { return type_; }
  QString label() const { return label_; }
  QString name() const { return name_; }
  QString details() const { return details_; }

  bool isFavorite() const { return type_ == NAV_TYPE_FAVORITE; }
  bool isRecent() const { return type_ == NAV_TYPE_RECENT; }

  bool operator==(const NavDestination &rhs) {
    return type_ == rhs.type_ && label_ == rhs.label_ && name_ == rhs.name_ &&
           details_ == rhs.details_ && latitude_ == rhs.latitude_ && longitude_ == rhs.longitude_;
  }

  QJsonObject toJson() const {
    QJsonObject obj;
    obj["save_type"] = type_;
    obj["label"] = label_;
    obj["place_name"] = name_;
    obj["place_details"] = details_;
    obj["latitude"] = latitude_;
    obj["longitude"] = longitude_;
    return obj;
  }

private:
  QString type_, label_, name_, details_;
  double latitude_, longitude_;
};

class DestinationWidget : public QPushButton {
  Q_OBJECT
public:
  explicit DestinationWidget(QWidget *parent = nullptr);

  void set(NavDestination *, bool current = false);
  void unset(const QString &label, bool current = false);

signals:
  void actionClicked();

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
};
