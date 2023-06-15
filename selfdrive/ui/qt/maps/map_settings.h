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

const QString NAV_TYPE_FAVORITE = "favorite";
const QString NAV_TYPE_RECENT = "recent";

const QString NAV_FAVORITE_LABEL_HOME = "home";
const QString NAV_FAVORITE_LABEL_WORK = "work";

const auto NAV_ICON_HOME =
    loadPixmap("../assets/navigation/icon_home.svg", {72, 72});
const auto NAV_ICON_WORK =
    loadPixmap("../assets/navigation/icon_work.svg", {72, 72});
const auto NAV_ICON_FAVORITE =
    loadPixmap("../assets/navigation/icon_favorite.svg", {72, 72});
const auto NAV_ICON_RECENT =
    loadPixmap("../assets/navigation/icon_recent.svg", {72, 72});

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
  QVBoxLayout *recent_layout;
  QWidget *current_container;
  DestinationWidget *current_widget;

signals:
  void closeSettings();
};

class MapSettings::DestinationWidget : public QPushButton {
  Q_OBJECT
public:
  explicit DestinationWidget(QWidget *parent = nullptr);

  void set(const QString &type, const QString &label, const QString &name,
           const QString &details, bool current = false);
  void clear(const QString &label);

private:
  QLabel *icon, *title, *subtitle, *action;
};
