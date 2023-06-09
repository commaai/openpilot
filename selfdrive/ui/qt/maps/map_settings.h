#pragma once
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QStackedWidget>

#include "common/params.h"
#include "selfdrive/ui/qt/widgets/controls.h"

const int MAP_PANEL_ICON_SIZE = 200;

const QString NAV_TYPE_FAVORITE = "favorite";
const QString NAV_TYPE_RECENT = "recent";

const QString NAV_FAVORITE_LABEL_HOME = "home";
const QString NAV_FAVORITE_LABEL_WORK = "work";

class MapPanel : public QWidget {
  Q_OBJECT
public:
  explicit MapPanel(QWidget* parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void parseResponse(const QString &response, bool success);
  void updateCurrentRoute();
  void clear();

private:
  void showEvent(QShowEvent *event) override;
  void refresh();

  Params params;
  QString prev_destinations, cur_destinations;
  QPushButton *home_button, *work_button;
  QLabel *home_address, *work_address;
  QVBoxLayout *recent_layout;
  QWidget *current_widget;
  ButtonControl *current_route;

signals:
  void closeSettings();
};
