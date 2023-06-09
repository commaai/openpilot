#pragma once

#include <QFrame>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QStackedWidget>

#include "common/params.h"

class MapPanel : public QFrame {
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
  bool invalidated;
  QString cur_destinations;
  QPushButton *home_button, *work_button;
  QLabel *home_address, *work_address;
  QVBoxLayout *recent_layout;
  QWidget *current_widget;
  QLabel *current_route;

  const int ICON_SIZE = 200;

signals:
  void closeSettings();
};
