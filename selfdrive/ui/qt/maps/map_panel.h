#pragma once

#include <QFrame>
#include <QMapboxGL>
#include <QStackedLayout>

#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/qt/maps/map.h"

class MapPanel : public QFrame {
  Q_OBJECT

public:
  explicit MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent = nullptr);

signals:
  void mapPanelRequested();

public slots:
  void toggleMapSettings();

private:
  QStackedLayout *content_stack;
  MapWindow *map;
//  QWidget *map;
  MapSettings *settings;
//  QWidget *settings;
};
