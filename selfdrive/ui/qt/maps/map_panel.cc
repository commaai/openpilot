#include "selfdrive/ui/qt/maps/map_panel.h"

#include <QHBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"

MapPanel::MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent) : QFrame(parent) {
  content_stack = new QStackedLayout(this);
  content_stack->setContentsMargins(0, 0, 0, 0);

  auto map = new MapWindow(mapboxSettings);
  QObject::connect(uiState(), &UIState::offroadTransition, map, &MapWindow::offroadTransition);
  QObject::connect(device(), &Device::interactiveTimeout, [=]() {
    content_stack->setCurrentIndex(0);
  });
  QObject::connect(map, &MapWindow::requestVisible, this, &MapPanel::requestVisible);
//  QObject::connect(map, &MapWindow::requestVisible, [=](bool visible) {
//    // when we show the map for a new route, signal HomeWindow to hide the sidebar
//    if (visible) { emit mapPanelRequested(); }
//    setVisible(visible);
//  });
  QObject::connect(map, &MapWindow::requestSettings, this, &MapPanel::requestMapSettings);
//  QObject::connect(map, &MapWindow::requestSettings, [=](bool settings) {
//    content_stack->setCurrentIndex(settings ? 1 : 0);
//  });
  content_stack->addWidget(map);

  auto settings = new MapSettings(true, parent);
  QObject::connect(settings, &MapSettings::closeSettings, [=]() {
    content_stack->setCurrentIndex(0);
  });
  content_stack->addWidget(settings);
}

void MapPanel::requestVisible(bool visible) {
  if (visible != isVisible()) {
    // signal HomeWindow to hide the sidebar and switch to map window if showing
    // TODO: perhaps showEvent is better
    if (visible) {
      emit mapPanelRequested();
      content_stack->setCurrentIndex(0);
    }
    setVisible(visible);
  }
}

void MapPanel::toggleMapSettings() {
  requestVisible(true);
//  if (!isVisible()) {
//    emit mapPanelRequested(); qDebug() << "emit mapPanelRequested()";
//    setVisible(true);
//  }
  content_stack->setCurrentIndex((content_stack->currentIndex() + 1) % 2);
}

void MapPanel::requestMapSettings(bool settings) {
  content_stack->setCurrentIndex(settings ? 1 : 0);
//  emit mapPanelRequested();
//  setVisible(true);
//  content_stack->setCurrentIndex(1);
}
