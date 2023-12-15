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
  QObject::connect(device(), &Device::interactiveTimeout, this, [=]() {
    content_stack->setCurrentIndex(0);
  });
  QObject::connect(map, &MapWindow::requestVisible, this, [=](bool visible) {
    // when we show the map for a new route, signal HomeWindow to hide the sidebar
    if (visible) { emit mapPanelRequested(); }
    setVisible(visible);
  });
  QObject::connect(map, &MapWindow::requestSettings, this, [=](bool settings) {
    content_stack->setCurrentIndex(settings ? 1 : 0);
  });
  content_stack->addWidget(map);

  auto settings = new MapSettings(true, parent);
  QObject::connect(settings, &MapSettings::closeSettings, this, [=]() {
    content_stack->setCurrentIndex(0);
  });
  content_stack->addWidget(settings);
}

void MapPanel::toggleMapSettings() {
  // show settings if not visible, then toggle between map and settings
  int new_index = isVisible() ? (1 - content_stack->currentIndex()) : 1;
  content_stack->setCurrentIndex(new_index);
  emit mapPanelRequested();
  show();
}
