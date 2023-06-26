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
  QObject::connect(map, &MapWindow::requestVisible, [=](bool visible) {
    setVisible(visible);
  });
  QObject::connect(map, &MapWindow::openSettings, [=]() {
    content_stack->setCurrentIndex(1);
  });
  content_stack->addWidget(map);

  auto settings = new MapSettings(true, parent);
  QObject::connect(settings, &MapSettings::closeSettings, [=]() {
    content_stack->setCurrentIndex(0);
  });
  content_stack->addWidget(settings);
}

bool MapPanel::isShowingMap() const {
  return content_stack->currentIndex() == 0;
}
