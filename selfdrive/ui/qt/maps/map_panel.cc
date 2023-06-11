#include "selfdrive/ui/qt/maps/map_panel.h"

#include <QStackedLayout>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/ui.h"

MapPanel::MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent) : QFrame(parent) {
  stack = new QStackedLayout(this);
  stack->setContentsMargins(0, 0, 0, 0);

  auto map = new MapWindow(mapboxSettings);
  QObject::connect(uiState(), &UIState::offroadTransition, map, &MapWindow::offroadTransition);
  stack->addWidget(map);

  stack->addWidget(new MapSettings(parent));

  setStyleSheet(R"(
    MapSettings {
      background-color: #333333;
    }
    QLabel {
      color: white;
    }
    QPushButton {
      border: none;
    }
  )");
}

bool MapPanel::isShowingMap() const {
  return stack->currentIndex() == 0;
}
