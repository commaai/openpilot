#include "selfdrive/ui/qt/maps/map_panel.h"

#include <QHBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/ui.h"

MapPanel::MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent) : QFrame(parent) {
  content_stack = new QStackedLayout(this);
  content_stack->setContentsMargins(0, 0, 0, 0);

  auto map_container = new QWidget(this);
  map_container->setObjectName("map_container");
  {
    auto map_stack = new QStackedLayout(map_container);
    map_stack->setContentsMargins(0, 0, 0, 0);
    map_stack->setStackingMode(QStackedLayout::StackAll);

    auto ui = new QWidget(this);
    {
      auto ui_layout = new QVBoxLayout(ui);
      ui_layout->setContentsMargins(0, 0, 0, 0);

      QPushButton *settings_btn = new QPushButton("Settings", this);
      settings_btn->setStyleSheet(R"(
        QPushButton {
          background-color: #292929;
          border-radius: 10px;
          color: white;
          font-size: 36px;
          padding: 20px;
        }
        QPushButton::pressed {
          background-color: #3B3B3B;
        }
      )");
      QObject::connect(settings_btn, &QPushButton::clicked, [=]() {
        content_stack->setCurrentIndex(1);
      });
      ui_layout->addWidget(settings_btn, 0, Qt::AlignBottom | Qt::AlignRight);
    }
    map_stack->addWidget(ui);

    auto map = new MapWindow(mapboxSettings);
    QObject::connect(uiState(), &UIState::offroadTransition, map, &MapWindow::offroadTransition);
    map_stack->addWidget(map);
  }
  content_stack->addWidget(map_container);

  auto settings = new MapSettings(parent);
  QObject::connect(settings, &MapSettings::closeSettings, [=]() {
    content_stack->setCurrentIndex(0);
  });
  content_stack->addWidget(settings);

  setStyleSheet(R"(
    MapSettings {
      background-color: #333333;
    }
    MapWindow {
      border: 1px solid red;
    }
    #map_container {
      border: 1px solid pink;
    }
    MapPanel {
      border: 1px solid green;
    }
    QLabel {
      color: white;
    }
    QPushButton {
      border: 1px solid purple;
    }
  )");
}

bool MapPanel::isShowingMap() const {
  return content_stack->currentIndex() == 0;
}
