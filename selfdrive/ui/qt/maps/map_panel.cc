#include "selfdrive/ui/qt/maps/map_panel.h"

#include <QHBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"

MapPanel::MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent) : QFrame(parent) {
  QSize icon_size(120, 120);
  directions_icon = loadPixmap("../assets/navigation/icon_directions.svg", icon_size);

  content_stack = new QStackedLayout(this);
  content_stack->setContentsMargins(0, 0, 0, 0);

  auto map_container = new QWidget(this);
  {
    auto map_stack = new QStackedLayout(map_container);
    map_stack->setContentsMargins(0, 0, 0, 0);
    map_stack->setStackingMode(QStackedLayout::StackAll);

    auto ui = new QWidget(this);
    {
      auto ui_layout = new QVBoxLayout(ui);
      ui_layout->setContentsMargins(0, 0, 32, 32);

      QPushButton *settings_btn = new QPushButton(directions_icon, "", this);
      settings_btn->setIconSize(icon_size);
      settings_btn->setStyleSheet(R"(
        QPushButton {
          background-color: #77000000;
          border-radius: 30px;
          padding: 30px;
        }
        QPushButton:pressed {
          background-color: #AA000000;
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
    QObject::connect(map, &MapWindow::requestVisible, [=](bool visible) {
      setVisible(visible);
    });
    map_stack->addWidget(map);
  }
  content_stack->addWidget(map_container);

  auto settings = new MapSettings(true, parent);
  QObject::connect(settings, &MapSettings::closeSettings, [=]() {
    content_stack->setCurrentIndex(0);
  });
  content_stack->addWidget(settings);

  setStyleSheet(R"(
    QLabel {
      color: white;
    }
    QPushButton {
      border: none;
    }
  )");
}

bool MapPanel::isShowingMap() const {
  return content_stack->currentIndex() == 0;
}
