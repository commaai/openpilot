#pragma once

#include <QPushButton>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/controls.h"

// Tether enable toggle
class TetherToggle : public ToggleControl {
  Q_OBJECT

public:
  TetherToggle() : ToggleControl("Enable tethering when driving", "", "", Hardware::get_tether_enabled()) {
    QObject::connect(this, &TetherToggle::toggleFlipped, [=](bool state) {
      Hardware::set_tether_enabled(state);
    });
  }
};
