#pragma once

#include <QPushButton>

#include "system/hardware/hw.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#define ButtonControl ButtonControlSP
#define ToggleControl ToggleControlSP
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#endif

// SSH enable toggle
class SshToggle : public ToggleControl {
  Q_OBJECT

public:
  SshToggle() : ToggleControl(tr("Enable SSH"), "", "", Hardware::get_ssh_enabled()) {
    QObject::connect(this, &SshToggle::toggleFlipped, [=](bool state) {
      Hardware::set_ssh_enabled(state);
    });
  }
};

// SSH key management widget
class SshControl : public ButtonControl {
  Q_OBJECT

public:
  SshControl();

private:
  Params params;

  void refresh();
  void getUserKeys(const QString &username);
};
