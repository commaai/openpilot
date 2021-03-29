#pragma once

#include <QPushButton>

#include "widgets/controls.hpp"
#include "selfdrive/hardware/hw.h"

// SSH enable toggle
class SshToggle : public ToggleControl {
  Q_OBJECT

public:
  SshToggle() : ToggleControl("Enable SSH", "", "", Hardware::get_ssh_enabled()) {
    QObject::connect(this, &SshToggle::toggleFlipped, [=](bool state) {
      Hardware::set_ssh_enabled(state);
    });
  }
};

// SSH key management widget
class SshControl : public AbstractControl {
  Q_OBJECT

public:
  SshControl();

private:
  QPushButton btn;
  QString username;
  QLabel username_label;

  void refresh();
  void getUserKeys(QString username);
};
