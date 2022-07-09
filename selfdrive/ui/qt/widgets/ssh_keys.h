#pragma once

#include <QPushButton>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/controls.h"

// SSH enable toggle
class SshToggle : public ToggleControl, public UI {
  Q_OBJECT

public:
  SshToggle() : ToggleControl("", "", "", Hardware::get_ssh_enabled()) {
    QObject::connect(this, &SshToggle::toggleFlipped, [=](bool state) {
      Hardware::set_ssh_enabled(state);
    });
    translateUi();
  }
  void translateUi() override {
    setTitle(tr("Enable SSH"));
  }
};

// SSH key management widget
class SshControl : public ButtonControl, public UI{
  Q_OBJECT

public:
  SshControl();
  void translateUi() override;

private:
  Params params;

  QLabel username_label;

  void refresh();
  void getUserKeys(const QString &username);
};
