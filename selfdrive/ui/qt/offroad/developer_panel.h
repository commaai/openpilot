#pragma once

#include "selfdrive/ui/qt/offroad/settings.h"

class DeveloperPanel : public ListWidget {
  Q_OBJECT
public:
  explicit DeveloperPanel(SettingsWindow *parent);

private:
  Params params;
  ButtonControl *sshControlButton;
  ButtonControl *joystickDebugModeButton;
  ButtonControl *LongitudinalManeuverModeButton;
};
