#pragma once

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#else
#include "selfdrive/ui/qt/offroad/settings.h"
#endif

class DeveloperPanel : public ListWidget {
  Q_OBJECT
public:
  explicit DeveloperPanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

private:
  Params params;
  ParamControl* adbToggle;
  ParamControl* joystickToggle;
  ButtonControl* errorLogBtn;
  ParamControl* longManeuverToggle;
  ParamControl* experimentalLongitudinalToggle;
  ParamControl* hyundaiRadarTracksToggle;
  ParamControl* enableGithubRunner;
  bool is_release;
  bool offroad = false;

private slots:
  void updateToggles(bool _offroad);
};
