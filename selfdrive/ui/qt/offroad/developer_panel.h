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
  ParamControl* joystickToggle;
  ParamControl* longManeuverToggle;
  ParamControl* hyundaiRadarTracksToggle;
  bool is_release;
  bool offroad;

private slots:
  void updateToggles(bool _offroad);
};
