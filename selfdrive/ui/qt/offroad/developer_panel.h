#pragma once

#ifdef SUNNYPILOT
#include "../../sunnypilot/selfdrive/ui/qt/offroad/settings/settings.h"
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
  bool is_release;
  bool offroad;

private slots:
  void updateToggles(bool _offroad);
};
