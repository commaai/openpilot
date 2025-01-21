#pragma once

#include "selfdrive/ui/qt/offroad/settings.h"

class DeveloperPanel : public ListWidget {
  Q_OBJECT
public:
  explicit DeveloperPanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

private:
  void initDevToggles();

  Params params;
  ParamControl* joystickToggle;
  ParamControl* longManeuverToggle;
  ParamControl* alphaLongToggle;

  bool offroad = false;

private slots:
  void updateToggles(bool _offroad);
};
