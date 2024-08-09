#pragma once

#include <map>
#include <string>

#include "selfdrive/ui/qt/offroad/settings.h"

class TogglesPanel : public ListWidget {
  Q_OBJECT
public:
  explicit TogglesPanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

  public slots:
    void expandToggleDescription(const QString &param);

  private slots:
    void updateState(const UIState &s);

private:
  Params params;
  std::map<std::string, ParamControl*> toggles;
  ButtonParamControl *long_personality_setting;

  void updateToggles();
};
