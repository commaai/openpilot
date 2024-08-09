#pragma once

#include "selfdrive/ui/qt/offroad/settings.h"

class DevicePanel : public ListWidget {
  Q_OBJECT
public:
  explicit DevicePanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

  signals:
    void reviewTrainingGuide();
  void showDriverView();

  private slots:
    void poweroff();
  void reboot();
  void updateCalibDescription();

private:
  Params params;
  ButtonControl *pair_device;
};
