#pragma once

#include <map>

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"

class AbstractAlert : public QFrame {
  Q_OBJECT

protected:
  AbstractAlert(bool hasRebootBtn, QWidget *parent = nullptr);

  QPushButton *snooze_btn;
  QVBoxLayout *scrollable_layout;
  Params params;

signals:
  void dismiss();
};

class UpdateAlert : public AbstractAlert {
  Q_OBJECT

public:
  UpdateAlert(QWidget *parent = 0);
  bool refresh();

private:
  QLabel *releaseNotes = nullptr;
};

class OffroadAlert : public AbstractAlert {
  Q_OBJECT

public:
  enum Severity {
    normal,
    high,
  };
  const std::vector<std::tuple<std::string, Severity, QString>> offroad_alerts = {
    {"Offroad_TemperatureTooHigh", Severity::high, tr("Device temperature too high. System won't start.")},
    {"Offroad_ConnectivityNeeded", Severity::high, tr("Connect to internet to check for updates. openpilot won't automatically start until it connects to internet to check for updates.")},
    {"Offroad_UpdateFailed", Severity::high, tr("Unable to download updates\n%1")},
    {"Offroad_InvalidTime", Severity::high, tr("Invalid date and time settings, system won't start. Connect to internet to set time.")},
    {"Offroad_UnofficialHardware", Severity::high, tr("Device failed to register. It will not connect to or upload to comma.ai servers, and receives no support from comma.ai. If this is an official device, contact support@comma.ai.")},
    {"Offroad_StorageMissing", Severity::high, tr("NVMe drive not mounted.")},
    {"Offroad_BadNvme", Severity::high, tr("Unsupported NVMe drive detected. Device may draw significantly more power and overheat due to the unsupported NVMe.")},
    {"Offroad_CarUnrecognized", Severity::normal, tr("openpilot was unable to identify your car. Your car is either unsupported or its ECUs are not recognized. Please submit a pull request to add the firmware versions to the proper vehicle. Need help? Join discord.comma.ai.")},
    {"Offroad_ConnectivityNeededPrompt", Severity::normal, tr("Immediately connect to the internet to check for updates. If you do not connect to the internet, openpilot won't engage in %1")},
    {"Offroad_NoFirmware", Severity::normal, tr("openpilot was unable to identify your car. Check integrity of cables and ensure all connections are secure, particularly that the comma power is fully inserted in the OBD-II port of the vehicle. Need help? Join discord.comma.ai.")},
    {"Offroad_IsTakingSnapshot", Severity::normal, tr("Taking camera snapshots. System won't start until finished.")},
    {"Offroad_NeosUpdate", Severity::normal, tr("An update to your device's operating system is downloading in the background. You will be prompted to update when it's ready to install.")},
  };

  explicit OffroadAlert(QWidget *parent = 0);
  int refresh();

private:
  struct Alert {
    QString text;
    QLabel *label;
  };
  std::map<std::string, Alert> alerts;
};
