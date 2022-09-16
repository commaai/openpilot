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

enum class Severity {
  normal,
  high,
};

const std::vector<std::tuple<std::string, Severity, QString>> OFFROAD_ALERTS = {
  {"Offroad_TemperatureTooHigh", Severity::high, QObject::tr("Device temperature too high. System won't start.")},
  {"Offroad_ConnectivityNeeded", Severity::high, QObject::tr("Connect to internet to check for updates. openpilot won't automatically start until it connects to internet to check for updates.")},
  {"Offroad_UpdateFailed", Severity::high, QObject::tr("Unable to download updates\n%1")},
  {"Offroad_InvalidTime", Severity::high, QObject::tr("Invalid date and time settings, system won't start. Connect to internet to set time.")},
  {"Offroad_UnofficialHardware", Severity::high, QObject::tr("Device failed to register. It will not connect to or upload to comma.ai servers, and receives no support from comma.ai. If this is an official device, contact support@comma.ai.")},
  {"Offroad_StorageMissing", Severity::high, QObject::tr("NVMe drive not mounted.")},
  {"Offroad_BadNvme", Severity::high, QObject::tr("Unsupported NVMe drive detected. Device may draw significantly more power and overheat due to the unsupported NVMe.")},
  {"Offroad_CarUnrecognized", Severity::normal, QObject::tr("openpilot was unable to identify your car. Your car is either unsupported or its ECUs are not recognized. Please submit a pull request to add the firmware versions to the proper vehicle. Need help? Join discord.comma.ai.")},
  {"Offroad_ConnectivityNeededPrompt", Severity::normal, QObject::tr("Immediately connect to the internet to check for updates. If you do not connect to the internet, openpilot won't engage in %1")},
  {"Offroad_NoFirmware", Severity::normal, QObject::tr("openpilot was unable to identify your car. Check integrity of cables and ensure all connections are secure, particularly that the comma power is fully inserted in the OBD-II port of the vehicle. Need help? Join discord.comma.ai.")},
  {"Offroad_IsTakingSnapshot", Severity::normal, QObject::tr("Taking camera snapshots. System won't start until finished.")},
  {"Offroad_NeosUpdate", Severity::normal, QObject::tr("An update to your device's operating system is downloading in the background. You will be prompted to update when it's ready to install.")},
};

class OffroadAlert : public AbstractAlert {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  int refresh();

private:
  struct Alert {
    QString text;
    QLabel *label;
  };
  std::map<std::string, Alert> alerts;
};
