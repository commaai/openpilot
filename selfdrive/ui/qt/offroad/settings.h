#pragma once

#include <map>
#include <string>

#include <QButtonGroup>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"

// ********** settings window + top-level panels **********
class SettingsWindow : public QFrame {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);
  void setCurrentPanel(int index, const QString &param = "");

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void closeSettings();
  void reviewTrainingGuide();
  void showDriverView();
  void expandToggleDescription(const QString &param);

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedWidget *panel_widget;
};

class DevicePanel : public ListWidget {
  Q_OBJECT
public:
  explicit DevicePanel(SettingsWindow *parent);

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

class SoftwarePanel : public ListWidget {
  Q_OBJECT
public:
  explicit SoftwarePanel(QWidget* parent = nullptr);

private:
  void showEvent(QShowEvent *event) override;
  void updateLabels();
  void checkForUpdates();

  bool is_onroad = false;

  QLabel *onroadLbl;
  LabelControl *versionLbl;
  ButtonControl *installBtn;
  ButtonControl *downloadBtn;
  ButtonControl *targetBranchBtn;

  Params params;
  ParamWatcher *fs_watch;
};

class DeveloperPanel : public ListWidget {
  Q_OBJECT

public:
  explicit DeveloperPanel(QWidget *parent = nullptr) : ListWidget(parent) {
    setSpacing(50);

    addItem(new LabelControl(tr("Developer Options"), tr("Settings for developers")));

    // SSH Key Settings
    addItem(new SshControl());

    // Joystick Control
    ButtonControl *joystickBtn = new ButtonControl(tr("Joystick Control"), tr("RUN"));
    QObject::connect(joystickBtn, &ButtonControl::clicked, [=]() {
      std::system(("python " + std::string(getenv("OPENPILOT_PREFIX")) + "/tools/joystick/joystick_control.py").c_str());
    });
    addItem(joystickBtn);

    // Longitudinal Maneuver Report Mode
    ButtonControl *longitudinalManeuverReportModeBtn = new ButtonControl(tr("Longitudinal Maneuver Report Mode"), tr("ACTIVATE"));
    QObject::connect(longitudinalManeuverReportModeBtn, &ButtonControl::clicked, [=]() {
      std::system("echo -n 1 > /data/params/d/LongitudinalManeuverMode");
      longitudinalManeuverReportModeBtn->setText(tr("ACTIVE"));
      longitudinalManeuverReportModeBtn->setEnabled(false);
    });
    addItem(longitudinalManeuverReportModeBtn);
  }
};
