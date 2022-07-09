#pragma once

#include <QButtonGroup>
#include <QFileSystemWatcher>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>


#include "selfdrive/ui/qt/widgets/controls.h"

// ********** settings window + top-level panels **********
class SettingsWindow : public QFrame,  public UI {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

protected:
  void translateUi() override;
  void showEvent(QShowEvent *event) override;

signals:
  void closeSettings();
  void reviewTrainingGuide();
  void showDriverView();
  void changeLanguage(const QString &lang);

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedWidget *panel_widget;
  QList<QPair<QPushButton*, QWidget *>> panels;
};

class DevicePanel : public ListWidget, public UI {
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
  void translateUi() override;

  Params params;
  LabelControl *dongle_id = nullptr;
  LabelControl *serial = nullptr;
  ButtonControl *dcamBtn = nullptr;
  ButtonControl *resetCalibBtn = nullptr;
  ButtonControl *retrainingBtn = nullptr;
  ButtonControl *regulatoryBtn = nullptr;
  ButtonControl *translateBtn = nullptr;
  QPushButton *reboot_btn = nullptr;
  QPushButton *poweroff_btn = nullptr;
};

class TogglesPanel : public ListWidget, public UI {
  Q_OBJECT
public:
  explicit TogglesPanel(SettingsWindow *parent);

private:
  void translateUi() override;
};

class SoftwarePanel : public ListWidget, public UI {
  Q_OBJECT
public:
  explicit SoftwarePanel(QWidget* parent = nullptr);

private:
  void showEvent(QShowEvent *event) override;
  void updateLabels();
  void translateUi() override;

  LabelControl *gitBranchLbl;
  LabelControl *gitCommitLbl;
  LabelControl *osVersionLbl;
  LabelControl *versionLbl;
  LabelControl *lastUpdateLbl;
  ButtonControl *updateBtn;

  Params params;
  QFileSystemWatcher *fs_watch;
};
