#pragma once

#include <QButtonGroup>
#include <QFileSystemWatcher>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>
#include <QSlider>
#include <QScrollArea>


#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/widgets/slider.h"


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
};

class TogglesPanel : public ListWidget {
  Q_OBJECT
public:
  explicit TogglesPanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

public slots:
  void expandToggleDescription(const QString &param);

private:
  Params params;
  std::map<std::string, ParamControl*> toggles;

  void updateToggles();
};


struct SliderDefinition {
    QString paramName;
    QString title;
    QString unit;
    double paramMin;
    double paramMax;
    double defaultVal;
    CustomSlider::CerealSetterFunction cerealSetFunc;
};

class BehaviorPanel : public ListWidget {
  Q_OBJECT

public:
  explicit BehaviorPanel(SettingsWindow *parent = nullptr);

public slots:
    void sendAllSliderValues();

private:
  std::unique_ptr<PubMaster> pm;
  Params params;
  std::map<std::string, QWidget *> sliderItems;
  QMap<QString, CustomSlider *> sliders;
  QTimer *timer;
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
  QFileSystemWatcher *fs_watch;
};
