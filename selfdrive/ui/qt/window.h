#pragma once

#include <QStackedLayout>
#include <QWidget>

#include "selfdrive/ui/qt/home.h"
#include "selfdrive/ui/qt/offroad/onboarding.h"
#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/ui.h"

class MainWindow : public QWidget {
  Q_OBJECT

protected:
  bool eventFilter(QObject *obj, QEvent *event) override;

public:
  explicit MainWindow(QWidget *parent = 0);

private:
  Device device;
  QUIState qs;

  QStackedLayout *main_layout;
  HomeWindow *homeWindow;
  SettingsWindow *settingsWindow;
  OnboardingWindow *onboardingWindow;
  bool onboardingDone = false;

public slots:
  void offroadTransition(bool offroad);
  void openSettings();
  void closeSettings();
  void reviewTrainingGuide();
};
