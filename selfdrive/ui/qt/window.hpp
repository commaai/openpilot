#pragma once

#include <QWidget>
#include <QStackedLayout>

#include "offroad/settings.hpp"
#include "offroad/onboarding.hpp"
#include "home.hpp"
#include "../ui.hpp"

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

public slots:
  void offroadTransition(bool offroad);
  void openSettings();
  void closeSettings();
  void reviewTrainingGuide();
};
