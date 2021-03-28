#include <cstdlib>

#include "window.hpp"

#ifdef QCOM
#include "selfdrive/hardware/eon/hardware.h"
#endif

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout;
  main_layout->setMargin(0);

  homeWindow = new HomeWindow(this);
  main_layout->addWidget(homeWindow);

  settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);

  onboardingWindow = new OnboardingWindow(this);
  main_layout->addWidget(onboardingWindow);

  QObject::connect(homeWindow, SIGNAL(openSettings()), this, SLOT(openSettings()));
  QObject::connect(homeWindow, SIGNAL(closeSettings()), this, SLOT(closeSettings()));
  QObject::connect(homeWindow, SIGNAL(offroadTransition(bool)), this, SLOT(offroadTransition(bool)));
  QObject::connect(settingsWindow, SIGNAL(closeSettings()), this, SLOT(closeSettings()));
  QObject::connect(settingsWindow, SIGNAL(reviewTrainingGuide()), this, SLOT(reviewTrainingGuide()));

  // start at onboarding
  main_layout->setCurrentWidget(onboardingWindow);
  QObject::connect(onboardingWindow, SIGNAL(onboardingDone()), this, SLOT(closeSettings()));
  onboardingWindow->updateActiveScreen();

  // no outline to prevent the focus rectangle
  setLayout(main_layout);
  setStyleSheet(R"(
    * {
      font-family: Inter;
      outline: none;
    }
  )");
}

void MainWindow::offroadTransition(bool offroad){
  if(!offroad){
    closeSettings();
  }
}

void MainWindow::openSettings() {
  main_layout->setCurrentWidget(settingsWindow);
}

void MainWindow::closeSettings() {
  main_layout->setCurrentWidget(homeWindow);
}

void MainWindow::reviewTrainingGuide() {
  main_layout->setCurrentWidget(onboardingWindow);
  onboardingWindow->updateActiveScreen();
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event){
  // wake screen on tap
  if (event->type() == QEvent::MouseButtonPress) {
    homeWindow->glWindow->wake();
  }

  // filter out touches when in android activity
#ifdef QCOM
  if (HardwareEon::launched_activity) {
    switch(event->type()) {
      case QEvent::MouseButtonPress:
      case QEvent::MouseMove:
      case QEvent::TouchBegin:
      case QEvent::TouchUpdate:
      case QEvent::TouchEnd: {
        HardwareEon::check_activity();
        if (HardwareEon::launched_activity) {
          qDebug() << "rejecting touch";
          return true;
        }
      }
      default:
        break;
    }
  }
#endif

  return false;
}
