#include "window.hpp"
#include "selfdrive/hardware/hw.h"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout;
  main_layout->setMargin(0);

  homeWindow = new HomeWindow(this);
  main_layout->addWidget(homeWindow);
  QObject::connect(homeWindow, &HomeWindow::openSettings, this, &MainWindow::openSettings);
  QObject::connect(homeWindow, &HomeWindow::closeSettings, this, &MainWindow::closeSettings);
  QObject::connect(&qs, &QUIState::uiUpdate, homeWindow, &HomeWindow::update);
  QObject::connect(&qs, &QUIState::offroadTransition, homeWindow, &HomeWindow::offroadTransition);
  QObject::connect(&device, &Device::displayPowerChanged, homeWindow, &HomeWindow::displayPowerChanged);

  settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);
  QObject::connect(settingsWindow, &SettingsWindow::closeSettings, this, &MainWindow::closeSettings);
  QObject::connect(&qs, &QUIState::offroadTransition, settingsWindow, &SettingsWindow::offroadTransition);
  QObject::connect(settingsWindow, SIGNAL(reviewTrainingGuide()), this, SLOT(reviewTrainingGuide()));

  onboardingWindow = new OnboardingWindow(this);
  main_layout->addWidget(onboardingWindow);

  main_layout->setCurrentWidget(onboardingWindow);
  QObject::connect(onboardingWindow, SIGNAL(onboardingDone()), this, SLOT(closeSettings()));
  onboardingWindow->updateActiveScreen();

  device.setAwake(true, true);
  QObject::connect(&qs, &QUIState::uiUpdate, &device, &Device::update);
  QObject::connect(&qs, &QUIState::offroadTransition, this, &MainWindow::offroadTransition);
  QObject::connect(&device, &Device::displayPowerChanged, this, &MainWindow::closeSettings);

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
    device.setAwake(true, true);
  }

#ifdef QCOM
  // filter out touches while in android activity
  const QList<QEvent::Type> filter_events = {QEvent::MouseButtonPress, QEvent::MouseMove, QEvent::TouchBegin, QEvent::TouchUpdate, QEvent::TouchEnd};
  if (HardwareEon::launched_activity && filter_events.contains(event->type())) {
    HardwareEon::check_activity();
    if (HardwareEon::launched_activity) {
      return true;
    }
  }
#endif
  return false;
}
