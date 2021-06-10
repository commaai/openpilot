#include "selfdrive/ui/qt/window.h"

#include <QFontDatabase>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->setMargin(0);

  QObject::connect(&qs, &QUIState::uiUpdate, signalMap(), &SignalMap::uiUpdate);
  QObject::connect(&qs, &QUIState::offroadTransition, signalMap(), &SignalMap::offroadTransition);
  QObject::connect(&device, &Device::displayPowerChanged, signalMap(), &SignalMap::displayPowerChanged);

  homeWindow = new HomeWindow(this);
  main_layout->addWidget(homeWindow);
  connect(signalMap(), &SignalMap::offroadTransition, homeWindow, &HomeWindow::offroadTransition);
  
  settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);
  QObject::connect(signalMap(), &SignalMap::reviewTrainingGuide, this, &MainWindow::reviewTrainingGuide);
  QObject::connect(signalMap(), &SignalMap::showDriverView, [=] {
    homeWindow->showDriverView(true);
  });

  QObject::connect(signalMap(), &SignalMap::closeSettings, this, &MainWindow::closeSettings);
  QObject::connect(signalMap(), &SignalMap::openSettings, this, &MainWindow::openSettings);

  onboardingWindow = new OnboardingWindow(this);
  onboardingDone = onboardingWindow->isOnboardingDone();
  main_layout->addWidget(onboardingWindow);

  main_layout->setCurrentWidget(onboardingWindow);
  QObject::connect(onboardingWindow, &OnboardingWindow::onboardingDone, [=]() {
    onboardingDone = true;
    closeSettings();
  });
  onboardingWindow->updateActiveScreen();

  device.setAwake(true, true);
  QObject::connect(&qs, &QUIState::uiUpdate, &device, &Device::update);
  QObject::connect(&qs, &QUIState::offroadTransition, this, &MainWindow::offroadTransition);
  QObject::connect(&device, &Device::displayPowerChanged, this, &MainWindow::closeSettings);

  // load fonts
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_regular.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_bold.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_semibold.ttf");

  // no outline to prevent the focus rectangle
  setStyleSheet(R"(
    * {
      font-family: Inter;
      outline: none;
    }
  )");
}

void MainWindow::offroadTransition(bool offroad) {
  if(!offroad) {
    closeSettings();
  }
}

void MainWindow::openSettings() {
  main_layout->setCurrentWidget(settingsWindow);
}

void MainWindow::closeSettings() {
  if(onboardingDone) {
    main_layout->setCurrentWidget(homeWindow);
  }
}

void MainWindow::reviewTrainingGuide() {
  onboardingDone = false;
  main_layout->setCurrentWidget(onboardingWindow);
  onboardingWindow->updateActiveScreen();
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
  // wake screen on tap
  if (event->type() == QEvent::MouseButtonPress) {
    device.setAwake(true, true);
  }

#ifdef QCOM
  // filter out touches while in android activity
  const static QSet<QEvent::Type> filter_events({QEvent::MouseButtonPress, QEvent::MouseMove, QEvent::TouchBegin, QEvent::TouchUpdate, QEvent::TouchEnd});
  if (HardwareEon::launched_activity && filter_events.contains(event->type())) {
    HardwareEon::check_activity();
    if (HardwareEon::launched_activity) {
      return true;
    }
  }
#endif
  return false;
}
