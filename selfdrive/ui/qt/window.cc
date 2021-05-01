#include "window.h"

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
  QObject::connect(&qs, &QUIState::offroadTransition, homeWindow, &HomeWindow::offroadTransitionSignal);
  QObject::connect(&device, &Device::displayPowerChanged, homeWindow, &HomeWindow::displayPowerChanged);

  settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);
  QObject::connect(settingsWindow, &SettingsWindow::closeSettings, this, &MainWindow::closeSettings);
  QObject::connect(&qs, &QUIState::offroadTransition, settingsWindow, &SettingsWindow::offroadTransition);
  QObject::connect(settingsWindow, &SettingsWindow::reviewTrainingGuide, this, &MainWindow::viewTrainingGuide);

  Params params;
  bool accepted_terms = params.get("HasAcceptedTerms") == params.get("TermsVersion");
  bool training_done = params.get("CompletedTrainingVersion") == params.get("TrainingVersion");
  if (!accepted_terms) {
    viewTerms(training_done);
  } else if (!training_done) {
    viewTrainingGuide();
  }

  device.setAwake(true, true);
  QObject::connect(&qs, &QUIState::uiUpdate, &device, &Device::update);
  QObject::connect(&qs, &QUIState::offroadTransition, this, &MainWindow::offroadTransition);
  QObject::connect(&device, &Device::displayPowerChanged, this, &MainWindow::closeSettings);

  // load fonts
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_regular.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_bold.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/opensans_semibold.ttf");

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

void MainWindow::viewTerms(bool training_done) {
  TermsPage *terms_page = new TermsPage(this);
  main_layout->addWidget(terms_page);
  main_layout->setCurrentWidget(terms_page);
  connect(terms_page, &TermsPage::acceptedTerms, [=] {
    Params params;
    params.put("HasAcceptedTerms", params.get("TermsVersion"));
    main_layout->removeWidget(terms_page);
    terms_page->deleteLater();

    if (!training_done) {
      viewTrainingGuide();
    } else {
      main_layout->setCurrentWidget(homeWindow);
    }
  });
}

void MainWindow::viewTrainingGuide() {
  TrainingGuide *tr = new TrainingGuide(this);
  main_layout->addWidget(tr);
  main_layout->setCurrentWidget(tr);
  connect(tr, &TrainingGuide::completedTraining, [=] {
    Params params;
    params.put("CompletedTrainingVersion", params.get("TrainingVersion"));
    main_layout->removeWidget(tr);
    main_layout->setCurrentWidget(homeWindow);
    tr->deleteLater();
  });
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
