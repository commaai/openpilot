#include "selfdrive/ui/qt/window.h"

#include <QFontDatabase>
#include <QDebug>

#include "selfdrive/hardware/hw.h"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->setMargin(0);

  onboardingWindow = new OnboardingWindow(this);
  main_layout->addWidget(onboardingWindow);
  QObject::connect(onboardingWindow, &OnboardingWindow::onboardingDone, [=]() {
    main_layout->setCurrentWidget(homeWindow);
  });

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
  QObject::connect(settingsWindow, &SettingsWindow::reviewTrainingGuide, [=]() {
    main_layout->setCurrentWidget(onboardingWindow);
  });
  QObject::connect(settingsWindow, &SettingsWindow::showDriverView, [=] {
    homeWindow->showDriverView(true);
  });

  device.setAwake(true, true);
  QObject::connect(&qs, &QUIState::uiUpdate, &device, &Device::update);
  QObject::connect(&qs, &QUIState::offroadTransition, [=](bool offroad) {
    if (!offroad) {
      closeSettings();
    }
  });
  QObject::connect(&device, &Device::displayPowerChanged, [=]() {
     if(main_layout->currentWidget() != onboardingWindow) {
       closeSettings();
     }
  });

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

void MainWindow::openSettings() {
  main_layout->setCurrentWidget(settingsWindow);
}

void MainWindow::closeSettings() {
  main_layout->setCurrentWidget(homeWindow);

  if (QUIState::ui_state.scene.started) {
    emit homeWindow->showSidebar(false);
  }
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

  if (event->type() == QEvent::TouchBegin || event->type() == QEvent::TouchUpdate || event->type() == QEvent::TouchEnd || event->type() == QEvent::TouchCancel) {
    QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
    for (const QTouchEvent::TouchPoint &tp : touchEvent->touchPoints()) {
      Qt::MouseButton button = Qt::LeftButton;
      Qt::MouseButtons buttons = Qt::LeftButton;
      Qt::KeyboardModifiers modifiers;
      QPointF localPos;
      QPointF screenPos;
      localPos.setX(tp.startPos().x());
      localPos.setY(tp.startPos().y());
      screenPos.setX(tp.screenPos().x());
      screenPos.setY(tp.screenPos().y());
      if (tp.state() == Qt::TouchPointPressed) {
        QMouseEvent eventNew(QEvent::MouseButtonPress, localPos, screenPos, button, buttons, modifiers);  // TODO fix local pos if needed
        QApplication::sendEvent(obj, &eventNew);
        qDebug() << "Sent mouse press with pos:" << localPos;
      } else if (tp.state() == Qt::TouchPointReleased) {
        QMouseEvent eventNew(QEvent::MouseButtonRelease, localPos, screenPos, button, buttons, modifiers);
        QApplication::sendEvent(obj, &eventNew);
        qDebug() << "Sent mouse release with pos:" << localPos;
      } else if (tp.state() == Qt::TouchPointMoved) {
        button = Qt::NoButton;
        QMouseEvent eventNew(QEvent::MouseMove, localPos, screenPos, button, buttons, modifiers);
        QApplication::sendEvent(obj, &eventNew);
      }
    }
  }

  return false;
}
