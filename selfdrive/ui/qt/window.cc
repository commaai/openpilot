#include "selfdrive/ui/qt/window.h"

#include <QFontDatabase>
#include <QTranslator>

#include "system/hardware/hw.h"

QWidget* TopLevelParentWidget (QWidget* widget) {
  while (widget -> parentWidget() != Q_NULLPTR) {
    widget = widget -> parentWidget();
    qDebug() << widget->metaObject()->className();
  }
  return widget;
}

QWidget* getFirstCustomParentWidget(QWidget* widget) {
  QStringList ignoredWidgets = {"QWidget", "QStackedWidget", "ParamControl", "LabelControl", "QLabel"};
  while (widget->parentWidget() != Q_NULLPTR && ignoredWidgets.contains(widget->metaObject()->className())) {
    widget = widget->parentWidget();
    qDebug() << widget->metaObject()->className();
  }
  return widget;
}

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  qDebug() << "MainWindow";
  main_layout = new QStackedLayout(this);
  main_layout->setMargin(0);

  homeWindow = new HomeWindow(this);
  main_layout->addWidget(homeWindow);
  QObject::connect(homeWindow, &HomeWindow::openSettings, this, &MainWindow::openSettings);
  QObject::connect(homeWindow, &HomeWindow::closeSettings, this, &MainWindow::closeSettings);

  settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);
  QObject::connect(settingsWindow, &SettingsWindow::closeSettings, this, &MainWindow::closeSettings);
  QObject::connect(settingsWindow, &SettingsWindow::reviewTrainingGuide, [=]() {
    onboardingWindow->showTrainingGuide();
    main_layout->setCurrentWidget(onboardingWindow);
  });
  QObject::connect(settingsWindow, &SettingsWindow::showDriverView, [=] {
    homeWindow->showDriverView(true);
  });

  onboardingWindow = new OnboardingWindow(this);
  main_layout->addWidget(onboardingWindow);
  QObject::connect(onboardingWindow, &OnboardingWindow::onboardingDone, [=]() {
    main_layout->setCurrentWidget(homeWindow);
  });
  if (!onboardingWindow->completed()) {
    main_layout->setCurrentWidget(onboardingWindow);
  }

  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    if (!offroad) {
      closeSettings();
    }
  });
  QObject::connect(&device, &Device::interactiveTimout, [=]() {
    if (main_layout->currentWidget() == settingsWindow) {
      closeSettings();
    }
  });

  // load fonts
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-Black.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-Bold.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-ExtraBold.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-ExtraLight.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-Medium.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-Regular.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-SemiBold.ttf");
  QFontDatabase::addApplicationFont("../assets/fonts/Inter-Thin.ttf");

  // no outline to prevent the focus rectangle
  setStyleSheet(R"(
    * {
      font-family: Inter;
      outline: none;
    }
  )");
  setAttribute(Qt::WA_NoSystemBackground);

  QTimer::singleShot(100, [=]{
    QTranslator translator;
    if (!translator.load("main_fr", "translations")) {
      qDebug() << "Failed to load translation file!";
    } else {
      qDebug() << "Loaded successfully";
    }
//  qApp->installTranslator(&translator);

//    dumpObjectTree();
//    qDebug() << translator.translate("DevicfePanel", "Serial");
//    for (auto w : QObject::findChildren<QPushButton*>()) {
//      TopLevelParentWidget(w)->metaObject()->className();
//      qDebug() << w->text();
////      qDebug() << "superClass" << w->metaObject()->superClass()->className();
////      qDebug() << "window" << w->window();
////      QString context = w->parentWidget()->parentWidget()->metaObject()->className();
////      qDebug() << context;
////      qDebug() << w->parentWidget()->objectName() << w->text() << translator.translate(context.toStdString().c_str(), w->text().toStdString().c_str());
//      qDebug() << "\n";
////      w->setText(translator.translate(context.toStdString().c_str(), w->text().toStdString().c_str()));
//    }
    for (auto w : QObject::findChildren<QLabel*>()) {
      QString context = getFirstCustomParentWidget(w)->metaObject()->className();
//      qDebug() << w->text();
//      QString context = w->parentWidget()->metaObject()->className();
      QString translation = translator.translate(context.toStdString().c_str(), w->text().toStdString().c_str());
      qDebug() << "Text:" << w->text() << "Context:" << context;
      qDebug() << "Translation:" << translation;
      if (!w->text().isEmpty() && translation.isEmpty()) {
        qDebug() << "Empty translation with non-empty source text!";
      }
//      qDebug() << w->parentWidget()->objectName() << w->text() << translator.translate("MainWindow", w->text().toStdString().c_str());
//      w->setText(translator.translate("", w->text().toStdString().c_str()));
      qDebug() << "\n";
    }
  });

}

void MainWindow::openSettings() {
  main_layout->setCurrentWidget(settingsWindow);
}

void MainWindow::closeSettings() {
  main_layout->setCurrentWidget(homeWindow);

  if (uiState()->scene.started) {
    homeWindow->showSidebar(false);
  }
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
  const static QSet<QEvent::Type> evts({QEvent::MouseButtonPress, QEvent::MouseMove,
                                 QEvent::TouchBegin, QEvent::TouchUpdate, QEvent::TouchEnd});

  if (evts.contains(event->type())) {
    device.resetInteractiveTimout();
  }

//  if (event->type() == QEvent::LanguageChange) {
//    obj->retranslateUi(this);
//  }

  return false;
}

//void MainWindow::changeEvent(QEvent *event) {
//  qDebug() << "changeEvent";
//  if (event->type() == QEvent::LanguageChange) {
//    qDebug() << "LanguageChange";
////    setTrs();
////    ui->retranslateUi(this);
//  }
//
//  QWidget::changeEvent(event);
//}
