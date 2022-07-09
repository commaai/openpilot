#pragma once

#include <QStackedLayout>
#include <QTranslator>
#include <QWidget>

#include "selfdrive/ui/qt/home.h"
#include "selfdrive/ui/qt/offroad/onboarding.h"
#include "selfdrive/ui/qt/offroad/settings.h"

class MainWindow : public QWidget {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);

public slots:
  void changeLanguage(const QString &lang);

private:
  bool eventFilter(QObject *obj, QEvent *event) override;
  void changeEvent(QEvent* event) override;
  void openSettings();
  void closeSettings();

  Device device;

  QStackedLayout *main_layout;
  HomeWindow *homeWindow;
  SettingsWindow *settingsWindow;
  OnboardingWindow *onboardingWindow;
  QTranslator translator;
};
