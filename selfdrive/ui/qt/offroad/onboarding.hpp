#pragma once

#include <QWidget>

class OnboardingWindow : public QWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

signals:
  void onboardingDone();

private slots:
  void setActiveScreen();
};
