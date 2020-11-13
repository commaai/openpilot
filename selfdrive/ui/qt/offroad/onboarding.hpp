#pragma once

#include <QWidget>
#include <QStackedLayout>

// TODO: this is defined in python too
#define LATEST_TERMS_VERSION "2"
#define LATEST_TRAINING_VERSION "0.2.0"

class OnboardingWindow : public QWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

signals:
  void onboardingDone();

private:
  QWidget * terms_screen();
  QWidget * training_screen();
  QStackedLayout *layout;

private slots:
  void updateActiveScreen();
};
