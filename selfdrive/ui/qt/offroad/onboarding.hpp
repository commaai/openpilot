#pragma once

#include <QWidget>
#include <QStackedWidget>

// TODO: this is defined in python too
#define LATEST_TERMS_VERSION "2"
#define LATEST_TRAINING_VERSION "0.3.0"

class TrainingGuide : public QWidget {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

protected:
  void mouseReleaseEvent(QMouseEvent*) override;

signals:
  void completedTraining();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

private:
  QWidget * terms_screen();
  QWidget * training_screen();

signals:
  void onboardingDone();

public slots:
  void updateActiveScreen();
};
