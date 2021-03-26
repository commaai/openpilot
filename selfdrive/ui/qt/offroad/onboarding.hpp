#pragma once

#include <QWidget>
#include <QStackedWidget>
#include <QTextEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QImage>

#include "selfdrive/common/params.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0) : QFrame(parent) {};

protected:
  void showEvent(QShowEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent* e) override;

private:
  QImage image;
  int currentIndex = 0;

  // Bounding boxes for the a given training guide step
  // (minx, maxx, miny, maxy)
  QVector<QVector<int>> boundingBox {{250, 930, 750, 900}, {280, 1280, 650, 950}, {330, 1130, 590, 900}, {910, 1580, 500, 1000}, {1180, 1300, 630, 720}, {290, 1050, 590, 960}, 
  {1090, 1240, 550, 660}, {1050, 1580, 250, 900}, {320, 1130, 670, 1020}, {1010, 1580, 410, 750}, {1040, 1500, 230, 1030}, {300, 1190, 590, 920}, {1050, 1310, 170, 870}, {950, 1530, 460, 770}, {190, 970, 750, 970}};

signals:
  void completedTraining();
};


class TermsPage : public QFrame {
  Q_OBJECT

public:
  explicit TermsPage(QWidget *parent = 0);

private:
  QPushButton *accept_btn;

public slots:
  void enableAccept();

signals:
  void acceptedTerms();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

private:
  Params params;
  std::string current_terms_version;
  std::string current_training_version;

signals:
  void onboardingDone();
  void resetTrainingGuide();

public slots:
  void updateActiveScreen();
};
