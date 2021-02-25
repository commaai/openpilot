#pragma once

#include <QWidget>
#include <QStackedWidget>
#include <QStackedLayout>
#include <QMouseEvent>
#include <QPointF>
#include <QPushButton>

#ifndef QCOM
#include <QtWebEngine>
#include <QWebEngineView>
#include <QWebEngineSettings>
#endif


class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

protected:
  void mouseReleaseEvent(QMouseEvent* e) override;

private:
  int currentIndex = 0;
  QStackedLayout* slayout;
  // Vector of bounding boxes for the a given training guide step. (minx, maxx, miny, maxy)
  QVector<QVector<int>> boundingBox {{250, 930, 750, 900}, {280, 1280, 650, 950}, {330, 1130, 590, 900}, {910, 1580, 500, 1000}, {1180, 1300, 630, 720}, {290, 1050, 590, 960}, 
  {1090, 1240, 550, 660}, {1050, 1580, 250, 900}, {320, 1130, 670, 1020}, {1010, 1580, 410, 750}, {1040, 1500, 230, 1030}, {300, 1190, 590, 920}, {1050, 1310, 170, 870}, {950, 1530, 460, 770}, {190, 970, 750, 970}};

signals:
  void completedTraining();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

private:
  std::string current_terms_version;
  std::string current_training_version;
  QWidget * terms_screen();
  QWidget * training_screen();
  QPushButton* accept_btn;

#ifndef QCOM
  QWebEngineView* view;
#endif

signals:
  void onboardingDone();

public slots:
  void updateActiveScreen();

private slots:
  void scrollPositionChanged(QPointF position);
};
