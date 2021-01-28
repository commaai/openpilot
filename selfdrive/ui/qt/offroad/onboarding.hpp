#pragma once

#include <QWidget>
#include <QStackedWidget>
#include <QStackedLayout>
#include <QMouseEvent>

// TODO: this is defined in python too
#define LATEST_TERMS_VERSION "2"
#define LATEST_TRAINING_VERSION "0.3.0"

class TrainingGuide : public QWidget {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

protected:
  void mouseReleaseEvent(QMouseEvent* e) override;
private:
  int currentIndex = 0;
  const int numberOfFrames = 15;
  QStackedLayout* slayout;
  //Vector of bounding boxes for the next step. (minx, maxx, miny, maxy)
  QVector<QVector<int>> boundingBox {{520,1200,750,900},{550,1550,650,950},{600,1400,590,900},{1180,1850,500,1000},{1450,1570,630,720},{560,1320,590,960},{1360,1510,550,660},
                                    {1320,1850,250,900},{590,1400,670,1020},{1280,1850,410,750},{1310,1770,230,1030},{570,1460,590,920},{1320,1580,170,870},{1220,1800,460,770},{460,1240,750,970}};
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
