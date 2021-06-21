#pragma once

#include <QImage>
#include <QMouseEvent>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "selfdrive/common/params.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0) : QFrame(parent) {};

private:
  void showEvent(QShowEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent* e) override;

  QImage image;
  QPoint imageCorner;
  int currentIndex = 0;

  // Bounding boxes for the a given training guide step
  // (minx, maxx, miny, maxy)
  QVector<QVector<int>> boundingBox {
    {650, 1375, 700, 900},
    {1600, 1920, 0, 1080},
    {1600, 1920, 0, 1080},
    {1240, 1750, 480, 1080},
    {1570, 1780, 620, 750},
    {1600, 1920, 0, 1080},
    {1630, 1740, 620, 780},
    {1200, 1920, 0, 1080},
    {1455, 1850, 400, 660},
    {1460, 1800, 195, 520},
    {1600, 1920, 0, 1080},
    {1350, 1920, 65, 1080},
    {1600, 1920, 0, 1080},
    {1570, 1900, 130, 1000},
    {1350, 1770, 500, 700},
    {1600, 1920, 0, 1080},
    {1600, 1920, 0, 1080},
    {1000, 1800, 760, 954},
  };

signals:
  void completedTraining();
};


class TermsPage : public QFrame {
  Q_OBJECT

public:
  explicit TermsPage(QWidget *parent = 0) : QFrame(parent) {};

public slots:
  void enableAccept();

private:
  void showEvent(QShowEvent *event) override;

  QPushButton *accept_btn;

signals:
  void acceptedTerms();
  void declinedTerms();
};

class DeclinePage : public QFrame {
  Q_OBJECT

public:
  explicit DeclinePage(QWidget *parent = 0) : QFrame(parent) {};

private:
  void showEvent(QShowEvent *event) override;

signals:
  void getBack();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

private:
  void showEvent(QShowEvent *event) override;
  void updateActiveScreen();

  Params params;
  std::string current_terms_version;
  std::string current_training_version;

signals:
  void onboardingDone();
};
