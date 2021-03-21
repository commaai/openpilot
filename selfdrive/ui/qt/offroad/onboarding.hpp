#pragma once

#include <QWidget>
#include <QStackedWidget>
#include <QTextEdit>
#include <QMouseEvent>
#include <QImage>

#include "selfdrive/common/params.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

protected:
  void mouseReleaseEvent(QMouseEvent* e) override;
  void paintEvent(QPaintEvent *event) override;

private:
  int currentIndex = 0;
  QImage image;
  
  // Array of bounding rects for the a given training guide step.
  static inline constexpr QRect boundingBox[] = {
      {QPoint{250, 750}, QPoint{930, 900}},
      {QPoint{280, 640}, QPoint{1280, 950}},
      {QPoint{330, 590}, QPoint{1130, 900}},
      {QPoint{910, 500}, QPoint{1580, 1000}},
      {QPoint{1180, 630}, QPoint{1300, 720}},
      {QPoint{290, 590}, QPoint{1050, 960}},
      {QPoint{1090, 550}, QPoint{1240, 660}},
      {QPoint{1050, 250}, QPoint{1580, 900}},
      {QPoint{320, 670}, QPoint{1130, 1020}},
      {QPoint{1010, 410}, QPoint{1580, 750}},
      {QPoint{1040, 230}, QPoint{1500, 1030}},
      {QPoint{300, 590}, QPoint{1190, 920}},
      {QPoint{1050, 170}, QPoint{1310, 870}},
      {QPoint{950, 460}, QPoint{1530, 770}},
      {QPoint{190, 750}, QPoint{970, 970}},
  };

signals:
  void completedTraining();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);

private:
  Params params;
  std::string current_terms_version;
  std::string current_training_version;

  QTextEdit *terms_text;
  QWidget *terms_screen();
  QWidget *training_screen();

signals:
  void onboardingDone();

public slots:
  void updateActiveScreen();
};
