#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QMouseEvent>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/qt_window.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

private:
  void showEvent(QShowEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent* e) override;

  QImage image;
  int currentIndex = 0;

  // Bounding boxes for each training guide step
  const QRect continueBtnStandard = {1620, 0, 300, 1080};
  QVector<QRect> boundingRectStandard {
    QRect(112, 804, 619, 166),
    continueBtnStandard,
    continueBtnStandard,  // done
    QRect(1488, 564, 229, 309),  // path
    QRect(1515, 547, 158, 73),  //triangle
    continueBtnStandard,
    QRect(1611, 664, 174, 153),  // set
    QRect(1220, 0, 700, 680),  // dm face
    QRect(1353, 516, 406, 112),  // alert
    QRect(112, 820, 996, 148),  // dm yes/no
    QRect(1412, 199, 316, 333),  // set speed
    continueBtnStandard,
    QRect(1237, 66, 683, 1014), // steering wheel
    continueBtnStandard,
    QRect(1438, 112, 351, 854),  // traffic light
    QRect(1242, 504, 405, 258),  // brake pedal
    continueBtnStandard,
    continueBtnStandard,
    QRect(630, 804, 626, 164),
    QRect(108, 804, 426, 164),
  };

  const QRect continueBtnWide = {1840, 0, 320, 1080};
  QVector<QRect> boundingRectWide {
    QRect(112, 804, 619, 165),
    continueBtnWide,
    continueBtnWide,
    QRect(1641, 558, 210, 313),  // path
    QRect(1662, 528, 184, 108), // triangle
    continueBtnWide,
    QRect(1823, 631, 177, 154), // set
    QRect(1350, 0, 810, 650),
    QRect(1553, 516, 406, 112), // alert
    QRect(112, 804, 1126, 164), // dm yesno
    QRect(1598, 199, 316, 333),  // set speed
    continueBtnWide,
    QRect(1364, 90, 796, 990),  // steering wheel
    continueBtnWide,
    QRect(1593, 114, 318, 853),  // traffic light
    QRect(1379, 511, 391, 243), // brake pedal
    continueBtnWide,
    continueBtnWide,
    QRect(630, 804, 626, 164),
    QRect(108, 804, 426, 164),
  };

  QString img_path;
  QVector<QRect> boundingRect;
  QElapsedTimer click_timer;

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
  inline void showTrainingGuide() { setCurrentIndex(1); }
  inline bool completed() const { return accepted_terms && training_done; }

private:
  void updateActiveScreen();

  Params params;
  bool accepted_terms = false, training_done = false;

signals:
  void onboardingDone();
};
