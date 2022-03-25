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
    continueBtnStandard,
    QRect(1476, 565, 253, 308),
    QRect(1501, 529, 184, 108),
    continueBtnStandard,
    QRect(1613, 665, 178, 153),
    QRect(1220, 0, 420, 730),
    QRect(1335, 499, 440, 147),
    QRect(112, 820, 996, 148),
    QRect(1412, 199, 316, 333),
    continueBtnStandard,
    QRect(1237, 63, 683, 1017),
    continueBtnStandard,
    QRect(1455, 110, 313, 860),
    QRect(1253, 519, 383, 228),
    continueBtnStandard,
    continueBtnStandard,
    QRect(630, 804, 626, 164),
    QRect(108, 804, 426, 164),
  };

  const QRect continueBtnWide = {1840, 0, 320, 1080};
  QVector<QRect> boundingRectWide {
    QRect(112, 804, 618, 164),
    continueBtnWide,
    continueBtnWide,
    QRect(1641, 558, 210, 313),
    QRect(1662, 528, 184, 108),
    continueBtnWide,
    QRect(1814, 621, 211, 170),
    QRect(1350, 0, 497, 755),
    QRect(1553, 516, 406, 112),
    QRect(112, 804, 1126, 164),
    QRect(1598, 199, 316, 333),
    continueBtnWide,
    QRect(1364, 90, 796, 990),
    continueBtnWide,
    QRect(1593, 114, 318, 853),
    QRect(1379, 511, 391, 243),
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
