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
  const QRect continueBtnStandard = {1610, 0, 310, 1080};
  QVector<QRect> boundingRectStandard {
    QRect(650, 710, 720, 190),
    continueBtnStandard,
    continueBtnStandard,
    QRect(1442, 565, 230, 310),
    QRect(1515, 562, 133, 60),
    continueBtnStandard,
    QRect(1580, 630, 215, 130),
    QRect(1210, 0, 485, 590),
    QRect(1460, 400, 375, 210),
    QRect(166, 842, 1019, 148),
    QRect(1460, 210, 300, 310),
    continueBtnStandard,
    QRect(1375, 80, 545, 1000),
    continueBtnStandard,
    QRect(1610, 130, 280, 800),
    QRect(1385, 485, 400, 270),
    continueBtnStandard,
    continueBtnStandard,
    QRect(1036, 769, 718, 189),
    QRect(201, 769, 718, 189),
  };

  const QRect continueBtnWide = {1850, 0, 310, 1080};
  QVector<QRect> boundingRectWide {
    QRect(654, 721, 718, 189),
    continueBtnWide,
    continueBtnWide,
    QRect(1690, 570, 165, 300),
    QRect(1690, 560, 133, 60),
    continueBtnWide,
    QRect(1820, 630, 180, 155),
    QRect(1360, 0, 460, 620),
    QRect(1570, 400, 375, 215),
    QRect(167, 842, 1018, 148),
    QRect(1610, 210, 295, 310),
    continueBtnWide,
    QRect(1555, 90, 610, 990),
    continueBtnWide,
    QRect(1600, 140, 280, 790),
    QRect(1385, 490, 750, 270),
    continueBtnWide,
    continueBtnWide,
    QRect(1138, 755, 718, 189),
    QRect(303, 755, 718, 189),
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
