#include <QLabel>
#include <QString>
#include <QPushButton>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QDebug>
#include <QGuiApplication>
#include <QScreen>
#include <QApplication>
#include <QDesktopWidget>

#include "onboarding.hpp"
#include "common/params.h"
#include "home.hpp"


QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(font-size: 100px; font-weight: bold;)");
  return l;
}

QWidget* layout2Widget(QLayout* l){
  QWidget* q = new QWidget;
  q->setLayout(l);
  return q;
}




void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  int leftOffset = (geometry().width()-1620)/2;
  int mousex = e->x()-leftOffset;
  int mousey = e->y();

  // Check for restart
  if (currentIndex == numberOfFrames-1) {
    if (1050 <= mousex && mousex <= 1500 && 773 <= mousey && mousey <= 954){
      slayout->setCurrentIndex(0);
      currentIndex = 0;
      return;
    }
  }

  if (boundingBox[currentIndex][0] <= mousex && mousex <= boundingBox[currentIndex][1] && boundingBox[currentIndex][2] <= mousey && mousey <= boundingBox[currentIndex][3]) {
    slayout->setCurrentIndex(++currentIndex);
  }
  if (currentIndex >= numberOfFrames) {
    emit completedTraining();
    return;
  }
}

TrainingGuide::TrainingGuide(QWidget* parent) {
  QHBoxLayout* hlayout = new QHBoxLayout;

  slayout = new QStackedLayout(this);
  for (int i = 0; i <= 14; i++) {
    QWidget* w = new QWidget;
    w->setStyleSheet(".QWidget {background-image: url(../assets/training/step" + QString::number(i) + ".jpg);}");
    w->setFixedSize(1620, 1080);
    slayout->addWidget(w);
  }

  QWidget* sw = layout2Widget(slayout);
  hlayout->addWidget(sw, 1, Qt::AlignCenter);
  setLayout(hlayout);
}


QWidget* OnboardingWindow::terms_screen() {

  QGridLayout *main_layout = new QGridLayout();
  main_layout->setMargin(100);
  main_layout->setSpacing(30);

  main_layout->addWidget(title_label("Review Terms"), 0, 0, 1, -1);

  QLabel *terms = new QLabel("See terms at https://my.comma.ai/terms");
  terms->setAlignment(Qt::AlignCenter);
  terms->setStyleSheet(R"(
    font-size: 75px;
    border-radius: 10px;
    background-color: #292929;
  )");
  main_layout->addWidget(terms, 1, 0, 1, -1);
  main_layout->setRowStretch(1, 1);

  QPushButton *accept_btn = new QPushButton("Accept");
  main_layout->addWidget(accept_btn, 2, 1);
  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    Params().write_db_value("HasAcceptedTerms", current_terms_version.toStdString());
    updateActiveScreen();
  });

  main_layout->addWidget(new QPushButton("Decline"), 2, 0);

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  widget->setStyleSheet(R"(
    QPushButton {
      font-size: 50px;
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  return widget;
}

void OnboardingWindow::updateActiveScreen() {
  Params params = Params();
   
  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(current_terms_version.toStdString()) == 0;
  bool training_done = params.get("CompletedTrainingVersion", false).compare(current_training_version.toStdString()) == 0;
  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  Params params = Params();
  current_terms_version = QString::fromStdString(params.get("TermsVersion", false));
  current_training_version = QString::fromStdString(params.get("TrainingVersion", false));
  addWidget(terms_screen());
  TrainingGuide* tr = new TrainingGuide(this);
  connect(tr, &TrainingGuide::completedTraining, [=](){Params().write_db_value("CompletedTrainingVersion", current_training_version.toStdString()); updateActiveScreen();});
  addWidget(tr);

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  updateActiveScreen();
}
