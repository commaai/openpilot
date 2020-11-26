#include <QLabel>
#include <QString>
#include <QPushButton>
#include <QGridLayout>
#include <QVBoxLayout>

#include "onboarding.hpp"
#include "common/params.h"


QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(font-size: 100px;)");
  return l;
}

QWidget * OnboardingWindow::terms_screen() {

  QGridLayout *main_layout = new QGridLayout();
  main_layout->setMargin(30);
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
    Params().write_db_value("HasAcceptedTerms", LATEST_TERMS_VERSION);
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

QWidget * OnboardingWindow::training_screen() {

  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(30);
  main_layout->setSpacing(30);

  main_layout->addWidget(title_label("Training Guide"));

  main_layout->addWidget(new QLabel(), 1); // just a spacer

  QPushButton *btn = new QPushButton("Continue");
  main_layout->addWidget(btn);
  QObject::connect(btn, &QPushButton::released, [=]() {
    Params().write_db_value("CompletedTrainingVersion", LATEST_TRAINING_VERSION);
    updateActiveScreen();
  });

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  return widget;
}

void OnboardingWindow::updateActiveScreen() {

  Params params = Params();
  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(LATEST_TERMS_VERSION) == 0;
  bool training_done = params.get("CompletedTrainingVersion", false).compare(LATEST_TRAINING_VERSION) == 0;

  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) {
  addWidget(terms_screen());
  addWidget(training_screen());

  setStyleSheet(R"(
    * {
      background-color: black;
    }
    QPushButton {
      font-size: 50px;
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  updateActiveScreen();
}
