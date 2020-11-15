#include <QLabel>
#include <QString>
#include <QPushButton>
#include <QGridLayout>

#include "onboarding.hpp"
#include "common/params.h"


QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(font-size: 100px;)");
  l->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  return l;
}

QWidget * OnboardingWindow::terms_screen() {

  QGridLayout *main_layout = new QGridLayout();
  main_layout->setMargin(30);
  main_layout->setSpacing(30);

  main_layout->addWidget(title_label("Review Terms"), 0, 0, 1, -1);

  QLabel *terms = new QLabel("See terms at https://my.comma.ai/terms");
  terms->setAlignment(Qt::AlignCenter);
  terms->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  terms->setStyleSheet(R"(
    font-size: 75px;
    border-radius: 10px;
    background-color: #292929;
  )");
  main_layout->addWidget(terms, 1, 0, 1, -1);

  main_layout->addWidget(new QPushButton("Decline"), 2, 0);

  QPushButton *accept_btn = new QPushButton("Accept");
  main_layout->addWidget(accept_btn, 2, 1);
  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    Params().write_db_value("HasAcceptedTerms", LATEST_TERMS_VERSION);
    updateActiveScreen();
  });

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

  QGridLayout *main_layout = new QGridLayout();
  main_layout->setMargin(30);
  main_layout->setSpacing(30);

  main_layout->addWidget(title_label("Training Guide"), 0, 0);

  QPushButton *btn = new QPushButton("Continue");
  main_layout->addWidget(btn, 1, 0);
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
    layout->setCurrentIndex(0);
  } else if (!training_done) {
    layout->setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QWidget(parent) {
  layout = new QStackedLayout();
  layout->addWidget(terms_screen());
  layout->addWidget(training_screen());

  setLayout(layout);
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
