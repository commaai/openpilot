#include <string>

#include "onboarding.hpp"

#include <QStackedLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>

#include "common/params.h"


QWidget * OnboardingWindow::terms_screen() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(30);
  main_layout->setSpacing(30);

  QLabel *title = new QLabel("Review Terms");
  title->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  title->setStyleSheet(R"(
    QLabel {
      font-size: 80px;
      text-align: left;
      margin: 0;
      padding: 0;
    }
  )");
  main_layout->addWidget(title);

  QLabel *terms = new QLabel("See terms at https://my.comma.ai/terms");
  terms->setAlignment(Qt::AlignCenter);
  terms->setStyleSheet(R"(
    QLabel {
      font-size: 35px;
      border-radius: 10px;
      text-align: center;
      background-color: #292929;
    }
  )");
  main_layout->addWidget(terms, Qt::AlignTop);

  QHBoxLayout *btn_layout = new QHBoxLayout();
  //btn_layout->setSpacing(30);

  QPushButton *decline_btn = new QPushButton("Decline");
  btn_layout->addWidget(decline_btn);
  QPushButton *accept_btn = new QPushButton("Accept");
  btn_layout->addWidget(accept_btn);
  main_layout->addLayout(btn_layout);

  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    Params().write_db_value("HasAcceptedTerms", LATEST_TERMS_VERSION);
    updateActiveScreen();
  });

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  widget->setStyleSheet(R"(
    QLabel {
      color: white;
    }
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

  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(LATEST_TERMS_VERSION) == 0;

  if (accepted_terms) {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QWidget(parent) {

  // Onboarding flow: terms -> account pairing -> training


  QStackedLayout *main_layout = new QStackedLayout;
  main_layout->addWidget(terms_screen());
  setLayout(main_layout);
  setStyleSheet(R"(
    * {
      background-color: black;
    }
  )");

  // TODO: implement the training guide
  Params().write_db_value("CompletedTrainingVersion", LATEST_TRAINING_VERSION);

  updateActiveScreen();
}
