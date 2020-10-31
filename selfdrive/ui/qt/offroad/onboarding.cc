#include <string>

#include "onboarding.hpp"

#include <QString>
#include <QStackedLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>

#include "common/params.h"


QWidget * OnboardingWindow::terms_screen() {
  QVBoxLayout *main_layout = new QVBoxLayout();

  QLabel *title = new QLabel("Review Terms");
  title->setStyleSheet(R"(
    QLabel {
      font-size: 80px;
      text-align: left;
    }
  )");
  main_layout->addWidget(title, Qt::AlignTop);

  QLabel *terms = new QLabel("See terms at https://my.comma.ai/terms");
  terms->setStyleSheet(R"(
    QLabel {
      font-size: 30px;
    }
  )");
  main_layout->addWidget(terms, Qt::AlignTop);

  QHBoxLayout *btn_layout = new QHBoxLayout();
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
      padding-left: 10px;
    }
    QPushButton {
      font-size: 45px;
      padding: 50px;
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

  // Onboarding flow: terms -> training guide -> connect/prime

  QStackedLayout *main_layout = new QStackedLayout;
  main_layout->addWidget(terms_screen());
  setLayout(main_layout);

  updateActiveScreen();
}
