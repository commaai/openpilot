#include <QLabel>
#include <QPainter>
#include <QVBoxLayout>
#include <QQuickWidget>
#include <QQmlContext>
#include <QDesktopWidget>

#include "common/params.h"
#include "onboarding.hpp"
#include "home.hpp"
#include "util.h"


void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  int leftOffset = (geometry().width()-1620)/2;
  int mousex = e->x()-leftOffset;
  int mousey = e->y();

  // Check for restart
  if (currentIndex == (boundingBox.size() - 1) && 1050 <= mousex && mousex <= 1500 &&
      773 <= mousey && mousey <= 954) {
    currentIndex = 0;
  } else if (boundingBox[currentIndex][0] <= mousex && mousex <= boundingBox[currentIndex][1] &&
             boundingBox[currentIndex][2] <= mousey && mousey <= boundingBox[currentIndex][3]) {
    currentIndex += 1;
  }

  if (currentIndex >= boundingBox.size()) {
    emit completedTraining();
  } else {
    image.load("../assets/training/step" + QString::number(currentIndex) + ".jpg");
    update();
  }
}

void TrainingGuide::showEvent(QShowEvent *event) {
  currentIndex = 0;
  image.load("../assets/training/step0.jpg");
}

void TrainingGuide::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  QRect devRect(0, 0, painter.device()->width(), painter.device()->height());
  QBrush bgBrush("#072339");
  painter.fillRect(devRect, bgBrush);

  QRect rect(image.rect());
  rect.moveCenter(devRect.center());
  painter.drawImage(rect.topLeft(), image);
}

TermsPage::TermsPage(QWidget *parent) : QFrame(parent){

  QVBoxLayout *main_layout = new QVBoxLayout;
  main_layout->setMargin(40);
  main_layout->setSpacing(40);

  QQuickWidget *text = new QQuickWidget(this);
  text->setResizeMode(QQuickWidget::SizeRootObjectToView);
  text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  text->setAttribute(Qt::WA_AlwaysStackOnTop);
  text->setClearColor(Qt::transparent);

  text->rootContext()->setContextProperty("font_size", 55);

  QString text_view = util::read_file("../assets/offroad/tc.html").c_str();
  text->rootContext()->setContextProperty("text_view", text_view);

  text->setSource(QUrl::fromLocalFile("qt/offroad/text_view.qml"));

  main_layout->addWidget(text);

  // TODO: add decline page
  QHBoxLayout* buttons = new QHBoxLayout;
  main_layout->addLayout(buttons);

  buttons->addWidget(new QPushButton("Decline"));
  buttons->addSpacing(50);

  accept_btn = new QPushButton("Scroll to accept");
  accept_btn->setEnabled(false);
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    emit acceptedTerms();
  });

  QObject *obj = (QObject*)text->rootObject();
  QObject::connect(obj, SIGNAL(qmlSignal()), SLOT(enableAccept()));
  setLayout(main_layout);
  setStyleSheet(R"(
    * {
      font-size: 50px;
    }
    QPushButton {
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");
}

void TermsPage::enableAccept(){
  accept_btn->setText("Accept");
  accept_btn->setEnabled(true);
  return;
}

void OnboardingWindow::updateActiveScreen() {
  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(current_terms_version) == 0;
  bool training_done = params.get("CompletedTrainingVersion", false).compare(current_training_version) == 0;

  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  params = Params();
  current_terms_version = params.get("TermsVersion", false);
  current_training_version = params.get("TrainingVersion", false);

  TermsPage* terms = new TermsPage(this);
  addWidget(terms);

  connect(terms, &TermsPage::acceptedTerms, [=](){
    Params().write_db_value("HasAcceptedTerms", current_terms_version);
    updateActiveScreen();
  });

  TrainingGuide* tr = new TrainingGuide(this);
  connect(tr, &TrainingGuide::completedTraining, [=](){
    Params().write_db_value("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });
  addWidget(tr);


  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      border-radius: 30px;
      background-color: #292929;
    }
    QPushButton:disabled {
      color: #777777;
      background-color: #222222;
    }
  )");

  updateActiveScreen();
}
