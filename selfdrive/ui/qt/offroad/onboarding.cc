#include "selfdrive/ui/qt/offroad/onboarding.h"

#include <QDesktopWidget>
#include <QLabel>
#include <QPainter>
#include <QQmlContext>
#include <QQuickWidget>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/home.h"
#include "selfdrive/ui/qt/widgets/input.h"


void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  QPoint touch = QPoint(e->x(), e->y()) - imageCorner;
  //qDebug() << touch.x() << ", " << touch.y();

  // Check for restart
  if (currentIndex == (boundingBox.size() - 1) && 200 <= touch.x() && touch.x() <= 920 &&
      760 <= touch.y() && touch.y() <= 960) {
    currentIndex = 0;
  } else if (boundingBox[currentIndex][0] <= touch.x() && touch.x() <= boundingBox[currentIndex][1] &&
             boundingBox[currentIndex][2] <= touch.y() && touch.y() <= boundingBox[currentIndex][3]) {
    currentIndex += 1;
  }

  if (currentIndex >= boundingBox.size()) {
    emit completedTraining();
  } else {
    image.load("../assets/training/step" + QString::number(currentIndex) + ".png");
    update();
  }
}

void TrainingGuide::showEvent(QShowEvent *event) {
  currentIndex = 0;
  image.load("../assets/training/step0.png");
}

void TrainingGuide::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  QRect bg(0, 0, painter.device()->width(), painter.device()->height());
  QBrush bgBrush("#000000");
  painter.fillRect(bg, bgBrush);

  QRect rect(image.rect());
  rect.moveCenter(bg.center());
  painter.drawImage(rect.topLeft(), image);
  imageCorner = rect.topLeft();
}

void TermsPage::showEvent(QShowEvent *event) {
  // late init, building QML widget takes 200ms
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout;
  main_layout->setMargin(40);
  main_layout->setSpacing(40);

  QQuickWidget *text = new QQuickWidget(this);
  text->setResizeMode(QQuickWidget::SizeRootObjectToView);
  text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  text->setAttribute(Qt::WA_AlwaysStackOnTop);
  text->setClearColor(Qt::transparent);

  QString text_view = util::read_file("../assets/offroad/tc.html").c_str();
  text->rootContext()->setContextProperty("text_view", text_view);
  text->rootContext()->setContextProperty("font_size", 55);

  text->setSource(QUrl::fromLocalFile("qt/offroad/text_view.qml"));

  main_layout->addWidget(text);

  QObject *obj = (QObject*)text->rootObject();
  QObject::connect(obj, SIGNAL(qmlSignal()), SLOT(enableAccept()));

  QHBoxLayout* buttons = new QHBoxLayout;
  main_layout->addLayout(buttons);

  decline_btn = new QPushButton("Decline");
  buttons->addWidget(decline_btn);
  QObject::connect(decline_btn, &QPushButton::released, this, &TermsPage::declinedTerms);

  buttons->addSpacing(50);

  accept_btn = new QPushButton("Scroll to accept");
  accept_btn->setEnabled(false);
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::released, this, &TermsPage::acceptedTerms);

  setLayout(main_layout);
  setStyleSheet(R"(
    QPushButton {
      padding: 50px;
      font-size: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");
}

void TermsPage::enableAccept() {
  accept_btn->setText("Accept");
  accept_btn->setEnabled(true);
  return;
}

void DeclinePage::showEvent(QShowEvent *event) {
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout;
  main_layout->setMargin(40);
  main_layout->setSpacing(40);

  QLabel *text = new QLabel(this);
  text->setText("You must accept the Terms and Conditions in order to use openpilot!");
  text->setStyleSheet(R"(font-size: 50px;)");
  main_layout->addWidget(text, 0, Qt::AlignCenter);

  QHBoxLayout* buttons = new QHBoxLayout;
  main_layout->addLayout(buttons);

  back_btn = new QPushButton("Back");
  buttons->addWidget(back_btn);
  buttons->addSpacing(50);

  QObject::connect(back_btn, &QPushButton::released, this, &DeclinePage::getBack);

  uninstall_btn = new QPushButton("Decline, uninstall openpilot");
  uninstall_btn->setStyleSheet("background-color: #E22C2C;");
  buttons->addWidget(uninstall_btn);

  QObject::connect(uninstall_btn, &QPushButton::released, [=]() {
    if (ConfirmationDialog::confirm("Are you sure you want to uninstall?", this)) {
      Params().putBool("DoUninstall", true);
    }
  });

  setLayout(main_layout);
  setStyleSheet(R"(
    QPushButton {
      padding: 50px;
      font-size: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");
}

void OnboardingWindow::updateActiveScreen() {
  updateOnboardingStatus();

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

  connect(terms, &TermsPage::acceptedTerms, [=]() {
    Params().put("HasAcceptedTerms", current_terms_version);
    updateActiveScreen();
  });

  TrainingGuide* tr = new TrainingGuide(this);
  connect(tr, &TrainingGuide::completedTraining, [=]() {
    Params().put("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });
  addWidget(tr);

  DeclinePage* declinePage = new DeclinePage(this);
  addWidget(declinePage);

  connect(terms, &TermsPage::declinedTerms, [=]() {
    setCurrentIndex(2);
  });

  connect(declinePage, &DeclinePage::getBack, [=]() {
    updateActiveScreen();
  });

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

void OnboardingWindow::updateOnboardingStatus() {
  accepted_terms = params.get("HasAcceptedTerms", false).compare(current_terms_version) == 0;
  training_done = params.get("CompletedTrainingVersion", false).compare(current_training_version) == 0;
}

bool OnboardingWindow::isOnboardingDone() {
  updateOnboardingStatus();
  return accepted_terms && training_done;
}
