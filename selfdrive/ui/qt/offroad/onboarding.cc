#include "selfdrive/ui/qt/offroad/onboarding.h"

#include <QLabel>
#include <QPainter>
#include <QQmlContext>
#include <QQuickWidget>
#include <QVBoxLayout>

#include "common/util.h"
#include "common/params.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/input.h"

TrainingGuide::TrainingGuide(QWidget *parent) : QFrame(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
}

void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  if (click_timer.elapsed() < 250) {
    return;
  }
  click_timer.restart();

  if (pages[currentIndex].second.contains(e->pos())) {
    if (currentIndex == 9) {
      Params().putBool("RecordFront", dm_yes.contains(e->pos()));
    }
    currentIndex += 1;
  } else if (currentIndex == (pages.size() - 1) && restart_training.contains(e->pos())) {
    currentIndex = 0;
  }

  if (currentIndex > (pages.size() - 1)) {
    emit completedTraining();
  } else {
    image.load(img_path + "step" + QString::number(currentIndex) + ".png");
    update();
  }
}

void TrainingGuide::showEvent(QShowEvent *event) {
  img_path = width() == WIDE_WIDTH ? "../assets/training_wide/" : "../assets/training/";
  pages = width() == WIDE_WIDTH ? widePages : standardPages;

  currentIndex = 0;
  image.load(img_path + "step0.png");
  click_timer.start();
}

void TrainingGuide::step0(QPainter &p) {
  drawBody(p, tr("Welcome to openpilot alpha"),
           tr("It is important to understand the functionality and limitations of the openpilot alpha software before testing"),
           {}, 500, false);
  drawButton(p, pages[0].second, tr("Begin training"), Qt::white, Qt::black);
}

void TrainingGuide::step1(QPainter &p) {
  drawBody(p, tr("What is openpilot"),
           tr("openpilot is an advanced driver assistance program.A driver assistance system is not a self-driving car.This means openpilot is designed to work with you,not without you.your attention is required at all times to drive with openpilot"));
}

void TrainingGuide::step2(QPainter &p) {
  drawBody(p, tr("Before we continue,please confirm the following"),
           tr("✔️ I will keep my eyes on the road.\n\n✔️ I will be ready to take over at any time.\n\n✔️ I will be ready to take over at any time!"));
}

void TrainingGuide::step3(QPainter &p) {
  drawBody(p, tr("The Driving Path"),
           tr("openpilot uses the road-facing camera to plan the best path to drive.\n\nAlongside this path,lane lines are displayed in white and road edges are displayed in red."),
           tr("Tap the driving path to continue"));
}

void TrainingGuide::step4(QPainter &p) {
  drawBody(p, tr("The Lead Car Indicator"),
           tr("The lead car indicator is displayed as a triangle under the lead car.openpilot can detect 2 cars simultaneously.The second triangle will appear when a cut-in is detected.if no triangle is present,your car may be using its stock ACC system,and some openpilot features may not be available."),
           tr("Tap the lead car indicator"));
}
void TrainingGuide::step5(QPainter &p) {
  drawBody(p, tr("How to engage and control openpilot"),
           tr("openpilot is controlled using your car's cruise control inputs,which are usually located eigher on the steering wheel or on a control level near the steering column."));
}
void TrainingGuide::step6(QPainter &p) {
  drawBody(p, tr("Engage openpilot"),
           tr("When you are already to engage openpilot at a comfortable speed,engage the cruise control system and press \"SET\" to begin.A green border will appear around the screen whenever openpilot is engaged."),
           tr("Tap \"SET\" in the image to continue"));
}

void TrainingGuide::step7(QPainter &p) {
  drawBody(p, tr("Driver Monitoring"),
           tr("openpilot monitors your awareness through the device's driving-facing camera.if openpilot senses that your eyes aren't on the road,the system will go through a series of alerts and actions."),
           tr("Tap the driver to continue"));
}

void TrainingGuide::step8(QPainter &p) {
  drawBody(p, tr("Distracted Driving"),
           tr("You must pay attention at all times.If you are distracted,openpilot will show alerts of increasing severity,and you will be locked out from engaging.if the problem persists,the system will begin decelerating the vehicle"),
           tr("Tap the alert to continue"));
}

void TrainingGuide::step9(QPainter &p) {
  drawBody(p, tr("Improve Driver Monitoring"),
           tr("Help improve driver monitoring by including your driving data in the traning data set.Your preference can be changed at any time in Settings.Would you like to share your data?"));
  drawButton(p, dm_no, tr("No"), Qt::black, Qt::white);
  drawButton(p, dm_yes, tr("Yes"), Qt::white, Qt::black);
}

void TrainingGuide::step10(QPainter &p) {
  drawBody(p, tr("Adjust the max speed"),
           tr("You can adjust your maximum speed by pressing + or - on your vehicle's cruise control inputs.The max speed is shown in the upper left corner of the display."),
           tr("Tap the max speed continue"));
}

void TrainingGuide::step11(QPainter &p) {
  drawBody(p, tr("How to change lanes with openpilot"),
           tr("If you are traveling above 30 mph,openpilot can perform a lane change.Keep in mind that it is not capable of checking if a lane change is safe.This is your job.Once initiated,openpilot will change lanes regardless if another vehicle is present."));
}

void TrainingGuide::step12(QPainter &p) {
  drawBody(p, tr("Perform Lane Change"),
           tr("With openpilot engaged,turn on your signal and check your surroudings.When it's safe,gently nudge the steering wheel towards your desired lane.The sequence of turn signal and wheel nudge will prompt openpilot to change lanes."),
           tr("Tap the steering wheel continue"));
}

void TrainingGuide::step13(QPainter &p) {
  drawBody(p, tr("How to disengage openpilot"),
           tr("When you encounter a potentially unsafe situation or need to exit a highway,disengage openpilot by pressing the brake pedal."));
}

void TrainingGuide::step14(QPainter &p) {
  drawBody(p, tr("Limited Features"),
           tr("Keep in mind that openpilot will not recognize certain scenarios including stop lights,close cut-ins,or pedestrians.\n\nYou must stay alert and always be ready to retake control of the vehicle."),
           tr("Tap the light to continue"));
}

void TrainingGuide::step15(QPainter &p) {
  drawBody(p, tr("Disengage openpilot"),
           tr("While openpilot is engaged,you may keep your hands on the wheel to override steering controls.Both steering and distance to the lead car will be managed by openpilot until the brake pendal is pressed to disengage."),
           tr("Tap the brake pedal to continue"));
}

void TrainingGuide::step16(QPainter &p) {
  drawBody(p, tr("Let's review.openpilot can:"),
           tr("✔️ Determine a path to drive.\n✔️ Maintain a maximum speed.\n✔️ Maintain a safe distance from a lead car.\n✔️ Change lanes with driver assistance"));
}

void TrainingGuide::step17(QPainter &p) {
  drawBody(p, tr("openpilot cannot:"),
           tr("☓ Stay engaged while the driver is distracted.\n☓ See other cars during a lane change.\n☓ Stop for red lights,stop signs or pedestrians.\n☓ React to unsafe situations like close vehicle cut-ins or road hazards."));
}

void TrainingGuide::step18(QPainter &p) {
  drawBody(p, tr("Congratulations!You have completed openpilot training."),
           tr("This guid can be replayed at any time from the device settings.To read more about openpilot,read the wiki and join the community at discord.comma.ai."));
  drawButton(p, restart_training, tr("Restart"), Qt::black, Qt::white);
  drawButton(p, finish_training, tr("Finish Traning"), Qt::white, Qt::black);
}

void TrainingGuide::drawBody(QPainter &p, const QString &title, const QString &text, const QString &foot, int right_margin, bool has_icon) {
  // draw title
  const qreal icon_width = has_icon ? 200 : 0;
  QRect rc = rect().adjusted(LEFT_MARGIN + icon_width, TOP_MARGIN, -right_margin, -BOTTOM_MARGIN);
  configFont(p, "Inter", 80, "SemiBold");
  p.drawText(rc, Qt::AlignLeft | Qt::AlignTop, title);
  int body_y = rc.y() + getTextRect(p, Qt::AlignLeft | Qt::AlignTop, title).height();

  // draw text
  configFont(p, "Inter", 60, "Regular");
  p.drawText(rect().adjusted(LEFT_MARGIN, body_y + 50, -right_margin, -BOTTOM_MARGIN), text);

  // draw footer
  configFont(p, "Inter", 55, "SemiBold");
  p.drawText(rect().adjusted(LEFT_MARGIN, rect().height() - BOTTOM_MARGIN - 200, -right_margin, -BOTTOM_MARGIN), Qt::AlignBottom, foot);
}

void TrainingGuide::drawButton(QPainter &p, const QRect &rect, const QString &text, const QColor &bg, const QColor &f) {
  p.setBrush(bg);
  p.setPen(Qt::NoPen);
  p.drawRoundedRect(rect, 16, 16);
  p.setPen(f);
  configFont(p, "Inter", 55, "SemiBold");
  p.drawText(rect, Qt::AlignCenter, text);
}

void TrainingGuide::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  int page_idx = std::clamp(currentIndex, 0, pages.size() - 1);
  QRect bg(0, 0, painter.device()->width(), painter.device()->height());
  painter.fillRect(bg, QColor("#000000"));

  QRect rect(image.rect());
  rect.moveCenter(bg.center());
  painter.drawImage(rect.topLeft(), image);

  (this->*(pages[page_idx].first))(painter);
  // progress bar
  const int h = 20;
  const int w = (page_idx / (float)(pages.size() - 1)) * width();
  painter.fillRect(QRect(0, height() - h, w, h), QColor("#465BEA"));
}

void TermsPage::showEvent(QShowEvent *event) {
  // late init, building QML widget takes 200ms
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(45, 35, 45, 45);
  main_layout->setSpacing(0);

  QLabel *title = new QLabel(tr("Terms & Conditions"));
  title->setStyleSheet("font-size: 90px; font-weight: 600;");
  main_layout->addWidget(title);

  main_layout->addSpacing(30);

  QQuickWidget *text = new QQuickWidget(this);
  text->setResizeMode(QQuickWidget::SizeRootObjectToView);
  text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  text->setAttribute(Qt::WA_AlwaysStackOnTop);
  text->setClearColor(QColor("#1B1B1B"));

  QString text_view = util::read_file("../assets/offroad/tc.html").c_str();
  text->rootContext()->setContextProperty("text_view", text_view);

  text->setSource(QUrl::fromLocalFile("qt/offroad/text_view.qml"));

  main_layout->addWidget(text, 1);
  main_layout->addSpacing(50);

  QObject *obj = (QObject*)text->rootObject();
  QObject::connect(obj, SIGNAL(scroll()), SLOT(enableAccept()));

  QHBoxLayout* buttons = new QHBoxLayout;
  buttons->setMargin(0);
  buttons->setSpacing(45);
  main_layout->addLayout(buttons);

  QPushButton *decline_btn = new QPushButton(tr("Decline"));
  buttons->addWidget(decline_btn);
  QObject::connect(decline_btn, &QPushButton::clicked, this, &TermsPage::declinedTerms);

  accept_btn = new QPushButton(tr("Scroll to accept"));
  accept_btn->setEnabled(false);
  accept_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #465BEA;
    }
    QPushButton:disabled {
      background-color: #4F4F4F;
    }
  )");
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::clicked, this, &TermsPage::acceptedTerms);
}

void TermsPage::enableAccept() {
  accept_btn->setText(tr("Agree"));
  accept_btn->setEnabled(true);
}

void DeclinePage::showEvent(QShowEvent *event) {
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(45);
  main_layout->setSpacing(40);

  QLabel *text = new QLabel(this);
  text->setText(tr("You must accept the Terms and Conditions in order to use openpilot."));
  text->setStyleSheet(R"(font-size: 80px; font-weight: 300; margin: 200px;)");
  text->setWordWrap(true);
  main_layout->addWidget(text, 0, Qt::AlignCenter);

  QHBoxLayout* buttons = new QHBoxLayout;
  buttons->setSpacing(45);
  main_layout->addLayout(buttons);

  QPushButton *back_btn = new QPushButton(tr("Back"));
  buttons->addWidget(back_btn);

  QObject::connect(back_btn, &QPushButton::clicked, this, &DeclinePage::getBack);

  QPushButton *uninstall_btn = new QPushButton(tr("Decline, uninstall %1").arg(getBrand()));
  uninstall_btn->setStyleSheet("background-color: #B73D3D");
  buttons->addWidget(uninstall_btn);
  QObject::connect(uninstall_btn, &QPushButton::clicked, [=]() {
    Params().putBool("DoUninstall", true);
  });
}

void OnboardingWindow::updateActiveScreen() {
  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done && !params.getBool("Passive")) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  std::string current_terms_version = params.get("TermsVersion");
  std::string current_training_version = params.get("TrainingVersion");
  accepted_terms = params.get("HasAcceptedTerms") == current_terms_version;
  training_done = params.get("CompletedTrainingVersion") == current_training_version;

  TermsPage* terms = new TermsPage(this);
  addWidget(terms);
  connect(terms, &TermsPage::acceptedTerms, [=]() {
    Params().put("HasAcceptedTerms", current_terms_version);
    accepted_terms = true;
    updateActiveScreen();
  });
  connect(terms, &TermsPage::declinedTerms, [=]() { setCurrentIndex(2); });

  TrainingGuide* tr = new TrainingGuide(this);
  addWidget(tr);
  connect(tr, &TrainingGuide::completedTraining, [=]() {
    training_done = true;
    Params().put("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });

  DeclinePage* declinePage = new DeclinePage(this);
  addWidget(declinePage);
  connect(declinePage, &DeclinePage::getBack, [=]() { updateActiveScreen(); });

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      height: 160px;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #4F4F4F;
    }
  )");
  updateActiveScreen();
}
