#include "selfdrive/ui/qt/setup/setup.h"

#include <cstdio>
#include <cstdlib>

#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>

#include <curl/curl.h>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/offroad/networking.h"
#include "selfdrive/ui/qt/widgets/input.h"

const char* USER_AGENT = "AGNOSSetup-0.1";
const QString DASHCAM_URL = "https://dashcam.comma.ai";

void Setup::download(QString url) {
  CURL *curl = curl_easy_init();
  if (!curl) {
    emit finished(false);
    return;
  }

  char tmpfile[] = "/tmp/installer_XXXXXX";
  FILE *fp = fdopen(mkstemp(tmpfile), "w");

  curl_easy_setopt(curl, CURLOPT_URL, url.toStdString().c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);

  int ret = curl_easy_perform(curl);
  if (ret != CURLE_OK) {
    emit finished(false);
    return;
  }
  curl_easy_cleanup(curl);
  fclose(fp);

  rename(tmpfile, "/tmp/installer");
  emit finished(true);
}

QWidget * Setup::low_voltage() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 0, 55, 55);
  main_layout->setSpacing(0);

  // inner text layout: warning icon, title, and body
  QVBoxLayout *inner_layout = new QVBoxLayout();
  inner_layout->setContentsMargins(110, 144, 365, 0);
  main_layout->addLayout(inner_layout);

  QLabel *triangle = new QLabel();
  triangle->setPixmap(QPixmap(":/offroad/icon_warning.png"));
  inner_layout->addWidget(triangle, 0, Qt::AlignTop | Qt::AlignLeft);
  inner_layout->addSpacing(80);

  QLabel *title = new QLabel("WARNING: Low Voltage");
  title->setStyleSheet("font-size: 90px; font-weight: 500; color: #FF594F;");
  inner_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  inner_layout->addSpacing(25);

  QLabel *body = new QLabel("Power your device in a car with a harness or proceed at your own risk.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  body->setStyleSheet("font-size: 80px; font-weight: 300;");
  inner_layout->addWidget(body);

  inner_layout->addStretch();

  // power off + continue buttons
  QHBoxLayout *blayout = new QHBoxLayout();
  blayout->setSpacing(50);
  main_layout->addLayout(blayout, 0);

  QPushButton *poweroff = new QPushButton("Power off");
  poweroff->setObjectName("navBtn");
  blayout->addWidget(poweroff);
  QObject::connect(poweroff, &QPushButton::clicked, this, [=]() {
    Hardware::poweroff();
  });

  QPushButton *cont = new QPushButton("Continue");
  cont->setObjectName("navBtn");
  blayout->addWidget(cont);
  QObject::connect(cont, &QPushButton::clicked, this, &Setup::nextPage);

  return widget;
}

QWidget * Setup::getting_started() {
  QWidget *widget = new QWidget();

  QHBoxLayout *main_layout = new QHBoxLayout(widget);
  main_layout->setMargin(0);

  QVBoxLayout *vlayout = new QVBoxLayout();
  vlayout->setContentsMargins(165, 280, 100, 0);
  main_layout->addLayout(vlayout);

  QLabel *title = new QLabel("Getting Started");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  vlayout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  vlayout->addSpacing(90);
  QLabel *desc = new QLabel("Before we get on the road, let’s finish installation and cover some details.");
  desc->setWordWrap(true);
  desc->setStyleSheet("font-size: 80px; font-weight: 300;");
  vlayout->addWidget(desc, 0, Qt::AlignTop | Qt::AlignLeft);

  vlayout->addStretch();

  QPushButton *btn = new QPushButton();
  btn->setIcon(QIcon(":/img_continue_triangle.svg"));
  btn->setIconSize(QSize(54, 106));
  btn->setFixedSize(310, 1080);
  btn->setProperty("primary", true);
  btn->setStyleSheet("border: none;");
  main_layout->addWidget(btn, 0, Qt::AlignRight);
  QObject::connect(btn, &QPushButton::clicked, this, &Setup::nextPage);

  return widget;
}

QWidget * Setup::network_setup() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 50, 55, 50);

  // title
  QLabel *title = new QLabel("Connect to Wi-Fi");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

  main_layout->addSpacing(25);

  // wifi widget
  Networking *networking = new Networking(this, false);
  networking->setStyleSheet("Networking {background-color: #292929; border-radius: 13px;}");
  main_layout->addWidget(networking, 1);

  main_layout->addSpacing(35);

  // back + continue buttons
  QHBoxLayout *blayout = new QHBoxLayout;
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  QPushButton *back = new QPushButton("Back");
  back->setObjectName("navBtn");
  QObject::connect(back, &QPushButton::clicked, this, &Setup::prevPage);
  blayout->addWidget(back);

  QPushButton *cont = new QPushButton();
  cont->setObjectName("navBtn");
  cont->setProperty("primary", true);
  QObject::connect(cont, &QPushButton::clicked, this, &Setup::nextPage);
  blayout->addWidget(cont);

  // setup timer for testing internet connection
  HttpRequest *request = new HttpRequest(this, false, 2500);
  QObject::connect(request, &HttpRequest::requestDone, [=](bool success) {
    cont->setEnabled(success);
    if (success) {
      const bool cell = networking->wifi->currentNetworkType() == NetworkType::CELL;
      cont->setText(cell ? "Continue without Wi-Fi" : "Continue");
    } else {
      cont->setText("Waiting for internet");
    }
    repaint();
  });
  request->sendRequest(DASHCAM_URL);
  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    if (!request->active() && cont->isVisible()) {
      request->sendRequest(DASHCAM_URL);
    }
  });
  timer->start(1000);

  return widget;
}

QWidget * radio_button(QString title, QButtonGroup *group) {
  QPushButton *btn = new QPushButton(title);
  btn->setCheckable(true);
  group->addButton(btn);
  btn->setStyleSheet(R"(
    QPushButton {
      height: 230;
      padding-left: 100px;
      padding-right: 100px;
      text-align: left;
      font-size: 80px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #4F4F4F;
    }
    QPushButton:checked {
      background-color: #465BEA;
    }
  )");

  // checkmark icon
  QPixmap pix(":/img_circled_check.svg");
  btn->setIcon(pix);
  btn->setIconSize(QSize(0, 0));
  btn->setLayoutDirection(Qt::RightToLeft);
  QObject::connect(btn, &QPushButton::toggled, [=](bool checked) {
    btn->setIconSize(checked ? QSize(104, 104) : QSize(0, 0));
  });
  return btn;
}

QWidget * Setup::software_selection() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 50, 55, 50);
  main_layout->setSpacing(0);

  // title
  QLabel *title = new QLabel("Choose Software to Install");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

  main_layout->addSpacing(50);

  // dashcam + custom radio buttons
  QButtonGroup *group = new QButtonGroup(widget);
  group->setExclusive(true);

  QWidget *dashcam = radio_button("Dashcam", group);
  main_layout->addWidget(dashcam);

  main_layout->addSpacing(30);

  QWidget *custom = radio_button("Custom Software", group);
  main_layout->addWidget(custom);

  main_layout->addStretch();

  // back + continue buttons
  QHBoxLayout *blayout = new QHBoxLayout;
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  QPushButton *back = new QPushButton("Back");
  back->setObjectName("navBtn");
  QObject::connect(back, &QPushButton::clicked, this, &Setup::prevPage);
  blayout->addWidget(back);

  QPushButton *cont = new QPushButton("Continue");
  cont->setObjectName("navBtn");
  cont->setEnabled(false);
  cont->setProperty("primary", true);
  blayout->addWidget(cont);

  QObject::connect(cont, &QPushButton::clicked, [=]() {
    auto w = currentWidget();
    QTimer::singleShot(0, [=]() {
      setCurrentWidget(downloading_widget);
    });
    QString url = DASHCAM_URL;
    if (group->checkedButton() != dashcam) {
      url = InputDialog::getText("Enter URL", this, "for Custom Software");
    }
    if (!url.isEmpty()) {
      QTimer::singleShot(1000, this, [=]() {
        download(url);
      });
    } else {
      setCurrentWidget(w);
    }
  });

  connect(group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), [=](QAbstractButton *btn) {
    btn->setChecked(true);
    cont->setEnabled(true);
  });

  return widget;
}

QWidget * Setup::downloading() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  QLabel *txt = new QLabel("Downloading...");
  txt->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(txt, 0, Qt::AlignCenter);
  return widget;
}

QWidget * Setup::download_failed() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 225, 55, 55);
  main_layout->setSpacing(0);

  QLabel *title = new QLabel("Download Failed");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  main_layout->addSpacing(67);

  QLabel *body = new QLabel("Ensure the entered URL is valid, and the device’s internet connection is good.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  body->setStyleSheet("font-size: 80px; font-weight: 300; margin-right: 100px;");
  main_layout->addWidget(body);

  main_layout->addStretch();

  // reboot + start over buttons
  QHBoxLayout *blayout = new QHBoxLayout();
  blayout->setSpacing(50);
  main_layout->addLayout(blayout, 0);

  QPushButton *reboot = new QPushButton("Reboot device");
  reboot->setObjectName("navBtn");
  blayout->addWidget(reboot);
  QObject::connect(reboot, &QPushButton::clicked, this, [=]() {
    Hardware::reboot();
  });

  QPushButton *restart = new QPushButton("Start over");
  restart->setObjectName("navBtn");
  restart->setProperty("primary", true);
  blayout->addWidget(restart);
  QObject::connect(restart, &QPushButton::clicked, this, [=]() {
    setCurrentIndex(2);
  });

  widget->setStyleSheet(R"(
    QLabel {
      margin-left: 117;
    }
  )");
  return widget;
}

void Setup::prevPage() {
  setCurrentIndex(currentIndex() - 1);
}

void Setup::nextPage() {
  setCurrentIndex(currentIndex() + 1);
}

Setup::Setup(QWidget *parent) : QStackedWidget(parent) {
  std::stringstream buffer;
  buffer << std::ifstream("/sys/class/hwmon/hwmon1/in1_input").rdbuf();
  float voltage = (float)std::atoi(buffer.str().c_str()) / 1000.;
  if (voltage < 7) {
    addWidget(low_voltage());
  }

  addWidget(getting_started());
  addWidget(network_setup());
  addWidget(software_selection());

  downloading_widget = downloading();
  addWidget(downloading_widget);

  failed_widget = download_failed();
  addWidget(failed_widget);

  QObject::connect(this, &Setup::finished, [=](bool success) {
    // hide setup on success
    qDebug() << "finished" << success;
    if (success) {
      QTimer::singleShot(3000, this, &QWidget::hide);
    } else {
      setCurrentWidget(failed_widget);
    }
  });

  // TODO: revisit pressed bg color
  setStyleSheet(R"(
    * {
      color: white;
      font-family: Inter;
    }
    Setup {
      background-color: black;
    }
    QPushButton#navBtn {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #333333;
    }
    QPushButton#navBtn:disabled, QPushButton[primary='true']:disabled {
      color: #808080;
      background-color: #333333;
    }
    QPushButton#navBtn:pressed {
      background-color: #444444;
    }
    QPushButton[primary='true'], #navBtn[primary='true'] {
      background-color: #465BEA;
    }
    QPushButton[primary='true']:pressed, #navBtn:pressed[primary='true'] {
      background-color: #3049F4;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Setup setup;
  setMainWindow(&setup);
  return a.exec();
}
