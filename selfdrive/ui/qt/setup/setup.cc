#include "selfdrive/ui/qt/setup/setup.h"

#include <cstdio>
#include <cstdlib>

#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>

#include <curl/curl.h>

#include "selfdrive/hardware/hw.h"

#include "selfdrive/ui/qt/offroad/networking.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/qt_window.h"

#define USER_AGENT "AGNOSSetup-0.1"

void Setup::download(QString url) {
  QCoreApplication::processEvents(QEventLoop::AllEvents, 1000);
  setCurrentIndex(count() - 2);

  CURL *curl = curl_easy_init();
  if (!curl) {
    emit downloadFailed();
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
    emit downloadFailed();
  }
  curl_easy_cleanup(curl);
  fclose(fp);

  rename(tmpfile, "/tmp/installer");
}

QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(
    font-size: 100px;
    font-weight: 500;
  )");
  return l;
}

QWidget * Setup::build_page(QString title, QWidget *content, bool next, bool prev) {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);

  main_layout->setMargin(50);
  main_layout->addWidget(title_label(title), 0, Qt::AlignLeft | Qt::AlignTop);

  main_layout->addWidget(content);

  QHBoxLayout *nav_layout = new QHBoxLayout();

  if (prev) {
    QPushButton *back_btn = new QPushButton("Back");
    nav_layout->addWidget(back_btn, 1, Qt::AlignBottom | Qt::AlignLeft);
    QObject::connect(back_btn, &QPushButton::released, this, &Setup::prevPage);
  }

  if (next) {
    QPushButton *continue_btn = new QPushButton("Continue");
    nav_layout->addWidget(continue_btn, 0, Qt::AlignBottom | Qt::AlignRight);
    QObject::connect(continue_btn, &QPushButton::released, this, &Setup::nextPage);
  }

  main_layout->addLayout(nav_layout, 0);
  return widget;
}

QWidget * Setup::getting_started() {
  QWidget *widget = new QWidget();

  QHBoxLayout *main_layout = new QHBoxLayout(widget);
  main_layout->setMargin(0);

  QVBoxLayout *vlayout = new QVBoxLayout();
  vlayout->setContentsMargins(165, 280, 0, 0);
  main_layout->addLayout(vlayout);

  QLabel *title = new QLabel("Getting Started");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  vlayout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

  vlayout->addSpacing(90);
  QLabel *desc = new QLabel("Before we get on the road, let’s finish installation and cover some details.");
  desc->setWordWrap(true);
  desc->setStyleSheet("font-size: 80px; font-weight: 300; margin-right: 50px;");
  vlayout->addWidget(desc, 0, Qt::AlignLeft | Qt::AlignTop);

  vlayout->addStretch();

  QPushButton *btn = new QPushButton(">");
  btn->setFixedSize(310, 1080);
  btn->setProperty("primary", true);
  btn->setStyleSheet(R"(
    QPushButton {
      font-size: 90px;
      border: none;
      border-radius: 0;
    }
  )");
  main_layout->addWidget(btn, 0, Qt::AlignRight);
  QObject::connect(btn, &QPushButton::clicked, this, &Setup::nextPage);

  return widget;
}

QWidget * Setup::network_setup() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 50, 55, 50);

  // title
  QLabel *title = new QLabel("Connect to WiFi");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

  // wifi widget
  Networking *wifi = new Networking(this, false);
  main_layout->addWidget(wifi, 1);

  // back + cotninue buttons
  QHBoxLayout *blayout = new QHBoxLayout;
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  QPushButton *back = new QPushButton("Back");
  back->setObjectName("navBtn");
  QObject::connect(back, &QPushButton::clicked, this, &Setup::prevPage);
  blayout->addWidget(back);

  QPushButton *cont = new QPushButton("Continue");
  cont->setObjectName("navBtn");
  QObject::connect(cont, &QPushButton::clicked, this, &Setup::nextPage);
  blayout->addWidget(cont);

  widget->setStyleSheet(R"(
  )");

  return widget;
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

  QPushButton *dashcam = new QPushButton("Dashcam");
  dashcam->setObjectName("radioBtn");
  dashcam->setCheckable(true);
  group->addButton(dashcam);
  main_layout->addWidget(dashcam);

  main_layout->addSpacing(30);

  QPushButton *custom = new QPushButton("Custom Software");
  custom->setObjectName("radioBtn");
  custom->setCheckable(true);
  group->addButton(custom);
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

  // TODO: disabled state color?
  QPushButton *cont = new QPushButton("Continue");
  cont->setObjectName("navBtn");
  cont->setEnabled(false);
  QObject::connect(cont, &QPushButton::clicked, this, &Setup::nextPage);
  blayout->addWidget(cont);

  connect(group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), [=](QAbstractButton *btn) {
    btn->setChecked(true);
    cont->setEnabled(true);
  });

  widget->setStyleSheet(R"(
    QPushButton#radioBtn {
      height: 230;
      padding-left: 100px;
      padding-right: 100px;
      text-align: left;
      font-size: 80px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #4F4F4F;
    }
    QPushButton#radioBtn:checked {
      background-color: #465BEA;
    }
  )");

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
  main_layout->setContentsMargins(55, 55, 55, 55);

  QLabel *title = new QLabel("Download Failed");
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

  title->move(224, 184);

  QLabel *body = new QLabel("Ensure the entered URL is valid, and the device's network connection is good.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignHCenter);
  body->setStyleSheet("font-size: 80px; font-weight: 300;");
  main_layout->addWidget(body);

  QHBoxLayout *nav_layout = new QHBoxLayout();

  QPushButton *reboot_btn = new QPushButton("Reboot Device");
  reboot_btn->setStyleSheet("background-color: #333333;");
  nav_layout->addWidget(reboot_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(reboot_btn, &QPushButton::released, this, [=]() {
    Hardware::reboot();
  });

  QPushButton *restart_btn = new QPushButton("Start over");
  restart_btn->setStyleSheet("background-color: #465BEA;");
  nav_layout->addWidget(restart_btn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(restart_btn, &QPushButton::released, this, [=]() {
    setCurrentIndex(0);
  });

  main_layout->addLayout(nav_layout, 0);
  return widget;
}

void Setup::prevPage() {
  setCurrentIndex(currentIndex() - 1);
}

void Setup::nextPage() {
  setCurrentIndex(currentIndex() + 1);
}

Setup::Setup(QWidget *parent) : QStackedWidget(parent) {
  addWidget(getting_started());
  addWidget(network_setup());
  addWidget(software_selection());
  addWidget(downloading());
  addWidget(download_failed());

  QObject::connect(this, &Setup::downloadFailed, this, &Setup::nextPage);

  // TODO: revisit pressed bg color
  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
    }
    Setup {
      background-color: black;
    }
    *[primary='true'] {
      background-color: #465BEA;
    }
    *[primary='true']:pressed {
      background-color: #3049F4;
    }
    QPushButton#navBtn {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #333333;
    }
    QPushButton#navBtn:pressed {
      background-color: #444444;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Setup setup;
  setMainWindow(&setup);
  return a.exec();
}
