#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include "setup.hpp"
#include "offroad/networking.hpp"
#include "widgets/input_field.hpp"
#include "qt_window.hpp"

#define USER_AGENT "AGNOSSetup-0.1"

void Setup::download(QString url) {
  setCurrentIndex(count() - 1);

  CURL *curl;
  curl = curl_easy_init();
  // TODO: exit with return code
  if (!curl) return;

  char tmpfile[] = "/tmp/installer_XXXXXX";
  FILE *fp = fdopen(mkstemp(tmpfile), "w");

  curl_easy_setopt(curl, CURLOPT_URL, url.toStdString().c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);
  curl_easy_perform(curl);
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

QPushButton * nav_btn(QString text) {
  QPushButton *btn = new QPushButton(text);
  btn->setStyleSheet(R"(
    QPushButton {
      background: none;
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 7px solid white;
      border-radius: 20px;
      font-size: 50px;
    }
  )");
  return btn;
}

QWidget * Setup::getting_started() {

  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(50, 50, 50, 50);

  main_layout->addWidget(title_label("Getting Started"), 0, Qt::AlignLeft | Qt::AlignTop);

  QLabel *body = new QLabel("Before we get on the road, let's finish\ninstallation and cover some details.");
  body->setStyleSheet(R"(font-size: 65px;)");
  main_layout->addWidget(body, 0, Qt::AlignCenter);

  main_layout->addSpacing(100);

  QPushButton *btn = new QPushButton("Continue");
  main_layout->addWidget(btn);
  QObject::connect(btn, SIGNAL(released()), this, SLOT(nextPage()));

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * Setup::network_setup() {

  // TODO: wait for internet, make it nice

  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(50, 50, 50, 50);

  main_layout->addWidget(title_label("Connect to WiFi"), 0, Qt::AlignTop);

  Networking *wifi = new Networking(this, false);
  main_layout->addWidget(wifi);
  QObject::connect(wifi, &Networking::openKeyboard, this, [=]() {
    this->continue_btn->setVisible(false);
  });
  QObject::connect(wifi, &Networking::closeKeyboard, this, [=]() {
    this->continue_btn->setVisible(true);
  });

  continue_btn = nav_btn("Continue");
  main_layout->addWidget(continue_btn, 0, Qt::AlignRight);
  QObject::connect(continue_btn, SIGNAL(released()), this, SLOT(nextPage()));

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * Setup::software_selection() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  main_layout->addWidget(title_label("Choose Software"), 0, Qt::AlignCenter);

  main_layout->addSpacing(50);

  QPushButton *dashcam_btn = new QPushButton("Dashcam");
  main_layout->addWidget(dashcam_btn);
  QObject::connect(dashcam_btn, &QPushButton::released, this,  [=]() {
    this->download("https://dashcam.comma.ai");
  });

  main_layout->addSpacing(50);

  QPushButton *custom_btn = new QPushButton("Custom");
  main_layout->addWidget(custom_btn);
  QObject::connect(custom_btn, SIGNAL(released()), this, SLOT(nextPage()));

  main_layout->addSpacing(100);

  QPushButton *prev_btn = new QPushButton("Back");
  main_layout->addWidget(prev_btn);
  QObject::connect(prev_btn, SIGNAL(released()), this, SLOT(prevPage()));

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * Setup::custom_software() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(50);

  main_layout->addWidget(title_label("Custom Software"), Qt::AlignTop | Qt::AlignHCenter);

  InputField *input = new InputField();
  input->setPromptText("Enter URL");
  main_layout->addWidget(input);

  QObject::connect(input, SIGNAL(emitText(QString)), this, SLOT(download(QString)));

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * Setup::downloading() {
  QVBoxLayout *main_layout = new QVBoxLayout();

  main_layout->addWidget(title_label("Downloading..."), 0, Qt::AlignCenter);

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

void Setup::prevPage() {
  setCurrentIndex(currentIndex() - 1);
}

void Setup::nextPage() {
  setCurrentIndex(currentIndex() + 1);
}

Setup::Setup(QWidget *parent) {
  addWidget(getting_started());
  addWidget(network_setup());
  addWidget(software_selection());
  addWidget(custom_software());
  addWidget(downloading());

  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 7px solid white;
      border-radius: 20px;
      font-size: 50px;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Setup setup;
  setMainWindow(&setup);
  return a.exec();
}
