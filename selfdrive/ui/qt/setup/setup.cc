#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "setup.hpp"
#include "qt_window.hpp"

#define USER_AGENT "AGNOS-0.1"

int download(std::string url) {
  CURL *curl;
  curl = curl_easy_init();
  if (!curl) return -1;

  char tmpfile[] = "/tmp/installer_XXXXXX";
  FILE *fp = fdopen(mkstemp(tmpfile), "w");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);
  curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  fclose(fp);

  rename(tmpfile, "/tmp/installer");
  return 0;
}

QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(
    font-size: 100px;
    font-weight: bold;
  )");
  return l;
}

QWidget * Setup::getting_started() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(200, 100, 200, 100);

  main_layout->addWidget(title_label("Getting Started"), 0, Qt::AlignCenter);

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
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  main_layout->addWidget(title_label("Connect to WiFi"), 0, Qt::AlignCenter);

  QPushButton *btn = new QPushButton("Continue");
  main_layout->addWidget(btn);
  QObject::connect(btn, SIGNAL(released()), this, SLOT(nextPage()));

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * Setup::software_selection() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  main_layout->addWidget(title_label("Choose Software"), 0, Qt::AlignCenter);

  QPushButton *dashcam_btn = new QPushButton("Dashcam");
  main_layout->addWidget(dashcam_btn);
  QObject::connect(dashcam_btn, SIGNAL(released()), this, SLOT(nextPage()));

  main_layout->addSpacing(50);

  const char* env_url = getenv("CUSTOM_URL");
  QString default_url = env_url == NULL ? "" : QString::fromStdString(env_url);
  url_input = new QLineEdit(default_url);
  url_input->setStyleSheet(R"(
    color: black;
    background-color: white;
    font-size: 55px;
    padding: 40px;
  )");
  main_layout->addWidget(url_input);

  QPushButton *custom_btn = new QPushButton("Custom");
  main_layout->addWidget(custom_btn);
  QObject::connect(custom_btn, SIGNAL(released()), this, SLOT(nextPage()));

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

void Setup::nextPage() {
  setCurrentIndex(currentIndex() + 1);

  // start download
  if (currentIndex() == count() - 1)  {
    std::string url = url_input->text().toStdString();
    download(url);
  }
}

Setup::Setup(QWidget *parent) {
  addWidget(getting_started());
  addWidget(network_setup());
  addWidget(software_selection());
  addWidget(downloading());

  setStyleSheet(R"(
    QWidget {
      color: white;
      background-color: black;
    }
    QPushButton {
      font-size: 60px;
      padding: 60px;
      width: 800px;
      color: white;
      background-color: blue;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Setup setup;
  setMainWindow(&setup);
  return a.exec();
}
