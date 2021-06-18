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
  QLabel *body = new QLabel("Before we get on the road, let's finish\ninstallation and cover some details.");
  body->setAlignment(Qt::AlignHCenter);
  body->setStyleSheet(R"(font-size: 80px;)");
  return build_page("Getting Started", body, true, false);
}

QWidget * Setup::network_setup() {
  Networking *wifi = new Networking(this, false);
  return build_page("Connect to WiFi", wifi, true, true);
}

QWidget * Setup::software_selection() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);

  QPushButton *dashcam_btn = new QPushButton("Dashcam");
  main_layout->addWidget(dashcam_btn);
  QObject::connect(dashcam_btn, &QPushButton::released, this, [=]() {
    this->download("https://dashcam.comma.ai");
  });

  main_layout->addSpacing(50);

  QPushButton *custom_btn = new QPushButton("Custom");
  main_layout->addWidget(custom_btn);
  QObject::connect(custom_btn, &QPushButton::released, this, [=]() {
    QString input_url = InputDialog::getText("Enter URL");
    if (input_url.size()) {
      this->download(input_url);
    }
  });
  return build_page("Choose Software", widget, false, true);
}

QWidget * Setup::downloading() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->addWidget(title_label("Downloading..."), 0, Qt::AlignCenter);
  return widget;
}

QWidget * Setup::download_failed() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(50, 50, 50, 50);
  main_layout->addWidget(title_label("Download Failed"), 0, Qt::AlignLeft | Qt::AlignTop);

  QLabel *body = new QLabel("Ensure the entered URL is valid, and the device's network connection is good.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignHCenter);
  body->setStyleSheet(R"(font-size: 80px;)");
  main_layout->addWidget(body);

  QHBoxLayout *nav_layout = new QHBoxLayout();

  QPushButton *reboot_btn = new QPushButton("Reboot");
  nav_layout->addWidget(reboot_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(reboot_btn, &QPushButton::released, this, [=]() {
    if (Hardware::TICI()) {
      std::system("sudo reboot");
    }
  });

  QPushButton *restart_btn = new QPushButton("Start over");
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
