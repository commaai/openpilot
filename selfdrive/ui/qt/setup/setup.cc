#include "selfdrive/ui/qt/setup/setup.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>

#include <curl/curl.h>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/network/networking.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/input.h"

const std::string USER_AGENT = "AGNOSSetup-";
const QString TEST_URL = "https://openpilot.comma.ai";

bool is_elf(char *fname) {
  FILE *fp = fopen(fname, "rb");
  if (fp == NULL) {
    return false;
  }
  char buf[4];
  size_t n = fread(buf, 1, 4, fp);
  fclose(fp);
  return n == 4 && buf[0] == 0x7f && buf[1] == 'E' && buf[2] == 'L' && buf[3] == 'F';
}

void Setup::download(QString url) {
  // autocomplete incomplete urls
  if (QRegularExpression("^([^/.]+)/([^/]+)$").match(url).hasMatch()) {
    url.prepend("https://installer.comma.ai/");
  }

  CURL *curl = curl_easy_init();
  if (!curl) {
    emit finished(url, tr("Something went wrong. Reboot the device."));
    return;
  }

  auto version = util::read_file("/VERSION");

  struct curl_slist *list = NULL;
  list = curl_slist_append(list, ("X-openpilot-serial: " + Hardware::get_serial()).c_str());

  char tmpfile[] = "/tmp/installer_XXXXXX";
  FILE *fp = fdopen(mkstemp(tmpfile), "w");

  curl_easy_setopt(curl, CURLOPT_URL, url.toStdString().c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, (USER_AGENT + version).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

  int ret = curl_easy_perform(curl);
  long res_status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &res_status);

  if (ret != CURLE_OK || res_status != 200) {
    emit finished(url, tr("Ensure the entered URL is valid, and the device’s internet connection is good."));
  } else if (!is_elf(tmpfile)) {
    emit finished(url, tr("No custom software found at this URL."));
  } else {
    rename(tmpfile, "/tmp/installer");

    FILE *fp_url = fopen("/tmp/installer_url", "w");
    fprintf(fp_url, "%s", url.toStdString().c_str());
    fclose(fp_url);

    emit finished(url);
  }

  curl_slist_free_all(list);
  curl_easy_cleanup(curl);
  fclose(fp);
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
  triangle->setPixmap(QPixmap(ASSET_PATH + "offroad/icon_warning.png"));
  inner_layout->addWidget(triangle, 0, Qt::AlignTop | Qt::AlignLeft);
  inner_layout->addSpacing(80);

  QLabel *title = new QLabel(tr("WARNING: Low Voltage"));
  title->setStyleSheet("font-size: 90px; font-weight: 500; color: #FF594F;");
  inner_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  inner_layout->addSpacing(25);

  QLabel *body = new QLabel(tr("Power your device in a car with a harness or proceed at your own risk."));
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  body->setStyleSheet("font-size: 80px; font-weight: 300;");
  inner_layout->addWidget(body);

  inner_layout->addStretch();

  // power off + continue buttons
  QHBoxLayout *blayout = new QHBoxLayout();
  blayout->setSpacing(50);
  main_layout->addLayout(blayout, 0);

  QPushButton *poweroff = new QPushButton(tr("Power off"));
  poweroff->setObjectName("navBtn");
  blayout->addWidget(poweroff);
  QObject::connect(poweroff, &QPushButton::clicked, this, [=]() {
    Hardware::poweroff();
  });

  QPushButton *cont = new QPushButton(tr("Continue"));
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

  QLabel *title = new QLabel(tr("Getting Started"));
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  vlayout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  vlayout->addSpacing(90);
  QLabel *desc = new QLabel(tr("Before we get on the road, let’s finish installation and cover some details."));
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
  QLabel *title = new QLabel(tr("Connect to Wi-Fi"));
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

  QPushButton *back = new QPushButton(tr("Back"));
  back->setObjectName("navBtn");
  QObject::connect(back, &QPushButton::clicked, this, &Setup::prevPage);
  blayout->addWidget(back);

  QPushButton *cont = new QPushButton();
  cont->setObjectName("navBtn");
  cont->setProperty("primary", true);
  QObject::connect(cont, &QPushButton::clicked, [=]() {
    auto w = currentWidget();
    QTimer::singleShot(0, [=]() {
      setCurrentWidget(downloading_widget);
    });
    QString url = InputDialog::getText(tr("Enter URL"), this, tr("for Custom Software"));
    if (!url.isEmpty()) {
      QTimer::singleShot(1000, this, [=]() {
        download(url);
      });
    } else {
      setCurrentWidget(w);
    }
  });
  blayout->addWidget(cont);

  // setup timer for testing internet connection
  HttpRequest *request = new HttpRequest(this, false, 2500);
  QObject::connect(request, &HttpRequest::requestDone, [=](const QString &, bool success) {
    cont->setEnabled(success);
    if (success) {
      const bool wifi = networking->wifi->currentNetworkType() == NetworkType::WIFI;
      cont->setText(wifi ? tr("Continue") : tr("Continue without Wi-Fi"));
    } else {
      cont->setText(tr("Waiting for internet"));
    }
    repaint();
  });
  request->sendRequest(TEST_URL);
  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    if (!request->active() && cont->isVisible()) {
      request->sendRequest(TEST_URL);
    }
  });
  timer->start(1000);

  return widget;
}

QWidget * Setup::downloading() {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  QLabel *txt = new QLabel(tr("Downloading..."));
  txt->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(txt, 0, Qt::AlignCenter);
  return widget;
}

QWidget * Setup::download_failed(QLabel *url, QLabel *body) {
  QWidget *widget = new QWidget();
  QVBoxLayout *main_layout = new QVBoxLayout(widget);
  main_layout->setContentsMargins(55, 185, 55, 55);
  main_layout->setSpacing(0);

  QLabel *title = new QLabel(tr("Download Failed"));
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  main_layout->addSpacing(67);

  url->setWordWrap(true);
  url->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  url->setStyleSheet("font-family: \"JetBrains Mono\"; font-size: 64px; font-weight: 400; margin-right: 100px;");
  main_layout->addWidget(url);

  main_layout->addSpacing(48);

  body->setWordWrap(true);
  body->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  body->setStyleSheet("font-size: 80px; font-weight: 300; margin-right: 100px;");
  main_layout->addWidget(body);

  main_layout->addStretch();

  // reboot + start over buttons
  QHBoxLayout *blayout = new QHBoxLayout();
  blayout->setSpacing(50);
  main_layout->addLayout(blayout, 0);

  QPushButton *reboot = new QPushButton(tr("Reboot device"));
  reboot->setObjectName("navBtn");
  blayout->addWidget(reboot);
  QObject::connect(reboot, &QPushButton::clicked, this, [=]() {
    Hardware::reboot();
  });

  QPushButton *restart = new QPushButton(tr("Start over"));
  restart->setObjectName("navBtn");
  restart->setProperty("primary", true);
  blayout->addWidget(restart);
  QObject::connect(restart, &QPushButton::clicked, this, [=]() {
    setCurrentIndex(1);
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
  if (std::getenv("MULTILANG")) {
    selectLanguage();
  }

  std::stringstream buffer;
  buffer << std::ifstream("/sys/class/hwmon/hwmon1/in1_input").rdbuf();
  float voltage = (float)std::atoi(buffer.str().c_str()) / 1000.;
  if (voltage < 7) {
    addWidget(low_voltage());
  }

  addWidget(getting_started());
  addWidget(network_setup());

  downloading_widget = downloading();
  addWidget(downloading_widget);

  QLabel *url_label = new QLabel();
  QLabel *body_label = new QLabel();
  failed_widget = download_failed(url_label, body_label);
  addWidget(failed_widget);

  QObject::connect(this, &Setup::finished, [=](const QString &url, const QString &error) {
    qDebug() << "finished" << url << error;
    if (error.isEmpty()) {
      // hide setup on success
      QTimer::singleShot(3000, this, &QWidget::hide);
    } else {
      url_label->setText(url);
      body_label->setText(error);
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

void Setup::selectLanguage() {
  QMap<QString, QString> langs = getSupportedLanguages();
  QString selection = MultiOptionDialog::getSelection(tr("Select a language"), langs.keys(), "", this);
  if (!selection.isEmpty()) {
    QString selectedLang = langs[selection];
    Params().put("LanguageSetting", selectedLang.toStdString());
    if (translator.load(":/" + selectedLang)) {
      qApp->installTranslator(&translator);
    }
  }
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  Setup setup;
  setMainWindow(&setup);
  return a.exec();
}
