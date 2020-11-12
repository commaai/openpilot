#include <stdio.h>
#include <curl/curl.h>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "setup.hpp"

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif


int download(std::string url) {
  CURL *curl;
  curl = curl_easy_init();
  if (!curl) return -1;

  FILE *fp;
  fp = fopen("/tmp/installer", "wb");
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt (curl, CURLOPT_VERBOSE, 1L);
  curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  fclose(fp);
  return 0;
}

QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(
    QLabel {
      font-size: 100px;
      font-weight: bold;
    }
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

  main_layout->addSpacing(100);

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
  layout->setCurrentIndex(layout->currentIndex() + 1);

  // start download
  if (layout->currentIndex() == layout->count() - 1)  {
    std::string url = url_input->text().toStdString();
    download(url);
  }
}

Setup::Setup(QWidget *parent) {
  layout = new QStackedLayout();
  layout->addWidget(getting_started());
  layout->addWidget(network_setup());
  layout->addWidget(software_selection());
  layout->addWidget(downloading());

  setLayout(layout);
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
#ifdef QCOM2
  int w = 2160, h = 1080;
#else
  int w = 1920, h = 1080;
#endif

  QApplication a(argc, argv);

  Setup setup;
  setup.setFixedSize(w, h);
  setup.show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", setup.windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  setup.showFullScreen();
#endif

  return a.exec();
}
