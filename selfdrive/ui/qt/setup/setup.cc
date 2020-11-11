#include <cstdlib>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QApplication>
#include <QDesktopWidget>

#include "setup.hpp"

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif


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
  Setup setup = Setup();
  setup.setFixedSize(w, h);
  setup.show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", window->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  setup.showFullScreen();
#endif

  return a.exec();
}
