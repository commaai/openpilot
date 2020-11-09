#include <cstdlib>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QApplication>
#include <QDesktopWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif


QWidget * getting_started() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  QLabel *title = new QLabel("Getting Started");
  title->setStyleSheet(R"(
    QLabel {
      font-size: 80px;
      font-weight: bold;
    }
  )");
  title->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(title, Qt::AlignCenter);

  QLabel *body = new QLabel("Before we get on the road, let's finish\ninstallation and cover some details.");
  body->setStyleSheet(R"(
    QLabel {
      font-size: 60px;
    }
  )");
  body->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(body, Qt::AlignCenter);

  QPushButton *btn = new QPushButton("Continue");
  btn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      padding: 60px;
      width: 800px;
      color: white;
      background-color: blue;
    }
  )");
  main_layout->addWidget(btn);

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * network_setup() {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(100);

  QLabel *title = new QLabel("Connect to WiFi");
  title->setStyleSheet(R"(
    QLabel {
      font-size: 80px;
      font-weight: bold;
    }
  )");
  title->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(title, Qt::AlignCenter);

  QPushButton *btn = new QPushButton("Continue");
  btn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      padding: 60px;
      width: 800px;
      color: white;
      background-color: blue;
    }
  )");
  main_layout->addWidget(btn);

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}




int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  QWidget *window = new QWidget();

  // TODO: get size from QScreen, doesn't work on tici
#ifdef QCOM2
  int w = 2160, h = 1080;
#else
  int w = 1920, h = 1080;
#endif
  window->setFixedSize(w, h);

  QStackedLayout *layout = new QStackedLayout();
  layout->addWidget(getting_started());
  layout->addWidget(network_setup());

  window->setLayout(layout);
  window->setStyleSheet(R"(
    QWidget {
      color: white;
      background-color: black;
    }
  )");
  window->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", window->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  window->showFullScreen();
#endif

  return a.exec();
}
