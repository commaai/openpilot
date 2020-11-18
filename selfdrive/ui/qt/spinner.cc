#include <stdio.h>
#include <string>
#include <iostream>

#include <QString>
#include <QGridLayout>
#include <QApplication>
#include <QDesktopWidget>

#ifdef QCOM2
#include <qpa/qplatformnativeinterface.h>
#include <QPlatformSurfaceEvent>
#include <wayland-client-protocol.h>
#endif

#include "spinner.hpp"


Spinner::Spinner(QWidget *parent) {
  QGridLayout *main_layout = new QGridLayout();
  main_layout->setSpacing(0);
  main_layout->setContentsMargins(200, 200, 200, 200);

  const int img_size = 360;

  comma = new QLabel();
  comma->setPixmap(QPixmap("../assets/img_spinner_comma.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  comma->setFixedSize(img_size, img_size);
  main_layout->addWidget(comma, 0, 0, Qt::AlignHCenter | Qt::AlignVCenter);

  track_img = QPixmap("../assets/img_spinner_track.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  track = new QLabel();
  track->setPixmap(track_img);
  track->setFixedSize(img_size, img_size);
  main_layout->addWidget(track, 0, 0, Qt::AlignHCenter | Qt::AlignVCenter);

  text = new QLabel();
  text->setVisible(false);
  main_layout->addWidget(text, 1, 0, Qt::AlignHCenter);

  progress_bar = new QProgressBar();
  progress_bar->setRange(5, 100);
  progress_bar->setTextVisible(false);
  progress_bar->setVisible(false);
  main_layout->addWidget(progress_bar, 1, 0, Qt::AlignHCenter);

  setLayout(main_layout);
  setStyleSheet(R"(
    Spinner {
      background-color: black;
    }
    QLabel {
      color: white;
      font-size: 80px;
    }
    QProgressBar {
      background-color: #373737;
      height: 20px;
      width: 1000px;
      border solid white;
      border-radius: 10px;
    }
    QProgressBar::chunk {
      border-radius: 10px;
      background-color: white;
    }
  )");

  rotate_timer = new QTimer(this);
  rotate_timer->start(1000/30.);
  QObject::connect(rotate_timer, SIGNAL(timeout()), this, SLOT(rotate()));

  notifier = new QSocketNotifier(fileno(stdin), QSocketNotifier::Read);
  QObject::connect(notifier, SIGNAL(activated(int)), this, SLOT(update(int)));
};

void Spinner::rotate() {
  transform.rotate(5);

  QPixmap r = track_img.transformed(transform.rotate(5), Qt::SmoothTransformation);
  int x = (r.width() - track->width()) / 2;
  int y = (r.height() - track->height()) / 2;
  track->setPixmap(r.copy(x, y, track->width(), track->height()));
};

void Spinner::update(int n) {
  std::string line;
  std::getline(std::cin, line);

  if (line.length()) {
    bool number = std::all_of(line.begin(), line.end(), ::isdigit);
    text->setVisible(!number);
    progress_bar->setVisible(number);
    text->setText(QString::fromStdString(line));
    if (number) {
      progress_bar->setValue(std::stoi(line));
    }
  }
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  Spinner *spinner = new Spinner();

  // TODO: get size from QScreen, doesn't work on tici
#ifdef QCOM2
  int w = 2160, h = 1080;
#else
  int w = 1920, h = 1080;
#endif
  spinner->setFixedSize(w, h);
  spinner->show();

#ifdef QCOM2
  QPlatformNativeInterface *native = QGuiApplication::platformNativeInterface();
  wl_surface *s = reinterpret_cast<wl_surface*>(native->nativeResourceForWindow("surface", spinner->windowHandle()));
  wl_surface_set_buffer_transform(s, WL_OUTPUT_TRANSFORM_270);
  wl_surface_commit(s);
  spinner->showFullScreen();
#endif

  return a.exec();
}
