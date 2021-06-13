#include "selfdrive/ui/qt/spinner.h"

#include <cstdio>
#include <iostream>
#include <string>

#include <QApplication>
#include <QGridLayout>
#include <QPainter>
#include <QString>
#include <QTransform>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

TrackWidget::TrackWidget(QWidget *parent) : QWidget(parent) {
  setFixedSize(spinner_size);
  setAutoFillBackground(false);

  comma_img = QPixmap("../assets/img_spinner_comma.png").scaled(spinner_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  // pre-compute all the track imgs. make this a gif instead?
  QTransform transform;
  QPixmap track_img = QPixmap("../assets/img_spinner_track.png").scaled(spinner_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  for (auto &img : track_imgs) {
    img = track_img.transformed(transform.rotate(360/spinner_fps), Qt::SmoothTransformation);
  }

  m_anim.setDuration(1000);
  m_anim.setStartValue(0);
  m_anim.setEndValue(int(track_imgs.size() -1));
  m_anim.setLoopCount(-1);
  m_anim.start();
  connect(&m_anim, SIGNAL(valueChanged(QVariant)), SLOT(update()));
}

void TrackWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  QRect bg(0, 0, painter.device()->width(), painter.device()->height());
  QBrush bgBrush("#000000");
  painter.fillRect(bg, bgBrush);

  int track_idx = m_anim.currentValue().toInt();
  QRect rect(track_imgs[track_idx].rect());
  rect.moveCenter(bg.center());
  painter.drawPixmap(rect.topLeft(), track_imgs[track_idx]);

  rect = comma_img.rect();
  rect.moveCenter(bg.center());
  painter.drawPixmap(rect.topLeft(), comma_img);
}

// Spinner

Spinner::Spinner(QWidget *parent) : QWidget(parent) {
  QGridLayout *main_layout = new QGridLayout(this);
  main_layout->setSpacing(0);
  main_layout->setMargin(200);

  main_layout->addWidget(new TrackWidget(this), 0, 0, Qt::AlignHCenter | Qt::AlignVCenter);

  text = new QLabel();
  text->setVisible(false);
  main_layout->addWidget(text, 1, 0, Qt::AlignHCenter);

  progress_bar = new QProgressBar();
  progress_bar->setRange(5, 100);
  progress_bar->setTextVisible(false);
  progress_bar->setVisible(false);
  progress_bar->setFixedHeight(20);
  main_layout->addWidget(progress_bar, 1, 0, Qt::AlignHCenter);

  setStyleSheet(R"(
    Spinner {
      background-color: black;
    }
    * {
      background-color: transparent;
    }
    QLabel {
      color: white;
      font-size: 80px;
    }
    QProgressBar {
      background-color: #373737;
      width: 1000px;
      border solid white;
      border-radius: 10px;
    }
    QProgressBar::chunk {
      border-radius: 10px;
      background-color: white;
    }
  )");

  notifier = new QSocketNotifier(fileno(stdin), QSocketNotifier::Read);
  QObject::connect(notifier, &QSocketNotifier::activated, this, &Spinner::update);
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
  setQtSurfaceFormat();

  Hardware::set_display_power(true);
  Hardware::set_brightness(65);

  QApplication a(argc, argv);
  Spinner spinner;
  setMainWindow(&spinner);
  return a.exec();
}
