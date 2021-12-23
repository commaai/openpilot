#include "selfdrive/ui/replay/replaywidget.h"

#include <QVBoxLayout>

RouteSelector::RouteSelector(QWidget *parent) : QWidget(parent) {
}

ThumbnailsWidget::ThumbnailsWidget(QWidget *parent) : QWidget(parent) {
}

void ThumbnailsWidget::paintEvent(QPaintEvent *event) {
  QWidget::paintEvent(event);
}

TimelineWidget::TimelineWidget(QWidget *parent) : QWidget(parent) {
  setFixedHeight(160);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  thumbnails = new ThumbnailsWidget(this);
  main_layout->addWidget(thumbnails);
  slider = new QSlider(Qt::Horizontal, this);
  QObject::connect(slider, &QSlider::sliderReleased, this, &TimelineWidget::sliderReleased);
  main_layout->addWidget(slider);
}

void TimelineWidget::sliderReleased() {
}

ReplayWidget::ReplayWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  cam_view = new CameraViewWidget("camerad", VISION_STREAM_RGB_BACK, false);
  main_layout->addWidget(cam_view);
  timeline = new TimelineWidget(this);
  main_layout->addWidget(timeline);
}

void ReplayWidget::replayRoute(const QString &route) {
  replay.reset(new Replay(route, {}, {}));
  if (!replay->load()) {
    qInfo() << "failed to load route " << route;
    return;
  }

  // get thumbnails
  // for (const auto& [n, seg] : replay->segments()) {

  // }

  replay->start();
}