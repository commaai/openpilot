#include "selfdrive/ui/replay/replaywidget.h"

#include <QPainter>
#include <QVBoxLayout>

RouteSelector::RouteSelector(QWidget *parent) : QWidget(parent) {}

ThumbnailsWidget::ThumbnailsWidget(QWidget *parent) : QWidget(parent) {}

void ThumbnailsWidget::paintEvent(QPaintEvent *event) {
  if (!thumbnails_ || thumbnails_->empty()) return;

  const int thumb_width = width() / thumbnails_->size();
  QPainter p(this);
  int x = 0;
  for (const auto &thumb : *thumbnails_) {
    p.drawPixmap(x, 0, thumb.scaled(thumb_width, thumb_width, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    x += thumb_width;
  }
}

TimelineWidget::TimelineWidget(QWidget *parent) : QWidget(parent) {
  setFixedHeight(160);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  thumbnails = new ThumbnailsWidget(this);
  main_layout->addWidget(thumbnails);
  slider = new QSlider(Qt::Horizontal, this);
  QObject::connect(slider, &QSlider::sliderReleased, [=]() {
    emit sliderReleased(slider->value());
  });
  main_layout->addWidget(slider);

  timer.callOnTimeout([=]() {
    slider->setValue(slider->value() + 1);
  });
}

void TimelineWidget::setThumbnail(std::vector<QPixmap> *t) {
  thumbnails->setThumbnail(t);
  slider->setMinimum(0);
  slider->setSingleStep(1);
  slider->setMaximum(t->size() * 60);
  slider->setValue(0);
  timer.start(1000);
}

ReplayWidget::ReplayWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  cam_view = new CameraViewWidget("camerad", VISION_STREAM_RGB_BACK, false);
  main_layout->addWidget(cam_view);
  timeline = new TimelineWidget(this);
  QObject::connect(timeline, &TimelineWidget::sliderReleased, this, &ReplayWidget::seekTo);
  main_layout->addWidget(timeline);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
      font-size: 60px;
    }
  )");
}

void ReplayWidget::replayRoute(const QString &route) {
  replay.reset(new Replay(route, {}, {}));
  if (!replay->load()) {
    qInfo() << "failed to load route " << route;
    return;
  }

  // get thumbnails
  // TODO: put in thread
  thumbnails.clear();
  for (const auto &[n, _] : replay->segments()) {
    LogReader log;
    if (!log.load(replay->route()->at(n).qlog.toStdString(), nullptr, true, 0, 3)) continue;

    for (const Event *e : log.events) {
      if (e->which == cereal::Event::Which::THUMBNAIL) {
        const auto thumb = e->event.getThumbnail().getThumbnail();
        QPixmap pixmap;
        bool ret = pixmap.loadFromData(thumb.begin(), thumb.size(), "jpeg");
        if (ret) {
          thumbnails.push_back(pixmap);
          break;
        }
      }
    }
  }
  timeline->setThumbnail(&thumbnails);
  replay->start();
}

void ReplayWidget::seekTo(int pos) {
}
