#include "selfdrive/ui/replay/replaywidget.h"

#include <QDir>
#include <QPainter>
#include <QRegExp>
#include <QVBoxLayout>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

RouteSelector::RouteSelector(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(25, 25, 25, 25);

  auto label = new QLabel("Local routes:");
  main_layout->addWidget(label);
  auto w = new ListWidget(this);
  ScrollView *scroll = new ScrollView(w);
  main_layout->addWidget(scroll);

  QDir log_dir(Path::log_root().c_str());
  for (const auto &folder : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
    if (int pos = folder.lastIndexOf("--"); pos != -1) {
      if (QString route = folder.left(pos); !route.isEmpty()) {
        route_names.insert(route);
      }
    }
  }
  for (auto &route : route_names) {
    ButtonControl *c = new ButtonControl(route, "view");
    QObject::connect(c, &ButtonControl::clicked, [=, r = route]() {
      emit selectRoute(r);
    });
    w->addItem(c);
  }
  main_layout->addStretch();

  setStyleSheet(R"(
    QPushButton {
     border:none;
      background-color: black;
    }
  )");
}

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
  stacked_layout = new QStackedLayout(this);

  route_selector = new RouteSelector(this);
  QObject::connect(route_selector, &RouteSelector::selectRoute, [=](const QString &route) {
    replayRoute(route, Path::log_root().c_str());
  });

  QLabel *loading = new QLabel("Loading...");
  loading->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
  loading->setStyleSheet("font-size: 150px; font-weight: 500;");

  QWidget *r = new QWidget(this);
  QVBoxLayout *main_layout = new QVBoxLayout(r);
  cam_view = new CameraViewWidget("camerad", VISION_STREAM_RGB_BACK, false);
  main_layout->addWidget(cam_view);
  timeline = new TimelineWidget(this);
  QObject::connect(timeline, &TimelineWidget::sliderReleased, this, &ReplayWidget::seekTo);
  main_layout->addWidget(timeline);

  stacked_layout->addWidget(route_selector);
  stacked_layout->addWidget(r);
  stacked_layout->addWidget(loading);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
      font-size: 60px;
    }
  )");
}

void ReplayWidget::replayRoute(const QString &route, const QString &data_dir) {
  stacked_layout->setCurrentIndex(2);

  replay.reset(new Replay(route, {}, {}, nullptr, REPLAY_FLAG_NONE, data_dir));
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

  replay->start();
  timeline->setThumbnail(&thumbnails);
  stacked_layout->setCurrentIndex(1);
}

void ReplayWidget::seekTo(int pos) {
  replay->seekTo(pos, false);
}
