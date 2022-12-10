#include "selfdrive/ui/qt/widgets/replay_widgets.h"

#include <QDir>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

// class RouteListWidget

RouteListWidget::RouteListWidget(QWidget *parent) : ListWidget(parent) {}

void RouteListWidget::showEvent(QShowEvent *event) {
  if (route_names.isEmpty()) {
    QDir log_dir(Path::log_root().c_str());
    for (const auto &folder : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
      if (int pos = folder.lastIndexOf("--"); pos != -1) {
        if (QString route = folder.left(pos); !route.isEmpty()) {
          route_names.insert(route);
        }
      }
    }
    for (auto &route : route_names) {
      ButtonControl *c = new ButtonControl(route, tr("replay"));
      QObject::connect(uiState(), &UIState::offroadTransition, c, &ButtonControl::setEnabled);
      QObject::connect(c, &ButtonControl::clicked, this, &RouteListWidget::buttonClicked);
      addItem(c);
    }
  }
  ListWidget::showEvent(event);
}

void RouteListWidget::buttonClicked() {
  ButtonControl *btn = qobject_cast<ButtonControl *>(sender());
  const QString route = btn->title();
  if (route == current_route) {
    current_route = "";
    emit uiState()->stopReplay();
    btn->setText(tr("replay"));
  } else {
    current_route = route;
    emit uiState()->startReplay(route, QString::fromStdString(Path::log_root()));
    btn->setText(tr("stop"));
  }
}

// class ReplayControls

ReplayControls::ReplayControls(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  QLabel *time_label = new QLabel("00:00");
  main_layout->addWidget(time_label);
  play_btn = new QPushButton("pause", this);
  main_layout->addWidget(play_btn);
  slider = new QSlider(Qt::Horizontal, this);
  slider->setSingleStep(0);
  main_layout->addWidget(slider);
  end_time_label = new QLabel(this);
  stop_btn = new QPushButton("stop", this);
  main_layout->addWidget(stop_btn);
  main_layout->addWidget(end_time_label);
  setStyleSheet(R"(
    QSlider {height: 60px;}
    QLabel {color:white;font-size:30px;}
  )");

  timer = new QTimer(this);
  timer->setInterval(1000);
  timer->callOnTimeout([this]() {
    slider->setValue(replay->currentSeconds());
  });

  QObject::connect(slider, &QSlider::sliderReleased, [this]() {
    replay->seekTo(slider->value(), true);
  });
  QObject::connect(stop_btn, &QPushButton::clicked, this, &ReplayControls::stop);
  QObject::connect(play_btn, &QPushButton::clicked, [this]() { replay->pause(!replay->isPaused()); });
  adjustPosition();
}

void ReplayControls::adjustPosition() {
  resize(parentWidget()->rect().width() - 100, sizeHint().height());
  move({50, parentWidget()->rect().height() - rect().height() - bdr_s});
}

void ReplayControls::start(const QString &route, const QString &data_dir) {
  QStringList allow = {
      "modelV2",
      "controlsState",
      "liveCalibration",
      "radarState",
      "roadCameraState",
      "roadEncodeIdx",
      "pandaStates",
      "carParams",
      "driverMonitoringState",
      "carState",
      "liveLocationKalman",
      "wideRoadCameraState",
      "managerState",
      "navInstruction",
      "navRoute",
      "gnssMeasurements",
  };
  QString route_name = "0000000000000000|" + route;
  replay.reset(new Replay(route_name, allow, {}, nullptr, REPLAY_FLAG_NONE, data_dir));
  if (replay->load()) {
    slider->setRange(0, replay->totalSeconds());
    end_time_label->setText(formatTime(replay->totalSeconds()));
    replay->start();
    timer->start();
  }
}

void ReplayControls::stop() {
  timer->stop();
  replay->stop();
  replay.reset(nullptr);
  emit uiState()->stopReplay();
}
