#include "selfdrive/ui/qt/offroad/replaypanel.h"

#include <QDir>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

ReplayPanel::ReplayPanel(QWidget *parent) : ListWidget(parent) {
}

void ReplayPanel::showEvent(QShowEvent *event) {
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
      ButtonControl *c = new ButtonControl(route, "replay");
      QObject::connect(c, &ButtonControl::clicked, [this, r = route]() {
        this->replayRoute(r);
      });
      addItem(c);
    }
  }
  ListWidget::showEvent(event);
}

void ReplayPanel::replayRoute(const QString &route) {
  QString route_name = "0000000000000000|" + route;
  replay.reset(new Replay(route_name, {}, {}, nullptr, REPLAY_FLAG_NONE,
                          QString::fromStdString(Path::log_root())));
  if (replay->load()) {
    replay->start();
    uiState()->replaying = true;
    emit uiState()->replayStarted();
  }
}

void ReplayPanel::stopReplay() {
  replay.reset(nullptr);
  uiState()->replaying = false;
  emit uiState()->replayStopped();
}
