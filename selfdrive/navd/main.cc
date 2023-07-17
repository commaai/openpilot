#include <QApplication>
#include <QDebug>
#include <csignal>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/navd/map_renderer.h"
#include "system/hardware/hw.h"

int main(int argc, char *argv[]) {
  qInstallMessageHandler(swagLogMessageHandler);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  MapRenderer m(get_mapbox_settings());
  return app.exec();
}
