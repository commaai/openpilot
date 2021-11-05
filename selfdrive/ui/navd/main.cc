#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>
#include <csignal>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/navd/route_engine.h"
#include "selfdrive/ui/navd/simple_map.h"
#include "selfdrive/hardware/hw.h"

const bool DRAW_MAP = true;


void sigHandler(int s) {
  qInfo() << "Shutting down";
  std::signal(s, SIG_DFL);

  qApp->quit();
}


int main(int argc, char *argv[]) {
  qInstallMessageHandler(swagLogMessageHandler);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  QCommandLineParser parser;
  parser.setApplicationDescription("Navigation server. Runs stand-alone, or using pre-computer route");
  parser.addHelpOption();
  parser.process(app);
  const QStringList args = parser.positionalArguments();


  RouteEngine* route_engine = new RouteEngine();

  if (DRAW_MAP) {
    QMapboxGLSettings settings;

    // TODO: Check if the cache is safe to access from two processes
    // if (!Hardware::PC()) {
    //   settings.setCacheDatabasePath("/data/mbgl-cache.db");
    // }
    settings.setApiBaseUrl(MAPS_HOST);
    settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
    settings.setAccessToken(get_mapbox_token());

    SimpleMap * m = new SimpleMap(settings);
    m->setFixedWidth(640);
    m->setFixedHeight(640);

    QObject::connect(route_engine, &RouteEngine::positionUpdated, m, &SimpleMap::updatePosition);

    m->show();
  }

  return app.exec();
}
