#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>
#include <csignal>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/navd/route_engine.h"
#include "selfdrive/ui/navd/map_renderer.h"
#include "selfdrive/hardware/hw.h"

const bool DRAW_MAP = getenv("DRAW_MAP") != nullptr;


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
    if (!Hardware::PC()) {
      settings.setCacheDatabasePath("/data/mbgl-cache-navd.db");
    }
    settings.setApiBaseUrl(MAPS_HOST);
    settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
    settings.setAccessToken(get_mapbox_token());

    MapRenderer * m = new MapRenderer(settings);

    QObject::connect(route_engine, &RouteEngine::positionUpdated, m, &MapRenderer::updatePosition);
    QObject::connect(route_engine, &RouteEngine::routeUpdated, m, &MapRenderer::updateRoute);
  }

  return app.exec();
}
