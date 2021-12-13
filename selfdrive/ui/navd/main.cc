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
#include "selfdrive/common/params.h"

void sigHandler(int s) {
  qInfo() << "Shutting down";
  std::signal(s, SIG_DFL);

  qApp->quit();
}


int main(int argc, char *argv[]) {
  qInstallMessageHandler(uiUtil::swagLogMessageHandler);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  QCommandLineParser parser;
  parser.setApplicationDescription("Navigation server. Runs stand-alone, or using pre-computer route");
  parser.addHelpOption();
  parser.process(app);
  const QStringList args = parser.positionalArguments();


  RouteEngine* route_engine = new RouteEngine();

  if (Params().getBool("NavdRender")) {
    MapRenderer * m = new MapRenderer(get_mapbox_settings());
    QObject::connect(route_engine, &RouteEngine::positionUpdated, m, &MapRenderer::updatePosition);
    QObject::connect(route_engine, &RouteEngine::routeUpdated, m, &MapRenderer::updateRoute);
  }

  return app.exec();
}
