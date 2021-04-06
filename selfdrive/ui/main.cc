#include <QApplication>

#include "qt/window.hpp"
#include "qt/qt_window.hpp"

#include <QThread>

#include "FrameReader.hpp"
#include "replay/replay.hpp"
#include "replay/Unlogger.hpp"

int main(int argc, char *argv[]) {
  QSurfaceFormat fmt;
#ifdef __APPLE__
  fmt.setVersion(3, 2);
  fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  fmt.setRenderableType(QSurfaceFormat::OpenGL);
#else
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
#endif
  QSurfaceFormat::setDefaultFormat(fmt);

#ifdef QCOM
  QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#endif

  QApplication a(argc, argv);
  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);

// ========================
  QString route(argv[1]);
  route = route.replace("|", "/");
  if (route == "") {
    printf("usage %s: <route>\n", argv[0]);
    exit(0);
  }

  int seek = QString(argv[2]).toInt();
  int use_api = QString::compare(QString("use_api"), route, Qt::CaseInsensitive) == 0;

	Replay *replay = new Replay(route, seek, use_api);
	replay->use_api = replay->use_api;
// ========================

  return a.exec();
}
