#include <QApplication>

#include "home.hpp"
#include "replay.hpp"
#include "qt/qt_window.hpp"

int main(int argc, char *argv[]){
  QSurfaceFormat fmt;
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication a(argc, argv);
	GLWindow *w = new GLWindow();
  setMainWindow((QWidget*)w);
  a.installEventFilter(w);

  QString route(argv[1]);
  route = route.replace("|", "/");
  if (route != "") {
    int seek = QString(argv[2]).toInt();
    int use_api = QString::compare(QString("use_api"), route, Qt::CaseInsensitive) == 0;

    Replay *replay = new Replay(route, seek, use_api);
    replay->replay();
  }

  return a.exec();
}
