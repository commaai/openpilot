#include <QApplication>

#include "replay.hpp"

int main(int argc, char *argv[]){

/*
  QSurfaceFormat fmt;
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
  QSurfaceFormat::setDefaultFormat(fmt);
*/

  QString route(argv[1]);
  route = route.replace("|", "/");
  if (route == "") {
  }

	int seek = QString(argv[2]).toInt();
	int use_api = QString::compare(QString("use_api"), route, Qt::CaseInsensitive) == 0;

  QApplication a(argc, argv);
	Replay *replay = new Replay(route, seek, use_api);

	replay->show();
  return a.exec();
}
