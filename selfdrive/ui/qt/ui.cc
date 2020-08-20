#include <QApplication>

#include "window.hpp"

int main(int argc, char *argv[])
{
  QSurfaceFormat fmt;
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication a(argc, argv);

  MainWindow w;
  w.setFixedSize(vwp_w, vwp_h);
  w.show();

  return a.exec();
}
