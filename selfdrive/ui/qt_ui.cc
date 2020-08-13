#include <QApplication>

#include "qt_window.hpp"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  QSurfaceFormat::setDefaultFormat(format);

  MainWindow w;
  w.setFixedSize(2160, 1080);
  w.show();

  return a.exec();
}
