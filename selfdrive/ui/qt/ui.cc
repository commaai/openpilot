#include <QApplication>

#include "window.hpp"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  MainWindow w;

  w.setFixedSize(vwp_w, vwp_h);
  w.show();

  return a.exec();
}
