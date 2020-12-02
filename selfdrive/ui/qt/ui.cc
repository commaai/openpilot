#include <QApplication>

#include "window.hpp"
#include "qt_window.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
