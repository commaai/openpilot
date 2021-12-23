#include <QApplication>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/replay/replaywidget.h"

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  ReplayWidget *w = new ReplayWidget;
  setMainWindow(w);
  if (argc > 1) {
    w->replayRoute(argv[1]);
  }
  return a.exec();
}
