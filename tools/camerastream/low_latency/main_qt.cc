#include <QApplication>
#include <QtWidgets>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

#include "directcameraview.h"

// __GL_MaxFramesAllowed=1 QT_SINGLE_BUFFER=1 ./low_latency

int main(int argc, char *argv[]) {
  initApp(argc, argv);

  QApplication a(argc, argv);
  QWidget w;
  setMainWindow(&w);

  QVBoxLayout *layout = new QVBoxLayout(&w);
  layout->setMargin(0);
  layout->setSpacing(0);
  QHBoxLayout *hlayout = new QHBoxLayout();
  layout->addLayout(hlayout);
  hlayout->addWidget(new DirectCameraViewWidget("roadEncodeData", "192.168.3.188"));

  return a.exec();
}