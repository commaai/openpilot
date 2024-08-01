#include <QApplication>
#include <QtWidgets>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"

int main(int argc, char *argv[]) {
  initApp(argc, argv);

  QApplication a(argc, argv);
  QWidget w;
  setMainWindow(&w);

  QVBoxLayout *layout = new QVBoxLayout(&w);
  layout->setMargin(0);
  layout->setSpacing(0);

  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_ROAD, false));
  }

  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_DRIVER, false));
    hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_WIDE_ROAD, false));
  }

  return a.exec();
}
