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

  auto newCamera = [](std::string stream_name, VisionStreamType stream_type) {
    auto cam = new CameraWidget(stream_name, stream_type, false);
    cam->setAutoUpdate(true);
    return cam;
  };
  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    hlayout->addWidget(newCamera("navd", VISION_STREAM_MAP));
    hlayout->addWidget(newCamera("camerad", VISION_STREAM_ROAD));
  }

  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    hlayout->addWidget(newCamera("camerad", VISION_STREAM_DRIVER));
    hlayout->addWidget(newCamera("camerad", VISION_STREAM_WIDE_ROAD));
  }

  return a.exec();
}
