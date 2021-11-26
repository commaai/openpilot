#include <QApplication>
#include <QtWidgets>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"

int main(int argc, char *argv[]) {
  QSurfaceFormat fmt;
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication a(argc, argv);
  QWidget w;
  setMainWindow(&w);

  QVBoxLayout *layout = new QVBoxLayout(&w);
  layout->setMargin(0);
  layout->setSpacing(0);

  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    // hlayout->addWidget(new CameraViewWidget("navd", VISION_STREAM_RGB_MAP, false));
    hlayout->addWidget(new CameraViewWidget("camerad", VISION_STREAM_YUV_BACK, false));
  }

  {
    QHBoxLayout *hlayout = new QHBoxLayout();
    layout->addLayout(hlayout);
    hlayout->addWidget(new CameraViewWidget("camerad", VISION_STREAM_YUV_FRONT, false));
    hlayout->addWidget(new CameraViewWidget("camerad", VISION_STREAM_YUV_WIDE, false));
  }

  return a.exec();
}
