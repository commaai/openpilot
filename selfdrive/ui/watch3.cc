#include <QApplication>
#include <QtWidgets>
#include <QCommandLineParser>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"

int main(int argc, char *argv[]) {
  initApp(argc, argv);

  QApplication app(argc, argv);
  QWidget w;
  setMainWindow(&w);

  QVBoxLayout *layout = new QVBoxLayout(&w);
  layout->setMargin(0);
  layout->setSpacing(0);

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  QCommandLineOption cams_option = QCommandLineOption("cams", "cameras to decode, separated by a comma", "0,1,2");
  cmd_parser.addOption(cams_option);
  cmd_parser.addOption({"nav", "decode nav"});
  cmd_parser.addOption({"qcam", "decode qcamera"});
  cmd_parser.addOption({"ecam", "decode wide road camera"});
  cmd_parser.addOption({"dcam", "decode driver camera"});
  cmd_parser.process(app);

  if (cmd_parser.isSet("cams")) {
    QStringList indexes = cmd_parser.value(cams_option).split(',');
    for (QString &index : indexes) {
      VisionStreamType streamType = static_cast<VisionStreamType>(index.toInt());
      layout->addWidget(new CameraWidget("camerad", streamType, false));
    }
  } else if (cmd_parser.optionNames().count() > 0) {
    if (cmd_parser.isSet("nav")) {
      layout->addWidget(new CameraWidget("navd", VISION_STREAM_MAP, false));
    }
    if (cmd_parser.isSet("qcam")) {
      layout->addWidget(new CameraWidget("camerad", VISION_STREAM_ROAD, false));
    }
    if (cmd_parser.isSet("dcam")) {
      layout->addWidget(new CameraWidget("camerad", VISION_STREAM_DRIVER, false));
    }
    if (cmd_parser.isSet("ecam")) {
      layout->addWidget(new CameraWidget("camerad", VISION_STREAM_WIDE_ROAD, false));
    }
  } else {
    {
      QHBoxLayout *hlayout = new QHBoxLayout();
      layout->addLayout(hlayout);
      hlayout->addWidget(new CameraWidget("navd", VISION_STREAM_MAP, false));
      hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_ROAD, false));
    }

    {
      QHBoxLayout *hlayout = new QHBoxLayout();
      layout->addLayout(hlayout);
      hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_DRIVER, false));
      hlayout->addWidget(new CameraWidget("camerad", VISION_STREAM_WIDE_ROAD, false));
    }
  }

  return app.exec();
}
