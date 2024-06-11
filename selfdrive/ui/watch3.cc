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
  cmd_parser.addOption({"cams", "cameras to decode e.g. 0 or 0,1 or 0,1,2"});
  cmd_parser.addOption({"nav", "load nav"});
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"ecam", "load wide road camera"});
  cmd_parser.addOption({"dcam", "load driver camera"});
  cmd_parser.process(app);
  const QStringList args = cmd_parser.positionalArguments();

  if (cmd_parser.isSet("cams")) {
    QStringList indexes = QString(cmd_parser.value("cams")).split(',');
    for (const QString &index : indexes) {
      VisionStreamType streamType = static_cast<VisionStreamType>(index.toInt());
      layout->addWidget(new CameraWidget("camerad", streamType, false));
    }
  } else if (!args.empty()) {
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
