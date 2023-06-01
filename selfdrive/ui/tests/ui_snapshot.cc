#include "selfdrive/ui/tests/ui_snapshot.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QTranslator>

#include "selfdrive/ui/qt/home.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"
#include "selfdrive/ui/ui.h"

void saveWidgetAsImage(QWidget *widget, const QString &fileName) {
  QImage image(widget->size(), QImage::Format_ARGB32);
  QPainter painter(&image);
  widget->render(&painter);
  image.save(fileName);
}

int main(int argc, char *argv[]) {
  initApp(argc, argv);

  QApplication app(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription("UI snapshot tool");
  parser.addHelpOption();
  parser.addPositionalArgument("file", "output file", "[file]", "snapshot.png");
  parser.process(app);

  const QString output = parser.positionalArguments().value(0);
  if (output.isEmpty()) {
    qCritical() << "No output file specified";
    return 1;
  }

  MainWindow w;
  w.setFixedSize(2160, 1080);
  w.show();
  app.installEventFilter(&w);

  // wait for the UI to update
  QTimer::singleShot(UI_FREQ, [&] {
    saveWidgetAsImage(&w, output);
    QTimer::singleShot(0, &app, &QApplication::quit);
  });

  return app.exec();
}
