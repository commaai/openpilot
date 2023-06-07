#include "selfdrive/ui/tests/ui_snapshot.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QImage>
#include <QPainter>

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
  parser.setApplicationDescription("Take a snapshot of the UI.");
  parser.addHelpOption();
  parser.addOption(QCommandLineOption(QStringList() << "o"
                                                    << "output",
                                      "Output image file path. The file's suffix is used to "
                                      "determine the format. Supports PNG and JPEG formats. "
                                      "Defaults to \"snapshot.png\".",
                                      "file", "snapshot.png"));
  parser.process(app);

  const QString output = parser.value("output");
  if (output.isEmpty()) {
    qCritical() << "No output file specified";
    return 1;
  }

  auto current = QDir::current();

  // change working directory to find assets
  if (!QDir::setCurrent(QCoreApplication::applicationDirPath() + QDir::separator() + "..")) {
    qCritical() << "Failed to set current directory";
    return 1;
  }

  MainWindow w;
  w.setFixedSize(2160, 1080);
  w.show();
  app.installEventFilter(&w);

  // restore working directory
  QDir::setCurrent(current.absolutePath());

  // wait for the UI to update
  QObject::connect(uiState(), &UIState::uiUpdate, [&](const UIState &s) {
    saveWidgetAsImage(&w, output);
    app.quit();
  });

  return app.exec();
}
