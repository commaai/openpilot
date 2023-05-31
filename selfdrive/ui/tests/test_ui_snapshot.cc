#include "selfdrive/ui/tests/test_ui_snapshot.h"

#include <QApplication>
#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QTranslator>

#include "selfdrive/ui/qt/home.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

void saveWidgetAsImage(QWidget *widget, const QString &fileName) {
  QImage image(widget->size(), QImage::Format_ARGB32);
  QPainter painter(&image);
  widget->render(&painter);
  image.save(fileName);
}

int main(int argc, char *argv[]) {
  /** SETUP **/
  initApp(argc, argv);

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(translation_file, "translations") && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  QApplication a(argc, argv);
  a.installTranslator(&translator);

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);

  /** TEST CASES **/
  std::vector<TestCase> testCases = {
    { [&]() { uiState()->setPrimeType(0); }, "no_prime" },
    { [&]() { uiState()->setPrimeType(1); }, "with_prime" },
  };

  for (const auto& testCase : testCases) {
    testCase.setupFunc();
    saveWidgetAsImage(&w, ("test_snapshot_" + testCase.name + ".png").c_str());
  }

  return 0;
}
