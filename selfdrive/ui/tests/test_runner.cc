#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QTranslator>

int main(int argc, char **argv) {
  // unit tests for Qt
  QApplication app(argc, argv);

  QString language_file = "main_test_en";
  qDebug() << "Loading language:" << language_file;

  QTranslator translator;
  QString translationsPath = QDir::cleanPath(qApp->applicationDirPath() + "/../translations");
  if (!translator.load(language_file, translationsPath)) {
    qDebug() << "Failed to load translation file!";
  }
  app.installTranslator(&translator);

  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}
