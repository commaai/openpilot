#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"
#include <QApplication>
#include <QTranslator>
#include <QDebug>

int main(int argc, char **argv) {
  // unit tests for Qt
  QApplication app(argc, argv);

  QString language_file = "main_test_en";
  qDebug() << "Loading language:" << language_file;

  QTranslator translator;
  if (!translator.load(language_file, "translations")) {
    qDebug() << "Failed to load translation file!";
  }
  app.installTranslator(&translator);

  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}
