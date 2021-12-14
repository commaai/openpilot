#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"
#include <QCoreApplication>

int main(int argc, char **argv) {
  // unit tests for Qt
  QCoreApplication app(argc, argv);
  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}
