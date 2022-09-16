#define CATCH_CONFIG_RUNNER
#include <QApplication>

#include "catch2/catch.hpp"
#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

int main(int argc, char **argv) {
  // unit tests for Qt
  QApplication app(argc, argv);
  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}

TEST_CASE("offroadAlerts") {
  std::vector<std::string> offroad_keys;
  for (auto key : Params().allKeys()) {
    if (key.find("Offroad_") == 0) {
      offroad_keys.push_back(key);
    }
  }

  OffroadAlert alert;
  REQUIRE(offroad_keys.size() == alert.offroad_alerts.size());
  for (const auto &k : offroad_keys) {
    auto result = std::find_if(alert.offroad_alerts.begin(), alert.offroad_alerts.end(), [=](auto &v) {
        return std::get<0>(v) == k;
    });
    REQUIRE(result != alert.offroad_alerts.end());
  }
}