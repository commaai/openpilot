#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <capnp/message.h>

#include "common/tests/native_test.h"
#include "tools/cabana/streams/cerealseries.h"

void test_cereal_series() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  auto state = event.initCarState();
  state.setVEgo(12.5f);
  state.setStandstill(false);
  state.setGearShifter(cereal::CarState::GearShifter::DRIVE);
  state.initCruiseState().setSpeed(25.0f);
  auto buttons = state.initButtonEvents(2);
  buttons[0].setPressed(true);
  buttons[1].setPressed(false);

  auto reader = event.asReader();
  auto paths = cabana::cerealSignalPaths(reader);
  REQUIRE(std::find(paths.begin(), paths.end(), "carState/vEgo") != paths.end());
  REQUIRE(std::find(paths.begin(), paths.end(), "carState/buttonEvents/1/pressed") != paths.end());
  REQUIRE(std::none_of(paths.begin(), paths.end(), [](const auto &path) { return path.find("controlsState/") == 0; }));
  REQUIRE(cabana::cerealSignalValue(reader, "carState/vEgo") == 12.5);
  REQUIRE(cabana::cerealSignalValue(reader, "carState/standstill") == 0.0);
  REQUIRE(cabana::cerealSignalValue(reader, "carState/cruiseState/speed") == 25.0);
  REQUIRE(cabana::cerealSignalValue(reader, "carState/buttonEvents/0/pressed") == 1.0);
  REQUIRE(cabana::cerealSignalValue(reader, "carState/buttonEvents/1/pressed") == 0.0);
  REQUIRE(cabana::cerealSignalValue(reader, "carState/gearShifter") ==
          static_cast<double>(static_cast<uint16_t>(cereal::CarState::GearShifter::DRIVE)));
  for (const auto *path : {"", "/carState/vEgo", "carState/vEgo/", "carState//vEgo", "carState/missing", "controlsState/vEgo",
                           "carState/buttonEvents/2/pressed", "carState/buttonEvents/-1/pressed",
                           "carState/buttonEvents/0junk/pressed", "carState/vEgo/x"}) {
    REQUIRE(!cabana::cerealSignalValue(reader, path));
  }
  state.setVEgo(std::numeric_limits<float>::quiet_NaN());
  REQUIRE(!cabana::cerealSignalValue(event.asReader(), "carState/vEgo"));
  state.setVEgo(std::numeric_limits<float>::infinity());
  REQUIRE(!cabana::cerealSignalValue(event.asReader(), "carState/vEgo"));

  auto model = event.initModelV2();
  auto x = model.initPosition().initX(3);
  x.set(0, 1.0f);
  x.set(1, 2.0f);
  x.set(2, 3.0f);
  REQUIRE(cabana::cerealSignalValue(event.asReader(), "modelV2/position/x/2") == 3.0);
  REQUIRE(!cabana::cerealSignalValue(event.asReader(), "carState/vEgo"));
  REQUIRE(!cabana::cerealSignalValue(event.asReader(), "modelV2/velocity/x/0"));

  cabana::CerealDiscovery discovery;
  discovery.observe(event.asReader());
  REQUIRE(discovery.names().size() == cabana::cerealSignalPaths(event.asReader()).size());
  model.initVelocity().initX(130).set(129, 7.0f);
  discovery.observe(event.asReader());
  auto names = discovery.names();
  REQUIRE(std::find(names.begin(), names.end(), "modelV2/velocity/x/129") != names.end());
  REQUIRE(cabana::cerealSignalValue(event.asReader(), "modelV2/velocity/x/129") == 7.0);
  event.setValid(false);
  event.setLogMonoTime(123000000000ULL);
  REQUIRE(cabana::cerealSignalValue(event.asReader(), "modelV2/__valid") == 0.0);
  REQUIRE(std::abs(*cabana::cerealSignalValue(event.asReader(), "modelV2/__logMonoTimeSeconds") - 123.0) < 1e-12);
  event.initCarState().setVEgo(3.0f);
  discovery.observe(event.asReader());
  names = discovery.names();
  REQUIRE(std::find(names.begin(), names.end(), "carState/vEgo") != names.end());
  REQUIRE(std::find(names.begin(), names.end(), "modelV2/velocity/x/129") != names.end());
  const auto labels = cabana::cerealEnumLabels("carState/gearShifter");
  REQUIRE(labels.at(static_cast<uint16_t>(cereal::CarState::GearShifter::DRIVE)) == "drive");
  REQUIRE(cabana::cerealEnumLabels("carState/vEgo").empty());
  REQUIRE(cabana::cerealEnumLabels("modelV2/position/x/nope").empty());
}

void test_transforms() {
  using cabana::CerealPoints;
  const CerealPoints points{{1000000000, 1}, {2000000000, 4}, {4000000000, 10}};
  REQUIRE(!cabana::heldValue(points, 999999999));
  REQUIRE(cabana::heldValue(points, 2000000000) == 4);
  REQUIRE(cabana::heldValue(points, 3999999999) == 4);
  REQUIRE(cabana::heldValue(points, 9000000000) == 10);
  REQUIRE(cabana::heldValue({{1, 2}, {1, 3}}, 1) == 3);
  cabana::SeriesTransform transform;
  transform.scale = 2;
  transform.offset = -1;
  auto output = cabana::transformSeries(points, transform);
  REQUIRE(output.size() == 3 && output[2].value == 19);
  transform.derivative = true;
  output = cabana::transformSeries(points, transform);
  REQUIRE(output.size() == 2 && output[0].value == 5 && output[1].value == 5);
  transform.derivative_dt = 0.5;
  output = cabana::transformSeries(points, transform);
  REQUIRE(output[0].value == 11 && output[1].value == 23);
  transform = {};
  transform.derivative = true;
  REQUIRE(cabana::transformSeries({{1, 1}, {1, 2}}, transform).empty());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  output = cabana::transformSeries({{1, 1}, {2, nan}, {3, 4}, {4, 5}}, transform);
  REQUIRE(output.size() == 3 && std::isnan(output[0].value) && std::isnan(output[1].value));
  REQUIRE(std::isfinite(output[2].value));
  bool rejected = false;
  try { cabana::transformSeries({{2, 1}, {1, 2}}, {}); } catch (const std::invalid_argument &) { rejected = true; }
  REQUIRE(rejected);
  rejected = false;
  transform.derivative_dt = -1;
  try { cabana::transformSeries(points, transform); } catch (const std::invalid_argument &) { rejected = true; }
  REQUIRE(rejected);
}

int main() {
  return run_native_test([]() { test_cereal_series(); test_transforms(); std::cout << "Cereal tests passed\n"; });
}
