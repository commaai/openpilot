#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include <capnp/message.h>

#include "common/tests/native_test.h"
#include "openpilot/cereal/gen/cpp/log.capnp.h"
#include "tools/cabana/core/logsignals.h"

namespace {

void test_numeric_fields() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  auto state = event.initCarState();
  state.setVEgo(12.5);
  state.setSteeringPressed(true);
  state.setGearShifter(cereal::CarState::GearShifter::DRIVE);
  state.initWheelSpeeds().setFl(13.0);
  cabana::LogSignals signals;
  cabana::appendLogSignal(capnp::toDynamic(state.asReader()), "carState", 42, signals);
  REQUIRE(signals.at("carState/vEgo")[0].value == 12.5);
  REQUIRE(signals.at("carState/vEgo")[0].mono_time == 42);
  REQUIRE(signals.at("carState/steeringPressed")[0].value == 1);
  REQUIRE(signals.at("carState/gasPressed")[0].value == 0);
  REQUIRE(signals.at("carState/gearShifter")[0].value == static_cast<int>(cereal::CarState::GearShifter::DRIVE));
  REQUIRE(signals.at("carState/wheelSpeeds/fl")[0].value == 13.0);
}

void test_lists_and_active_union() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  auto model = event.initModelV2();
  auto position = model.initPosition();
  auto xs = position.initX(3);
  xs.set(0, 1); xs.set(1, 2); xs.set(2, 3);
  cabana::LogSignals signals;
  cabana::appendLogSignal(capnp::toDynamic(event.asReader()), "event", 100, signals);
  REQUIRE(signals.at("event/modelV2/position/x/0")[0].value == 1);
  REQUIRE(signals.at("event/modelV2/position/x/2")[0].value == 3);
  REQUIRE(!signals.count("event/modelV2/position/x/3"));
  REQUIRE(!signals.count("event/carState/vEgo"));
  REQUIRE(!signals.count("event/modelV2/velocity/x/0"));
  // Switching the event union must not read the previous member.
  signals.clear();
  event.initCarState().setVEgo(2);
  cabana::appendLogSignal(capnp::toDynamic(event.asReader()), "event", 101, signals);
  REQUIRE(signals.at("event/carState/vEgo")[0].value == 2);
  REQUIRE(!signals.count("event/modelV2/position/x/0"));
  signals.clear();
  event.initCarParams().initLateralTuning().initTorque().setFriction(0.1);
  cabana::appendLogSignal(capnp::toDynamic(event.asReader()), "event", 102, signals);
  REQUIRE(std::abs(signals.at("event/carParams/lateralTuning/torque/friction")[0].value - 0.1) < 1e-6);
  REQUIRE(!signals.count("event/carParams/lateralTuning/pid/kpBP/0"));
}

void test_missing_and_non_numeric_fields() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  auto params = event.initCarParams();
  params.setCarFingerprint("TEST");
  cabana::LogSignals signals;
  cabana::appendLogSignal(capnp::toDynamic(params.asReader()), "carParams", 1, signals);
  REQUIRE(!signals.count("carParams/carFingerprint"));
  REQUIRE(!signals.count("carParams/carFw/0/ecu"));
}

void test_nonfinite_samples() {
  capnp::MallocMessageBuilder message;
  auto state = message.initRoot<cereal::Event>().initCarState();
  state.setVEgo(std::numeric_limits<float>::infinity());
  state.setAEgo(std::numeric_limits<float>::quiet_NaN());
  cabana::LogSignals signals;
  cabana::appendLogSignal(capnp::toDynamic(state.asReader()), "carState", 7, signals);
  REQUIRE(std::isnan(signals.at("carState/vEgo")[0].value));
  REQUIRE(std::isnan(signals.at("carState/aEgo")[0].value));
}

cabana::LogSegments sample_segments() {
  auto a = std::make_shared<cabana::LogSignals>();
  auto b = std::make_shared<cabana::LogSignals>();
  (*a)["carState/vEgo"] = {{2000000000, 4}, {1000000000, 2}};
  (*b)["carState/vEgo"] = {{2000000000, 5}, {3000000000, 8}};
  return {{0, a}, {1, b}};
}

void test_ordering_duplicates_and_timestamps() {
  auto points = cabana::logPoints(sample_segments(), {"carState/vEgo"}, 2000000000);
  REQUIRE(points.size() == 3);
  REQUIRE(points[0].x == -1 && points[0].y == 2);
  REQUIRE(points[1].x == 0 && points[1].y == 5);
  REQUIRE(points[2].x == 1 && points[2].y == 8);
  REQUIRE(cabana::logPoints(sample_segments(), {"missing"}, 0).empty());
  auto high = std::make_shared<cabana::LogSignals>();
  const uint64_t origin = 9007199254740992ULL;
  (*high)["x"] = {{origin - 1, 1}, {origin, 2}, {origin + 1, 3}};
  auto tiny = cabana::logPoints({{0, high}}, {"x"}, origin);
  REQUIRE(tiny[0].x == -1e-9 && tiny[2].x == 1e-9);
}

void test_transforms_and_eviction() {
  auto segments = sample_segments();
  auto points = cabana::logPoints(segments, {"carState/vEgo", 3.6, 1, true}, 0);
  REQUIRE(std::isnan(points[0].y));
  REQUIRE(std::abs(points[1].y - 10.8) < 1e-10);
  REQUIRE(std::abs(points[2].y - 10.8) < 1e-10);
  segments.erase(0);
  points = cabana::logPoints(segments, {"carState/vEgo"}, 0);
  REQUIRE(points.size() == 2 && points.front().x == 2);
  auto gap = std::make_shared<cabana::LogSignals>();
  (*gap)["x"] = {{0, 1}, {1000000000, std::numeric_limits<double>::quiet_NaN()}, {2000000000, 3}, {3000000000, 5}};
  points = cabana::logPoints({{0, gap}}, {"x", 1, 0, true}, 0);
  REQUIRE(std::isnan(points[1].y) && std::isnan(points[2].y) && points[3].y == 2);
  (*gap)["x"] = {{0, 1}};
  auto later = std::make_shared<cabana::LogSignals>();
  (*later)["x"] = {{180000000000, 3}, {181000000000, 4}};
  points = cabana::logPoints({{0, gap}, {3, later}}, {"x", 1, 0, true}, 0);
  REQUIRE(points.size() == 4);
  REQUIRE(std::isnan(points[1].y) && std::isnan(points[2].y) && points[3].y == 1);
}

void test_layout_roundtrip_and_validation() {
  std::vector<cabana::LogPlot> plots = {{{"carState/vEgo", 3.6, -1, true}, {"carState/steeringPressed"}}, {{"modelV2/position/x/0"}}};
  const auto text = cabana::serializeLogLayout(plots);
  const auto parsed = cabana::parseLogLayout(text);
  REQUIRE(parsed.size() == 2 && parsed[0].size() == 2);
  REQUIRE(parsed[0][0].scale == 3.6 && parsed[0][0].offset == -1 && parsed[0][0].derivative);
  REQUIRE(cabana::serializeLogLayout(parsed) == text);
  auto defaults = cabana::parseLogLayout(R"({"version":1,"plots":[[{"signal":"x"}]]})");
  REQUIRE(defaults[0][0].scale == 1 && defaults[0][0].offset == 0 && !defaults[0][0].derivative);
  for (const char *bad : {"garbage", "[]", R"({"version":2,"plots":[]})", R"({"version":1,"plots":[[]]})",
       R"({"version":1,"plots":[[{"signal":""}]]})", R"({"version":1,"plots":[[{"signal":"x","scale":"2"}]]})",
       R"({"version":1,"plots":[[{"signal":"x","derivative":1}]]})", R"({"version":1,"plots":[[{"signal":"x","offset":null}]]})"}) {
    bool rejected = false;
    try { cabana::parseLogLayout(bad); } catch (const std::runtime_error &) { rejected = true; }
    REQUIRE(rejected);
  }
}

void test_csv_range_and_escaping() {
  auto csv = cabana::logCsv(sample_segments(), {{{"carState/vEgo"}}}, 0, 1.5, 2.5);
  REQUIRE(csv == "plot,signal,time,value,scale,offset,derivative\n1,\"carState/vEgo\",2,5,1,0,0\n");
  auto signals = std::make_shared<cabana::LogSignals>();
  (*signals)["a,\"b"] = {{0, std::numeric_limits<double>::quiet_NaN()}};
  csv = cabana::logCsv({{0, signals}}, {{{"a,\"b"}}}, 0, 0, 0);
  REQUIRE(csv.find("1,\"a,\"\"b\",0,,1,0,0\n") != std::string::npos);
}

}  // namespace

int main() {
  return run_native_test([] {
    test_numeric_fields();
    test_lists_and_active_union();
    test_missing_and_non_numeric_fields();
    test_nonfinite_samples();
    test_ordering_duplicates_and_timestamps();
    test_transforms_and_eviction();
    test_layout_roundtrip_and_validation();
    test_csv_range_and_escaping();
    std::cout << "8 log signal test groups passed\n";
  });
}
