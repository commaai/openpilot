#include <capnp/message.h>

#include "common/tests/native_test.h"
#include "tools/cabana/streams/cereal_extractor.h"

namespace {

void test_scalars_and_defaults() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  event.setLogMonoTime(9000000000000000001ULL);
  auto state = event.initCarState();
  state.setVEgo(12.5);
  state.setSteeringPressed(false);
  cabana::CerealSeriesMap series;
  cabana::extractCerealEvent(event.asReader(), series);
  const auto &velocity = series.at("/carState/vEgo").samples;
  REQUIRE(velocity.size() == 1);
  REQUIRE(velocity[0].value == 12.5);
  REQUIRE(velocity[0].mono_time == 9000000000000000001ULL);
  REQUIRE(series.at("/carState/steeringPressed").samples[0].value == 0.0);
  REQUIRE(series.at("/carState/aEgo").samples[0].value == 0.0);
  REQUIRE(series.count("/logMonoTime") == 0);
}

void test_lists_and_union() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  auto control = event.initCarControl();
  auto orientation = control.initOrientationNED(3);
  orientation.set(0, -1.5);
  orientation.set(1, 0.0);
  orientation.set(2, 2.5);
  cabana::CerealSeriesMap series;
  cabana::extractCerealEvent(event.asReader(), series);
  REQUIRE(series.at("/carControl/orientationNED/0").samples[0].value == -1.5);
  REQUIRE(series.at("/carControl/orientationNED/1").samples[0].value == 0.0);
  REQUIRE(series.at("/carControl/orientationNED/2").samples[0].value == 2.5);
  REQUIRE(series.count("/carControl/orientationNED/3") == 0);

  series.clear();
  event.initControlsState().initLateralControlState().initPidState().setP(1.25);
  cabana::extractCerealEvent(event.asReader(), series);
  REQUIRE(series.at("/controlsState/lateralControlState/pidState/p").samples[0].value == 1.25);
  REQUIRE(series.count("/controlsState/lateralControlState/torqueState/p") == 0);
  REQUIRE(series.count("/carControl/orientationNED/0") == 0);
}

void test_enums_and_can_exclusion() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  event.initCarState();
  cabana::CerealSeriesMap series;
  cabana::extractCerealEvent(event.asReader(), series);
  const auto &gear = series.at("/carState/gearShifter");
  REQUIRE(gear.samples.size() == 1);
  REQUIRE(!gear.enum_names.empty());
  REQUIRE(gear.enum_names.count(static_cast<uint16_t>(gear.samples[0].value)) == 1);

  series.clear();
  event.initCan(1)[0].setAddress(123);
  cabana::extractCerealEvent(event.asReader(), series);
  REQUIRE(series.empty());
  event.initSendcan(1)[0].setAddress(123);
  cabana::extractCerealEvent(event.asReader(), series);
  REQUIRE(series.empty());
  event.setLogMessage("text is handled by the log viewer");
  cabana::extractCerealEvent(event.asReader(), series);
  REQUIRE(series.empty());
}

void test_segment_snapshots() {
  cabana::CerealSeriesStore store;
  cabana::CerealSeriesMap series;
  series["/test/value"].samples = {{30, 3}, {10, 1}, {10, 2}};
  store.replaceSegment(2, std::move(series));
  auto first = store.snapshot();
  const auto &samples = first.segments.at(2)->at("/test/value").samples;
  REQUIRE(samples.size() == 3);
  REQUIRE(samples[0].mono_time == 10);
  REQUIRE(samples[0].value == 1);
  REQUIRE(samples[1].value == 2);
  REQUIRE(samples[2].mono_time == 30);

  cabana::CerealSeriesMap replacement;
  replacement["/test/value"].samples = {{40, 4}};
  store.replaceSegment(2, std::move(replacement));
  auto second = store.snapshot();
  REQUIRE(second.revision > first.revision);
  REQUIRE(second.segments.size() == 1);
  REQUIRE(second.segments.at(2)->at("/test/value").samples.size() == 1);
  REQUIRE(samples.size() == 3);  // earlier snapshot remains valid
  store.retainSegments({2});
  REQUIRE(store.snapshot().revision == second.revision);
  store.retainSegments({3});
  REQUIRE(store.snapshot().segments.empty());
  REQUIRE(second.segments.at(2)->at("/test/value").samples[0].value == 4);
  store.clear();
  REQUIRE(store.snapshot().segments.empty());
}

}  // namespace

int main() {
  return run_native_test([]() {
    test_scalars_and_defaults();
    test_lists_and_union();
    test_enums_and_can_exclusion();
    test_segment_snapshots();
  });
}
