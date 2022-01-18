#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <random>

#include "catch2/catch.hpp"
#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/boardd.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"

void write_car_params(cereal::CarParams::SafetyModel safety_model, int16_t safety_param) {
  MessageBuilder msg;
  auto car_params = msg.initEvent().initCarParams();
  auto safety_config = car_params.initSafetyConfigs(1);
  safety_config[0].setSafetyModel(safety_model);
  safety_config[0].setSafetyParam(safety_param);

  char car_vin[17] = {'\0'};
  auto bytes = msg.toBytes();

  Params params;
  params.put("CarVin", car_vin, std::size(car_vin));
  params.putBool("ControlsReady", true);
  params.put("CarParams", (const char *)bytes.begin(), bytes.size());
}

TEST_CASE("safety setter thread") {
  Panda *panda = nullptr;
  while (!(panda = usb_connect())) {
    util::sleep_for(500);
  }

  std::vector<Panda *> pandas{panda};
  auto thread = std::thread(safety_setter_thread, pandas);

  int16_t safety_param = 1;
  auto safety_model = cereal::CarParams::SafetyModel::ALL_OUTPUT;
  write_car_params(safety_model, safety_param);

  thread.join();

  health_t state = panda->get_state();
  REQUIRE(state.safety_model == (uint8_t)safety_model);
  REQUIRE(state.safety_param == safety_param);

  delete panda;
}
