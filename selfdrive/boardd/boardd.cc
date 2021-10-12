#include <sched.h>
#include <sys/cdefs.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <unordered_map>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/locationd/ublox_msg.h"

#include "selfdrive/boardd/panda.h"
#include "selfdrive/boardd/pigeon.h"

// -- Multi-panda conventions --
// Ordering:
// - The internal panda will always be the first panda
// - Consecutive pandas will be sorted based on panda type, and then serial number
// Connecting:
// - If a panda connection is dropped, boardd wil reconnect to all pandas
// - If a panda is added, we will only reconnect when we are offroad
// CAN buses:
// - Each panda will have it's block of 4 buses. E.g.: the second panda will use
//   bus numbers 4, 5, 6 and 7
// - The internal panda will always be used for accessing the OBD2 port,
//   and thus firmware queries
// Safety:
// - SafetyConfig is a list, which is mapped to the connected pandas
// - If there are more pandas connected than there are SafetyConfigs,
//   the excess pandas will remain in "silent" ot "noOutput" mode
// Ignition:
// - If any of the ignition sources in any panda is high, ignition is high

#define MAX_IR_POWER 0.5f
#define MIN_IR_POWER 0.0f
#define CUTOFF_IL 200
#define SATURATE_IL 1600
#define NIBBLE_TO_HEX(n) ((n) < 10 ? (n) + '0' : ((n) - 10) + 'a')
using namespace std::chrono_literals;

std::atomic<bool> ignition(false);
std::atomic<bool> pigeon_active(false);

ExitHandler do_exit;

std::string get_time_str(const struct tm &time) {
  char s[30] = {'\0'};
  std::strftime(s, std::size(s), "%Y-%m-%d %H:%M:%S", &time);
  return s;
}

bool check_all_connected(std::vector<Panda *> pandas) {
  for (const auto& panda : pandas) {
    if (!panda->connected) return false;
  }
  return true;
}

bool safety_setter_thread(std::vector<Panda *> pandas) {
  LOGD("Starting safety setter thread");

  // there should be at least one panda connected
  if (pandas.size() == 0) {
    return false;
  }

  pandas[0]->set_safety_model(cereal::CarParams::SafetyModel::ELM327);

  Params p = Params();

  // switch to SILENT when CarVin param is read
  while (true) {
    if (do_exit || !check_all_connected(pandas) || !ignition) {
      return false;
    }

    std::string value_vin = p.get("CarVin");
    if (value_vin.size() > 0) {
      // sanity check VIN format
      assert(value_vin.size() == 17);
      LOGW("got CarVin %s", value_vin.c_str());
      break;
    }
    util::sleep_for(20);
  }

  pandas[0]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1);

  std::string params;
  LOGW("waiting for params to set safety model");
  while (true) {
    for (const auto& panda : pandas) {
      if (do_exit || !panda->connected || !ignition) {
        return false;
      }
    }

    if (p.getBool("ControlsReady")) {
      params = p.get("CarParams");
      if (params.size() > 0) break;
    }
    util::sleep_for(100);
  }
  LOGW("got %d bytes CarParams", params.size());

  AlignedBuffer aligned_buf;
  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(params.data(), params.size()));
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();
  cereal::CarParams::SafetyModel safety_model;
  int safety_param;

  auto safety_configs = car_params.getSafetyConfigs();
  for (uint32_t i=0; i<pandas.size(); i++) {
    auto panda = pandas[i];

    if (safety_configs.size() > i) {
      safety_model = safety_configs[i].getSafetyModel();
      safety_param = safety_configs[i].getSafetyParam();
    } else {
      // If no safety mode is specified, default to silent
      safety_model = cereal::CarParams::SafetyModel::SILENT;
      safety_param = 0;
    }

    LOGW("panda %d: setting safety model: %d with param %d", i, (int)safety_model, safety_param);

    panda->set_unsafe_mode(0);  // see safety_declarations.h for allowed values
    panda->set_safety_model(safety_model, safety_param);
  }

  return true;
}

Panda *usb_connect(std::string serial="", uint32_t index=0) {
  std::unique_ptr<Panda> panda;
  try {
    panda = std::make_unique<Panda>(serial, (index * PANDA_BUS_CNT));
  } catch (std::exception &e) {
    return nullptr;
  }

  Params params = Params();

  if (getenv("BOARDD_LOOPBACK")) {
    panda->set_loopback(true);
  }

  // power on charging, only the first time. Panda can also change mode and it causes a brief disconneciton
#ifndef __x86_64__
  static std::once_flag connected_once;
  std::call_once(connected_once, &Panda::set_usb_power_mode, panda, cereal::PeripheralState::UsbPowerMode::CDP);
#endif

  if (panda->has_rtc) {
    setenv("TZ","UTC",1);
    struct tm sys_time = util::get_time();
    struct tm rtc_time = panda->get_rtc();

    if (!util::time_valid(sys_time) && util::time_valid(rtc_time)) {
      LOGE("System time wrong, setting from RTC. System: %s RTC: %s",
           get_time_str(sys_time).c_str(), get_time_str(rtc_time).c_str());
      const struct timeval tv = {mktime(&rtc_time), 0};
      settimeofday(&tv, 0);
    }
  }

  return panda.release();
}

void can_recv(Panda *panda, PubMaster &pm) {
  kj::Array<capnp::word> can_data;
  panda->can_receive(can_data);
  auto bytes = can_data.asBytes();
  pm.send("can", bytes.begin(), bytes.size());
}

void can_send_thread(std::vector<Panda *> pandas, bool fake_send) {
  LOGD("start send thread");

  AlignedBuffer aligned_buf;
  Context * context = Context::create();
  SubSocket * subscriber = SubSocket::create(context, "sendcan");
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  // run as fast as messages come in
  while (!do_exit) {
    if (!check_all_connected(pandas)) {
      do_exit = true;
      break;
    }

    Message * msg = subscriber->receive();

    if (!msg) {
      if (errno == EINTR) {
        do_exit = true;
      }
      continue;
    }

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    //Dont send if older than 1 second
    if (nanos_since_boot() - event.getLogMonoTime() < 1e9) {
      if (!fake_send) {
        for (const auto& panda : pandas) {
          panda->can_send(event.getSendcan());
        }
      }
    }

    delete msg;
  }

  delete subscriber;
  delete context;
}

void can_recv_thread(std::vector<Panda *> pandas) {
  LOGD("start recv thread");

  // can = 8006
  PubMaster pm({"can"});

  // run at 100hz
  const uint64_t dt = 10000000ULL;
  uint64_t next_frame_time = nanos_since_boot() + dt;

  while (!do_exit) {
    if (!check_all_connected(pandas)){
      do_exit = true;
      break;
    }

    for (const auto& panda : pandas) {
      can_recv(panda, pm);
    }

    uint64_t cur_time = nanos_since_boot();
    int64_t remaining = next_frame_time - cur_time;
    if (remaining > 0) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(remaining));
    } else {
      if (ignition) {
        LOGW("missed cycles (%d) %lld", (int)-1*remaining/dt, remaining);
      }
      next_frame_time = cur_time;
    }

    next_frame_time += dt;
  }
}

void send_empty_peripheral_state(PubMaster *pm) {
  MessageBuilder msg;
  auto peripheralState  = msg.initEvent().initPeripheralState();
  peripheralState.setPandaType(cereal::PandaState::PandaType::UNKNOWN);
  pm->send("peripheralState", msg);
}

void send_empty_panda_state(PubMaster *pm) {
  MessageBuilder msg;
  auto pandaStates = msg.initEvent().initPandaStates(1);
  pandaStates[0].setPandaType(cereal::PandaState::PandaType::UNKNOWN);
  pm->send("pandaStates", msg);
}

bool send_panda_states(PubMaster *pm, std::vector<Panda *> pandas, bool spoofing_started) {
  bool ignition_local = false;

  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto pss = evt.initPandaStates(pandas.size());

  std::vector<health_t> pandaStates;
  for (const auto& panda : pandas){
    health_t pandaState = panda->get_state();

    if (spoofing_started) {
      pandaState.ignition_line = 1;
    }

    ignition_local |= ((pandaState.ignition_line != 0) || (pandaState.ignition_can != 0));

    pandaStates.push_back(pandaState);
  }

  for (uint32_t i=0; i<pandas.size(); i++) {
    auto panda = pandas[i];
    auto pandaState = pandaStates[i];

    // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
    if (pandaState.safety_model == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }

  #ifndef __x86_64__
    bool power_save_desired = !ignition_local && !pigeon_active;
    if (pandaState.power_save_enabled != power_save_desired) {
      panda->set_power_saving(power_save_desired);
    }

    // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
    if (!ignition_local && (pandaState.safety_model != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
  #endif

    // TODO: do we still need this?
    if (!panda->comms_healthy) {
      evt.setValid(false);
    }

    auto ps = pss[i];
    ps.setUptime(pandaState.uptime);
    ps.setIgnitionLine(pandaState.ignition_line);
    ps.setIgnitionCan(pandaState.ignition_can);
    ps.setControlsAllowed(pandaState.controls_allowed);
    ps.setGasInterceptorDetected(pandaState.gas_interceptor_detected);
    ps.setCanRxErrs(pandaState.can_rx_errs);
    ps.setCanSendErrs(pandaState.can_send_errs);
    ps.setCanFwdErrs(pandaState.can_fwd_errs);
    ps.setGmlanSendErrs(pandaState.gmlan_send_errs);
    ps.setPandaType(panda->hw_type);
    ps.setSafetyModel(cereal::CarParams::SafetyModel(pandaState.safety_model));
    ps.setSafetyParam(pandaState.safety_param);
    ps.setFaultStatus(cereal::PandaState::FaultStatus(pandaState.fault_status));
    ps.setPowerSaveEnabled((bool)(pandaState.power_save_enabled));
    ps.setHeartbeatLost((bool)(pandaState.heartbeat_lost));
    ps.setHarnessStatus(cereal::PandaState::HarnessStatus(pandaState.car_harness_status));

    // Convert faults bitset to capnp list
    std::bitset<sizeof(pandaState.faults) * 8> fault_bits(pandaState.faults);
    auto faults = ps.initFaults(fault_bits.count());

    size_t j = 0;
    for (size_t f = size_t(cereal::PandaState::FaultType::RELAY_MALFUNCTION);
        f <= size_t(cereal::PandaState::FaultType::INTERRUPT_RATE_TICK); f++) {
      if (fault_bits.test(f)) {
        faults.set(j, cereal::PandaState::FaultType(f));
        j++;
      }
    }
  }

  pm->send("pandaStates", msg);
  return ignition_local;
}

void send_peripheral_state(PubMaster *pm, Panda *panda) {
  health_t pandaState = panda->get_state();

  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  evt.setValid(panda->comms_healthy);

  auto ps = evt.initPeripheralState();
  ps.setPandaType(panda->hw_type);

  if (Hardware::TICI()) {
    double read_time = millis_since_boot();
    ps.setVoltage(std::atoi(util::read_file("/sys/class/hwmon/hwmon1/in1_input").c_str()));
    ps.setCurrent(std::atoi(util::read_file("/sys/class/hwmon/hwmon1/curr1_input").c_str()));
    read_time = millis_since_boot() - read_time;
    if (read_time > 50) {
      LOGW("reading hwmon took %lfms", read_time);
    }
  } else {
    ps.setVoltage(pandaState.voltage);
    ps.setCurrent(pandaState.current);
  }

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setUsbPowerMode(cereal::PeripheralState::UsbPowerMode(pandaState.usb_power_mode));
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void panda_state_thread(PubMaster *pm, std::vector<Panda *> pandas, bool spoofing_started) {
  Params params;
  Panda *peripheral_panda = pandas[0];
  bool ignition_last = false;
  std::future<bool> safety_future;

  LOGD("start panda state thread");

  // run at 2hz
  while (!do_exit) {
    if(!check_all_connected(pandas)) {
      do_exit = true;
      break;
    }

    send_peripheral_state(pm, peripheral_panda);
    ignition = send_panda_states(pm, pandas, spoofing_started);

    // check if we have new pandas and are offroad
    if (!ignition && (pandas.size() != Panda::list().size())) {
      LOGW("Reconnecting to changed amount of pandas!");
      do_exit = true;
      break;
    }

    // clear VIN, CarParams, and set new safety on car start
    if (ignition && !ignition_last) {
      params.clearAll(CLEAR_ON_IGNITION_ON);
      if (!safety_future.valid() || safety_future.wait_for(0ms) == std::future_status::ready) {
        safety_future = std::async(std::launch::async, safety_setter_thread, pandas);
      } else {
        LOGW("Safety setter thread already running");
      }
    } else if (!ignition && ignition_last) {
      params.clearAll(CLEAR_ON_IGNITION_OFF);
    }

    ignition_last = ignition;

    for (const auto &panda : pandas) {
      panda->send_heartbeat();
    }
    util::sleep_for(500);
  }
}


void peripheral_control_thread(Panda *panda) {
  LOGD("start peripheral control thread");
  SubMaster sm({"deviceState", "driverCameraState"});

  uint64_t last_front_frame_t = 0;
  uint16_t prev_fan_speed = 999;
  uint16_t ir_pwr = 0;
  uint16_t prev_ir_pwr = 999;
  bool prev_charging_disabled = false;
  unsigned int cnt = 0;

  FirstOrderFilter integ_lines_filter(0, 30.0, 0.05);

  while (!do_exit && panda->connected) {
    cnt++;
    sm.update(1000); // TODO: what happens if EINTR is sent while in sm.update?

    if (!Hardware::PC() && sm.updated("deviceState")) {
      // Charging mode
      bool charging_disabled = sm["deviceState"].getDeviceState().getChargingDisabled();
      if (charging_disabled != prev_charging_disabled) {
        if (charging_disabled) {
          panda->set_usb_power_mode(cereal::PeripheralState::UsbPowerMode::CLIENT);
          LOGW("TURN OFF CHARGING!\n");
        } else {
          panda->set_usb_power_mode(cereal::PeripheralState::UsbPowerMode::CDP);
          LOGW("TURN ON CHARGING!\n");
        }
        prev_charging_disabled = charging_disabled;
      }
    }

    // Other pandas don't have fan/IR to control
    if (panda->hw_type != cereal::PandaState::PandaType::UNO && panda->hw_type != cereal::PandaState::PandaType::DOS) continue;
    if (sm.updated("deviceState")) {
      // Fan speed
      uint16_t fan_speed = sm["deviceState"].getDeviceState().getFanSpeedPercentDesired();
      if (fan_speed != prev_fan_speed || cnt % 100 == 0) {
        panda->set_fan_speed(fan_speed);
        prev_fan_speed = fan_speed;
      }
    }
    if (sm.updated("driverCameraState")) {
      auto event = sm["driverCameraState"];
      int cur_integ_lines = event.getDriverCameraState().getIntegLines();
      float cur_gain = event.getDriverCameraState().getGain();

      if (Hardware::TICI()) {
        cur_integ_lines = integ_lines_filter.update(cur_integ_lines * cur_gain);
      }
      last_front_frame_t = event.getLogMonoTime();

      if (cur_integ_lines <= CUTOFF_IL) {
        ir_pwr = 100.0 * MIN_IR_POWER;
      } else if (cur_integ_lines > SATURATE_IL) {
        ir_pwr = 100.0 * MAX_IR_POWER;
      } else {
        ir_pwr = 100.0 * (MIN_IR_POWER + ((cur_integ_lines - CUTOFF_IL) * (MAX_IR_POWER - MIN_IR_POWER) / (SATURATE_IL - CUTOFF_IL)));
      }
    }
    // Disable ir_pwr on front frame timeout
    uint64_t cur_t = nanos_since_boot();
    if (cur_t - last_front_frame_t > 1e9) {
      ir_pwr = 0;
    }

    if (ir_pwr != prev_ir_pwr || cnt % 100 == 0 || ir_pwr >= 50.0) {
      panda->set_ir_pwr(ir_pwr);
      prev_ir_pwr = ir_pwr;
    }

    // Write to rtc once per minute when no ignition present
    if ((panda->has_rtc) && !ignition && (cnt % 120 == 1)) {
      // Write time to RTC if it looks reasonable
      setenv("TZ","UTC",1);
      struct tm sys_time = util::get_time();

      if (util::time_valid(sys_time)) {
        struct tm rtc_time = panda->get_rtc();
        double seconds = difftime(mktime(&rtc_time), mktime(&sys_time));

        if (std::abs(seconds) > 1.1) {
          panda->set_rtc(sys_time);
          LOGW("Updating panda RTC. dt = %.2f System: %s RTC: %s",
                seconds, get_time_str(sys_time).c_str(), get_time_str(rtc_time).c_str());
        }
      }
    }
  }
}

static void pigeon_publish_raw(PubMaster &pm, const std::string &dat) {
  // create message
  MessageBuilder msg;
  msg.initEvent().setUbloxRaw(capnp::Data::Reader((uint8_t*)dat.data(), dat.length()));
  pm.send("ubloxRaw", msg);
}

void pigeon_thread(Panda *panda) {
  PubMaster pm({"ubloxRaw"});
  bool ignition_last = false;

  Pigeon *pigeon = Hardware::TICI() ? Pigeon::connect("/dev/ttyHS0") : Pigeon::connect(panda);

  std::unordered_map<char, uint64_t> last_recv_time;
  std::unordered_map<char, int64_t> cls_max_dt = {
    {(char)ublox::CLASS_NAV, int64_t(900000000ULL)}, // 0.9s
    {(char)ublox::CLASS_RXM, int64_t(900000000ULL)}, // 0.9s
  };

  while (!do_exit && panda->connected) {
    bool need_reset = false;
    bool ignition_local = ignition;
    std::string recv = pigeon->receive();

    // Parse message header
    if (ignition_local && recv.length() >= 3) {
      if (recv[0] == (char)ublox::PREAMBLE1 && recv[1] == (char)ublox::PREAMBLE2) {
        const char msg_cls = recv[2];
        uint64_t t = nanos_since_boot();
        if (t > last_recv_time[msg_cls]) {
          last_recv_time[msg_cls] = t;
        }
      }
    }

    // Check based on message frequency
    for (const auto& [msg_cls, max_dt] : cls_max_dt) {
      int64_t dt = (int64_t)nanos_since_boot() - (int64_t)last_recv_time[msg_cls];
      if (ignition_last && ignition_local && dt > max_dt) {
        LOGD("ublox receive timeout, msg class: 0x%02x, dt %llu", msg_cls, dt);
        // TODO: turn on reset after verification of logs
        // need_reset = true;
      }
    }

    // Check based on null bytes
    if (ignition_local && recv.length() > 0 && recv[0] == (char)0x00) {
      need_reset = true;
      LOGW("received invalid ublox message while onroad, resetting panda GPS");
    }

    if (recv.length() > 0) {
      pigeon_publish_raw(pm, recv);
    }

    // init pigeon on rising ignition edge
    // since it was turned off in low power mode
    if((ignition_local && !ignition_last) || need_reset) {
      pigeon_active = true;
      pigeon->init();

      // Set receive times to current time
      uint64_t t = nanos_since_boot() + 10000000000ULL; // Give ublox 10 seconds to start
      for (const auto& [msg_cls, dt] : cls_max_dt) {
        last_recv_time[msg_cls] = t;
      }
    } else if (!ignition_local && ignition_last) {
      // power off on falling edge of ignition
      LOGD("powering off pigeon\n");
      pigeon->stop();
      pigeon->set_power(false);
      pigeon_active = false;
    }

    ignition_last = ignition_local;

    // 10ms - 100 Hz
    util::sleep_for(10);
  }

  delete pigeon;
}

int main(int argc, char* argv[]) {
  std::vector<std::thread> threads;
  std::vector<Panda *> pandas;
  Panda *peripheral_panda;

  LOGW("starting boardd");

  if (!Hardware::PC()) {
    int err;
    err = set_realtime_priority(54);
    assert(err == 0);
    err = set_core_affinity({Hardware::TICI() ? 4 : 3});
    assert(err == 0);
  }

  LOGW("attempting to connect");
  PubMaster pm({"pandaStates", "peripheralState"});

  // connect loop
  while (!do_exit) {
    std::vector<std::string> serials(argv + 1, argv + argc);
    if (serials.size() == 0) serials.push_back("");

    // connect to all provided serials
    for (const auto& s : serials) {
      Panda *p = usb_connect(s);
      if (p != NULL) {
        pandas.push_back(p);
      }
    }

    // send empty pandaState & peripheralState and try again
    if (pandas.size() == 0) {
      send_empty_panda_state(&pm);
      send_empty_peripheral_state(&pm);
      util::sleep_for(500);
    } else {
      break;
    }
  }

  peripheral_panda = pandas[0];

  LOGW("connected to board");

  threads.emplace_back(panda_state_thread, &pm, pandas, getenv("STARTED") != nullptr);
  threads.emplace_back(peripheral_control_thread, peripheral_panda);
  threads.emplace_back(pigeon_thread, peripheral_panda);

  threads.emplace_back(can_send_thread, pandas, getenv("FAKESEND") != nullptr);
  threads.emplace_back(can_recv_thread, pandas);

  for (auto &t : threads) t.join();

  // we have exited, clean up pandas
  for (const auto& panda : pandas){
    delete panda;
  }
}
