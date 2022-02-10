#include "selfdrive/boardd/boardd.h"

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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

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

static std::string get_time_str(const struct tm &time) {
  char s[30] = {'\0'};
  std::strftime(s, std::size(s), "%Y-%m-%d %H:%M:%S", &time);
  return s;
}

bool check_all_connected(const std::vector<Panda *> &pandas) {
  for (const auto& panda : pandas) {
    if (!panda->connected) {
      do_exit = true;
      return false;
    }
  }
  return true;
}

enum class SyncTimeDir { TO_PANDA, FROM_PANDA };

void sync_time(Panda *panda, SyncTimeDir dir) {
  if (!panda->has_rtc) return;

  setenv("TZ", "UTC", 1);
  struct tm sys_time = util::get_time();
  struct tm rtc_time = panda->get_rtc();

  if (dir == SyncTimeDir::TO_PANDA) {
    if (util::time_valid(sys_time)) {
      // Write time to RTC if it looks reasonable
      double seconds = difftime(mktime(&rtc_time), mktime(&sys_time));
      if (std::abs(seconds) > 1.1) {
        panda->set_rtc(sys_time);
        LOGW("Updating panda RTC. dt = %.2f System: %s RTC: %s",
              seconds, get_time_str(sys_time).c_str(), get_time_str(rtc_time).c_str());
      }
    }
  } else if (dir == SyncTimeDir::FROM_PANDA) {
    if (!util::time_valid(sys_time) && util::time_valid(rtc_time)) {
      const struct timeval tv = {mktime(&rtc_time), 0};
      settimeofday(&tv, 0);
      LOGE("System time wrong, setting from RTC. System: %s RTC: %s",
           get_time_str(sys_time).c_str(), get_time_str(rtc_time).c_str());
    }
  }
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
  uint16_t unsafe_mode = car_params.getUnsafeMode();
  for (uint32_t i = 0; i < pandas.size(); i++) {
    auto panda = pandas[i];

    if (safety_configs.size() > i) {
      safety_model = safety_configs[i].getSafetyModel();
      safety_param = safety_configs[i].getSafetyParam();
    } else {
      // If no safety mode is specified, default to silent
      safety_model = cereal::CarParams::SafetyModel::SILENT;
      safety_param = 0;
    }

    LOGW("panda %d: setting safety model: %d, param: %d, unsafe mode: %d", i, (int)safety_model, safety_param, unsafe_mode);
    panda->set_unsafe_mode(unsafe_mode);
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

  if (getenv("BOARDD_LOOPBACK")) {
    panda->set_loopback(true);
  }

  // power on charging, only the first time. Panda can also change mode and it causes a brief disconneciton
#ifndef __x86_64__
  static std::once_flag connected_once;
  std::call_once(connected_once, &Panda::set_usb_power_mode, panda, cereal::PeripheralState::UsbPowerMode::CDP);
#endif

  sync_time(panda.get(), SyncTimeDir::FROM_PANDA);
  return panda.release();
}

void can_send_thread(std::vector<Panda *> pandas, bool fake_send) {
  util::set_thread_name("boardd_can_send");

  AlignedBuffer aligned_buf;
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> subscriber(SubSocket::create(context.get(), "sendcan"));
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  // run as fast as messages come in
  while (!do_exit && check_all_connected(pandas)) {
    std::unique_ptr<Message> msg(subscriber->receive());
    if (!msg) {
      if (errno == EINTR) {
        do_exit = true;
      }
      continue;
    }

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg.get()));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    //Dont send if older than 1 second
    if ((nanos_since_boot() - event.getLogMonoTime() < 1e9) && !fake_send) {
      for (const auto& panda : pandas) {
        panda->can_send(event.getSendcan());
      }
    }
  }
}

void can_recv_thread(std::vector<Panda *> pandas) {
  util::set_thread_name("boardd_can_recv");

  // can = 8006
  PubMaster pm({"can"});

  // run at 100hz
  const uint64_t dt = 10000000ULL;
  uint64_t next_frame_time = nanos_since_boot() + dt;
  std::vector<can_frame> raw_can_data;

  while (!do_exit && check_all_connected(pandas)) {
    bool comms_healthy = true;
    raw_can_data.clear();
    for (const auto& panda : pandas) {
      comms_healthy &= panda->can_receive(raw_can_data);
    }

    MessageBuilder msg;
    auto evt = msg.initEvent();
    evt.setValid(comms_healthy);
    auto canData = evt.initCan(raw_can_data.size());
    for (uint i = 0; i<raw_can_data.size(); i++) {
      canData[i].setAddress(raw_can_data[i].address);
      canData[i].setBusTime(raw_can_data[i].busTime);
      canData[i].setDat(kj::arrayPtr((uint8_t*)raw_can_data[i].dat.data(), raw_can_data[i].dat.size()));
      canData[i].setSrc(raw_can_data[i].src);
    }
    pm.send("can", msg);

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

std::optional<bool> send_panda_states(PubMaster *pm, const std::vector<Panda *> &pandas, bool spoofing_started) {
  bool ignition_local = false;

  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto pss = evt.initPandaStates(pandas.size());

  std::vector<health_t> pandaStates;
  for (const auto& panda : pandas){
    auto health_opt = panda->get_state();
    if (!health_opt) {
      return std::nullopt;
    }

    health_t health = *health_opt;

    if (spoofing_started) {
      health.ignition_line_pkt = 1;
    }

    ignition_local |= ((health.ignition_line_pkt != 0) || (health.ignition_can_pkt != 0));

    pandaStates.push_back(health);
  }

  for (uint32_t i = 0; i < pandas.size(); i++) {
    auto panda = pandas[i];
    const auto &health = pandaStates[i];

    // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
    if (health.safety_mode_pkt == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }

  #ifndef __x86_64__
    bool power_save_desired = !ignition_local && !pigeon_active;
    if (health.power_save_enabled_pkt != power_save_desired) {
      panda->set_power_saving(power_save_desired);
    }

    // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
    if (!ignition_local && (health.safety_mode_pkt != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
  #endif

    if (!panda->comms_healthy) {
      evt.setValid(false);
    }

    auto ps = pss[i];
    ps.setUptime(health.uptime_pkt);
    ps.setBlockedCnt(health.blocked_msg_cnt_pkt);
    ps.setIgnitionLine(health.ignition_line_pkt);
    ps.setIgnitionCan(health.ignition_can_pkt);
    ps.setControlsAllowed(health.controls_allowed_pkt);
    ps.setGasInterceptorDetected(health.gas_interceptor_detected_pkt);
    ps.setCanRxErrs(health.can_rx_errs_pkt);
    ps.setCanSendErrs(health.can_send_errs_pkt);
    ps.setCanFwdErrs(health.can_fwd_errs_pkt);
    ps.setGmlanSendErrs(health.gmlan_send_errs_pkt);
    ps.setPandaType(panda->hw_type);
    ps.setSafetyModel(cereal::CarParams::SafetyModel(health.safety_mode_pkt));
    ps.setSafetyParam(health.safety_param_pkt);
    ps.setFaultStatus(cereal::PandaState::FaultStatus(health.fault_status_pkt));
    ps.setPowerSaveEnabled((bool)(health.power_save_enabled_pkt));
    ps.setHeartbeatLost((bool)(health.heartbeat_lost_pkt));
    ps.setUnsafeMode(health.unsafe_mode_pkt);
    ps.setHarnessStatus(cereal::PandaState::HarnessStatus(health.car_harness_status_pkt));

    // Convert faults bitset to capnp list
    std::bitset<sizeof(health.faults_pkt) * 8> fault_bits(health.faults_pkt);
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
  auto pandaState_opt = panda->get_state();
  if (!pandaState_opt) {
    return;
  }

  health_t pandaState = *pandaState_opt;

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
    ps.setVoltage(pandaState.voltage_pkt);
    ps.setCurrent(pandaState.current_pkt);
  }

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setUsbPowerMode(cereal::PeripheralState::UsbPowerMode(pandaState.usb_power_mode_pkt));
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void panda_state_thread(PubMaster *pm, std::vector<Panda *> pandas, bool spoofing_started) {
  util::set_thread_name("boardd_panda_state");

  Params params;
  SubMaster sm({"controlsState"});

  Panda *peripheral_panda = pandas[0];
  bool ignition_last = false;
  std::future<bool> safety_future;

  LOGD("start panda state thread");

  // run at 2hz
  while (!do_exit && check_all_connected(pandas)) {
    uint64_t start_time = nanos_since_boot();

    // send out peripheralState
    send_peripheral_state(pm, peripheral_panda);
    auto ignition_opt = send_panda_states(pm, pandas, spoofing_started);

    if (!ignition_opt) {
      continue;
    }

    ignition = *ignition_opt;

    // TODO: make this check fast, currently takes 16ms
    // check if we have new pandas and are offroad
    if (!ignition && (pandas.size() != Panda::list().size())) {
      LOGW("Reconnecting to changed amount of pandas!");
      do_exit = true;
      break;
    }

    // clear ignition-based params and set new safety on car start
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

    sm.update(0);
    const bool engaged = sm.allAliveAndValid({"controlsState"}) && sm["controlsState"].getControlsState().getEnabled();

    for (const auto &panda : pandas) {
      panda->send_heartbeat(engaged);
    }

    uint64_t dt = nanos_since_boot() - start_time;
    util::sleep_for(500 - dt / 1000000ULL);
  }
}


void peripheral_control_thread(Panda *panda) {
  util::set_thread_name("boardd_peripheral_control");

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
    if (!ignition && (cnt % 120 == 1)) {
      sync_time(panda, SyncTimeDir::TO_PANDA);
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
  util::set_thread_name("boardd_pigeon");

  PubMaster pm({"ubloxRaw"});
  bool ignition_last = false;

  std::unique_ptr<Pigeon> pigeon(Hardware::TICI() ? Pigeon::connect("/dev/ttyHS0") : Pigeon::connect(panda));

  while (!do_exit && panda->connected) {
    bool need_reset = false;
    bool ignition_local = ignition;
    std::string recv = pigeon->receive();

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
}

void boardd_main_thread(std::vector<std::string> serials) {
  if (serials.size() == 0) serials.push_back("");

  PubMaster pm({"pandaStates", "peripheralState"});
  LOGW("attempting to connect");

  // connect to all provided serials
  std::vector<Panda *> pandas;
  for (int i = 0; i < serials.size() && !do_exit; /**/) {
    Panda *p = usb_connect(serials[i], i);
    if (!p) {
      // send empty pandaState & peripheralState and try again
      send_empty_panda_state(&pm);
      send_empty_peripheral_state(&pm);
      util::sleep_for(500);
      continue;
    }

    pandas.push_back(p);
    ++i;
  }

  if (!do_exit) {
    LOGW("connected to board");
    Panda *peripheral_panda = pandas[0];
    std::vector<std::thread> threads;

    Params().put("LastPeripheralPandaType", std::to_string((int) peripheral_panda->get_hw_type()));

    threads.emplace_back(panda_state_thread, &pm, pandas, getenv("STARTED") != nullptr);
    threads.emplace_back(peripheral_control_thread, peripheral_panda);
    threads.emplace_back(pigeon_thread, peripheral_panda);

    threads.emplace_back(can_send_thread, pandas, getenv("FAKESEND") != nullptr);
    threads.emplace_back(can_recv_thread, pandas);

    for (auto &t : threads) t.join();
  }

  // we have exited, clean up pandas
  for (Panda *panda : pandas) {
    delete panda;
  }
}
