#include "selfdrive/boardd/boardd.h"

#include <sched.h>
#include <sys/cdefs.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
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

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "system/hardware/hw.h"

// -- Multi-panda conventions --
// Ordering:
// - The internal panda will always be the first panda
// - Consecutive pandas will be sorted based on panda type, and then serial number
// Connecting:
// - If a panda connection is dropped, boardd will reconnect to all pandas
// - If a panda is added, we will only reconnect when we are offroad
// CAN buses:
// - Each panda will have it's block of 4 buses. E.g.: the second panda will use
//   bus numbers 4, 5, 6 and 7
// - The internal panda will always be used for accessing the OBD2 port,
//   and thus firmware queries
// Safety:
// - SafetyConfig is a list, which is mapped to the connected pandas
// - If there are more pandas connected than there are SafetyConfigs,
//   the excess pandas will remain in "silent" or "noOutput" mode
// Ignition:
// - If any of the ignition sources in any panda is high, ignition is high

#define MAX_IR_POWER 0.5f
#define MIN_IR_POWER 0.0f
#define CUTOFF_IL 400
#define SATURATE_IL 1000
using namespace std::chrono_literals;

std::atomic<bool> ignition(false);

ExitHandler do_exit;

struct BoarddState {
  Params params;

  // sendcan
  AlignedBuffer aligned_buf;
  std::unique_ptr<SubSocket> subscriber;// (SubSocket::create(context.get(), "sendcan"));

  // can recv
  bool comms_healthy = true;
  std::vector<can_frame> raw_can_data;

  // panda state
  bool ignition_last = false;
  std::future<bool> safety_future;

  // peripheral control
  uint64_t last_front_frame_t = 0;
  uint16_t prev_fan_speed = 999;
  uint16_t prev_ir_pwr = 999;
  unsigned int cnt = 0;
  FirstOrderFilter integ_lines_filter{0, 30.0, 0.05};
};


static std::string get_time_str(const struct tm &time) {
  char s[30] = {'\0'};
  std::strftime(s, std::size(s), "%Y-%m-%d %H:%M:%S", &time);
  return s;
}

bool check_all_connected(const std::vector<Panda *> &pandas) {
  for (const auto& panda : pandas) {
    if (!panda->connected()) {
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

  Params p;

  // there should be at least one panda connected
  if (pandas.size() == 0) {
    return false;
  }

  // set to ELM327 for fingerprinting
  for (int i = 0; i < pandas.size(); i++) {
    const uint16_t safety_param = (i > 0) ? 1U : 0U;
    pandas[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, safety_param);
  }

  // wait for FW query at OBD port to finish
  while (true) {
    if (do_exit || !check_all_connected(pandas) || !ignition) {
      return false;
    }

    if (p.getBool("FirmwareObdQueryDone")) {
      LOGW("finished FW query at OBD port");
      break;
    }
    util::sleep_for(20);
  }

  // set to ELM327 to finish fingerprinting and for potential ECU knockouts
  for (Panda *panda : pandas) {
    panda->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
  }

  p.putBool("ObdMultiplexingDisabled", true);

  std::string params;
  LOGW("waiting for params to set safety model");
  while (true) {
    if (do_exit || !check_all_connected(pandas) || !ignition) {
      return false;
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
  uint16_t safety_param;

  auto safety_configs = car_params.getSafetyConfigs();
  uint16_t alternative_experience = car_params.getAlternativeExperience();
  for (uint32_t i = 0; i < pandas.size(); i++) {
    auto panda = pandas[i];

    if (safety_configs.size() > i) {
      safety_model = safety_configs[i].getSafetyModel();
      safety_param = safety_configs[i].getSafetyParam();
    } else {
      // If no safety mode is specified, default to silent
      safety_model = cereal::CarParams::SafetyModel::SILENT;
      safety_param = 0U;
    }

    LOGW("panda %d: setting safety model: %d, param: %d, alternative experience: %d", i, (int)safety_model, safety_param, alternative_experience);
    panda->set_alternative_experience(alternative_experience);
    panda->set_safety_model(safety_model, safety_param);
  }

  return true;
}

Panda *connect(std::string serial="", uint32_t index=0) {
  std::unique_ptr<Panda> panda;
  try {
    panda = std::make_unique<Panda>(serial, (index * PANDA_BUS_CNT));
  } catch (std::exception &e) {
    return nullptr;
  }

  // common panda config
  if (getenv("BOARDD_LOOPBACK")) {
    panda->set_loopback(true);
  }
  //panda->enable_deepsleep();

  sync_time(panda.get(), SyncTimeDir::FROM_PANDA);
  return panda.release();
}

void can_send(BoarddState &bs, std::vector<Panda *> pandas, SubSocket *sock, bool fake_send) {
  // run as fast as messages come in
  while (!do_exit && check_all_connected(pandas)) {
    std::unique_ptr<Message> msg(sock->receive());
    if (!msg) {
      break;
    }

    capnp::FlatArrayMessageReader cmsg(bs.aligned_buf.align(msg.get()));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    // Don't send if older than 1 second
    if ((nanos_since_boot() - event.getLogMonoTime() < 1e9) && !fake_send) {
      for (const auto& panda : pandas) {
        LOGT("sending sendcan to panda: %s", (panda->hw_serial()).c_str());
        panda->can_send(event.getSendcan());
        LOGT("sendcan sent to panda: %s", (panda->hw_serial()).c_str());
      }
    }
  }
}

void can_recv(BoarddState &bs, PubMaster &pm, std::vector<Panda *> pandas) {
  bs.raw_can_data.clear();
  for (const auto& panda : pandas) {
    bs.comms_healthy &= panda->can_receive(bs.raw_can_data);
  }

  MessageBuilder msg;
  auto evt = msg.initEvent();
  evt.setValid(bs.comms_healthy);
  auto canData = evt.initCan(bs.raw_can_data.size());
  for (uint i = 0; i < bs.raw_can_data.size(); i++) {
    canData[i].setAddress(bs.raw_can_data[i].address);
    canData[i].setBusTime(bs.raw_can_data[i].busTime);
    canData[i].setDat(kj::arrayPtr((uint8_t*)bs.raw_can_data[i].dat.data(), bs.raw_can_data[i].dat.size()));
    canData[i].setSrc(bs.raw_can_data[i].src);
  }
  pm.send("can", msg);
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
  const uint32_t pandas_cnt = pandas.size();

  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto pss = evt.initPandaStates(pandas_cnt);

  std::vector<health_t> pandaStates;
  pandaStates.reserve(pandas_cnt);

  std::vector<std::array<can_health_t, PANDA_CAN_CNT>> pandaCanStates;
  pandaCanStates.reserve(pandas_cnt);

  for (const auto& panda : pandas){
    auto health_opt = panda->get_state();
    if (!health_opt) {
      return std::nullopt;
    }

    health_t health = *health_opt;

    std::array<can_health_t, PANDA_CAN_CNT> can_health{};
    for (uint32_t i = 0; i < PANDA_CAN_CNT; i++) {
      auto can_health_opt = panda->get_can_state(i);
      if (!can_health_opt) {
        return std::nullopt;
      }
      can_health[i] = *can_health_opt;
    }
    pandaCanStates.push_back(can_health);

    if (spoofing_started) {
      health.ignition_line_pkt = 1;
    }

    ignition_local |= ((health.ignition_line_pkt != 0) || (health.ignition_can_pkt != 0));

    pandaStates.push_back(health);
  }

  for (uint32_t i = 0; i < pandas_cnt; i++) {
    auto panda = pandas[i];
    const auto &health = pandaStates[i];

    // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
    if (health.safety_mode_pkt == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }

  #ifndef __x86_64__
    bool power_save_desired = !ignition_local;
    if (health.power_save_enabled_pkt != power_save_desired) {
      panda->set_power_saving(power_save_desired);
    }

    // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
    if (!ignition_local && (health.safety_mode_pkt != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
  #endif

    if (!panda->comms_healthy()) {
      evt.setValid(false);
    }

    auto ps = pss[i];
    ps.setUptime(health.uptime_pkt);
    ps.setSafetyTxBlocked(health.safety_tx_blocked_pkt);
    ps.setSafetyRxInvalid(health.safety_rx_invalid_pkt);
    ps.setIgnitionLine(health.ignition_line_pkt);
    ps.setIgnitionCan(health.ignition_can_pkt);
    ps.setControlsAllowed(health.controls_allowed_pkt);
    ps.setGasInterceptorDetected(health.gas_interceptor_detected_pkt);
    ps.setTxBufferOverflow(health.tx_buffer_overflow_pkt);
    ps.setRxBufferOverflow(health.rx_buffer_overflow_pkt);
    ps.setGmlanSendErrs(health.gmlan_send_errs_pkt);
    ps.setPandaType(panda->hw_type);
    ps.setSafetyModel(cereal::CarParams::SafetyModel(health.safety_mode_pkt));
    ps.setSafetyParam(health.safety_param_pkt);
    ps.setFaultStatus(cereal::PandaState::FaultStatus(health.fault_status_pkt));
    ps.setPowerSaveEnabled((bool)(health.power_save_enabled_pkt));
    ps.setHeartbeatLost((bool)(health.heartbeat_lost_pkt));
    ps.setAlternativeExperience(health.alternative_experience_pkt);
    ps.setHarnessStatus(cereal::PandaState::HarnessStatus(health.car_harness_status_pkt));
    ps.setInterruptLoad(health.interrupt_load);
    ps.setFanPower(health.fan_power);
    ps.setSafetyRxChecksInvalid((bool)(health.safety_rx_checks_invalid));

    std::array<cereal::PandaState::PandaCanState::Builder, PANDA_CAN_CNT> cs = {ps.initCanState0(), ps.initCanState1(), ps.initCanState2()};

    for (uint32_t j = 0; j < PANDA_CAN_CNT; j++) {
      const auto &can_health = pandaCanStates[i][j];
      cs[j].setBusOff((bool)can_health.bus_off);
      cs[j].setBusOffCnt(can_health.bus_off_cnt);
      cs[j].setErrorWarning((bool)can_health.error_warning);
      cs[j].setErrorPassive((bool)can_health.error_passive);
      cs[j].setLastError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_error));
      cs[j].setLastStoredError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_stored_error));
      cs[j].setLastDataError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_data_error));
      cs[j].setLastDataStoredError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_data_stored_error));
      cs[j].setReceiveErrorCnt(can_health.receive_error_cnt);
      cs[j].setTransmitErrorCnt(can_health.transmit_error_cnt);
      cs[j].setTotalErrorCnt(can_health.total_error_cnt);
      cs[j].setTotalTxLostCnt(can_health.total_tx_lost_cnt);
      cs[j].setTotalRxLostCnt(can_health.total_rx_lost_cnt);
      cs[j].setTotalTxCnt(can_health.total_tx_cnt);
      cs[j].setTotalRxCnt(can_health.total_rx_cnt);
      cs[j].setTotalFwdCnt(can_health.total_fwd_cnt);
      cs[j].setCanSpeed(can_health.can_speed);
      cs[j].setCanDataSpeed(can_health.can_data_speed);
      cs[j].setCanfdEnabled(can_health.canfd_enabled);
      cs[j].setBrsEnabled(can_health.brs_enabled);
      cs[j].setCanfdNonIso(can_health.canfd_non_iso);
    }

    // Convert faults bitset to capnp list
    std::bitset<sizeof(health.faults_pkt) * 8> fault_bits(health.faults_pkt);
    auto faults = ps.initFaults(fault_bits.count());

    size_t j = 0;
    for (size_t f = size_t(cereal::PandaState::FaultType::RELAY_MALFUNCTION);
        f <= size_t(cereal::PandaState::FaultType::INTERRUPT_RATE_EXTI); f++) {
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
  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  evt.setValid(panda->comms_healthy());

  auto ps = evt.initPeripheralState();
  ps.setPandaType(panda->hw_type);

  // TODO: this rarely lags, should probably be moved to deviceState anyway
  double read_time = millis_since_boot();
  //ps.setVoltage(Hardware::get_voltage());
  //ps.setCurrent(Hardware::get_current());
  read_time = millis_since_boot() - read_time;
  if (read_time > 50) {
    LOGW("reading hwmon took %lfms", read_time);
  }

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void panda_state(BoarddState &bs, PubMaster *pm, SubMaster &sm, std::vector<Panda *> pandas, bool spoofing_started) {
  Panda *peripheral_panda = pandas[0];

  // send out peripheralState
  send_peripheral_state(pm, peripheral_panda);
  auto ignition_opt = send_panda_states(pm, pandas, spoofing_started);

  if (!ignition_opt) {
    return;
  }

  ignition = *ignition_opt;

  // clear ignition-based params and set new safety on car start
  if (ignition && !bs.ignition_last) {
    bs.params.clearAll(CLEAR_ON_IGNITION_ON);
    if (!bs.safety_future.valid() || bs.safety_future.wait_for(0ms) == std::future_status::ready) {
      bs.safety_future = std::async(std::launch::async, safety_setter_thread, pandas);
    } else {
      LOGW("Safety setter thread already running");
    }
  } else if (!ignition && bs.ignition_last) {
    bs.params.clearAll(CLEAR_ON_IGNITION_OFF);
  }

  bs.ignition_last = ignition;

  const bool engaged = sm.allAliveAndValid({"controlsState"}) && sm["controlsState"].getControlsState().getEnabled();
  for (const auto &panda : pandas) {
    panda->send_heartbeat(engaged);
  }
}


void peripheral_control(BoarddState &bs, Panda *panda, bool no_fan_control, SubMaster &sm) {
  bs.cnt++;

  if (sm.updated("deviceState") && !no_fan_control) {
    // Fan speed
    uint16_t fan_speed = sm["deviceState"].getDeviceState().getFanSpeedPercentDesired();
    if (fan_speed != bs.prev_fan_speed || bs.cnt % 100 == 0) {
      panda->set_fan_speed(fan_speed);
      bs.prev_fan_speed = fan_speed;
    }
  }

  uint16_t ir_pwr = 0;
  if (sm.updated("driverCameraState")) {
    auto event = sm["driverCameraState"];
    int cur_integ_lines = event.getDriverCameraState().getIntegLines();

    cur_integ_lines = bs.integ_lines_filter.update(cur_integ_lines);
    bs.last_front_frame_t = event.getLogMonoTime();

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
  if (cur_t - bs.last_front_frame_t > 1e9) {
    ir_pwr = 0;
  }

  if (ir_pwr != bs.prev_ir_pwr || bs.cnt % 100 == 0 || ir_pwr >= 50.0) {
    panda->set_ir_pwr(ir_pwr);
    bs.prev_ir_pwr = ir_pwr;
  }

  // Write to rtc once per minute when no ignition present
  if (!ignition && (bs.cnt % 120 == 1)) {
    sync_time(panda, SyncTimeDir::TO_PANDA);
  }
}


void boardd_main_thread(std::vector<std::string> serials) {
  LOGW("attempting to connect");

  BoarddState bs;
  PubMaster pm({"can", "pandaStates", "peripheralState"});
  SubMaster sm({"deviceState", "driverCameraState", "controlsState"});

  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sendcan_sock(SubSocket::create(context.get(), "sendcan"));
  assert(sendcan_sock != NULL);
  sendcan_sock->setTimeout(0);

  if (serials.size() == 0) {
    // connect to all
    serials = Panda::list();

    // exit if no pandas are connected
    if (serials.size() == 0) {
      LOGW("no pandas found, exiting");
      return;
    }
  }

  // connect to all provided serials
  std::vector<Panda *> pandas;
  for (int i = 0; i < serials.size() && !do_exit; /**/) {
    Panda *p = connect(serials[i], i);
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

  // 100Hz loop
  int frame = 0;
  const uint64_t dt = 10000000ULL;
  struct timespec ts;
  uint64_t next_frame_time = nanos_since_boot() + dt;
  while (!do_exit && check_all_connected(pandas)) {
    // 100Hz stuff
    can_recv(bs, pm, pandas);
    can_send(bs, pandas, sendcan_sock.get(), getenv("FAKESEND") != nullptr);

    // 2Hz stuff
    if ((frame % 50) == 0) {
      panda_state(bs, &pm, sm, pandas, getenv("STARTED") != nullptr);
      peripheral_control(bs, pandas[0], getenv("NO_FAN_CONTROL") != nullptr, sm);

      // TODO: make this check fast, currently takes 16ms
      // check if we have new pandas and are offroad
      if (!ignition && (pandas.size() != Panda::list().size())) {
        LOGW("Reconnecting to changed amount of pandas!");
        do_exit = true;
        return;
      }
    }

    // setup next loop
    uint64_t cur_time = nanos_since_boot();
    int64_t remaining = next_frame_time - cur_time;
    if (remaining > 0) {
      ts.tv_sec = 0;
      ts.tv_nsec = remaining;
      nanosleep(&ts, &ts);
    } else {
      if (ignition) {
        LOGW("missed cycles (%d) %lld", (int)-1*remaining/dt, remaining);
      }
      next_frame_time = cur_time;
    }

    frame++;
    next_frame_time += dt;
  }

  // we have exited, clean up pandas
  for (Panda *panda : pandas) {
    delete panda;
  }
}
