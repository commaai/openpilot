#include "selfdrive/boardd/boardd.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "common/params.h"
#include "common/ratekeeper.h"
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

bool check_all_connected(const std::vector<Panda *> &pandas) {
  for (const auto& panda : pandas) {
    if (!panda->connected()) {
      do_exit = true;
      return false;
    }
  }
  return true;
}

bool safety_setter_thread(std::vector<Panda *> pandas) {
  LOGD("Starting safety setter thread");

  Params p;

  // there should be at least one panda connected
  if (pandas.size() == 0) {
    return false;
  }

  // initialize to ELM327 without OBD multiplexing for fingerprinting
  bool obd_multiplexing_enabled = false;
  for (int i = 0; i < pandas.size(); i++) {
    pandas[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
  }

  // openpilot can switch between multiplexing modes for different FW queries
  while (true) {
    if (do_exit || !check_all_connected(pandas) || !ignition) {
      return false;
    }

    bool obd_multiplexing_requested = p.getBool("ObdMultiplexingEnabled");
    if (obd_multiplexing_requested != obd_multiplexing_enabled) {
      for (int i = 0; i < pandas.size(); i++) {
        const uint16_t safety_param = (i > 0 || !obd_multiplexing_requested) ? 1U : 0U;
        pandas[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, safety_param);
      }
      obd_multiplexing_enabled = obd_multiplexing_requested;
      p.putBool("ObdMultiplexingChanged", true);
    }

    if (p.getBool("FirmwareQueryDone")) {
      LOGW("finished FW query");
      break;
    }
    util::sleep_for(20);
  }

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
  LOGW("got %lu bytes CarParams", params.size());

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

  if (!panda->up_to_date() && !getenv("BOARDD_SKIP_FW_CHECK")) {
    throw std::runtime_error("Panda firmware out of date. Run pandad.py to update.");
  }

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

    // Don't send if older than 1 second
    if ((nanos_since_boot() - event.getLogMonoTime() < 1e9) && !fake_send) {
      for (const auto& panda : pandas) {
        LOGT("sending sendcan to panda: %s", (panda->hw_serial()).c_str());
        panda->can_send(event.getSendcan());
        LOGT("sendcan sent to panda: %s", (panda->hw_serial()).c_str());
      }
    } else {
      LOGE("sendcan too old to send: %" PRIu64 ", %" PRIu64, nanos_since_boot(), event.getLogMonoTime());
    }
  }
}

void can_recv_thread(std::vector<Panda *> pandas) {
  util::set_thread_name("boardd_can_recv");

  PubMaster pm({"can"});

  // run at 100Hz
  RateKeeper rk("boardd_can_recv", 100);
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

    rk.keepTime();
  }
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

  const bool red_panda_comma_three = (pandas.size() == 2) &&
                                     (pandas[0]->hw_type == cereal::PandaState::PandaType::DOS) &&
                                     (pandas[1]->hw_type == cereal::PandaState::PandaType::RED_PANDA);

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

    // on comma three setups with a red panda, the dos can
    // get false positive ignitions due to the harness box
    // without a harness connector, so ignore it
    if (red_panda_comma_three && (panda->hw_type == cereal::PandaState::PandaType::DOS)) {
      health.ignition_line_pkt = 0;
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

    bool power_save_desired = !ignition_local;
    if (health.power_save_enabled_pkt != power_save_desired) {
      panda->set_power_saving(power_save_desired);
    }

    // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
    if (!ignition_local && (health.safety_mode_pkt != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }

    if (!panda->comms_healthy()) {
      evt.setValid(false);
    }

    auto ps = pss[i];
    ps.setVoltage(health.voltage_pkt);
    ps.setCurrent(health.current_pkt);
    ps.setUptime(health.uptime_pkt);
    ps.setSafetyTxBlocked(health.safety_tx_blocked_pkt);
    ps.setSafetyRxInvalid(health.safety_rx_invalid_pkt);
    ps.setIgnitionLine(health.ignition_line_pkt);
    ps.setIgnitionCan(health.ignition_can_pkt);
    ps.setControlsAllowed(health.controls_allowed_pkt);
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
    ps.setInterruptLoad(health.interrupt_load_pkt);
    ps.setFanPower(health.fan_power);
    ps.setFanStallCount(health.fan_stall_count);
    ps.setSafetyRxChecksInvalid((bool)(health.safety_rx_checks_invalid_pkt));
    ps.setSpiChecksumErrorCount(health.spi_checksum_error_count_pkt);
    ps.setSbu1Voltage(health.sbu1_voltage_mV / 1000.0f);
    ps.setSbu2Voltage(health.sbu2_voltage_mV / 1000.0f);

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
      cs[j].setIrq0CallRate(can_health.irq0_call_rate);
      cs[j].setIrq1CallRate(can_health.irq1_call_rate);
      cs[j].setIrq2CallRate(can_health.irq2_call_rate);
      cs[j].setCanCoreResetCnt(can_health.can_core_reset_cnt);
    }

    // Convert faults bitset to capnp list
    std::bitset<sizeof(health.faults_pkt) * 8> fault_bits(health.faults_pkt);
    auto faults = ps.initFaults(fault_bits.count());

    size_t j = 0;
    for (size_t f = size_t(cereal::PandaState::FaultType::RELAY_MALFUNCTION);
         f <= size_t(cereal::PandaState::FaultType::HEARTBEAT_LOOP_WATCHDOG); f++) {
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

  double read_time = millis_since_boot();
  ps.setVoltage(Hardware::get_voltage());
  ps.setCurrent(Hardware::get_current());
  read_time = millis_since_boot() - read_time;
  if (read_time > 50) {
    LOGW("reading hwmon took %lfms", read_time);
  }

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void panda_state_thread(std::vector<Panda *> pandas, bool spoofing_started) {
  util::set_thread_name("boardd_panda_state");

  Params params;
  SubMaster sm({"controlsState"});
  PubMaster pm({"pandaStates", "peripheralState"});

  Panda *peripheral_panda = pandas[0];
  bool is_onroad = false;
  bool is_onroad_last = false;
  std::future<bool> safety_future;

  std::vector<std::string> connected_serials;
  for (Panda *p : pandas) {
    connected_serials.push_back(p->hw_serial());
  }

  LOGD("start panda state thread");

  // run at 10hz
  RateKeeper rk("panda_state_thread", 10);

  while (!do_exit && check_all_connected(pandas)) {
    // send out peripheralState at 2Hz
    if (sm.frame % 5 == 0) {
      send_peripheral_state(&pm, peripheral_panda);
    }

    auto ignition_opt = send_panda_states(&pm, pandas, spoofing_started);

    if (!ignition_opt) {
      LOGE("Failed to get ignition_opt");
      rk.keepTime();
      continue;
    }

    ignition = *ignition_opt;

    // check if we should have pandad reconnect
    if (!ignition) {
      bool comms_healthy = true;
      for (const auto &panda : pandas) {
        comms_healthy &= panda->comms_healthy();
      }

      if (!comms_healthy) {
        LOGE("Reconnecting, communication to pandas not healthy");
        do_exit = true;

      } else {
        // check for new pandas
        for (std::string &s : Panda::list(true)) {
          if (!std::count(connected_serials.begin(), connected_serials.end(), s)) {
            LOGW("Reconnecting to new panda: %s", s.c_str());
            do_exit = true;
            break;
          }
        }
      }

      if (do_exit) {
        break;
      }
    }

    is_onroad = params.getBool("IsOnroad");

    // set new safety on onroad transition, after params are cleared
    if (is_onroad && !is_onroad_last) {
      if (!safety_future.valid() || safety_future.wait_for(0ms) == std::future_status::ready) {
        safety_future = std::async(std::launch::async, safety_setter_thread, pandas);
      } else {
        LOGW("Safety setter thread already running");
      }
    }

    is_onroad_last = is_onroad;

    sm.update(0);
    const bool engaged = sm.allAliveAndValid({"controlsState"}) && sm["controlsState"].getControlsState().getEnabled();

    for (const auto &panda : pandas) {
      panda->send_heartbeat(engaged);
    }

    rk.keepTime();
  }
}


void peripheral_control_thread(Panda *panda, bool no_fan_control) {
  util::set_thread_name("boardd_peripheral_control");

  SubMaster sm({"deviceState", "driverCameraState"});

  uint64_t last_driver_camera_t = 0;
  uint16_t prev_fan_speed = 999;
  uint16_t ir_pwr = 0;
  uint16_t prev_ir_pwr = 999;

  FirstOrderFilter integ_lines_filter(0, 30.0, 0.05);

  while (!do_exit && panda->connected()) {
    sm.update(1000);

    if (sm.updated("deviceState") && !no_fan_control) {
      // Fan speed
      uint16_t fan_speed = sm["deviceState"].getDeviceState().getFanSpeedPercentDesired();
      if (fan_speed != prev_fan_speed || sm.frame % 100 == 0) {
        panda->set_fan_speed(fan_speed);
        prev_fan_speed = fan_speed;
      }
    }

    if (sm.updated("driverCameraState")) {
      auto event = sm["driverCameraState"];
      int cur_integ_lines = event.getDriverCameraState().getIntegLines();

      cur_integ_lines = integ_lines_filter.update(cur_integ_lines);
      last_driver_camera_t = event.getLogMonoTime();

      if (cur_integ_lines <= CUTOFF_IL) {
        ir_pwr = 100.0 * MIN_IR_POWER;
      } else if (cur_integ_lines > SATURATE_IL) {
        ir_pwr = 100.0 * MAX_IR_POWER;
      } else {
        ir_pwr = 100.0 * (MIN_IR_POWER + ((cur_integ_lines - CUTOFF_IL) * (MAX_IR_POWER - MIN_IR_POWER) / (SATURATE_IL - CUTOFF_IL)));
      }
    }

    // Disable IR on input timeout
    if (nanos_since_boot() - last_driver_camera_t > 1e9) {
      ir_pwr = 0;
    }

    if (ir_pwr != prev_ir_pwr || sm.frame % 100 == 0 || ir_pwr >= 50.0) {
      panda->set_ir_pwr(ir_pwr);
      prev_ir_pwr = ir_pwr;
    }
  }
}

void boardd_main_thread(std::vector<std::string> serials) {
  LOGW("launching boardd");

  if (serials.size() == 0) {
    serials = Panda::list();

    if (serials.size() == 0) {
      LOGW("no pandas found, exiting");
      return;
    }
  }

  std::string serials_str;
  for (int i = 0; i < serials.size(); i++) {
    serials_str += serials[i];
    if (i < serials.size() - 1) serials_str += ", ";
  }
  LOGW("connecting to pandas: %s", serials_str.c_str());

  // connect to all provided serials
  std::vector<Panda *> pandas;
  for (int i = 0; i < serials.size() && !do_exit; /**/) {
    Panda *p = connect(serials[i], i);
    if (!p) {
      util::sleep_for(100);
      continue;
    }

    pandas.push_back(p);
    ++i;
  }

  if (!do_exit) {
    LOGW("connected to all pandas");

    std::vector<std::thread> threads;

    threads.emplace_back(panda_state_thread, pandas, getenv("STARTED") != nullptr);
    threads.emplace_back(peripheral_control_thread, pandas[0], getenv("NO_FAN_CONTROL") != nullptr);

    threads.emplace_back(can_send_thread, pandas, getenv("FAKESEND") != nullptr);
    threads.emplace_back(can_recv_thread, pandas);

    for (auto &t : threads) t.join();
  }

  for (Panda *panda : pandas) {
    delete panda;
  }
}
