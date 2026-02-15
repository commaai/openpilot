#include "selfdrive/pandad/pandad.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cerrno>
#include <memory>
#include <thread>
#include <utility>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/messaging/messaging.h"
#include "cereal/services.h"
#include "common/ratekeeper.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "system/hardware/hw.h"

#define MAX_IR_PANDA_VAL 50
#define CUTOFF_IL 400
#define SATURATE_IL 1000

ExitHandler do_exit;

bool check_connected(Panda *panda) {
  if (!panda->connected()) {
    do_exit = true;
    return false;
  }
  return true;
}

Panda *connect(std::string serial) {
  std::unique_ptr<Panda> panda;
  try {
    panda = std::make_unique<Panda>(serial);
  } catch (std::exception &e) {
    return nullptr;
  }

  // common panda config
  if (getenv("BOARDD_LOOPBACK")) {
    panda->set_loopback(true);
  }
  //panda->enable_deepsleep();

  for (int i = 0; i < PANDA_CAN_CNT; i++) {
    panda->set_can_fd_auto(i, true);
  }

  if (!panda->up_to_date() && !getenv("BOARDD_SKIP_FW_CHECK")) {
    throw std::runtime_error("Panda firmware out of date. Run pandad.py to update.");
  }

  return panda.release();
}

void can_send_thread(Panda *panda, bool fake_send) {
  util::set_thread_name("pandad_can_send");

  AlignedBuffer aligned_buf;
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> subscriber(SubSocket::create(context.get(), "sendcan", "127.0.0.1", false, true, services.at("sendcan").queue_size));
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  // run as fast as messages come in
  while (!do_exit && check_connected(panda)) {
    std::unique_ptr<Message> msg(subscriber->receive());
    if (!msg) {
      continue;
    }

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg.get()));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    // Don't send if older than 1 second
    if ((nanos_since_boot() - event.getLogMonoTime() < 1e9) && !fake_send) {
      LOGT("sending sendcan to panda: %s", (panda->hw_serial()).c_str());
      panda->can_send(event.getSendcan());
      LOGT("sendcan sent to panda: %s", (panda->hw_serial()).c_str());
    } else {
      LOGE("sendcan too old to send: %" PRIu64 ", %" PRIu64, nanos_since_boot(), event.getLogMonoTime());
    }
  }
}

void can_recv(Panda *panda, PubMaster *pm) {
  static std::vector<can_frame> raw_can_data;
  {
    raw_can_data.clear();
    bool comms_healthy = panda->can_receive(raw_can_data);

    MessageBuilder msg;
    auto evt = msg.initEvent();
    evt.setValid(comms_healthy);
    auto canData = evt.initCan(raw_can_data.size());
    for (size_t i = 0; i < raw_can_data.size(); ++i) {
      canData[i].setAddress(raw_can_data[i].address);
      canData[i].setDat(kj::arrayPtr((uint8_t*)raw_can_data[i].dat.data(), raw_can_data[i].dat.size()));
      canData[i].setSrc(raw_can_data[i].src);
    }
    pm->send("can", msg);
  }
}

void fill_panda_state(cereal::PandaState::Builder &ps, cereal::PandaState::PandaType hw_type, const health_t &health) {
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
  ps.setPandaType(hw_type);
  ps.setSafetyModel(cereal::CarParams::SafetyModel(health.safety_mode_pkt));
  ps.setSafetyParam(health.safety_param_pkt);
  ps.setFaultStatus(cereal::PandaState::FaultStatus(health.fault_status_pkt));
  ps.setPowerSaveEnabled((bool)(health.power_save_enabled_pkt));
  ps.setHeartbeatLost((bool)(health.heartbeat_lost_pkt));
  ps.setAlternativeExperience(health.alternative_experience_pkt);
  ps.setHarnessStatus(cereal::PandaState::HarnessStatus(health.car_harness_status_pkt));
  ps.setInterruptLoad(health.interrupt_load_pkt);
  ps.setFanPower(health.fan_power);
  ps.setSafetyRxChecksInvalid((bool)(health.safety_rx_checks_invalid_pkt));
  ps.setSpiErrorCount(health.spi_error_count_pkt);
  ps.setSbu1Voltage(health.sbu1_voltage_mV / 1000.0f);
  ps.setSbu2Voltage(health.sbu2_voltage_mV / 1000.0f);
}

void fill_panda_can_state(cereal::PandaState::PandaCanState::Builder &cs, const can_health_t &can_health) {
  cs.setBusOff((bool)can_health.bus_off);
  cs.setBusOffCnt(can_health.bus_off_cnt);
  cs.setErrorWarning((bool)can_health.error_warning);
  cs.setErrorPassive((bool)can_health.error_passive);
  cs.setLastError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_error));
  cs.setLastStoredError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_stored_error));
  cs.setLastDataError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_data_error));
  cs.setLastDataStoredError(cereal::PandaState::PandaCanState::LecErrorCode(can_health.last_data_stored_error));
  cs.setReceiveErrorCnt(can_health.receive_error_cnt);
  cs.setTransmitErrorCnt(can_health.transmit_error_cnt);
  cs.setTotalErrorCnt(can_health.total_error_cnt);
  cs.setTotalTxLostCnt(can_health.total_tx_lost_cnt);
  cs.setTotalRxLostCnt(can_health.total_rx_lost_cnt);
  cs.setTotalTxCnt(can_health.total_tx_cnt);
  cs.setTotalRxCnt(can_health.total_rx_cnt);
  cs.setTotalFwdCnt(can_health.total_fwd_cnt);
  cs.setCanSpeed(can_health.can_speed);
  cs.setCanDataSpeed(can_health.can_data_speed);
  cs.setCanfdEnabled(can_health.canfd_enabled);
  cs.setBrsEnabled(can_health.brs_enabled);
  cs.setCanfdNonIso(can_health.canfd_non_iso);
  cs.setIrq0CallRate(can_health.irq0_call_rate);
  cs.setIrq1CallRate(can_health.irq1_call_rate);
  cs.setIrq2CallRate(can_health.irq2_call_rate);
  cs.setCanCoreResetCnt(can_health.can_core_reset_cnt);
}

std::optional<bool> send_panda_states(PubMaster *pm, Panda *panda, bool is_onroad, bool spoofing_started) {
  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto pss = evt.initPandaStates(1);

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

  if (spoofing_started) {
    health.ignition_line_pkt = 1;
  }

  bool ignition_local = ((health.ignition_line_pkt != 0) || (health.ignition_can_pkt != 0));

  // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
  if (health.safety_mode_pkt == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
    panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
  }

  bool power_save_desired = !ignition_local;
  if (health.power_save_enabled_pkt != power_save_desired) {
    panda->set_power_saving(power_save_desired);
  }

  // set safety mode to NO_OUTPUT when car is off or we're not onroad. ELM327 is an alternative if we want to leverage athenad/connect
  bool should_close_relay = !ignition_local || !is_onroad;
  if (should_close_relay && (health.safety_mode_pkt != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
    panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
  }

  if (!panda->comms_healthy()) {
    evt.setValid(false);
  }

  auto ps = pss[0];
  fill_panda_state(ps, panda->hw_type, health);

  auto cs = std::array{ps.initCanState0(), ps.initCanState1(), ps.initCanState2()};
  for (uint32_t j = 0; j < PANDA_CAN_CNT; j++) {
    fill_panda_can_state(cs[j], can_health[j]);
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

  pm->send("pandaStates", msg);
  return ignition_local;
}

void send_peripheral_state(Panda *panda, PubMaster *pm) {
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

  // fall back to panda's voltage and current measurement
  if (ps.getVoltage() == 0 && ps.getCurrent() == 0) {
    auto health_opt = panda->get_state();
    if (health_opt) {
      health_t health = *health_opt;
      ps.setVoltage(health.voltage_pkt);
      ps.setCurrent(health.current_pkt);
    }
  }

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void process_panda_state(Panda *panda, PubMaster *pm, bool engaged, bool is_onroad, bool spoofing_started) {
  auto ignition_opt = send_panda_states(pm, panda, is_onroad, spoofing_started);
  if (!ignition_opt) {
    LOGE("Failed to get ignition_opt");
    return;
  }

  // check if we should have pandad reconnect
  if (!ignition_opt.value()) {
    if (!panda->comms_healthy()) {
      LOGE("Reconnecting, communication to panda not healthy");
      do_exit = true;
    }
  }

  panda->send_heartbeat(engaged);
}

void process_peripheral_state(Panda *panda, PubMaster *pm, bool no_fan_control) {
  static Params params;
  static SubMaster sm({"deviceState", "driverCameraState"});

  static uint64_t last_driver_camera_t = 0;
  static uint16_t prev_fan_speed = 999;
  static int ir_pwr = 0;
  static int prev_ir_pwr = 999;
  static uint32_t prev_frame_id = UINT32_MAX;
  static bool driver_view = false;

  // TODO: can we merge these?
  static FirstOrderFilter integ_lines_filter(0, 30.0, 0.05);
  static FirstOrderFilter integ_lines_filter_driver_view(0, 5.0, 0.05);

  {
    sm.update(0);
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

      // reset the filter when camerad restarts
      if (event.getDriverCameraState().getFrameId() < prev_frame_id) {
        integ_lines_filter.reset(0);
        integ_lines_filter_driver_view.reset(0);
        driver_view = params.getBool("IsDriverViewEnabled");
      }
      prev_frame_id = event.getDriverCameraState().getFrameId();

      cur_integ_lines = (driver_view ? integ_lines_filter_driver_view : integ_lines_filter).update(cur_integ_lines);
      last_driver_camera_t = event.getLogMonoTime();

      if (cur_integ_lines <= CUTOFF_IL) {
        ir_pwr = 0;
      } else if (cur_integ_lines > SATURATE_IL) {
        ir_pwr = 100;
      } else {
        ir_pwr = 100 * (cur_integ_lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL);
      }
    }

    // Disable IR on input timeout
    if (nanos_since_boot() - last_driver_camera_t > 1e9) {
      ir_pwr = 0;
    }

    if (ir_pwr != prev_ir_pwr || sm.frame % 100 == 0) {
      int16_t ir_panda = util::map_val(ir_pwr, 0, 100, 0, MAX_IR_PANDA_VAL); 
      panda->set_ir_pwr(ir_panda);
      Hardware::set_ir_power(ir_pwr); 
      prev_ir_pwr = ir_pwr;
    }
  }
}

void pandad_run(Panda *panda) {
  const bool no_fan_control = getenv("NO_FAN_CONTROL") != nullptr;
  const bool spoofing_started = getenv("STARTED") != nullptr;
  const bool fake_send = getenv("FAKESEND") != nullptr;

  // Start the CAN send thread
  std::thread send_thread(can_send_thread, panda, fake_send);

  Params params;
  RateKeeper rk("pandad", 100);
  SubMaster sm({"selfdriveState"});
  PubMaster pm({"can", "pandaStates", "peripheralState"});
  PandaSafety panda_safety(panda);
  bool engaged = false;
  bool is_onroad = false;

  // Main loop: receive CAN data and process states
  while (!do_exit && check_connected(panda)) {
    can_recv(panda, &pm);

    // Process peripheral state at 20 Hz
    if (rk.frame() % 5 == 0) {
      process_peripheral_state(panda, &pm, no_fan_control);
    }

    // Process panda state at 10 Hz
    if (rk.frame() % 10 == 0) {
      sm.update(0);
      engaged = sm.allAliveAndValid({"selfdriveState"}) && sm["selfdriveState"].getSelfdriveState().getEnabled();
      is_onroad = params.getBool("IsOnroad");
      process_panda_state(panda, &pm, engaged, is_onroad, spoofing_started);
      panda_safety.configureSafetyMode(is_onroad);
    }

    // Send out peripheralState at 2Hz
    if (rk.frame() % 50 == 0) {
      send_peripheral_state(panda, &pm);
    }

    // Forward logs from panda to cloudlog if available
    std::string log = panda->serial_read();
    if (!log.empty()) {
      if (log.find("Register 0x") != std::string::npos) {
        // Log register divergent faults as errors
        LOGE("%s", log.c_str());
      } else {
        LOGD("%s", log.c_str());
      }
    }

    rk.keepTime();
  }

  // Close relay on exit to prevent a fault
  if (is_onroad && !engaged) {
    if (panda->connected()) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
  }

  send_thread.join();
}

void pandad_main_thread(std::string serial) {
  if (serial.empty()) {
    auto serials = Panda::list();

    if (serials.empty()) {
      LOGW("no pandas found, exiting");
      return;
    }
    serial = serials[0];
  }

  LOGW("connecting to panda: %s", serial.c_str());

  Panda *panda = nullptr;
  while (!do_exit) {
    panda = connect(serial);
    if (panda) break;
    util::sleep_for(100);
  }

  if (!do_exit) {
    LOGW("connected to panda");
    pandad_run(panda);
  }

  delete panda;
}
