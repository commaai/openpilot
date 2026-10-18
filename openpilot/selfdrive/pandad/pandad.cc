#include "selfdrive/pandad/pandad.h"
#include "selfdrive/pandad/tss3_startup_catcher.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include "openpilot/cereal/gen/cpp/car.capnp.h"
#include "openpilot/cereal/messaging/messaging.h"
#include "openpilot/cereal/services.h"
#include "common/ratekeeper.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "common/hardware/hw.h"

#define MAX_IR_PANDA_VAL 50
#define CUTOFF_IL 400
#define SATURATE_IL 1000

ExitHandler do_exit;

namespace {
constexpr uint32_t TSS3_EPS_TX = 0x7A1U;
constexpr uint32_t TSS3_EPS_RX = 0x7A9U;
constexpr uint32_t TSS3_WAKE_TRIGGER_ADDR = 0x45AU;
constexpr uint8_t TSS3_DIAG_BUS = 0U;
constexpr uint64_t TSS3_EXTENDED_PERIOD_NS = 20ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_PREWARM_PERIOD_NS = 100ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_PREWARM_MAX_NS = 60ULL * 1000ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_WAKE_IDLE_NS = 3ULL * 1000ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_CATCH_TIMEOUT_NS = 2ULL * 1000ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_STATE_POLL_NS = 10ULL * 1000ULL * 1000ULL;
constexpr uint64_t TSS3_PARAM_POLL_NS = 250ULL * 1000ULL * 1000ULL;
constexpr char TSS3_NATIVE_CATCH_PATH[] = "/tmp/tss3-oracle-native-catch.json";
constexpr char TSS3_NATIVE_NOTIFY_PATH[] = "/tmp/tss3-oracle-native-catch.sock";
// Mirrors ToyotaSafetyFlags.TSS3 in opendbc/car/toyota/values.py. Any Toyota
// platform OpenDBC marks TSS3 sets this safety param bit, so the catcher
// follows the platform list instead of matching one car.
constexpr uint16_t TOYOTA_PARAM_TSS3 = 16U << 8;
const std::string TSS3_EXTENDED_FRAME("\x02\x10\x03\x00\x00\x00\x00\x00", 8);
const std::string TSS3_PROGRAMMING_FRAME("\x02\x10\x02\x00\x00\x00\x00\x00", 8);

class Tss3OracleStartupCatcher {
public:
  void update(Panda *panda) {
    const uint64_t now = nanos_since_boot();
    if ((now - last_param_check_ns_) >= TSS3_PARAM_POLL_NS || !param_initialized_) {
      enabled_ = tss3_auto_enabled();
      last_param_check_ns_ = now;
      param_initialized_ = true;
      if (!enabled_ && active()) {
        LOGW("TSS3 oracle startup catcher disarmed");
        state_ = ignition_initialized_ && last_ignition_ ? State::WAIT_OFF : State::DISARMED;
      }
    }

    if ((now - last_state_check_ns_) >= TSS3_STATE_POLL_NS || !ignition_initialized_) {
      last_state_check_ns_ = now;
      auto health = panda->get_state();
      if (health) {
        const bool ignition = (health->flags_pkt & (HEALTH_FLAG_IGNITION_LINE | HEALTH_FLAG_IGNITION_CAN)) != 0U;
        update_ignition(panda, ignition, now);
      }
    }

    if (state_ == State::PREWARM) {
      if ((now - wake_ns_) > TSS3_PREWARM_MAX_NS) {
        LOGW("TSS3 oracle startup prewarm reached 60 s; waiting for sleep or ignition");
        state_ = State::WAIT_SLEEP;
      } else if ((now - last_native_rx_ns_) > TSS3_WAKE_IDLE_NS) {
        LOGW("TSS3 oracle startup prewarm saw vehicle bus return quiet; re-arming");
        reset_wake_attempt();
        state_ = State::ARMED;
      } else if (now >= next_extended_tx_ns_) {
        send_extended(panda, now, TSS3_PREWARM_PERIOD_NS);
      }
    } else if (state_ == State::WAIT_SLEEP) {
      if ((now - last_native_rx_ns_) > TSS3_WAKE_IDLE_NS) {
        LOGW("TSS3 oracle startup wake interval ended; re-arming");
        reset_wake_attempt();
        state_ = State::ARMED;
      }
    } else if (state_ == State::CATCHING) {
      if ((now - ignition_ns_) > TSS3_CATCH_TIMEOUT_NS) {
        LOGW("TSS3 oracle startup catcher timed out without exact positive 50 03");
        state_ = State::WAIT_OFF;
        return;
      }
      if (now >= next_extended_tx_ns_) {
        send_extended(panda, now, TSS3_EXTENDED_PERIOD_NS);
      }
    }
  }

  void process_rx(Panda *panda, const std::vector<can_frame> &frames) {
    for (const auto &frame : frames) {
      if (frame.src == TSS3_DIAG_BUS) {
        last_native_rx_ns_ = nanos_since_boot();
      }

      if (state_ == State::ARMED && frame.src == TSS3_DIAG_BUS && frame.address == TSS3_WAKE_TRIGGER_ADDR) {
        start_prewarm(panda, nanos_since_boot());
      }
      if (state_ == State::CATCHING && frame.address == TSS3_EPS_RX && frame.src == TSS3_DIAG_BUS &&
          is_tss3_exact_extended_response(frame.dat)) {
        const uint64_t positive_ns = nanos_since_boot();
        positive_extended_frame_ = frame.dat;
        panda->can_send(TSS3_EPS_TX, TSS3_PROGRAMMING_FRAME, TSS3_DIAG_BUS);
        const uint64_t programming_ns = nanos_since_boot();
        positive_extended_ns_ = positive_ns;
        programming_tx_ns_ = programming_ns;
        if (write_marker()) notify_watcher();
        state_ = State::CAUGHT;
        LOGW("TSS3 oracle startup catcher sent 10 02 %.3f ms after exact positive 50 03",
             (programming_ns - positive_ns) / 1e6);
        return;
      }
    }
  }

  bool active() const {
    // Pre-start states deliberately count as active so ordinary offroad safety
    // does not replace the preloaded ELM327 policy. Panda power saving remains
    // enabled through ARMED/PREWARM/WAIT_SLEEP; only the always-on main bus is
    // used until the native ignition edge.
    return state_ == State::ARMED || state_ == State::PREWARM || state_ == State::WAIT_SLEEP ||
           state_ == State::CATCHING || state_ == State::CAUGHT;
  }

  bool caught() const {
    return state_ == State::CAUGHT;
  }

private:
  enum class State { DISARMED, ARMED, PREWARM, WAIT_SLEEP, CATCHING, CAUGHT, WAIT_OFF };

  bool tss3_auto_enabled() {
    if (!params_.getBool("Tss3OracleAutoArm")) return false;
    const std::string cp_bytes = params_.get("CarParamsPersistent");
    if (cp_bytes.empty()) return false;
    try {
      AlignedBuffer aligned_buf;
      capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
      const auto CP = cmsg.getRoot<cereal::CarParams>();
      const auto safety_configs = CP.getSafetyConfigs();
      if (safety_configs.size() == 0) return false;
      const bool tss3 = (safety_configs[0].getSafetyModel() == cereal::CarParams::SafetyModel::TOYOTA) &&
                        ((safety_configs[0].getSafetyParam() & TOYOTA_PARAM_TSS3) != 0U);
      if (!tss3) return false;
      fingerprint_ = CP.getCarFingerprint().cStr();
      return true;
    } catch (const kj::Exception &e) {
      LOGE("TSS3 oracle startup catcher could not parse CarParamsPersistent: %s", e.getDescription().cStr());
      return false;
    }
  }

  void update_ignition(Panda *panda, bool ignition, uint64_t now) {
    if (!ignition_initialized_) {
      ignition_initialized_ = true;
      last_ignition_ = ignition;
      if (ignition) {
        state_ = State::WAIT_OFF;
        return;
      }
      if (enabled_) arm(panda);
      return;
    }

    if (ignition && !last_ignition_ &&
        (state_ == State::ARMED || state_ == State::PREWARM || state_ == State::WAIT_SLEEP)) {
      start(panda, now);
    } else if (!ignition && last_ignition_) {
      reset_for_off();
      if (enabled_) arm(panda);
    } else if (!ignition && enabled_ && state_ == State::DISARMED) {
      arm(panda);
    } else if (ignition && enabled_ && state_ == State::DISARMED) {
      state_ = State::WAIT_OFF;
    }
    last_ignition_ = ignition;
  }

  void reset_wake_attempt() {
    wake_ns_ = 0;
    first_extended_tx_ns_ = 0;
    next_extended_tx_ns_ = 0;
    positive_extended_frame_.clear();
  }

  void reset_for_off() {
    state_ = State::DISARMED;
    ignition_ns_ = 0;
    last_native_rx_ns_ = 0;
    reset_wake_attempt();
    positive_extended_ns_ = 0;
    programming_tx_ns_ = 0;
    first_extended_before_power_wake_ = false;
    power_wake_complete_ns_ = 0;
    std::remove(TSS3_NATIVE_CATCH_PATH);
  }

  void arm(Panda *panda) {
    std::remove(TSS3_NATIVE_CATCH_PATH);
    // Safety policy is a firmware-side filter and does not wake CAN hardware.
    // Preload it while OFF so the ignition hot path has no safety control
    // transfer ahead of the first diagnostic request.
    panda->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
    // Keep explicit packet formats authoritative after CAN-FD wake traffic,
    // just as normal TSS3 safety does.
    for (uint16_t bus = 0U; bus < PANDA_CAN_CNT; bus++) {
      panda->set_can_fd_auto(bus, false);
    }
    state_ = State::ARMED;
    LOGW("TSS3 oracle startup catcher armed in Panda power-save; waiting for vehicle wake");
  }

  void start_prewarm(Panda *panda, uint64_t now) {
    std::remove(TSS3_NATIVE_CATCH_PATH);
    wake_ns_ = now;
    last_native_rx_ns_ = now;
    state_ = State::PREWARM;
    // The exact 2026 Camry proximity capture observed 0x45A as the first
    // mirrored wake frame. Logical bus 0 is Panda's always-awake main bus, so
    // offer EXTENDED at low cadence without disabling Panda power-save.
    send_extended(panda, now, TSS3_PREWARM_PERIOD_NS);
    first_extended_before_power_wake_ = true;
    LOGW("TSS3 oracle startup prewarm triggered by native bus0 0x45A");
  }

  void start(Panda *panda, uint64_t now) {
    std::remove(TSS3_NATIVE_CATCH_PATH);
    ignition_ns_ = now;
    state_ = State::CATCHING;
    // Logical bus 0 is the harness main bus and its transceiver remains enabled
    // in Panda power-save for CAN ignition detection. Queue 10 03 before the
    // control transfer that wakes the remaining transceivers.
    send_extended(panda, nanos_since_boot(), TSS3_EXTENDED_PERIOD_NS);
    first_extended_before_power_wake_ = true;
    panda->set_power_saving(false);
    power_wake_complete_ns_ = nanos_since_boot();
    LOGW("TSS3 oracle startup catcher promoted to fast catch on native ignition edge");
  }

  void send_extended(Panda *panda, uint64_t now, uint64_t period_ns) {
    panda->can_send(TSS3_EPS_TX, TSS3_EXTENDED_FRAME, TSS3_DIAG_BUS);
    const uint64_t sent_ns = nanos_since_boot();
    if (first_extended_tx_ns_ == 0) first_extended_tx_ns_ = sent_ns;
    next_extended_tx_ns_ = now + period_ns;
  }

  bool write_marker() const {
    const std::string tmp = std::string(TSS3_NATIVE_CATCH_PATH) + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) {
      LOGE("failed to open TSS3 native catch marker: errno=%d", errno);
      return false;
    }
    out << "{\n"
        << "  \"schema\": \"tss3-oracle-native-catch-v1\",\n"
        << "  \"target\": \"" << fingerprint_ << "\",\n"
        << "  \"pandad_wrapper_pid\": " << getppid() << ",\n"
        << "  \"wake_trigger_monotonic_ns\": " << wake_ns_ << ",\n"
        << "  \"wake_trigger_address\": " << TSS3_WAKE_TRIGGER_ADDR << ",\n"
        << "  \"ignition_monotonic_ns\": " << ignition_ns_ << ",\n"
        << "  \"first_extended_tx_monotonic_ns\": " << first_extended_tx_ns_ << ",\n"
        << "  \"positive_extended_monotonic_ns\": " << positive_extended_ns_ << ",\n"
        << "  \"positive_extended_frame_hex\": \""
        << util::hexdump(reinterpret_cast<const uint8_t *>(positive_extended_frame_.data()), positive_extended_frame_.size()) << "\",\n"
        << "  \"programming_tx_monotonic_ns\": " << programming_tx_ns_ << ",\n"
        << "  \"first_extended_before_power_wake\": " << (first_extended_before_power_wake_ ? "true" : "false") << ",\n"
        << "  \"power_wake_complete_monotonic_ns\": " << power_wake_complete_ns_ << ",\n"
        << "  \"programming_after_50_03_ms\": " << ((programming_tx_ns_ - positive_extended_ns_) / 1e6) << ",\n"
        << "  \"verdict\": \"programming_request_sent_after_exact_50_03\"\n"
        << "}\n";
    out.close();
    if (std::rename(tmp.c_str(), TSS3_NATIVE_CATCH_PATH) != 0) {
      LOGE("failed to publish TSS3 native catch marker: errno=%d", errno);
      return false;
    }
    return true;
  }

  void notify_watcher() const {
    const int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) return;
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    fcntl(fd, F_SETFL, O_NONBLOCK);
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", TSS3_NATIVE_NOTIFY_PATH);
    const char notification = 1;
    if (sendto(fd, &notification, sizeof(notification), 0, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
      LOGW("TSS3 oracle native catch notification unavailable: errno=%d", errno);
    }
    close(fd);
  }

  std::string fingerprint_;
  Params params_;
  State state_ = State::DISARMED;
  bool enabled_ = false;
  bool param_initialized_ = false;
  bool ignition_initialized_ = false;
  bool last_ignition_ = false;
  uint64_t last_param_check_ns_ = 0;
  uint64_t last_state_check_ns_ = 0;
  uint64_t wake_ns_ = 0;
  uint64_t last_native_rx_ns_ = 0;
  uint64_t ignition_ns_ = 0;
  uint64_t first_extended_tx_ns_ = 0;
  uint64_t positive_extended_ns_ = 0;
  std::string positive_extended_frame_;
  uint64_t programming_tx_ns_ = 0;
  uint64_t next_extended_tx_ns_ = 0;
  bool first_extended_before_power_wake_ = false;
  uint64_t power_wake_complete_ns_ = 0;
};
}  // namespace

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
  std::unique_ptr<SubSocket> subscriber(SubSocket::create("sendcan", false, services.at("sendcan").queue_size));
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
      panda->can_send(event.getSendcan());
    } else {
      LOGE("sendcan too old to send: %" PRIu64 ", %" PRIu64, nanos_since_boot(), event.getLogMonoTime());
    }
  }
}

void can_recv(Panda *panda, PubMaster *pm, Tss3OracleStartupCatcher *startup_catcher) {
  static std::vector<can_frame> raw_can_data;
  {
    raw_can_data.clear();
    bool comms_healthy = panda->can_receive(raw_can_data);
    if (startup_catcher != nullptr) startup_catcher->process_rx(panda, raw_can_data);

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
  ps.setIgnitionLine((health.flags_pkt & HEALTH_FLAG_IGNITION_LINE) != 0U);
  ps.setIgnitionCan((health.flags_pkt & HEALTH_FLAG_IGNITION_CAN) != 0U);
  ps.setControlsAllowed((health.flags_pkt & HEALTH_FLAG_CONTROLS_ALLOWED) != 0U);
  ps.setTxBufferOverflow(health.tx_buffer_overflow_pkt);
  ps.setRxBufferOverflow(health.rx_buffer_overflow_pkt);
  ps.setPandaType(hw_type);
  ps.setSafetyModel(cereal::CarParams::SafetyModel(health.safety_mode_pkt));
  ps.setSafetyParam(health.safety_param_pkt);
  ps.setFaultStatus(cereal::PandaState::FaultStatus(health.fault_status_pkt));
  ps.setPowerSaveEnabled((health.flags_pkt & HEALTH_FLAG_POWER_SAVE_ENABLED) != 0U);
  ps.setHeartbeatLost((health.flags_pkt & HEALTH_FLAG_HEARTBEAT_LOST) != 0U);
  ps.setAlternativeExperience(health.alternative_experience_pkt);
  ps.setHarnessStatus(cereal::PandaState::HarnessStatus(health.car_harness_status_pkt));
  ps.setInterruptLoad(health.interrupt_load_pkt / 255.0f);
  ps.setFanPower(health.fan_power);
  ps.setSafetyRxChecksInvalid((health.flags_pkt & HEALTH_FLAG_SAFETY_RX_CHECKS_INVALID) != 0U);
  ps.setNmiReset((health.flags_pkt & HEALTH_FLAG_NMI_RESET) != 0U);
  ps.setHardfaultReset((health.flags_pkt & HEALTH_FLAG_HARDFAULT_RESET) != 0U);
  ps.setSpiErrorCount(health.spi_error_count_pkt);
  ps.setSbu1Voltage(health.sbu1_voltage_mV / 1000.0f);
  ps.setSbu2Voltage(health.sbu2_voltage_mV / 1000.0f);
  ps.setSoundOutputLevel(health.sound_output_level_pkt);
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

std::optional<bool> send_panda_states(PubMaster *pm, Panda *panda, bool is_onroad, bool spoofing_started, bool startup_catcher_active) {
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
    health.flags_pkt |= HEALTH_FLAG_IGNITION_LINE;
  }

  bool ignition_local = (health.flags_pkt & (HEALTH_FLAG_IGNITION_LINE | HEALTH_FLAG_IGNITION_CAN)) != 0U;

  // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
  if (health.safety_mode_pkt == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
    panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
  }

  bool power_save_desired = !ignition_local;
  if (((health.flags_pkt & HEALTH_FLAG_POWER_SAVE_ENABLED) != 0U) != power_save_desired) {
    panda->set_power_saving(power_save_desired);
  }

  // set safety mode to NO_OUTPUT when car is off or we're not onroad. ELM327 is an alternative if we want to leverage athenad/connect
  bool should_close_relay = !ignition_local || !is_onroad;
  if (!startup_catcher_active && should_close_relay && (health.safety_mode_pkt != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
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
  // bits above the last fault type would leave unset entries, which read as relayMalfunction
  std::bitset<sizeof(health.faults_pkt) * 8> fault_bits(health.faults_pkt & ((2U << size_t(cereal::PandaState::FaultType::SAFETY_REDUNDANCY)) - 1U));
  auto faults = ps.initFaults(fault_bits.count());

  size_t j = 0;
  for (size_t f = size_t(cereal::PandaState::FaultType::RELAY_MALFUNCTION);
       f <= size_t(cereal::PandaState::FaultType::SAFETY_REDUNDANCY); f++) {
    if (fault_bits.test(f)) {
      faults.set(j, cereal::PandaState::FaultType(f));
      j++;
    }
  }

  pm->send("pandaStates", msg);
  return ignition_local;
}

void send_peripheral_state(Panda *panda, PubMaster *pm) {
  auto health_opt = panda->get_state();
  if (!health_opt) {
    return;
  }

  // build msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  evt.setValid(panda->comms_healthy());

  auto ps = evt.initPeripheralState();
  ps.setPandaType(panda->hw_type);

  health_t health = *health_opt;
  ps.setVoltage(health.voltage_pkt);
  ps.setCurrent(health.current_pkt);

  uint16_t fan_speed_rpm = panda->get_fan_speed();
  ps.setFanSpeedRpm(fan_speed_rpm);

  pm->send("peripheralState", msg);
}

void process_panda_state(Panda *panda, PubMaster *pm, bool engaged, bool is_onroad, bool spoofing_started, bool startup_catcher_active) {
  auto ignition_opt = send_panda_states(pm, panda, is_onroad, spoofing_started, startup_catcher_active);
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

void process_peripheral_state(Panda *panda, PubMaster *pm, bool no_fan_control, bool is_onroad) {
  static Params params;
  static SubMaster sm({"deviceState", "cabinCameraState"});

  static uint64_t last_cabin_camera_t = 0;
  static uint16_t prev_fan_speed = 999;
  static int ir_pwr = 0;
  static int prev_ir_pwr = 999;
  static uint32_t prev_frame_id = UINT32_MAX;
  static bool driver_view = false;
  static bool not_car = false;
  static bool not_car_checked = false;

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

    if (sm.updated("cabinCameraState")) {
      auto event = sm["cabinCameraState"];
      int cur_integ_lines = event.getCabinCameraState().getIntegLines();

      // reset the filter when camerad restarts
      if (event.getCabinCameraState().getFrameId() < prev_frame_id) {
        integ_lines_filter.reset(0);
        integ_lines_filter_driver_view.reset(0);
        driver_view = params.getBool("IsDriverViewEnabled");
      }
      prev_frame_id = event.getCabinCameraState().getFrameId();

      cur_integ_lines = (driver_view ? integ_lines_filter_driver_view : integ_lines_filter).update(cur_integ_lines);
      last_cabin_camera_t = event.getLogMonoTime();

      if (cur_integ_lines <= CUTOFF_IL) {
        ir_pwr = 0;
      } else if (cur_integ_lines > SATURATE_IL) {
        ir_pwr = 100;
      } else {
        ir_pwr = 100 * (cur_integ_lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL);
      }
    }

    // Disable IR on input timeout or when requested offroad.
    if (nanos_since_boot() - last_cabin_camera_t > 1e9 || (!is_onroad && params.getBool("DisableDriverCameraIR"))) {
      ir_pwr = 0;
    }

    // turn off IR leds if body
    if (!not_car_checked && is_onroad) {
      std::string cp_bytes = params.get("CarParams");
      if (cp_bytes.size() > 0) {
        AlignedBuffer aligned_buf;
        capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
        cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
        not_car = CP.getNotCar();
        not_car_checked = true;
      }
    }
    if (not_car) {
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

bool pandad_run(Panda *panda) {
  const bool no_fan_control = getenv("NO_FAN_CONTROL") != nullptr;
  const bool spoofing_started = getenv("STARTED") != nullptr;
  const bool fake_send = getenv("FAKESEND") != nullptr;

  // Start helper thread for event-driven sendcan.
  std::thread send_thread(can_send_thread, panda, fake_send);

  RateKeeper rk("pandad", 100);
  SubMaster sm({"selfdriveState", "deviceState"});
  PubMaster pm({"can", "pandaStates", "peripheralState"});
  PandaSafety panda_safety(panda);
  Tss3OracleStartupCatcher tss3_startup_catcher;
  bool engaged = false;
  bool is_onroad = false;
  bool oracle_handoff = false;

  // Main loop: receive CAN first, then process lower priority panda and peripheral state.
  while (!do_exit && check_connected(panda)) {
    tss3_startup_catcher.update(panda);
    can_recv(panda, &pm, &tss3_startup_catcher);
    if (tss3_startup_catcher.caught()) {
      // The warm uploader is already connected passively and the catch marker
      // has been published. Stop all native Panda traffic immediately so the
      // managed wrapper can grant its cooperative lease without waiting for a
      // second signal-driven shutdown round trip.
      oracle_handoff = true;
      do_exit = true;
      break;
    }

    // Process peripheral state at 20 Hz
    if (rk.frame() % 5 == 0) {
      process_peripheral_state(panda, &pm, no_fan_control, is_onroad);
    }

    // Process panda state at 10 Hz
    if (rk.frame() % 10 == 0) {
      sm.update(0);
      engaged = sm.allAliveAndValid({"selfdriveState"}) && sm["selfdriveState"].getSelfdriveState().getEnabled();
      if (sm.updated("deviceState")) {
        is_onroad = sm["deviceState"].getDeviceState().getStarted();
      }
      process_panda_state(panda, &pm, engaged, is_onroad, spoofing_started, tss3_startup_catcher.active());
      if (!tss3_startup_catcher.active()) panda_safety.configureSafetyMode(is_onroad);
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
  if (!oracle_handoff && is_onroad && !engaged) {
    if (panda->connected()) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
  }

  send_thread.join();
  return oracle_handoff;
}

int pandad_main_thread(std::string serial) {
  if (serial.empty()) {
    auto serials = Panda::list();

    if (serials.empty()) {
      LOGW("no pandas found, exiting");
      return 0;
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

  bool oracle_handoff = false;
  if (!do_exit) {
    LOGW("connected to panda");
    oracle_handoff = pandad_run(panda);
  }

  delete panda;
  return oracle_handoff ? TSS3_ORACLE_HANDOFF_EXIT_CODE : 0;
}
