#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <ctime>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <thread>
#include <atomic>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"

#include "common/util.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "messaging.hpp"

#include "panda.h"
#include "pigeon.h"


#define MAX_IR_POWER 0.5f
#define MIN_IR_POWER 0.0f
#define CUTOFF_IL 200
#define SATURATE_IL 1600
#define NIBBLE_TO_HEX(n) ((n) < 10 ? (n) + '0' : ((n) - 10) + 'a')

Panda * panda = NULL;
std::atomic<bool> safety_setter_thread_running(false);
volatile sig_atomic_t do_exit = 0;
bool spoofing_started = false;
bool fake_send = false;
bool connected_once = false;
bool ignition = false;

struct tm get_time(){
  time_t rawtime;
  time(&rawtime);

  struct tm sys_time;
  gmtime_r(&rawtime, &sys_time);

  return sys_time;
}

bool time_valid(struct tm sys_time){
  return 1900 + sys_time.tm_year >= 2019;
}

void safety_setter_thread() {
  LOGD("Starting safety setter thread");
  // diagnostic only is the default, needed for VIN query
  panda->set_safety_model(cereal::CarParams::SafetyModel::ELM327);

  // switch to SILENT when CarVin param is read
  while (1) {
    if (do_exit || !panda->connected){
      safety_setter_thread_running = false;
      return;
    };

    std::vector<char> value_vin = Params().read_db_bytes("CarVin");
    if (value_vin.size() > 0) {
      // sanity check VIN format
      assert(value_vin.size() == 17);
      std::string str_vin(value_vin.begin(), value_vin.end());
      LOGW("got CarVin %s", str_vin.c_str());
      break;
    }
    usleep(100*1000);
  }

  // VIN query done, stop listening to OBDII
  panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);

  std::vector<char> params;
  LOGW("waiting for params to set safety model");
  while (1) {
    if (do_exit || !panda->connected){
      safety_setter_thread_running = false;
      return;
    };

    params = Params().read_db_bytes("CarParams");
    if (params.size() > 0) break;
    usleep(100*1000);
  }
  LOGW("got %d bytes CarParams", params.size());

  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((params.size() / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), params.data(), params.size());

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();
  cereal::CarParams::SafetyModel safety_model = car_params.getSafetyModel();

  panda->set_unsafe_mode(0);  // see safety_declarations.h for allowed values

  auto safety_param = car_params.getSafetyParam();
  LOGW("setting safety model: %d with param %d", (int)safety_model, safety_param);

  panda->set_safety_model(safety_model, safety_param);

  safety_setter_thread_running = false;
}


bool usb_connect() {
  try {
    assert(panda == NULL);
    panda = new Panda();
  } catch (std::exception &e) {
    return false;
  }

  Params params = Params();

  if (getenv("BOARDD_LOOPBACK")) {
    panda->set_loopback(true);
  }

  const char *fw_sig_buf = panda->get_firmware_version();
  if (fw_sig_buf){
    params.write_db_value("PandaFirmware", fw_sig_buf, 128);

    // Convert to hex for offroad
    char fw_sig_hex_buf[16] = {0};
    for (size_t i = 0; i < 8; i++){
      fw_sig_hex_buf[2*i] = NIBBLE_TO_HEX((uint8_t)fw_sig_buf[i] >> 4);
      fw_sig_hex_buf[2*i+1] = NIBBLE_TO_HEX((uint8_t)fw_sig_buf[i] & 0xF);
    }

    params.write_db_value("PandaFirmwareHex", fw_sig_hex_buf, 16);
    LOGW("fw signature: %.*s", 16, fw_sig_hex_buf);

    delete[] fw_sig_buf;
  } else { return false; }

  // get panda serial
  const char *serial_buf = panda->get_serial();
  if (serial_buf) {
    size_t serial_sz = strnlen(serial_buf, 16);

    params.write_db_value("PandaDongleId", serial_buf, serial_sz);
    LOGW("panda serial: %.*s", serial_sz, serial_buf);

    delete[] serial_buf;
  } else { return false; }

  // power on charging, only the first time. Panda can also change mode and it causes a brief disconneciton
#ifndef __x86_64__
  if (!connected_once) {
    panda->set_usb_power_mode(cereal::HealthData::UsbPowerMode::CDP);
  }
#endif

  if (panda->has_rtc){
    struct tm sys_time = get_time();
    struct tm rtc_time = panda->get_rtc();

    if (!time_valid(sys_time) && time_valid(rtc_time)) {
      LOGE("System time wrong, setting from RTC");

      setenv("TZ","UTC",1);
      const struct timeval tv = {mktime(&rtc_time), 0};
      settimeofday(&tv, 0);
    }
  }

  connected_once = true;
  return true;
}

// must be called before threads or with mutex
void usb_retry_connect() {
  LOGW("attempting to connect");
  while (!usb_connect()) { usleep(100*1000); }
  LOGW("connected to board");
}

void can_recv(PubMaster &pm) {
  // create message
  MessageBuilder msg;
  auto event = msg.initEvent();
  panda->can_receive(event);
  pm.send("can", msg);
}

void can_send_thread() {
  LOGD("start send thread");

  Context * context = Context::create();
  SubSocket * subscriber = SubSocket::create(context, "sendcan");
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  // run as fast as messages come in
  while (!do_exit && panda->connected) {
    Message * msg = subscriber->receive();

    if (!msg){
      if (errno == EINTR) {
        do_exit = true;
      }
      continue;
    }

    auto amsg = kj::heapArray<capnp::word>((msg->getSize() / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), msg->getData(), msg->getSize());

    capnp::FlatArrayMessageReader cmsg(amsg);
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    //Dont send if older than 1 second
    if (nanos_since_boot() - event.getLogMonoTime() < 1e9) {
      if (!fake_send){
        panda->can_send(event.getSendcan());
      }
    }

    delete msg;
  }

  delete subscriber;
  delete context;
}

void can_recv_thread() {
  LOGD("start recv thread");

  // can = 8006
  PubMaster pm({"can"});

  // run at 100hz
  const uint64_t dt = 10000000ULL;
  uint64_t next_frame_time = nanos_since_boot() + dt;

  while (!do_exit && panda->connected) {
    can_recv(pm);

    uint64_t cur_time = nanos_since_boot();
    int64_t remaining = next_frame_time - cur_time;
    if (remaining > 0){
      useconds_t sleep = remaining / 1000;
      usleep(sleep);
    } else {
      if (ignition){
        LOGW("missed cycles (%d) %lld", (int)-1*remaining/dt, remaining);
      }
      next_frame_time = cur_time;
    }

    next_frame_time += dt;
  }
}

void can_health_thread() {
  LOGD("start health thread");
  PubMaster pm({"health"});

  uint32_t no_ignition_cnt = 0;
  bool ignition_last = false;
  Params params = Params();

  // Broadcast empty health message when panda is not yet connected
  while (!panda){
    MessageBuilder msg;
    auto healthData  = msg.initEvent().initHealth();

    healthData.setHwType(cereal::HealthData::HwType::UNKNOWN);
    pm.send("health", msg);
    usleep(500*1000);
  }

  // run at 2hz
  while (!do_exit && panda->connected) {
    MessageBuilder msg;
    auto healthData = msg.initEvent().initHealth();

    health_t health = panda->get_health();

    if (spoofing_started) {
      health.ignition_line = 1;
    }

    // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
    if (health.safety_model == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }

    ignition = ((health.ignition_line != 0) || (health.ignition_can != 0));

    if (ignition) {
      no_ignition_cnt = 0;
    } else {
      no_ignition_cnt += 1;
    }

#ifndef __x86_64__
    bool power_save_desired = !ignition;
    if (health.power_save_enabled != power_save_desired){
      panda->set_power_saving(power_save_desired);
    }

    // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
    if (!ignition && (health.safety_model != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
      panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    }
#endif

    // clear VIN, CarParams, and set new safety on car start
    if (ignition && !ignition_last) {
      int result = params.delete_db_value("CarVin");
      assert((result == 0) || (result == ERR_NO_VALUE));
      result = params.delete_db_value("CarParams");
      assert((result == 0) || (result == ERR_NO_VALUE));

      if (!safety_setter_thread_running) {
        safety_setter_thread_running = true;
        std::thread(safety_setter_thread).detach();
      } else {
        LOGW("Safety setter thread already running");
      }
    }

    // Write to rtc once per minute when no ignition present
    if ((panda->has_rtc) && !ignition && (no_ignition_cnt % 120 == 1)){
      // Write time to RTC if it looks reasonable
      struct tm sys_time = get_time();
      if (time_valid(sys_time)){
        panda->set_rtc(sys_time);
      }
    }

    ignition_last = ignition;
    uint16_t fan_speed_rpm = panda->get_fan_speed();

    // set fields
    healthData.setUptime(health.uptime);
    healthData.setVoltage(health.voltage);
    healthData.setCurrent(health.current);
    healthData.setIgnitionLine(health.ignition_line);
    healthData.setIgnitionCan(health.ignition_can);
    healthData.setControlsAllowed(health.controls_allowed);
    healthData.setGasInterceptorDetected(health.gas_interceptor_detected);
    healthData.setHasGps(panda->is_pigeon);
    healthData.setCanRxErrs(health.can_rx_errs);
    healthData.setCanSendErrs(health.can_send_errs);
    healthData.setCanFwdErrs(health.can_fwd_errs);
    healthData.setGmlanSendErrs(health.gmlan_send_errs);
    healthData.setHwType(panda->hw_type);
    healthData.setUsbPowerMode(cereal::HealthData::UsbPowerMode(health.usb_power_mode));
    healthData.setSafetyModel(cereal::CarParams::SafetyModel(health.safety_model));
    healthData.setFanSpeedRpm(fan_speed_rpm);
    healthData.setFaultStatus(cereal::HealthData::FaultStatus(health.fault_status));
    healthData.setPowerSaveEnabled((bool)(health.power_save_enabled));

    // Convert faults bitset to capnp list
    std::bitset<sizeof(health.faults) * 8> fault_bits(health.faults);
    auto faults = healthData.initFaults(fault_bits.count());

    size_t i = 0;
    for (size_t f = size_t(cereal::HealthData::FaultType::RELAY_MALFUNCTION);
        f <= size_t(cereal::HealthData::FaultType::INTERRUPT_RATE_TIM9); f++){
      if (fault_bits.test(f)) {
        faults.set(i, cereal::HealthData::FaultType(f));
        i++;
      }
    }
    pm.send("health", msg);
    panda->send_heartbeat();
    usleep(500*1000);
  }
}

void hardware_control_thread() {
  LOGD("start hardware control thread");
  SubMaster sm({"thermal", "frontFrame"});

  uint64_t last_front_frame_t = 0;
  uint16_t prev_fan_speed = 999;
  uint16_t ir_pwr = 0;
  uint16_t prev_ir_pwr = 999;
#ifdef QCOM
  bool prev_charging_disabled = false;
#endif
  unsigned int cnt = 0;

  while (!do_exit && panda->connected) {
    cnt++;
    sm.update(1000); // TODO: what happens if EINTR is sent while in sm.update?

#ifdef QCOM
    if (sm.updated("thermal")){
      // Charging mode
      bool charging_disabled = sm["thermal"].getThermal().getChargingDisabled();
      if (charging_disabled != prev_charging_disabled){
        if (charging_disabled){
          panda->set_usb_power_mode(cereal::HealthData::UsbPowerMode::CLIENT);
          LOGW("TURN OFF CHARGING!\n");
        } else {
          panda->set_usb_power_mode(cereal::HealthData::UsbPowerMode::CDP);
          LOGW("TURN ON CHARGING!\n");
        }
        prev_charging_disabled = charging_disabled;
      }
    }
#endif

    // Other pandas don't have fan/IR to control
    if (panda->hw_type != cereal::HealthData::HwType::UNO && panda->hw_type != cereal::HealthData::HwType::DOS) continue;
    if (sm.updated("thermal")){
      // Fan speed
      uint16_t fan_speed = sm["thermal"].getThermal().getFanSpeed();
      if (fan_speed != prev_fan_speed || cnt % 100 == 0){
        panda->set_fan_speed(fan_speed);
        prev_fan_speed = fan_speed;
      }
    }
    if (sm.updated("frontFrame")){
      auto event = sm["frontFrame"];
      int cur_integ_lines = event.getFrontFrame().getIntegLines();
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
    if (cur_t - last_front_frame_t > 1e9){
      ir_pwr = 0;
    }

    if (ir_pwr != prev_ir_pwr || cnt % 100 == 0 || ir_pwr >= 50.0){
      panda->set_ir_pwr(ir_pwr);
      prev_ir_pwr = ir_pwr;
    }

  }
}

static void pigeon_publish_raw(PubMaster &pm, std::string dat) {
  // create message
  MessageBuilder msg;
  auto ublox_raw = msg.initEvent().initUbloxRaw(dat.length());
  memcpy(ublox_raw.begin(), dat.data(), dat.length());

  pm.send("ubloxRaw", msg);
}


void pigeon_thread() {
  if (!panda->is_pigeon){ return; };

  // ubloxRaw = 8042
  PubMaster pm({"ubloxRaw"});

#ifdef QCOM2
  Pigeon * pigeon = Pigeon::connect("/dev/ttyHS0");
#else
  Pigeon * pigeon = Pigeon::connect(panda);
#endif

  pigeon->init();

  while (!do_exit && panda->connected) {
    std::string recv = pigeon->receive();
    if (recv.length() > 0) {
      if (recv[0] == (char)0x00){
        LOGW("received invalid ublox message, resetting panda GPS");
        pigeon->init();
      } else {
        pigeon_publish_raw(pm, recv);
      }
    }

    // 10ms - 100 Hz
    usleep(10*1000);
  }

  delete pigeon;
}


int main() {
  int err;
  LOGW("starting boardd");

  // set process priority and affinity
  err = set_realtime_priority(54);
  LOG("set priority returns %d", err);
  err = set_core_affinity(3);
  LOG("set affinity returns %d", err);

  // check the environment
  if (getenv("STARTED")) {
    spoofing_started = true;
  }

  if (getenv("FAKESEND")) {
    fake_send = true;
  }

  panda_set_power(true);

  while (!do_exit){
    std::vector<std::thread> threads;
    threads.push_back(std::thread(can_health_thread));

    // connect to the board
    usb_retry_connect();

    threads.push_back(std::thread(can_send_thread));
    threads.push_back(std::thread(can_recv_thread));
    threads.push_back(std::thread(hardware_control_thread));
    threads.push_back(std::thread(pigeon_thread));

    for (auto &t : threads) t.join();

    delete panda;
    panda = NULL;
  }
}
