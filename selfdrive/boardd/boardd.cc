#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <assert.h>
#include <pthread.h>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"

#include "common/util.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "messaging.hpp"

#include <algorithm>
#include <bitset>

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

#define MAX_IR_POWER 0.5f
#define MIN_IR_POWER 0.0f
#define CUTOFF_IL 200
#define SATURATE_IL 1600
#define NIBBLE_TO_HEX(n) ((n) < 10 ? (n) + '0' : ((n) - 10) + 'a')
#define VOLTAGE_K 0.091  // LPF gain for 5s tau (dt/tau / (dt/tau + 1))

namespace {

volatile sig_atomic_t do_exit = 0;

struct __attribute__((packed)) timestamp_t {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t weekday;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
};

libusb_context *ctx = NULL;
libusb_device_handle *dev_handle = NULL;
pthread_mutex_t usb_lock;

bool spoofing_started = false;
bool fake_send = false;
bool loopback_can = false;
cereal::HealthData::HwType hw_type = cereal::HealthData::HwType::UNKNOWN;
bool is_pigeon = false;
float voltage_f = 12.5;  // filtered voltage
uint32_t no_ignition_cnt = 0;
bool connected_once = false;
bool ignition_last = false;

#ifndef __x86_64__
const uint32_t NO_IGNITION_CNT_MAX = 2 * 60 * 60 * 30;  // turn off charge after 30 hrs
const float VBATT_START_CHARGING = 11.5;
const float VBATT_PAUSE_CHARGING = 11.0;
#endif

bool safety_setter_thread_initialized = false;
pthread_t safety_setter_thread_handle;

bool pigeon_thread_initialized = false;
pthread_t pigeon_thread_handle;

bool pigeon_needs_init;

void pigeon_init();
void *pigeon_thread(void *crap);

void *safety_setter_thread(void *s) {
  // diagnostic only is the default, needed for VIN query
  pthread_mutex_lock(&usb_lock);
  libusb_control_transfer(dev_handle, 0x40, 0xdc, (uint16_t)(cereal::CarParams::SafetyModel::ELM327), 0, NULL, 0, TIMEOUT);
  pthread_mutex_unlock(&usb_lock);

  // switch to SILENT when CarVin param is read
  while (1) {
    if (do_exit) return NULL;
    std::vector<char> value_vin = read_db_bytes("CarVin");
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
  pthread_mutex_lock(&usb_lock);
  libusb_control_transfer(dev_handle, 0x40, 0xdc, (uint16_t)(cereal::CarParams::SafetyModel::NO_OUTPUT), 0, NULL, 0, TIMEOUT);
  pthread_mutex_unlock(&usb_lock);

  std::vector<char> params;
  LOGW("waiting for params to set safety model");
  while (1) {
    if (do_exit) return NULL;

    params = read_db_bytes("CarParams");
    if (params.size() > 0) break;
    usleep(100*1000);
  }
  LOGW("got %d bytes CarParams", params.size());

  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((params.size() / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), params.data(), params.size());

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  int safety_model = int(car_params.getSafetyModel());
  auto safety_param = car_params.getSafetyParam();
  LOGW("setting safety model: %d with param %d", safety_model, safety_param);

  pthread_mutex_lock(&usb_lock);

  // set in the mutex to avoid race
  safety_setter_thread_initialized = false;

  libusb_control_transfer(dev_handle, 0x40, 0xdc, safety_model, safety_param, NULL, 0, TIMEOUT);

  pthread_mutex_unlock(&usb_lock);

  return NULL;
}

// must be called before threads or with mutex
bool usb_connect() {
  int err, err2;
  unsigned char hw_query[1] = {0};
  unsigned char fw_sig_buf[128];
  unsigned char fw_sig_hex_buf[16];
  unsigned char serial_buf[16];
  const char *serial;
  int serial_sz = 0;

  ignition_last = false;

  if (dev_handle != NULL){
    libusb_close(dev_handle);
    dev_handle = NULL;
  }

  dev_handle = libusb_open_device_with_vid_pid(ctx, 0xbbaa, 0xddcc);
  if (dev_handle == NULL) { goto fail; }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  if (loopback_can) {
    libusb_control_transfer(dev_handle, 0xc0, 0xe5, 1, 0, NULL, 0, TIMEOUT);
  }

  // get panda fw
  err = libusb_control_transfer(dev_handle, 0xc0, 0xd3, 0, 0, fw_sig_buf, 64, TIMEOUT);
  err2 = libusb_control_transfer(dev_handle, 0xc0, 0xd4, 0, 0, fw_sig_buf + 64, 64, TIMEOUT);
  if ((err == 64) && (err2 == 64)) {
    printf("FW signature read\n");
    write_db_value("PandaFirmware", (const char *)fw_sig_buf, 128);

    for (size_t i = 0; i < 8; i++){
      fw_sig_hex_buf[2*i] = NIBBLE_TO_HEX(fw_sig_buf[i] >> 4);
      fw_sig_hex_buf[2*i+1] = NIBBLE_TO_HEX(fw_sig_buf[i] & 0xF);
    }
    write_db_value("PandaFirmwareHex", (const char *)fw_sig_hex_buf, 16);
  }
  else { goto fail; }

  // get panda serial
  err = libusb_control_transfer(dev_handle, 0xc0, 0xd0, 0, 0, serial_buf, 16, TIMEOUT);

  if (err > 0) {
    serial = (const char *)serial_buf;
    serial_sz = strnlen(serial, err);
    write_db_value("PandaDongleId", serial, serial_sz);
    printf("panda serial: %.*s\n", serial_sz, serial);
  }
  else { goto fail; }

  // power on charging, only the first time. Panda can also change mode and it causes a brief disconneciton
#ifndef __x86_64__
  if (!connected_once) {
    libusb_control_transfer(dev_handle, 0xc0, 0xe6, (uint16_t)(cereal::HealthData::UsbPowerMode::CDP), 0, NULL, 0, TIMEOUT);
  }
#endif
  connected_once = true;

  libusb_control_transfer(dev_handle, 0xc0, 0xc1, 0, 0, hw_query, 1, TIMEOUT);

  hw_type = (cereal::HealthData::HwType)(hw_query[0]);
  is_pigeon = (hw_type == cereal::HealthData::HwType::GREY_PANDA) ||
              (hw_type == cereal::HealthData::HwType::BLACK_PANDA) ||
              (hw_type == cereal::HealthData::HwType::UNO);
  if (is_pigeon) {
    LOGW("panda with gps detected");
    pigeon_needs_init = true;
    if (!pigeon_thread_initialized) {
      err = pthread_create(&pigeon_thread_handle, NULL, pigeon_thread, NULL);
      assert(err == 0);
      pigeon_thread_initialized = true;
    }
  }

  if (hw_type == cereal::HealthData::HwType::UNO){
    // Get time from system
    time_t rawtime;
    time(&rawtime);

    struct tm sys_time;
    gmtime_r(&rawtime, &sys_time);

    // Get time from RTC
    timestamp_t rtc_time;
    libusb_control_transfer(dev_handle, 0xc0, 0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time), TIMEOUT);

    //printf("System: %d-%d-%d\t%d:%d:%d\n", 1900 + sys_time.tm_year, 1 + sys_time.tm_mon, sys_time.tm_mday, sys_time.tm_hour, sys_time.tm_min, sys_time.tm_sec);
    //printf("RTC: %d-%d-%d\t%d:%d:%d\n", rtc_time.year, rtc_time.month, rtc_time.day, rtc_time.hour, rtc_time.minute, rtc_time.second);

    // Update system time from RTC if it looks off, and RTC time is good
    if (1900 + sys_time.tm_year < 2019 && rtc_time.year >= 2019){
      LOGE("System time wrong, setting from RTC");

      struct tm new_time = { 0 };
      new_time.tm_year = rtc_time.year - 1900;
      new_time.tm_mon  = rtc_time.month - 1;
      new_time.tm_mday = rtc_time.day;
      new_time.tm_hour = rtc_time.hour;
      new_time.tm_min  = rtc_time.minute;
      new_time.tm_sec  = rtc_time.second;

      setenv("TZ","UTC",1);
      const struct timeval tv = {mktime(&new_time), 0};
      settimeofday(&tv, 0);
    }
  }

  return true;
fail:
  return false;
}

// must be called before threads or with mutex
void usb_retry_connect() {
  LOG("attempting to connect");
  while (!usb_connect()) { usleep(100*1000); }
  LOGW("connected to board");
}

void handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == -4) {
    LOGE("lost connection");
    usb_retry_connect();
  }
  // TODO: check other errors, is simply retrying okay?
}

void can_recv(PubMaster &pm) {
  int err;
  uint32_t data[RECV_SIZE/4];
  int recv;

  uint64_t start_time = nanos_since_boot();

  // do recv
  pthread_mutex_lock(&usb_lock);

  do {
    err = libusb_bulk_transfer(dev_handle, 0x81, (uint8_t*)data, RECV_SIZE, &recv, TIMEOUT);
    if (err != 0) { handle_usb_issue(err, __func__); }
    if (err == -8) { LOGE_100("overflow got 0x%x", recv); };

    // timeout is okay to exit, recv still happened
    if (err == -7) { break; }
  } while(err != 0);

  pthread_mutex_unlock(&usb_lock);

  // return if length is 0
  if (recv <= 0) {
    return;
  } else if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(start_time);
  size_t num_msg = recv / 0x10;
  auto canData = event.initCan(num_msg);

  // populate message
  for (int i = 0; i < num_msg; i++) {
    if (data[i*4] & 4) {
      // extended
      canData[i].setAddress(data[i*4] >> 3);
      //printf("got extended: %x\n", data[i*4] >> 3);
    } else {
      // normal
      canData[i].setAddress(data[i*4] >> 21);
    }
    canData[i].setBusTime(data[i*4+1] >> 16);
    int len = data[i*4+1]&0xF;
    canData[i].setDat(kj::arrayPtr((uint8_t*)&data[i*4+2], len));
    canData[i].setSrc((data[i*4+1] >> 4) & 0xff);
  }

  pm.send("can", msg);
}

void can_health(PubMaster &pm) {
  int cnt;
  int err;

  // copied from panda/board/main.c
  struct __attribute__((packed)) health {
    uint32_t uptime;
    uint32_t voltage;
    uint32_t current;
    uint32_t can_rx_errs;
    uint32_t can_send_errs;
    uint32_t can_fwd_errs;
    uint32_t gmlan_send_errs;
    uint32_t faults;
    uint8_t ignition_line;
    uint8_t ignition_can;
    uint8_t controls_allowed;
    uint8_t gas_interceptor_detected;
    uint8_t car_harness_status;
    uint8_t usb_power_mode;
    uint8_t safety_model;
    uint8_t fault_status;
    uint8_t power_save_enabled;
  } health;

  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto healthData = event.initHealth();

  bool received = false;

  // recv from board
  if (dev_handle != NULL) {
    pthread_mutex_lock(&usb_lock);
    cnt = libusb_control_transfer(dev_handle, 0xc0, 0xd2, 0, 0, (unsigned char*)&health, sizeof(health), TIMEOUT);
    pthread_mutex_unlock(&usb_lock);

    received = (cnt == sizeof(health));
  }

  // No panda connected, send empty health packet
  if (!received){
    healthData.setHwType(cereal::HealthData::HwType::UNKNOWN);
    pm.send("health", msg);
    return;
  }

  if (spoofing_started) {
    health.ignition_line = 1;
  }

  voltage_f = VOLTAGE_K * (health.voltage / 1000.0) + (1.0 - VOLTAGE_K) * voltage_f;  // LPF

  // Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent and there is only one other CAN node
  if (health.safety_model == (uint8_t)(cereal::CarParams::SafetyModel::SILENT)) {
    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0x40, 0xdc, (uint16_t)(cereal::CarParams::SafetyModel::NO_OUTPUT), 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);
  }

  bool ignition = ((health.ignition_line != 0) || (health.ignition_can != 0));

  if (ignition) {
    no_ignition_cnt = 0;
  } else {
    no_ignition_cnt += 1;
  }

#ifndef __x86_64__
  bool cdp_mode = health.usb_power_mode == (uint8_t)(cereal::HealthData::UsbPowerMode::CDP);
  bool no_ignition_exp = no_ignition_cnt > NO_IGNITION_CNT_MAX;
  if ((no_ignition_exp || (voltage_f < VBATT_PAUSE_CHARGING)) && cdp_mode && !ignition) {
    std::vector<char> disable_power_down = read_db_bytes("DisablePowerDown");
    if (disable_power_down.size() != 1 || disable_power_down[0] != '1') {
      printf("TURN OFF CHARGING!\n");
      pthread_mutex_lock(&usb_lock);
      libusb_control_transfer(dev_handle, 0xc0, 0xe6, (uint16_t)(cereal::HealthData::UsbPowerMode::CLIENT), 0, NULL, 0, TIMEOUT);
      pthread_mutex_unlock(&usb_lock);
      printf("POWER DOWN DEVICE\n");
      system("service call power 17 i32 0 i32 1");
    }
  }
  if (!no_ignition_exp && (voltage_f > VBATT_START_CHARGING) && !cdp_mode) {
    printf("TURN ON CHARGING!\n");
    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0xc0, 0xe6, (uint16_t)(cereal::HealthData::UsbPowerMode::CDP), 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);
  }
  // set power save state enabled when car is off and viceversa when it's on
  if (ignition && (health.power_save_enabled == 1)) {
    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0xc0, 0xe7, 0, 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);
  }
  if (!ignition && (health.power_save_enabled == 0)) {
    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0xc0, 0xe7, 1, 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);
  }
  // set safety mode to NO_OUTPUT when car is off. ELM327 is an alternative if we want to leverage athenad/connect
  if (!ignition && (health.safety_model != (uint8_t)(cereal::CarParams::SafetyModel::NO_OUTPUT))) {
    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0x40, 0xdc, (uint16_t)(cereal::CarParams::SafetyModel::NO_OUTPUT), 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);
  }
#endif

  // clear VIN, CarParams, and set new safety on car start
  if (ignition && !ignition_last) {
    int result = delete_db_value("CarVin");
    assert((result == 0) || (result == ERR_NO_VALUE));
    result = delete_db_value("CarParams");
    assert((result == 0) || (result == ERR_NO_VALUE));

    if (!safety_setter_thread_initialized) {
      err = pthread_create(&safety_setter_thread_handle, NULL, safety_setter_thread, NULL);
      assert(err == 0);
      safety_setter_thread_initialized = true;
    }
  }

  // Get fan RPM
  uint16_t fan_speed_rpm = 0;

  pthread_mutex_lock(&usb_lock);
  libusb_control_transfer(dev_handle, 0xc0, 0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm), TIMEOUT);
  pthread_mutex_unlock(&usb_lock);

  // Write to rtc once per minute when no ignition present
  if ((hw_type == cereal::HealthData::HwType::UNO) && !ignition && (no_ignition_cnt % 120 == 1)){
    // Get time from system
    time_t rawtime;
    time(&rawtime);

    struct tm sys_time;
    gmtime_r(&rawtime, &sys_time);

    // Write time to RTC if it looks reasonable
    if (1900 + sys_time.tm_year >= 2019){
      pthread_mutex_lock(&usb_lock);
      libusb_control_transfer(dev_handle, 0x40, 0xa1, (uint16_t)(1900 + sys_time.tm_year), 0, NULL, 0, TIMEOUT);
      libusb_control_transfer(dev_handle, 0x40, 0xa2, (uint16_t)(1 + sys_time.tm_mon), 0, NULL, 0, TIMEOUT);
      libusb_control_transfer(dev_handle, 0x40, 0xa3, (uint16_t)sys_time.tm_mday, 0, NULL, 0, TIMEOUT);
      // libusb_control_transfer(dev_handle, 0x40, 0xa4, (uint16_t)(1 + sys_time.tm_wday), 0, NULL, 0, TIMEOUT);
      libusb_control_transfer(dev_handle, 0x40, 0xa5, (uint16_t)sys_time.tm_hour, 0, NULL, 0, TIMEOUT);
      libusb_control_transfer(dev_handle, 0x40, 0xa6, (uint16_t)sys_time.tm_min, 0, NULL, 0, TIMEOUT);
      libusb_control_transfer(dev_handle, 0x40, 0xa7, (uint16_t)sys_time.tm_sec, 0, NULL, 0, TIMEOUT);
      pthread_mutex_unlock(&usb_lock);
    }
  }

  ignition_last = ignition;

  // set fields
  healthData.setUptime(health.uptime);
  healthData.setVoltage(health.voltage);
  healthData.setCurrent(health.current);
  healthData.setIgnitionLine(health.ignition_line);
  healthData.setIgnitionCan(health.ignition_can);
  healthData.setControlsAllowed(health.controls_allowed);
  healthData.setGasInterceptorDetected(health.gas_interceptor_detected);
  healthData.setHasGps(is_pigeon);
  healthData.setCanRxErrs(health.can_rx_errs);
  healthData.setCanSendErrs(health.can_send_errs);
  healthData.setCanFwdErrs(health.can_fwd_errs);
  healthData.setGmlanSendErrs(health.gmlan_send_errs);
  healthData.setHwType(hw_type);
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
       f <= size_t(cereal::HealthData::FaultType::INTERRUPT_RATE_KLINE_INIT); f++){
    if (fault_bits.test(f)) {
      faults.set(i, cereal::HealthData::FaultType(f));
      i++;
    }
  }
  // send to health
  pm.send("health", msg);

  // send heartbeat back to panda
  pthread_mutex_lock(&usb_lock);
  libusb_control_transfer(dev_handle, 0x40, 0xf3, 1, 0, NULL, 0, TIMEOUT);
  pthread_mutex_unlock(&usb_lock);
}


void can_send(cereal::Event::Reader &event) {
  int err;
  // recv from sendcan
  if (nanos_since_boot() - event.getLogMonoTime() > 1e9) {
    //Older than 1 second. Dont send.
    return;
  }

  auto can_data_list = event.getSendcan();
  int msg_count = can_data_list.size();

  uint32_t *send = (uint32_t*)malloc(msg_count*0x10);
  memset(send, 0, msg_count*0x10);

  for (int i = 0; i < msg_count; i++) {
    auto cmsg = can_data_list[i];
    if (cmsg.getAddress() >= 0x800) {
      // extended
      send[i*4] = (cmsg.getAddress() << 3) | 5;
    } else {
      // normal
      send[i*4] = (cmsg.getAddress() << 21) | 1;
    }
    auto can_data = cmsg.getDat();
    assert(can_data.size() <= 8);
    send[i*4+1] = can_data.size() | (cmsg.getSrc() << 4);
    memcpy(&send[i*4+2], can_data.begin(), can_data.size());
  }

  // send to board
  int sent;
  pthread_mutex_lock(&usb_lock);

  if (!fake_send) {
    do {
      // Try sending can messages. If the receive buffer on the panda is full it will NAK
      // and libusb will try again. After 5ms, it will time out. We will drop the messages.
      err = libusb_bulk_transfer(dev_handle, 3, (uint8_t*)send, msg_count*0x10, &sent, 5);
      if (err == LIBUSB_ERROR_TIMEOUT) {
        LOGW("Transmit buffer full");
        break;
      } else if (err != 0 || msg_count*0x10 != sent) {
        LOGW("Error");
        handle_usb_issue(err, __func__);
      }
    } while(err != 0);
  }

  pthread_mutex_unlock(&usb_lock);

  // done
  free(send);
}

// **** threads ****

void *can_send_thread(void *crap) {
  LOGD("start send thread");

  Context * context = Context::create();
  SubSocket * subscriber = SubSocket::create(context, "sendcan");
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  // run as fast as messages come in
  while (!do_exit) {
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
    can_send(event);
    delete msg;
  }

  delete subscriber;
  delete context;

  return NULL;
}

void *can_recv_thread(void *crap) {
  LOGD("start recv thread");

  // can = 8006
  PubMaster pm({"can"});

  // run at 100hz
  const uint64_t dt = 10000000ULL;
  uint64_t next_frame_time = nanos_since_boot() + dt;

  while (!do_exit) {
    can_recv(pm);

    uint64_t cur_time = nanos_since_boot();
    int64_t remaining = next_frame_time - cur_time;
    if (remaining > 0){
      useconds_t sleep = remaining / 1000;
      usleep(sleep);
    } else {
      LOGW("missed cycle");
      next_frame_time = cur_time;
    }

    next_frame_time += dt;
  }
  return NULL;
}

void *can_health_thread(void *crap) {
  LOGD("start health thread");
  // health = 8011
  PubMaster pm({"health"});

  // run at 2hz
  while (!do_exit) {
    can_health(pm);
    usleep(500*1000);
  }

  return NULL;
}

void *hardware_control_thread(void *crap) {
  LOGD("start hardware control thread");
  SubMaster sm({"thermal", "frontFrame"});

  // Wait for hardware type to be set.
  while (hw_type == cereal::HealthData::HwType::UNKNOWN){
    usleep(100*1000);
  }
  // Only control fan speed on UNO
  if (hw_type != cereal::HealthData::HwType::UNO) return NULL;


  uint64_t last_front_frame_t = 0;
  uint16_t prev_fan_speed = 999;
  uint16_t ir_pwr = 0;
  uint16_t prev_ir_pwr = 999;
  unsigned int cnt = 0;

  while (!do_exit) {
    cnt++;
    sm.update(1000);
    if (sm.updated("thermal")){
      uint16_t fan_speed = sm["thermal"].getThermal().getFanSpeed();
      if (fan_speed != prev_fan_speed || cnt % 100 == 0){
        pthread_mutex_lock(&usb_lock);
        libusb_control_transfer(dev_handle, 0x40, 0xb1, fan_speed, 0, NULL, 0, TIMEOUT);
        pthread_mutex_unlock(&usb_lock);

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
      pthread_mutex_lock(&usb_lock);
      libusb_control_transfer(dev_handle, 0x40, 0xb0, ir_pwr, 0, NULL, 0, TIMEOUT);
      pthread_mutex_unlock(&usb_lock);
      prev_ir_pwr = ir_pwr;
    }

  }

  return NULL;
}

#define pigeon_send(x) _pigeon_send(x, sizeof(x)-1)

void hexdump(unsigned char *d, int l) __attribute__((unused));
void hexdump(unsigned char *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i!=0 && i%0x10 == 0) printf("\n");
    printf("%2.2X ", d[i]);
  }
  printf("\n");
}

void _pigeon_send(const char *dat, int len) {
  int sent;
  unsigned char a[0x20];
  int err;
  a[0] = 1;
  for (int i=0; i<len; i+=0x20) {
    int ll = std::min(0x20, len-i);
    memcpy(&a[1], &dat[i], ll);
    pthread_mutex_lock(&usb_lock);
    err = libusb_bulk_transfer(dev_handle, 2, a, ll+1, &sent, TIMEOUT);
    if (err < 0) { handle_usb_issue(err, __func__); }
    /*assert(err == 0);
    assert(sent == ll+1);*/
    //hexdump(a, ll+1);
    pthread_mutex_unlock(&usb_lock);
  }
}

void pigeon_set_power(int power) {
  pthread_mutex_lock(&usb_lock);
  int err = libusb_control_transfer(dev_handle, 0xc0, 0xd9, power, 0, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  pthread_mutex_unlock(&usb_lock);
}

void pigeon_set_baud(int baud) {
  int err;
  pthread_mutex_lock(&usb_lock);
  err = libusb_control_transfer(dev_handle, 0xc0, 0xe2, 1, 0, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  err = libusb_control_transfer(dev_handle, 0xc0, 0xe4, 1, baud/300, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  pthread_mutex_unlock(&usb_lock);
}

void pigeon_init() {
  usleep(1000*1000);
  LOGW("panda GPS start");

  // power off pigeon
  pigeon_set_power(0);
  usleep(100*1000);

  // 9600 baud at init
  pigeon_set_baud(9600);

  // power on pigeon
  pigeon_set_power(1);
  usleep(500*1000);

  // baud rate upping
  pigeon_send("\x24\x50\x55\x42\x58\x2C\x34\x31\x2C\x31\x2C\x30\x30\x30\x37\x2C\x30\x30\x30\x33\x2C\x34\x36\x30\x38\x30\x30\x2C\x30\x2A\x31\x35\x0D\x0A");
  usleep(100*1000);

  // set baud rate to 460800
  pigeon_set_baud(460800);
  usleep(100*1000);

  // init from ubloxd
  // To generate this data, run test/ubloxd.py with the print statements enabled in the write function in panda/python/serial.py
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F");
  pigeon_send("\xB5\x62\x06\x3E\x00\x00\x44\xD2");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85");
  pigeon_send("\xB5\x62\x06\x00\x00\x00\x06\x18");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x01\x08\x22");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x02\x09\x23");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x03\x0A\x24");
  pigeon_send("\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10");
  pigeon_send("\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63");
  pigeon_send("\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37");
  pigeon_send("\xB5\x62\x06\x24\x00\x00\x2A\x84");
  pigeon_send("\xB5\x62\x06\x23\x00\x00\x29\x81");
  pigeon_send("\xB5\x62\x06\x1E\x00\x00\x24\x72");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x0A\x09\x01\x1E\x70");

  LOGW("panda GPS on");
}

static void pigeon_publish_raw(PubMaster &pm, unsigned char *dat, int alen) {
  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto ublox_raw = event.initUbloxRaw(alen);
  memcpy(ublox_raw.begin(), dat, alen);

  pm.send("ubloxRaw", msg);
}


void *pigeon_thread(void *crap) {
  // ubloxRaw = 8042
  PubMaster pm({"ubloxRaw"});

  // run at ~100hz
  unsigned char dat[0x1000];
  uint64_t cnt = 0;
  while (!do_exit) {
    if (pigeon_needs_init) {
      pigeon_needs_init = false;
      pigeon_init();
    }
    int alen = 0;
    while (alen < 0xfc0) {
      pthread_mutex_lock(&usb_lock);
      int len = libusb_control_transfer(dev_handle, 0xc0, 0xe0, 1, 0, dat+alen, 0x40, TIMEOUT);
      if (len < 0) { handle_usb_issue(len, __func__); }
      pthread_mutex_unlock(&usb_lock);
      if (len <= 0) break;

      //printf("got %d\n", len);
      alen += len;
    }
    if (alen > 0) {
      if (dat[0] == (char)0x00){
        LOGW("received invalid ublox message, resetting panda GPS");
        pigeon_init();
      } else {
        pigeon_publish_raw(pm, dat, alen);
      }
    }

    // 10ms
    usleep(10*1000);
    cnt++;
  }
  return NULL;
}

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

  if (getenv("BOARDD_LOOPBACK")){
    loopback_can = true;
  }

  err = pthread_mutex_init(&usb_lock, NULL);
  assert(err == 0);

  // init libusb
  err = libusb_init(&ctx);
  assert(err == 0);

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(ctx, 3);
#endif

  pthread_t can_health_thread_handle;
  err = pthread_create(&can_health_thread_handle, NULL,
                       can_health_thread, NULL);
  assert(err == 0);

  // connect to the board
  pthread_mutex_lock(&usb_lock);
  usb_retry_connect();
  pthread_mutex_unlock(&usb_lock);

  // create threads
  pthread_t can_send_thread_handle;
  err = pthread_create(&can_send_thread_handle, NULL,
                       can_send_thread, NULL);
  assert(err == 0);

  pthread_t can_recv_thread_handle;
  err = pthread_create(&can_recv_thread_handle, NULL,
                       can_recv_thread, NULL);
  assert(err == 0);

  pthread_t hardware_control_thread_handle;
  err = pthread_create(&hardware_control_thread_handle, NULL,
                       hardware_control_thread, NULL);
  assert(err == 0);

  // join threads

  err = pthread_join(can_recv_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_send_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_health_thread_handle, NULL);
  assert(err == 0);

  //while (!do_exit) usleep(1000);

  // destruct libusb

  libusb_close(dev_handle);
  libusb_exit(ctx);
}
