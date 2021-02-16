#include <stdexcept>
#include <cassert>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "common/swaglog.h"
#include "common/gpio.h"
#include "common/util.h"
#include "messaging.hpp"
#include "panda.h"
#include "panda_flashing.hpp"

void panda_set_power(bool power) {
#ifdef QCOM2
  int err = 0;

  err += gpio_init(GPIO_STM_RST_N, true);
  err += gpio_init(GPIO_STM_BOOT0, true);

  err += gpio_set(GPIO_STM_RST_N, true);
  err += gpio_set(GPIO_STM_BOOT0, false);

  util::sleep_for(100); // 100 ms

  err += gpio_set(GPIO_STM_RST_N, !power);
  assert(err == 0);
#endif
}

#define LIBUSB_ENDPOINT_DIR_MASK 0x80
#define LIBUSB_ENDPOINT_IN 0x80
#define LIBUSB_ENDPOINT_OUT 0x00

#define REQUEST_IN 192

PandaComm::PandaComm(uint16_t vid, uint16_t pid) {
  int err = libusb_init(&ctx);
  if (err != 0) { goto fail; }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(ctx, 3);
#endif

  dev_handle = libusb_open_device_with_vid_pid(ctx, vid, pid);
  if (dev_handle == NULL) { goto fail; }
  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }
  std::cout<<"Opened device with vid:pid "<< std::hex <<vid<<":"<<std::hex<<pid<<std::endl;
  return;

fail:
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

PandaComm::~PandaComm() {
  std::lock_guard lk(usb_lock);
  cleanup();
  connected = false;
}

void PandaComm::cleanup() {
  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }
}

int PandaComm::control_read(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  std::lock_guard lk(usb_lock);
  return libusb_control_transfer(dev_handle, (request_type & ~LIBUSB_ENDPOINT_DIR_MASK) | LIBUSB_ENDPOINT_IN, bRequest, wValue, wIndex, data, wLength, timeout);
}

int PandaComm::control_write(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  std::lock_guard lk(usb_lock);
  return libusb_control_transfer(dev_handle, (request_type & ~LIBUSB_ENDPOINT_DIR_MASK) | LIBUSB_ENDPOINT_OUT, bRequest, wValue, wIndex, data, wLength, timeout);
}

int PandaComm::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int PandaComm::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int PandaComm::usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);
  do {
    // Try sending can messages. If the receive buffer on the panda is full it will NAK
    // and libusb will try again. After 5ms, it will time out. We will drop the messages.
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  } while(err != 0 && connected);

  return transferred;
}

int PandaComm::usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);

  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

    if (err == LIBUSB_ERROR_TIMEOUT) {
      break; // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }

  } while(err != 0 && connected);

  return transferred;
}

void PandaComm::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}


DynamicPanda::DynamicPanda() {
  reconnect();
}

// Attempts to connect once
void DynamicPanda::connect() {
  // Find the first panda device in the list of devices
  pandaExists = false;

  int err = 0;
  libusb_context* ctx;
  err = libusb_init(&ctx);
  assert(err == 0);
  libusb_device **list = NULL;
  int count = 0;
  count = libusb_get_device_list(ctx, &list);
  assert(count > 0);
  std::cout<<"Looking for panda"<<std::endl;
  for (int idx = 0; idx < count; idx++) {
    libusb_device *device = list[idx];
    libusb_device_descriptor desc;
    err = libusb_get_device_descriptor(device, &desc);
    assert(err == 0);
    uint16_t vid = desc.idVendor;
    uint16_t pid = desc.idProduct;
    if (vid == 0x0483 && pid == 0xdf11) { // Panda in DFU 
      std::cout<<"Found panda in DFU mode, should not occur"<<std::endl;
      pandaExists = false;
      bootstub = false;
      libusb_free_device_list(list, count);
      libusb_exit(ctx);
      throw std::runtime_error("Found DFU panda...");
    }else if (vid == 0xbbaa && pid == 0xddcc) { //Normal panda running the firmware
      c = new PandaComm(0xbbaa, 0xddcc);
      std::cout<<"Found panda in a good state, exiting"<<std::endl;
      pandaExists = true;
      bootstub = false;
      break;
    }else if (vid == 0xbbaa && pid == 0xddee) {
      c = new PandaComm(0xbbaa, 0xddee);
      std::cout<<"Found panda in bootstub mode, some more work to do"<<std::endl;
      pandaExists = true;
      bootstub = true;
      break;
    }
  }
  libusb_free_device_list(list, count);
  libusb_exit(ctx);
  if (!pandaExists) {
    std::cout<<"Dynamic panda cannot find any non DFU panda"<<std::endl;
    throw std::runtime_error("Panda not found...");
  }
}

std::string DynamicPanda::get_version() {
  std::vector<uint8_t> fw_sig_buf(0x40);
  c->control_read(REQUEST_IN, 0xd6, 0, 0, &fw_sig_buf[0], 0x40);
  return std::string(fw_sig_buf.begin(), fw_sig_buf.end());
}

std::string DynamicPanda::get_signature() {
  std::vector<uint8_t> fw_sig_buf(128);
  c->control_read(REQUEST_IN, 0xd3, 0, 0, &fw_sig_buf[0], 64);
  c->control_read(REQUEST_IN, 0xd4, 0, 0, &fw_sig_buf[64], 64);
  return std::string(fw_sig_buf.begin(), fw_sig_buf.end());
}

void DynamicPanda::flash(std::string fw_fn) {
  std::cout<<std::endl<<"Firmware string: "<<fw_fn<<std::endl;
  std::cout<<"flash: main version is "<<get_version()<<std::endl;
  if (!bootstub) {
    reset(true, false);
  }
  assert(bootstub);

  std::string code = util::read_file(fw_fn);
  unsigned char code_data[code.length()];
  for(int i = 0 ; i < code.length() ; i++) {
    code_data[i]=code[i];
  }
  std::cout<<"Firmware code length: "<<code.length()<<std::endl;
  std::cout<<"flash: bootstub version is "<<get_version()<<std::endl;

  // confirm flashed is present
  std::vector<uint8_t> buf(12);
  c->control_read(REQUEST_IN, 0xb0, 0, 0, &buf[0], 12);
  assert(buf[4] == 0xde && buf[5] == 0xad && buf[6] == 0xd0 && buf[7] == 0x0d);

  //unlock flash
  std::cout<<"flash: unlocking"<<std::endl;
  c->control_write(REQUEST_IN, 0xb1, 0, 0, nullptr, 0);

  // erase sectors 1-3
  std::cout<<"flash: erasing"<<std::endl;
  for (int i = 1 ; i < 4 ; i++) {
    c->control_write(REQUEST_IN, 0xb2, i, 0, nullptr, 0);
  }

  // flash over EP2
  int STEP = 0x10;
  std::cout<<"flash: flashing"<<std::endl;
  for(int i = 0 ; i < code.length() ; i += STEP) {
    c->usb_bulk_write(2, code_data + i, STEP);
  }
  //reset
  std::cout<<"flash: resetting"<<std::endl;
  try{
    c->control_write(REQUEST_IN, 0xd8, 0, 0, nullptr, 0);
  } catch (std::runtime_error &e) {}

  reconnect();
}

void DynamicPanda::reconnect() {
  util::sleep_for(1000);
  for (int i = 0 ; i < 15 ; i++) {
    try{
     connect();
     return;
    }catch(std::runtime_error &e) {
      std::cout<<"reconnecting is taking "<<i+1<<" seconds..."<<std::endl;
      PandaComm* dfu;
      try{
        dfu = new PandaComm(0x0483, 0xdf11);
        dfu_recover(dfu);
      }catch(std::runtime_error &e) {}
      delete(dfu);
      util::sleep_for(1000);
    }
  }
  throw std::runtime_error("Reconnecting timed out");
}

void DynamicPanda::reset(bool enter_bootstub, bool enter_bootloader) {
  try {
    if (enter_bootloader) {
      c->control_write(REQUEST_IN, 0xd1, 0, 0, nullptr, 0);
    } else if (enter_bootstub) {
      c->control_write(REQUEST_IN, 0xd1, 1, 0, nullptr, 0);
    } else {
      c->control_write(REQUEST_IN, 0xd8, 0, 0, nullptr, 0);
    }
  } catch(std::runtime_error &e) {}

  if (!enter_bootloader) {
    reconnect();
  }
}

void DynamicPanda::recover() {
  reset(true, false);
  reset(false, true);
  while(true) {
    std::cout<<"Waiting for DFU"<<std::endl;
    util::sleep_for(100);
    try {
      PandaComm dfuPanda(0x0483, 0xdf11); // Throws exception if Panda is not in DFU mdoe
      dfu_recover(&dfuPanda);
      break;
    } catch(std::runtime_error &e) {}
  }

  do {
    try {
    connect();
    } catch(std::runtime_error &e) {};
    
    std::cout<<"Looking for panda"<<std::endl;
    util::sleep_for(1000);
  } while(!pandaExists);

  flash(get_firmware_fn());
}

DynamicPanda::~DynamicPanda() {
  delete(c);
}


Panda::Panda() {
  // init communication
  c = new PandaComm(0xbbaa, 0xddcc);

  hw_type = get_hw_type();

  is_pigeon =
    (hw_type == cereal::HealthData::PandaType::GREY_PANDA) ||
    (hw_type == cereal::HealthData::PandaType::BLACK_PANDA) ||
    (hw_type == cereal::HealthData::PandaType::UNO) ||
    (hw_type == cereal::HealthData::PandaType::DOS);
  has_rtc = (hw_type == cereal::HealthData::PandaType::UNO) ||
    (hw_type == cereal::HealthData::PandaType::DOS);

}

Panda::~Panda() {
  cleanup();
}

void Panda::cleanup() {
  delete(c);
}

bool Panda::connected() {
  return c->connected;
}

int Panda::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  return c->usb_write(bRequest, wValue, wIndex, timeout);
}

int Panda::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  return c->usb_read(bRequest, wValue, wIndex, data, wLength, timeout);
}

int Panda::usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return c->usb_bulk_write(endpoint, data, length, timeout);
}

int Panda::usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return c->usb_bulk_read(endpoint, data, length, timeout);
}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param) {
  usb_write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_unsafe_mode(uint16_t unsafe_mode) {
  usb_write(0xdf, unsafe_mode, 0);
}

cereal::HealthData::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  usb_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::HealthData::PandaType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time) {
  // tm struct has year defined as years since 1900
  usb_write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  usb_write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  usb_write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // usb_write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  usb_write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  usb_write(0xa6, (uint16_t)sys_time.tm_min, 0);
  usb_write(0xa7, (uint16_t)sys_time.tm_sec, 0);
}

struct tm Panda::get_rtc() {
  struct __attribute__((packed)) timestamp_t {
    uint16_t year; // Starts at 0
    uint8_t month;
    uint8_t day;
    uint8_t weekday;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
  } rtc_time = {0};

  usb_read(0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time));

  struct tm new_time = { 0 };
  new_time.tm_year = rtc_time.year - 1900; // tm struct has year defined as years since 1900
  new_time.tm_mon  = rtc_time.month - 1;
  new_time.tm_mday = rtc_time.day;
  new_time.tm_hour = rtc_time.hour;
  new_time.tm_min  = rtc_time.minute;
  new_time.tm_sec  = rtc_time.second;

  return new_time;
}

void Panda::set_fan_speed(uint16_t fan_speed) {
  usb_write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed() {
  uint16_t fan_speed_rpm = 0;
  usb_read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  usb_write(0xb0, ir_pwr, 0);
}

health_t Panda::get_health() {
  health_t health {0};
  usb_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return health;
}

void Panda::set_loopback(bool loopback) {
  usb_write(0xe5, loopback, 0);
}

std::optional<std::vector<uint8_t>> Panda::get_firmware_version() {
  std::vector<uint8_t> fw_sig_buf(128);
  int read_1 = usb_read(0xd3, 0, 0, &fw_sig_buf[0], 64);
  int read_2 = usb_read(0xd4, 0, 0, &fw_sig_buf[64], 64);
  return ((read_1 == 64) && (read_2 == 64)) ? std::make_optional(fw_sig_buf) : std::nullopt;
}

std::optional<std::string> Panda::get_serial() {
  char serial_buf[17] = {'\0'};
  int err = usb_read(0xd0, 0, 0, (uint8_t*)serial_buf, 16);
  return err >= 0 ? std::make_optional(serial_buf) : std::nullopt;
}

void Panda::set_power_saving(bool power_saving) {
  usb_write(0xe7, power_saving, 0);
}

void Panda::set_usb_power_mode(cereal::HealthData::UsbPowerMode power_mode) {
  usb_write(0xe6, (uint16_t)power_mode, 0);
}

void Panda::send_heartbeat() {
  usb_write(0xf3, 1, 0);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  static std::vector<uint32_t> send;
  const int msg_count = can_data_list.size();

  send.resize(msg_count*0x10);

  for (int i = 0; i < msg_count; i++) {
    auto cmsg = can_data_list[i];
    if (cmsg.getAddress() >= 0x800) { // extended
      send[i*4] = (cmsg.getAddress() << 3) | 5;
    } else { // normal
      send[i*4] = (cmsg.getAddress() << 21) | 1;
    }
    auto can_data = cmsg.getDat();
    assert(can_data.size() <= 8);
    send[i*4+1] = can_data.size() | (cmsg.getSrc() << 4);
    memcpy(&send[i*4+2], can_data.begin(), can_data.size());
  }

  usb_bulk_write(3, (unsigned char*)send.data(), send.size(), 5);
}

int Panda::can_receive(kj::Array<capnp::word>& out_buf) {
  uint32_t data[RECV_SIZE/4];
  int recv = usb_bulk_read(0x81, (unsigned char*)data, RECV_SIZE);

  // Not sure if this can happen
  if (recv < 0) recv = 0;

  if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

  size_t num_msg = recv / 0x10;
  MessageBuilder msg;
  auto canData = msg.initEvent().initCan(num_msg);

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
  out_buf = capnp::messageToFlatArray(msg);
  return recv;
}
