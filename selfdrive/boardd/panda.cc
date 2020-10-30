#include <stdexcept>
#include <cassert>
#include <iostream>

#include <unistd.h>

#include "common/swaglog.h"
#include "common/gpio.h"

#include "panda.h"

void panda_set_power(bool power){
#ifdef QCOM2
  int err = 0;
  err += gpio_init(GPIO_STM_RST_N, true);
  err += gpio_init(GPIO_STM_BOOT0, true);

  err += gpio_set(GPIO_STM_RST_N, false);
  err += gpio_set(GPIO_STM_BOOT0, false);

  usleep(100*1000); // 100 ms

  err += gpio_set(GPIO_STM_RST_N, power);
  assert(err == 0);
#endif
}

Panda::Panda(){
  int err;

  err = pthread_mutex_init(&usb_lock, NULL);
  if (err != 0) { goto fail; }

  // init libusb
  err = libusb_init(&ctx);
  if (err != 0) { goto fail; }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(ctx, 3);
#endif

  dev_handle = libusb_open_device_with_vid_pid(ctx, 0xbbaa, 0xddcc);
  if (dev_handle == NULL) { goto fail; }

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  hw_type = get_hw_type();
  is_pigeon =
    (hw_type == cereal::HealthData::HwType::GREY_PANDA) ||
    (hw_type == cereal::HealthData::HwType::BLACK_PANDA) ||
    (hw_type == cereal::HealthData::HwType::UNO) ||
    (hw_type == cereal::HealthData::HwType::DOS);
  has_rtc = (hw_type == cereal::HealthData::HwType::UNO) ||
    (hw_type == cereal::HealthData::HwType::DOS);

  return;

fail:
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

Panda::~Panda(){
  pthread_mutex_lock(&usb_lock);
  cleanup();
  connected = false;
  pthread_mutex_unlock(&usb_lock);
}

void Panda::cleanup(){
  if (dev_handle){
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }
}

void Panda::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}

int Panda::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected){
    return LIBUSB_ERROR_NO_DEVICE;
  }

  pthread_mutex_lock(&usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  pthread_mutex_unlock(&usb_lock);

  return err;
}

int Panda::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  pthread_mutex_lock(&usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);
  pthread_mutex_unlock(&usb_lock);

  return err;
}

int Panda::usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected){
    return 0;
  }

  pthread_mutex_lock(&usb_lock);
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

  pthread_mutex_unlock(&usb_lock);
  return transferred;
}

int Panda::usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected){
    return 0;
  }

  pthread_mutex_lock(&usb_lock);

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

  pthread_mutex_unlock(&usb_lock);

  return transferred;
}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param){
  usb_write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_unsafe_mode(uint16_t unsafe_mode) {
  usb_write(0xdf, unsafe_mode, 0);
}

cereal::HealthData::HwType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  usb_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::HealthData::HwType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time){
  // tm struct has year defined as years since 1900
  usb_write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  usb_write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  usb_write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // usb_write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  usb_write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  usb_write(0xa6, (uint16_t)sys_time.tm_min, 0);
  usb_write(0xa7, (uint16_t)sys_time.tm_sec, 0);
}

struct tm Panda::get_rtc(){
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

void Panda::set_fan_speed(uint16_t fan_speed){
  usb_write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed(){
  uint16_t fan_speed_rpm = 0;
  usb_read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  usb_write(0xb0, ir_pwr, 0);
}

health_t Panda::get_health(){
  health_t health {0};
  usb_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return health;
}

void Panda::set_loopback(bool loopback){
  usb_write(0xe5, loopback, 0);
}

const char* Panda::get_firmware_version(){
  const char* fw_sig_buf = new char[128]();

  int read_1 = usb_read(0xd3, 0, 0, (unsigned char*)fw_sig_buf, 64);
  int read_2 = usb_read(0xd4, 0, 0, (unsigned char*)fw_sig_buf + 64, 64);

  if ((read_1 == 64) && (read_2 == 64)) {
    return fw_sig_buf;
  }

  delete[] fw_sig_buf;
  return NULL;
}

const char* Panda::get_serial(){
  const char* serial_buf = new char[16]();

  int err = usb_read(0xd0, 0, 0, (unsigned char*)serial_buf, 16);

  if (err >= 0) {
    return serial_buf;
  }

  delete[] serial_buf;
  return NULL;
}

void Panda::set_power_saving(bool power_saving){
  usb_write(0xe7, power_saving, 0);
}

void Panda::set_usb_power_mode(cereal::HealthData::UsbPowerMode power_mode){
  usb_write(0xe6, (uint16_t)power_mode, 0);
}

void Panda::send_heartbeat(){
  usb_write(0xf3, 1, 0);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list){
  int msg_count = can_data_list.size();

  uint32_t *send = new uint32_t[msg_count*0x10]();

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

  usb_bulk_write(3, (unsigned char*)send, msg_count*0x10, 5);

  delete[] send;
}

int Panda::can_receive(cereal::Event::Builder &event){
  uint32_t data[RECV_SIZE/4];
  int recv = usb_bulk_read(0x81, (unsigned char*)data, RECV_SIZE);

  // Not sure if this can happen
  if (recv < 0) recv = 0;

  if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

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

  return recv;
}
