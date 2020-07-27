#include <stdexcept>
#include <cassert>
#include <iostream>

#include "common/swaglog.h"

#include "panda.h"

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

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  hw_type = get_hw_type();
  is_pigeon =
    (hw_type == cereal::HealthData::HwType::GREY_PANDA) ||
    (hw_type == cereal::HealthData::HwType::BLACK_PANDA) ||
    (hw_type == cereal::HealthData::HwType::UNO);
  has_rtc = (hw_type == cereal::HealthData::HwType::UNO);

  return;

 fail:
  if (dev_handle){
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }

  throw std::runtime_error("Error connecting to panda");
}

Panda::~Panda(){
  pthread_mutex_lock(&usb_lock);

  if (dev_handle){
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }
  connected = -1;
  pthread_mutex_unlock(&usb_lock);
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
  int transferred;

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
  int transferred;

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
  } rtc_time;

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

health_t Panda::get_health(){
  health_t health {0};
  usb_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return health;
}

void Panda::set_loopback(bool loopback){
  usb_write(0xe5, loopback, 0);
}

const char* Panda::get_firmware_version(){
  const char* fw_sig_buf = new char[128];

  int read_1 = usb_read(0xd3, 0, 0, (unsigned char*)fw_sig_buf, 64);
  int read_2 = usb_read(0xd4, 0, 0, (unsigned char*)fw_sig_buf + 64, 64);

  if ((read_1 == 64) && (read_2 == 64)) {
    return fw_sig_buf;
  }

  delete[] fw_sig_buf;
  return NULL;
}
