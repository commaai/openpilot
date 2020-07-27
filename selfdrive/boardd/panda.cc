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
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  pthread_mutex_lock(&usb_lock);
  if (!connected){
    pthread_mutex_unlock(&usb_lock);
    return -1;
  }

  int err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
  if (err < 0) handle_usb_issue(err, __func__);

  pthread_mutex_unlock(&usb_lock);

  return err;
}

int Panda::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  pthread_mutex_lock(&usb_lock);
  if (!connected){
    pthread_mutex_unlock(&usb_lock);
    return -1;
  }

  int err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
  if (err < 0) handle_usb_issue(err, __func__);

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
  } while(err != 0);

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

  } while(err != 0);

  pthread_mutex_unlock(&usb_lock);

  return transferred;
}

int Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param){
  return usb_write(0xdc, (uint16_t)safety_model, safety_param);
}
