#include "selfdrive/boardd/usbdevice.h"

#include <cassert>
#include <map>
#include <memory>

#include "selfdrive/common/swaglog.h"

USBContext::USBContext() {
  int err = libusb_init(&context);
  if (err != 0) {
    LOGE("libusb initialization error %d", err);
    assert(0);
  }
#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(context, 3);
#endif
}

USBContext::~USBContext() {
  libusb_exit(context);
}

USBDeviceList::USBDeviceList(libusb_context *ctx) {
  num_devices = libusb_get_device_list(ctx, &dev_list);
}

USBDeviceList::~USBDeviceList() {
  if (dev_list) libusb_free_device_list(dev_list, 1);
}

libusb_device_handle *USBDeviceList::open(const std::string &serial) {
  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc = {};
    int ret = libusb_get_device_descriptor(dev_list[i], &desc);
    if (ret < 0 || desc.idVendor != USB_VID || desc.idProduct != USB_PID) continue;

    libusb_device_handle *handle = nullptr;
    if (libusb_open(dev_list[i], &handle) == 0) {
      unsigned char s[256] = {'\0'};
      libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, s, std::size(s) - 1);
      if (serial.empty() || serial == (char *)s) {
        return handle;
      }
      libusb_close(handle);
    }
  }
  return nullptr;
}

int USBDeviceList::size() {
  int cnt = 0;
  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc = {};
    int ret = libusb_get_device_descriptor(dev_list[i], &desc);
    cnt += ret >= 0 && desc.idVendor == USB_VID && desc.idProduct == USB_PID;
  }
  return cnt;
}

bool USBDevice::open(const std::string &serial) {
  dev_handle = USBDeviceList(ctx.context).open(serial);
  if (!dev_handle) return false;

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  if (libusb_set_configuration(dev_handle, 1) != 0 ||
      libusb_claim_interface(dev_handle, 0) != 0) {
    return false;
  }

  return true;
}

USBDevice::~USBDevice() {
  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }
}

int USBDevice::control_transfer(libusb_endpoint_direction dir, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, uint8_t *data, uint16_t length, uint32_t timeout) {
  std::lock_guard lk(usb_lock);

  int err = LIBUSB_ERROR_NO_DEVICE;
  const uint8_t bmRequestType = dir | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;
  while (connected) {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, length, timeout);
    if (err >= 0) break;

    handle_usb_issue(err, __func__);
  }
  return err;
}

int USBDevice::bulk_read(uint8_t endpoint, uint8_t *data, int length, uint32_t timeout) {
  std::lock_guard lk(usb_lock);

  int transferred = 0;
  while (connected) {
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == 0) break;

    if (err == LIBUSB_ERROR_TIMEOUT) {
      break;  // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else {
      handle_usb_issue(err, __func__);
    }
  }
  return transferred;
}

int USBDevice::bulk_write(uint8_t endpoint, uint8_t *data, int length, uint32_t timeout) {
  std::lock_guard lk(usb_lock);

  int transferred = 0;
  while (connected) {
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == 0 && transferred == length) break;

    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else {
      handle_usb_issue(err, __func__);
    }
  }
  return transferred;
}

void USBDevice::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}
