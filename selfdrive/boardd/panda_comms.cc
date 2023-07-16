#include "selfdrive/boardd/panda.h"

#include <cassert>
#include <stdexcept>

#include "common/swaglog.h"

#define PANDA_VENDOR_ID 0xBBAA
#define PANDA_PRODUCT_ID 0xDDCC

static int init_usb_ctx(libusb_context **context) {
  assert(context != nullptr);

  int err = libusb_init(context);
  if (err != 0) {
    LOGE("libusb initialization error");
    return err;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(*context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(*context, 3);
#endif

  return err;
}

struct usbDevice {
  usbDevice(libusb_context *ctx) {
    num_devices = libusb_get_device_list(ctx, &dev_list);
    if (num_devices < 0) LOGE("libusb can't get device list");
  }
  ~usbDevice() { libusb_free_device_list(dev_list, 1); }

  auto list(uint16_t vid = PANDA_VENDOR_ID, uint16_t pid = PANDA_PRODUCT_ID) {
    std::vector<std::pair<libusb_device *, std::string>> result;
    for (size_t i = 0; i < num_devices; ++i) {
      libusb_device_descriptor desc = {};
      libusb_device_handle *handle = nullptr;

      int ret = libusb_get_device_descriptor(dev_list[i], &desc);
      if (ret == 0 && desc.idVendor == vid && desc.idProduct == pid && libusb_open(dev_list[i], &handle) == 0) {
        unsigned char serial[256] = {'\0'};
        ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, serial, std::size(serial) - 1);
        if (ret >= 0) result.emplace_back(dev_list[i], std::string((char *)serial, ret));
        libusb_close(handle);
      }
    }
    return result;
  }

  libusb_device **dev_list = nullptr;
  ssize_t num_devices = 0;
};

PandaUsbHandle::PandaUsbHandle(std::string serial) : PandaCommsHandle(serial) {
  int err = init_usb_ctx(&ctx);
  if (err != 0) { goto fail; }

  // connect by serial
  for (auto &[dev, serial_name] : usbDevice(ctx).list()) {
    if (serial.empty() || serial == serial_name) {
      libusb_open(dev, &dev_handle);
      break;
    }
  }
  if (dev_handle == NULL) goto fail;

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  return;

fail:
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

PandaUsbHandle::~PandaUsbHandle() {
  std::lock_guard lk(hw_lock);
  cleanup();
  connected = false;
}

void PandaUsbHandle::cleanup() {
  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }
}

std::vector<std::string> PandaUsbHandle::list() {
  libusb_context *context = NULL;
  std::vector<std::string> serials;
  if (init_usb_ctx(&context) == 0) {
    for (auto &[_, serial] : usbDevice(context).list()) {
      serials.push_back(serial);
    }
  }
  return serials;
}

void PandaUsbHandle::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}

int PandaUsbHandle::control_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(hw_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int PandaUsbHandle::control_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(hw_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int PandaUsbHandle::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(hw_lock);
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

int PandaUsbHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(hw_lock);

  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

    if (err == LIBUSB_ERROR_TIMEOUT) {
      break; // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }

  } while(err != 0 && connected);

  return transferred;
}
