#include "common/gpio.h"

#ifdef __APPLE__
int gpio_init(int pin_nr, bool output) {
  return 0;
}

int gpio_set(int pin_nr, bool high) {
  return 0;
}

int gpiochip_get_ro_value_fd(const char* consumer_label, int gpiochiop_id, int pin_nr) {
  return 0;
}

#else

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <linux/gpio.h>
#include <sys/ioctl.h>

#include "common/util.h"
#include "common/swaglog.h"

int gpio_init(int pin_nr, bool output) {
  char pin_dir_path[50];
  int pin_dir_path_len = snprintf(pin_dir_path, sizeof(pin_dir_path),
                           "/sys/class/gpio/gpio%d/direction", pin_nr);
  if(pin_dir_path_len <= 0) {
    return -1;
  }
  const char *value = output ? "out" : "in";
  return util::write_file(pin_dir_path, (void*)value, strlen(value));
}

int gpio_set(int pin_nr, bool high) {
  char pin_val_path[50];
  int pin_val_path_len = snprintf(pin_val_path, sizeof(pin_val_path),
                           "/sys/class/gpio/gpio%d/value", pin_nr);
  if(pin_val_path_len <= 0) {
    return -1;
  }
  return util::write_file(pin_val_path, (void*)(high ? "1" : "0"), 1);
}

int gpiochip_get_ro_value_fd(const char* consumer_label, int gpiochiop_id, int pin_nr) {

  // Assumed that all interrupt pins are unexported and rights are given to
  // read from gpiochip0.
  std::string gpiochip_path = "/dev/gpiochip" + std::to_string(gpiochiop_id);
  int fd = open(gpiochip_path.c_str(), O_RDONLY);
  if (fd < 0) {
    LOGE("Error opening gpiochip0 fd")
    return -1;
  }

  // Setup event
  struct gpioevent_request rq;
  rq.lineoffset = pin_nr;
  rq.handleflags = GPIOHANDLE_REQUEST_INPUT;

  /* Requesting both edges as the data ready pulse from the lsm6ds sensor is
     very short(75us) and is mostly detected as falling edge instead of rising.
     So if it is detected as rising the following falling edge is skipped. */
  rq.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES;

  strncpy(rq.consumer_label, consumer_label, std::size(rq.consumer_label) - 1);
  int ret = ioctl(fd, GPIO_GET_LINEEVENT_IOCTL, &rq);
  if (ret == -1) {
    LOGE("Unable to get line event from ioctl : %s", strerror(errno));
    close(fd);
    return -1;
  }

  close(fd);
  return rq.fd;
}

#endif
