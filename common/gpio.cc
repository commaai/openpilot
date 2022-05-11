#include "common/gpio.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>

#include "common/util.h"

// We assume that all pins have already been exported on boot,
// and that we have permission to write to them.

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

int gpio_set_edge(int pin_nr, Edgetype etype) {
  char pin_dir_path[50];
  int pin_dir_path_len = snprintf(pin_dir_path, sizeof(pin_dir_path),
                           "/sys/class/gpio/gpio%d/edge", pin_nr);
  if(pin_dir_path_len <= 0) {
    return -1;
  }

  std::string value;
  switch(etype) {
      case Rising  : value = "rising"; break;
      case Falling : value = "falling"; break;
      case Both    : value = "both";  break;
      case None    : return 0;
      default      : return -1;
  }

  return util::write_file(pin_dir_path, (void*)value.c_str(), value.size());
}

int gpio_get_ro_value_fd(int pin_nr) {
  char pin_dir_path[50];

  int pin_dir_path_len = snprintf(pin_dir_path, sizeof(pin_dir_path),
                          "/sys/class/gpio/gpio%d/value", pin_nr);
  if(pin_dir_path_len <= 0) {
    return -1;
  }

  return open(pin_dir_path, O_RDONLY);
}
