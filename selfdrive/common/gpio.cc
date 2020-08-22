#include "gpio.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

// We assume that all pins have already been exported on boot,
// and that we have permission to write to them.

int gpio_init(int pin_nr, bool output){
  int ret = 0;
  int fd = -1, tmp;

  char pin_dir_path[50];
  int pin_dir_path_len = snprintf(pin_dir_path, sizeof(pin_dir_path),
                           "/sys/class/gpio/gpio%d/direction", pin_nr);
  if(pin_dir_path_len <= 0){
    ret = -1;
    goto cleanup;
  }

  fd = open(pin_dir_path, O_WRONLY);
  if(fd == -1){
    ret = -1;
    goto cleanup;
  }
  if(output){
    tmp = write(fd, "out", 3);
    if(tmp != 3){
      ret = -1;
      goto cleanup;
    }
  } else {
    tmp = write(fd, "in", 2);
    if(tmp != 2){
      ret = -1;
      goto cleanup;
    }
  }

cleanup:
  if(fd >= 0){
    close(fd);
  }
  return ret;
}

int gpio_set(int pin_nr, bool high){
  int ret = 0;
  int fd = -1, tmp;

  char pin_val_path[50];
  int pin_val_path_len = snprintf(pin_val_path, sizeof(pin_val_path),
                           "/sys/class/gpio/gpio%d/value", pin_nr);
  if(pin_val_path_len <= 0){
    ret = -1;
    goto cleanup;
  }

  fd = open(pin_val_path, O_WRONLY);
  if(fd == -1){
    ret = -1;
    goto cleanup;
  }
  tmp = write(fd, high ? "1" : "0", 1);
  if(tmp != 1){
    ret = -1;
    goto cleanup;
  }

cleanup:
  if(fd >= 0){
    close(fd);
  }
  return ret;
}
