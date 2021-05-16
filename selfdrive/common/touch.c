#include "selfdrive/common/touch.h"

#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/poll.h>
#include <unistd.h>

/* this macro is used to tell if "bit" is set in "array"
 * it selects a byte from the array, and does a boolean AND
 * operation with a byte that only has the relevant bit set.
 * eg. to check for the 12th bit, we do (array[1] & 1<<4)
 */
#define test_bit(bit, array)    (array[bit/8] & (1<<(bit%8)))

static int find_dev() {
  int err;

  int ret = -1;

  DIR *dir = opendir("/dev/input");
  assert(dir);
  struct dirent* de = NULL;
  while ((de = readdir(dir))) {
    if (strncmp(de->d_name, "event", 5)) continue;

    int fd = openat(dirfd(dir), de->d_name, O_RDONLY);
    assert(fd >= 0);

    unsigned char ev_bits[KEY_MAX / 8 + 1];
    err = ioctl(fd, EVIOCGBIT(EV_ABS, sizeof(ev_bits)), ev_bits);
    assert(err >= 0);

    if (test_bit(ABS_MT_POSITION_X, ev_bits) && test_bit(ABS_MT_POSITION_Y, ev_bits)) {
      ret = fd;
      break;
    }
    close(fd);
  }
  closedir(dir);

  return ret;
}

void touch_init(TouchState *s) {
  s->fd = find_dev();
  assert(s->fd >= 0);
}

int touch_poll(TouchState *s, int* out_x, int* out_y, int timeout) {
  assert(out_x && out_y);
  bool up = false;
  while (true) {
    struct pollfd polls[] = {{
      .fd = s->fd,
      .events = POLLIN,
    }};
    int err = poll(polls, 1, timeout);
    if (err < 0) {
      return -1;
    }
    if (!(polls[0].revents & POLLIN)) {
      break;
    }

    struct input_event event;
    err = read(polls[0].fd, &event, sizeof(event));
    if (err < sizeof(event)) {
      return -1;
    }

    switch (event.type) {
    case EV_ABS:
      if (event.code == ABS_MT_POSITION_X) {
        s->last_x = event.value;
      } else if (event.code == ABS_MT_POSITION_Y) {
        s->last_y = event.value;
      } else if (event.code == ABS_MT_TRACKING_ID && event.value != -1) {
        up = true;
      }
      break;
    default:
      break;
    }
  }
  if (up) {
    // adjust for flippening
    *out_x = s->last_y;
    *out_y = 1080 - s->last_x;
  }
  return up;
}
