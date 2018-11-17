/*
  Use modified touch.h code so we can track press+hold/drag events
*/

#ifndef TOUCH_H
#define TOUCH_H
 
#ifdef __cplusplus
extern "C" {
#endif

typedef struct TouchState {
  int fd;
  int last_x, last_y;
} TouchState;

void touch_init(TouchState *s);
int touch_poll(TouchState *s, int *out_x, int *out_y, int timeout);

#ifdef __cplusplus
}
#endif

#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/poll.h>
#include <linux/input.h>

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

    char name[128] = {0};
    err = ioctl(fd, EVIOCGNAME(sizeof(name) - 1), &name);
    assert(err >= 0);

    unsigned long ev_bits[8] = {0};
    err = ioctl(fd, EVIOCGBIT(0, sizeof(ev_bits)), ev_bits);
    assert(err >= 0);

    if (strncmp(name, "synaptics", 9) == 0 && ev_bits[0] == 0xb) {
      ret = fd;
      break;
    }
    close(fd);
  }
  closedir(dir);

  return ret;
}

void custom_touch_init(TouchState *s) {
  s->fd = find_dev();
  assert(s->fd >= 0);
}

int custom_touch_poll(TouchState *s, int* out_x, int* out_y, int timeout, int* key_up) {
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
      }
      break;
    case EV_KEY:
      if (event.code == BTN_TOOL_FINGER && event.value == 0) {
        // finger up
        up = true;
      }
      break;
    default:
      break;
    }
    *out_x = s->last_y;
    *out_y = 1080 - s->last_x;
  }
  if (up) {
    // adjust for flippening
    *out_x = s->last_y;
    *out_y = 1080 - s->last_x;
    *key_up = up;
  }
  return up;
}

