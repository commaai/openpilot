#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/poll.h>
#include <linux/input.h>

#include "touch.h"

void touch_init(TouchState *s) {
  // synaptics touch screen on oneplus 3
  s->fd = open("/dev/input/event4", O_RDONLY);
  assert(s->fd >= 0);
}

int touch_poll(TouchState *s, int* out_x, int* out_y) {
  assert(out_x && out_y);
  bool up = false;
  while (true) {
    struct pollfd polls[] = {{
      .fd = s->fd,
      .events = POLLIN,
    }};
    int err = poll(polls, 1, 0);
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
  }
  if (up) {
    // adjust for landscape
    *out_x = 1920 - s->last_y;
    *out_y = s->last_x;
  }
  return up;
}

