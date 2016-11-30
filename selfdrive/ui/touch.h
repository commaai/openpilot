#ifndef TOUCH_H
#define TOUCH_H

typedef struct TouchState {
  int fd;
  int last_x, last_y;
} TouchState;

void touch_init(TouchState *s);
int touch_poll(TouchState *s, int *out_x, int *out_y);

#endif
