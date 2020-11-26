#pragma once

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
