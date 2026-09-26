#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define ALLOW_DEBUG

// TODO: time should just be passed into the hooks we expose
uint32_t timer_cnt = 0;
uint32_t microsecond_timer_get(void);
uint32_t microsecond_timer_get(void) {
  return timer_cnt;
}
