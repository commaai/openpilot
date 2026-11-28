#ifndef BOARD_UTILS_H
#define BOARD_UTILS_H

// cppcheck-suppress-macro [misra-c2012-1.2, misra-c2012-17.3]; allow __typeof__ extension
#define MIN(a, b) ({ \
  __typeof__ (a) min_a = (a); \
  __typeof__ (b) min_b = (b); \
  (min_a < min_b) ? min_a : min_b; \
})

// cppcheck-suppress-macro [misra-c2012-1.2, misra-c2012-17.3]; allow __typeof__ extension
#define MAX(a, b) ({ \
  __typeof__ (a) max_a = (a); \
  __typeof__ (b) max_b = (b); \
  (max_a > max_b) ? max_a : max_b; \
})

// cppcheck-suppress-macro [misra-c2012-1.2, misra-c2012-17.3]; allow __typeof__ extension
#define CLAMP(x, low, high) ({ \
  __typeof__(x) clamp_x = (x); \
  __typeof__(low) clamp_low = (low);\
  __typeof__(high) clamp_high = (high);\
  (clamp_x > clamp_high) ? clamp_high : ((clamp_x < clamp_low) ? clamp_low : clamp_x); \
})

// cppcheck-suppress-macro [misra-c2012-1.2, misra-c2012-17.3]; allow __typeof__ extension
#define ABS(a) ({ \
  __typeof__ (a) abs_a = (a); \
  (abs_a > 0) ? abs_a : (-abs_a); \
})

#ifndef NULL
// this just provides a standard implementation of NULL
// in lieu of including libc in the panda build
// cppcheck-suppress [misra-c2012-21.1]
#define NULL ((void*)0)
#endif

// STM32 HAL defines this
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#define COMPILE_TIME_ASSERT(pred) ((void)sizeof(char[1 - (2 * (!(pred) ? 1 : 0))]))

// compute the time elapsed (in microseconds) from 2 counter samples
// case where ts < ts_last is ok: overflow is properly re-casted into uint32_t
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts - ts_last;
}

#endif  // BOARD_UTILS_H
