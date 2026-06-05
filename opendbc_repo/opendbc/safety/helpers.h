#include "opendbc/safety/declarations.h"

// cppcheck-suppress-macro misra-c2012-1.2; allow __typeof__ extension
// cppcheck-suppress-macro misra-c2012-17.3; suppress false implicit declaration alert on typeof extension
#define SAFETY_MIN(a, b) ({ \
  __typeof__(a) _a = (a); \
  __typeof__(b) _b = (b); \
  (_a < _b) ? _a : _b; \
})

// cppcheck-suppress-macro misra-c2012-1.2; allow __typeof__ extension
// cppcheck-suppress-macro misra-c2012-17.3; suppress false implicit declaration alert on typeof extension
#define SAFETY_MAX(a, b) ({ \
  __typeof__(a) _a = (a); \
  __typeof__(b) _b = (b); \
  (_a > _b) ? _a : _b; \
})

// cppcheck-suppress-macro misra-c2012-1.2; allow __typeof__ extension
// cppcheck-suppress-macro misra-c2012-17.3; suppress false implicit declaration alert on typeof extension
#define SAFETY_CLAMP(x, low, high) ({ \
  __typeof__(x) __x = (x); \
  __typeof__(low) __low = (low);\
  __typeof__(high) __high = (high);\
  (__x > __high) ? __high : ((__x < __low) ? __low : __x); \
})

// cppcheck-suppress-macro misra-c2012-1.2; allow __typeof__ extension
// cppcheck-suppress-macro misra-c2012-17.3; suppress false implicit declaration alert on typeof extension
#define SAFETY_ABS(a) ({ \
  __typeof__(a) _a = (a); \
  (_a > 0) ? _a : (-_a); \
})

#define SAFETY_UNUSED(x) ((void)(x))

// compute the time elapsed (in microseconds) from 2 counter samples
// case where ts < ts_last is ok: overflow is properly re-casted into uint32_t
static inline uint32_t safety_get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts - ts_last;
}

static bool safety_max_limit_check(int val, const int MAX_VAL, const int MIN_VAL) {
  return (val > MAX_VAL) || (val < MIN_VAL);
}

// interp function that holds extreme values
static float safety_interpolate(struct lookup_t xy, float x) {
  int size = sizeof(xy.x) / sizeof(xy.x[0]);
  float ret = xy.y[size - 1];  // default output is last point

  // x is lower than the first point in the x array. Return the first point
  if (x <= xy.x[0]) {
    ret = xy.y[0];

  } else {
    // find the index such that (xy.x[i] <= x < xy.x[i+1]) and linearly interp
    for (int i=0; i < (size - 1); i++) {
      if (x < xy.x[i+1]) {
        float x0 = xy.x[i];
        float y0 = xy.y[i];
        float dx = xy.x[i+1] - x0;
        float dy = xy.y[i+1] - y0;
        // dx should not be zero as xy.x is supposed to be monotonic
        dx = SAFETY_MAX(dx, 0.0001);
        ret = (dy * (x - x0) / dx) + y0;
        break;
      }
    }
  }
  return ret;
}
