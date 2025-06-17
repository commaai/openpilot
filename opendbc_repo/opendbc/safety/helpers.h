#include "opendbc/safety/safety_declarations.h"

static bool max_limit_check(int val, const int MAX_VAL, const int MIN_VAL) {
  return (val > MAX_VAL) || (val < MIN_VAL);
}

// interp function that holds extreme values
static float interpolate(struct lookup_t xy, float x) {

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
        dx = MAX(dx, 0.0001);
        ret = (dy * (x - x0) / dx) + y0;
        break;
      }
    }
  }
  return ret;
}
