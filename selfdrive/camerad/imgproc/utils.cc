#include "utils.h"
#include <stdio.h>
#include <algorithm>
#include <cmath>
// calculate score based on laplacians in one area
uint16_t get_lapmap_one(const int16_t *lap, int x_pitch, int y_pitch) {
  const int size = x_pitch * y_pitch;
  // avg and max of roi
  int16_t max = 0;
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    const int16_t v = lap[i % x_pitch + (i / x_pitch) * x_pitch];
    sum += v;
    if (v > max) max = v;
  }

  const int16_t mean = sum / size;

  // var of roi
  int var = 0;
  for (int i = 0; i < size; ++i) {
    var += std::pow(lap[i % x_pitch + (i / x_pitch) * x_pitch] - mean, 2);
  }

  const float fvar = (float)var / size;
  return std::min(5 * fvar + max, (float)65535);
}

bool is_blur(const uint16_t *lapmap, const size_t size) {
  float bad_sum = 0;
  for (int i = 0; i < size; i++) {
    if (lapmap[i] < LM_THRESH) {
      bad_sum += 1 / (float)size;
    }
  }
  return (bad_sum > LM_PREC_THRESH);
}