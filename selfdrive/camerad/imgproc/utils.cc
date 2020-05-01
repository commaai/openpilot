#include "utils.h"
#include <stdio.h>
#include <algorithm>

// calculate score based on laplacians in one area
void get_lapmap_one(int16_t *lap, uint16_t *res, int x_pitch, int y_pitch) {
  int size = x_pitch * y_pitch;
  // avg and max of roi
  float fsum = 0;
  int16_t mean, max;
  max = 0;

  for (int i = 0; i < size; i++) {
    int x_offset = i % x_pitch;
    int y_offset = i / x_pitch;
    fsum += lap[x_offset + y_offset*x_pitch];
    max = std::max(lap[x_offset + y_offset*x_pitch], max);
  }

  mean = fsum / size;

  // var of roi
  float fvar = 0;
  for (int i = 0; i < size; i++) {
    int x_offset = i % x_pitch;
    int y_offset = i / x_pitch;
    fvar += (float)((lap[x_offset + y_offset*x_pitch] - mean) * (lap[x_offset + y_offset*x_pitch] - mean));
  }

  fvar = fvar / size;

  *res = std::min(5 * fvar + max, (float)65535);
}

bool is_blur(uint16_t *lapmap) {
  int n_roi = (ROI_X_MAX - ROI_X_MIN + 1) * (ROI_Y_MAX - ROI_Y_MIN + 1);
  float bad_sum = 0;
  for (int i = 0; i < n_roi; i++) {
    if (*(lapmap + i) < LM_THRESH) {
      bad_sum += 1/(float)n_roi;
    }
  }
  return (bad_sum > LM_PREC_THRESH);
}