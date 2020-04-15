#include "utils.h"
#include <stdio.h>

// replaced by var_pool cl kernel
void get_lapmap(int8_t *lap, int8_t *lapmap, int x_pitch, int y_pitch) {
  int size = x_pitch * y_pitch;
  for (int xidx=ROI_X_MIN;xidx<=ROI_X_MAX;xidx++) {
    for (int yidx=ROI_Y_MIN;yidx<=ROI_Y_MAX;yidx++) {
      // avg and max of roi
      float fsum = 0;
      int8_t mean, max;
      
      for (int i = 0; i < size; i++) {
        int x_offset = i % x_pitch;
        int y_offset = i / x_pitch;
        fsum += lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X];
        max = lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]>max?lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]:max;
      }

      mean = fsum / size;
      // printf("mean %f\n", mean);

      // var of roi
      int8_t var = 0;
      float fvar = 0;
      for (int i = 0; i < size; i++) {
        int x_offset = i % x_pitch;
        int y_offset = i / x_pitch;
        fvar += (lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean) * (lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean);
        // printf("item %f\n", (lap[xidx*x_pitch + yidx*FULL_STRIDE_X + x_offset + y_offset*x_pitch] - mean) * (lap[xidx*x_pitch + yidx*FULL_STRIDE_X + x_offset + y_offset*x_pitch] - mean));
      }

      var = (int8_t)(fvar / size);
      // printf("var %f\n", var);

      lapmap[(xidx-ROI_X_MIN)+(yidx-ROI_Y_MIN)*(ROI_X_MAX-ROI_X_MIN+1)] = 5 * var + max;
    }
  }
}

bool is_blur(uint16_t *lapmap) {
  int n_roi = (ROI_X_MAX - ROI_X_MIN + 1) * (ROI_Y_MAX - ROI_Y_MIN + 1);
  float bad_sum = 0;
  for (int i = 0; i < n_roi; i++) {
    printf("%d- %d\n", i, lapmap[i]);
    if (lapmap[i] < LM_THRESH) {
      bad_sum += 1/(float)n_roi;
    }
  }
  return (bool)(bad_sum > LM_PREC_THRESH);
}