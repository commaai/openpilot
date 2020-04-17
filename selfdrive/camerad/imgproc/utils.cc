#include "utils.h"
#include <stdio.h>

uint16_t clamp_uint16(float x) {
  if (x < 0)
  {
    return 0;
  }
  else if (x > 65535)
  {
    return 65535;
  }
  else
  {
    return (uint16_t) x;
  }
}

uint8_t clamp_uint8(float x) {
  if (x < 0)
  {
    return 0;
  }
  else if (x > 255)
  {
    return 255;
  }
  else
  {
    return (uint8_t) x;
  }
}

// replaced by var_pool cl kernel
void get_lapmap(int16_t *lap, uint16_t *lapmap, int x_pitch, int y_pitch) {
  int size = x_pitch * y_pitch;
  for (int xidx=ROI_X_MIN;xidx<=ROI_X_MAX;xidx++) {
    for (int yidx=ROI_Y_MIN;yidx<=ROI_Y_MAX;yidx++) {
      // avg and max of roi
      float fsum = 0;
      int16_t mean, max;
      max = 0;
      
      for (int i = 0; i < size; i++) {
        int x_offset = i % x_pitch;
        int y_offset = i / x_pitch;
        fsum += lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X];
        max = lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]>max?lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]:max;
      }

      mean = fsum / size;
      // printf("mean %d\n", mean);

      // var of roi
      // uint16_t var = 0;
      float fvar = 0;
      for (int i = 0; i < size; i++) {
        int x_offset = i % x_pitch;
        int y_offset = i / x_pitch;
        fvar += (lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean) * (lap[xidx*x_pitch + yidx*y_pitch*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean);
      }

      fvar = fvar / size;
      // printf("fvar %f\n", fvar);

      lapmap[(xidx-ROI_X_MIN)+(yidx-ROI_Y_MIN)*(ROI_X_MAX-ROI_X_MIN+1)] = clamp_uint16(5 * fvar + max);
    }
  }
}

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
    max = lap[x_offset + y_offset*x_pitch]>max?lap[x_offset + y_offset*x_pitch]:max;
  }

  mean = fsum / size;
  // printf("mean %d\n", mean);

  // var of roi
  // uint16_t var = 0;
  float fvar = 0;
  for (int i = 0; i < size; i++) {
    int x_offset = i % x_pitch;
    int y_offset = i / x_pitch;
    fvar += (float)((lap[x_offset + y_offset*x_pitch] - mean) * (lap[x_offset + y_offset*x_pitch] - mean));
  }

  fvar = fvar / size;
  // printf("fvar %f\n", fvar);

  *res = clamp_uint16(5 * fvar + max);
}

bool is_blur(uint16_t *lapmap) {
  int n_roi = (ROI_X_MAX - ROI_X_MIN + 1) * (ROI_Y_MAX - ROI_Y_MIN + 1);
  float bad_sum = 0;
  for (int i = 0; i < n_roi; i++) {
    printf("%d- %d\n", i, *(lapmap + i));
    if (*(lapmap + i) < LM_THRESH) {
      bad_sum += 1/(float)n_roi;
    }
  }
  return (bool)(bad_sum > LM_PREC_THRESH);
}