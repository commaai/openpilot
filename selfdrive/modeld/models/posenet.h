#ifndef POSENET_H
#define POSENET_H

#include <stdint.h>
#include "runners/run.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PosenetState {
  float output[12];
  float *input;
  RunModel *m;
} PosenetState;

void posenet_init(PosenetState *s);
void posenet_push(PosenetState *s, uint8_t *yuv_ptr_y, int yuv_width);
void posenet_eval(PosenetState *s);
void posenet_free(PosenetState *s);

#ifdef __cplusplus
}
#endif

#endif

