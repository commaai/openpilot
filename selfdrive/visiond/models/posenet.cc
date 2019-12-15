#include <string.h>
#include <math.h>
#include "posenet.h"

void posenet_init(PosenetState *s) {
  s->input = (float*)malloc(2*200*532*sizeof(float));
  s->m = new DefaultRunModel("../../models/posenet.dlc", s->output, sizeof(s->output)/sizeof(float), USE_GPU_RUNTIME);
}

void posenet_push(PosenetState *s, uint8_t *yuv_ptr_y, int yuv_width) {
  // move second frame to first frame
  memmove(&s->input[0], &s->input[1], sizeof(float)*(200*532*2 - 1));

  // fill posenet input
  float a;
  // posenet uses a half resolution cropped frame
  // with upper left corner: [50, 237] and
  // bottom right corner: [1114, 637]
  // So the resulting crop is 532 X 200
  for (int y=237; y<637; y+=2) {
    int yy = (y-237)/2;
    for (int x = 50; x < 1114; x+=2) {
      int xx = (x-50)/2;
      a = 0;
      a += yuv_ptr_y[yuv_width*(y+0) + (x+1)];
      a += yuv_ptr_y[yuv_width*(y+1) + (x+1)];
      a += yuv_ptr_y[yuv_width*(y+0) + (x+0)];
      a += yuv_ptr_y[yuv_width*(y+1) + (x+0)];
      // The posenet takes a normalized image input
      // like the driving model so [0,255] is remapped
      // to [-1,1]
      s->input[(yy*532+xx)*2 + 1] = (a/512.0 - 1.0);
    }
  }
}

void posenet_eval(PosenetState *s) {
  s->m->execute(s->input);

  // fix stddevs
  for (int i = 6; i < 12; i++) {
    s->output[i] = log1p(exp(s->output[i])) + 1e-6;
  }
  // to radians
  for (int i = 3; i < 6; i++) {
    s->output[i] = M_PI * s->output[i] / 180.0;
  }
  // to radians
  for (int i = 9; i < 12; i++) {
    s->output[i] = M_PI * s->output[i] / 180.0;
  }
}

void posenet_free(PosenetState *s) {
  delete s->m;
  free(s->input);
}

