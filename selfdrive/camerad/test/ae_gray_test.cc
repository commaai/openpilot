// unittest for set_exposure_target

#include <assert.h>

#include "selfdrive/camerad/cameras/camera_common.h"

#define W 240
#define H 160

void camera_autoexposure(CameraState *s, float grey_frac) {}

int main() {
  // set up fake camerabuf
  CameraBuf cb = {};
  VisionBuf vb = {};
  uint8_t * fb_y = new uint8_t[W*H];
  vb.y = fb_y;
  cb.cur_yuv_buf = &vb;
  cb.rgb_width = W;
  cb.rgb_height = H;

  printf("WxH %dx%d\n", cb.rgb_width, cb.rgb_height);

  // calculate EV
  for (int round=0; round<3; round++) {
    for (int i=0; i<80; i++) {
      uint8_t pc, sc;
      if (round==0) {
        pc = 235;
      } else {
        pc = 117;
      }
      if (round==2) {
        sc = 235;
      } else {
        sc = 0;
      }
      for (int r=0; r<H; r++) {
        for (int c=0; c<W; c++) {
          fb_y[r*W+c] = r > i * (H/80) ? pc : sc;
        }
      }
      float ev;
      ev = set_exposure_target((const CameraBuf*) &cb, 0, W-1, 1, 0, H-1, 1, 0);
      printf("ev is %f\n", ev);
    }
  }
  return 0;
}
