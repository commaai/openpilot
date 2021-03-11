// unittest for set_exposure_target

#include <assert.h>
#include <cstring>

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/camerad/test/ae_gray_test.h"

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

  printf("AE test patterns %dx%d\n", cb.rgb_width, cb.rgb_height);

  // mix of 4 tones
  uint8_t l[4] = {0, 32, 96, 235}; // 235 is yuv max

  // generate pattern and calculate EV
  for (int i_0=0; i_0<5; i_0++) {
    for (int i_1=0; i_1<5; i_1++) {
      for (int i_2=0; i_2<5; i_2++) {
        int h_0 = i_0 * H / 5;
        int h_1 = i_1 * (H - h_0) / 5;
        int h_2 = i_2 * (H - h_0 - h_1) / 5;
        int h_3 = H - h_0 - h_1 - h_2;
        memset(fb_y, l[0], h_0*W);
        memset(fb_y+h_0*W, l[1], h_1*W);
        memset(fb_y+h_0*W+h_1*W, l[0], h_2*W);
        memset(fb_y+h_0*W+h_1*W+h_2*W, l[0], h_3*W);
        float ev = set_exposure_target((const CameraBuf*) &cb, 0, W-1, 1, 0, H-1, 1, 0);
        printf("%d/%d/%d/%d ev is %f\n", h_0, h_1, h_2, h_3, ev);
      }
    }
  }

  delete[] fb_y;
  return 0;
}
