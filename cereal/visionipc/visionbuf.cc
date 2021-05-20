#include "visionbuf.h"

#define ALIGN(x, align) (((x) + (align)-1) & ~((align)-1))

#ifdef QCOM
// from libadreno_utils.so
extern "C" void compute_aligned_width_and_height(int width,
                                                 int height,
                                                 int bpp,
                                                 int tile_mode,
                                                 int raster_mode,
                                                 int padding_threshold,
                                                 int *aligned_w,
                                                 int *aligned_h);
#endif

void visionbuf_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h) {
#if defined(QCOM) && !defined(QCOM_REPLAY)
  compute_aligned_width_and_height(ALIGN(width, 32), ALIGN(height, 32), 3, 0, 0, 512, aligned_w, aligned_h);
#else
  *aligned_w = width; *aligned_h = height;
#endif
}

void VisionBuf::init_rgb(size_t width, size_t height, size_t stride) {
  this->rgb = true;
  this->width = width;
  this->height = height;
  this->stride = stride;
}

void VisionBuf::init_yuv(size_t width, size_t height){
  this->rgb = false;
  this->width = width;
  this->height = height;

  this->y = (uint8_t *)this->addr;
  this->u = this->y + (width * height);
  this->v = this->u + (width / 2 * height / 2);
}
