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
#ifdef QCOM
  compute_aligned_width_and_height(ALIGN(width, 32), ALIGN(height, 32), 3, 0, 0, 512, aligned_w, aligned_h);
#else
  *aligned_w = width; *aligned_h = height;
#endif
}

void VisionBuf::init_rgb(size_t init_width, size_t init_height, size_t init_stride) {
  this->rgb = true;
  this->width = init_width;
  this->height = init_height;
  this->stride = init_stride;
}

void VisionBuf::init_yuv(size_t init_width, size_t init_height){
  this->rgb = false;
  this->width = init_width;
  this->height = init_height;

  this->y = (uint8_t *)this->addr;
  this->u = this->y + (this->width * this->height);
  this->v = this->u + (this->width / 2 * this->height / 2);
}


uint64_t VisionBuf::get_frame_id() {
  return *frame_id;
}

void VisionBuf::set_frame_id(uint64_t id) {
  *frame_id = id;
}
