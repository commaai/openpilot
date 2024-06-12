#include "msgq/visionipc/visionbuf.h"

#define ALIGN(x, align) (((x) + (align)-1) & ~((align)-1))

void visionbuf_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h) {
  *aligned_w = width;
  *aligned_h = height;
}

void VisionBuf::init_rgb(size_t init_width, size_t init_height, size_t init_stride) {
  this->rgb = true;
  this->width = init_width;
  this->height = init_height;
  this->stride = init_stride;
}

void VisionBuf::init_yuv(size_t init_width, size_t init_height, size_t init_stride, size_t init_uv_offset){
  this->rgb = false;
  this->width = init_width;
  this->height = init_height;
  this->stride = init_stride;
  this->uv_offset = init_uv_offset;

  this->y = (uint8_t *)this->addr;
  this->uv = this->y + this->uv_offset;
}


uint64_t VisionBuf::get_frame_id() {
  return *frame_id;
}

void VisionBuf::set_frame_id(uint64_t id) {
  *frame_id = id;
}
