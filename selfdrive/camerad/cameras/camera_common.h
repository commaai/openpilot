#ifndef CAMERA_COMMON_H
#define CAMERA_COMMON_H

#include <stdint.h>
#include <stdbool.h>

#define CAMERA_ID_IMX298 0
#define CAMERA_ID_IMX179 1
#define CAMERA_ID_S5K3P8SP 2
#define CAMERA_ID_OV8865 3
#define CAMERA_ID_IMX298_FLIPPED 4
#define CAMERA_ID_OV10640 5
#define CAMERA_ID_MAX 6

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CameraInfo {
  const char* name;
  int frame_width, frame_height;
  int frame_stride;
  bool bayer;
  int bayer_flip;
  bool hdr;
} CameraInfo;

typedef struct FrameMetadata {
  uint32_t frame_id;
  uint64_t timestamp_eof;
  unsigned int frame_length;
  unsigned int integ_lines;
  unsigned int global_gain;
  unsigned int lens_pos;
  float lens_sag;
  float lens_err;
  float lens_true_pos;
  float gain_frac;
} FrameMetadata;

extern CameraInfo cameras_supported[CAMERA_ID_MAX];

#ifdef __cplusplus
}
#endif

#endif
