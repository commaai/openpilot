#include "system/loggerd/encoder/msm_vidc_encoder.h"

#include <cassert>
#include <cstring>

#include "system/loggerd/encoder/msm_vidc.h"
#include "common/util.h"

#define BUF_IN_COUNT 9

MsmVidcEncoder::MsmVidcEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options)
    : V4LEncoder(encoder_info, in_width, in_height, options,
                 HANDLE_EINTR(open("/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1", O_RDWR|O_NONBLOCK)),
                 V4L2_MEMORY_USERPTR, BUF_IN_COUNT) {
  struct v4l2_capability cap;
  util::safe_ioctl(fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP failed");
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  assert(strcmp((const char *)cap.driver, "msm_vidc_driver") == 0);
  assert(strcmp((const char *)cap.card, "msm_vidc_venc") == 0);

  // msm_vidc scales the input to the bitstream size
  set_formats(options.input_format);
  set_codec_controls(encoder_info.get_settings(in_width), options.max_performance);
  start(false);
}

void MsmVidcEncoder::set_codec_controls(const EncoderSettings &encoder_settings, bool max_performance) {
  // shared ctrls
  V4LEncoder::set_controls({
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = encoder_settings.gop_size - encoder_settings.b_frames - 1},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = encoder_settings.b_frames},
    { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, .value = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
  });
  if (max_performance) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY,
      .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL offline encode failed");
  }

  if (is_h265) {
    V4LEncoder::set_controls({
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO, .value = V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED},
    });
  } else {
    V4LEncoder::set_controls({
      { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = encoder_info.is_live ? V4L2_MPEG_VIDEO_H264_LEVEL_3_1 : V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN},
      { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = V4L2_MPEG_VIDEO_H264_LOOP_FILTER_MODE_ENABLED},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, .value = 0},
    });
  }
}

V4LEncoder::Bitstream MsmVidcEncoder::parse_bitstream(uint32_t flags, const uint8_t *dat, size_t len) {
  // the stream ends with an empty EOS buffer, and the header comes in its own CODECCONFIG buffer
  if (flags & V4L2_QCOM_BUF_FLAG_EOS) return {.eos = true};
  if (flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG) return {.header_len = len};
  return {.frame = true};
}

int MsmVidcEncoder::acquire_input(const VisionBuf *) {
  return in_slots.acquire();
}

void MsmVidcEncoder::set_plane(v4l2_plane *plane, const VisionBuf *buf) {
  // msm_vidc takes the ION fd along with the pointer
  plane->m.userptr = (unsigned long)buf->addr;
  plane->reserved[0] = buf->fd;
}

uint32_t MsmVidcEncoder::keyframe_control() const {
  return V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME;
}

MsmVidcEncoder::~MsmVidcEncoder() {
  encoder_close();
}
