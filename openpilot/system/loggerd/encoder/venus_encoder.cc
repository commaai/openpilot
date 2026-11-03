#include "system/loggerd/encoder/venus_encoder.h"

#include <cassert>
#include <cstring>

#include "system/loggerd/encoder/v4l_device.h"
#include "common/util.h"

// upstream controls the downstream kernel's headers don't have, for builds on its AGNOS
#ifndef V4L2_CID_MPEG_VIDEO_HEVC_PROFILE
#define V4L2_CID_MPEG_VIDEO_HEVC_PROFILE (V4L2_CID_MPEG_BASE + 615)
#define V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN 0
#define V4L2_CID_MPEG_VIDEO_HEVC_LEVEL (V4L2_CID_MPEG_BASE + 616)
#define V4L2_MPEG_VIDEO_HEVC_LEVEL_5 7
#endif

// a VisionIPC buffer is queued at its buffer index, so vb2 keeps it attached and mapped. that's one for each of
// camerad's 18 buffers or the decoder's 16. a source with more shares indices, which only costs a remap
#define VENUS_IN_COUNT 18

VenusEncoder::VenusEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options)
    : V4LEncoder(encoder_info, in_width, in_height, options, open_video_device("qcom-venus-encoder"), V4L2_MEMORY_DMABUF,
                 VENUS_IN_COUNT) {
  struct v4l2_capability cap;
  util::safe_ioctl(fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP failed");
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  assert(strcmp((const char *)cap.driver, "qcom-venus") == 0);
  // Venus encodes linear NV12
  assert(options.input_format == V4L2_PIX_FMT_NV12);

  set_formats(V4L2_PIX_FMT_NV12);
  // Venus aligns the frame size to 128x32 for its buffers, and encodes all of it unless it's cropped to the frame
  struct v4l2_selection sel = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT,
    .target = V4L2_SEL_TGT_CROP,
    .r = {.left = 0, .top = 0, .width = (uint32_t)in_width, .height = (uint32_t)in_height},
  };
  util::safe_ioctl(fd, VIDIOC_S_SELECTION, &sel, "VIDIOC_S_SELECTION failed");
  // the source format and crop reset the bitstream size to the crop. a smaller one is a scaled encode, like msm_vidc's,
  // so it's set again. the stream's header has the exact size (H.264 frame cropping, the HEVC conformance window)
  if (out_width != in_width || out_height != in_height) {
    struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
    util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
    fmt.fmt.pix_mp.width = out_width;
    fmt.fmt.pix_mp.height = out_height;
    util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");
    // a kernel without the scaling patch crops the source to that size instead
    util::safe_ioctl(fd, VIDIOC_G_SELECTION, &sel, "VIDIOC_G_SELECTION failed");
    assert(sel.r.width == (uint32_t)in_width && sel.r.height == (uint32_t)in_height);
  }
  set_codec_controls(encoder_info.get_settings(in_width));
  start(true);
}

void VenusEncoder::set_codec_controls(const EncoderSettings &encoder_settings) {
  V4LEncoder::set_controls({
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
    // VBR without frame skipping is the firmware's VBR_CFR, like msm_vidc's
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE, .value = V4L2_MPEG_VIDEO_BITRATE_MODE_VBR},
    { .id = V4L2_CID_MPEG_VIDEO_GOP_SIZE, .value = encoder_settings.gop_size},
    { .id = V4L2_CID_MPEG_VIDEO_B_FRAMES, .value = encoder_settings.b_frames},
    // parse_bitstream splits the header off the IDR frames
    { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME},
  });

  if (is_h265) {
    // Venus has no tier control, so this is Main tier, where msm_vidc's is High. that only raises the bitrate limit,
    // which at level 5 is 25 Mbps for Main, more than the 10 Mbps the road cameras use
    V4LEncoder::set_controls({
      { .id = V4L2_CID_MPEG_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN},
      { .id = V4L2_CID_MPEG_VIDEO_HEVC_LEVEL, .value = V4L2_MPEG_VIDEO_HEVC_LEVEL_5},
    });
  } else {
    V4LEncoder::set_controls({
      { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = h264_level(out_width, out_height, encoder_info.fps, encoder_info.is_live)},
      { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = V4L2_MPEG_VIDEO_H264_LOOP_FILTER_MODE_ENABLED},
    });
  }
}

V4LEncoder::Bitstream VenusEncoder::parse_bitstream(uint32_t flags, const uint8_t *dat, size_t len) {
  // the header is in front of the first slice of every IDR frame. the drain ends with a LAST buffer, which may hold
  // the last frame
  const size_t header_len = (flags & V4L2_BUF_FLAG_KEYFRAME) ? header_size(dat, len, is_h265) : 0;
  return {.header_len = header_len, .frame = len > header_len, .eos = (flags & V4L2_BUF_FLAG_LAST) != 0};
}

int VenusEncoder::acquire_input(const VisionBuf *buf) {
  return in_slots.acquire(buf->idx % in_slots.size());
}

void VenusEncoder::set_plane(v4l2_plane *plane, const VisionBuf *buf) {
  plane->m.fd = buf->fd;
}

void VenusEncoder::resume() {
  // the drain in encoder_close stopped the encoder, restart it with an IDR frame for the new segment
  struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_START };
  util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
  request_keyframe();
}

VenusEncoder::~VenusEncoder() {
  // before the buffers it queued are freed
  encoder_close();
}
