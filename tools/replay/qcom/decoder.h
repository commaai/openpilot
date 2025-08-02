#pragma once
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <stdint.h>
#include <limits.h>
#include <poll.h>
#include <list>

#include "third_party/linux/include/media/msm_vidc.h"
#include "msgq/visionipc/visionbuf.h"
#include "sde_rotator.h"

extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
}

#define VIDEO_DEVICE "/dev/video32"
#define OUTPUT_BUFFER_COUNT 	8
#define CAPTURE_BUFFER_COUNT 	4
#define CAP_PLANES 						2
#define OUT_PLANES						1
#define FPS 									20

#define TIMESTAMP_NONE	((uint64_t)-1)
#define V4L2_QCOM_BUF_TIMESTAMP_INVALID		0x00080000

#define V4L2_EVENT_MSM_VIDC_START	(V4L2_EVENT_PRIVATE_START + 0x00001000)
#define V4L2_EVENT_MSM_VIDC_FLUSH_DONE	(V4L2_EVENT_MSM_VIDC_START + 1)
#define V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_SUFFICIENT	\
	  (V4L2_EVENT_MSM_VIDC_START + 2)
#define V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT	\
	  (V4L2_EVENT_MSM_VIDC_START + 3)
/*
 * Bitdepth changed insufficient is deprecated now, however retaining
 * to prevent changing the values of the other macros after bitdepth
 */
#define V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_BITDEPTH_CHANGED_INSUFFICIENT \
	  (V4L2_EVENT_MSM_VIDC_START + 4)
#define V4L2_EVENT_MSM_VIDC_SYS_ERROR	(V4L2_EVENT_MSM_VIDC_START + 5)
#define V4L2_EVENT_MSM_VIDC_RELEASE_BUFFER_REFERENCE \
	  (V4L2_EVENT_MSM_VIDC_START + 6)
#define V4L2_EVENT_MSM_VIDC_RELEASE_UNQUEUED_BUFFER \
	  (V4L2_EVENT_MSM_VIDC_START + 7)
#define V4L2_EVENT_MSM_VIDC_HW_OVERLOAD (V4L2_EVENT_MSM_VIDC_START + 8)
#define V4L2_EVENT_MSM_VIDC_MAX_CLIENTS (V4L2_EVENT_MSM_VIDC_START + 9)
#define V4L2_EVENT_MSM_VIDC_HW_UNSUPPORTED (V4L2_EVENT_MSM_VIDC_START + 10)


#define V4L2_CID_MPEG_MSM_VIDC_BASE 0x00992000
#define V4L2_CID_MPEG_VIDC_VIDEO_EXTRADATA \
    (V4L2_CID_MPEG_MSM_VIDC_BASE + 17)
#define V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT \
		(V4L2_CID_MPEG_MSM_VIDC_BASE + 44)
#define V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE \
	  (V4L2_CID_MPEG_MSM_VIDC_BASE + 53)
#define V4L2_CID_MPEG_VIDC_VIDEO_CONCEAL_COLOR_8BIT	\
	(V4L2_CID_MPEG_MSM_VIDC_BASE + 109)
#define V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE \
		(V4L2_CID_MPEG_MSM_VIDC_BASE + 22)


enum v4l2_mpeg_vidc_video_decoder_multi_stream {
	V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_PRIMARY = 0,
	V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_SECONDARY = 1,
};

enum v4l2_mpeg_vidc_extradata {
	V4L2_MPEG_VIDC_EXTRADATA_NONE = 0,
	V4L2_MPEG_VIDC_EXTRADATA_MB_QUANTIZATION = 1,
	V4L2_MPEG_VIDC_EXTRADATA_INTERLACE_VIDEO = 2,
	V4L2_MPEG_VIDC_EXTRADATA_VC1_FRAMEDISP = 3,
	V4L2_MPEG_VIDC_EXTRADATA_VC1_SEQDISP = 4,
	V4L2_MPEG_VIDC_EXTRADATA_TIMESTAMP = 5,
	V4L2_MPEG_VIDC_EXTRADATA_S3D_FRAME_PACKING = 6,
	V4L2_MPEG_VIDC_EXTRADATA_FRAME_RATE = 7,
	V4L2_MPEG_VIDC_EXTRADATA_PANSCAN_WINDOW = 8,
	V4L2_MPEG_VIDC_EXTRADATA_RECOVERY_POINT_SEI = 9,
	V4L2_MPEG_VIDC_EXTRADATA_MULTISLICE_INFO = 10,
	V4L2_MPEG_VIDC_EXTRADATA_NUM_CONCEALED_MB = 11,
	V4L2_MPEG_VIDC_EXTRADATA_METADATA_FILLER = 12,
	V4L2_MPEG_VIDC_EXTRADATA_INPUT_CROP = 13,
	V4L2_MPEG_VIDC_EXTRADATA_DIGITAL_ZOOM = 14,
	V4L2_MPEG_VIDC_EXTRADATA_ASPECT_RATIO = 15,
	V4L2_MPEG_VIDC_EXTRADATA_MPEG2_SEQDISP = 16,
	V4L2_MPEG_VIDC_EXTRADATA_STREAM_USERDATA = 17,
	V4L2_MPEG_VIDC_EXTRADATA_FRAME_QP = 18,
	V4L2_MPEG_VIDC_EXTRADATA_FRAME_BITS_INFO = 19,
	V4L2_MPEG_VIDC_EXTRADATA_LTR = 20,
	V4L2_MPEG_VIDC_EXTRADATA_METADATA_MBI = 21,
	V4L2_MPEG_VIDC_EXTRADATA_VQZIP_SEI = 22,
	V4L2_MPEG_VIDC_EXTRADATA_YUV_STATS = 23,
	V4L2_MPEG_VIDC_EXTRADATA_ROI_QP = 24,
#define V4L2_MPEG_VIDC_EXTRADATA_OUTPUT_CROP \
	V4L2_MPEG_VIDC_EXTRADATA_OUTPUT_CROP
	V4L2_MPEG_VIDC_EXTRADATA_OUTPUT_CROP = 25,
#define V4L2_MPEG_VIDC_EXTRADATA_DISPLAY_COLOUR_SEI \
	V4L2_MPEG_VIDC_EXTRADATA_DISPLAY_COLOUR_SEI
	V4L2_MPEG_VIDC_EXTRADATA_DISPLAY_COLOUR_SEI = 26,
#define V4L2_MPEG_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI \
	V4L2_MPEG_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI
	V4L2_MPEG_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI = 27,
#define V4L2_MPEG_VIDC_EXTRADATA_PQ_INFO \
	V4L2_MPEG_VIDC_EXTRADATA_PQ_INFO
	V4L2_MPEG_VIDC_EXTRADATA_PQ_INFO = 28,
#define V4L2_MPEG_VIDC_EXTRADATA_VUI_DISPLAY \
	V4L2_MPEG_VIDC_EXTRADATA_VUI_DISPLAY
	V4L2_MPEG_VIDC_EXTRADATA_VUI_DISPLAY = 29,
#define V4L2_MPEG_VIDC_EXTRADATA_VPX_COLORSPACE \
	V4L2_MPEG_VIDC_EXTRADATA_VPX_COLORSPACE
	V4L2_MPEG_VIDC_EXTRADATA_VPX_COLORSPACE = 30,
#define V4L2_MPEG_VIDC_EXTRADATA_UBWC_CR_STATS_INFO \
	V4L2_MPEG_VIDC_EXTRADATA_UBWC_CR_STATS_INFO
	V4L2_MPEG_VIDC_EXTRADATA_UBWC_CR_STATS_INFO = 31,
#define V4L2_MPEG_VIDC_EXTRADATA_ENC_FRAME_QP \
	V4L2_MPEG_VIDC_EXTRADATA_ENC_FRAME_QP
	V4L2_MPEG_VIDC_EXTRADATA_ENC_FRAME_QP = 32,
};

enum v4l2_mpeg_vidc_video_dpb_color_format {
	V4L2_MPEG_VIDC_VIDEO_DPB_COLOR_FMT_NONE = 0,
	V4L2_MPEG_VIDC_VIDEO_DPB_COLOR_FMT_UBWC = 1,
	V4L2_MPEG_VIDC_VIDEO_DPB_COLOR_FMT_TP10_UBWC = 2
};
#define V4L2_QCOM_CMD_FLUSH_CAPTURE (1 << 1)
#define V4L2_QCOM_CMD_FLUSH      		(4)

class MsmVidc {
public:
  MsmVidc() = default;
  ~MsmVidc();

  bool init(const char* dev, size_t width, size_t height, uint64_t codec);
  VisionBuf* decodeFrame(AVPacket* pkt, VisionBuf* buf);

  AVFormatContext* avctx = nullptr;
  int fd = 0;
  int sigfd = 0;

private:
  bool initialized = false;
  bool reconfigure_pending = false;
  bool need_more_frames = false;
  bool frame_ready = false;

  VisionBuf* current_output_buf = nullptr;
  VisionBuf out_buf;                          // Single input buffer
  VisionBuf ext_buf;
  VisionBuf cap_bufs[CAPTURE_BUFFER_COUNT];   // Capture (output) buffers

  size_t w = 1928, h = 1208;
  size_t cap_height = 0, cap_width = 0;

  int cap_buf_size = 0;
  int out_buf_size = 0;

  size_t cap_plane_off[CAPTURE_BUFFER_COUNT] = {0};
  size_t cap_plane_stride[CAPTURE_BUFFER_COUNT] = {0};
  bool cap_buf_flag[CAPTURE_BUFFER_COUNT] = {false};

  size_t ext_buf_off[CAPTURE_BUFFER_COUNT] = {0};
  void* ext_buf_addr[CAPTURE_BUFFER_COUNT] = {0};

  size_t out_buf_off[OUTPUT_BUFFER_COUNT] = {0};
  void* out_buf_addr[OUTPUT_BUFFER_COUNT] = {0};
  bool out_buf_flag[OUTPUT_BUFFER_COUNT] = {false};
	const int out_buf_cnt = OUTPUT_BUFFER_COUNT;

  uint32_t cap_buf_format = 0;

  const int subscriptions[2] = {
    V4L2_EVENT_MSM_VIDC_FLUSH_DONE,
    V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT
  };

  enum { EV_VIDEO, EV_SIGNAL, EV_COUNT };
  struct pollfd pfd[EV_COUNT] = {0};
  int ev[EV_COUNT] = {-1, -1};
  int nfds = 0;

  bool setupOutput();
  bool subscribeEvents();
  bool setPlaneFormat(v4l2_buf_type type, uint32_t fourcc);
  bool setFPS(uint32_t fps);
  bool setControls();
  bool restartCapture();
  bool queueCaptureBuffer(int i);
  bool queueOutputBuffer(int i, size_t size);
  bool setDBP(enum v4l2_mpeg_vidc_video_dpb_color_format format);
  bool setupPolling();
  bool sendPacket(int buf_index, AVPacket* pkt);
  int getBufferUnlocked();
  int handleSignal();
  VisionBuf* handleCapture();
  bool handleOutput();
  bool handleEvent();

  SdeRotator rotator;
};
