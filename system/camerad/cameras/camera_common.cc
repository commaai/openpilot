#include "system/camerad/cameras/camera_common.h"

#include <cassert>
#include <string>

#include "common/swaglog.h"
#include "system/camerad/cameras/spectra.h"


void CameraBuf::init(cl_device_id device_id, cl_context context, SpectraCamera *cam, VisionIpcServer * v, int frame_cnt, VisionStreamType type) {
  vipc_server = v;
  stream_type = type;
  frame_buf_count = frame_cnt;

  const SensorInfo *sensor = cam->sensor.get();

  // RAW frames from ISP
  if (cam->cc.output_type != ISP_IFE_PROCESSED) {
    camera_bufs_raw = std::make_unique<VisionBuf[]>(frame_buf_count);

    const int raw_frame_size = (sensor->frame_height + sensor->extra_height) * sensor->frame_stride;
    for (int i = 0; i < frame_buf_count; i++) {
      camera_bufs_raw[i].allocate(raw_frame_size);
      camera_bufs_raw[i].init_cl(device_id, context);
    }
    LOGD("allocated %d CL buffers", frame_buf_count);
  }

  // the encoder HW tells us the size it wants after setting it up.
  // TODO: VENUS_BUFFER_SIZE should give the size, but it's too small. dependent on encoder settings?
  size_t nv12_size = (out_img_width <= 1344 ? 2900 : 2346)*cam->stride;

  vipc_server->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT, out_img_width, out_img_height, nv12_size, cam->stride, cam->uv_offset);
  LOGD("created %d YUV vipc buffers with size %dx%d", VIPC_BUFFER_COUNT, cam->stride, cam->y_height);
}

CameraBuf::~CameraBuf() {
  if (camera_bufs_raw != nullptr) {
    for (int i = 0; i < frame_buf_count; i++) {
      camera_bufs_raw[i].free();
    }
  }
}

void CameraBuf::sendFrameToVipc() {
  assert(cur_buf_idx >=0 && cur_buf_idx < frame_buf_count);

  if (camera_bufs_raw) {
    cur_camera_buf = &camera_bufs_raw[cur_buf_idx];
  }

  cur_yuv_buf = vipc_server->get_buffer(stream_type, cur_buf_idx);

  VisionIpcBufExtra extra = {
    cur_frame_data.frame_id,
    cur_frame_data.timestamp_sof,
    cur_frame_data.timestamp_eof,
  };
  cur_yuv_buf->set_frame_id(cur_frame_data.frame_id);
  vipc_server->send(cur_yuv_buf, &extra);
}

// common functions

kj::Array<uint8_t> get_raw_frame_image(const CameraBuf *b) {
  const uint8_t *dat = (const uint8_t *)b->cur_camera_buf->addr;

  kj::Array<uint8_t> frame_image = kj::heapArray<uint8_t>(b->cur_camera_buf->len);
  uint8_t *resized_dat = frame_image.begin();

  memcpy(resized_dat, dat, b->cur_camera_buf->len);

  return kj::mv(frame_image);
}

float calculate_exposure_value(const CameraBuf *b, Rect ae_xywh, int x_skip, int y_skip) {
  int lum_med;
  uint32_t lum_binning[256] = {0};
  const uint8_t *pix_ptr = b->cur_yuv_buf->y;

  unsigned int lum_total = 0;
  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = pix_ptr[(y * b->out_img_width) + x];
      lum_binning[lum]++;
      lum_total += 1;
    }
  }

  // Find mean lumimance value
  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];

    if (lum_cur >= lum_total / 2) {
      break;
    }
  }

  return lum_med / 256.0;
}

int open_v4l_by_name_and_index(const char name[], int index, int flags) {
  for (int v4l_index = 0; /**/; ++v4l_index) {
    std::string v4l_name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", v4l_index));
    if (v4l_name.empty()) return -1;
    if (v4l_name.find(name) == 0) {
      if (index == 0) {
        return HANDLE_EINTR(open(util::string_format("/dev/v4l-subdev%d", v4l_index).c_str(), flags));
      }
      index--;
    }
  }
}
