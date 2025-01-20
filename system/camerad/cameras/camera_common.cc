#include "system/camerad/cameras/camera_common.h"

#include <cassert>
#include <string>

#include <jpeglib.h>

#include "common/clutil.h"
#include "common/swaglog.h"

#include "system/camerad/cameras/spectra.h"


class ImgProc {
public:
  ImgProc(cl_device_id device_id, cl_context context, const CameraBuf *b, const SensorInfo *sensor, int camera_num, int buf_width, int uv_offset) {
    char args[4096];
    snprintf(args, sizeof(args),
             "-cl-fast-relaxed-math -cl-denorms-are-zero -Isensors "
             "-DFRAME_WIDTH=%d -DFRAME_HEIGHT=%d -DFRAME_STRIDE=%d -DFRAME_OFFSET=%d "
             "-DRGB_WIDTH=%d -DRGB_HEIGHT=%d -DYUV_STRIDE=%d -DUV_OFFSET=%d "
             "-DSENSOR_ID=%hu -DHDR_OFFSET=%d -DVIGNETTING=%d ",
             sensor->frame_width, sensor->frame_height, sensor->hdr_offset > 0 ? sensor->frame_stride * 2 : sensor->frame_stride, sensor->frame_offset,
             b->out_img_width, b->out_img_height, buf_width, uv_offset,
             static_cast<unsigned short>(sensor->image_sensor), sensor->hdr_offset, camera_num == 1);
    const char *cl_file = "cameras/process_raw.cl";
    cl_program prg_imgproc = cl_program_from_file(context, device_id, cl_file, args);
    krnl_ = CL_CHECK_ERR(clCreateKernel(prg_imgproc, "process_raw", &err));
    CL_CHECK(clReleaseProgram(prg_imgproc));

    const cl_queue_properties props[] = {0};  //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
    queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device_id, props, &err));
  }

  void runKernel(cl_mem cam_buf_cl, cl_mem buf_cl, int width, int height, int expo_time) {
    CL_CHECK(clSetKernelArg(krnl_, 0, sizeof(cl_mem), &cam_buf_cl));
    CL_CHECK(clSetKernelArg(krnl_, 1, sizeof(cl_mem), &buf_cl));
    CL_CHECK(clSetKernelArg(krnl_, 2, sizeof(cl_int), &expo_time));

    const size_t globalWorkSize[] = {size_t(width / 2), size_t(height / 2)};
    const int imgproc_local_worksize = 16;
    const size_t localWorkSize[] = {imgproc_local_worksize, imgproc_local_worksize};

    cl_event event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, krnl_, 2, NULL, globalWorkSize, localWorkSize, 0, 0, &event));
    clWaitForEvents(1, &event);
    CL_CHECK(clReleaseEvent(event));
  }

  ~ImgProc() {
    CL_CHECK(clReleaseKernel(krnl_));
    CL_CHECK(clReleaseCommandQueue(queue));
  }

private:
  cl_kernel krnl_;
  cl_command_queue queue;
};

void CameraBuf::init(cl_device_id device_id, cl_context context, SpectraCamera *cam, VisionIpcServer * v, int frame_cnt, VisionStreamType type) {
  vipc_server = v;
  stream_type = type;
  frame_buf_count = frame_cnt;

  const SensorInfo *sensor = cam->sensor.get();

  is_raw = cam->is_raw;
  frame_metadata = std::make_unique<FrameMetadata[]>(frame_buf_count);

  // RAW frames from ISP
  if (is_raw) {
    camera_bufs_raw = std::make_unique<VisionBuf[]>(frame_buf_count);

    const int raw_frame_size = (sensor->frame_height + sensor->extra_height) * sensor->frame_stride;
    for (int i = 0; i < frame_buf_count; i++) {
      camera_bufs_raw[i].allocate(raw_frame_size);
      camera_bufs_raw[i].init_cl(device_id, context);
    }
    LOGD("allocated %d CL buffers", frame_buf_count);
  }

  out_img_width = sensor->frame_width;
  out_img_height = sensor->hdr_offset > 0 ? (sensor->frame_height - sensor->hdr_offset) / 2 : sensor->frame_height;

  // the encoder HW tells us the size it wants after setting it up.
  // TODO: VENUS_BUFFER_SIZE should give the size, but it's too small. dependent on encoder settings?
  size_t nv12_size = (out_img_width <= 1344 ? 2900 : 2346)*cam->stride;

  vipc_server->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT, out_img_width, out_img_height, nv12_size, cam->stride, cam->uv_offset);
  LOGD("created %d YUV vipc buffers with size %dx%d", VIPC_BUFFER_COUNT, cam->stride, cam->y_height);

  if (is_raw) imgproc = new ImgProc(device_id, context, this, sensor, cam->cc.camera_num, cam->stride, cam->uv_offset);
}

CameraBuf::~CameraBuf() {
  if (camera_bufs_raw != nullptr) {
    for (int i = 0; i < frame_buf_count; i++) {
      camera_bufs_raw[i].free();
    }
  }
  if (imgproc) delete imgproc;
}

bool CameraBuf::acquire(int expo_time) {
  if (!safe_queue.try_pop(cur_buf_idx, 50)) return false;

  if (frame_metadata[cur_buf_idx].frame_id == -1) {
    LOGE("no frame data? wtf");
    return false;
  }

  cur_frame_data = frame_metadata[cur_buf_idx];
  cur_camera_buf = &camera_bufs_raw[cur_buf_idx];
  if (is_raw) {
    cur_yuv_buf = vipc_server->get_buffer(stream_type);

    double start_time = millis_since_boot();
    imgproc->runKernel(camera_bufs_raw[cur_buf_idx].buf_cl, cur_yuv_buf->buf_cl, out_img_width, out_img_height, expo_time);
    cur_frame_data.processing_time = (millis_since_boot() - start_time) / 1000.0;
  } else {
    cur_yuv_buf = vipc_server->get_buffer(stream_type, cur_buf_idx);
    cur_frame_data.processing_time = (double)(cur_frame_data.timestamp_end_of_isp - cur_frame_data.timestamp_eof)*1e-9;
  }

  VisionIpcBufExtra extra = {
    cur_frame_data.frame_id,
    cur_frame_data.timestamp_sof,
    cur_frame_data.timestamp_eof,
  };
  cur_yuv_buf->set_frame_id(cur_frame_data.frame_id);
  vipc_server->send(cur_yuv_buf, &extra);

  return true;
}

void CameraBuf::queue(size_t buf_idx) {
  safe_queue.push(buf_idx);
}

// common functions

kj::Array<uint8_t> get_raw_frame_image(const CameraBuf *b) {
  const uint8_t *dat = (const uint8_t *)b->cur_camera_buf->addr;

  kj::Array<uint8_t> frame_image = kj::heapArray<uint8_t>(b->cur_camera_buf->len);
  uint8_t *resized_dat = frame_image.begin();

  memcpy(resized_dat, dat, b->cur_camera_buf->len);

  return kj::mv(frame_image);
}

static kj::Array<capnp::byte> yuv420_to_jpeg(const CameraBuf *b, int thumbnail_width, int thumbnail_height) {
  int downscale = b->cur_yuv_buf->width / thumbnail_width;
  assert(downscale * thumbnail_height == b->cur_yuv_buf->height);
  int in_stride = b->cur_yuv_buf->stride;

  // make the buffer big enough. jpeg_write_raw_data requires 16-pixels aligned height to be used.
  std::unique_ptr<uint8_t[]> buf(new uint8_t[(thumbnail_width * ((thumbnail_height + 15) & ~15) * 3) / 2]);
  uint8_t *y_plane = buf.get();
  uint8_t *u_plane = y_plane + thumbnail_width * thumbnail_height;
  uint8_t *v_plane = u_plane + (thumbnail_width * thumbnail_height) / 4;
  {
    // subsampled conversion from nv12 to yuv
    for (int hy = 0; hy < thumbnail_height/2; hy++) {
      for (int hx = 0; hx < thumbnail_width/2; hx++) {
        int ix = hx * downscale + (downscale-1)/2;
        int iy = hy * downscale + (downscale-1)/2;
        y_plane[(hy*2 + 0)*thumbnail_width + (hx*2 + 0)] = b->cur_yuv_buf->y[(iy*2 + 0) * in_stride + ix*2 + 0];
        y_plane[(hy*2 + 0)*thumbnail_width + (hx*2 + 1)] = b->cur_yuv_buf->y[(iy*2 + 0) * in_stride + ix*2 + 1];
        y_plane[(hy*2 + 1)*thumbnail_width + (hx*2 + 0)] = b->cur_yuv_buf->y[(iy*2 + 1) * in_stride + ix*2 + 0];
        y_plane[(hy*2 + 1)*thumbnail_width + (hx*2 + 1)] = b->cur_yuv_buf->y[(iy*2 + 1) * in_stride + ix*2 + 1];
        u_plane[hy*thumbnail_width/2 + hx] = b->cur_yuv_buf->uv[iy*in_stride + ix*2 + 0];
        v_plane[hy*thumbnail_width/2 + hx] = b->cur_yuv_buf->uv[iy*in_stride + ix*2 + 1];
      }
    }
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  uint8_t *thumbnail_buffer = nullptr;
  size_t thumbnail_len = 0;
  jpeg_mem_dest(&cinfo, &thumbnail_buffer, &thumbnail_len);

  cinfo.image_width = thumbnail_width;
  cinfo.image_height = thumbnail_height;
  cinfo.input_components = 3;

  jpeg_set_defaults(&cinfo);
  jpeg_set_colorspace(&cinfo, JCS_YCbCr);
  // configure sampling factors for yuv420.
  cinfo.comp_info[0].h_samp_factor = 2;  // Y
  cinfo.comp_info[0].v_samp_factor = 2;
  cinfo.comp_info[1].h_samp_factor = 1;  // U
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;  // V
  cinfo.comp_info[2].v_samp_factor = 1;
  cinfo.raw_data_in = TRUE;

  jpeg_set_quality(&cinfo, 50, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW y[16], u[8], v[8];
  JSAMPARRAY planes[3]{y, u, v};

  for (int line = 0; line < cinfo.image_height; line += 16) {
    for (int i = 0; i < 16; ++i) {
      y[i] = y_plane + (line + i) * cinfo.image_width;
      if (i % 2 == 0) {
        int offset = (cinfo.image_width / 2) * ((i + line) / 2);
        u[i / 2] = u_plane + offset;
        v[i / 2] = v_plane + offset;
      }
    }
    jpeg_write_raw_data(&cinfo, planes, 16);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  kj::Array<capnp::byte> dat = kj::heapArray<capnp::byte>(thumbnail_buffer, thumbnail_len);
  free(thumbnail_buffer);
  return dat;
}

void publish_thumbnail(PubMaster *pm, const CameraBuf *b) {
  auto thumbnail = yuv420_to_jpeg(b, b->out_img_width / 4, b->out_img_height / 4);
  if (thumbnail.size() == 0) return;

  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initThumbnail();
  thumbnaild.setFrameId(b->cur_frame_data.frame_id);
  thumbnaild.setTimestampEof(b->cur_frame_data.timestamp_eof);
  thumbnaild.setThumbnail(thumbnail);

  pm->send("thumbnail", msg);
}

float set_exposure_target(const CameraBuf *b, Rect ae_xywh, int x_skip, int y_skip) {
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
