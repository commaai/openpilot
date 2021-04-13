#include "selfdrive/camerad/cameras/camera_common.h"

#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <chrono>
#include <thread>

#include "libyuv.h"
#include <jpeglib.h>

#include "selfdrive/camerad/imgproc/utils.h"
#include "selfdrive/common/clutil.h"
#include "selfdrive/common/modeldata.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

#ifdef QCOM
#include "selfdrive/camerad/cameras/camera_qcom.h"
#elif QCOM2
#include "selfdrive/camerad/cameras/camera_qcom2.h"
#elif WEBCAM
#include "selfdrive/camerad/cameras/camera_webcam.h"
#else
#include "selfdrive/camerad/cameras/camera_frame_stream.h"
#endif
#ifdef QCOM
#include "CL/cl_ext_qcom.h"
#endif

const int YUV_COUNT = 100;

static cl_program build_debayer_program(cl_device_id device_id, cl_context context, const CameraInfo *ci, const CameraBuf *b, const CameraState *s) {
  char args[4096];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DFRAME_WIDTH=%d -DFRAME_HEIGHT=%d -DFRAME_STRIDE=%d "
           "-DRGB_WIDTH=%d -DRGB_HEIGHT=%d -DRGB_STRIDE=%d "
           "-DBAYER_FLIP=%d -DHDR=%d -DCAM_NUM=%d",
           ci->frame_width, ci->frame_height, ci->frame_stride,
           b->rgb_width, b->rgb_height, b->rgb_stride,
           ci->bayer_flip, ci->hdr, s->camera_num);
  const char *cl_file = Hardware::TICI() ? "cameras/real_debayer.cl" : "cameras/debayer.cl";
  return cl_program_from_file(context, device_id, cl_file, args);
}

void CameraBuf::init(CameraServer* server, CameraState *s, int frame_cnt, release_cb release_callback) {
  vipc_server = server->vipc_server;
  if (s == &server->road_cam) {
    this->rgb_type = VISION_STREAM_RGB_BACK;
    this->yuv_type = VISION_STREAM_YUV_BACK;
  } else if (s == &server->driver_cam) {
    this->rgb_type = VISION_STREAM_RGB_FRONT;
    this->yuv_type = VISION_STREAM_YUV_FRONT;
  } else {
    this->rgb_type = VISION_STREAM_RGB_WIDE;
    this->yuv_type = VISION_STREAM_YUV_WIDE;
  }

  this->release_callback = release_callback;

  const CameraInfo *ci = &s->ci;
  camera_state = s;
  frame_buf_count = frame_cnt;

  // RAW frame
  const int frame_size = ci->frame_height * ci->frame_stride;
  camera_bufs = std::make_unique<VisionBuf[]>(frame_buf_count);
  camera_bufs_metadata = std::make_unique<FrameMetadata[]>(frame_buf_count);

  for (int i = 0; i < frame_buf_count; i++) {
    camera_bufs[i].allocate(frame_size);
    camera_bufs[i].init_cl(server->device_id, server->context);
  }

  rgb_width = ci->frame_width;
  rgb_height = ci->frame_height;

  if (!Hardware::TICI() && ci->bayer) {
    // debayering does a 2x downscale
    rgb_width = ci->frame_width / 2;
    rgb_height = ci->frame_height / 2;
  }

  yuv_transform = get_model_yuv_transform(ci->bayer);

  vipc_server->create_buffers(rgb_type, UI_BUF_COUNT, true, rgb_width, rgb_height);
  rgb_stride = vipc_server->get_buffer(rgb_type)->stride;

  vipc_server->create_buffers(yuv_type, YUV_COUNT, false, rgb_width, rgb_height);

  if (ci->bayer) {
    cl_program prg_debayer = build_debayer_program(server->device_id, server->context, ci, this, s);
    krnl_debayer = CL_CHECK_ERR(clCreateKernel(prg_debayer, "debayer10", &err));
    CL_CHECK(clReleaseProgram(prg_debayer));
  }

  rgb2yuv = std::make_unique<Rgb2Yuv>(server->context, server->device_id, rgb_width, rgb_height, rgb_stride);

#ifdef __APPLE__
  q = CL_CHECK_ERR(clCreateCommandQueue(server->context, server->device_id, 0, &err));
#else
  const cl_queue_properties props[] = {0};  //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  q = CL_CHECK_ERR(clCreateCommandQueueWithProperties(server->context, server->device_id, props, &err));
#endif
}

CameraBuf::~CameraBuf() {
  for (int i = 0; i < frame_buf_count; i++) {
    camera_bufs[i].free();
  }

  if (krnl_debayer) CL_CHECK(clReleaseKernel(krnl_debayer));
  if (q) CL_CHECK(clReleaseCommandQueue(q));
}

bool CameraBuf::acquire() {
  if (!safe_queue.try_pop(cur_buf_idx, 1)) return false;

  if (camera_bufs_metadata[cur_buf_idx].frame_id == -1) {
    LOGE("no frame data? wtf");
    release();
    return false;
  }

  cur_frame_data = camera_bufs_metadata[cur_buf_idx];
  cur_rgb_buf = vipc_server->get_buffer(rgb_type);

  cl_event debayer_event;
  cl_mem camrabuf_cl = camera_bufs[cur_buf_idx].buf_cl;
  if (camera_state->ci.bayer) {
    CL_CHECK(clSetKernelArg(krnl_debayer, 0, sizeof(cl_mem), &camrabuf_cl));
    CL_CHECK(clSetKernelArg(krnl_debayer, 1, sizeof(cl_mem), &cur_rgb_buf->buf_cl));
#ifdef QCOM2
    constexpr int localMemSize = (DEBAYER_LOCAL_WORKSIZE + 2 * (3 / 2)) * (DEBAYER_LOCAL_WORKSIZE + 2 * (3 / 2)) * sizeof(short int);
    const size_t globalWorkSize[] = {size_t(camera_state->ci.frame_width), size_t(camera_state->ci.frame_height)};
    const size_t localWorkSize[] = {DEBAYER_LOCAL_WORKSIZE, DEBAYER_LOCAL_WORKSIZE};
    CL_CHECK(clSetKernelArg(krnl_debayer, 2, localMemSize, 0));
    CL_CHECK(clEnqueueNDRangeKernel(q, krnl_debayer, 2, NULL, globalWorkSize, localWorkSize,
                                    0, 0, &debayer_event));
#else
    float digital_gain = camera_state->digital_gain;
    if ((int)digital_gain == 0) {
      digital_gain = 1.0;
    }
    CL_CHECK(clSetKernelArg(krnl_debayer, 2, sizeof(float), &digital_gain));
    const size_t debayer_work_size = rgb_height;  // doesn't divide evenly, is this okay?
    CL_CHECK(clEnqueueNDRangeKernel(q, krnl_debayer, 1, NULL,
                                    &debayer_work_size, NULL, 0, 0, &debayer_event));
#endif
  } else {
    assert(rgb_stride == camera_state->ci.frame_stride);
    CL_CHECK(clEnqueueCopyBuffer(q, camrabuf_cl, cur_rgb_buf->buf_cl, 0, 0,
                               cur_rgb_buf->len, 0, 0, &debayer_event));
  }

  clWaitForEvents(1, &debayer_event);
  CL_CHECK(clReleaseEvent(debayer_event));

  cur_yuv_buf = vipc_server->get_buffer(yuv_type);
  rgb2yuv->queue(q, cur_rgb_buf->buf_cl, cur_yuv_buf->buf_cl);

  VisionIpcBufExtra extra = {
                        cur_frame_data.frame_id,
                        cur_frame_data.timestamp_sof,
                        cur_frame_data.timestamp_eof,
  };
  vipc_server->send(cur_rgb_buf, &extra);
  vipc_server->send(cur_yuv_buf, &extra);

  return true;
}

void CameraBuf::release() {
  if (release_callback) {
    release_callback((void*)camera_state, cur_buf_idx);
  }
}

void CameraBuf::queue(size_t buf_idx) {
  safe_queue.push(buf_idx);
}

// common functions

static void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data) {
  framed.setFrameId(frame_data.frame_id);
  framed.setTimestampEof(frame_data.timestamp_eof);
  framed.setTimestampSof(frame_data.timestamp_sof);
  framed.setFrameLength(frame_data.frame_length);
  framed.setIntegLines(frame_data.integ_lines);
  framed.setGain(frame_data.gain);
  framed.setHighConversionGain(frame_data.high_conversion_gain);
  framed.setMeasuredGreyFraction(frame_data.measured_grey_fraction);
  framed.setTargetGreyFraction(frame_data.target_grey_fraction);
  framed.setLensPos(frame_data.lens_pos);
  framed.setLensSag(frame_data.lens_sag);
  framed.setLensErr(frame_data.lens_err);
  framed.setLensTruePos(frame_data.lens_true_pos);
}

kj::Array<uint8_t> get_frame_image(const CameraBuf *b) {
  static const int x_min = util::getenv("XMIN", 0);
  static const int y_min = util::getenv("YMIN", 0);
  static const int env_xmax = util::getenv("XMAX", -1);
  static const int env_ymax = util::getenv("YMAX", -1);
  static const int scale = util::getenv("SCALE", 1);

  assert(b->cur_rgb_buf);

  const int x_max = env_xmax != -1 ? env_xmax : b->rgb_width - 1;
  const int y_max = env_ymax != -1 ? env_ymax : b->rgb_height - 1;
  const int new_width = (x_max - x_min + 1) / scale;
  const int new_height = (y_max - y_min + 1) / scale;
  const uint8_t *dat = (const uint8_t *)b->cur_rgb_buf->addr;

  kj::Array<uint8_t> frame_image = kj::heapArray<uint8_t>(new_width*new_height*3);
  uint8_t *resized_dat = frame_image.begin();
  int goff = x_min*3 + y_min*b->rgb_stride;
  for (int r=0;r<new_height;r++) {
    for (int c=0;c<new_width;c++) {
      memcpy(&resized_dat[(r*new_width+c)*3], &dat[goff+r*b->rgb_stride*scale+c*3*scale], 3*sizeof(uint8_t));
    }
  }
  return kj::mv(frame_image);
}

static void publish_thumbnail(PubMaster *pm, const CameraBuf *b) {
  uint8_t* thumbnail_buffer = NULL;
  unsigned long thumbnail_len = 0;

  unsigned char *row = (unsigned char *)malloc(b->rgb_width/4*3);

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &thumbnail_buffer, &thumbnail_len);

  cinfo.image_width = b->rgb_width / 4;
  cinfo.image_height = b->rgb_height / 4;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
#ifndef __APPLE__
  jpeg_set_quality(&cinfo, 50, true);
  jpeg_start_compress(&cinfo, true);
#else
  jpeg_set_quality(&cinfo, 50, static_cast<boolean>(true) );
  jpeg_start_compress(&cinfo, static_cast<boolean>(true) );
#endif

  JSAMPROW row_pointer[1];
  const uint8_t *bgr_ptr = (const uint8_t *)b->cur_rgb_buf->addr;
  for (int ii = 0; ii < b->rgb_height/4; ii+=1) {
    for (int j = 0; j < b->rgb_width*3; j+=12) {
      for (int k = 0; k < 3; k++) {
        uint16_t dat = 0;
        int i = ii * 4;
        dat += bgr_ptr[b->rgb_stride*i + j + k];
        dat += bgr_ptr[b->rgb_stride*i + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+1) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+1) + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+2) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+2) + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+3) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+3) + j+3 + k];
        row[(j/4) + (2-k)] = dat/8;
      }
    }
    row_pointer[0] = row;
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  free(row);

  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initThumbnail();
  thumbnaild.setFrameId(b->cur_frame_data.frame_id);
  thumbnaild.setTimestampEof(b->cur_frame_data.timestamp_eof);
  thumbnaild.setThumbnail(kj::arrayPtr((const uint8_t*)thumbnail_buffer, thumbnail_len));

  pm->send("thumbnail", msg);
  free(thumbnail_buffer);
}

float set_exposure_target(const CameraBuf *b, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip) {
  int lum_med;
  uint32_t lum_binning[256] = {0};
  const uint8_t *pix_ptr = b->cur_yuv_buf->y;

  unsigned int lum_total = 0;
  for (int y = rect.y1; y < rect.y2; y += rect.y_skip) {
    for (int x = rect.x1; x < rect.x2; x += rect.x_skip) {
      uint8_t lum = pix_ptr[(y * b->rgb_width) + x];
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

static void road_cam_auto_exposure(CameraState *c) {
#ifndef QCOM2
  const ExpRect rect = {290, 850, 1, 322, 636, 1};
  int analog_gain = -1;
#else
  const ExpRect = (c == &s->wide_road_cam) ? {96, 1830, 2, 250, 774, 2}
                                           : {96, 1830, 2, 160, 1146, 2};
  int analog_gain = (int)c->analog_gain;
#endif
  camera_autoexposure(c, set_exposure_target(&c->buf, rect, analog_gain, false, false));
}

static void driver_cam_auto_exposure(CameraState *c) {
  static const bool is_rhd = Params().getBool("IsRHD");
  static SubMaster sm({"driverState"});

  const CameraBuf *b = &c->buf;

  int x_offset = 0, y_offset = 0;
  int frame_width = b->rgb_width, frame_height = b->rgb_height;


  ExpRect def_rect;
  if (Hardware::TICI()) {
    x_offset = 630, y_offset = 156;
    frame_width = 668, frame_height = frame_width / 1.33;
    def_rect = {96, 1832, 2, 242, 1148, 4};
  } else {
    def_rect = {is_rhd ? 0 : b->rgb_width * 3 / 5, is_rhd ? b->rgb_width * 2 / 5 : b->rgb_width, 2,
                b->rgb_height / 3, b->rgb_height, 1};
  }

  static ExpRect rect = def_rect;
  // use driver face crop for AE
  if (Hardware::EON() && sm.updated("driverState")) {
    if (auto state = sm["driverState"].getDriverState(); state.getFaceProb() > 0.4) {
      auto face_position = state.getFacePosition();
      int x = is_rhd ? 0 : frame_width - (0.5 * frame_height);
      x += (face_position[0] * (is_rhd ? -1.0 : 1.0) + 0.5) * (0.5 * frame_height) + x_offset;
      int y = (face_position[1] + 0.5) * frame_height + y_offset;
      rect = {std::max(0, x - 72), std::min(b->rgb_width - 1, x + 72), 2,
              std::max(0, y - 72), std::min(b->rgb_height - 1, y + 72), 1};
    }
  }

  camera_autoexposure(c, set_exposure_target(b, rect.x1, rect.x2, rect.x_skip, rect.y1, rect.y2, rect.y_skip));
}

void common_process_driver_camera(PubMaster *pm, CameraState *c, int cnt) {
  int j = Hardware::TICI() ? 1 : 3;
  if (cnt % j == 0) {
    driver_cam_auto_exposure(c);
  }
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data);
  if (env_send_driver) {
    framed.setImage(get_frame_image(&c->buf));
  }
  pm->send("driverCameraState", msg);
}

// CameraServerBase

CameraServerBase::CameraServerBase() {
  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  // TODO: do this for QCOM2 too
#if defined(QCOM)
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  context = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));
#else
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
#endif

  pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"
#ifdef QCOM2
    , "wideRoadCameraState"
#endif
  });
 
  vipc_server = new VisionIpcServer("camerad", device_id, context);
}

CameraServerBase::~CameraServerBase() {
  delete vipc_server;
  delete pm;
  CL_CHECK(clReleaseContext(context));
}

void CameraServerBase::start() {
  vipc_server->start_listener();
  run();
  LOG(" ************** stopping camera server **************");
  for (auto &t : camera_threads) t.join();
}

void CameraServerBase::start_process_thread(CameraState *cs, process_thread_cb callback, bool is_frame_stream) {
  camera_threads.push_back(std::thread(&CameraServerBase::process_camera, this, cs, callback, is_frame_stream));
}

ExitHandler do_exit;

void CameraServerBase::process_camera(CameraState *cs, process_thread_cb callback, bool is_frame_stream) {
  CameraServer *server = (CameraServer *)this;
  const char *thread_name, *cam_state_name;
  bool set_image = false, set_transform = false, pub_thumbnail = false;
  ::cereal::FrameData::Builder (cereal::Event::Builder::*init_cam_state_func)() = nullptr;

  if (cs == &server->road_cam) {
    thread_name = "RoadCamera";
    cam_state_name = "roadCameraState";
    set_image = getenv("SEND_ROAD") != NULL;
    pub_thumbnail = set_transform = true;
    init_cam_state_func = &cereal::Event::Builder::initRoadCameraState;
  } else if (cs == &server->driver_cam) {
    thread_name = "DriverCamera";
    cam_state_name = "driverCameraState";
    set_image = getenv("SEND_DRIVER") != NULL;
    init_cam_state_func = &cereal::Event::Builder::initDriverCameraState;
  } else {
    thread_name = "WideRoadCamera";
    cam_state_name = "wideRoadCameraState";
    set_image = getenv("SEND_WIDE_ROAD") != NULL;
    init_cam_state_func = &cereal::Event::Builder::initWideRoadCameraState;
  }

  set_thread_name(thread_name);

  uint32_t cnt = 0;
  while (!do_exit) {
    if (!cs->buf.acquire()) continue;

    // process camera buffer
    if (!is_frame_stream) {
      MessageBuilder msg;
      cereal::FrameData::Builder framed = (msg.initEvent().*init_cam_state_func)();

      // fill and send FrameData
      fill_frame_data(framed, cs->buf.cur_frame_data);
      if (set_transform) {
        framed.setTransform(cs->buf.yuv_transform.v);
      }
      if (set_image) {
        framed.setImage(get_frame_image(&cs->buf));
      }
      if (callback) {
        callback(server, cs, framed, cnt);
      }
      pm->send(cam_state_name, msg);

      // auto exposure
      if (cnt % 3 == 0) {
        cs == &server->driver_cam ? driver_cam_auto_exposure(cs)
                                  : road_cam_auto_exposure(cs);
      }

      // pub thumbnail
      if (pub_thumbnail && cnt % 100 == 3) {
        // this takes 10ms???
        publish_thumbnail(pm, &(cs->buf));
      }
    }

    cs->buf.release();
    ++cnt;
  }
}
