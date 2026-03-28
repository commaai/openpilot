#include "system/camerad/cameras/camera_v4l2.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/media.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"

extern ExitHandler do_exit;
extern const bool env_debug_frames;
extern const bool env_log_raw_frames;
extern const bool env_ctrl_exp_from_params;

// V4L2 pipeline config for each camera on comma 3X (tizi)
static const V4L2CameraConfig V4L2_WIDE_CONFIG = {
  .sensor_name = "ox03c10 16-0036",
  .csiphy_name = "msm_csiphy0",
  .csid_name = "msm_csid0",
  .vfe_pix_name = "msm_vfe0_pix",
  .video_dev_index = 3,
};

static const V4L2CameraConfig V4L2_ROAD_CONFIG = {
  .sensor_name = "ox03c10 16-0010",
  .csiphy_name = "msm_csiphy1",
  .csid_name = "msm_csid1",
  .vfe_pix_name = "msm_vfe1_pix",
  .video_dev_index = 7,
};

static const V4L2CameraConfig *v4l2_config_for_camera(int camera_num) {
  switch (camera_num) {
    case 0: return &V4L2_WIDE_CONFIG;
    case 1: return &V4L2_ROAD_CONFIG;
    default: return nullptr;
  }
}

// *** Media controller helpers ***

// Use media-ctl shell commands for pipeline setup (handles format negotiation correctly)
static void media_ctl_set_fmt(const char *entity, int pad, const char *fmt_str) {
  std::string cmd = util::string_format(
    "media-ctl -d /dev/media0 --set-v4l2 '\"%s\":%d[fmt:%s]'", entity, pad, fmt_str);
  system(cmd.c_str());
}

static void media_ctl_set_link(const char *src, int src_pad, const char *sink, int sink_pad) {
  std::string cmd = util::string_format(
    "media-ctl -d /dev/media0 -l '\"%s\":%d->\"%s\":%d[1]'", src, src_pad, sink, sink_pad);
  system(cmd.c_str());
}

// *** V4L2Camera implementation ***

V4L2Camera::V4L2Camera(const CameraConfig &config, const V4L2CameraConfig &v4l2_config)
    : enabled(config.enabled), cc(config), v4l2_cc(v4l2_config) {
  sensor = std::make_unique<OX03C10>();

  // NV12 output dimensions (ISP produces 1920x1208 from 1928x1224 input)
  stride = 2048;      // aligned stride
  y_height = 1208;
  uv_height = 604;
  uv_offset = stride * y_height;
  yuv_size = stride * y_height + stride * uv_height;
}

V4L2Camera::~V4L2Camera() {
  camera_close();
}

bool is_mainline_camss() {
  // Mainline CAMSS creates /dev/media0 but has no cam-req-mgr
  return util::file_exists("/dev/media0") &&
         open_v4l_by_name_and_index("cam-req-mgr", 0, O_RDWR | O_NONBLOCK) < 0;
}

int V4L2Camera::find_sensor_subdev() {
  for (int i = 0; i < 30; i++) {
    std::string name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", i));
    if (name.empty()) break;
    if (name.find(v4l2_cc.sensor_name) == 0) {
      return HANDLE_EINTR(open(util::string_format("/dev/v4l-subdev%d", i).c_str(), O_RDWR));
    }
  }
  return -1;
}

void V4L2Camera::setup_media_links() {
  media_ctl_set_link(v4l2_cc.csiphy_name, 1, v4l2_cc.csid_name, 0);
  media_ctl_set_link(v4l2_cc.csid_name, 4, v4l2_cc.vfe_pix_name, 0);
}

void V4L2Camera::setup_formats() {
  const char *bayer = "SGRBG12_1X12/1928x1224";
  media_ctl_set_fmt(v4l2_cc.sensor_name, 0, bayer);
  media_ctl_set_fmt(v4l2_cc.csiphy_name, 1, bayer);
  media_ctl_set_fmt(v4l2_cc.csid_name, 0, bayer);
  media_ctl_set_fmt(v4l2_cc.csid_name, 4, bayer);
  media_ctl_set_fmt(v4l2_cc.vfe_pix_name, 0, bayer);
}

void V4L2Camera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  LOG("V4L2: opening camera %d (%s)", cc.camera_num, v4l2_cc.sensor_name);

  // Open media device
  media_fd = HANDLE_EINTR(open("/dev/media0", O_RDWR));
  assert(media_fd >= 0);

  // Set up media links and formats
  setup_media_links();
  setup_formats();

  // Open video capture device
  std::string video_path = util::string_format("/dev/video%d", v4l2_cc.video_dev_index);
  video_fd = HANDLE_EINTR(open(video_path.c_str(), O_RDWR));
  assert(video_fd >= 0);

  // Set NV12 output format
  struct v4l2_format fmt = {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.width = 1920;
  fmt.fmt.pix_mp.height = 1224;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  fmt.fmt.pix_mp.num_planes = 1;
  int ret = ioctl(video_fd, VIDIOC_S_FMT, &fmt);
  assert(ret == 0);

  // Read back actual format
  ioctl(video_fd, VIDIOC_G_FMT, &fmt);
  stride = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
  LOG("V4L2: format %dx%d stride=%d", fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height, stride);

  // Request buffers
  struct v4l2_requestbuffers reqbufs = {};
  reqbufs.count = NUM_BUFFERS;
  reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbufs.memory = V4L2_MEMORY_MMAP;
  ret = ioctl(video_fd, VIDIOC_REQBUFS, &reqbufs);
  assert(ret == 0);

  // mmap buffers
  for (uint32_t i = 0; i < reqbufs.count; i++) {
    struct v4l2_buffer vbuf = {};
    struct v4l2_plane planes[1] = {};
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    vbuf.index = i;
    vbuf.m.planes = planes;
    vbuf.length = 1;
    ret = ioctl(video_fd, VIDIOC_QUERYBUF, &vbuf);
    assert(ret == 0);

    mmap_lengths[i] = planes[0].length;
    mmap_bufs[i] = mmap(NULL, planes[0].length, PROT_READ | PROT_WRITE,
                        MAP_SHARED, video_fd, planes[0].m.mem_offset);
    assert(mmap_bufs[i] != MAP_FAILED);
  }

  // Open sensor subdev for exposure control
  sensor_subdev_fd = find_sensor_subdev();
  assert(sensor_subdev_fd >= 0);

  // Initialize CameraBuf with VIPC buffers
  buf.out_img_width = 1920;
  buf.out_img_height = y_height;
  buf.init(v, VIPC_BUFFER_COUNT, cc.stream_type, yuv_size, stride, uv_offset);

  LOG("V4L2: camera %d opened, %d buffers", cc.camera_num, reqbufs.count);
}

void V4L2Camera::sensors_start() {
  if (!enabled) return;

  // Set initial exposure
  set_exposure(5, 256);

  // Queue all buffers
  for (int i = 0; i < NUM_BUFFERS; i++) {
    queue_buffer(i);
  }

  // Start streaming
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  int ret = ioctl(video_fd, VIDIOC_STREAMON, &type);
  assert(ret == 0);

  LOG("V4L2: camera %d streaming", cc.camera_num);
}

void V4L2Camera::queue_buffer(int idx) {
  struct v4l2_buffer qbuf = {};
  struct v4l2_plane planes[1] = {};
  qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.index = idx;
  qbuf.m.planes = planes;
  qbuf.length = 1;
  ioctl(video_fd, VIDIOC_QBUF, &qbuf);
}

int V4L2Camera::capture_frame() {
  struct v4l2_buffer vbuf = {};
  struct v4l2_plane planes[1] = {};
  vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vbuf.memory = V4L2_MEMORY_MMAP;
  vbuf.m.planes = planes;
  vbuf.length = 1;

  int ret = ioctl(video_fd, VIDIOC_DQBUF, &vbuf);
  if (ret < 0) return -1;

  int idx = vbuf.index;

  // Copy frame data to VIPC buffer
  VisionBuf *vipc_buf = buf.vipc_server->get_buffer(cc.stream_type, idx % VIPC_BUFFER_COUNT);
  memcpy(vipc_buf->addr, mmap_bufs[idx], std::min((size_t)yuv_size, mmap_lengths[idx]));

  // Fill frame metadata
  static uint32_t frame_counter = 0;
  buf.cur_buf_idx = idx % VIPC_BUFFER_COUNT;
  buf.cur_frame_data.frame_id = frame_counter++;
  buf.cur_frame_data.request_id = buf.cur_frame_data.frame_id;
  buf.cur_frame_data.timestamp_sof = (uint64_t)vbuf.timestamp.tv_sec * 1000000000ULL +
                                      (uint64_t)vbuf.timestamp.tv_usec * 1000ULL;
  buf.cur_frame_data.timestamp_eof = buf.cur_frame_data.timestamp_sof + sensor->readout_time_ns;
  buf.cur_frame_data.processing_time = (nanos_since_boot() - buf.cur_frame_data.timestamp_eof) / 1e9;

  return idx;
}

void V4L2Camera::set_exposure(int exposure_time, int analogue_gain) {
  if (sensor_subdev_fd < 0) return;

  struct v4l2_control ctrl = {};
  ctrl.id = V4L2_CID_EXPOSURE;
  ctrl.value = exposure_time;
  ioctl(sensor_subdev_fd, VIDIOC_S_CTRL, &ctrl);

  ctrl.id = V4L2_CID_ANALOGUE_GAIN;
  ctrl.value = analogue_gain;
  ioctl(sensor_subdev_fd, VIDIOC_S_CTRL, &ctrl);
}

void V4L2Camera::camera_close() {
  if (video_fd >= 0) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);

    for (int i = 0; i < NUM_BUFFERS; i++) {
      if (mmap_bufs[i]) {
        munmap(mmap_bufs[i], mmap_lengths[i]);
        mmap_bufs[i] = nullptr;
      }
    }
    close(video_fd);
    video_fd = -1;
  }
  if (sensor_subdev_fd >= 0) { close(sensor_subdev_fd); sensor_subdev_fd = -1; }
  if (media_fd >= 0) { close(media_fd); media_fd = -1; }
}

// *** CameraState for V4L2 (reuses AE logic) ***

class V4L2CameraState {
public:
  V4L2Camera camera;
  int exposure_time = 5;
  bool dc_gain_enabled = false;
  int dc_gain_weight = 0;
  int gain_idx = 0;
  float analog_gain_frac = 0;
  float cur_ev[3] = {};
  float best_ev_score = 0;
  int new_exp_g = 0;
  int new_exp_t = 0;
  Rect ae_xywh = {};
  float measured_grey_fraction = 0;
  float target_grey_fraction = 0.125;
  float fl_pix = 0;
  std::unique_ptr<PubMaster> pm;

  V4L2CameraState(const CameraConfig &config, const V4L2CameraConfig &v4l2_config)
      : camera(config, v4l2_config) {}

  void init(VisionIpcServer *v) {
    camera.camera_open(v);
    if (!camera.enabled) return;

    fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm / camera.sensor->out_scale;
    set_exposure_rect();
    dc_gain_weight = camera.sensor->dc_gain_min_weight;
    gain_idx = camera.sensor->analog_gain_rec_idx;
    cur_ev[0] = cur_ev[1] = cur_ev[2] = get_gain_factor() * camera.sensor->sensor_analog_gains[gain_idx] * exposure_time;
    pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});
  }

  float get_gain_factor() const {
    return (1 + dc_gain_weight * (camera.sensor->dc_gain_factor - 1) / camera.sensor->dc_gain_max_weight);
  }

  void set_exposure_rect() {
    std::vector<std::pair<Rect, float>> ae_targets = {
      std::make_pair((Rect){96, 400, 1734, 524}, 567.0),
      std::make_pair((Rect){96, 160, 1734, 986}, 2648.0),
      std::make_pair((Rect){96, 242, 1736, 906}, 567.0)
    };
    int h_ref = 1208;
    auto ae_target = ae_targets[camera.cc.camera_num];
    Rect xywh_ref = ae_target.first;
    float fl_ref = ae_target.second;
    ae_xywh = (Rect){
      std::max(0, (int)camera.buf.out_img_width / 2 - (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
      std::max(0, (int)camera.buf.out_img_height / 2 - (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y))),
      std::min((int)(fl_pix / fl_ref * xywh_ref.w), (int)camera.buf.out_img_width / 2 + (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
      std::min((int)(fl_pix / fl_ref * xywh_ref.h), (int)camera.buf.out_img_height / 2 + (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y)))
    };
  }

  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
    float score = camera.sensor->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
    if (score < best_ev_score) {
      new_exp_t = exp_t;
      new_exp_g = exp_g_idx;
      best_ev_score = score;
    }
  }

  void set_camera_exposure(float grey_frac) {
    if (!camera.enabled) return;
    std::vector<double> target_grey_minimums = {0.1, 0.1, 0.125};

    const float dt = 0.05;
    const float ts_grey = 10.0;
    const float ts_ev = 0.05;
    const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
    const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

    const auto &s = camera.sensor;
    const float cur_ev_ = cur_ev[(camera.buf.cur_frame_data.frame_id - 1) % 3] * s->ev_scale;

    float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + s->target_grey_factor * cur_ev_) / log2(6000.0),
                                       target_grey_minimums[camera.cc.camera_num], 0.4);
    float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;
    float desired_ev = std::clamp(cur_ev_ / s->ev_scale * target_grey / grey_frac, s->min_ev, s->max_ev);
    float k = (1.0 - k_ev) / 3.0;
    desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

    best_ev_score = 1e6;
    new_exp_g = 0;
    new_exp_t = 0;

    bool enable_dc_gain = dc_gain_enabled;
    if (!enable_dc_gain && target_grey < s->dc_gain_on_grey) {
      enable_dc_gain = true;
      dc_gain_weight = s->dc_gain_min_weight;
    } else if (enable_dc_gain && target_grey > s->dc_gain_off_grey) {
      enable_dc_gain = false;
      dc_gain_weight = s->dc_gain_max_weight;
    }
    if (enable_dc_gain && dc_gain_weight < s->dc_gain_max_weight) dc_gain_weight += 1;
    if (!enable_dc_gain && dc_gain_weight > s->dc_gain_min_weight) dc_gain_weight -= 1;

    int min_g = std::max(gain_idx - 1, s->analog_gain_min_idx);
    int max_g = std::min(gain_idx + 1, s->analog_gain_max_idx);
    for (int g = min_g; g <= max_g; g++) {
      float gain = s->sensor_analog_gains[g] * get_gain_factor();
      int t = std::clamp(int(std::round(desired_ev / gain)), s->exposure_time_min, s->exposure_time_max);
      if (g < s->analog_gain_rec_idx && t > 20 && g < gain_idx) continue;
      update_exposure_score(desired_ev, t, g, gain);
    }

    measured_grey_fraction = grey_frac;
    target_grey_fraction = target_grey;
    analog_gain_frac = s->sensor_analog_gains[new_exp_g];
    gain_idx = new_exp_g;
    exposure_time = new_exp_t;
    dc_gain_enabled = enable_dc_gain;

    float gain = analog_gain_frac * get_gain_factor();
    cur_ev[camera.buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

    // Set exposure via V4L2 controls (map gain_idx to register value)
    camera.set_exposure(exposure_time, (int)(analog_gain_frac * 256));
  }

  void sendState() {
    camera.buf.sendFrameToVipc();

    MessageBuilder msg;
    auto framed = (msg.initEvent().*camera.cc.init_camera_state)();
    const FrameMetadata &meta = camera.buf.cur_frame_data;
    framed.setFrameId(meta.frame_id);
    framed.setRequestId(meta.request_id);
    framed.setTimestampEof(meta.timestamp_eof);
    framed.setTimestampSof(meta.timestamp_sof);
    framed.setIntegLines(exposure_time);
    framed.setGain(analog_gain_frac * get_gain_factor());
    framed.setHighConversionGain(dc_gain_enabled);
    framed.setMeasuredGreyFraction(measured_grey_fraction);
    framed.setTargetGreyFraction(target_grey_fraction);
    framed.setProcessingTime(meta.processing_time);

    const float ev = cur_ev[meta.frame_id % 3];
    const float perc = util::map_val(ev, camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f);
    framed.setExposureValPercent(perc);
    framed.setSensor(camera.sensor->image_sensor);

    set_camera_exposure(calculate_exposure_value(&camera.buf, ae_xywh, 2,
                        camera.cc.stream_type != VISION_STREAM_DRIVER ? 2 : 4));

    pm->send(camera.cc.publish_name, msg);
  }
};

// *** Main V4L2 camera thread ***

void camerad_thread_v4l2() {
  fprintf(stderr, "V4L2: starting camerad with mainline CAMSS driver\n");
  LOG("V4L2: starting camerad with mainline CAMSS driver");

  VisionIpcServer v("camerad");

  std::vector<std::unique_ptr<V4L2CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    const V4L2CameraConfig *v4l2_cfg = v4l2_config_for_camera(config.camera_num);
    if (!v4l2_cfg) {
      LOG("V4L2: skipping camera %d (no V4L2 config)", config.camera_num);
      continue;
    }
    auto cam = std::make_unique<V4L2CameraState>(config, *v4l2_cfg);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  LOG("V4L2: starting streams");
  for (auto &cam : cams) cam->camera.sensors_start();

  // Build poll FD array
  std::vector<struct pollfd> fds;
  for (auto &cam : cams) {
    fds.push_back({.fd = cam->camera.video_fd, .events = POLLIN});
  }

  LOG("V4L2: entering capture loop");
  while (!do_exit) {
    int ret = poll(fds.data(), fds.size(), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("V4L2: poll failed (%d)", errno);
      break;
    }

    for (size_t i = 0; i < cams.size(); i++) {
      if (!(fds[i].revents & POLLIN)) continue;

      int buf_idx = cams[i]->camera.capture_frame();
      if (buf_idx >= 0) {
        cams[i]->sendState();
        cams[i]->camera.queue_buffer(buf_idx);
      }
    }
  }

  LOG("V4L2: stopping cameras");
  for (auto &cam : cams) cam->camera.camera_close();
}
