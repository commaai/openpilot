#include "system/camerad/cameras/camera_common.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
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
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/nv12_info.h"
#include "system/camerad/sensors/sensor.h"


ExitHandler do_exit;

const bool env_debug_frames = getenv("DEBUG_FRAMES") != nullptr;
const int dragon_target_fps = 20;
const int dragon_frame_height = 1232;
const int dragon_frame_length_lines = 5123;

// Dragon Q6A camera routing:
//   RDI mode (raw passthrough): CSIPHY -> CSID pad 1 -> VFE RDI0 -> raw Bayer
// RDI requires CSID N -> VFE N pairing.
struct DragonCamConfig {
  uint32_t csiphy_entity;
  uint32_t csid_entity;
  uint32_t vfe_rdi_entity;
  int rdi_video_dev;
  const char *sensor_name;
  int csiphy_subdev;
  int csid_subdev;
  int vfe_rdi_subdev;
};

static int find_v4l_dev(const char *prefix, const char *name) {
  for (int i = 0; i < 64; i++) {
    auto path = util::string_format("/sys/class/video4linux/%s%d/name", prefix, i);
    auto dev_name = util::read_file(path);
    if (!dev_name.empty() && dev_name.find(name) == 0) return i;
  }
  return -1;
}

static uint32_t find_media_entity(int media_fd, const char *name) {
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strcmp(ent.name, name) == 0) return ent.id;
  }
  return 0;
}

static DragonCamConfig resolve_cam_config(int media_fd, int cam_idx) {
  static const struct { int csiphy; int csid; int rdi_vfe; const char *sensor; } routing[] = {
    {2, 2, 2, "imx219 18-0010"},
    {3, 3, 3, "imx219 21-0010"},
  };
  auto &r = routing[cam_idx];
  DragonCamConfig cfg = {};
  cfg.sensor_name = r.sensor;
  cfg.csiphy_entity = find_media_entity(media_fd, util::string_format("msm_csiphy%d", r.csiphy).c_str());
  cfg.csid_entity = find_media_entity(media_fd, util::string_format("msm_csid%d", r.csid).c_str());
  cfg.vfe_rdi_entity = find_media_entity(media_fd, util::string_format("msm_vfe%d_rdi0", r.rdi_vfe).c_str());
  cfg.rdi_video_dev = find_v4l_dev("video", util::string_format("msm_vfe%d_video0", r.rdi_vfe).c_str());
  cfg.csiphy_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csiphy%d", r.csiphy).c_str());
  cfg.csid_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csid%d", r.csid).c_str());
  cfg.vfe_rdi_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_vfe%d_rdi0", r.rdi_vfe).c_str());
  return cfg;
}

static DragonCamConfig dragon_cams[2];


struct V4LBuf {
  void *start;
  size_t length;
};

static inline uint16_t raw10_pixel(const uint8_t *raw, int row, int col, int stride) {
  int group = col / 4;
  int idx = col % 4;
  int off = row * stride + group * 5;
  return ((uint16_t)raw[off + idx] << 2) | ((raw[off + 4] >> (idx * 2)) & 0x3);
}

static uint8_t gamma_lut[1024];
static void __attribute__((constructor)) init_gamma_lut() {
  for (int i = 0; i < 1024; i++) {
    float v = i / 1023.0f;
    gamma_lut[i] = (uint8_t)std::min((int)(powf(v, 1.0f / 2.2f) * 255.0f + 0.5f), 255);
  }
}

static void set_dragon_sensor_timing(int sensor_fd, int cam_idx) {
  struct v4l2_control vblank_ctrl = {};
  vblank_ctrl.id = V4L2_CID_VBLANK;
  vblank_ctrl.value = dragon_frame_length_lines - dragon_frame_height;
  if (ioctl(sensor_fd, VIDIOC_S_CTRL, &vblank_ctrl) != 0) {
    LOGE("cam %d: set VBLANK failed: %d (%s)", cam_idx, errno, strerror(errno));
    return;
  }

  if (ioctl(sensor_fd, VIDIOC_G_CTRL, &vblank_ctrl) != 0) {
    LOGE("cam %d: read VBLANK failed after set: %d (%s)", cam_idx, errno, strerror(errno));
    return;
  }

  const int frame_length_lines = dragon_frame_height + vblank_ctrl.value;
  if (frame_length_lines != dragon_frame_length_lines) {
    LOGE("cam %d: VBLANK readback %d gives FLL=%d, expected FLL=%d for %dfps",
         cam_idx, vblank_ctrl.value, frame_length_lines, dragon_frame_length_lines, dragon_target_fps);
  } else {
    LOG("cam %d: VBLANK set to %d (FLL=%d, %dfps target)",
        cam_idx, vblank_ctrl.value, frame_length_lines, dragon_target_fps);
  }
}

static void debayer_raw10_to_nv12(const uint8_t *raw, int raw_stride,
                                   uint8_t *y_out, uint8_t *uv_out,
                                   int raw_width, int output_width, int height, int out_stride) {
  const int wb_r = 410;   // ~1.60x
  const int wb_g = 256;   // 1.00x
  const int wb_b = 390;   // ~1.52x
  const int black = 64;

  for (int y = 0; y < height - 1; y += 2) {
    const int dst_y0 = height - 2 - y;
    const int dst_y1 = height - 1 - y;
    const int dst_uv_y = height / 2 - 1 - y / 2;

    for (int x = 0; x < output_width - 1; x += 2) {
      int src_x = ((x * raw_width) / output_width) & ~1;
      src_x = std::min(src_x, raw_width - 2);
      const int dst_x0 = output_width - 2 - x;
      const int dst_x1 = output_width - 1 - x;

      int R  = std::max((int)raw10_pixel(raw, y, src_x, raw_stride) - black, 0);
      int Gr = std::max((int)raw10_pixel(raw, y, src_x + 1, raw_stride) - black, 0);
      int Gb = std::max((int)raw10_pixel(raw, y + 1, src_x, raw_stride) - black, 0);
      int B  = std::max((int)raw10_pixel(raw, y + 1, src_x + 1, raw_stride) - black, 0);

      int r10 = std::min((R * wb_r) >> 8, 1023);
      int g10_r = std::min((Gr * wb_g) >> 8, 1023);
      int g10_b = std::min((Gb * wb_g) >> 8, 1023);
      int b10 = std::min((B * wb_b) >> 8, 1023);

      // Gamma-corrected 8-bit for Y (perceptual luminance)
      uint8_t r8 = gamma_lut[r10];
      uint8_t gr8 = gamma_lut[g10_r];
      uint8_t gb8 = gamma_lut[g10_b];
      uint8_t b8 = gamma_lut[b10];

      uint8_t g8 = (gr8 + gb8 + 1) >> 1;
      y_out[dst_y1 * out_stride + dst_x1] = std::clamp((int)r8, 16, 235);
      y_out[dst_y1 * out_stride + dst_x0] = std::clamp((int)gr8, 16, 235);
      y_out[dst_y0 * out_stride + dst_x1] = std::clamp((int)gb8, 16, 235);
      y_out[dst_y0 * out_stride + dst_x0] = std::clamp((int)b8, 16, 235);

      // UV from gamma-corrected RGB (BT.601)
      uv_out[dst_uv_y * out_stride + dst_x0]     = std::clamp(((-38*r8 - 74*g8 + 112*b8 + 128) >> 8) + 128, 16, 240);
      uv_out[dst_uv_y * out_stride + dst_x0 + 1] = std::clamp(((112*r8 - 94*g8 - 18*b8 + 128) >> 8) + 128, 16, 240);
    }
  }
}

class DragonCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  CameraBuf buf;
  bool enabled;

  int video_fd = -1;
  int sensor_fd = -1;

  V4LBuf v4l_bufs[4];
  int n_bufs = 4;
  uint32_t frame_size = 0;

  uint32_t output_width = 0, output_height = 0;
  uint32_t stride = 0, y_height = 0, uv_height = 0, yuv_size = 0, uv_offset = 0;

  DragonCamera(const CameraConfig &config) : cc(config), enabled(config.enabled) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void setup_media_links();
  void set_formats();
  void queue_all_buffers();
  void stream_on();
  void start_streaming();
  void stop_streaming();
  int dequeue_frame(uint64_t *timestamp);
  void queue_frame(int index);
  void set_exposure(int exposure_time, int gain_idx);

  VisionIpcServer *vipc_server = nullptr;
  VisionStreamType stream_type;
};


static void reset_all_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) return;

  for (int i = 0; i < 20; i++) {
    std::string vpath = util::string_format("/dev/video%d", i);
    int vfd = open(vpath.c_str(), O_RDWR);
    if (vfd >= 0) {
      int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      close(vfd);
    }
  }

  std::vector<uint32_t> csid_ents, vfe_ents, csiphy_ents;
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strncmp(ent.name, "msm_csiphy", 10) == 0) csiphy_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_csid", 8) == 0) csid_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_vfe", 7) == 0) vfe_ents.push_back(ent.id);
  }
  for (uint32_t csiphy : csiphy_ents) {
    for (uint32_t csid : csid_ents) {
      struct media_link_desc link = {};
      link.source = {.entity = csiphy, .index = 1};
      link.sink = {.entity = csid, .index = 0};
      link.flags = 0;
      ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
    }
  }
  for (uint32_t csid : csid_ents) {
    for (uint32_t vfe : vfe_ents) {
      for (int pad = 1; pad <= 4; pad++) {
        struct media_link_desc link = {};
        link.source = {.entity = csid, .index = (uint16_t)pad};
        link.sink = {.entity = vfe, .index = 0};
        link.flags = 0;
        ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
      }
    }
  }
  close(media_fd);
  LOG("reset all media links");
}

void DragonCamera::setup_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }

  int cam_idx = cc.camera_num;
  auto &dcfg = dragon_cams[cam_idx];

  struct media_link_desc link = {};

  // CSIPHY → CSID (source pad 1 → sink pad 0)
  link.source = {.entity = (uint32_t)dcfg.csiphy_entity, .index = 1};
  link.sink = {.entity = (uint32_t)dcfg.csid_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csiphy->csid link FAILED: %d (%s)", cam_idx, errno, strerror(errno));

  memset(&link, 0, sizeof(link));
  // CSID pad 1 -> VFE_RDI0 pad 0
  link.source = {.entity = (uint32_t)dcfg.csid_entity, .index = 1};
  link.sink = {.entity = (uint32_t)dcfg.vfe_rdi_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csid->vfe_rdi link FAILED: %d (%s)", cam_idx, errno, strerror(errno));

  close(media_fd);
  LOG("cam %d: media links set up (RDI mode)", cam_idx);
}

void DragonCamera::set_formats() {
  int cam_idx = cc.camera_num;
  auto &dcfg = dragon_cams[cam_idx];

  // set format on CSIPHY subdev
  int csiphy_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csiphy_subdev).c_str(), O_RDWR);
  if (csiphy_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = 0x300f;  // MEDIA_BUS_FMT_SRGGB10_1X10
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    sfmt.pad = 1;
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csiphy_fd);
  }

  // set format on CSID subdev
  int csid_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csid_subdev).c_str(), O_RDWR);
  if (csid_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = 0x300f;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);

    // RDI source pad (pad 1)
    sfmt.pad = 1;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csid_fd);
  }

  // set format on sensor subdev
  if (sensor_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = 0x300f;
    ioctl(sensor_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
  }

  // set format on VFE RDI subdev (sink pad)
  int vfe_rdi_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.vfe_rdi_subdev).c_str(), O_RDWR);
  if (vfe_rdi_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = 0x300f;
    ioctl(vfe_rdi_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(vfe_rdi_fd);
  }

  // set raw Bayer format on RDI video device
  struct v4l2_format vfmt = {};
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vfmt.fmt.pix_mp.width = sensor->frame_width;
  vfmt.fmt.pix_mp.height = sensor->frame_height;
  vfmt.fmt.pix_mp.pixelformat = v4l2_fourcc('p', 'R', 'A', 'A');  // SRGGB10P
  vfmt.fmt.pix_mp.num_planes = 1;
  if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) == 0) {
    frame_size = vfmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    LOG("cam %d: RDI format set %dx%d, frame_size=%u", cam_idx,
        vfmt.fmt.pix_mp.width, vfmt.fmt.pix_mp.height, frame_size);
  }
}

void DragonCamera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  sensor = std::make_unique<IMX219>();
  vipc_server = v;
  stream_type = cc.stream_type;

  int cam_idx = cc.camera_num;
  auto &dcfg = dragon_cams[cam_idx];

  // open video device
  int dev = dcfg.rdi_video_dev;
  std::string path = util::string_format("/dev/video%d", dev);
  video_fd = open(path.c_str(), O_RDWR);
  if (video_fd < 0) {
    LOGE("cam %d: failed to open %s: %d", cam_idx, path.c_str(), errno);
    enabled = false;
    return;
  }
  LOG("cam %d: opened %s (RDI mode)", cam_idx, path.c_str());

  // find sensor subdev
  for (int i = 0; i < 32; i++) {
    std::string name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", i));
    if (name.find(dcfg.sensor_name) == 0) {
      sensor_fd = open(util::string_format("/dev/v4l-subdev%d", i).c_str(), O_RDWR);
      break;
    }
  }
  if (sensor_fd < 0) {
    LOGE("cam %d: sensor subdev '%s' not found, disabling", cam_idx, dcfg.sensor_name);
    enabled = false;
    return;
  }

  setup_media_links();
  set_formats();
  set_dragon_sensor_timing(sensor_fd, cam_idx);

  output_width = sensor->frame_width;
  output_height = sensor->frame_height;
  auto [s, yh, uvh, sz] = get_nv12_info(output_width, output_height);
  stride = s;
  y_height = yh;
  uv_height = uvh;
  yuv_size = sz;
  uv_offset = stride * y_height;

  v->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT,
                               output_width, output_height,
                               yuv_size, stride, uv_offset);

  // V4L2 MMAP buffers for raw RDI frames
  struct v4l2_requestbuffers req = {};
  req.count = n_bufs;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_MMAP;
  int reqbufs_ret = ioctl(video_fd, VIDIOC_REQBUFS, &req);
  if (reqbufs_ret != 0) {
    LOGE("cam %d: REQBUFS failed: %d (%s)", cam_idx, errno, strerror(errno));
    enabled = false;
    return;
  }
  n_bufs = req.count;

  for (int i = 0; i < n_bufs; i++) {
    struct v4l2_buffer qbuf = {};
    struct v4l2_plane planes[1] = {};
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = i;
    qbuf.length = 1;
    qbuf.m.planes = planes;
    if (ioctl(video_fd, VIDIOC_QUERYBUF, &qbuf) != 0) {
      LOGE("cam %d: QUERYBUF %d failed: %d", cam_idx, i, errno);
      continue;
    }
    v4l_bufs[i].length = planes[0].length;
    v4l_bufs[i].start = mmap(NULL, planes[0].length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, video_fd, planes[0].m.mem_offset);
  }

  LOG("cam %d: VIPC buffers created (NV12 sw debayer, %u bytes, stride=%u)", cam_idx,
      yuv_size, stride);
}

void DragonCamera::queue_all_buffers() {
  if (!enabled) return;

  for (int i = 0; i < n_bufs; i++) {
    queue_frame(i);
  }
}

void DragonCamera::stream_on() {
  if (!enabled) return;

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl(video_fd, VIDIOC_STREAMON, &type) != 0) {
    LOGE("cam %d: STREAMON failed: %d (%s)", cc.camera_num, errno, strerror(errno));
    enabled = false;
    return;
  }

  LOG("cam %d: RDI streaming started", cc.camera_num);
}

void DragonCamera::start_streaming() {
  queue_all_buffers();
  stream_on();
}

void DragonCamera::stop_streaming() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  ioctl(video_fd, VIDIOC_STREAMOFF, &type);
}

void DragonCamera::queue_frame(int index) {
  struct v4l2_buffer vbuf = {};
  struct v4l2_plane planes[1] = {};
  vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vbuf.memory = V4L2_MEMORY_MMAP;
  vbuf.index = index;
  vbuf.length = 1;
  vbuf.m.planes = planes;
  int ret = ioctl(video_fd, VIDIOC_QBUF, &vbuf);
  if (ret != 0) LOGE("cam %d: QBUF idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
}

int DragonCamera::dequeue_frame(uint64_t *timestamp) {
  struct pollfd pfd = {video_fd, POLLIN, 0};
  int ret = poll(&pfd, 1, 20);
  if (ret == 0) return -ETIMEDOUT;
  if (ret < 0) return -errno;
  if (!(pfd.revents & POLLIN)) return -EIO;

  struct v4l2_buffer dbuf = {};
  struct v4l2_plane planes[1] = {};
  dbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  dbuf.memory = V4L2_MEMORY_MMAP;
  dbuf.length = 1;
  dbuf.m.planes = planes;
  if (ioctl(video_fd, VIDIOC_DQBUF, &dbuf) != 0) return -errno;

  *timestamp = (uint64_t)dbuf.timestamp.tv_sec * 1000000000ULL +
               (uint64_t)dbuf.timestamp.tv_usec * 1000ULL;
  return dbuf.index;
}

void DragonCamera::set_exposure(int exposure_time, int gain_idx) {
  if (sensor_fd < 0) return;

  struct v4l2_control ctrl = {};
  ctrl.id = V4L2_CID_EXPOSURE;
  ctrl.value = exposure_time;
  ioctl(sensor_fd, VIDIOC_S_CTRL, &ctrl);

  const uint32_t imx219_gains[] = {0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 232};
  ctrl.id = V4L2_CID_ANALOGUE_GAIN;
  ctrl.value = imx219_gains[gain_idx];
  ioctl(sensor_fd, VIDIOC_S_CTRL, &ctrl);
}

void DragonCamera::camera_close() {
  if (video_fd >= 0) {
    stop_streaming();
    for (int i = 0; i < n_bufs; i++) {
      if (v4l_bufs[i].start) {
        munmap(v4l_bufs[i].start, v4l_bufs[i].length);
        v4l_bufs[i].start = nullptr;
      }
    }
    close(video_fd);
    video_fd = -1;
  }
  if (sensor_fd >= 0) {
    close(sensor_fd);
    sensor_fd = -1;
  }
}


class CameraState {
public:
  DragonCamera camera;
  int exposure_time = 1600;
  bool dc_gain_enabled = false;
  int dc_gain_weight = 0;
  int gain_idx = 8;
  float analog_gain_frac = 0;
  float cur_ev[3] = {};
  float best_ev_score = 0;
  int new_exp_g = 0;
  int new_exp_t = 0;

  Rect ae_xywh = {};
  float measured_grey_fraction = 0;
  float target_grey_fraction = 0.125;

  uint32_t frame_id = 0;
  std::unique_ptr<PubMaster> pm;
  float fl_pix = 0;

  CameraState(const CameraConfig &config) : camera(config) {}
  ~CameraState() { camera.camera_close(); }

  void init(VisionIpcServer *v);
  void process_rdi_frame(int buf_idx, uint64_t timestamp);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);
  void set_exposure_rect();

  float get_gain_factor() const {
    return (1 + dc_gain_weight * (camera.sensor->dc_gain_factor-1) / camera.sensor->dc_gain_max_weight);
  }
};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);
  if (!camera.enabled) return;

  // The Dragon starts the road sensor first, then wide almost one period later.
  // Number wide one frame ahead so same-numbered road/wide frames refer to the
  // same 20 Hz capture instant.
  if (camera.cc.stream_type == VISION_STREAM_WIDE_ROAD) {
    frame_id = 1;
  }

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm;
  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});

  float gain = camera.sensor->sensor_analog_gains[gain_idx];
  cur_ev[0] = cur_ev[1] = cur_ev[2] = gain * exposure_time;
  camera.set_exposure(exposure_time, gain_idx);

  set_exposure_rect();
}

void CameraState::set_exposure_rect() {
  // AE rectangle for NV12 frames
  ae_xywh = {
    (int)(camera.sensor->frame_width * 0.05f),
    (int)(camera.sensor->frame_height * 0.15f),
    (int)(camera.sensor->frame_width * 0.9f),
    (int)(camera.sensor->frame_height * 0.75f),
  };
}

static float calculate_exposure_value_nv12(const uint8_t *y_plane, int stride, Rect ae_xywh, int x_skip, int y_skip) {
  int lum_med;
  uint32_t lum_binning[256] = {0};

  unsigned int lum_total = 0;
  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = y_plane[(y * stride) + x];
      lum_binning[lum]++;
      lum_total += 1;
    }
  }

  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];
    if (lum_cur >= lum_total / 2) break;
  }

  return lum_med / 256.0f;
}

void CameraState::update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
  float score = camera.sensor->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
  if (score < best_ev_score) {
    new_exp_t = exp_t;
    new_exp_g = exp_g_idx;
    best_ev_score = score;
  }
}

void CameraState::set_camera_exposure(float grey_frac) {
  if (!camera.enabled) return;

  const float dt = 0.05;
  const float ts_grey = 10.0;
  const float ts_ev = 0.05;
  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  const auto &sens = camera.sensor;
  const float cur_ev_ = cur_ev[(frame_id - 1) % 3];

  float new_target_grey = std::clamp(0.4f - 0.3f * (float)(log2(1.0 + sens->target_grey_factor*cur_ev_) / log2(6000.0)), 0.1f, 0.4f);
  float target_grey = (1.0f - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, sens->min_ev, sens->max_ev);
  float k = (1.0f - k_ev) / 3.0f;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  int min_g = std::max(gain_idx - 1, sens->analog_gain_min_idx);
  int max_g = std::min(gain_idx + 1, sens->analog_gain_max_idx);
  for (int g = min_g; g <= max_g; g++) {
    float gain = sens->sensor_analog_gains[g];
    int t = std::clamp((int)std::round(desired_ev / gain), sens->exposure_time_min, sens->exposure_time_max);
    update_exposure_score(desired_ev, t, g, gain);
  }

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;
  analog_gain_frac = sens->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;

  cur_ev[frame_id % 3] = exposure_time * analog_gain_frac;
  camera.set_exposure(exposure_time, gain_idx);
}

void CameraState::process_rdi_frame(int buf_idx, uint64_t timestamp) {
  frame_id++;
  uint64_t timestamp_eof = timestamp + camera.sensor->readout_time_ns;

  VisionBuf *vb = camera.vipc_server->get_buffer(camera.stream_type, frame_id % VIPC_BUFFER_COUNT);
  if (vb && camera.v4l_bufs[buf_idx].start) {
    const uint8_t *raw = (const uint8_t *)camera.v4l_bufs[buf_idx].start;
    uint8_t *y_plane = (uint8_t *)vb->addr;
    uint8_t *uv_plane = y_plane + camera.uv_offset;
    int raw_stride = camera.frame_size / camera.sensor->frame_height;
    debayer_raw10_to_nv12(raw, raw_stride, y_plane, uv_plane,
                           camera.sensor->frame_width, camera.output_width, camera.output_height,
                           camera.stride);
    set_camera_exposure(calculate_exposure_value_nv12(y_plane, camera.stride, ae_xywh, 4, 4));
  }

  VisionIpcBufExtra extra = {frame_id, timestamp, timestamp_eof};
  vb->set_frame_id(frame_id);
  camera.vipc_server->send(vb, &extra);

  MessageBuilder msg;
  auto framed = (msg.initEvent().*camera.cc.init_camera_state)();
  framed.setFrameId(frame_id);
  framed.setTimestampEof(timestamp_eof);
  framed.setTimestampSof(timestamp);
  framed.setIntegLines(exposure_time);
  framed.setGain(camera.sensor->sensor_analog_gains[gain_idx]);
  framed.setSensor(camera.sensor->image_sensor);
  framed.setMeasuredGreyFraction(measured_grey_fraction);
  framed.setTargetGreyFraction(target_grey_fraction);
  framed.setExposureValPercent(util::map_val(cur_ev[frame_id % 3],
    camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f));
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  LOG("-- Dragon camerad starting (RDI sw debayer)");

  VisionIpcServer v("camerad");

  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }
  for (int i = 0; i < 2; i++) {
    dragon_cams[i] = resolve_cam_config(media_fd, i);
    LOG("cam %d: csiphy=%u csid=%u vfe_rdi=%u rdi_dev=%d",
        i, dragon_cams[i].csiphy_entity, dragon_cams[i].csid_entity,
        dragon_cams[i].vfe_rdi_entity, dragon_cams[i].rdi_video_dev);
  }
  close(media_fd);

  reset_all_media_links();

  bool single_cam = getenv("SINGLE_CAM") != nullptr;
  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    if (config.output_type == ISP_BPS_PROCESSED) continue;
    if (config.camera_num > 1) continue;
    if (single_cam && config.camera_num != 0) continue;
    auto cam = std::make_unique<CameraState>(config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  for (auto &cam : cams) {
    cam->camera.queue_all_buffers();
  }
  for (auto it = cams.rbegin(); it != cams.rend(); ++it) {
    (*it)->camera.stream_on();
    if ((*it)->camera.cc.stream_type == VISION_STREAM_ROAD) {
      usleep(28000);
    }
  }

  LOG("-- Dragon camerad streaming");

  while (!do_exit) {
    for (auto &cam : cams) {
      if (!cam->camera.enabled) continue;

      uint64_t timestamp;
      int buf_idx = cam->camera.dequeue_frame(&timestamp);
      if (buf_idx < 0) {
        if (env_debug_frames) printf("cam %d: dequeue timeout\n", cam->camera.cc.camera_num);
        if (buf_idx != -ETIMEDOUT) {
          LOGW_100("cam %d: dequeue failed: %d (%s)", cam->camera.cc.camera_num, -buf_idx, strerror(-buf_idx));
          usleep(1000);
        }
        continue;
      }

      cam->process_rdi_frame(buf_idx, timestamp);

      if (env_debug_frames) {
        printf("cam %d frame %u buf %d ts %.2f ms (%s)\n", cam->camera.cc.camera_num, cam->frame_id, buf_idx, timestamp / 1e6,
               "RDI");
      }

      cam->camera.queue_frame(buf_idx);
    }
  }

  LOG("-- Dragon camerad stopping");
  for (auto &cam : cams) {
    cam->camera.stop_streaming();
  }
}
