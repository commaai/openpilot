#include "system/camerad/cameras/camera_common.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <linux/media.h>
#include <linux/videodev2.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "media/qcom-camss-vfe.h"

#include "common/params.h"
#include "common/swaglog.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/ife.h"
#include "system/camerad/cameras/nv12_info.h"
#include "system/camerad/sensors/sensor.h"


ExitHandler do_exit;

const bool env_debug_frames = getenv("DEBUG_FRAMES") != nullptr;
const bool env_log_raw_frames = getenv("LOG_RAW_FRAMES") != nullptr;
const bool env_ctrl_exp_from_params = getenv("CTRL_EXP_FROM_PARAMS") != nullptr;


// sensor factory
static std::unique_ptr<SensorInfo> create_sensor(int camera_num) {
  if (camera_num == 0 || camera_num == 1) return std::make_unique<OX03C10>();
  if (camera_num == 2) return std::make_unique<OS04C10>();
  return nullptr;
}


class MainlineCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  CameraBuf buf;
  bool enabled;

  // NV12 geometry
  uint32_t stride, y_height, uv_height, yuv_size, uv_offset;

  MainlineCamera(const CameraConfig &config) : cc(config), enabled(config.enabled) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void sensors_i2c(const i2c_random_wr_payload *dat, int len, bool data_word);
  void sensors_start();
  void enqueue_frame(int buf_idx);

  int vfe_fd = -1;
  int sensor_fd = -1;

private:
  uint64_t iova_map[VIPC_BUFFER_COUNT] = {};
  uint32_t wm_y = 3, wm_uv = 4;  // PIX write master indices (same as downstream)

  int vfe_ioctl(unsigned long cmd, void *arg) {
    return HANDLE_EINTR(ioctl(vfe_fd, cmd, arg));
  }
};


void MainlineCamera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  sensor = create_sensor(cc.camera_num);
  assert(sensor);

  // NV12 output geometry
  auto [out_w, out_h] = std::make_pair(sensor->frame_width, sensor->frame_height);
  buf.out_img_width = out_w;
  buf.out_img_height = out_h;
  auto [s, yh, uvh, sz] = get_nv12_info(out_w, out_h);
  stride = s;
  y_height = yh;
  uv_height = uvh;
  yuv_size = sz;
  uv_offset = stride * y_height;

  // open VFE PIX video device
  // PIX is video3 for VFE0, video7 for VFE1
  {
    const int pix_devs[] = {3, 7};  // VFE0 PIX, VFE1 PIX
    std::string path = util::string_format("/dev/video%d", pix_devs[cc.camera_num]);
    vfe_fd = HANDLE_EINTR(open(path.c_str(), O_RDWR));
    LOG("camera %d: opening %s fd=%d", cc.camera_num, path.c_str(), vfe_fd);
  }
  if (vfe_fd < 0) {
    LOGE("failed to open VFE PIX for camera %d: %d", cc.camera_num, errno);
    enabled = false;
    return;
  }

  // open sensor subdev for register writes
  // can't use open_v4l_by_name_and_index because subdev numbering has gaps
  {
    int match_idx = 0;
    for (int i = 0; i < 32; i++) {
      std::string name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", i));
      if (name.find("ox03c10") == 0 || name.find("os04c10") == 0) {
        if (match_idx == cc.camera_num) {
          sensor_fd = HANDLE_EINTR(open(util::string_format("/dev/v4l-subdev%d", i).c_str(), O_RDWR));
          break;
        }
        match_idx++;
      }
    }
  }
  if (sensor_fd < 0) {
    LOGE("failed to open sensor subdev for camera %d", cc.camera_num);
    enabled = false;
    return;
  }

  // power on camera subsystem (titan_top_gdsc) before sensor init
  // this enables CAMCC clocks including MCLK for sensors
  {
    std::string power_ctrl = "/sys/bus/platform/devices/acb3000.camss/power/control";
    int pfd = open(power_ctrl.c_str(), O_WRONLY);
    if (pfd >= 0) {
      write(pfd, "on", 2);
      close(pfd);
      usleep(1000);  // let GDSC stabilize
    }
  }

  // write sensor init registers
  sensors_i2c(sensor->init_reg_array.data(), sensor->init_reg_array.size(), sensor->data_word);
  LOG("camera %d: wrote %zu sensor init registers", cc.camera_num, sensor->init_reg_array.size());

  // subscribe to V4L2 events (SOF)
  struct v4l2_event_subscription sub = {};
  sub.type = V4L2_EVENT_FRAME_SYNC;
  HANDLE_EINTR(ioctl(vfe_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));

  // allocate VisionIPC buffers
  buf.vipc_server = v;
  buf.stream_type = cc.stream_type;
  v->create_buffers_with_sizes(cc.stream_type, VIPC_BUFFER_COUNT, out_w, out_h, yuv_size, stride, uv_offset);
  LOG("camera %d: created %d vipc buffers %dx%d stride=%d", cc.camera_num, VIPC_BUFFER_COUNT, out_w, out_h, stride);

  // map VisionIPC buffers into VFE IOMMU
  for (int i = 0; i < VIPC_BUFFER_COUNT; i++) {
    VisionBuf *vb = v->get_buffer(cc.stream_type, i);
    struct vfe_map_buf_cmd cmd = {};
    cmd.fd = vb->fd;
    int ret = vfe_ioctl(VFE_MAP_BUF, &cmd);
    if (ret != 0) {
      LOGE("VFE_MAP_BUF failed for buf %d: %d", i, errno);
      enabled = false;
      return;
    }
    iova_map[i] = cmd.iova;
    LOGD("camera %d: mapped buf %d fd=%d iova=0x%llx", cc.camera_num, i, vb->fd, (unsigned long long)cmd.iova);
  }

  // write initial ISP config
  auto [regs, dmis] = build_initial_config_flat(cc, sensor.get(), out_w, out_h);

  // submit register writes
  struct vfe_write_regs_cmd regs_cmd = {};
  regs_cmd.regs = (uint64_t)regs.data();
  regs_cmd.count = regs.size();
  int ret = vfe_ioctl(VFE_WRITE_REGS, &regs_cmd);
  if (ret != 0) LOGE("VFE_WRITE_REGS init failed: %d", errno);

  // upload DMI LUTs
  for (const auto &dmi : dmis) {
    struct vfe_dmi_cmd dmi_cmd = {};
    dmi_cmd.dmi_cfg_offset = dmi.cfg_offset;
    dmi_cmd.ram_select = dmi.ram_select;
    dmi_cmd.count = dmi.count;
    dmi_cmd.data = (uint64_t)dmi.data;
    ret = vfe_ioctl(VFE_WRITE_DMI, &dmi_cmd);
    if (ret != 0) LOGE("VFE_WRITE_DMI (sel=%d) failed: %d", dmi.ram_select, errno);
  }

  // fire reg_update to latch initial config
  ret = vfe_ioctl(VFE_REG_UPDATE, nullptr);
  if (ret != 0) LOGE("VFE_REG_UPDATE init failed: %d", errno);

  LOG("camera %d: ISP configured (%zu regs, %zu DMI uploads)", cc.camera_num, regs.size(), dmis.size());
}

void MainlineCamera::camera_close() {
  if (vfe_fd >= 0) {
    // unmap all buffers
    for (int i = 0; i < VIPC_BUFFER_COUNT; i++) {
      if (iova_map[i] != 0) {
        struct vfe_unmap_buf_cmd cmd = {.iova = iova_map[i]};
        vfe_ioctl(VFE_UNMAP_BUF, &cmd);
        iova_map[i] = 0;
      }
    }
    close(vfe_fd);
    vfe_fd = -1;
  }
  if (sensor_fd >= 0) {
    close(sensor_fd);
    sensor_fd = -1;
  }
}

void MainlineCamera::sensors_i2c(const i2c_random_wr_payload *dat, int len, bool data_word) {
  if (sensor_fd < 0 || len == 0) return;

  // translate i2c_random_wr_payload to sensor_reg_write
  std::vector<sensor_reg_write> regs(len);
  for (int i = 0; i < len; i++) {
    regs[i].addr = (uint16_t)dat[i].reg_addr;
    regs[i].data = (uint16_t)dat[i].reg_data;
  }

  struct sensor_write_regs_cmd cmd = {};
  cmd.regs = (uint64_t)regs.data();
  cmd.count = len;
  cmd.data_width = data_word ? 2 : 1;

  int ret = HANDLE_EINTR(ioctl(sensor_fd, SENSOR_WRITE_REGS, &cmd));
  if (ret != 0) {
    LOGE("SENSOR_WRITE_REGS failed (%d regs): %d", len, errno);
  }
}

void MainlineCamera::sensors_start() {
  if (!enabled) return;

  // start VFE pipeline (powers on CSIPHY, CSID, VFE, enables IRQs)
  int ret = vfe_ioctl(VFE_START, nullptr);
  if (ret != 0) LOGE("VFE_START failed: %d", errno);

  // start sensor streaming
  sensors_i2c(sensor->start_reg_array.data(), sensor->start_reg_array.size(), sensor->data_word);
  LOG("camera %d: streaming started", cc.camera_num);
}

void MainlineCamera::enqueue_frame(int buf_idx) {
  // per-frame ISP register update
  auto regs = build_update_flat(cc, sensor.get());
  struct vfe_write_regs_cmd regs_cmd = {};
  regs_cmd.regs = (uint64_t)regs.data();
  regs_cmd.count = regs.size();
  vfe_ioctl(VFE_WRITE_REGS, &regs_cmd);

  // set output buffer for Y and UV planes
  struct vfe_set_buf_cmd y_cmd = {};
  y_cmd.wm_index = wm_y;
  y_cmd.iova = iova_map[buf_idx];
  y_cmd.stride = stride;
  y_cmd.frame_inc = stride * y_height;
  vfe_ioctl(VFE_SET_BUF, &y_cmd);

  struct vfe_set_buf_cmd uv_cmd = {};
  uv_cmd.wm_index = wm_uv;
  uv_cmd.iova = iova_map[buf_idx] + uv_offset;
  uv_cmd.stride = stride;
  uv_cmd.frame_inc = stride * uv_height;
  vfe_ioctl(VFE_SET_BUF, &uv_cmd);

  // latch all register writes
  vfe_ioctl(VFE_REG_UPDATE, nullptr);
}


// CameraState wraps MainlineCamera with auto-exposure logic
// (same structure as downstream camera_qcom2.cc CameraState)
class CameraState {
public:
  MainlineCamera camera;
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

  CameraState(const CameraConfig &config) : camera(config) {}
  ~CameraState() { camera.camera_close(); }

  void init(VisionIpcServer *v);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);
  void set_exposure_rect();
  void sendState();

  float get_gain_factor() const {
    return (1 + dc_gain_weight * (camera.sensor->dc_gain_factor-1) / camera.sensor->dc_gain_max_weight);
  }
};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);
  if (!camera.enabled) return;

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm / camera.sensor->out_scale;
  set_exposure_rect();

  dc_gain_weight = camera.sensor->dc_gain_min_weight;
  gain_idx = camera.sensor->analog_gain_rec_idx;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = get_gain_factor() * camera.sensor->sensor_analog_gains[gain_idx] * exposure_time;

  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});
}

void CameraState::set_exposure_rect() {
  std::vector<std::pair<Rect, float>> ae_targets = {
    std::make_pair((Rect){96, 400, 1734, 524}, 567.0),  // wide
    std::make_pair((Rect){96, 160, 1734, 986}, 2648.0), // road
    std::make_pair((Rect){96, 242, 1736, 906}, 567.0)   // driver
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
  std::vector<double> target_grey_minimums = {0.1, 0.1, 0.125};

  const float dt = 0.05;
  const float ts_grey = 10.0;
  const float ts_ev = 0.05;
  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  const auto &sens = camera.sensor;
  const float cur_ev_ = cur_ev[(camera.buf.cur_frame_data.frame_id - 1) % 3] * sens->ev_scale;

  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + sens->target_grey_factor*cur_ev_) / log2(6000.0), target_grey_minimums[camera.cc.camera_num], 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ / sens->ev_scale * target_grey / grey_frac, sens->min_ev, sens->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < sens->dc_gain_on_grey) {
    enable_dc_gain = true;
    dc_gain_weight = sens->dc_gain_min_weight;
  } else if (enable_dc_gain && target_grey > sens->dc_gain_off_grey) {
    enable_dc_gain = false;
    dc_gain_weight = sens->dc_gain_max_weight;
  }

  if (enable_dc_gain && dc_gain_weight < sens->dc_gain_max_weight) dc_gain_weight += 1;
  if (!enable_dc_gain && dc_gain_weight > sens->dc_gain_min_weight) dc_gain_weight -= 1;

  std::string gain_bytes, time_bytes;
  if (env_ctrl_exp_from_params) {
    static Params params;
    gain_bytes = params.get("CameraDebugExpGain");
    time_bytes = params.get("CameraDebugExpTime");
  }

  if (gain_bytes.size() > 0 && time_bytes.size() > 0) {
    gain_idx = std::stoi(gain_bytes);
    exposure_time = std::stoi(time_bytes);
    new_exp_g = gain_idx;
    new_exp_t = exposure_time;
    enable_dc_gain = false;
  } else {
    int min_g = std::max(gain_idx - 1, sens->analog_gain_min_idx);
    int max_g = std::min(gain_idx + 1, sens->analog_gain_max_idx);
    for (int g = min_g; g <= max_g; g++) {
      float gain = sens->sensor_analog_gains[g] * get_gain_factor();
      int t = std::clamp(int(std::round(desired_ev / gain)), sens->exposure_time_min, sens->exposure_time_max);
      if (g < sens->analog_gain_rec_idx && t > 20 && g < gain_idx) continue;
      update_exposure_score(desired_ev, t, g, gain);
    }
  }

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;
  analog_gain_frac = sens->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * get_gain_factor();
  cur_ev[camera.buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  auto exp_reg_array = sens->getExposureRegisters(exposure_time, new_exp_g, dc_gain_enabled);
  camera.sensors_i2c(exp_reg_array.data(), exp_reg_array.size(), camera.sensor->data_word);
}

void CameraState::sendState() {
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

  if (env_log_raw_frames && camera.cc.stream_type == VISION_STREAM_ROAD && meta.frame_id % 100 == 5) {
    framed.setImage(get_raw_frame_image(&camera.buf));
  }

  set_camera_exposure(calculate_exposure_value(&camera.buf, ae_xywh, 2, camera.cc.stream_type != VISION_STREAM_DRIVER ? 2 : 4));

  pm->send(camera.cc.publish_name, msg);
}


// frame sync across cameras (same logic as spectra.cc syncFirstFrame)
static std::map<int, uint64_t> sync_timestamps;
static std::map<int, uint64_t> frame_id_offsets;
static bool first_frame_synced = false;

static bool sync_first_frame(int camera_id, uint64_t frame_id_raw, uint64_t timestamp) {
  if (first_frame_synced) return true;

  sync_timestamps[camera_id] = timestamp;
  frame_id_offsets[camera_id] = frame_id_raw + 1;

  // need all enabled cameras
  int enabled_count = 0;
  for (const auto &cfg : ALL_CAMERA_CONFIGS) {
    if (cfg.enabled && cfg.output_type != ISP_BPS_PROCESSED) enabled_count++;
  }
  if ((int)sync_timestamps.size() < enabled_count) return false;

  // check timestamps are within 0.2ms (both cameras are non-staggered)
  const uint64_t tolerance_ns = 200000ULL;
  for (const auto &[cam_a, ts_a] : sync_timestamps) {
    for (const auto &[cam_b, ts_b] : sync_timestamps) {
      if (cam_a >= cam_b) continue;
      uint64_t diff = std::max(ts_a, ts_b) - std::min(ts_a, ts_b);
      if (diff > tolerance_ns) {
        sync_timestamps.clear();
        frame_id_offsets.clear();
        return false;
      }
    }
  }

  first_frame_synced = true;
  for (const auto &[cam, ts] : sync_timestamps) {
    LOGW("camera %d synced on frame_id_offset %lu timestamp %lu", cam, frame_id_offsets[cam], ts);
  }
  return true;
}


void camerad_thread() {
  VisionIpcServer v("camerad");

  // init cameras (wide + road only, driver disabled)
  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    if (config.output_type == ISP_BPS_PROCESSED) continue;  // skip driver camera
    auto cam = std::make_unique<CameraState>(config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  // start sensors and pre-queue frames
  LOG("-- Starting cameras");
  for (auto &cam : cams) {
    cam->camera.sensors_start();
    for (int i = 0; i < VIPC_BUFFER_COUNT; i++) {
      cam->camera.enqueue_frame(i);
    }
  }

  // main event loop
  LOG("-- Polling for events");
  uint32_t frame_counter = 0;
  while (!do_exit) {
    // build poll fds for all cameras
    std::vector<struct pollfd> fds;
    for (auto &cam : cams) {
      if (cam->camera.enabled) {
        fds.push_back({.fd = cam->camera.vfe_fd, .events = POLLPRI});
      }
    }

    int ret = poll(fds.data(), fds.size(), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    for (size_t i = 0; i < fds.size(); i++) {
      if (!(fds[i].revents & POLLPRI)) continue;

      auto &cam = cams[i];
      struct v4l2_event ev = {};
      ret = HANDLE_EINTR(ioctl(fds[i].fd, VIDIOC_DQEVENT, &ev));
      if (ret != 0) continue;

      // SOF event with timestamp
      uint64_t timestamp = ev.timestamp.tv_sec * 1000000000ULL + ev.timestamp.tv_nsec;
      frame_counter++;

      if (env_debug_frames) {
        printf("cam %d frame %u ts %.2f ms\n", cam->camera.cc.camera_num, frame_counter, timestamp / 1e6);
        if (frame_counter > 20) do_exit = true;
      }

      if (!sync_first_frame(cam->camera.cc.camera_num, frame_counter, timestamp)) {
        continue;
      }

      uint64_t timestamp_eof = timestamp + cam->camera.sensor->readout_time_ns;

      int buf_idx = frame_counter % VIPC_BUFFER_COUNT;
      cam->camera.buf.cur_buf_idx = buf_idx;
      cam->camera.buf.cur_frame_data = {
        .frame_id = (uint32_t)(frame_counter - frame_id_offsets[cam->camera.cc.camera_num]),
        .request_id = frame_counter,
        .timestamp_sof = timestamp,
        .timestamp_eof = timestamp_eof,
        .processing_time = 0,
      };

      cam->sendState();
      cam->camera.enqueue_frame(buf_idx);
    }
  }

  LOG("-- Stopping cameras");
}
