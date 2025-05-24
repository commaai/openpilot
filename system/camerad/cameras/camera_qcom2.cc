#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/spectra.h"

#include <poll.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifdef QCOM2
#include "CL/cl_ext_qcom.h"
#else
#define CL_PRIORITY_HINT_HIGH_QCOM NULL
#define CL_CONTEXT_PRIORITY_HINT_QCOM NULL
#endif

#include "media/cam_sensor_cmn_header.h"

#include "common/clutil.h"
#include "common/params.h"
#include "common/swaglog.h"


ExitHandler do_exit;

// for debugging
const bool env_debug_frames = getenv("DEBUG_FRAMES") != nullptr;
const bool env_log_raw_frames = getenv("LOG_RAW_FRAMES") != nullptr;
const bool env_ctrl_exp_from_params = getenv("CTRL_EXP_FROM_PARAMS") != nullptr;


class CameraState {
public:
  SpectraCamera camera;
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
  float target_grey_fraction = 0.3;

  float fl_pix = 0;
  std::unique_ptr<PubMaster> pm;

  CameraState(SpectraMaster *master, const CameraConfig &config) : camera(master, config) {};
  ~CameraState();
  void init(VisionIpcServer *v, cl_device_id device_id, cl_context ctx);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);
  void set_exposure_rect();
  void sendState();

  float get_gain_factor() const {
    return (1 + dc_gain_weight * (camera.sensor->dc_gain_factor-1) / camera.sensor->dc_gain_max_weight);
  }
};

void CameraState::init(VisionIpcServer *v, cl_device_id device_id, cl_context ctx) {
  camera.camera_open(v, device_id, ctx);

  if (!camera.enabled) return;

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm / camera.sensor->out_scale;
  set_exposure_rect();

  dc_gain_weight = camera.sensor->dc_gain_min_weight;
  gain_idx = camera.sensor->analog_gain_rec_idx;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = get_gain_factor() * camera.sensor->sensor_analog_gains[gain_idx] * exposure_time;

  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});
}

CameraState::~CameraState() {}

void CameraState::set_exposure_rect() {
  // set areas for each camera, shouldn't be changed
  std::vector<std::pair<Rect, float>> ae_targets = {
    // (Rect, F)
    std::make_pair((Rect){96, 250, 1734, 524}, 567.0),  // wide
    std::make_pair((Rect){96, 160, 1734, 986}, 2648.0), // road
    std::make_pair((Rect){96, 242, 1736, 906}, 567.0)   // driver
  };
  int h_ref = 1208;
  /*
    exposure target intrinsics is
    [
      [F, 0, 0.5*ae_xywh[2]]
      [0, F, 0.5*H-ae_xywh[1]]
      [0, 0, 1]
    ]
  */
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
  std::vector<double> target_grey_minimums = {0.1, 0.1, 0.125}; // wide, road, driver

  const float dt = 0.05;

  const float ts_grey = 10.0;
  const float ts_ev = 0.05;

  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  // It takes 3 frames for the commanded exposure settings to take effect. The first frame is already started by the time
  // we reach this function, the other 2 are due to the register buffering in the sensor.
  // Therefore we use the target EV from 3 frames ago, the grey fraction that was just measured was the result of that control action.
  // TODO: Lower latency to 2 frames, by using the histogram outputted by the sensor we can do AE before the debayering is complete

  const auto &sensor = camera.sensor;
  // Offset idx by one to not get stuck in self loop
  const float cur_ev_ = cur_ev[(camera.buf.cur_frame_data.frame_id - 1) % 3] * sensor->ev_scale;

  // Scale target grey between min and 0.4 depending on lighting conditions
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + sensor->target_grey_factor*cur_ev_) / log2(6000.0), target_grey_minimums[camera.cc.camera_num], 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ / sensor->ev_scale * target_grey / grey_frac, sensor->min_ev, sensor->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < sensor->dc_gain_on_grey) {
    enable_dc_gain = true;
    dc_gain_weight = sensor->dc_gain_min_weight;
  } else if (enable_dc_gain && target_grey > sensor->dc_gain_off_grey) {
    enable_dc_gain = false;
    dc_gain_weight = sensor->dc_gain_max_weight;
  }

  if (enable_dc_gain && dc_gain_weight < sensor->dc_gain_max_weight) {dc_gain_weight += 1;}
  if (!enable_dc_gain && dc_gain_weight > sensor->dc_gain_min_weight) {dc_gain_weight -= 1;}

  std::string gain_bytes, time_bytes;
  if (env_ctrl_exp_from_params) {
    static Params params;
    gain_bytes = params.get("CameraDebugExpGain");
    time_bytes = params.get("CameraDebugExpTime");
  }

  if (gain_bytes.size() > 0 && time_bytes.size() > 0) {
    // Override gain and exposure time
    gain_idx = std::stoi(gain_bytes);
    exposure_time = std::stoi(time_bytes);

    new_exp_g = gain_idx;
    new_exp_t = exposure_time;
    enable_dc_gain = false;
  } else {
    // Simple brute force optimizer to choose sensor parameters to reach desired EV
    int min_g = std::max(gain_idx - 1, sensor->analog_gain_min_idx);
    int max_g = std::min(gain_idx + 1, sensor->analog_gain_max_idx);
    for (int g = min_g; g <= max_g; g++) {
      float gain = sensor->sensor_analog_gains[g] * get_gain_factor();

      // Compute optimal time for given gain
      int t = std::clamp(int(std::round(desired_ev / gain)), sensor->exposure_time_min, sensor->exposure_time_max);

      // Only go below recommended gain when absolutely necessary to not overexpose
      if (g < sensor->analog_gain_rec_idx && t > 20 && g < gain_idx) {
        continue;
      }

      update_exposure_score(desired_ev, t, g, gain);
    }
  }

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = sensor->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * get_gain_factor();
  cur_ev[camera.buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", camera.cc.camera_num, 1e-9 * nanos_since_boot(), 1e-9 * camera.buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - camera.buf.cur_frame_data.timestamp_sof));

  auto exp_reg_array = sensor->getExposureRegisters(exposure_time, new_exp_g, dc_gain_enabled);
  camera.sensors_i2c(exp_reg_array.data(), exp_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, camera.sensor->data_word);
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

  // Log raw frames for road camera
  if (env_log_raw_frames && camera.cc.stream_type == VISION_STREAM_ROAD && meta.frame_id % 100 == 5) {  // no overlap with qlog decimation
    framed.setImage(get_raw_frame_image(&camera.buf));
  }

  set_camera_exposure(calculate_exposure_value(&camera.buf, ae_xywh, 2, camera.cc.stream_type != VISION_STREAM_DRIVER ? 2 : 4));

  // Send the message
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  // TODO: centralize enabled handling

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  cl_context ctx = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));

  VisionIpcServer v("camerad", device_id, ctx);

  // *** initial ISP init ***
  SpectraMaster m;
  m.init();

  // *** per-cam init ***
  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    auto cam = std::make_unique<CameraState>(&m, config);
    cam->init(&v, device_id, ctx);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  // start devices
  LOG("-- Starting devices");
  for (auto &cam : cams) cam->camera.sensors_start();

  // poll events
  LOG("-- Dequeueing Video events");
  while (!do_exit) {
    struct pollfd fds[1] = {{.fd = m.video0_fd, .events = POLLPRI}};
    int ret = poll(fds, std::size(fds), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    if (!(fds[0].revents & POLLPRI)) continue;

    struct v4l2_event ev = {0};
    ret = HANDLE_EINTR(ioctl(fds[0].fd, VIDIOC_DQEVENT, &ev));
    if (ret == 0) {
      if (ev.type == V4L_EVENT_CAM_REQ_MGR_EVENT) {
        struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)ev.u.data;
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl,
                 event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp/1e6, event_data->u.frame_msg.sof_status);
          do_exit = do_exit || event_data->u.frame_msg.frame_id > (1*20);
        }

        for (auto &cam : cams) {
          if (event_data->session_hdl == cam->camera.session_handle) {
            if (cam->camera.handle_camera_event(event_data)) {
              cam->sendState();
            }
            break;
          }
        }
      } else {
        LOGE("unhandled event %d\n", ev.type);
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }
}
