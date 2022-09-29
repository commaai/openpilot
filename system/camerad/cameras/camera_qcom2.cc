#include "system/camerad/cameras/camera_qcom2.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

// For debugging:
// echo "4294967295" > /sys/module/cam_debug_util/parameters/debug_mdl

extern ExitHandler do_exit;

int CameraState::clear_req_queue() {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = camera->session_handle;
  req_mgr_flush_request.link_hdl = camera->link_handle;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
  return ret;
}

// ************** high level camera helpers ****************

void CameraState::enqueue_buffer(int i, bool dp) {
  int ret;
  int request_id = request_ids[i];

  if (buf_handle[i] && sync_objs[i]) {
    // wait
    struct cam_sync_wait sync_wait = {0};
    sync_wait.sync_obj = sync_objs[i];
    sync_wait.timeout_ms = 50; // max dt tolerance, typical should be 23
    ret = do_cam_control(multi_cam_state->cam_sync_fd, CAM_SYNC_WAIT, &sync_wait, sizeof(sync_wait));
    if (ret != 0) {
      LOGE("failed to wait for sync: %d %d", ret, sync_wait.sync_obj);
      // TODO: handle frame drop cleanly
    }

    buf.camera_bufs_metadata[i].timestamp_eof = (uint64_t)nanos_since_boot(); // set true eof
    if (dp) buf.queue(i);

    // destroy old output fence
    struct cam_sync_info sync_destroy = {0};
    sync_destroy.sync_obj = sync_objs[i];
    ret = do_cam_control(multi_cam_state->cam_sync_fd, CAM_SYNC_DESTROY, &sync_destroy, sizeof(sync_destroy));
    if (ret != 0) {
      LOGE("failed to destroy sync object: %d %d", ret, sync_destroy.sync_obj);
    }
  }

  // create output fence
  struct cam_sync_info sync_create = {0};
  strcpy(sync_create.name, "NodeOutputPortFence");
  ret = do_cam_control(multi_cam_state->cam_sync_fd, CAM_SYNC_CREATE, &sync_create, sizeof(sync_create));
  if (ret != 0) {
    LOGE("failed to create fence: %d %d", ret, sync_create.sync_obj)
  }
  sync_objs[i] = sync_create.sync_obj;

  // schedule request with camera request manager
  struct cam_req_mgr_sched_request req_mgr_sched_request = {0};
  req_mgr_sched_request.session_hdl = camera->session_handle;
  req_mgr_sched_request.link_hdl = camera->link_handle;
  req_mgr_sched_request.req_id = request_id;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_SCHED_REQ, &req_mgr_sched_request, sizeof(req_mgr_sched_request));
  if (ret != 0) {
    LOGE("failed to schedule cam mgr request: %d %d", ret, request_id);
  }

  // poke sensor, must happen after schedule
  camera->sensors_poke(request_id);

  // submit request to the ife
  camera->config_isp(buf_handle[i], sync_objs[i], request_id, camera->buf0_handle, 65632*(i+1));
}

void CameraState::enqueue_req_multi(int start, int n, bool dp) {
  for (int i=start;i<start+n;++i) {
    request_ids[(i - 1) % FRAME_BUF_COUNT] = i;
    enqueue_buffer((i - 1) % FRAME_BUF_COUNT, dp);
  }
}

// ******************* camera *******************

void CameraState::camera_set_parameters() {
  target_grey_fraction = 0.3;
  dc_gain_enabled = false;
  dc_gain_weight = camera->dc_gain_min_weight;
  gain_idx = camera->analog_gain_rec_idx;
  exposure_time = 5;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = (1 + dc_gain_weight * (camera->dc_gain_factor-1) / camera->dc_gain_max_weight) * camera->sensor_analog_gains[gain_idx] * exposure_time;
}

void CameraState::camera_map_bufs(MultiCameraState *s) {
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // configure ISP to put the image in place
    struct cam_mem_mgr_map_cmd mem_mgr_map_cmd = {0};
    mem_mgr_map_cmd.mmu_hdls[0] = s->device_iommu;
    mem_mgr_map_cmd.num_hdl = 1;
    mem_mgr_map_cmd.flags = CAM_MEM_FLAG_HW_READ_WRITE;
    mem_mgr_map_cmd.fd = buf.camera_bufs[i].fd;
    int ret = do_cam_control(s->video0_fd, CAM_REQ_MGR_MAP_BUF, &mem_mgr_map_cmd, sizeof(mem_mgr_map_cmd));
    LOGD("map buf req: (fd: %d) 0x%x %d", buf.camera_bufs[i].fd, mem_mgr_map_cmd.out.buf_handle, ret);
    buf_handle[i] = mem_mgr_map_cmd.out.buf_handle;
  }
  enqueue_req_multi(1, FRAME_BUF_COUNT, 0);
}

void CameraState::camera_init(MultiCameraState *s, VisionIpcServer * v, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType yuv_type) {
  if (!camera->enabled) return;

  LOGD("camera init %d", camera->camera_num);
  request_id_last = 0;
  skipped = true;
  camera_set_parameters();

  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, yuv_type);
  camera_map_bufs(s);
}

void CameraState::camera_open(MultiCameraState *multi_cam_state_, int camera_num_, bool enabled_) {
  multi_cam_state = multi_cam_state_;
  // TODO: use build flag instead?
  camera.reset(new CameraAR0231());
  int ret = camera->open(multi_cam_state_, camera_num_, enabled_);
  if (ret != 0) {
    camera.reset(new CameraOX03C10());
    camera->open(multi_cam_state_, camera_num_, enabled_);
  }
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->driver_cam.camera_init(s, v, 20, device_id, ctx, VISION_STREAM_DRIVER);
  s->road_cam.camera_init(s, v, 20, device_id, ctx, VISION_STREAM_ROAD);
  s->wide_road_cam.camera_init(s, v, 20, device_id, ctx, VISION_STREAM_WIDE_ROAD);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "wideRoadCameraState", "thumbnail"});
}

void cameras_open(MultiCameraState *s) {
  int ret;

  LOG("-- Opening devices");
  // video0 is req_mgr, the target of many ioctls
  s->video0_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-soc:qcom_cam-req-mgr-video-index0", O_RDWR | O_NONBLOCK));
  assert(s->video0_fd >= 0);
  LOGD("opened video0");

  // video1 is cam_sync, the target of some ioctls
  s->cam_sync_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-cam_sync-video-index0", O_RDWR | O_NONBLOCK));
  assert(s->cam_sync_fd >= 0);
  LOGD("opened video1 (cam_sync)");

  // looks like there's only one of these
  s->isp_fd = open_v4l_by_name_and_index("cam-isp");
  assert(s->isp_fd >= 0);
  LOGD("opened isp");

  // query icp for MMU handles
  LOG("-- Query ICP for MMU handles");
  struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  ret = do_cam_control(s->isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  s->device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  s->cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;

  // subscribe
  LOG("-- Subscribing");
  struct v4l2_event_subscription sub = {0};
  sub.type = V4L_EVENT_CAM_REQ_MGR_EVENT;
  sub.id = V4L_EVENT_CAM_REQ_MGR_SOF_BOOT_TS;
  ret = HANDLE_EINTR(ioctl(s->video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));
  LOGD("req mgr subscribe: %d", ret);

  s->driver_cam.camera_open(s, 2, !env_disable_driver);
  LOGD("driver camera opened");
  s->road_cam.camera_open(s, 1, !env_disable_road);
  LOGD("road camera opened");
  s->wide_road_cam.camera_open(s, 0, !env_disable_wide_road);
  LOGD("wide road camera opened");
}

void CameraState::camera_close() {
  camera->close();
  if (camera->enabled) {
    for (int i = 0; i < FRAME_BUF_COUNT; i++) {
      release(multi_cam_state->video0_fd, buf_handle[i]);
    }
    LOGD("released buffers");
  }
}

void cameras_close(MultiCameraState *s) {
  s->driver_cam.camera_close();
  s->road_cam.camera_close();
  s->wide_road_cam.camera_close();

  delete s->pm;
}

void CameraState::handle_camera_event(void *evdat) {
  if (!camera->enabled) return;
  struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)evdat;
  assert(event_data->session_hdl == camera->session_handle);
  assert(event_data->u.frame_msg.link_hdl == camera->link_handle);

  uint64_t timestamp = event_data->u.frame_msg.timestamp;
  int main_id = event_data->u.frame_msg.frame_id;
  int real_id = event_data->u.frame_msg.request_id;

  if (real_id != 0) { // next ready
    if (real_id == 1) {idx_offset = main_id;}
    int buf_idx = (real_id - 1) % FRAME_BUF_COUNT;

    // check for skipped frames
    if (main_id > frame_id_last + 1 && !skipped) {
      LOGE("camera %d realign", camera->camera_num);
      clear_req_queue();
      enqueue_req_multi(real_id + 1, FRAME_BUF_COUNT - 1, 0);
      skipped = true;
    } else if (main_id == frame_id_last + 1) {
      skipped = false;
    }

    // check for dropped requests
    if (real_id > request_id_last + 1) {
      LOGE("camera %d dropped requests %d %d", camera->camera_num, real_id, request_id_last);
      enqueue_req_multi(request_id_last + 1 + FRAME_BUF_COUNT, real_id - (request_id_last + 1), 0);
    }

    // metas
    frame_id_last = main_id;
    request_id_last = real_id;

    auto &meta_data = buf.camera_bufs_metadata[buf_idx];
    meta_data.frame_id = main_id - idx_offset;
    meta_data.timestamp_sof = timestamp;
    exp_lock.lock();
    meta_data.gain = analog_gain_frac * (1 + dc_gain_weight * (camera->dc_gain_factor-1) / camera->dc_gain_max_weight);
    meta_data.high_conversion_gain = dc_gain_enabled;
    meta_data.integ_lines = exposure_time;
    meta_data.measured_grey_fraction = measured_grey_fraction;
    meta_data.target_grey_fraction = target_grey_fraction;
    exp_lock.unlock();

    // dispatch
    enqueue_req_multi(real_id + FRAME_BUF_COUNT, 1, 1);
  } else { // not ready
    if (main_id > frame_id_last + 10) {
      LOGE("camera %d reset after half second of no response", camera->camera_num);
      clear_req_queue();
      enqueue_req_multi(request_id_last + 1, FRAME_BUF_COUNT, 0);
      frame_id_last = main_id;
      skipped = true;
    }
  }
}

void CameraState::set_camera_exposure(float grey_frac) {
  if (!camera->enabled) return;
  const float dt = 0.05;

  const float ts_grey = 10.0;
  const float ts_ev = 0.05;

  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  // It takes 3 frames for the commanded exposure settings to take effect. The first frame is already started by the time
  // we reach this function, the other 2 are due to the register buffering in the sensor.
  // Therefore we use the target EV from 3 frames ago, the grey fraction that was just measured was the result of that control action.
  // TODO: Lower latency to 2 frames, by using the histogram outputted by the sensor we can do AE before the debayering is complete

  const float cur_ev_ = cur_ev[buf.cur_frame_data.frame_id % 3];

  // Scale target grey between 0.1 and 0.4 depending on lighting conditions
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + camera->target_grey_factor*cur_ev_) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, camera->min_ev, camera->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  float best_ev_score = 1e6;
  int new_g = 0;
  int new_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < camera->dc_gain_on_grey) {
    enable_dc_gain = true;
    dc_gain_weight = camera->dc_gain_min_weight;
  } else if (enable_dc_gain && target_grey > camera->dc_gain_off_grey) {
    enable_dc_gain = false;
    dc_gain_weight = camera->dc_gain_max_weight;
  }

  if (enable_dc_gain && dc_gain_weight < camera->dc_gain_max_weight) {dc_gain_weight += 1;}
  if (!enable_dc_gain && dc_gain_weight > camera->dc_gain_min_weight) {dc_gain_weight -= 1;}

  std::string gain_bytes, time_bytes;
  if (env_ctrl_exp_from_params) {
    gain_bytes = Params().get("CameraDebugExpGain");
    time_bytes = Params().get("CameraDebugExpTime");
  }

  if (gain_bytes.size() > 0 && time_bytes.size() > 0) {
    // Override gain and exposure time
    gain_idx = std::stoi(gain_bytes);
    exposure_time = std::stoi(time_bytes);

    new_g = gain_idx;
    new_t = exposure_time;
    enable_dc_gain = false;
  } else {
    // Simple brute force optimizer to choose sensor parameters
    // to reach desired EV
    for (int g = std::max((int)camera->analog_gain_min_idx, gain_idx - 1); g <= std::min((int)camera->analog_gain_max_idx, gain_idx + 1); g++) {
      float gain = camera->sensor_analog_gains[g] * (1 + dc_gain_weight * (camera->dc_gain_factor-1) / camera->dc_gain_max_weight);

      // Compute optimal time for given gain
      int t = std::clamp(int(std::round(desired_ev / gain)), camera->exposure_time_min, camera->exposure_time_max);

      // Only go below recommended gain when absolutely necessary to not overexpose
      if (g < camera->analog_gain_rec_idx && t > 20 && g < gain_idx) {
        continue;
      }

      // Compute error to desired ev
      float score = std::abs(desired_ev - (t * gain)) * 10;

      // Going below recommended gain needs lower penalty to not overexpose
      float m = g > camera->analog_gain_rec_idx ? 5.0 : 0.1;
      score += std::abs(g - (int)camera->analog_gain_rec_idx) * m;

      // LOGE("cam: %d - gain: %d, t: %d (%.2f), score %.2f, score + gain %.2f, %.3f, %.3f", camera_num, g, t, desired_ev / gain, score, score + std::abs(g - gain_idx) * (score + 1.0) / 10.0, desired_ev, min_ev);

      // Small penalty on changing gain
      score += std::abs(g - gain_idx) * (score + 1.0) / 10.0;

      if (score < best_ev_score) {
        new_t = t;
        new_g = g;
        best_ev_score = score;
      }
    }
  }

  exp_lock.lock();

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = camera->sensor_analog_gains[new_g];
  gain_idx = new_g;
  exposure_time = new_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * (1 + dc_gain_weight * (camera->dc_gain_factor-1) / camera->dc_gain_max_weight);
  cur_ev[buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - buf.cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", camera_num, 1e-9 * nanos_since_boot(), 1e-9 * buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - buf.cur_frame_data.timestamp_sof));

  auto exp_vector = camera->getExposureVector(new_g, dc_gain_enabled, exposure_time, dc_gain_weight);
  camera->sensors_i2c(exp_vector, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
}

static void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  c->set_camera_exposure(set_exposure_target(&c->buf, 96, 1832, 2, 242, 1148, 4));

  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data, c);

  c->camera->processRegisters(c->buf.cur_camera_buf->addr, framed);
  s->pm->send("driverCameraState", msg);
}

void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;

  MessageBuilder msg;
  auto framed = c == &s->road_cam ? msg.initEvent().initRoadCameraState() : msg.initEvent().initWideRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data, c);
  if (env_log_raw_frames && c == &s->road_cam && cnt % 100 == 5) {  // no overlap with qlog decimation
    framed.setImage(get_raw_frame_image(b));
  }
  LOGT(c->buf.cur_frame_data.frame_id, "%s: Image set", c == &s->road_cam ? "RoadCamera" : "WideRoadCamera");
  if (c == &s->road_cam) {
    framed.setTransform(b->yuv_transform.v);
    LOGT(c->buf.cur_frame_data.frame_id, "%s: Transformed", "RoadCamera");
  }

  c->camera->processRegisters(c->buf.cur_camera_buf->addr, framed);
  s->pm->send(c == &s->road_cam ? "roadCameraState" : "wideRoadCameraState", msg);

  const auto [x, y, w, h] = (c == &s->wide_road_cam) ? std::tuple(96, 250, 1734, 524) : std::tuple(96, 160, 1734, 986);
  const int skip = 2;
  c->set_camera_exposure(set_exposure_target(b, x, x + w, skip, y, y + h, skip));
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  if (s->driver_cam.camera->enabled) threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));
  if (s->road_cam.camera->enabled) threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  if (s->wide_road_cam.camera->enabled) threads.push_back(start_process_thread(s, &s->wide_road_cam, process_road_camera));

  // start devices
  LOG("-- Starting devices");
  s->driver_cam.camera->sensors_start();
  s->road_cam.camera->sensors_start();
  s->wide_road_cam.camera->sensors_start();

  // poll events
  LOG("-- Dequeueing Video events");
  while (!do_exit) {
    struct pollfd fds[1] = {{0}};

    fds[0].fd = s->video0_fd;
    fds[0].events = POLLPRI;

    int ret = poll(fds, std::size(fds), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    if (!fds[0].revents) continue;

    struct v4l2_event ev = {0};
    ret = HANDLE_EINTR(ioctl(fds[0].fd, VIDIOC_DQEVENT, &ev));
    if (ret == 0) {
      if (ev.type == V4L_EVENT_CAM_REQ_MGR_EVENT) {
        struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)ev.u.data;
        // LOGD("v4l2 event: sess_hdl 0x%X, link_hdl 0x%X, frame_id %d, req_id %lld, timestamp 0x%llx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp/1e6, event_data->u.frame_msg.sof_status);
        }

        if (event_data->session_hdl == s->road_cam.camera->session_handle) {
          s->road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->wide_road_cam.camera->session_handle) {
          s->wide_road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->driver_cam.camera->session_handle) {
          s->driver_cam.handle_camera_event(event_data);
        } else {
          LOGE("Unknown vidioc event source");
          assert(false);
        }
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}
