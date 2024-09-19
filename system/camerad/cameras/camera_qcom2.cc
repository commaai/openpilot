#include "system/camerad/cameras/camera_qcom2.h"

#include <poll.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "CL/cl_ext_qcom.h"

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_req_mgr.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"

#include "common/clutil.h"
#include "common/params.h"
#include "common/swaglog.h"

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

ExitHandler do_exit;


int SpectraCamera::clear_req_queue() {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = session_handle;
  req_mgr_flush_request.link_hdl = link_handle;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
  return ret;
}

// ************** high level camera helpers ****************

void SpectraCamera::config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  if (io_mem_handle != 0) {
    size += sizeof(struct cam_buf_io_cfg);
  }
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = 0;
  // YUV has kmd_cmd_buf_offset = 1780
  // I guess this is the ISP command
  // YUV also has patch_offset = 0x1030 and num_patches = 10

  if (io_mem_handle != 0) {
    pkt->io_configs_offset = sizeof(struct cam_cmd_buf_desc)*pkt->num_cmd_buf;
    pkt->num_io_configs = 1;
  }

  if (io_mem_handle != 0) {
    pkt->header.op_code = 0xf000001;
    pkt->header.request_id = request_id;
  } else {
    pkt->header.op_code = 0xf000000;
    pkt->header.request_id = 1;
  }
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;
  struct cam_buf_io_cfg *io_cfg = (struct cam_buf_io_cfg *)((char*)&pkt->payload + pkt->io_configs_offset);

  // TODO: support MMU
  buf_desc[0].size = 65624;
  buf_desc[0].length = 0;
  buf_desc[0].type = CAM_CMD_BUF_DIRECT;
  buf_desc[0].meta_data = 3;
  buf_desc[0].mem_handle = buf0_mem_handle;
  buf_desc[0].offset = buf0_offset;

  // parsed by cam_isp_packet_generic_blob_handler
  struct isp_packet {
    uint32_t type_0;
    cam_isp_resource_hfr_config resource_hfr;

    uint32_t type_1;
    cam_isp_clock_config clock;
    uint64_t extra_rdi_hz[3];

    uint32_t type_2;
    cam_isp_bw_config bw;
    struct cam_isp_bw_vote extra_rdi_vote[6];
  } __attribute__((packed)) tmp;
  memset(&tmp, 0, sizeof(tmp));

  tmp.type_0 = CAM_ISP_GENERIC_BLOB_TYPE_HFR_CONFIG;
  tmp.type_0 |= sizeof(cam_isp_resource_hfr_config) << 8;
  static_assert(sizeof(cam_isp_resource_hfr_config) == 0x20);
  tmp.resource_hfr = {
    .num_ports = 1,  // 10 for YUV (but I don't think we need them)
    .port_hfr_config[0] = {
      .resource_type = CAM_ISP_IFE_OUT_RES_RDI_0, // CAM_ISP_IFE_OUT_RES_FULL for YUV
      .subsample_pattern = 1,
      .subsample_period = 0,
      .framedrop_pattern = 1,
      .framedrop_period = 0,
    }};

  tmp.type_1 = CAM_ISP_GENERIC_BLOB_TYPE_CLOCK_CONFIG;
  tmp.type_1 |= (sizeof(cam_isp_clock_config) + sizeof(tmp.extra_rdi_hz)) << 8;
  static_assert((sizeof(cam_isp_clock_config) + sizeof(tmp.extra_rdi_hz)) == 0x38);
  tmp.clock = {
    .usage_type = 1, // dual mode
    .num_rdi = 4,
    .left_pix_hz = 404000000,
    .right_pix_hz = 404000000,
    .rdi_hz[0] = 404000000,
  };


  tmp.type_2 = CAM_ISP_GENERIC_BLOB_TYPE_BW_CONFIG;
  tmp.type_2 |= (sizeof(cam_isp_bw_config) + sizeof(tmp.extra_rdi_vote)) << 8;
  static_assert((sizeof(cam_isp_bw_config) + sizeof(tmp.extra_rdi_vote)) == 0xe0);
  tmp.bw = {
    .usage_type = 1, // dual mode
    .num_rdi = 4,
    .left_pix_vote = {
      .resource_id = 0,
      .cam_bw_bps = 450000000,
      .ext_bw_bps = 450000000,
    },
    .rdi_vote[0] = {
      .resource_id = 0,
      .cam_bw_bps = 8706200000,
      .ext_bw_bps = 8706200000,
    },
  };

  static_assert(offsetof(struct isp_packet, type_2) == 0x60);

  buf_desc[1].size = sizeof(tmp);
  buf_desc[1].offset = io_mem_handle != 0 ? 0x60 : 0;
  buf_desc[1].length = buf_desc[1].size - buf_desc[1].offset;
  buf_desc[1].type = CAM_CMD_BUF_GENERIC;
  buf_desc[1].meta_data = CAM_ISP_PACKET_META_GENERIC_BLOB_COMMON;
  auto buf2 = mm.alloc<uint32_t>(buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle);
  memcpy(buf2.get(), &tmp, sizeof(tmp));

  if (io_mem_handle != 0) {
    io_cfg[0].mem_handle[0] = io_mem_handle;
    io_cfg[0].planes[0] = (struct cam_plane_cfg){
      .width = sensor->frame_width,
      .height = sensor->frame_height + sensor->extra_height,
      .plane_stride = sensor->frame_stride,
      .slice_height = sensor->frame_height + sensor->extra_height,
      .meta_stride = 0x0,  // YUV has meta(stride=0x400, size=0x5000)
      .meta_size = 0x0,
      .meta_offset = 0x0,
      .packer_config = 0x0,  // 0xb for YUV
      .mode_config = 0x0,    // 0x9ef for YUV
      .tile_config = 0x0,
      .h_init = 0x0,
      .v_init = 0x0,
    };
    io_cfg[0].format = sensor->mipi_format;                    // CAM_FORMAT_UBWC_TP10 for YUV
    io_cfg[0].color_space = CAM_COLOR_SPACE_BASE;          // CAM_COLOR_SPACE_BT601_FULL for YUV
    io_cfg[0].color_pattern = 0x5;                         // 0x0 for YUV
    io_cfg[0].bpp = (sensor->mipi_format == CAM_FORMAT_MIPI_RAW_10 ? 0xa : 0xc);  // bits per pixel
    io_cfg[0].resource_type = CAM_ISP_IFE_OUT_RES_RDI_0;   // CAM_ISP_IFE_OUT_RES_FULL for YUV
    io_cfg[0].fence = fence;
    io_cfg[0].direction = CAM_BUF_OUTPUT;
    io_cfg[0].subsample_pattern = 0x1;
    io_cfg[0].framedrop_pattern = 0x1;
  }

  int ret = device_config(m->isp_fd, session_handle, isp_dev_handle, cam_packet_handle);
  assert(ret == 0);
  if (ret != 0) {
    LOGE("isp config failed");
  }
}

void SpectraCamera::enqueue_buffer(int i, bool dp) {
  int ret;
  uint64_t request_id = request_ids[i];

  if (buf_handle[i] && sync_objs[i]) {
    // wait
    struct cam_sync_wait sync_wait = {0};
    sync_wait.sync_obj = sync_objs[i];
    sync_wait.timeout_ms = 50; // max dt tolerance, typical should be 23
    ret = do_cam_control(m->cam_sync_fd, CAM_SYNC_WAIT, &sync_wait, sizeof(sync_wait));
    if (ret != 0) {
      LOGE("failed to wait for sync: %d %d", ret, sync_wait.sync_obj);
      // TODO: handle frame drop cleanly
    }

    buf.camera_bufs_metadata[i].timestamp_eof = (uint64_t)nanos_since_boot(); // set true eof
    if (dp) buf.queue(i);

    // destroy old output fence
    struct cam_sync_info sync_destroy = {0};
    sync_destroy.sync_obj = sync_objs[i];
    ret = do_cam_control(m->cam_sync_fd, CAM_SYNC_DESTROY, &sync_destroy, sizeof(sync_destroy));
    if (ret != 0) {
      LOGE("failed to destroy sync object: %d %d", ret, sync_destroy.sync_obj);
    }
  }

  // create output fence
  struct cam_sync_info sync_create = {0};
  strcpy(sync_create.name, "NodeOutputPortFence");
  ret = do_cam_control(m->cam_sync_fd, CAM_SYNC_CREATE, &sync_create, sizeof(sync_create));
  if (ret != 0) {
    LOGE("failed to create fence: %d %d", ret, sync_create.sync_obj);
  }
  sync_objs[i] = sync_create.sync_obj;

  // schedule request with camera request manager
  struct cam_req_mgr_sched_request req_mgr_sched_request = {0};
  req_mgr_sched_request.session_hdl = session_handle;
  req_mgr_sched_request.link_hdl = link_handle;
  req_mgr_sched_request.req_id = request_id;
  ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_SCHED_REQ, &req_mgr_sched_request, sizeof(req_mgr_sched_request));
  if (ret != 0) {
    LOGE("failed to schedule cam mgr request: %d %lu", ret, request_id);
  }

  // poke sensor, must happen after schedule
  sensors_poke(request_id);

  // submit request to the ife
  config_isp(buf_handle[i], sync_objs[i], request_id, buf0_handle, 65632*(i+1));
}

// ******************* camera *******************

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
    exposure target intrinics is
    [
      [F, 0, 0.5*ae_xywh[2]]
      [0, F, 0.5*H-ae_xywh[1]]
      [0, 0, 1]
    ]
  */
  auto ae_target = ae_targets[cc.camera_num];
  Rect xywh_ref = ae_target.first;
  float fl_ref = ae_target.second;

  ae_xywh = (Rect){
    std::max(0, buf.rgb_width / 2 - (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
    std::max(0, buf.rgb_height / 2 - (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y))),
    std::min((int)(fl_pix / fl_ref * xywh_ref.w), buf.rgb_width / 2 + (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
    std::min((int)(fl_pix / fl_ref * xywh_ref.h), buf.rgb_height / 2 + (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y)))
  };
}

void CameraState::sensor_set_parameters() {
  dc_gain_weight = sensor->dc_gain_min_weight;
  gain_idx = sensor->analog_gain_rec_idx;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = (1 + dc_gain_weight * (sensor->dc_gain_factor-1) / sensor->dc_gain_max_weight) * sensor->sensor_analog_gains[gain_idx] * exposure_time;
}

void SpectraCamera::camera_map_bufs() {
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // configure ISP to put the image in place
    struct cam_mem_mgr_map_cmd mem_mgr_map_cmd = {0};
    mem_mgr_map_cmd.mmu_hdls[0] = m->device_iommu;
    mem_mgr_map_cmd.num_hdl = 1;
    mem_mgr_map_cmd.flags = CAM_MEM_FLAG_HW_READ_WRITE;
    mem_mgr_map_cmd.fd = buf.camera_bufs[i].fd;
    int ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_MAP_BUF, &mem_mgr_map_cmd, sizeof(mem_mgr_map_cmd));
    LOGD("map buf req: (fd: %d) 0x%x %d", buf.camera_bufs[i].fd, mem_mgr_map_cmd.out.buf_handle, ret);
    buf_handle[i] = mem_mgr_map_cmd.out.buf_handle;
  }
  enqueue_req_multi(1, FRAME_BUF_COUNT, 0);
}

bool SpectraCamera::openSensor() {
  sensor_fd = open_v4l_by_name_and_index("cam-sensor-driver", cc.camera_num);
  assert(sensor_fd >= 0);
  LOGD("opened sensor for %d", cc.camera_num);

  // init memorymanager for this camera
  mm.init(m->video0_fd);

  LOGD("-- Probing sensor %d", cc.camera_num);

  auto init_sensor_lambda = [this](SensorInfo *s) {
    sensor.reset(s);
    int ret = sensors_init();
    if (ret == 0) {
      // TODO: add this back
      //sensor_set_parameters();
    }
    return ret == 0;
  };

  // Try different sensors one by one until it success.
  if (!init_sensor_lambda(new AR0231) &&
      !init_sensor_lambda(new OX03C10) &&
      !init_sensor_lambda(new OS04C10)) {
    LOGE("** sensor %d FAILED bringup, disabling", cc.camera_num);
    enabled = false;
    return false;
  }
  LOGD("-- Probing sensor %d success", cc.camera_num);

  // create session
  struct cam_req_mgr_session_info session_info = {};
  int ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_CREATE_SESSION, &session_info, sizeof(session_info));
  LOGD("get session: %d 0x%X", ret, session_info.session_hdl);
  session_handle = session_info.session_hdl;

  // access the sensor
  LOGD("-- Accessing sensor");
  auto sensor_dev_handle_ = device_acquire(sensor_fd, session_handle, nullptr);
  assert(sensor_dev_handle_);
  sensor_dev_handle = *sensor_dev_handle_;
  LOGD("acquire sensor dev");

  LOG("-- Configuring sensor");
  sensors_i2c(sensor->init_reg_array.data(), sensor->init_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, sensor->data_word);
  return true;
}

void SpectraCamera::configISP() {
  // NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
  // If you don't do this, the strobe GPIO is an output (even in reset it seems!)
  if (!enabled) return;

  struct cam_isp_in_port_info in_port_info = {
    .res_type = cc.phy,
    .lane_type = CAM_ISP_LANE_TYPE_DPHY,
    .lane_num = 4,
    .lane_cfg = 0x3210,

    .vc = 0x0,
    .dt = sensor->frame_data_type,
    .format = sensor->mipi_format,

    .test_pattern = 0x2,  // 0x3?
    .usage_type = 0x0,

    .left_start = 0,
    .left_stop = sensor->frame_width - 1,
    .left_width = sensor->frame_width,

    .right_start = 0,
    .right_stop = sensor->frame_width - 1,
    .right_width = sensor->frame_width,

    .line_start = 0,
    .line_stop = sensor->frame_height + sensor->extra_height - 1,
    .height = sensor->frame_height + sensor->extra_height,

    .pixel_clk = 0x0,
    .batch_size = 0x0,
    .dsp_mode = CAM_ISP_DSP_MODE_NONE,
    .hbi_cnt = 0x0,
    .custom_csid = 0x0,

    .num_out_res = 0x1,
    .data[0] = (struct cam_isp_out_port_info){
      .res_type = CAM_ISP_IFE_OUT_RES_RDI_0,
      .format = sensor->mipi_format,
      .width = sensor->frame_width,
      .height = sensor->frame_height + sensor->extra_height,
      .comp_grp_id = 0x0, .split_point = 0x0, .secure_mode = 0x0,
    },
  };
  struct cam_isp_resource isp_resource = {
    .resource_id = CAM_ISP_RES_ID_PORT,
    .handle_type = CAM_HANDLE_USER_POINTER,
    .res_hdl = (uint64_t)&in_port_info,
    .length = sizeof(in_port_info),
  };

  auto isp_dev_handle_ = device_acquire(m->isp_fd, session_handle, &isp_resource);
  assert(isp_dev_handle_);
  isp_dev_handle = *isp_dev_handle_;
  LOGD("acquire isp dev");

  // config ISP
  alloc_w_mmu_hdl(m->video0_fd, 984480, (uint32_t*)&buf0_handle, 0x20, CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS |
                  CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, m->device_iommu, m->cdm_iommu);
  config_isp(0, 0, 1, buf0_handle, 0);
}

void SpectraCamera::configCSIPHY() {
  csiphy_fd = open_v4l_by_name_and_index("cam-csiphy-driver", cc.camera_num);
  assert(csiphy_fd >= 0);
  LOGD("opened csiphy for %d", cc.camera_num);

  struct cam_csiphy_acquire_dev_info csiphy_acquire_dev_info = {.combo_mode = 0};
  auto csiphy_dev_handle_ = device_acquire(csiphy_fd, session_handle, &csiphy_acquire_dev_info);
  assert(csiphy_dev_handle_);
  csiphy_dev_handle = *csiphy_dev_handle_;
  LOGD("acquire csiphy dev");

  // config csiphy
  LOG("-- Config CSI PHY");
  {
    uint32_t cam_packet_handle = 0;
    int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
    auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
    pkt->num_cmd_buf = 1;
    pkt->kmd_cmd_buf_index = -1;
    pkt->header.size = size;
    struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

    buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_csiphy_info);
    buf_desc[0].type = CAM_CMD_BUF_GENERIC;

    auto csiphy_info = mm.alloc<struct cam_csiphy_info>(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
    csiphy_info->lane_mask = 0x1f;
    csiphy_info->lane_assign = 0x3210;// skip clk. How is this 16 bit for 5 channels??
    csiphy_info->csiphy_3phase = 0x0; // no 3 phase, only 2 conductors per lane
    csiphy_info->combo_mode = 0x0;
    csiphy_info->lane_cnt = 0x4;
    csiphy_info->secure_mode = 0x0;
    csiphy_info->settle_time = MIPI_SETTLE_CNT * 200000000ULL;
    csiphy_info->data_rate = 48000000;  // Calculated by camera_freqs.py

    int ret_ = device_config(csiphy_fd, session_handle, csiphy_dev_handle, cam_packet_handle);
    assert(ret_ == 0);
  }
}

void SpectraCamera::linkDevices() {
  LOG("-- Link devices");
  struct cam_req_mgr_link_info req_mgr_link_info = {0};
  req_mgr_link_info.session_hdl = session_handle;
  req_mgr_link_info.num_devices = 2;
  req_mgr_link_info.dev_hdls[0] = isp_dev_handle;
  req_mgr_link_info.dev_hdls[1] = sensor_dev_handle;
  int ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_LINK, &req_mgr_link_info, sizeof(req_mgr_link_info));
  link_handle = req_mgr_link_info.link_hdl;
  LOGD("link: %d session: 0x%X isp: 0x%X sensors: 0x%X link: 0x%X", ret, session_handle, isp_dev_handle, sensor_dev_handle, link_handle);

  struct cam_req_mgr_link_control req_mgr_link_control = {0};
  req_mgr_link_control.ops = CAM_REQ_MGR_LINK_ACTIVATE;
  req_mgr_link_control.session_hdl = session_handle;
  req_mgr_link_control.num_links = 1;
  req_mgr_link_control.link_hdls[0] = link_handle;
  ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
  LOGD("link control: %d", ret);

  ret = device_control(csiphy_fd, CAM_START_DEV, session_handle, csiphy_dev_handle);
  LOGD("start csiphy: %d", ret);
  ret = device_control(m->isp_fd, CAM_START_DEV, session_handle, isp_dev_handle);
  LOGD("start isp: %d", ret);
  assert(ret == 0);

  // TODO: this is unneeded, should we be doing the start i2c in a different way?
  //ret = device_control(sensor_fd, CAM_START_DEV, session_handle, sensor_dev_handle);
  //LOGD("start sensor: %d", ret);
}

void SpectraCamera::camera_close() {
  // stop devices
  LOG("-- Stop devices %d", cc.camera_num);

  if (enabled) {
    // ret = device_control(sensor_fd, CAM_STOP_DEV, session_handle, sensor_dev_handle);
    // LOGD("stop sensor: %d", ret);
    int ret = device_control(m->isp_fd, CAM_STOP_DEV, session_handle, isp_dev_handle);
    LOGD("stop isp: %d", ret);
    ret = device_control(csiphy_fd, CAM_STOP_DEV, session_handle, csiphy_dev_handle);
    LOGD("stop csiphy: %d", ret);
    // link control stop
    LOG("-- Stop link control");
    struct cam_req_mgr_link_control req_mgr_link_control = {0};
    req_mgr_link_control.ops = CAM_REQ_MGR_LINK_DEACTIVATE;
    req_mgr_link_control.session_hdl = session_handle;
    req_mgr_link_control.num_links = 1;
    req_mgr_link_control.link_hdls[0] = link_handle;
    ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
    LOGD("link control stop: %d", ret);

    // unlink
    LOG("-- Unlink");
    struct cam_req_mgr_unlink_info req_mgr_unlink_info = {0};
    req_mgr_unlink_info.session_hdl = session_handle;
    req_mgr_unlink_info.link_hdl = link_handle;
    ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_UNLINK, &req_mgr_unlink_info, sizeof(req_mgr_unlink_info));
    LOGD("unlink: %d", ret);

    // release devices
    LOGD("-- Release devices");
    ret = device_control(m->isp_fd, CAM_RELEASE_DEV, session_handle, isp_dev_handle);
    LOGD("release isp: %d", ret);
    ret = device_control(csiphy_fd, CAM_RELEASE_DEV, session_handle, csiphy_dev_handle);
    LOGD("release csiphy: %d", ret);

    for (int i = 0; i < FRAME_BUF_COUNT; i++) {
      release(m->video0_fd, buf_handle[i]);
    }
    LOGD("released buffers");
  }

  int ret = device_control(sensor_fd, CAM_RELEASE_DEV, session_handle, sensor_dev_handle);
  LOGD("release sensor: %d", ret);

  // destroyed session
  struct cam_req_mgr_session_info session_info = {.session_hdl = session_handle};
  ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_DESTROY_SESSION, &session_info, sizeof(session_info));
  LOGD("destroyed session %d: %d", cc.camera_num, ret);
}

void CameraState::handle_camera_event(void *evdat) {
  if (!enabled) return;
  struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)evdat;
  assert(event_data->session_hdl == session_handle);
  assert(event_data->u.frame_msg.link_hdl == link_handle);

  uint64_t timestamp = event_data->u.frame_msg.timestamp;
  uint64_t main_id = event_data->u.frame_msg.frame_id;
  uint64_t real_id = event_data->u.frame_msg.request_id;

  if (real_id != 0) { // next ready
    if (real_id == 1) {idx_offset = main_id;}
    int buf_idx = (real_id - 1) % FRAME_BUF_COUNT;

    // check for skipped frames
    if (main_id > frame_id_last + 1 && !skipped) {
      LOGE("camera %d realign", cc.camera_num);
      clear_req_queue();
      enqueue_req_multi(real_id + 1, FRAME_BUF_COUNT - 1, 0);
      skipped = true;
    } else if (main_id == frame_id_last + 1) {
      skipped = false;
    }

    // check for dropped requests
    if (real_id > request_id_last + 1) {
      LOGE("camera %d dropped requests %ld %ld", cc.camera_num, real_id, request_id_last);
      enqueue_req_multi(request_id_last + 1 + FRAME_BUF_COUNT, real_id - (request_id_last + 1), 0);
    }

    // metas
    frame_id_last = main_id;
    request_id_last = real_id;

    auto &meta_data = buf.camera_bufs_metadata[buf_idx];
    meta_data.frame_id = main_id - idx_offset;
    meta_data.request_id = real_id;
    meta_data.timestamp_sof = timestamp;
    exp_lock.lock();
    meta_data.gain = analog_gain_frac * (1 + dc_gain_weight * (sensor->dc_gain_factor-1) / sensor->dc_gain_max_weight);
    meta_data.high_conversion_gain = dc_gain_enabled;
    meta_data.integ_lines = exposure_time;
    meta_data.measured_grey_fraction = measured_grey_fraction;
    meta_data.target_grey_fraction = target_grey_fraction;
    exp_lock.unlock();

    // dispatch
    enqueue_req_multi(real_id + FRAME_BUF_COUNT, 1, 1);
  } else { // not ready
    if (main_id > frame_id_last + 10) {
      LOGE("camera %d reset after half second of no response", cc.camera_num);
      clear_req_queue();
      enqueue_req_multi(request_id_last + 1, FRAME_BUF_COUNT, 0);
      frame_id_last = main_id;
      skipped = true;
    }
  }
}

void CameraState::update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
  float score = sensor->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
  if (score < best_ev_score) {
    new_exp_t = exp_t;
    new_exp_g = exp_g_idx;
    best_ev_score = score;
  }
}

void CameraState::set_camera_exposure(float grey_frac) {
  if (!enabled) return;
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
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + sensor->target_grey_factor*cur_ev_) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, sensor->min_ev, sensor->max_ev);
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
    // Simple brute force optimizer to choose sensor parameters
    // to reach desired EV
    for (int g = std::max((int)sensor->analog_gain_min_idx, gain_idx - 1); g <= std::min((int)sensor->analog_gain_max_idx, gain_idx + 1); g++) {
      float gain = sensor->sensor_analog_gains[g] * (1 + dc_gain_weight * (sensor->dc_gain_factor-1) / sensor->dc_gain_max_weight);

      // Compute optimal time for given gain
      int t = std::clamp(int(std::round(desired_ev / gain)), sensor->exposure_time_min, sensor->exposure_time_max);

      // Only go below recommended gain when absolutely necessary to not overexpose
      if (g < sensor->analog_gain_rec_idx && t > 20 && g < gain_idx) {
        continue;
      }

      update_exposure_score(desired_ev, t, g, gain);
    }
  }

  exp_lock.lock();

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = sensor->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * (1 + dc_gain_weight * (sensor->dc_gain_factor-1) / sensor->dc_gain_max_weight);
  cur_ev[buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - buf.cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", cc.camera_num, 1e-9 * nanos_since_boot(), 1e-9 * buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - buf.cur_frame_data.timestamp_sof));

  auto exp_reg_array = sensor->getExposureRegisters(exposure_time, new_exp_g, dc_gain_enabled);
  sensors_i2c(exp_reg_array.data(), exp_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, sensor->data_word);
}

void CameraState::run() {
  util::set_thread_name(cc.publish_name);

  std::vector<const char*> pubs = {cc.publish_name};
  if (cc.stream_type == VISION_STREAM_ROAD) pubs.push_back("thumbnail");
  PubMaster pm(pubs);

  for (uint32_t cnt = 0; !do_exit; ++cnt) {
    // Acquire the buffer; continue if acquisition fails
    if (!buf.acquire()) continue;

    MessageBuilder msg;
    auto framed = (msg.initEvent().*cc.init_camera_state)();
    fill_frame_data(framed, buf.cur_frame_data, this);

    // Log raw frames for road camera
    if (env_log_raw_frames && cc.stream_type == VISION_STREAM_ROAD && cnt % 100 == 5) {  // no overlap with qlog decimation
      framed.setImage(get_raw_frame_image(&buf));
    }

    // Process camera registers and set camera exposure
    sensor->processRegisters((uint8_t *)buf.cur_camera_buf->addr, framed);
    set_camera_exposure(set_exposure_target(&buf, ae_xywh, 2, cc.stream_type != VISION_STREAM_DRIVER ? 2 : 4));

    // Send the message
    pm.send(cc.publish_name, msg);
    if (cc.stream_type == VISION_STREAM_ROAD && cnt % 100 == 3) {
      publish_thumbnail(&pm, &buf);  // this takes 10ms???
    }
  }
}

void camerad_thread() {
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  cl_context ctx = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));

  VisionIpcServer v("camerad", device_id, ctx);

  SpectraMaster m;
  SpectraMaster *s = &m;
  s->init();

  CameraState road_cam(s, ROAD_CAMERA_CONFIG);
  CameraState wide_road_cam(s, WIDE_ROAD_CAMERA_CONFIG);
  CameraState driver_cam(s, DRIVER_CAMERA_CONFIG);

  // open + init
  driver_cam.camera_open();
  LOGD("driver camera opened");
  road_cam.camera_open();
  LOGD("road camera opened");
  wide_road_cam.camera_open();
  LOGD("wide road camera opened");

  driver_cam.camera_init(&v, device_id, ctx);
  road_cam.camera_init(&v, device_id, ctx);
  wide_road_cam.camera_init(&v, device_id, ctx);

  v.start_listener();

  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  if (driver_cam.enabled) threads.emplace_back(&CameraState::run, &driver_cam);
  if (road_cam.enabled) threads.emplace_back(&CameraState::run, &road_cam);
  if (wide_road_cam.enabled) threads.emplace_back(&CameraState::run, &wide_road_cam);

  // start devices
  LOG("-- Starting devices");
  driver_cam.sensors_start();
  road_cam.sensors_start();
  wide_road_cam.sensors_start();

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
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl,
                 event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp/1e6, event_data->u.frame_msg.sof_status);
          do_exit = do_exit || event_data->u.frame_msg.frame_id > (1*20);
        }

        if (event_data->session_hdl == road_cam.session_handle) {
          road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == wide_road_cam.session_handle) {
          wide_road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == driver_cam.session_handle) {
          driver_cam.handle_camera_event(event_data);
        } else {
          LOGE("Unknown vidioc event source");
          assert(false);
        }
      } else {
        LOGE("unhandled event %d\n", ev.type);
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();
}
