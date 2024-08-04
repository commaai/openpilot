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

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_req_mgr.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"
#include "common/swaglog.h"

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

// For debugging:
// echo "4294967295" > /sys/module/cam_debug_util/parameters/debug_mdl

ExitHandler do_exit;

CameraState::CameraState(MultiCameraState *multi_camera_state, const CameraConfig &config)
  : multi_cam_state(multi_camera_state),
    camera_num(config.camera_num),
    stream_type(config.stream_type),
    focal_len(config.focal_len),
    publish_name(config.publish_name),
    init_camera_state(config.init_camera_state),
    enabled(config.enabled) {
}

int CameraState::clear_req_queue() {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = session_handle;
  req_mgr_flush_request.link_hdl = link_handle;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
  return ret;
}

// ************** high level camera helpers ****************

void CameraState::sensors_start() {
  if (!enabled) return;
  LOGD("starting sensor %d", camera_num);
  sensors_i2c(ci->start_reg_array.data(), ci->start_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, ci->data_word);
}

void CameraState::sensors_poke(int request_id) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet);
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
  pkt->num_cmd_buf = 0;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = CAM_SENSOR_PACKET_OPCODE_SENSOR_NOP;
  pkt->header.request_id = request_id;

  int ret = device_config(sensor_fd, session_handle, sensor_dev_handle, cam_packet_handle);
  if (ret != 0) {
    LOGE("** sensor %d FAILED poke, disabling", camera_num);
    enabled = false;
    return;
  }
}

void CameraState::sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word) {
  // LOGD("sensors_i2c: %d", len);
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
  pkt->num_cmd_buf = 1;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = op_code;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct i2c_rdwr_header) + len*sizeof(struct i2c_random_wr_payload);
  buf_desc[0].type = CAM_CMD_BUF_I2C;

  auto i2c_random_wr = mm.alloc<struct cam_cmd_i2c_random_wr>(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  i2c_random_wr->header.count = len;
  i2c_random_wr->header.op_code = 1;
  i2c_random_wr->header.cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_RNDM_WR;
  i2c_random_wr->header.data_type = data_word ? CAMERA_SENSOR_I2C_TYPE_WORD : CAMERA_SENSOR_I2C_TYPE_BYTE;
  i2c_random_wr->header.addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  memcpy(i2c_random_wr->random_wr_payload, dat, len*sizeof(struct i2c_random_wr_payload));

  int ret = device_config(sensor_fd, session_handle, sensor_dev_handle, cam_packet_handle);
  if (ret != 0) {
    LOGE("** sensor %d FAILED i2c, disabling", camera_num);
    enabled = false;
    return;
  }
}

static cam_cmd_power *power_set_wait(cam_cmd_power *power, int16_t delay_ms) {
  cam_cmd_unconditional_wait *unconditional_wait = (cam_cmd_unconditional_wait *)((char *)power + (sizeof(struct cam_cmd_power) + (power->count - 1) * sizeof(struct cam_power_settings)));
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = delay_ms;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  return (struct cam_cmd_power *)(unconditional_wait + 1);
}

int CameraState::sensors_init() {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = 0x1000000 | CAM_SENSOR_PACKET_OPCODE_SENSOR_PROBE;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  auto i2c_info = mm.alloc<struct cam_cmd_i2c_info>(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  auto probe = (struct cam_cmd_probe *)(i2c_info.get() + 1);

  probe->camera_id = camera_num;
  i2c_info->slave_addr = ci->getSlaveAddress(camera_num);
  // 0(I2C_STANDARD_MODE) = 100khz, 1(I2C_FAST_MODE) = 400khz
  //i2c_info->i2c_freq_mode = I2C_STANDARD_MODE;
  i2c_info->i2c_freq_mode = I2C_FAST_MODE;
  i2c_info->cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_INFO;

  probe->data_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->op_code = 3;   // don't care?
  probe->cmd_type = CAMERA_SENSOR_CMD_TYPE_PROBE;
  probe->reg_addr = ci->probe_reg_addr;
  probe->expected_data = ci->probe_expected_data;
  probe->data_mask = 0;

  //buf_desc[1].size = buf_desc[1].length = 148;
  buf_desc[1].size = buf_desc[1].length = 196;
  buf_desc[1].type = CAM_CMD_BUF_I2C;
  auto power_settings = mm.alloc<struct cam_cmd_power>(buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle);

  // power on
  struct cam_cmd_power *power = power_settings.get();
  power->count = 4;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 3; // clock??
  power->power_settings[1].power_seq_type = 1; // analog
  power->power_settings[2].power_seq_type = 2; // digital
  power->power_settings[3].power_seq_type = 8; // reset low
  power = power_set_wait(power, 1);

  // set clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = ci->mclk_frequency;
  power = power_set_wait(power, 1);

  // reset high
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 1;
  // wait 650000 cycles @ 19.2 mhz = 33.8 ms
  power = power_set_wait(power, 34);

  // probe happens here

  // disable clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = 0;
  power = power_set_wait(power, 1);

  // reset high
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 1;
  power = power_set_wait(power, 1);

  // reset low
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 0;
  power = power_set_wait(power, 1);

  // power off
  power->count = 3;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 2;
  power->power_settings[1].power_seq_type = 1;
  power->power_settings[2].power_seq_type = 3;

  int ret = do_cam_control(sensor_fd, CAM_SENSOR_PROBE_CMD, (void *)(uintptr_t)cam_packet_handle, 0);
  LOGD("probing the sensor: %d", ret);
  return ret;
}

void CameraState::config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset) {
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
      .width = ci->frame_width,
      .height = ci->frame_height + ci->extra_height,
      .plane_stride = ci->frame_stride,
      .slice_height = ci->frame_height + ci->extra_height,
      .meta_stride = 0x0,  // YUV has meta(stride=0x400, size=0x5000)
      .meta_size = 0x0,
      .meta_offset = 0x0,
      .packer_config = 0x0,  // 0xb for YUV
      .mode_config = 0x0,    // 0x9ef for YUV
      .tile_config = 0x0,
      .h_init = 0x0,
      .v_init = 0x0,
    };
    io_cfg[0].format = ci->mipi_format;                    // CAM_FORMAT_UBWC_TP10 for YUV
    io_cfg[0].color_space = CAM_COLOR_SPACE_BASE;          // CAM_COLOR_SPACE_BT601_FULL for YUV
    io_cfg[0].color_pattern = 0x5;                         // 0x0 for YUV
    io_cfg[0].bpp = (ci->mipi_format == CAM_FORMAT_MIPI_RAW_10 ? 0xa : 0xc);  // bits per pixel
    io_cfg[0].resource_type = CAM_ISP_IFE_OUT_RES_RDI_0;   // CAM_ISP_IFE_OUT_RES_FULL for YUV
    io_cfg[0].fence = fence;
    io_cfg[0].direction = CAM_BUF_OUTPUT;
    io_cfg[0].subsample_pattern = 0x1;
    io_cfg[0].framedrop_pattern = 0x1;
  }

  int ret = device_config(multi_cam_state->isp_fd, session_handle, isp_dev_handle, cam_packet_handle);
  assert(ret == 0);
  if (ret != 0) {
    LOGE("isp config failed");
  }
}

void CameraState::enqueue_buffer(int i, bool dp) {
  int ret;
  uint64_t request_id = request_ids[i];

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
    LOGE("failed to create fence: %d %d", ret, sync_create.sync_obj);
  }
  sync_objs[i] = sync_create.sync_obj;

  // schedule request with camera request manager
  struct cam_req_mgr_sched_request req_mgr_sched_request = {0};
  req_mgr_sched_request.session_hdl = session_handle;
  req_mgr_sched_request.link_hdl = link_handle;
  req_mgr_sched_request.req_id = request_id;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_SCHED_REQ, &req_mgr_sched_request, sizeof(req_mgr_sched_request));
  if (ret != 0) {
    LOGE("failed to schedule cam mgr request: %d %lu", ret, request_id);
  }

  // poke sensor, must happen after schedule
  sensors_poke(request_id);

  // submit request to the ife
  config_isp(buf_handle[i], sync_objs[i], request_id, buf0_handle, 65632*(i+1));
}

void CameraState::enqueue_req_multi(uint64_t start, int n, bool dp) {
  for (uint64_t i = start; i < start + n; ++i) {
    request_ids[(i - 1) % FRAME_BUF_COUNT] = i;
    enqueue_buffer((i - 1) % FRAME_BUF_COUNT, dp);
  }
}

// ******************* camera *******************

void CameraState::set_exposure_rect() {
  // set areas for each camera, shouldn't be changed
  std::vector<std::pair<Rect, float>> ae_targets = {
    // (Rect, F)
    std::make_pair((Rect){96, 250, 1734, 524}, 567.0), // wide
    std::make_pair((Rect){96, 160, 1734, 986}, 2648.0), // road
    std::make_pair((Rect){96, 242, 1736, 906}, 567.0) // driver
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
  auto ae_target = ae_targets[camera_num];
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
  dc_gain_weight = ci->dc_gain_min_weight;
  gain_idx = ci->analog_gain_rec_idx;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = (1 + dc_gain_weight * (ci->dc_gain_factor-1) / ci->dc_gain_max_weight) * ci->sensor_analog_gains[gain_idx] * exposure_time;
}

void CameraState::camera_map_bufs() {
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // configure ISP to put the image in place
    struct cam_mem_mgr_map_cmd mem_mgr_map_cmd = {0};
    mem_mgr_map_cmd.mmu_hdls[0] = multi_cam_state->device_iommu;
    mem_mgr_map_cmd.num_hdl = 1;
    mem_mgr_map_cmd.flags = CAM_MEM_FLAG_HW_READ_WRITE;
    mem_mgr_map_cmd.fd = buf.camera_bufs[i].fd;
    int ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_MAP_BUF, &mem_mgr_map_cmd, sizeof(mem_mgr_map_cmd));
    LOGD("map buf req: (fd: %d) 0x%x %d", buf.camera_bufs[i].fd, mem_mgr_map_cmd.out.buf_handle, ret);
    buf_handle[i] = mem_mgr_map_cmd.out.buf_handle;
  }
  enqueue_req_multi(1, FRAME_BUF_COUNT, 0);
}

void CameraState::camera_init(VisionIpcServer * v, cl_device_id device_id, cl_context ctx) {
  if (!enabled) return;

  LOGD("camera init %d", camera_num);
  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, stream_type);
  camera_map_bufs();

  fl_pix = focal_len / ci->pixel_size_mm;
  set_exposure_rect();
}

void CameraState::camera_open() {
  if (!enabled) return;

  if (!openSensor()) {
    return;
  }

  configISP();
  configCSIPHY();
  linkDevices();
}

bool CameraState::openSensor() {
  sensor_fd = open_v4l_by_name_and_index("cam-sensor-driver", camera_num);
  assert(sensor_fd >= 0);
  LOGD("opened sensor for %d", camera_num);

  // init memorymanager for this camera
  mm.init(multi_cam_state->video0_fd);

  LOGD("-- Probing sensor %d", camera_num);

  auto init_sensor_lambda = [this](SensorInfo *sensor) {
    ci.reset(sensor);
    int ret = sensors_init();
    if (ret == 0) {
      sensor_set_parameters();
    }
    return ret == 0;
  };

  // Try different sensors one by one until it success.
  if (!init_sensor_lambda(new AR0231) &&
      !init_sensor_lambda(new OX03C10) &&
      !init_sensor_lambda(new OS04C10)) {
    LOGE("** sensor %d FAILED bringup, disabling", camera_num);
    enabled = false;
    return false;
  }
  LOGD("-- Probing sensor %d success", camera_num);

  // create session
  struct cam_req_mgr_session_info session_info = {};
  int ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_CREATE_SESSION, &session_info, sizeof(session_info));
  LOGD("get session: %d 0x%X", ret, session_info.session_hdl);
  session_handle = session_info.session_hdl;

  // access the sensor
  LOGD("-- Accessing sensor");
  auto sensor_dev_handle_ = device_acquire(sensor_fd, session_handle, nullptr);
  assert(sensor_dev_handle_);
  sensor_dev_handle = *sensor_dev_handle_;
  LOGD("acquire sensor dev");

  LOG("-- Configuring sensor");
  sensors_i2c(ci->init_reg_array.data(), ci->init_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, ci->data_word);
  return true;
}

void CameraState::configISP() {
  // NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
  // If you don't do this, the strobe GPIO is an output (even in reset it seems!)
  if (!enabled) return;

  struct cam_isp_in_port_info in_port_info = {
    .res_type = (uint32_t[]){CAM_ISP_IFE_IN_RES_PHY_0, CAM_ISP_IFE_IN_RES_PHY_1, CAM_ISP_IFE_IN_RES_PHY_2}[camera_num],

    .lane_type = CAM_ISP_LANE_TYPE_DPHY,
    .lane_num = 4,
    .lane_cfg = 0x3210,

    .vc = 0x0,
    .dt = ci->frame_data_type,
    .format = ci->mipi_format,

    .test_pattern = 0x2,  // 0x3?
    .usage_type = 0x0,

    .left_start = 0,
    .left_stop = ci->frame_width - 1,
    .left_width = ci->frame_width,

    .right_start = 0,
    .right_stop = ci->frame_width - 1,
    .right_width = ci->frame_width,

    .line_start = 0,
    .line_stop = ci->frame_height + ci->extra_height - 1,
    .height = ci->frame_height + ci->extra_height,

    .pixel_clk = 0x0,
    .batch_size = 0x0,
    .dsp_mode = CAM_ISP_DSP_MODE_NONE,
    .hbi_cnt = 0x0,
    .custom_csid = 0x0,

    .num_out_res = 0x1,
    .data[0] = (struct cam_isp_out_port_info){
      .res_type = CAM_ISP_IFE_OUT_RES_RDI_0,
      .format = ci->mipi_format,
      .width = ci->frame_width,
      .height = ci->frame_height + ci->extra_height,
      .comp_grp_id = 0x0, .split_point = 0x0, .secure_mode = 0x0,
    },
  };
  struct cam_isp_resource isp_resource = {
    .resource_id = CAM_ISP_RES_ID_PORT,
    .handle_type = CAM_HANDLE_USER_POINTER,
    .res_hdl = (uint64_t)&in_port_info,
    .length = sizeof(in_port_info),
  };

  auto isp_dev_handle_ = device_acquire(multi_cam_state->isp_fd, session_handle, &isp_resource);
  assert(isp_dev_handle_);
  isp_dev_handle = *isp_dev_handle_;
  LOGD("acquire isp dev");

  // config ISP
  alloc_w_mmu_hdl(multi_cam_state->video0_fd, 984480, (uint32_t*)&buf0_handle, 0x20, CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS |
                  CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, multi_cam_state->device_iommu, multi_cam_state->cdm_iommu);
  config_isp(0, 0, 1, buf0_handle, 0);
}

void CameraState::configCSIPHY() {
  csiphy_fd = open_v4l_by_name_and_index("cam-csiphy-driver", camera_num);
  assert(csiphy_fd >= 0);
  LOGD("opened csiphy for %d", camera_num);

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

void CameraState::linkDevices() {
  LOG("-- Link devices");
  struct cam_req_mgr_link_info req_mgr_link_info = {0};
  req_mgr_link_info.session_hdl = session_handle;
  req_mgr_link_info.num_devices = 2;
  req_mgr_link_info.dev_hdls[0] = isp_dev_handle;
  req_mgr_link_info.dev_hdls[1] = sensor_dev_handle;
  int ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_LINK, &req_mgr_link_info, sizeof(req_mgr_link_info));
  link_handle = req_mgr_link_info.link_hdl;
  LOGD("link: %d session: 0x%X isp: 0x%X sensors: 0x%X link: 0x%X", ret, session_handle, isp_dev_handle, sensor_dev_handle, link_handle);

  struct cam_req_mgr_link_control req_mgr_link_control = {0};
  req_mgr_link_control.ops = CAM_REQ_MGR_LINK_ACTIVATE;
  req_mgr_link_control.session_hdl = session_handle;
  req_mgr_link_control.num_links = 1;
  req_mgr_link_control.link_hdls[0] = link_handle;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
  LOGD("link control: %d", ret);

  ret = device_control(csiphy_fd, CAM_START_DEV, session_handle, csiphy_dev_handle);
  LOGD("start csiphy: %d", ret);
  ret = device_control(multi_cam_state->isp_fd, CAM_START_DEV, session_handle, isp_dev_handle);
  LOGD("start isp: %d", ret);
  assert(ret == 0);

  // TODO: this is unneeded, should we be doing the start i2c in a different way?
  //ret = device_control(sensor_fd, CAM_START_DEV, session_handle, sensor_dev_handle);
  //LOGD("start sensor: %d", ret);
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->driver_cam.camera_init(v, device_id, ctx);
  s->road_cam.camera_init(v, device_id, ctx);
  s->wide_road_cam.camera_init(v, device_id, ctx);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "wideRoadCameraState", "thumbnail"});
}

void cameras_open(MultiCameraState *s) {
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
  int ret = do_cam_control(s->isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
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

  s->driver_cam.camera_open();
  LOGD("driver camera opened");
  s->road_cam.camera_open();
  LOGD("road camera opened");
  s->wide_road_cam.camera_open();
  LOGD("wide road camera opened");
}

void CameraState::camera_close() {
  // stop devices
  LOG("-- Stop devices %d", camera_num);

  if (enabled) {
    // ret = device_control(sensor_fd, CAM_STOP_DEV, session_handle, sensor_dev_handle);
    // LOGD("stop sensor: %d", ret);
    int ret = device_control(multi_cam_state->isp_fd, CAM_STOP_DEV, session_handle, isp_dev_handle);
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
    ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
    LOGD("link control stop: %d", ret);

    // unlink
    LOG("-- Unlink");
    struct cam_req_mgr_unlink_info req_mgr_unlink_info = {0};
    req_mgr_unlink_info.session_hdl = session_handle;
    req_mgr_unlink_info.link_hdl = link_handle;
    ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_UNLINK, &req_mgr_unlink_info, sizeof(req_mgr_unlink_info));
    LOGD("unlink: %d", ret);

    // release devices
    LOGD("-- Release devices");
    ret = device_control(multi_cam_state->isp_fd, CAM_RELEASE_DEV, session_handle, isp_dev_handle);
    LOGD("release isp: %d", ret);
    ret = device_control(csiphy_fd, CAM_RELEASE_DEV, session_handle, csiphy_dev_handle);
    LOGD("release csiphy: %d", ret);

    for (int i = 0; i < FRAME_BUF_COUNT; i++) {
      release(multi_cam_state->video0_fd, buf_handle[i]);
    }
    LOGD("released buffers");
  }

  int ret = device_control(sensor_fd, CAM_RELEASE_DEV, session_handle, sensor_dev_handle);
  LOGD("release sensor: %d", ret);

  // destroyed session
  struct cam_req_mgr_session_info session_info = {.session_hdl = session_handle};
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_DESTROY_SESSION, &session_info, sizeof(session_info));
  LOGD("destroyed session %d: %d", camera_num, ret);
}

void cameras_close(MultiCameraState *s) {
  s->driver_cam.camera_close();
  s->road_cam.camera_close();
  s->wide_road_cam.camera_close();

  delete s->pm;
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
      LOGE("camera %d realign", camera_num);
      clear_req_queue();
      enqueue_req_multi(real_id + 1, FRAME_BUF_COUNT - 1, 0);
      skipped = true;
    } else if (main_id == frame_id_last + 1) {
      skipped = false;
    }

    // check for dropped requests
    if (real_id > request_id_last + 1) {
      LOGE("camera %d dropped requests %ld %ld", camera_num, real_id, request_id_last);
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
    meta_data.gain = analog_gain_frac * (1 + dc_gain_weight * (ci->dc_gain_factor-1) / ci->dc_gain_max_weight);
    meta_data.high_conversion_gain = dc_gain_enabled;
    meta_data.integ_lines = exposure_time;
    meta_data.measured_grey_fraction = measured_grey_fraction;
    meta_data.target_grey_fraction = target_grey_fraction;
    exp_lock.unlock();

    // dispatch
    enqueue_req_multi(real_id + FRAME_BUF_COUNT, 1, 1);
  } else { // not ready
    if (main_id > frame_id_last + 10) {
      LOGE("camera %d reset after half second of no response", camera_num);
      clear_req_queue();
      enqueue_req_multi(request_id_last + 1, FRAME_BUF_COUNT, 0);
      frame_id_last = main_id;
      skipped = true;
    }
  }
}

void CameraState::update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
  float score = ci->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
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
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + ci->target_grey_factor*cur_ev_) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, ci->min_ev, ci->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < ci->dc_gain_on_grey) {
    enable_dc_gain = true;
    dc_gain_weight = ci->dc_gain_min_weight;
  } else if (enable_dc_gain && target_grey > ci->dc_gain_off_grey) {
    enable_dc_gain = false;
    dc_gain_weight = ci->dc_gain_max_weight;
  }

  if (enable_dc_gain && dc_gain_weight < ci->dc_gain_max_weight) {dc_gain_weight += 1;}
  if (!enable_dc_gain && dc_gain_weight > ci->dc_gain_min_weight) {dc_gain_weight -= 1;}

  std::string gain_bytes, time_bytes;
  if (env_ctrl_exp_from_params) {
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
    for (int g = std::max((int)ci->analog_gain_min_idx, gain_idx - 1); g <= std::min((int)ci->analog_gain_max_idx, gain_idx + 1); g++) {
      float gain = ci->sensor_analog_gains[g] * (1 + dc_gain_weight * (ci->dc_gain_factor-1) / ci->dc_gain_max_weight);

      // Compute optimal time for given gain
      int t = std::clamp(int(std::round(desired_ev / gain)), ci->exposure_time_min, ci->exposure_time_max);

      // Only go below recommended gain when absolutely necessary to not overexpose
      if (g < ci->analog_gain_rec_idx && t > 20 && g < gain_idx) {
        continue;
      }

      update_exposure_score(desired_ev, t, g, gain);
    }
  }

  exp_lock.lock();

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = ci->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * (1 + dc_gain_weight * (ci->dc_gain_factor-1) / ci->dc_gain_max_weight);
  cur_ev[buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - buf.cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", camera_num, 1e-9 * nanos_since_boot(), 1e-9 * buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - buf.cur_frame_data.timestamp_sof));

  auto exp_reg_array = ci->getExposureRegisters(exposure_time, new_exp_g, dc_gain_enabled);
  sensors_i2c(exp_reg_array.data(), exp_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, ci->data_word);
}

void CameraState::run() {
  util::set_thread_name(publish_name);

  for (uint32_t cnt = 0; !do_exit; ++cnt) {
    // Acquire the buffer; continue if acquisition fails
    if (!buf.acquire()) continue;

    MessageBuilder msg;
    auto framed = (msg.initEvent().*init_camera_state)();
    fill_frame_data(framed, buf.cur_frame_data, this);

    // Log raw frames for road camera
    if (env_log_raw_frames && stream_type == VISION_STREAM_ROAD && cnt % 100 == 5) {  // no overlap with qlog decimation
      framed.setImage(get_raw_frame_image(&buf));
    }
    // Log frame id for road and wide road cameras
    if (stream_type != VISION_STREAM_DRIVER) {
      LOGT(buf.cur_frame_data.frame_id, "%s: Image set", publish_name);
    }

    // Process camera registers and set camera exposure
    ci->processRegisters(this, framed);
    set_camera_exposure(set_exposure_target(&buf, ae_xywh, 2, stream_type != VISION_STREAM_DRIVER ? 2 : 4));

    // Send the message
    multi_cam_state->pm->send(publish_name, msg);
    if (stream_type == VISION_STREAM_ROAD && cnt % 100 == 3) {
      publish_thumbnail(multi_cam_state->pm, &buf);  // this takes 10ms???
    }
  }
}

MultiCameraState::MultiCameraState()
  : driver_cam(this, DRIVER_CAMERA_CONFIG),
    road_cam(this, ROAD_CAMERA_CONFIG),
    wide_road_cam(this, WIDE_ROAD_CAMERA_CONFIG) {
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  if (s->driver_cam.enabled) threads.emplace_back(&CameraState::run, &s->driver_cam);
  if (s->road_cam.enabled) threads.emplace_back(&CameraState::run, &s->road_cam);
  if (s->wide_road_cam.enabled) threads.emplace_back(&CameraState::run, &s->wide_road_cam);

  // start devices
  LOG("-- Starting devices");
  s->driver_cam.sensors_start();
  s->road_cam.sensors_start();
  s->wide_road_cam.sensors_start();

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
        }

        // for debugging
        //do_exit = do_exit || event_data->u.frame_msg.frame_id > (30*20);

        if (event_data->session_hdl == s->road_cam.session_handle) {
          s->road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->wide_road_cam.session_handle) {
          s->wide_road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->driver_cam.session_handle) {
          s->driver_cam.handle_camera_event(event_data);
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

  cameras_close(s);
}
