#include "system/camerad/cameras/camera.h"

#include "system/camerad/cameras/camera_qcom2.h"

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

int AbstractCamera::open(MultiCameraState *multi_cam_state_, int camera_num_, bool enabled_) {
  multi_cam_state = multi_cam_state_;
  camera_num = camera_num_;
  enabled = enabled_;
  if (!enabled) return 0;

  sensor_fd = open_v4l_by_name_and_index("cam-sensor-driver", camera_num);
  assert(sensor_fd >= 0);
  LOGD("opened sensor for %d", camera_num);

  // init memorymanager for this camera
  mm.init(multi_cam_state->video0_fd);

  // probe the sensor
  LOGD("-- Probing sensor %d", camera_num);
  // camera.reset(new CameraAR0231());
  int ret = sensors_init();
  LOGD("-- Probing sensor %d done with %d", camera_num, ret);
  if (ret != 0) {
    LOGE("** sensor %d FAILED bringup, disabling", camera_num);
    enabled = false;
    return ret;
  }

  // create session
  struct cam_req_mgr_session_info session_info = {};
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_CREATE_SESSION, &session_info, sizeof(session_info));
  LOGD("get session: %d 0x%X", ret, session_info.session_hdl);
  session_handle = session_info.session_hdl;

  // access the sensor
  LOGD("-- Accessing sensor");
  auto sensor_dev_handle_ = device_acquire(sensor_fd, session_handle, nullptr);
  assert(sensor_dev_handle_);
  sensor_dev_handle = *sensor_dev_handle_;
  LOGD("acquire sensor dev");

  LOG("-- Configuring sensor");
  sensors_i2c(init_array, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
  printf("dt is %x\n", in_port_info_dt);

  // NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
  // If you don't do this, the strobe GPIO is an output (even in reset it seems!)
  if (!enabled) return 0;

  struct cam_isp_in_port_info in_port_info = {
      .res_type = (uint32_t[]){CAM_ISP_IFE_IN_RES_PHY_0, CAM_ISP_IFE_IN_RES_PHY_1, CAM_ISP_IFE_IN_RES_PHY_2}[camera_num],

      .lane_type = CAM_ISP_LANE_TYPE_DPHY,
      .lane_num = 4,
      .lane_cfg = 0x3210,

      .vc = 0x0,
      .dt = in_port_info_dt,
      .format = CAM_FORMAT_MIPI_RAW_12,

      .test_pattern = 0x2,  // 0x3?
      .usage_type = 0x0,

      .left_start = 0,
      .left_stop = ci.frame_width - 1,
      .left_width = ci.frame_width,

      .right_start = 0,
      .right_stop = ci.frame_width - 1,
      .right_width = ci.frame_width,

      .line_start = 0,
      .line_stop = ci.frame_height + ci.extra_height - 1,
      .height = ci.frame_height + ci.extra_height,

      .pixel_clk = 0x0,
      .batch_size = 0x0,
      .dsp_mode = CAM_ISP_DSP_MODE_NONE,
      .hbi_cnt = 0x0,
      .custom_csid = 0x0,

      .num_out_res = 0x1,
      .data[0] = (struct cam_isp_out_port_info){
          .res_type = CAM_ISP_IFE_OUT_RES_RDI_0,
          .format = CAM_FORMAT_MIPI_RAW_12,
          .width = ci.frame_width,
          .height = ci.frame_height + ci.extra_height,
          .comp_grp_id = 0x0,
          .split_point = 0x0,
          .secure_mode = 0x0,
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

  csiphy_fd = open_v4l_by_name_and_index("cam-csiphy-driver", camera_num);
  assert(csiphy_fd >= 0);
  LOGD("opened csiphy for %d", camera_num);

  struct cam_csiphy_acquire_dev_info csiphy_acquire_dev_info = {.combo_mode = 0};
  auto csiphy_dev_handle_ = device_acquire(csiphy_fd, session_handle, &csiphy_acquire_dev_info);
  assert(csiphy_dev_handle_);
  csiphy_dev_handle = *csiphy_dev_handle_;
  LOGD("acquire csiphy dev");

  // config ISP
  alloc_w_mmu_hdl(multi_cam_state->video0_fd, 984480, (uint32_t *)&buf0_handle, 0x20, CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, multi_cam_state->device_iommu, multi_cam_state->cdm_iommu);
  config_isp(0, 0, 1, buf0_handle, 0);

  // config csiphy
  LOG("-- Config CSI PHY");
  {
    uint32_t cam_packet_handle = 0;
    int size = sizeof(struct cam_packet) + sizeof(struct cam_cmd_buf_desc) * 1;
    struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
    pkt->num_cmd_buf = 1;
    pkt->kmd_cmd_buf_index = -1;
    pkt->header.size = size;
    struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

    buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_csiphy_info);
    buf_desc[0].type = CAM_CMD_BUF_GENERIC;

    struct cam_csiphy_info *csiphy_info = (struct cam_csiphy_info *)mm.alloc(buf_desc[0].size, (uint32_t *)&buf_desc[0].mem_handle);
    csiphy_info->lane_mask = 0x1f;
    csiphy_info->lane_assign = 0x3210;  // skip clk. How is this 16 bit for 5 channels??
    csiphy_info->csiphy_3phase = 0x0;   // no 3 phase, only 2 conductors per lane
    csiphy_info->combo_mode = 0x0;
    csiphy_info->lane_cnt = 0x4;
    csiphy_info->secure_mode = 0x0;
    csiphy_info->settle_time = MIPI_SETTLE_CNT * 200000000ULL;
    csiphy_info->data_rate = 48000000;  // Calculated by camera_freqs.py

    int ret_ = device_config(csiphy_fd, session_handle, csiphy_dev_handle, cam_packet_handle);
    assert(ret_ == 0);

    mm.free(csiphy_info);
    mm.free(pkt);
  }

  // link devices
  LOG("-- Link devices");
  struct cam_req_mgr_link_info req_mgr_link_info = {0};
  req_mgr_link_info.session_hdl = session_handle;
  req_mgr_link_info.num_devices = 2;
  req_mgr_link_info.dev_hdls[0] = isp_dev_handle;
  req_mgr_link_info.dev_hdls[1] = sensor_dev_handle;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_LINK, &req_mgr_link_info, sizeof(req_mgr_link_info));
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

  // TODO: this is unneeded, should we be doing the start i2c in a different way?
  // ret = device_control(sensor_fd, CAM_START_DEV, session_handle, sensor_dev_handle);
  // LOGD("start sensor: %d", ret);
  return 0;
}

void AbstractCamera::close() {
  int ret;
  // stop devices
  LOG("-- Stop devices %d", camera_num);

  if (enabled) {
    // ret = device_control(sensor_fd, CAM_STOP_DEV, session_handle, sensor_dev_handle);
    // LOGD("stop sensor: %d", ret);
    ret = device_control(multi_cam_state->isp_fd, CAM_STOP_DEV, session_handle, isp_dev_handle);
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

    // for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    //   release(multi_cam_state->video0_fd, buf_handle[i]);
    // }
    // LOGD("released buffers");
  }

  ret = device_control(sensor_fd, CAM_RELEASE_DEV, session_handle, sensor_dev_handle);
  LOGD("release sensor: %d", ret);

  // destroyed session
  struct cam_req_mgr_session_info session_info = {.session_hdl = session_handle};
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_DESTROY_SESSION, &session_info, sizeof(session_info));
  LOGD("destroyed session %d: %d", camera_num, ret);
}

void AbstractCamera::sensors_i2c(const std::vector<i2c_random_wr_payload> &dat, int op_code) {
  // LOGD("sensors_i2c: %d", len);
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet) + sizeof(struct cam_cmd_buf_desc) * 1;
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
  pkt->num_cmd_buf = 1;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = op_code;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct i2c_rdwr_header) + dat.size() * sizeof(struct i2c_random_wr_payload);
  buf_desc[0].type = CAM_CMD_BUF_I2C;

  struct cam_cmd_i2c_random_wr *i2c_random_wr = (struct cam_cmd_i2c_random_wr *)mm.alloc(buf_desc[0].size, (uint32_t *)&buf_desc[0].mem_handle);
  i2c_random_wr->header.count = dat.size();
  i2c_random_wr->header.op_code = 1;
  i2c_random_wr->header.cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_RNDM_WR;
  i2c_random_wr->header.data_type = i2c_type;
  i2c_random_wr->header.addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  memcpy(i2c_random_wr->random_wr_payload, dat.data(), dat.size() * sizeof(struct i2c_random_wr_payload));

  int ret = device_config(sensor_fd, session_handle, sensor_dev_handle, cam_packet_handle);
  if (ret != 0) {
    LOGE("** sensor %d FAILED i2c, disabling", camera_num);
    enabled = false;
    return;
  }

  mm.free(i2c_random_wr);
  mm.free(pkt);
}

static cam_cmd_power *power_set_wait(cam_cmd_power *power, int16_t delay_ms) {
  cam_cmd_unconditional_wait *unconditional_wait = (cam_cmd_unconditional_wait *)((char *)power + (sizeof(struct cam_cmd_power) + (power->count - 1) * sizeof(struct cam_power_settings)));
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = delay_ms;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  return (struct cam_cmd_power *)(unconditional_wait + 1);
};

int AbstractCamera::sensors_init() {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet) + sizeof(struct cam_cmd_buf_desc) * 2;
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = 0x1000000 | CAM_SENSOR_PACKET_OPCODE_SENSOR_PROBE;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  struct cam_cmd_i2c_info *i2c_info = (struct cam_cmd_i2c_info *)mm.alloc(buf_desc[0].size, (uint32_t *)&buf_desc[0].mem_handle);
  auto probe = (struct cam_cmd_probe *)(i2c_info + 1);

  probe->camera_id = camera_num;
  i2c_info->slave_addr = getSlaveAddress(camera_num);

  // 0(I2C_STANDARD_MODE) = 100khz, 1(I2C_FAST_MODE) = 400khz
  // i2c_info->i2c_freq_mode = I2C_STANDARD_MODE;
  i2c_info->i2c_freq_mode = I2C_FAST_MODE;
  i2c_info->cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_INFO;

  probe->data_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->op_code = 3;  // don't care?
  probe->cmd_type = CAMERA_SENSOR_CMD_TYPE_PROBE;
  probe->reg_addr = reg_addr;
  probe->expected_data = expected_data;
  probe->data_mask = 0;

  // buf_desc[1].size = buf_desc[1].length = 148;
  buf_desc[1].size = buf_desc[1].length = 196;
  buf_desc[1].type = CAM_CMD_BUF_I2C;
  struct cam_cmd_power *power_settings = (struct cam_cmd_power *)mm.alloc(buf_desc[1].size, (uint32_t *)&buf_desc[1].mem_handle);
  memset(power_settings, 0, buf_desc[1].size);

  // power on
  struct cam_cmd_power *power = power_settings;
  power->count = 4;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 3;  // clock??
  power->power_settings[1].power_seq_type = 1;  // analog
  power->power_settings[2].power_seq_type = 2;  // digital
  power->power_settings[3].power_seq_type = 8;  // reset low
  power = power_set_wait(power, 1);

  // set clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = config_val_low;
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

  mm.free(i2c_info);
  mm.free(power_settings);
  mm.free(pkt);

  return ret;
}

void AbstractCamera::config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet) + sizeof(struct cam_cmd_buf_desc) * 2;
  if (io_mem_handle != 0) {
    size += sizeof(struct cam_buf_io_cfg);
  }
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = 0;
  // YUV has kmd_cmd_buf_offset = 1780
  // I guess this is the ISP command
  // YUV also has patch_offset = 0x1030 and num_patches = 10

  if (io_mem_handle != 0) {
    pkt->io_configs_offset = sizeof(struct cam_cmd_buf_desc) * pkt->num_cmd_buf;
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
  struct cam_buf_io_cfg *io_cfg = (struct cam_buf_io_cfg *)((char *)&pkt->payload + pkt->io_configs_offset);

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
          .resource_type = CAM_ISP_IFE_OUT_RES_RDI_0,  // CAM_ISP_IFE_OUT_RES_FULL for YUV
          .subsample_pattern = 1,
          .subsample_period = 0,
          .framedrop_pattern = 1,
          .framedrop_period = 0,
      }};

  tmp.type_1 = CAM_ISP_GENERIC_BLOB_TYPE_CLOCK_CONFIG;
  tmp.type_1 |= (sizeof(cam_isp_clock_config) + sizeof(tmp.extra_rdi_hz)) << 8;
  static_assert((sizeof(cam_isp_clock_config) + sizeof(tmp.extra_rdi_hz)) == 0x38);
  tmp.clock = {
      .usage_type = 1,  // dual mode
      .num_rdi = 4,
      .left_pix_hz = 404000000,
      .right_pix_hz = 404000000,
      .rdi_hz[0] = 404000000,
  };

  tmp.type_2 = CAM_ISP_GENERIC_BLOB_TYPE_BW_CONFIG;
  tmp.type_2 |= (sizeof(cam_isp_bw_config) + sizeof(tmp.extra_rdi_vote)) << 8;
  static_assert((sizeof(cam_isp_bw_config) + sizeof(tmp.extra_rdi_vote)) == 0xe0);
  tmp.bw = {
      .usage_type = 1,  // dual mode
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
  uint32_t *buf2 = (uint32_t *)mm.alloc(buf_desc[1].size, (uint32_t *)&buf_desc[1].mem_handle);
  memcpy(buf2, &tmp, sizeof(tmp));

  if (io_mem_handle != 0) {
    io_cfg[0].mem_handle[0] = io_mem_handle;
    io_cfg[0].planes[0] = (struct cam_plane_cfg){
        .width = ci.frame_width,
        .height = ci.frame_height + ci.extra_height,
        .plane_stride = ci.frame_stride,
        .slice_height = ci.frame_height + ci.extra_height,
        .meta_stride = 0x0,  // YUV has meta(stride=0x400, size=0x5000)
        .meta_size = 0x0,
        .meta_offset = 0x0,
        .packer_config = 0x0,  // 0xb for YUV
        .mode_config = 0x0,    // 0x9ef for YUV
        .tile_config = 0x0,
        .h_init = 0x0,
        .v_init = 0x0,
    };
    io_cfg[0].format = CAM_FORMAT_MIPI_RAW_12;     // CAM_FORMAT_UBWC_TP10 for YUV
    io_cfg[0].color_space = CAM_COLOR_SPACE_BASE;  // CAM_COLOR_SPACE_BT601_FULL for YUV
    io_cfg[0].color_pattern = 0x5;                 // 0x0 for YUV
    io_cfg[0].bpp = 0xc;
    io_cfg[0].resource_type = CAM_ISP_IFE_OUT_RES_RDI_0;  // CAM_ISP_IFE_OUT_RES_FULL for YUV
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

  mm.free(buf2);
  mm.free(pkt);
}

void AbstractCamera::sensors_poke(int request_id) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet);
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
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

  mm.free(pkt);
}

void AbstractCamera::sensors_start() {
  if (!enabled) return;
  LOGD("starting sensor %d", camera_num);
  sensors_i2c(start_reg_array, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
}
