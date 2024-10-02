#include <stdint.h>
#include <cassert>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"

#include "common/util.h"
#include "common/swaglog.h"
#include "system/camerad/cameras/spectra.h"

// For debugging:
// echo "4294967295" > /sys/module/cam_debug_util/parameters/debug_mdl

// ************** low level camera helpers ****************

int do_cam_control(int fd, int op_code, void *handle, int size) {
  struct cam_control camcontrol = {0};
  camcontrol.op_code = op_code;
  camcontrol.handle = (uint64_t)handle;
  if (size == 0) {
    camcontrol.size = 8;
    camcontrol.handle_type = CAM_HANDLE_MEM_HANDLE;
  } else {
    camcontrol.size = size;
    camcontrol.handle_type = CAM_HANDLE_USER_POINTER;
  }

  int ret = HANDLE_EINTR(ioctl(fd, VIDIOC_CAM_CONTROL, &camcontrol));
  if (ret == -1) {
    LOGE("VIDIOC_CAM_CONTROL error: op_code %d - errno %d", op_code, errno);
  }
  return ret;
}

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources) {
  struct cam_acquire_dev_cmd cmd = {
    .session_handle = session_handle,
    .handle_type = CAM_HANDLE_USER_POINTER,
    .num_resources = (uint32_t)(data ? num_resources : 0),
    .resource_hdl = (uint64_t)data,
  };
  int err = do_cam_control(fd, CAM_ACQUIRE_DEV, &cmd, sizeof(cmd));
  return err == 0 ? std::make_optional(cmd.dev_handle) : std::nullopt;
}

int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle) {
  struct cam_config_dev_cmd cmd = {
    .session_handle = session_handle,
    .dev_handle = dev_handle,
    .packet_handle = packet_handle,
  };
  return do_cam_control(fd, CAM_CONFIG_DEV, &cmd, sizeof(cmd));
}

int device_control(int fd, int op_code, int session_handle, int dev_handle) {
  // start stop and release are all the same
  struct cam_start_stop_dev_cmd cmd { .session_handle = session_handle, .dev_handle = dev_handle };
  return do_cam_control(fd, op_code, &cmd, sizeof(cmd));
}

void *alloc_w_mmu_hdl(int video0_fd, int len, uint32_t *handle, int align, int flags, int mmu_hdl, int mmu_hdl2) {
  struct cam_mem_mgr_alloc_cmd mem_mgr_alloc_cmd = {0};
  mem_mgr_alloc_cmd.len = len;
  mem_mgr_alloc_cmd.align = align;
  mem_mgr_alloc_cmd.flags = flags;
  mem_mgr_alloc_cmd.num_hdl = 0;
  if (mmu_hdl != 0) {
    mem_mgr_alloc_cmd.mmu_hdls[0] = mmu_hdl;
    mem_mgr_alloc_cmd.num_hdl++;
  }
  if (mmu_hdl2 != 0) {
    mem_mgr_alloc_cmd.mmu_hdls[1] = mmu_hdl2;
    mem_mgr_alloc_cmd.num_hdl++;
  }

  do_cam_control(video0_fd, CAM_REQ_MGR_ALLOC_BUF, &mem_mgr_alloc_cmd, sizeof(mem_mgr_alloc_cmd));
  *handle = mem_mgr_alloc_cmd.out.buf_handle;

  void *ptr = NULL;
  if (mem_mgr_alloc_cmd.out.fd > 0) {
    ptr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, mem_mgr_alloc_cmd.out.fd, 0);
    assert(ptr != MAP_FAILED);
  }

  // LOGD("allocated: %x %d %llx mapped %p", mem_mgr_alloc_cmd.out.buf_handle, mem_mgr_alloc_cmd.out.fd, mem_mgr_alloc_cmd.out.vaddr, ptr);

  return ptr;
}

void release(int video0_fd, uint32_t handle) {
  struct cam_mem_mgr_release_cmd mem_mgr_release_cmd = {0};
  mem_mgr_release_cmd.buf_handle = handle;

  int ret = do_cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

static cam_cmd_power *power_set_wait(cam_cmd_power *power, int16_t delay_ms) {
  cam_cmd_unconditional_wait *unconditional_wait = (cam_cmd_unconditional_wait *)((char *)power + (sizeof(struct cam_cmd_power) + (power->count - 1) * sizeof(struct cam_power_settings)));
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = delay_ms;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  return (struct cam_cmd_power *)(unconditional_wait + 1);
}

// *** MemoryManager ***

void *MemoryManager::alloc_buf(int size, uint32_t *handle) {
  lock.lock();
  void *ptr;
  if (!cached_allocations[size].empty()) {
    ptr = cached_allocations[size].front();
    cached_allocations[size].pop();
    *handle = handle_lookup[ptr];
  } else {
    ptr = alloc_w_mmu_hdl(video0_fd, size, handle);
    handle_lookup[ptr] = *handle;
    size_lookup[ptr] = size;
  }
  lock.unlock();
  memset(ptr, 0, size);
  return ptr;
}

void MemoryManager::free(void *ptr) {
  lock.lock();
  cached_allocations[size_lookup[ptr]].push(ptr);
  lock.unlock();
}

MemoryManager::~MemoryManager() {
  for (auto& x : cached_allocations) {
    while (!x.second.empty()) {
      void *ptr = x.second.front();
      x.second.pop();
      LOGD("freeing cached allocation %p with size %d", ptr, size_lookup[ptr]);
      munmap(ptr, size_lookup[ptr]);

      // release fd
      close(handle_lookup[ptr] >> 16);
      release(video0_fd, handle_lookup[ptr]);

      handle_lookup.erase(ptr);
      size_lookup.erase(ptr);
    }
  }
}

// *** SpectraMaster ***

void SpectraMaster::init() {
  LOG("-- Opening devices");
  // video0 is req_mgr, the target of many ioctls
  video0_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-soc:qcom_cam-req-mgr-video-index0", O_RDWR | O_NONBLOCK));
  assert(video0_fd >= 0);
  LOGD("opened video0");

  // video1 is cam_sync, the target of some ioctls
  cam_sync_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-cam_sync-video-index0", O_RDWR | O_NONBLOCK));
  assert(cam_sync_fd >= 0);
  LOGD("opened video1 (cam_sync)");

  // looks like there's only one of these
  isp_fd = open_v4l_by_name_and_index("cam-isp");
  assert(isp_fd >= 0);
  LOGD("opened isp");

  // query icp for MMU handles
  LOG("-- Query ICP for MMU handles");
  struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  int ret = do_cam_control(isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;

  // subscribe
  LOG("-- Subscribing");
  struct v4l2_event_subscription sub = {0};
  sub.type = V4L_EVENT_CAM_REQ_MGR_EVENT;
  sub.id = V4L_EVENT_CAM_REQ_MGR_SOF_BOOT_TS;
  ret = HANDLE_EINTR(ioctl(video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));
  LOGD("req mgr subscribe: %d", ret);
}

// *** SpectraCamera ***

SpectraCamera::SpectraCamera(SpectraMaster *master, const CameraConfig &config)
  : m(master),
    enabled(config.enabled) ,
    cc(config) {
  mm.init(m->video0_fd);
}

SpectraCamera::~SpectraCamera() {
  if (open) {
    camera_close();
  }
}

int SpectraCamera::clear_req_queue() {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = session_handle;
  req_mgr_flush_request.link_hdl = link_handle;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret = do_cam_control(m->video0_fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
  return ret;
}

void SpectraCamera::camera_open(VisionIpcServer *v, cl_device_id device_id, cl_context ctx) {
  if (!enabled) return;

  if (!openSensor()) {
    return;
  }

  open = true;
  configISP();
  configCSIPHY();
  linkDevices();

  LOGD("camera init %d", cc.camera_num);
  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, cc.stream_type);
  camera_map_bufs();
}

void SpectraCamera::enqueue_req_multi(uint64_t start, int n, bool dp) {
  for (uint64_t i = start; i < start + n; ++i) {
    request_ids[(i - 1) % FRAME_BUF_COUNT] = i;
    enqueue_buffer((i - 1) % FRAME_BUF_COUNT, dp);
  }
}

void SpectraCamera::sensors_start() {
  if (!enabled) return;
  LOGD("starting sensor %d", cc.camera_num);
  sensors_i2c(sensor->start_reg_array.data(), sensor->start_reg_array.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, sensor->data_word);
}

void SpectraCamera::sensors_poke(int request_id) {
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
    LOGE("** sensor %d FAILED poke, disabling", cc.camera_num);
    enabled = false;
    return;
  }
}

void SpectraCamera::sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word) {
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
    LOGE("** sensor %d FAILED i2c, disabling", cc.camera_num);
    enabled = false;
    return;
  }
}

int SpectraCamera::sensors_init() {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = CSLDeviceTypeImageSensor | CAM_SENSOR_PACKET_OPCODE_SENSOR_PROBE;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  auto i2c_info = mm.alloc<struct cam_cmd_i2c_info>(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  auto probe = (struct cam_cmd_probe *)(i2c_info.get() + 1);

  probe->camera_id = cc.camera_num;
  i2c_info->slave_addr = sensor->getSlaveAddress(cc.camera_num);
  // 0(I2C_STANDARD_MODE) = 100khz, 1(I2C_FAST_MODE) = 400khz
  //i2c_info->i2c_freq_mode = I2C_STANDARD_MODE;
  i2c_info->i2c_freq_mode = I2C_FAST_MODE;
  i2c_info->cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_INFO;

  probe->data_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->op_code = 3;   // don't care?
  probe->cmd_type = CAMERA_SENSOR_CMD_TYPE_PROBE;
  probe->reg_addr = sensor->probe_reg_addr;
  probe->expected_data = sensor->probe_expected_data;
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
  power->power_settings[0].config_val_low = sensor->mclk_frequency;
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

void SpectraCamera::config_ife(int io_mem_handle, int fence, int request_id, int buf0_idx) {
  /*
    Handles initial + per-frame IFE config.
    IFE = Image Front End
  */
  int size = sizeof(struct cam_packet) + sizeof(struct cam_cmd_buf_desc)*2;
  if (io_mem_handle != 0) {
    size += sizeof(struct cam_buf_io_cfg);
  }

  uint32_t cam_packet_handle = 0;
  auto pkt = mm.alloc<struct cam_packet>(size, &cam_packet_handle);

  if (io_mem_handle != 0) {
    pkt->header.op_code =  CSLDeviceTypeIFE | OpcodesIFEUpdate;  // 0xf000001
    pkt->header.request_id = request_id;
  } else {
    pkt->header.op_code = CSLDeviceTypeIFE | OpcodesIFEInitialConfig; // 0xf000000
    pkt->header.request_id = 1;
  }
  pkt->header.size = size;

  // *** kmd cmd buf ***
  {
    pkt->kmd_cmd_buf_index = 0;
    pkt->kmd_cmd_buf_offset = 0;
  }

  // *** cmd buf ***
  {
    struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;
    pkt->num_cmd_buf = 2;

    // *** first command ***
    // TODO: support MMU
    buf_desc[0].size = buf0_size;
    buf_desc[0].length = 0;
    buf_desc[0].type = CAM_CMD_BUF_DIRECT;
    buf_desc[0].meta_data = 3;
    buf_desc[0].mem_handle = buf0_handle;
    buf_desc[0].offset = ALIGNED_SIZE(buf0_size, buf0_alignment)*buf0_idx;

    // *** second command ***
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
  }

  // *** io config ***
  if (io_mem_handle != 0) {
    // configure output frame
    pkt->num_io_configs = 1;
    pkt->io_configs_offset = sizeof(struct cam_cmd_buf_desc)*pkt->num_cmd_buf;

    struct cam_buf_io_cfg *io_cfg = (struct cam_buf_io_cfg *)((char*)&pkt->payload + pkt->io_configs_offset);
    io_cfg[0].offsets[0] = 0;
    io_cfg[0].mem_handle[0] = io_mem_handle;

    io_cfg[0].planes[0] = (struct cam_plane_cfg){
      .width = sensor->frame_width,
      .height = sensor->frame_height + sensor->extra_height,
      .plane_stride = sensor->frame_stride,
      .slice_height = sensor->frame_height + sensor->extra_height,

      // these are for UBWC, we'll want that eventually
      .meta_stride = 0x0,
      .meta_size = 0x0,
      .meta_offset = 0x0,
      .packer_config = 0x0,
      .mode_config = 0x0,
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

  // *** patches ***
  {
    pkt->num_patches = 0;
    pkt->patch_offset = sizeof(struct cam_cmd_buf_desc)*pkt->num_cmd_buf + sizeof(struct cam_buf_io_cfg)*pkt->num_io_configs;
  }

  int ret = device_config(m->isp_fd, session_handle, isp_dev_handle, cam_packet_handle);
  assert(ret == 0);
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
  config_ife(buf_handle[i], sync_objs[i], request_id, i);
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

  LOGD("-- Probing sensor %d", cc.camera_num);

  auto init_sensor_lambda = [this](SensorInfo *s) {
    sensor.reset(s);
    return (sensors_init() == 0);
  };

  // Figure out which sensor we have
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
  if (!enabled) return;

  struct cam_isp_in_port_info in_port_info = {
    // ISP input to the CSID
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

    // ISP outputs
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

  // config IFE
  alloc_w_mmu_hdl(m->video0_fd, FRAME_BUF_COUNT*ALIGNED_SIZE(buf0_size, buf0_alignment), (uint32_t*)&buf0_handle, buf0_alignment,
                  CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE,
                  m->device_iommu, m->cdm_iommu);
  config_ife(0, 0, 1, 0);
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
}

void SpectraCamera::camera_close() {
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

void SpectraCamera::handle_camera_event(const cam_req_mgr_message *event_data) {
  if (!enabled) return;

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
