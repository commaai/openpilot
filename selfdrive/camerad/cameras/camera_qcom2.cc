#include "selfdrive/camerad/cameras/camera_qcom2.h"

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

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_sensor.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"
#include "common/swaglog.h"
#include "selfdrive/camerad/cameras/sensor2_i2c.h"

// For debugging:
// echo "4294967295" > /sys/module/cam_debug_util/parameters/debug_mdl

extern ExitHandler do_exit;

const size_t FRAME_WIDTH = 1928;
const size_t FRAME_HEIGHT = 1208;
const size_t FRAME_STRIDE = 2896;  // for 12 bit output. 1928 * 12 / 8 + 4 (alignment)

const size_t AR0231_REGISTERS_HEIGHT = 2;
const size_t AR0231_STATS_HEIGHT = 2;

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_AR0231] = {
    .frame_width = FRAME_WIDTH,
    .frame_height = FRAME_HEIGHT,
    .frame_stride = FRAME_STRIDE,
    .extra_height = AR0231_REGISTERS_HEIGHT + AR0231_STATS_HEIGHT,

    .registers_offset = 0,
    .frame_offset = AR0231_REGISTERS_HEIGHT,
    .stats_offset = AR0231_REGISTERS_HEIGHT + FRAME_HEIGHT,

    .bayer = true,
    .bayer_flip = 1,
    .hdr = false,
  },
  [CAMERA_ID_IMX390] = {
    .frame_width = FRAME_WIDTH,
    .frame_height = FRAME_HEIGHT,
    .frame_stride = FRAME_STRIDE,

    .bayer = true,
    .bayer_flip = 1,
    .hdr = false,
  },
};

const float DC_GAIN = 2.5;
const float sensor_analog_gains[] = {
  1.0/8.0, 2.0/8.0, 2.0/7.0, 3.0/7.0, // 0, 1, 2, 3
  3.0/6.0, 4.0/6.0, 4.0/5.0, 5.0/5.0, // 4, 5, 6, 7
  5.0/4.0, 6.0/4.0, 6.0/3.0, 7.0/3.0, // 8, 9, 10, 11
  7.0/2.0, 8.0/2.0, 8.0/1.0};         // 12, 13, 14, 15 = bypass

const int ANALOG_GAIN_MIN_IDX = 0x1; // 0.25x
const int ANALOG_GAIN_REC_IDX = 0x6; // 0.8x
const int ANALOG_GAIN_MAX_IDX = 0xD; // 4.0x

const int EXPOSURE_TIME_MIN = 2; // with HDR, fastest ss
const int EXPOSURE_TIME_MAX = 0x0855; // with HDR, slowest ss, 40ms

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

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources=1) {
  struct cam_acquire_dev_cmd cmd = {
      .session_handle = session_handle,
      .handle_type = CAM_HANDLE_USER_POINTER,
      .num_resources = (uint32_t)(data ? num_resources : 0),
      .resource_hdl = (uint64_t)data,
  };
  int err = do_cam_control(fd, CAM_ACQUIRE_DEV, &cmd, sizeof(cmd));
  return err == 0 ? std::make_optional(cmd.dev_handle) : std::nullopt;
};

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

void *alloc_w_mmu_hdl(int video0_fd, int len, uint32_t *handle, int align = 8, int flags = CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE,
                      int mmu_hdl = 0, int mmu_hdl2 = 0) {
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
  int ret;
  struct cam_mem_mgr_release_cmd mem_mgr_release_cmd = {0};
  mem_mgr_release_cmd.buf_handle = handle;

  ret = do_cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

void release_fd(int video0_fd, uint32_t handle) {
  // handle to fd
  close(handle>>16);
  release(video0_fd, handle);
}

void *MemoryManager::alloc(int size, uint32_t *handle) {
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
      release_fd(video0_fd, handle_lookup[ptr]);
      handle_lookup.erase(ptr);
      size_lookup.erase(ptr);
    }
  }
}

int CameraState::clear_req_queue() {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = session_handle;
  req_mgr_flush_request.link_hdl = link_handle;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
  return ret;
}

// ************** high level camera helpers ****************

void CameraState::sensors_start() {
  if (!enabled) return;
  LOGD("starting sensor %d", camera_num);
  if (camera_id == CAMERA_ID_AR0231) {
    sensors_i2c(start_reg_array_ar0231, std::size(start_reg_array_ar0231), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, true);
  } else if (camera_id == CAMERA_ID_IMX390) {
    sensors_i2c(start_reg_array_imx390, std::size(start_reg_array_imx390), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, false);
  } else {
    assert(false);
  }
}

void CameraState::sensors_poke(int request_id) {
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

void CameraState::sensors_i2c(struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word) {
  // LOGD("sensors_i2c: %d", len);
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
  pkt->num_cmd_buf = 1;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = op_code;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct i2c_rdwr_header) + len*sizeof(struct i2c_random_wr_payload);
  buf_desc[0].type = CAM_CMD_BUF_I2C;

  struct cam_cmd_i2c_random_wr *i2c_random_wr = (struct cam_cmd_i2c_random_wr *)mm.alloc(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
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

int CameraState::sensors_init() {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = 0x1000000 | CAM_SENSOR_PACKET_OPCODE_SENSOR_PROBE;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  struct cam_cmd_i2c_info *i2c_info = (struct cam_cmd_i2c_info *)mm.alloc(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  auto probe = (struct cam_cmd_probe *)(i2c_info + 1);

  probe->camera_id = camera_num;
  switch (camera_num) {
    case 0:
      // port 0
      i2c_info->slave_addr = (camera_id == CAMERA_ID_AR0231) ? 0x20 : 0x34;
      break;
    case 1:
      // port 1
      i2c_info->slave_addr = (camera_id == CAMERA_ID_AR0231) ? 0x30 : 0x36;
      break;
    case 2:
      // port 2
      i2c_info->slave_addr = (camera_id == CAMERA_ID_AR0231) ? 0x20 : 0x34;
      break;
  }

  // 0(I2C_STANDARD_MODE) = 100khz, 1(I2C_FAST_MODE) = 400khz
  //i2c_info->i2c_freq_mode = I2C_STANDARD_MODE;
  i2c_info->i2c_freq_mode = I2C_FAST_MODE;
  i2c_info->cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_INFO;

  probe->data_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  probe->op_code = 3;   // don't care?
  probe->cmd_type = CAMERA_SENSOR_CMD_TYPE_PROBE;
  if (camera_id == CAMERA_ID_AR0231) {
    probe->reg_addr = 0x3000;
    probe->expected_data = 0x354;
  } else if (camera_id == CAMERA_ID_IMX390) {
    probe->reg_addr = 0x330;
    probe->expected_data = 0x1538;
  } else {
    assert(false);
  }
  probe->data_mask = 0;

  //buf_desc[1].size = buf_desc[1].length = 148;
  buf_desc[1].size = buf_desc[1].length = 196;
  buf_desc[1].type = CAM_CMD_BUF_I2C;
  struct cam_cmd_power *power_settings = (struct cam_cmd_power *)mm.alloc(buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle);
  memset(power_settings, 0, buf_desc[1].size);

  // power on
  struct cam_cmd_power *power = power_settings;
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
  power->power_settings[0].config_val_low = (camera_id == CAMERA_ID_AR0231) ? 19200000 : 24000000; //Hz
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

  LOGD("probing the sensor");
  int ret = do_cam_control(sensor_fd, CAM_SENSOR_PROBE_CMD, (void *)(uintptr_t)cam_packet_handle, 0);

  mm.free(i2c_info);
  mm.free(power_settings);
  mm.free(pkt);

  return ret;
}

void CameraState::config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
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
  uint32_t *buf2 = (uint32_t *)mm.alloc(buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle);
  memcpy(buf2, &tmp, sizeof(tmp));

  if (io_mem_handle != 0) {
    io_cfg[0].mem_handle[0] = io_mem_handle;
		io_cfg[0].planes[0] = (struct cam_plane_cfg){
		 .width = ci.frame_width,
		 .height = ci.frame_height + ci.extra_height,
		 .plane_stride = ci.frame_stride,
		 .slice_height = ci.frame_height + ci.extra_height,
		 .meta_stride = 0x0,    // YUV has meta(stride=0x400, size=0x5000)
		 .meta_size = 0x0,
		 .meta_offset = 0x0,
		 .packer_config = 0x0,  // 0xb for YUV
		 .mode_config = 0x0,    // 0x9ef for YUV
		 .tile_config = 0x0,
		 .h_init = 0x0,
		 .v_init = 0x0,
		};
    io_cfg[0].format = CAM_FORMAT_MIPI_RAW_12;             // CAM_FORMAT_UBWC_TP10 for YUV
    io_cfg[0].color_space = CAM_COLOR_SPACE_BASE;          // CAM_COLOR_SPACE_BT601_FULL for YUV
    io_cfg[0].color_pattern = 0x5;                         // 0x0 for YUV
    io_cfg[0].bpp = 0xc;
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

  mm.free(buf2);
  mm.free(pkt);
}

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
  req_mgr_sched_request.session_hdl = session_handle;
  req_mgr_sched_request.link_hdl = link_handle;
  req_mgr_sched_request.req_id = request_id;
  ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_SCHED_REQ, &req_mgr_sched_request, sizeof(req_mgr_sched_request));
  if (ret != 0) {
    LOGE("failed to schedule cam mgr request: %d %d", ret, request_id);
  }

  // poke sensor, must happen after schedule
  sensors_poke(request_id);

  // submit request to the ife
  config_isp(buf_handle[i], sync_objs[i], request_id, buf0_handle, 65632*(i+1));
}

void CameraState::enqueue_req_multi(int start, int n, bool dp) {
  for (int i=start;i<start+n;++i) {
    request_ids[(i - 1) % FRAME_BUF_COUNT] = i;
    enqueue_buffer((i - 1) % FRAME_BUF_COUNT, dp);
  }
}

// ******************* camera *******************

void CameraState::camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer * v, int camera_id_, int camera_num_, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type, bool enabled_) {
  multi_cam_state = multi_cam_state_;
  camera_id = camera_id_;
  camera_num = camera_num_;
  enabled = enabled_;
  if (!enabled) return;

  LOGD("camera init %d", camera_num);
  assert(camera_id < std::size(cameras_supported));
  ci = cameras_supported[camera_id];
  assert(ci.frame_width != 0);

  request_id_last = 0;
  skipped = true;

  min_ev = EXPOSURE_TIME_MIN * sensor_analog_gains[ANALOG_GAIN_MIN_IDX];
  max_ev = EXPOSURE_TIME_MAX * sensor_analog_gains[ANALOG_GAIN_MAX_IDX] * DC_GAIN;
  target_grey_fraction = 0.3;

  dc_gain_enabled = false;
  gain_idx = ANALOG_GAIN_REC_IDX;
  exposure_time = 5;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = (dc_gain_enabled ? DC_GAIN : 1) * sensor_analog_gains[gain_idx] * exposure_time;

  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, rgb_type, yuv_type);
}

void CameraState::camera_open() {
  int ret;
  sensor_fd = open_v4l_by_name_and_index("cam-sensor-driver", camera_num);
  assert(sensor_fd >= 0);
  LOGD("opened sensor for %d", camera_num);

  // init memorymanager for this camera
  mm.init(multi_cam_state->video0_fd);

  // probe the sensor
  LOGD("-- Probing sensor %d", camera_num);
  ret = sensors_init();
  if (ret != 0) {
    LOGD("AR0231 init failed, trying IMX390");
    camera_id = CAMERA_ID_IMX390;
    ret = sensors_init();
  }
  LOGD("-- Probing sensor %d done with %d", camera_num, ret);
  if (ret != 0) {
    LOGE("** sensor %d FAILED bringup, disabling", camera_num);
    enabled = false;
    return;
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
  if (camera_id == CAMERA_ID_AR0231) {
    sensors_i2c(init_array_ar0231, std::size(init_array_ar0231), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, true);
  } else if (camera_id == CAMERA_ID_IMX390) {
    sensors_i2c(init_array_imx390, std::size(init_array_imx390), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, false);
  } else {
    assert(false);
  }

  // NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
  // If you don't do this, the strobe GPIO is an output (even in reset it seems!)
  if (!enabled) return;

  struct cam_isp_in_port_info in_port_info = {
      .res_type = (uint32_t[]){CAM_ISP_IFE_IN_RES_PHY_0, CAM_ISP_IFE_IN_RES_PHY_1, CAM_ISP_IFE_IN_RES_PHY_2}[camera_num],

      .lane_type = CAM_ISP_LANE_TYPE_DPHY,
      .lane_num = 4,
      .lane_cfg = 0x3210,

      .vc = 0x0,
      .dt = 0x12, // Changing stats to 0x2C doesn't work, so change pixels to 0x12 instead
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

  csiphy_fd = open_v4l_by_name_and_index("cam-csiphy-driver", camera_num);
  assert(csiphy_fd >= 0);
  LOGD("opened csiphy for %d", camera_num);

  struct cam_csiphy_acquire_dev_info csiphy_acquire_dev_info = {.combo_mode = 0};
  auto csiphy_dev_handle_ = device_acquire(csiphy_fd, session_handle, &csiphy_acquire_dev_info);
  assert(csiphy_dev_handle_);
  csiphy_dev_handle = *csiphy_dev_handle_;
  LOGD("acquire csiphy dev");

  // config ISP
  alloc_w_mmu_hdl(multi_cam_state->video0_fd, 984480, (uint32_t*)&buf0_handle, 0x20, CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, multi_cam_state->device_iommu, multi_cam_state->cdm_iommu);
  config_isp(0, 0, 1, buf0_handle, 0);

  // config csiphy
  LOG("-- Config CSI PHY");
  {
    uint32_t cam_packet_handle = 0;
    int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
    struct cam_packet *pkt = (struct cam_packet *)mm.alloc(size, &cam_packet_handle);
    pkt->num_cmd_buf = 1;
    pkt->kmd_cmd_buf_index = -1;
    pkt->header.size = size;
    struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

    buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_csiphy_info);
    buf_desc[0].type = CAM_CMD_BUF_GENERIC;

    struct cam_csiphy_info *csiphy_info = (struct cam_csiphy_info *)mm.alloc(buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
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

  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // configure ISP to put the image in place
    struct cam_mem_mgr_map_cmd mem_mgr_map_cmd = {0};
    mem_mgr_map_cmd.mmu_hdls[0] = multi_cam_state->device_iommu;
    mem_mgr_map_cmd.num_hdl = 1;
    mem_mgr_map_cmd.flags = CAM_MEM_FLAG_HW_READ_WRITE;
    mem_mgr_map_cmd.fd = buf.camera_bufs[i].fd;
    ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_MAP_BUF, &mem_mgr_map_cmd, sizeof(mem_mgr_map_cmd));
    LOGD("map buf req: (fd: %d) 0x%x %d", buf.camera_bufs[i].fd, mem_mgr_map_cmd.out.buf_handle, ret);
    buf_handle[i] = mem_mgr_map_cmd.out.buf_handle;
  }

  // TODO: this is unneeded, should we be doing the start i2c in a different way?
  //ret = device_control(sensor_fd, CAM_START_DEV, session_handle, sensor_dev_handle);
  //LOGD("start sensor: %d", ret);

  enqueue_req_multi(1, FRAME_BUF_COUNT, 0);
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->driver_cam.camera_init(s, v, CAMERA_ID_AR0231, 2, 20, device_id, ctx, VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER, !env_disable_driver);
  s->road_cam.camera_init(s, v, CAMERA_ID_AR0231, 1, 20, device_id, ctx, VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD, !env_disable_road);
  s->wide_road_cam.camera_init(s, v, CAMERA_ID_AR0231, 0, 20, device_id, ctx, VISION_STREAM_RGB_WIDE_ROAD, VISION_STREAM_WIDE_ROAD, !env_disable_wide_road);

  s->sm = new SubMaster({"driverStateV2"});
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
  static struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  static struct cam_query_cap_cmd query_cap_cmd = {0};
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
  static struct v4l2_event_subscription sub = {0};
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
    static struct cam_req_mgr_link_control req_mgr_link_control = {0};
    req_mgr_link_control.ops = CAM_REQ_MGR_LINK_DEACTIVATE;
    req_mgr_link_control.session_hdl = session_handle;
    req_mgr_link_control.num_links = 1;
    req_mgr_link_control.link_hdls[0] = link_handle;
    ret = do_cam_control(multi_cam_state->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
    LOGD("link control stop: %d", ret);

    // unlink
    LOG("-- Unlink");
    static struct cam_req_mgr_unlink_info req_mgr_unlink_info = {0};
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

  ret = device_control(sensor_fd, CAM_RELEASE_DEV, session_handle, sensor_dev_handle);
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

  delete s->sm;
  delete s->pm;
}

std::map<uint16_t, std::pair<int, int>> CameraState::ar0231_build_register_lut(uint8_t *data) {
  // This function builds a lookup table from register address, to a pair of indices in the
  // buffer where to read this address. The buffer contains padding bytes,
  // as well as markers to indicate the type of the next byte.
  //
  // 0xAA is used to indicate the MSB of the address, 0xA5 for the LSB of the address.
  // Every byte of data (MSB and LSB) is preceded by 0x5A. Specifying an address is optional
  // for contigous ranges. See page 27-29 of the AR0231 Developer guide for more information.

  int max_i[] = {1828 / 2 * 3, 1500 / 2 * 3};
  auto get_next_idx = [](int cur_idx) {
    return (cur_idx % 3 == 1) ? cur_idx + 2 : cur_idx + 1; // Every third byte is padding
  };

  std::map<uint16_t, std::pair<int, int>> registers;
  for (int register_row = 0; register_row < 2; register_row++) {
    uint8_t *registers_raw = data + ci.frame_stride * register_row;
    assert(registers_raw[0] == 0x0a); // Start of line

    int value_tag_count = 0;
    int first_val_idx = 0;
    uint16_t cur_addr = 0;

    for (int i = 1; i <= max_i[register_row]; i = get_next_idx(get_next_idx(i))) {
      int val_idx = get_next_idx(i);

      uint8_t tag = registers_raw[i];
      uint16_t val = registers_raw[val_idx];

      if (tag == 0xAA) { // Register MSB tag
        cur_addr = val << 8;
      } else if (tag == 0xA5) { // Register LSB tag
        cur_addr |= val;
        cur_addr -= 2; // Next value tag will increment address again
      } else if (tag == 0x5A) { // Value tag

        // First tag
        if (value_tag_count % 2 == 0) {
          cur_addr += 2;
          first_val_idx = val_idx;
        } else {
          registers[cur_addr] = std::make_pair(first_val_idx + ci.frame_stride * register_row, val_idx + ci.frame_stride * register_row);
        }

        value_tag_count++;
      }
    }
  }
  return registers;
}

std::map<uint16_t, uint16_t> CameraState::ar0231_parse_registers(uint8_t *data, std::initializer_list<uint16_t> addrs) {
  if (ar0231_register_lut.empty()) {
    ar0231_register_lut = ar0231_build_register_lut(data);
  }

  std::map<uint16_t, uint16_t> registers;
  for (uint16_t addr : addrs) {
    auto offset = ar0231_register_lut[addr];
    registers[addr] = ((uint16_t)data[offset.first] << 8) | data[offset.second];
  }
  return registers;
}

void CameraState::handle_camera_event(void *evdat) {
  if (!enabled) return;
  struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)evdat;
  assert(event_data->session_hdl == session_handle);
  assert(event_data->u.frame_msg.link_hdl == link_handle);

  uint64_t timestamp = event_data->u.frame_msg.timestamp;
  int main_id = event_data->u.frame_msg.frame_id;
  int real_id = event_data->u.frame_msg.request_id;

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
      LOGE("camera %d dropped requests %d %d", camera_num, real_id, request_id_last);
      enqueue_req_multi(request_id_last + 1 + FRAME_BUF_COUNT, real_id - (request_id_last + 1), 0);
    }

    // metas
    frame_id_last = main_id;
    request_id_last = real_id;

    auto &meta_data = buf.camera_bufs_metadata[buf_idx];
    meta_data.frame_id = main_id - idx_offset;
    meta_data.timestamp_sof = timestamp;
    exp_lock.lock();
    meta_data.gain = dc_gain_enabled ? analog_gain_frac * DC_GAIN : analog_gain_frac;
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
  // TODO: Lower latency to 2 frames, by using the histogram outputed by the sensor we can do AE before the debayering is complete

  const float cur_ev_ = cur_ev[buf.cur_frame_data.frame_id % 3];

  // Scale target grey between 0.1 and 0.4 depending on lighting conditions
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + cur_ev_) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, min_ev, max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  float best_ev_score = 1e6;
  int new_g = 0;
  int new_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < 0.2) {
    enable_dc_gain = true;
  } else if (enable_dc_gain && target_grey > 0.3) {
    enable_dc_gain = false;
  }

  // Simple brute force optimizer to choose sensor parameters
  // to reach desired EV
  for (int g = std::max((int)ANALOG_GAIN_MIN_IDX, gain_idx - 1); g <= std::min((int)ANALOG_GAIN_MAX_IDX, gain_idx + 1); g++) {
    float gain = sensor_analog_gains[g] * (enable_dc_gain ? DC_GAIN : 1);

    // Compute optimal time for given gain
    int t = std::clamp(int(std::round(desired_ev / gain)), EXPOSURE_TIME_MIN, EXPOSURE_TIME_MAX);

    // Only go below recomended gain when absolutely necessary to not overexpose
    if (g < ANALOG_GAIN_REC_IDX && t > 20 && g < gain_idx) {
      continue;
    }

    // Compute error to desired ev
    float score = std::abs(desired_ev - (t * gain)) * 10;

    // Going below recomended gain needs lower penalty to not overexpose
    float m = g > ANALOG_GAIN_REC_IDX ? 5.0 : 0.1;
    score += std::abs(g - (int)ANALOG_GAIN_REC_IDX) * m;

    // LOGE("cam: %d - gain: %d, t: %d (%.2f), score %.2f, score + gain %.2f, %.3f, %.3f", camera_num, g, t, desired_ev / gain, score, score + std::abs(g - gain_idx) * (score + 1.0) / 10.0, desired_ev, min_ev);

    // Small penalty on changing gain
    score += std::abs(g - gain_idx) * (score + 1.0) / 10.0;

    if (score < best_ev_score) {
      new_t = t;
      new_g = g;
      best_ev_score = score;
    }
  }

  exp_lock.lock();

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = sensor_analog_gains[new_g];
  gain_idx = new_g;
  exposure_time = new_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * (dc_gain_enabled ? DC_GAIN : 1.0);
  cur_ev[buf.cur_frame_data.frame_id % 3] = exposure_time * gain;

  exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - buf.cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", camera_num, 1e-9 * nanos_since_boot(), 1e-9 * buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - buf.cur_frame_data.timestamp_sof));

  if (camera_id == CAMERA_ID_AR0231) {
    uint16_t analog_gain_reg = 0xFF00 | (new_g << 4) | new_g;
    struct i2c_random_wr_payload exp_reg_array[] = {
                                                    {0x3366, analog_gain_reg},
                                                    {0x3362, (uint16_t)(dc_gain_enabled ? 0x1 : 0x0)},
                                                    {0x3012, (uint16_t)exposure_time},
                                                  };
    sensors_i2c(exp_reg_array, sizeof(exp_reg_array)/sizeof(struct i2c_random_wr_payload), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, true);
  } else if (camera_id == CAMERA_ID_IMX390) {
    // if gain is sub 1, we have to use exposure to mimic sub 1 gains
    uint32_t real_exposure_time = (gain < 1.0) ? (exposure_time*gain) : exposure_time;
    // invert real_exposure_time, max exposure is 2
    real_exposure_time = (exposure_time >= 0x7cf) ? 2 : (0x7cf - exposure_time);
    uint32_t real_gain = int((10*log10(fmax(1.0, gain)))/0.3);
    //printf("%d expose: %d gain: %f = %d\n", camera_num, exposure_time, gain, real_gain);
    struct i2c_random_wr_payload exp_reg_array[] = {
      {0x000c, real_exposure_time&0xFF}, {0x000d, real_exposure_time>>8},
      {0x0010, real_exposure_time&0xFF}, {0x0011, real_exposure_time>>8},
      {0x0018, real_gain&0xFF}, {0x0019, real_gain>>8},
    };
    sensors_i2c(exp_reg_array, sizeof(exp_reg_array)/sizeof(struct i2c_random_wr_payload), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, false);
  }
}

void camera_autoexposure(CameraState *s, float grey_frac) {
  s->set_camera_exposure(grey_frac);
}

static float ar0231_parse_temp_sensor(uint16_t calib1, uint16_t calib2, uint16_t data_reg) {
  // See AR0231 Developer Guide - page 36
  float slope = (125.0 - 55.0) / ((float)calib1 - (float)calib2);
  float t0 = 55.0 - slope * (float)calib2;
  return t0 + slope * (float)data_reg;
}

static void ar0231_process_registers(MultiCameraState *s, CameraState *c, cereal::FrameData::Builder &framed){
  const uint8_t expected_preamble[] = {0x0a, 0xaa, 0x55, 0x20, 0xa5, 0x55};
  uint8_t *data = (uint8_t*)c->buf.cur_camera_buf->addr + c->ci.registers_offset;

  if (memcmp(data, expected_preamble, std::size(expected_preamble)) != 0){
    LOGE("unexpected register data found");
    return;
  }

  auto registers = c->ar0231_parse_registers(data, {0x2000, 0x2002, 0x20b0, 0x20b2, 0x30c6, 0x30c8, 0x30ca, 0x30cc});

  uint32_t frame_id = ((uint32_t)registers[0x2000] << 16) | registers[0x2002];
  framed.setFrameIdSensor(frame_id);

  float temp_0 = ar0231_parse_temp_sensor(registers[0x30c6], registers[0x30c8], registers[0x20b0]);
  float temp_1 = ar0231_parse_temp_sensor(registers[0x30ca], registers[0x30cc], registers[0x20b2]);
  framed.setTemperaturesC({temp_0, temp_1});
}

static void driver_cam_auto_exposure(CameraState *c, SubMaster &sm) {
  struct ExpRect {int x1, x2, x_skip, y1, y2, y_skip;};
  const CameraBuf *b = &c->buf;
  static ExpRect rect = {96, 1832, 2, 242, 1148, 4};
  camera_autoexposure(c, set_exposure_target(b, rect.x1, rect.x2, rect.x_skip, rect.y1, rect.y2, rect.y_skip));
}

static void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  s->sm->update(0);
  driver_cam_auto_exposure(c, *(s->sm));

  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data);
  if (env_send_driver) {
    framed.setImage(get_frame_image(&c->buf));
  }
  if (c->camera_id == CAMERA_ID_AR0231) {
    ar0231_process_registers(s, c, framed);
  }
  s->pm->send("driverCameraState", msg);
}

void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;

  MessageBuilder msg;
  auto framed = c == &s->road_cam ? msg.initEvent().initRoadCameraState() : msg.initEvent().initWideRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  if ((c == &s->road_cam && env_send_road) || (c == &s->wide_road_cam && env_send_wide_road)) {
    framed.setImage(get_frame_image(b));
  } else if (env_log_raw_frames && c == &s->road_cam && cnt % 100 == 5) {  // no overlap with qlog decimation
    framed.setImage(get_raw_frame_image(b));
  }
  LOGT(c->buf.cur_frame_data.frame_id, "%s: Image set", c == &s->road_cam ? "RoadCamera" : "WideRoadCamera");
  if (c == &s->road_cam) {
    framed.setTransform(b->yuv_transform.v);
    LOGT(c->buf.cur_frame_data.frame_id, "%s: Transformed", "RoadCamera");
  }

  if (c->camera_id == CAMERA_ID_AR0231) {
    ar0231_process_registers(s, c, framed);
  }

  s->pm->send(c == &s->road_cam ? "roadCameraState" : "wideRoadCameraState", msg);

  const auto [x, y, w, h] = (c == &s->wide_road_cam) ? std::tuple(96, 250, 1734, 524) : std::tuple(96, 160, 1734, 986);
  const int skip = 2;
  camera_autoexposure(c, set_exposure_target(b, x, x + w, skip, y, y + h, skip));
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  if (s->driver_cam.enabled) threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));
  if (s->road_cam.enabled) threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  if (s->wide_road_cam.enabled) threads.push_back(start_process_thread(s, &s->wide_road_cam, process_road_camera));

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
        // LOGD("v4l2 event: sess_hdl 0x%X, link_hdl 0x%X, frame_id %d, req_id %lld, timestamp 0x%llx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp/1e6, event_data->u.frame_msg.sof_status);
        }

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
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}

