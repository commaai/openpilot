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
#include "selfdrive/common/swaglog.h"
#include "selfdrive/camerad/cameras/sensor2_i2c.h"

extern ExitHandler do_exit;

const size_t FRAME_WIDTH = 1928;
const size_t FRAME_HEIGHT = 1208;
const size_t FRAME_STRIDE = 2416;  // for 10 bit output

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_AR0231] = {
    .frame_width = FRAME_WIDTH,
    .frame_height = FRAME_HEIGHT,
    .frame_stride = FRAME_STRIDE,
    .bayer = true,
    .bayer_flip = 1,
    .hdr = false
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
const int EXPOSURE_TIME_MAX = 1904; // with HDR, slowest ss

// ************** low level camera helpers ****************
int cam_control(int fd, int op_code, void *handle, int size) {
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
    printf("OP CODE ERR - %d \n", op_code);
    perror("wat");
  }
  return ret;
}

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data) {
  struct cam_acquire_dev_cmd cmd = {
      .session_handle = session_handle,
      .handle_type = CAM_HANDLE_USER_POINTER,
      .num_resources = (uint32_t)(data ? 1 : 0),
      .resource_hdl = (uint64_t)data,
  };
  int err = cam_control(fd, CAM_ACQUIRE_DEV, &cmd, sizeof(cmd));
  return err == 0 ? std::make_optional(cmd.dev_handle) : std::nullopt;
};

int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle) {
  struct cam_config_dev_cmd cmd = {
      .session_handle = session_handle,
      .dev_handle = dev_handle,
      .packet_handle = packet_handle,
  };
  return cam_control(fd, CAM_CONFIG_DEV, &cmd, sizeof(cmd));
}

int device_control(int fd, int op_code, int session_handle, int dev_handle) {
  // start stop and release are all the same
  struct cam_start_stop_dev_cmd cmd { .session_handle = session_handle, .dev_handle = dev_handle };
  return cam_control(fd, op_code, &cmd, sizeof(cmd));
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

  cam_control(video0_fd, CAM_REQ_MGR_ALLOC_BUF, &mem_mgr_alloc_cmd, sizeof(mem_mgr_alloc_cmd));
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

  ret = cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

void release_fd(int video0_fd, uint32_t handle) {
  // handle to fd
  close(handle>>16);
  release(video0_fd, handle);
}

void clear_req_queue(int fd, int32_t session_hdl, int32_t link_hdl) {
  struct cam_req_mgr_flush_info req_mgr_flush_request = {0};
  req_mgr_flush_request.session_hdl = session_hdl;
  req_mgr_flush_request.link_hdl = link_hdl;
  req_mgr_flush_request.flush_type = CAM_REQ_MGR_FLUSH_TYPE_ALL;
  int ret;
  ret = cam_control(fd, CAM_REQ_MGR_FLUSH_REQ, &req_mgr_flush_request, sizeof(req_mgr_flush_request));
  // LOGD("flushed all req: %d", ret);
}

// ************** high level camera helpers ****************

void sensors_poke(struct CameraState *s, int request_id) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet);
  struct cam_packet *pkt = (struct cam_packet *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, size, &cam_packet_handle);
  pkt->num_cmd_buf = 0;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = 0x7f;
  pkt->header.request_id = request_id;

  int ret = device_config(s->sensor_fd, s->session_handle, s->sensor_dev_handle, cam_packet_handle);
  assert(ret == 0);

  munmap(pkt, size);
  release_fd(s->multi_cam_state->video0_fd, cam_packet_handle);
}

void sensors_i2c(struct CameraState *s, struct i2c_random_wr_payload* dat, int len, int op_code) {
  // LOGD("sensors_i2c: %d", len);
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
  struct cam_packet *pkt = (struct cam_packet *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, size, &cam_packet_handle);
  pkt->num_cmd_buf = 1;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.size = size;
  pkt->header.op_code = op_code;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct i2c_rdwr_header) + len*sizeof(struct i2c_random_wr_payload);
  buf_desc[0].type = CAM_CMD_BUF_I2C;

  struct cam_cmd_i2c_random_wr *i2c_random_wr = (struct cam_cmd_i2c_random_wr *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  i2c_random_wr->header.count = len;
  i2c_random_wr->header.op_code = 1;
  i2c_random_wr->header.cmd_type = CAMERA_SENSOR_CMD_TYPE_I2C_RNDM_WR;
  i2c_random_wr->header.data_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  i2c_random_wr->header.addr_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  memcpy(i2c_random_wr->random_wr_payload, dat, len*sizeof(struct i2c_random_wr_payload));

  int ret = device_config(s->sensor_fd, s->session_handle, s->sensor_dev_handle, cam_packet_handle);
  assert(ret == 0);

  munmap(i2c_random_wr, buf_desc[0].size);
  release_fd(s->multi_cam_state->video0_fd, buf_desc[0].mem_handle);
  munmap(pkt, size);
  release_fd(s->multi_cam_state->video0_fd, cam_packet_handle);
}
static cam_cmd_power *power_set_wait(cam_cmd_power *power, int16_t delay_ms) {
  cam_cmd_unconditional_wait *unconditional_wait = (cam_cmd_unconditional_wait *)((char *)power + (sizeof(struct cam_cmd_power) + (power->count - 1) * sizeof(struct cam_power_settings)));
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = delay_ms;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  return (struct cam_cmd_power *)(unconditional_wait + 1);
};

void sensors_init(int video0_fd, int sensor_fd, int camera_num) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  struct cam_packet *pkt = (struct cam_packet *)alloc_w_mmu_hdl(video0_fd, size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = 0x1000003;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  struct cam_cmd_i2c_info *i2c_info = (struct cam_cmd_i2c_info *)alloc_w_mmu_hdl(video0_fd, buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
  struct cam_cmd_probe *probe = (struct cam_cmd_probe *)((uint8_t *)i2c_info) + sizeof(struct cam_cmd_i2c_info);

  switch (camera_num) {
    case 0:
      // port 0
      i2c_info->slave_addr = 0x20;
      probe->camera_id = 0;
      break;
    case 1:
      // port 1
      i2c_info->slave_addr = 0x30;
      probe->camera_id = 1;
      break;
    case 2:
      // port 2
      i2c_info->slave_addr = 0x20;
      probe->camera_id = 2;
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
  probe->reg_addr = 0x3000; //0x300a; //0x300b;
  probe->expected_data = 0x354; //0x7750; //0x885a;
  probe->data_mask = 0;

  //buf_desc[1].size = buf_desc[1].length = 148;
  buf_desc[1].size = buf_desc[1].length = 196;
  buf_desc[1].type = CAM_CMD_BUF_I2C;
  struct cam_cmd_power *power_settings = (struct cam_cmd_power *)alloc_w_mmu_hdl(video0_fd, buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle);
  memset(power_settings, 0, buf_desc[1].size);
  // 7750
  /*power->count = 2;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 2;
  power->power_settings[1].power_seq_type = 8;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));*/

  // 885a
  struct cam_cmd_power *power = power_settings;
  power->count = 4;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 3; // clock??
  power->power_settings[1].power_seq_type = 1; // analog
  power->power_settings[2].power_seq_type = 2; // digital
  power->power_settings[3].power_seq_type = 8; // reset low
  power = power_set_wait(power, 5);

  // set clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = 19200000; //Hz
  power = power_set_wait(power, 10);

  // 8,1 is this reset?
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 1;
  power = power_set_wait(power, 100);

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

  // 7750
  /*power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 2;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));*/

  // 885a
  power->count = 3;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 2;
  power->power_settings[1].power_seq_type = 1;
  power->power_settings[2].power_seq_type = 3;

  LOGD("probing the sensor");
  int ret = cam_control(sensor_fd, CAM_SENSOR_PROBE_CMD, (void *)(uintptr_t)cam_packet_handle, 0);
  assert(ret == 0);

  munmap(i2c_info, buf_desc[0].size);
  release_fd(video0_fd, buf_desc[0].mem_handle);
  munmap(power_settings, buf_desc[1].size);
  release_fd(video0_fd, buf_desc[1].mem_handle);
  munmap(pkt, size);
  release_fd(video0_fd, cam_packet_handle);
}

void config_isp(struct CameraState *s, int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  if (io_mem_handle != 0) {
    size += sizeof(struct cam_buf_io_cfg);
  }
  struct cam_packet *pkt = (struct cam_packet *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, size, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = 0;

  if (io_mem_handle != 0) {
    pkt->io_configs_offset = sizeof(struct cam_cmd_buf_desc)*2;
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

   // cam_isp_packet_generic_blob_handler
  uint32_t tmp[] = {
    // size is 0x20, type is 0(CAM_ISP_GENERIC_BLOB_TYPE_HFR_CONFIG)
    0x2000,
    0x1, 0x0, CAM_ISP_IFE_OUT_RES_RDI_0, 0x1, 0x0, 0x1, 0x0, 0x0, // 1 port, CAM_ISP_IFE_OUT_RES_RDI_0
    // size is 0x38, type is 1(CAM_ISP_GENERIC_BLOB_TYPE_CLOCK_CONFIG), clocks
    0x3801,
    0x1, 0x4, // Dual mode, 4 RDI wires
    0x18148d00, 0x0, 0x18148d00, 0x0, 0x18148d00, 0x0, // rdi clock
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0,  // junk?
    // offset 0x60
    // size is 0xe0, type is 2(CAM_ISP_GENERIC_BLOB_TYPE_BW_CONFIG), bandwidth
    0xe002,
    0x1, 0x4, // 4 RDI
    0x0, 0x0, 0x1ad27480, 0x0, 0x1ad27480, 0x0, // left_pix_vote
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, // right_pix_vote
    0x0, 0x0, 0x6ee11c0, 0x2, 0x6ee11c0, 0x2,  // rdi_vote
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

  buf_desc[1].size = sizeof(tmp);
  buf_desc[1].offset = io_mem_handle != 0 ? 0x60 : 0;
  buf_desc[1].length = buf_desc[1].size - buf_desc[1].offset;
  buf_desc[1].type = CAM_CMD_BUF_GENERIC;
  buf_desc[1].meta_data = CAM_ISP_PACKET_META_GENERIC_BLOB_COMMON;
  uint32_t *buf2 = (uint32_t *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, buf_desc[1].size, (uint32_t*)&buf_desc[1].mem_handle, 0x20);
  memcpy(buf2, tmp, sizeof(tmp));

  if (io_mem_handle != 0) {
    io_cfg[0].mem_handle[0] = io_mem_handle;
		io_cfg[0].planes[0] = (struct cam_plane_cfg){
		 .width = FRAME_WIDTH,
		 .height = FRAME_HEIGHT,
		 .plane_stride = FRAME_STRIDE,
		 .slice_height = FRAME_HEIGHT,
		 .meta_stride = 0x0,
		 .meta_size = 0x0,
		 .meta_offset = 0x0,
		 .packer_config = 0x0,
		 .mode_config = 0x0,
		 .tile_config = 0x0,
		 .h_init = 0x0,
		 .v_init = 0x0,
		};
    io_cfg[0].format = CAM_FORMAT_MIPI_RAW_10;
    io_cfg[0].color_pattern = 0x5;
    io_cfg[0].bpp = 0xc;
    io_cfg[0].resource_type = CAM_ISP_IFE_OUT_RES_RDI_0;
    io_cfg[0].fence = fence;
    io_cfg[0].direction = CAM_BUF_OUTPUT;
    io_cfg[0].subsample_pattern = 0x1;
    io_cfg[0].framedrop_pattern = 0x1;
  }

  int ret = device_config(s->multi_cam_state->isp_fd, s->session_handle, s->isp_dev_handle, cam_packet_handle);
  assert(ret == 0);
  if (ret != 0) {
    printf("ISP CONFIG FAILED\n");
  }

  munmap(buf2, buf_desc[1].size);
  release_fd(s->multi_cam_state->video0_fd, buf_desc[1].mem_handle);
  // release_fd(s->multi_cam_state->video0_fd, buf_desc[0].mem_handle);
  munmap(pkt, size);
  release_fd(s->multi_cam_state->video0_fd, cam_packet_handle);
}

void enqueue_buffer(struct CameraState *s, int i, bool dp) {
  int ret;
  int request_id = s->request_ids[i];

  if (s->buf_handle[i]) {
    release(s->multi_cam_state->video0_fd, s->buf_handle[i]);
    // wait
    struct cam_sync_wait sync_wait = {0};
    sync_wait.sync_obj = s->sync_objs[i];
    sync_wait.timeout_ms = 50; // max dt tolerance, typical should be 23
    ret = cam_control(s->multi_cam_state->video1_fd, CAM_SYNC_WAIT, &sync_wait, sizeof(sync_wait));
    // LOGD("fence wait: %d %d", ret, sync_wait.sync_obj);

    s->buf.camera_bufs_metadata[i].timestamp_eof = (uint64_t)nanos_since_boot(); // set true eof
    if (dp) s->buf.queue(i);

    // destroy old output fence
    struct cam_sync_info sync_destroy = {0};
    strcpy(sync_destroy.name, "NodeOutputPortFence");
    sync_destroy.sync_obj = s->sync_objs[i];
    ret = cam_control(s->multi_cam_state->video1_fd, CAM_SYNC_DESTROY, &sync_destroy, sizeof(sync_destroy));
    // LOGD("fence destroy: %d %d", ret, sync_destroy.sync_obj);
  }

  // do stuff
  struct cam_req_mgr_sched_request req_mgr_sched_request = {0};
  req_mgr_sched_request.session_hdl = s->session_handle;
  req_mgr_sched_request.link_hdl = s->link_handle;
  req_mgr_sched_request.req_id = request_id;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_SCHED_REQ, &req_mgr_sched_request, sizeof(req_mgr_sched_request));
  // LOGD("sched req: %d %d", ret, request_id);

  // create output fence
  struct cam_sync_info sync_create = {0};
  strcpy(sync_create.name, "NodeOutputPortFence");
  ret = cam_control(s->multi_cam_state->video1_fd, CAM_SYNC_CREATE, &sync_create, sizeof(sync_create));
  // LOGD("fence req: %d %d", ret, sync_create.sync_obj);
  s->sync_objs[i] = sync_create.sync_obj;

  // configure ISP to put the image in place
  struct cam_mem_mgr_map_cmd mem_mgr_map_cmd = {0};
  mem_mgr_map_cmd.mmu_hdls[0] = s->multi_cam_state->device_iommu;
  mem_mgr_map_cmd.num_hdl = 1;
  mem_mgr_map_cmd.flags = CAM_MEM_FLAG_HW_READ_WRITE;
  mem_mgr_map_cmd.fd = s->buf.camera_bufs[i].fd;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_MAP_BUF, &mem_mgr_map_cmd, sizeof(mem_mgr_map_cmd));
  // LOGD("map buf req: (fd: %d) 0x%x %d", s->bufs[i].fd, mem_mgr_map_cmd.out.buf_handle, ret);
  s->buf_handle[i] = mem_mgr_map_cmd.out.buf_handle;

  // poke sensor
  sensors_poke(s, request_id);
  // LOGD("Poked sensor");

  // push the buffer
  config_isp(s, s->buf_handle[i], s->sync_objs[i], request_id, s->buf0_handle, 65632*(i+1));
}

void enqueue_req_multi(struct CameraState *s, int start, int n, bool dp) {
   for (int i=start;i<start+n;++i) {
     s->request_ids[(i - 1) % FRAME_BUF_COUNT] = i;
     enqueue_buffer(s, (i - 1) % FRAME_BUF_COUNT, dp);
   }
}

// ******************* camera *******************

static void camera_init(MultiCameraState *multi_cam_state, VisionIpcServer * v, CameraState *s, int camera_id, int camera_num, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  LOGD("camera init %d", camera_num);
  s->multi_cam_state = multi_cam_state;
  assert(camera_id < std::size(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_num;

  s->request_id_last = 0;
  s->skipped = true;

  s->min_ev = EXPOSURE_TIME_MIN * sensor_analog_gains[ANALOG_GAIN_MIN_IDX];
  s->max_ev = EXPOSURE_TIME_MAX * sensor_analog_gains[ANALOG_GAIN_MAX_IDX] * DC_GAIN;
  s->target_grey_fraction = 0.3;

  s->dc_gain_enabled = false;
  s->gain_idx = ANALOG_GAIN_REC_IDX;
  s->exposure_time = 5;
  s->cur_ev[0] = s->cur_ev[1] = s->cur_ev[2] = (s->dc_gain_enabled ? DC_GAIN : 1) * sensor_analog_gains[s->gain_idx] * s->exposure_time;

  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type);
}

int open_v4l_by_name_and_index(const char name[], int index, int flags = O_RDWR | O_NONBLOCK) {
  for (int v4l_index = 0; /**/; ++v4l_index) {
    std::string v4l_name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", v4l_index));
    if (v4l_name.empty()) return -1;
    if (v4l_name.find(name) == 0) {
      if (index == 0) {
        return open(util::string_format("/dev/v4l-subdev%d", v4l_index).c_str(), flags);
      }
      index--;
    }
  }
}

static void camera_open(CameraState *s) {
  s->sensor_fd = open_v4l_by_name_and_index("cam-sensor-driver", s->camera_num);
  assert(s->sensor_fd >= 0);
  LOGD("opened sensor");

  s->csiphy_fd = open_v4l_by_name_and_index("cam-csiphy-driver", s->camera_num);
  assert(s->csiphy_fd >= 0);
  LOGD("opened csiphy");

  // probe the sensor
  LOGD("-- Probing sensor %d", s->camera_num);
  sensors_init(s->multi_cam_state->video0_fd, s->sensor_fd, s->camera_num);

  // create session
  struct cam_req_mgr_session_info session_info = {}; 
  int ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_CREATE_SESSION, &session_info, sizeof(session_info));
  LOGD("get session: %d 0x%X", ret, session_info.session_hdl);
  s->session_handle = session_info.session_hdl;

  // access the sensor
  LOGD("-- Accessing sensor");
  auto sensor_dev_handle = device_acquire(s->sensor_fd, s->session_handle, nullptr);
  assert(sensor_dev_handle);
  s->sensor_dev_handle = *sensor_dev_handle;
  LOGD("acquire sensor dev");

  struct cam_isp_in_port_info in_port_info = {
      .res_type = (uint32_t[]){CAM_ISP_IFE_IN_RES_PHY_0, CAM_ISP_IFE_IN_RES_PHY_1, CAM_ISP_IFE_IN_RES_PHY_2}[s->camera_num],

      .lane_type = CAM_ISP_LANE_TYPE_DPHY,
      .lane_num = 4,
      .lane_cfg = 0x3210,

      .vc = 0x0,
      // .dt = 0x2C; //CSI_RAW12
      .dt = 0x2B,  //CSI_RAW10
      .format = CAM_FORMAT_MIPI_RAW_10,

      .test_pattern = 0x2,  // 0x3?
      .usage_type = 0x0,

      .left_start = 0,
      .left_stop = FRAME_WIDTH - 1,
      .left_width = FRAME_WIDTH,

      .right_start = 0,
      .right_stop = FRAME_WIDTH - 1,
      .right_width = FRAME_WIDTH,

      .line_start = 0,
      .line_stop = FRAME_HEIGHT - 1,
      .height = FRAME_HEIGHT,

      .pixel_clk = 0x0,
      .batch_size = 0x0,
      .dsp_mode = 0x0,
      .hbi_cnt = 0x0,
      .custom_csid = 0x0,

      .num_out_res = 0x1,
      .data[0] = (struct cam_isp_out_port_info){
          .res_type = CAM_ISP_IFE_OUT_RES_RDI_0,
          .format = CAM_FORMAT_MIPI_RAW_10,
          .width = FRAME_WIDTH,
          .height = FRAME_HEIGHT,
          .comp_grp_id = 0x0, .split_point = 0x0, .secure_mode = 0x0,
      },
  };
  struct cam_isp_resource isp_resource = {
      .resource_id = CAM_ISP_RES_ID_PORT,
      .handle_type = CAM_HANDLE_USER_POINTER,
      .res_hdl = (uint64_t)&in_port_info,
      .length = sizeof(in_port_info),
  };

  auto isp_dev_handle = device_acquire(s->multi_cam_state->isp_fd, s->session_handle, &isp_resource);
  assert(isp_dev_handle);
  s->isp_dev_handle = *isp_dev_handle; 
  LOGD("acquire isp dev");

  struct cam_csiphy_acquire_dev_info csiphy_acquire_dev_info = {.combo_mode = 0};
  auto csiphy_dev_handle = device_acquire(s->csiphy_fd, s->session_handle, &csiphy_acquire_dev_info);
  assert(csiphy_dev_handle);
  s->csiphy_dev_handle = *csiphy_dev_handle;
  LOGD("acquire csiphy dev");

  // config ISP
  alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, 984480, (uint32_t*)&s->buf0_handle, 0x20, CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, s->multi_cam_state->device_iommu, s->multi_cam_state->cdm_iommu);
  config_isp(s, 0, 0, 1, s->buf0_handle, 0);

  LOG("-- Configuring sensor");
  sensors_i2c(s, init_array_ar0231, std::size(init_array_ar0231), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
  //sensors_i2c(s, start_reg_array, std::size(start_reg_array), CAM_SENSOR_PACKET_OPCODE_SENSOR_STREAMON);
  //sensors_i2c(s, stop_reg_array, std::size(stop_reg_array), CAM_SENSOR_PACKET_OPCODE_SENSOR_STREAMOFF);

  // config csiphy
  LOG("-- Config CSI PHY");
  {
    uint32_t cam_packet_handle = 0;
    int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*1;
    struct cam_packet *pkt = (struct cam_packet *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, size, &cam_packet_handle);
    pkt->num_cmd_buf = 1;
    pkt->kmd_cmd_buf_index = -1;
    pkt->header.size = size;
    struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

    buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_csiphy_info);
    buf_desc[0].type = CAM_CMD_BUF_GENERIC;

    struct cam_csiphy_info *csiphy_info = (struct cam_csiphy_info *)alloc_w_mmu_hdl(s->multi_cam_state->video0_fd, buf_desc[0].size, (uint32_t*)&buf_desc[0].mem_handle);
    csiphy_info->lane_mask = 0x1f;
    csiphy_info->lane_assign = 0x3210;// skip clk. How is this 16 bit for 5 channels??
    csiphy_info->csiphy_3phase = 0x0; // no 3 phase, only 2 conductors per lane
    csiphy_info->combo_mode = 0x0;
    csiphy_info->lane_cnt = 0x4;
    csiphy_info->secure_mode = 0x0;
    csiphy_info->settle_time = MIPI_SETTLE_CNT * 200000000ULL;
    csiphy_info->data_rate = 48000000;  // Calculated by camera_freqs.py

    int ret_ = device_config(s->csiphy_fd, s->session_handle, s->csiphy_dev_handle, cam_packet_handle);
    assert(ret_ == 0);

    munmap(csiphy_info, buf_desc[0].size);
    release_fd(s->multi_cam_state->video0_fd, buf_desc[0].mem_handle);
    munmap(pkt, size);
    release_fd(s->multi_cam_state->video0_fd, cam_packet_handle);
  }

  // link devices
  LOG("-- Link devices");
  struct cam_req_mgr_link_info req_mgr_link_info = {0};
  req_mgr_link_info.session_hdl = s->session_handle;
  req_mgr_link_info.num_devices = 2;
  req_mgr_link_info.dev_hdls[0] = s->isp_dev_handle;
  req_mgr_link_info.dev_hdls[1] = s->sensor_dev_handle;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_LINK, &req_mgr_link_info, sizeof(req_mgr_link_info));
  LOGD("link: %d", ret);
  s->link_handle = req_mgr_link_info.link_hdl;

  struct cam_req_mgr_link_control req_mgr_link_control = {0};
  req_mgr_link_control.ops = CAM_REQ_MGR_LINK_ACTIVATE;
  req_mgr_link_control.session_hdl = s->session_handle;
  req_mgr_link_control.num_links = 1;
  req_mgr_link_control.link_hdls[0] = s->link_handle;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
  LOGD("link control: %d", ret);

  ret = device_control(s->csiphy_fd, CAM_START_DEV, s->session_handle, s->csiphy_dev_handle);
  LOGD("start csiphy: %d", ret);
  ret = device_control(s->multi_cam_state->isp_fd, CAM_START_DEV, s->session_handle, s->isp_dev_handle);
  LOGD("start isp: %d", ret);
  ret = device_control(s->sensor_fd, CAM_START_DEV, s->session_handle, s->sensor_dev_handle);
  LOGD("start sensor: %d", ret);

  enqueue_req_multi(s, 1, FRAME_BUF_COUNT, 0);
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  camera_init(s, v, &s->road_cam, CAMERA_ID_AR0231, 1, 20, device_id, ctx,
              VISION_STREAM_RGB_BACK, VISION_STREAM_ROAD); // swap left/right
  printf("road camera initted \n");
  camera_init(s, v, &s->wide_road_cam, CAMERA_ID_AR0231, 0, 20, device_id, ctx,
              VISION_STREAM_RGB_WIDE, VISION_STREAM_WIDE_ROAD);
  printf("wide road camera initted \n");
  camera_init(s, v, &s->driver_cam, CAMERA_ID_AR0231, 2, 20, device_id, ctx,
              VISION_STREAM_RGB_FRONT, VISION_STREAM_DRIVER);
  printf("driver camera initted \n");

  s->sm = new SubMaster({"driverState"});
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
  s->video1_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-cam_sync-video-index0", O_RDWR | O_NONBLOCK));
  assert(s->video1_fd >= 0);
  LOGD("opened video1");

  // looks like there's only one of these
  s->isp_fd = HANDLE_EINTR(open("/dev/v4l-subdev1", O_RDWR | O_NONBLOCK));
  assert(s->isp_fd >= 0);
  LOGD("opened isp");

  // query icp for MMU handles
  LOG("-- Query ICP for MMU handles");
  static struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  static struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  ret = cam_control(s->isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  s->device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  s->cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;

  // subscribe
  LOG("-- Subscribing");
  static struct v4l2_event_subscription sub = {0};
  sub.type = V4L_EVENT_CAM_REQ_MGR_EVENT;
  sub.id = 2; // should use boot time for sof
  ret = HANDLE_EINTR(ioctl(s->video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));
  printf("req mgr subscribe: %d\n", ret);

  camera_open(&s->road_cam);
  printf("road camera opened \n");
  camera_open(&s->wide_road_cam);
  printf("wide road camera opened \n");
  camera_open(&s->driver_cam);
  printf("driver camera opened \n");
}

static void camera_close(CameraState *s) {
  int ret;

  // stop devices
  LOG("-- Stop devices");
  // ret = device_control(s->sensor_fd, CAM_STOP_DEV, s->session_handle, s->sensor_dev_handle);
  // LOGD("stop sensor: %d", ret);
  ret = device_control(s->multi_cam_state->isp_fd, CAM_STOP_DEV, s->session_handle, s->isp_dev_handle);
  LOGD("stop isp: %d", ret);
  ret = device_control(s->csiphy_fd, CAM_STOP_DEV, s->session_handle, s->csiphy_dev_handle);
  LOGD("stop csiphy: %d", ret);
  // link control stop
  LOG("-- Stop link control");
  static struct cam_req_mgr_link_control req_mgr_link_control = {0};
  req_mgr_link_control.ops = CAM_REQ_MGR_LINK_DEACTIVATE;
  req_mgr_link_control.session_hdl = s->session_handle;
  req_mgr_link_control.num_links = 1;
  req_mgr_link_control.link_hdls[0] = s->link_handle;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_LINK_CONTROL, &req_mgr_link_control, sizeof(req_mgr_link_control));
  LOGD("link control stop: %d", ret);

  // unlink
  LOG("-- Unlink");
  static struct cam_req_mgr_unlink_info req_mgr_unlink_info = {0};
  req_mgr_unlink_info.session_hdl = s->session_handle;
  req_mgr_unlink_info.link_hdl = s->link_handle;
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_UNLINK, &req_mgr_unlink_info, sizeof(req_mgr_unlink_info));
  LOGD("unlink: %d", ret);

  // release devices
  LOGD("-- Release devices");
  ret = device_control(s->sensor_fd, CAM_RELEASE_DEV, s->session_handle, s->sensor_dev_handle);
  LOGD("release sensor: %d", ret);
  ret = device_control(s->multi_cam_state->isp_fd, CAM_RELEASE_DEV, s->session_handle, s->isp_dev_handle);
  LOGD("release isp: %d", ret);
  ret = device_control(s->csiphy_fd, CAM_RELEASE_DEV, s->session_handle, s->csiphy_dev_handle);
  LOGD("release csiphy: %d", ret);

  // destroyed session
  struct cam_req_mgr_session_info session_info = {.session_hdl = s->session_handle};
  ret = cam_control(s->multi_cam_state->video0_fd, CAM_REQ_MGR_DESTROY_SESSION, &session_info, sizeof(session_info));
  LOGD("destroyed session: %d", ret);
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->road_cam);
  camera_close(&s->wide_road_cam);
  camera_close(&s->driver_cam);

  delete s->sm;
  delete s->pm;
}

// ******************* just a helper *******************

void handle_camera_event(CameraState *s, void *evdat) {
  struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)evdat;

  uint64_t timestamp = event_data->u.frame_msg.timestamp;
  int main_id = event_data->u.frame_msg.frame_id;
  int real_id = event_data->u.frame_msg.request_id;

  if (real_id != 0) { // next ready
    if (real_id == 1) {s->idx_offset = main_id;}
    int buf_idx = (real_id - 1) % FRAME_BUF_COUNT;

    // check for skipped frames
    if (main_id > s->frame_id_last + 1 && !s->skipped) {
      // realign
      clear_req_queue(s->multi_cam_state->video0_fd, event_data->session_hdl, event_data->u.frame_msg.link_hdl);
      enqueue_req_multi(s, real_id + 1, FRAME_BUF_COUNT - 1, 0);
      s->skipped = true;
    } else if (main_id == s->frame_id_last + 1) {
      s->skipped = false;
    }

    // check for dropped requests
    if (real_id > s->request_id_last + 1) {
      enqueue_req_multi(s, s->request_id_last + 1 + FRAME_BUF_COUNT, real_id - (s->request_id_last + 1), 0);
    }

    // metas
    s->frame_id_last = main_id;
    s->request_id_last = real_id;

    auto &meta_data = s->buf.camera_bufs_metadata[buf_idx];
    meta_data.frame_id = main_id - s->idx_offset;
    meta_data.timestamp_sof = timestamp;
    s->exp_lock.lock();
    meta_data.gain = s->dc_gain_enabled ? s->analog_gain_frac * DC_GAIN : s->analog_gain_frac;
    meta_data.high_conversion_gain = s->dc_gain_enabled;
    meta_data.integ_lines = s->exposure_time;
    meta_data.measured_grey_fraction = s->measured_grey_fraction;
    meta_data.target_grey_fraction = s->target_grey_fraction;
    s->exp_lock.unlock();

    // dispatch
    enqueue_req_multi(s, real_id + FRAME_BUF_COUNT, 1, 1);
  } else { // not ready
    // reset after half second of no response
    if (main_id > s->frame_id_last + 10) {
      clear_req_queue(s->multi_cam_state->video0_fd, event_data->session_hdl, event_data->u.frame_msg.link_hdl);
      enqueue_req_multi(s, s->request_id_last + 1, FRAME_BUF_COUNT, 0);
      s->frame_id_last = main_id;
      s->skipped = true;
    }
  }
}

static void set_camera_exposure(CameraState *s, float grey_frac) {
  const float dt = 0.05;

  const float ts_grey = 10.0;
  const float ts_ev = 0.05;

  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  // It takes 3 frames for the commanded exposure settings to take effect. The first frame is already started by the time
  // we reach this function, the other 2 are due to the register buffering in the sensor.
  // Therefore we use the target EV from 3 frames ago, the grey fraction that was just measured was the result of that control action.
  // TODO: Lower latency to 2 frames, by using the histogram outputed by the sensor we can do AE before the debayering is complete

  const float cur_ev = s->cur_ev[s->buf.cur_frame_data.frame_id % 3];

  // Scale target grey between 0.1 and 0.4 depending on lighting conditions
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + cur_ev) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * s->target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev * target_grey / grey_frac, s->min_ev, s->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * s->cur_ev[0]) + (k * s->cur_ev[1]) + (k * s->cur_ev[2]) + (k_ev * desired_ev);

  float best_ev_score = 1e6;
  int new_g = 0;
  int new_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = s->dc_gain_enabled;
  if (!enable_dc_gain && target_grey < 0.2) {
    enable_dc_gain = true;
  } else if (enable_dc_gain && target_grey > 0.3) {
    enable_dc_gain = false;
  }

  // Simple brute force optimizer to choose sensor parameters
  // to reach desired EV
  for (int g = std::max((int)ANALOG_GAIN_MIN_IDX, s->gain_idx - 1); g <= std::min((int)ANALOG_GAIN_MAX_IDX, s->gain_idx + 1); g++) {
    float gain = sensor_analog_gains[g] * (enable_dc_gain ? DC_GAIN : 1);

    // Compute optimal time for given gain
    int t = std::clamp(int(std::round(desired_ev / gain)), EXPOSURE_TIME_MIN, EXPOSURE_TIME_MAX);

    // Only go below recomended gain when absolutely necessary to not overexpose
    if (g < ANALOG_GAIN_REC_IDX && t > 20 && g < s->gain_idx) {
      continue;
    }

    // Compute error to desired ev
    float score = std::abs(desired_ev - (t * gain)) * 10;

    // Going below recomended gain needs lower penalty to not overexpose
    float m = g > ANALOG_GAIN_REC_IDX ? 5.0 : 0.1;
    score += std::abs(g - (int)ANALOG_GAIN_REC_IDX) * m;

    // LOGE("cam: %d - gain: %d, t: %d (%.2f), score %.2f, score + gain %.2f, %.3f, %.3f", s->camera_num, g, t, desired_ev / gain, score, score + std::abs(g - s->gain_idx) * (score + 1.0) / 10.0, desired_ev, s->min_ev);

    // Small penalty on changing gain
    score += std::abs(g - s->gain_idx) * (score + 1.0) / 10.0;

    if (score < best_ev_score) {
      new_t = t;
      new_g = g;
      best_ev_score = score;
    }
  }

  s->exp_lock.lock();

  s->measured_grey_fraction = grey_frac;
  s->target_grey_fraction = target_grey;

  s->analog_gain_frac = sensor_analog_gains[new_g];
  s->gain_idx = new_g;
  s->exposure_time = new_t;
  s->dc_gain_enabled = enable_dc_gain;

  float gain = s->analog_gain_frac * (s->dc_gain_enabled ? DC_GAIN : 1.0);
  s->cur_ev[s->buf.cur_frame_data.frame_id % 3] = s->exposure_time * gain;

  s->exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - s->buf.cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", s->camera_num, 1e-9 * nanos_since_boot(), 1e-9 * s->buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - s->buf.cur_frame_data.timestamp_sof));

  uint16_t analog_gain_reg = 0xFF00 | (new_g << 4) | new_g;
  struct i2c_random_wr_payload exp_reg_array[] = {
                                                  {0x3366, analog_gain_reg},
                                                  {0x3362, (uint16_t)(s->dc_gain_enabled ? 0x1 : 0x0)},
                                                  {0x3012, (uint16_t)s->exposure_time},
                                                };
  sensors_i2c(s, exp_reg_array, sizeof(exp_reg_array)/sizeof(struct i2c_random_wr_payload),
              CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);

}

void camera_autoexposure(CameraState *s, float grey_frac) {
  set_camera_exposure(s, grey_frac);
}

// called by processing_thread
void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;

  MessageBuilder msg;
  auto framed = c == &s->road_cam ? msg.initEvent().initRoadCameraState() : msg.initEvent().initWideRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  if ((c == &s->road_cam && env_send_road) || (c == &s->wide_road_cam && env_send_wide_road)) {
    framed.setImage(get_frame_image(b));
  }
  if (c == &s->road_cam) {
    framed.setTransform(b->yuv_transform.v);
  }
  s->pm->send(c == &s->road_cam ? "roadCameraState" : "wideRoadCameraState", msg);

  const auto [x, y, w, h] = (c == &s->wide_road_cam) ? std::tuple(96, 250, 1734, 524) : std::tuple(96, 160, 1734, 986);
  const int skip = 2;
  camera_autoexposure(c, set_exposure_target(b, x, x + w, skip, y, y + h, skip));
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  threads.push_back(start_process_thread(s, &s->driver_cam, common_process_driver_camera));
  threads.push_back(start_process_thread(s, &s->wide_road_cam, process_road_camera));

  // start devices
  LOG("-- Starting devices");
  int start_reg_len = sizeof(start_reg_array) / sizeof(struct i2c_random_wr_payload);
  sensors_i2c(&s->road_cam, start_reg_array, start_reg_len, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
  sensors_i2c(&s->wide_road_cam, start_reg_array, start_reg_len, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);
  sensors_i2c(&s->driver_cam, start_reg_array, start_reg_len, CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG);

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
        // LOGD("v4l2 event: sess_hdl %d, link_hdl %d, frame_id %d, req_id %lld, timestamp 0x%llx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);
        // printf("sess_hdl %d, link_hdl %d, frame_id %lu, req_id %lu, timestamp 0x%lx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);

        if (event_data->session_hdl == s->road_cam.session_handle) {
          handle_camera_event(&s->road_cam, event_data);
        } else if (event_data->session_hdl == s->wide_road_cam.session_handle) {
          handle_camera_event(&s->wide_road_cam, event_data);
        } else if (event_data->session_hdl == s->driver_cam.session_handle) {
          handle_camera_event(&s->driver_cam, event_data);
        } else {
          printf("Unknown vidioc event source\n");
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
