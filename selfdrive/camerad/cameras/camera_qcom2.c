#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "camera_qcom2.h"

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sensor.h"
#include "media/cam_sync.h"

#define FRAME_WIDTH  1928
#define FRAME_HEIGHT 1208
#define FRAME_STRIDE 1936

extern volatile sig_atomic_t do_exit;

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_AR0231] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_STRIDE,
      .bayer = false,
      .bayer_flip = false,
  },
};

// ************** low level camera helpers ****************

int cam_control(int fd, int op_code, void *handle, int size) {
  struct cam_control camcontrol = {0};
  camcontrol.op_code = op_code;
  camcontrol.handle = (uint64_t)handle;
  if (size == 0) { camcontrol.size = 8;
    camcontrol.handle_type = CAM_HANDLE_MEM_HANDLE;
  } else {
    camcontrol.size = size;
    camcontrol.handle_type = CAM_HANDLE_USER_POINTER;
  }

  int ret = ioctl(fd, VIDIOC_CAM_CONTROL, &camcontrol);
  if (ret == -1) {
    perror("wat");
  }
  return ret;
}

int device_control(int fd, int op_code, int session_handle, int dev_handle) {
  // start stop and release are all the same
  static struct cam_release_dev_cmd release_dev_cmd;
  release_dev_cmd.session_handle = session_handle;
  release_dev_cmd.dev_handle = dev_handle;
  return cam_control(fd, op_code, &release_dev_cmd, sizeof(release_dev_cmd));
}

void *alloc_w_mmu_hdl(int video0_fd, int len, int align, int flags, uint32_t *handle, int mmu_hdl, int mmu_hdl2) {
  int ret;

  static struct cam_mem_mgr_alloc_cmd mem_mgr_alloc_cmd = {0};
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

  LOGD("alloced: %x %d %llx mapped %p", mem_mgr_alloc_cmd.out.buf_handle, mem_mgr_alloc_cmd.out.fd, mem_mgr_alloc_cmd.out.vaddr, ptr);

  return ptr;
}

void *alloc(int video0_fd, int len, int align, int flags, uint32_t *handle) {
  return alloc_w_mmu_hdl(video0_fd, len, align, flags, handle, 0, 0);
}

void release(int video0_fd, uint32_t handle) {
  int ret;
  static struct cam_mem_mgr_release_cmd mem_mgr_release_cmd = {0};
  mem_mgr_release_cmd.buf_handle = handle;

  ret = cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

// ************** high level camera helpers ****************

void sensors_init(int video0_fd, int sensor_fd, int camera_num) {
  uint32_t cam_packet_handle = 0;
  int size = sizeof(struct cam_packet)+sizeof(struct cam_cmd_buf_desc)*2;
  struct cam_packet *pkt = alloc(video0_fd, size, 8,
    CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, &cam_packet_handle);
  pkt->num_cmd_buf = 2;
  pkt->kmd_cmd_buf_index = -1;
  pkt->header.op_code = 0x1000003;
  pkt->header.size = size;
  struct cam_cmd_buf_desc *buf_desc = (struct cam_cmd_buf_desc *)&pkt->payload;

  buf_desc[0].size = buf_desc[0].length = sizeof(struct cam_cmd_i2c_info) + sizeof(struct cam_cmd_probe);
  buf_desc[0].type = CAM_CMD_BUF_LEGACY;
  struct cam_cmd_i2c_info *i2c_info = alloc(video0_fd, buf_desc[0].size, 8, CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, &buf_desc[0].mem_handle);
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
  struct cam_cmd_power *power = alloc(video0_fd, buf_desc[1].size, 8, CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE, &buf_desc[1].mem_handle);
  memset(power, 0, buf_desc[1].size);
  struct cam_cmd_unconditional_wait *unconditional_wait;

  void *ptr = power;
  // 7750
  /*power->count = 2;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 2;
  power->power_settings[1].power_seq_type = 8;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));*/

  // 885a
  power->count = 4;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 3; // clock??
  power->power_settings[1].power_seq_type = 1; // analog
  power->power_settings[2].power_seq_type = 2; // digital
  power->power_settings[3].power_seq_type = 8; // reset low
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));
  
  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 5;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

  // set clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = 24000000; //Hz
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 10; // ms
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

  // 8,1 is this reset?
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_UP;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 1;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 100; // ms
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

  // probe happens here

  // disable clock
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 0;
  power->power_settings[0].config_val_low = 0;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 1;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

  // reset high
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 1;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 1;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

  // reset low
  power->count = 1;
  power->cmd_type = CAMERA_SENSOR_CMD_TYPE_PWR_DOWN;
  power->power_settings[0].power_seq_type = 8;
  power->power_settings[0].config_val_low = 0;
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  unconditional_wait = (void*)power;
  unconditional_wait->cmd_type = CAMERA_SENSOR_CMD_TYPE_WAIT;
  unconditional_wait->delay = 1;
  unconditional_wait->op_code = CAMERA_SENSOR_WAIT_OP_SW_UCND;
  power = (void*)power + sizeof(struct cam_cmd_unconditional_wait);

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
  power = (void*)power + (sizeof(struct cam_cmd_power) + (power->count-1)*sizeof(struct cam_power_settings));

  LOGD("probing the sensor");
  int ret = cam_control(sensor_fd, CAM_SENSOR_PROBE_CMD, (void *)cam_packet_handle, 0);
  assert(ret == 0);

  release(video0_fd, buf_desc[0].mem_handle);
  release(video0_fd, buf_desc[1].mem_handle);
  release(video0_fd, cam_packet_handle);
}


// ******************* camera *******************

static void camera_release_buffer(void* cookie, int buf_idx) {
  CameraState *s = cookie;

  // printf("camera_release_buffer %d\n", buf_idx);
  /*s->ss[0].qbuf_info[buf_idx].dirty_buf = 1;
  ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &s->ss[0].qbuf_info[buf_idx]);*/
}

static void camera_init(CameraState *s, int camera_id, int camera_num, unsigned int fps) {
  LOGD("camera init %d", camera_num);

  // TODO: this is copied code from camera_webcam
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_num;
  s->frame_size = s->ci.frame_height * s->ci.frame_stride;

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer, s);

  s->transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};
}

static void camera_open(CameraState *s, VisionBuf* b) {
  int ret;
  s->bufs = b;

  // /dev/v4l-subdev10 is sensor, 11, 12, 13 are the other sensors
  switch (s->camera_num) {
    case 0:
      s->sensor_fd = open("/dev/v4l-subdev10", O_RDWR | O_NONBLOCK);
      break;
    case 1:
      s->sensor_fd = open("/dev/v4l-subdev11", O_RDWR | O_NONBLOCK);
      break;
    case 2:
      s->sensor_fd = open("/dev/v4l-subdev12", O_RDWR | O_NONBLOCK);
      break;
  }
  assert(s->sensor_fd >= 0);
  LOGD("opened sensor");

  // also at /dev/v4l-subdev3, 4, 5, 6
  switch (s->camera_num) {
    case 0:
      s->csiphy_fd = open("/dev/v4l-subdev3", O_RDWR | O_NONBLOCK);
      break;
    case 1:
      s->csiphy_fd = open("/dev/v4l-subdev4", O_RDWR | O_NONBLOCK);
      break;
    case 2:
      s->csiphy_fd = open("/dev/v4l-subdev5", O_RDWR | O_NONBLOCK);
      break;
  }
  assert(s->csiphy_fd >= 0);
  LOGD("opened csiphy");

  // probe the sensor
  LOGD("-- Probing sensor");
  sensors_init(s->video0_fd, s->sensor_fd, s->camera_num);

  memset(&s->req_mgr_session_info, 0, sizeof(s->req_mgr_session_info));
  ret = cam_control(s->video0_fd, CAM_REQ_MGR_CREATE_SESSION, &s->req_mgr_session_info, sizeof(s->req_mgr_session_info));
  LOGD("get session: %d 0x%X", ret, s->req_mgr_session_info.session_hdl);
  s->session_handle = s->req_mgr_session_info.session_hdl;

  // access the sensor
  LOGD("-- Accessing sensor");
  static struct cam_acquire_dev_cmd acquire_dev_cmd = {0};
  acquire_dev_cmd.session_handle = s->session_handle;
  acquire_dev_cmd.handle_type = CAM_HANDLE_USER_POINTER;
  ret = cam_control(s->sensor_fd, CAM_ACQUIRE_DEV, &acquire_dev_cmd, sizeof(acquire_dev_cmd));
  LOGD("acquire sensor dev: %d", ret);
  s->sensor_dev_handle = acquire_dev_cmd.dev_handle;

  static struct cam_isp_resource isp_resource = {0};

  acquire_dev_cmd.session_handle = s->session_handle;
  acquire_dev_cmd.handle_type = CAM_HANDLE_USER_POINTER;
  acquire_dev_cmd.num_resources = 1;
  acquire_dev_cmd.resource_hdl = (uint64_t)&isp_resource;
  
  isp_resource.resource_id = CAM_ISP_RES_ID_PORT;
  isp_resource.length = sizeof(struct cam_isp_in_port_info) + sizeof(struct cam_isp_out_port_info)*(1-1);
  isp_resource.handle_type = CAM_HANDLE_USER_POINTER;

  struct cam_isp_in_port_info *in_port_info = malloc(isp_resource.length);
  isp_resource.res_hdl = (uint64_t)in_port_info;

  switch (s->camera_num) {
    case 0:
      in_port_info->res_type = CAM_ISP_IFE_IN_RES_PHY_0;
      break;
    case 1:
      in_port_info->res_type = CAM_ISP_IFE_IN_RES_PHY_1;
      break;
    case 2:
      in_port_info->res_type = CAM_ISP_IFE_IN_RES_PHY_2;
      break;
  }

  in_port_info->lane_type = CAM_ISP_LANE_TYPE_DPHY;
  in_port_info->lane_num = 4;
  in_port_info->lane_cfg = 0x3210;

  in_port_info->vc = 0x0;
  in_port_info->dt = 0x2C; //CSI_RAW12
  in_port_info->format = CAM_FORMAT_MIPI_RAW_12;

  in_port_info->test_pattern = 0x2; // 0x3?
  in_port_info->usage_type = 0x0;
  
  in_port_info->left_start = 0x0;
  in_port_info->left_stop = FRAME_WIDTH - 1;
  in_port_info->left_width = FRAME_WIDTH;

  in_port_info->right_start = 0x0;
  in_port_info->right_stop = FRAME_WIDTH - 1;
  in_port_info->right_width = FRAME_WIDTH;

  in_port_info->line_start = 0x0;
  in_port_info->line_stop = FRAME_HEIGHT - 1;
  in_port_info->height = FRAME_HEIGHT;

  in_port_info->pixel_clk = 0x0;
  in_port_info->batch_size = 0x0;
  in_port_info->dsp_mode = 0x0;
  in_port_info->hbi_cnt = 0x0;
  in_port_info->custom_csid = 0x0;

  in_port_info->num_out_res = 0x1;
  in_port_info->data[0] = (struct cam_isp_out_port_info){
    .res_type = CAM_ISP_IFE_OUT_RES_RDI_0, 
    .format = CAM_FORMAT_MIPI_RAW_12,
    .width = FRAME_WIDTH, 
    .height = FRAME_HEIGHT,
    .comp_grp_id = 0x0, .split_point = 0x0, .secure_mode = 0x0,
  };

  ret = cam_control(s->isp_fd, CAM_ACQUIRE_DEV, &acquire_dev_cmd, sizeof(acquire_dev_cmd));
  LOGD("acquire isp dev: %d", ret);
  free(in_port_info);
  s->isp_dev_handle = acquire_dev_cmd.dev_handle;

  static struct cam_csiphy_acquire_dev_info csiphy_acquire_dev_info = {0};
  csiphy_acquire_dev_info.combo_mode = 0;

  acquire_dev_cmd.session_handle = s->session_handle;
  acquire_dev_cmd.handle_type = CAM_HANDLE_USER_POINTER;
  acquire_dev_cmd.num_resources = 1;
  acquire_dev_cmd.resource_hdl = (uint64_t)&csiphy_acquire_dev_info;

  ret = cam_control(s->csiphy_fd, CAM_ACQUIRE_DEV, &acquire_dev_cmd, sizeof(acquire_dev_cmd));
  LOGD("acquire csiphy dev: %d", ret);
  s->csiphy_dev_handle = acquire_dev_cmd.dev_handle;

  // acquires done

}

void cameras_init(DualCameraState *s) {
  camera_init(&s->rear, CAMERA_ID_AR0231, 0, 20);
  camera_init(&s->wide, CAMERA_ID_AR0231, 1, 20);
  camera_init(&s->front, CAMERA_ID_AR0231, 2, 20);
}

void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front) {
  int ret;

  LOG("-- Opening devices");
  // video0 is the target of many ioctls
  s->video0_fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
  assert(s->video0_fd >= 0);
  LOGD("opened video0");
  s->rear.video0_fd = s->front.video0_fd = s->wide.video0_fd = s->video0_fd;

  // video1 is the target of some ioctls
  s->video1_fd = open("/dev/video1", O_RDWR | O_NONBLOCK);
  assert(s->video1_fd >= 0);
  LOGD("opened video1");
  s->rear.video1_fd = s->front.video1_fd = s->wide.video1_fd = s->video1_fd;

  // looks like there's only one of these
  s->isp_fd = open("/dev/v4l-subdev1", O_RDWR | O_NONBLOCK);
  assert(s->isp_fd >= 0);
  LOGD("opened isp");
  s->rear.isp_fd = s->front.isp_fd = s->wide.isp_fd = s->isp_fd;

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
  int device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  int cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;
  s->rear.device_iommu = s->front.device_iommu = s->wide.device_iommu = device_iommu;
  s->rear.cdm_iommu = s->front.cdm_iommu = s->wide.cdm_iommu = cdm_iommu;

  // subscribe  
  LOG("-- Subscribing");
  static struct v4l2_event_subscription sub = {0};
  sub.type = 0x8000000;
  sub.id = 0;
  ret = ioctl(s->video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  LOGD("isp subscribe: %d", ret);
  sub.id = 1;
  ret = ioctl(s->video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  LOGD("isp subscribe: %d", ret);

  camera_open(&s->rear, camera_bufs_rear);
  camera_open(&s->front, camera_bufs_front);
  // TODO: add bufs for camera wide
}

static void camera_close(CameraState *s) {
  int ret;

  // release devices
  LOGD("-- Release devices");
  ret = device_control(s->sensor_fd, CAM_RELEASE_DEV, s->session_handle, s->sensor_dev_handle);
  LOGD("release sensor: %d", ret);
  ret = device_control(s->isp_fd, CAM_RELEASE_DEV, s->session_handle, s->isp_dev_handle);
  LOGD("release isp: %d", ret);
  ret = device_control(s->csiphy_fd, CAM_RELEASE_DEV, s->session_handle, s->csiphy_dev_handle);
  LOGD("release csiphy: %d", ret);

  ret = cam_control(s->video0_fd, CAM_REQ_MGR_DESTROY_SESSION, &s->req_mgr_session_info, sizeof(s->req_mgr_session_info));
  LOGD("destroyed session: %d", ret);

  tbuffer_stop(&s->camera_tb);
}

static void cameras_close(DualCameraState *s) {
  camera_close(&s->rear);
  camera_close(&s->front);
  //camera_close(&s->wide);
}

void cameras_run(DualCameraState *s) {
  while (!do_exit) {
    sleep(1);
  }

  LOG(" ************** STOPPING **************");
  cameras_close(s);
}

void camera_autoexposure(CameraState *s, float grey_frac) {
}

