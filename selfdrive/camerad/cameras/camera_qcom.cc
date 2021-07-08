#include "selfdrive/camerad/cameras/camera_qcom.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>

#include <cutils/properties.h>
#include <linux/media.h>

#include "selfdrive/camerad/cameras/sensor_i2c.h"
#include "selfdrive/camerad/include/msm_cam_sensor.h"
#include "selfdrive/camerad/include/msmb_camera.h"
#include "selfdrive/camerad/include/msmb_isp.h"
#include "selfdrive/camerad/include/msmb_ispif.h"
#include "selfdrive/common/clutil.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

// leeco actuator (DW9800W H-Bridge Driver IC)
// from sniff
const uint16_t INFINITY_DAC = 364;

extern ExitHandler do_exit;

static int cam_ioctl(int fd, unsigned long int request, void *arg, const char *log_msg = nullptr) {
  int err = ioctl(fd, request, arg);
  if (err != 0 && log_msg) {
    LOG(util::string_format("%s: %d", log_msg, err).c_str());
  }
  return err;
}
// global var for AE/AF ops
std::atomic<CameraExpInfo> road_cam_exp{{0}};
std::atomic<CameraExpInfo> driver_cam_exp{{0}};

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
    .frame_width = 2328,
    .frame_height = 1748,
    .frame_stride = 2912,
    .bayer = true,
    .bayer_flip = 3,
    .hdr = true
  },
  [CAMERA_ID_OV8865] = {
    .frame_width = 1632,
    .frame_height = 1224,
    .frame_stride = 2040, // seems right
    .bayer = true,
    .bayer_flip = 3,
    .hdr = false
  },
  // this exists to get the kernel to build for the LeEco in release
  [CAMERA_ID_IMX298_FLIPPED] = {
    .frame_width = 2328,
    .frame_height = 1748,
    .frame_stride = 2912,
    .bayer = true,
    .bayer_flip = 3,
    .hdr = true
  },
  [CAMERA_ID_OV10640] = {
    .frame_width = 1280,
    .frame_height = 1080,
    .frame_stride = 2040,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = true
  },
};

static void camera_release_buffer(void* cookie, int buf_idx) {
  CameraState *s = (CameraState *)cookie;
  // printf("camera_release_buffer %d\n", buf_idx);
  s->ss[0].qbuf_info[buf_idx].dirty_buf = 1;
  ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &s->ss[0].qbuf_info[buf_idx]);
}

int sensor_write_regs(CameraState *s, struct msm_camera_i2c_reg_array* arr, size_t size, msm_camera_i2c_data_type data_type) {
  struct msm_camera_i2c_reg_setting out_settings = {
    .reg_setting = arr,
    .size = (uint16_t)size,
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = data_type,
    .delay = 0,
  };
  sensorb_cfg_data cfg_data = {.cfgtype = CFG_WRITE_I2C_ARRAY, .cfg.setting = &out_settings};
  return ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &cfg_data);
}

static int imx298_apply_exposure(CameraState *s, int gain, int integ_lines, uint32_t frame_length) {
  int analog_gain = std::min(gain, 448);
  s->digital_gain = gain > 448 ? (512.0/(512-(gain))) / 8.0 : 1.0;
  //printf("%5d/%5d %5d %f\n", s->cur_integ_lines, s->frame_length, analog_gain, s->digital_gain);

  struct msm_camera_i2c_reg_array reg_array[] = {
    // REG_HOLD
    {0x104,0x1,0},
    {0x3002,0x0,0}, // long autoexposure off

    // FRM_LENGTH
    {0x340, (uint16_t)(frame_length >> 8), 0}, {0x341, (uint16_t)(frame_length & 0xff), 0},
    // INTEG_TIME aka coarse_int_time_addr aka shutter speed
    {0x202, (uint16_t)(integ_lines >> 8), 0}, {0x203, (uint16_t)(integ_lines & 0xff),0},
    // global_gain_addr
    // if you assume 1x gain is 32, 448 is 14x gain, aka 2^14=16384
    {0x204, (uint16_t)(analog_gain >> 8), 0}, {0x205, (uint16_t)(analog_gain & 0xff),0},

    // digital gain for colors: gain_greenR, gain_red, gain_blue, gain_greenB
    /*{0x20e, digital_gain_gr >> 8, 0}, {0x20f,digital_gain_gr & 0xFF,0},
    {0x210, digital_gain_r >> 8, 0}, {0x211,digital_gain_r & 0xFF,0},
    {0x212, digital_gain_b >> 8, 0}, {0x213,digital_gain_b & 0xFF,0},
    {0x214, digital_gain_gb >> 8, 0}, {0x215,digital_gain_gb & 0xFF,0},*/

    // REG_HOLD
    {0x104,0x0,0},
  };
  return sensor_write_regs(s, reg_array, std::size(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
}

static int ov8865_apply_exposure(CameraState *s, int gain, int integ_lines, uint32_t frame_length) {
  //printf("driver camera: %d %d %d\n", gain, integ_lines, frame_length);
  int coarse_gain_bitmap, fine_gain_bitmap;

  // get bitmaps from iso
  static const int gains[] = {0, 100, 200, 400, 800};
  int i;
  for (i = 1; i < std::size(gains); i++) {
    if (gain >= gains[i - 1] && gain < gains[i])
      break;
  }
  int coarse_gain = i - 1;
  float fine_gain = (gain - gains[coarse_gain])/(float)(gains[coarse_gain+1]-gains[coarse_gain]);
  coarse_gain_bitmap = (1 << coarse_gain) - 1;
  fine_gain_bitmap = ((int)(16*fine_gain) << 3) + 128; // 7th is always 1, 0-2nd are always 0

  integ_lines *= 16; // The exposure value in reg is in 16ths of a line

  struct msm_camera_i2c_reg_array reg_array[] = {
    //{0x104,0x1,0},

    // FRM_LENGTH
    {0x380e, (uint16_t)(frame_length >> 8), 0}, {0x380f, (uint16_t)(frame_length & 0xff), 0},
    // AEC EXPO
    {0x3500, (uint16_t)(integ_lines >> 16), 0}, {0x3501, (uint16_t)(integ_lines >> 8), 0}, {0x3502, (uint16_t)(integ_lines & 0xff),0},
    // AEC MANUAL
    {0x3503, 0x4, 0},
    // AEC GAIN
    {0x3508, (uint16_t)(coarse_gain_bitmap), 0}, {0x3509, (uint16_t)(fine_gain_bitmap), 0},

    //{0x104,0x0,0},
  };
  return sensor_write_regs(s, reg_array, std::size(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
}

static void camera_init(VisionIpcServer *v, CameraState *s, int camera_id, int camera_num,
                        uint32_t pixel_clock, uint32_t line_length_pclk,
                        uint32_t max_gain, uint32_t fps, cl_device_id device_id, cl_context ctx,
                        VisionStreamType rgb_type, VisionStreamType yuv_type) {
  s->camera_num = camera_num;
  s->camera_id = camera_id;

  assert(camera_id < std::size(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->pixel_clock = pixel_clock;
  s->max_gain = max_gain;
  s->fps = fps;
  s->frame_length = s->pixel_clock / line_length_pclk / s->fps;
  s->self_recover = 0;

  s->apply_exposure = (camera_id == CAMERA_ID_IMX298) ? imx298_apply_exposure : ov8865_apply_exposure;
  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type, camera_release_buffer);
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  char project_name[1024] = {0};
  property_get("ro.boot.project_name", project_name, "");
  assert(strlen(project_name) == 0);

  // sensor is flipped in LP3
  // IMAGE_ORIENT = 3
  init_array_imx298[0].reg_data = 3;

  // 0   = ISO 100
  // 256 = ISO 200
  // 384 = ISO 400
  // 448 = ISO 800
  // 480 = ISO 1600
  // 496 = ISO 3200
  // 504 = ISO 6400, 8x digital gain
  // 508 = ISO 12800, 16x digital gain
  // 510 = ISO 25600, 32x digital gain

  camera_init(v, &s->road_cam, CAMERA_ID_IMX298, 0,
              /*pixel_clock=*/600000000, /*line_length_pclk=*/5536,
              /*max_gain=*/510,  //0 (ISO 100)- 448 (ISO 800, max analog gain) - 511 (super noisy)
#ifdef HIGH_FPS
              /*fps*/ 60,
#else
              /*fps*/ 20,
#endif
              device_id, ctx,
              VISION_STREAM_RGB_BACK, VISION_STREAM_YUV_BACK);

  camera_init(v, &s->driver_cam, CAMERA_ID_OV8865, 1,
              /*pixel_clock=*/72000000, /*line_length_pclk=*/1602,
              /*max_gain=*/510, 10, device_id, ctx,
              VISION_STREAM_RGB_FRONT, VISION_STREAM_YUV_FRONT);

  s->sm = new SubMaster({"driverState"});
  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});

  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // TODO: make lengths correct
    s->focus_bufs[i].allocate(0xb80);
    s->stats_bufs[i].allocate(0xb80);
  }
  std::fill_n(s->lapres, std::size(s->lapres), 16160);
  s->lap_conv = new LapConv(device_id, ctx, s->road_cam.buf.rgb_width, s->road_cam.buf.rgb_height, 3);
}

static void set_exposure(CameraState *s, float exposure_frac, float gain_frac) {
  int err = 0;
  uint32_t gain = s->cur_gain;
  uint32_t integ_lines = s->cur_integ_lines;

  if (exposure_frac >= 0) {
    exposure_frac = std::clamp(exposure_frac, 2.0f / s->frame_length, 1.0f);
    integ_lines = s->frame_length * exposure_frac;

    // See page 79 of the datasheet, this is the max allowed (-1 for phase adjust)
    integ_lines = std::min(integ_lines, s->frame_length - 11);
  }

  if (gain_frac >= 0) {
    // ISO200 is minimum gain
    gain_frac = std::clamp(gain_frac, 1.0f/64, 1.0f);

    // linearize gain response
    // TODO: will be wrong for driver camera
    // 0.125 -> 448
    // 0.25  -> 480
    // 0.5   -> 496
    // 1.0   -> 504
    // 512 - 512/(128*gain_frac)
    gain = (s->max_gain/510) * (512 - 512/(256*gain_frac));
  }

  if (gain != s->cur_gain || integ_lines != s->cur_integ_lines) {
    if (s->apply_exposure == ov8865_apply_exposure) {
      gain = 800 * gain_frac; // ISO
    }
    err = s->apply_exposure(s, gain, integ_lines, s->frame_length);
    if (err == 0) {
      std::lock_guard lk(s->frame_info_lock);
      s->cur_gain = gain;
      s->cur_integ_lines = integ_lines;
    } else {
      LOGE("camera %d apply_exposure err: %d", s->camera_num, err);
    }
  }

  if (err == 0) {
    s->cur_exposure_frac = exposure_frac;
    std::lock_guard lk(s->frame_info_lock);
    s->cur_gain_frac = gain_frac;
  }

  //LOGD("set exposure: %f %f - %d", exposure_frac, gain_frac, err);
}

static void do_autoexposure(CameraState *s, float grey_frac) {
  const float target_grey = 0.3;

  s->frame_info_lock.lock();
  s->measured_grey_fraction = grey_frac;
  s->target_grey_fraction = target_grey;
  s->frame_info_lock.unlock();

  if (s->apply_exposure == ov8865_apply_exposure) {
    // gain limits downstream
    const float gain_frac_min = 0.015625;
    const float gain_frac_max = 1.0;
    // exposure time limits
    const uint32_t exposure_time_min = 16;
    const uint32_t exposure_time_max = s->frame_length - 11; // copied from set_exposure()

    float cur_gain_frac = s->cur_gain_frac;
    float exposure_factor = pow(1.05, (target_grey - grey_frac) / 0.05);
    if (cur_gain_frac > 0.125 && exposure_factor < 1) {
      cur_gain_frac *= exposure_factor;
    } else if (s->cur_integ_lines * exposure_factor <= exposure_time_max && s->cur_integ_lines * exposure_factor >= exposure_time_min) { // adjust exposure time first
      s->cur_exposure_frac *= exposure_factor;
    } else if (cur_gain_frac * exposure_factor <= gain_frac_max && cur_gain_frac * exposure_factor >= gain_frac_min) {
      cur_gain_frac *= exposure_factor;
    }
    s->frame_info_lock.lock();
    s->cur_gain_frac = cur_gain_frac;
    s->frame_info_lock.unlock();

    set_exposure(s, s->cur_exposure_frac, cur_gain_frac);
  } else { // keep the old for others
    float new_exposure = s->cur_exposure_frac;
    new_exposure *= pow(1.05, (target_grey - grey_frac) / 0.05 );
    //LOGD("diff %f: %f to %f", target_grey - grey_frac, s->cur_exposure_frac, new_exposure);

    float new_gain = s->cur_gain_frac;
    if (new_exposure < 0.10) {
      new_gain *= 0.95;
    } else if (new_exposure > 0.40) {
      new_gain *= 1.05;
    }

    set_exposure(s, new_exposure, new_gain);
  }
}

static void sensors_init(MultiCameraState *s) {
  msm_camera_sensor_slave_info slave_infos[2] = {
  (msm_camera_sensor_slave_info){ // road camera
    .sensor_name = "imx298",
    .eeprom_name = "sony_imx298",
    .actuator_name = "dw9800w",
    .ois_name = "",
    .flash_name = "pmic",
    .camera_id = CAMERA_0,
    .slave_addr = 32,
    .i2c_freq_mode = I2C_FAST_MODE,
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .sensor_id_info = {.sensor_id_reg_addr = 22, .sensor_id = 664, .module_id = 9, .vcm_id = 6},
    .power_setting_array = {
      .power_setting_a = {
        {.seq_type = SENSOR_GPIO, .delay = 1},
        {.seq_type = SENSOR_VREG, .seq_val = 2},
        {.seq_type = SENSOR_GPIO, .seq_val = 5, .config_val = 2},
        {.seq_type = SENSOR_VREG, .seq_val = 1},
        {.seq_type = SENSOR_VREG, .seq_val = 3, .delay = 1},
        {.seq_type = SENSOR_CLK, .config_val = 24000000, .delay = 1},
        {.seq_type = SENSOR_GPIO, .config_val = 2, .delay = 10},
      },
      .size = 7,
      .power_down_setting_a = {
        {.seq_type = SENSOR_CLK, .delay = 1},
        {.seq_type = SENSOR_GPIO, .delay = 1},
        {.seq_type = SENSOR_VREG, .seq_val = 1},
        {.seq_type = SENSOR_GPIO, .seq_val = 5},
        {.seq_type = SENSOR_VREG, .seq_val = 2},
        {.seq_type = SENSOR_VREG, .seq_val = 3, .delay = 1},
      },
      .size_down = 6,
    },
    .is_init_params_valid = 0,
    .sensor_init_params = {.modes_supported = 1, .position = BACK_CAMERA_B, .sensor_mount_angle = 90},
    .output_format = MSM_SENSOR_BAYER,
  },
  (msm_camera_sensor_slave_info){ // driver camera
    .sensor_name = "ov8865_sunny",
    .eeprom_name = "ov8865_plus",
    .actuator_name = "",
    .ois_name = "",
    .flash_name = "",
    .camera_id = CAMERA_2,
    .slave_addr = 108,
    .i2c_freq_mode = I2C_FAST_MODE,
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .sensor_id_info = {.sensor_id_reg_addr = 12299, .sensor_id = 34917, .module_id = 2},
    .power_setting_array = {
      .power_setting_a = {
        {.seq_type = SENSOR_GPIO, .delay = 5},
        {.seq_type = SENSOR_VREG, .seq_val = 1},
        {.seq_type = SENSOR_VREG, .seq_val = 2},
        {.seq_type = SENSOR_VREG},
        {.seq_type = SENSOR_CLK, .config_val = 24000000, .delay = 1},
        {.seq_type = SENSOR_GPIO, .config_val = 2, .delay = 1},
      },
      .size = 6,
      .power_down_setting_a = {
        {.seq_type = SENSOR_GPIO, .delay = 5},
        {.seq_type = SENSOR_CLK, .delay = 1},
        {.seq_type = SENSOR_VREG},
        {.seq_type = SENSOR_VREG, .seq_val = 1},
        {.seq_type = SENSOR_VREG, .seq_val = 2, .delay = 1},
      },
      .size_down = 5,
    },
    .is_init_params_valid = 0,
    .sensor_init_params = {.modes_supported = 1, .position = FRONT_CAMERA_B, .sensor_mount_angle = 270},
    .output_format = MSM_SENSOR_BAYER,
  }};

  unique_fd sensorinit_fd = open("/dev/v4l-subdev11", O_RDWR | O_NONBLOCK);
  assert(sensorinit_fd >= 0);
  for (auto &info : slave_infos) {
    info.power_setting_array.power_setting = &info.power_setting_array.power_setting_a[0];
    info.power_setting_array.power_down_setting = &info.power_setting_array.power_down_setting_a[0];
    sensor_init_cfg_data sensor_init_cfg = {.cfgtype = CFG_SINIT_PROBE, .cfg.setting = &info};
    int err = cam_ioctl(sensorinit_fd, VIDIOC_MSM_SENSOR_INIT_CFG, &sensor_init_cfg, "sensor init cfg");
    assert(err >= 0);
  }
}

static void camera_open(CameraState *s, bool is_road_cam) {
  struct csid_cfg_data csid_cfg_data = {};
  struct v4l2_event_subscription sub = {};

  struct msm_actuator_cfg_data actuator_cfg_data = {};

  // open devices
  const char *sensor_dev;
  if (is_road_cam) {
    s->csid_fd = open("/dev/v4l-subdev3", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev0", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    sensor_dev = "/dev/v4l-subdev17";
    s->isp_fd = open("/dev/v4l-subdev13", O_RDWR | O_NONBLOCK);
    assert(s->isp_fd >= 0);
    s->actuator_fd = open("/dev/v4l-subdev7", O_RDWR | O_NONBLOCK);
    assert(s->actuator_fd >= 0);
  } else {
    s->csid_fd = open("/dev/v4l-subdev5", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev2", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    sensor_dev = "/dev/v4l-subdev18";
    s->isp_fd = open("/dev/v4l-subdev14", O_RDWR | O_NONBLOCK);
    assert(s->isp_fd >= 0);
  }

  // wait for sensor device
  // on first startup, these devices aren't present yet
  for (int i = 0; i < 10; i++) {
    s->sensor_fd = open(sensor_dev, O_RDWR | O_NONBLOCK);
    if (s->sensor_fd >= 0) break;
    LOGW("waiting for sensors...");
    util::sleep_for(1000); // sleep one second
  }
  assert(s->sensor_fd >= 0);

  // *** SHUTDOWN ALL ***

  // CSIPHY: release csiphy
  struct msm_camera_csi_lane_params csi_lane_params = {0};
  csi_lane_params.csi_lane_mask = 0x1f;
  csiphy_cfg_data csiphy_cfg_data = { .cfg.csi_lane_params = &csi_lane_params, .cfgtype = CSIPHY_RELEASE};
  int err = cam_ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data, "release csiphy");

  // CSID: release csid
  csid_cfg_data.cfgtype = CSID_RELEASE;
  cam_ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data, "release csid");

  // SENSOR: send power down
  struct sensorb_cfg_data sensorb_cfg_data = {.cfgtype = CFG_POWER_DOWN};
  cam_ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data, "sensor power down");

  // actuator powerdown
  actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERDOWN;
  cam_ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data, "actuator powerdown");

  // reset isp
  // struct msm_vfe_axi_halt_cmd halt_cmd = {
  //   .stop_camif = 1,
  //   .overflow_detected = 1,
  //   .blocking_halt = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_HALT, &halt_cmd);
  // printf("axi halt: %d\n", err);

  // struct msm_vfe_axi_reset_cmd reset_cmd = {
  //   .blocking = 1,
  //   .frame_id = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_RESET, &reset_cmd);
  // printf("axi reset: %d\n", err);

  // struct msm_vfe_axi_restart_cmd restart_cmd = {
  //   .enable_camif = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_RESTART, &restart_cmd);
  // printf("axi restart: %d\n", err);

  // **** GO GO GO ****
  LOG("******************** GO GO GO ************************");

  // CSID: init csid
  csid_cfg_data.cfgtype = CSID_INIT;
  cam_ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data, "init csid");

  // CSIPHY: init csiphy
  csiphy_cfg_data = {.cfgtype = CSIPHY_INIT};
  cam_ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data, "init csiphy");

  // SENSOR: stop stream
  struct msm_camera_i2c_reg_setting stop_settings = {
    .reg_setting = stop_reg_array,
    .size = std::size(stop_reg_array),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = MSM_CAMERA_I2C_BYTE_DATA,
    .delay = 0
  };
  sensorb_cfg_data.cfgtype = CFG_SET_STOP_STREAM_SETTING;
  sensorb_cfg_data.cfg.setting = &stop_settings;
  cam_ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data, "stop stream");

  // SENSOR: send power up
  sensorb_cfg_data = {.cfgtype = CFG_POWER_UP};
  cam_ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data, "sensor power up");

  // **** configure the sensor ****

  // SENSOR: send i2c configuration
  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, init_array_imx298, std::size(init_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    err = sensor_write_regs(s, init_array_ov8865, std::size(init_array_ov8865), MSM_CAMERA_I2C_BYTE_DATA);
  } else {
    assert(false);
  }
  LOG("sensor init i2c: %d", err);

  if (is_road_cam) {
    // init the actuator
    actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERUP;
    cam_ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data, "actuator powerup");

    actuator_cfg_data.cfgtype = CFG_ACTUATOR_INIT;
    cam_ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data, "actuator init");

    struct msm_actuator_reg_params_t actuator_reg_params[] = {
      {
        .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
        // MSB here at address 3
        .reg_addr = 3,
        .data_type = 9,
        .addr_type = 4,
      },
    };

    struct reg_settings_t actuator_init_settings[] = {
      { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=1, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },   // PD = power down
      { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=0, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },   // 0 = power up
      { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=2, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },   // RING = SAC mode
      { .reg_addr=6, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=64, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },  // 0x40 = SAC3 mode
      { .reg_addr=7, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=113, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },
      // 0x71 = DIV1 | DIV0 | SACT0 -- Tvib x 1/4 (quarter)
      // SAC Tvib = 6.3 ms + 0.1 ms = 6.4 ms / 4 = 1.6 ms
      // LSC 1-step = 252 + 1*4 = 256 ms / 4 = 64 ms
    };

    struct region_params_t region_params[] = {
      {.step_bound = {238, 0,}, .code_per_step = 235, .qvalue = 128}
    };

    actuator_cfg_data.cfgtype = CFG_SET_ACTUATOR_INFO;
    actuator_cfg_data.cfg.set_info = (struct msm_actuator_set_info_t){
      .actuator_params = {
        .act_type = ACTUATOR_BIVCM,
        .reg_tbl_size = 1,
        .data_size = 10,
        .init_setting_size = 5,
        .i2c_freq_mode = I2C_STANDARD_MODE,
        .i2c_addr = 24,
        .i2c_addr_type = MSM_ACTUATOR_BYTE_ADDR,
        .i2c_data_type = MSM_ACTUATOR_WORD_DATA,
        .reg_tbl_params = &actuator_reg_params[0],
        .init_settings = &actuator_init_settings[0],
        .park_lens = {.damping_step = 1023, .damping_delay = 14000, .hw_params = 11, .max_step = 20},
      },
      .af_tuning_params = {
        .initial_code = INFINITY_DAC,
        .pwd_step = 0,
        .region_size = 1,
        .total_steps = 238,
        .region_params = &region_params[0],
      },
    };

    cam_ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data, "actuator set info");
  }

  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, mode_setting_array_imx298, std::size(mode_setting_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
    LOG("sensor setup: %d", err);
  }

  // CSIPHY: configure csiphy
  struct msm_camera_csiphy_params csiphy_params = {};
  if (s->camera_id == CAMERA_ID_IMX298) {
    csiphy_params = {.lane_cnt = 4, .settle_cnt = 14, .lane_mask = 0x1f, .csid_core = 0};
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    // guess!
    csiphy_params = {.lane_cnt = 4, .settle_cnt = 24, .lane_mask = 0x1f, .csid_core = 2};
  }
  csiphy_cfg_data.cfgtype = CSIPHY_CFG;
  csiphy_cfg_data.cfg.csiphy_params = &csiphy_params;
  cam_ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data, "csiphy configure");

  // CSID: configure csid
#define CSI_STATS 0x35
#define CSI_PD 0x36
  struct msm_camera_csid_params csid_params = {
    .lane_cnt = 4,
    .lane_assign = 0x4320,
    .phy_sel = (uint8_t)(is_road_cam ? 0 : 2),
    .lut_params.num_cid = (uint8_t)(is_road_cam ? 3 : 1),
    .lut_params.vc_cfg_a = {
      {.cid = 0, .dt = CSI_RAW10, .decode_format = CSI_DECODE_10BIT},
      {.cid = 1, .dt = CSI_PD, .decode_format = CSI_DECODE_10BIT},
      {.cid = 2, .dt = CSI_STATS, .decode_format = CSI_DECODE_10BIT},
    },
  };

  csid_params.lut_params.vc_cfg[0] = &csid_params.lut_params.vc_cfg_a[0];
  csid_params.lut_params.vc_cfg[1] = &csid_params.lut_params.vc_cfg_a[1];
  csid_params.lut_params.vc_cfg[2] = &csid_params.lut_params.vc_cfg_a[2];

  csid_cfg_data.cfgtype = CSID_CFG;
  csid_cfg_data.cfg.csid_params = &csid_params;
  cam_ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data, "csid configure");

  // ISP: SMMU_ATTACH
  msm_vfe_smmu_attach_cmd smmu_attach_cmd = {.security_mode = 0, .iommu_attach_mode = IOMMU_ATTACH};
  cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_SMMU_ATTACH, &smmu_attach_cmd, "isp smmu attach");

  // ******************* STREAM RAW *****************************

  // configure QMET input
  struct msm_vfe_input_cfg input_cfg = {};
  for (int i = 0; i < (is_road_cam ? 3 : 1); i++) {
    StreamState *ss = &s->ss[i];

    memset(&input_cfg, 0, sizeof(struct msm_vfe_input_cfg));
    input_cfg.input_src = (msm_vfe_input_src)(VFE_RAW_0+i);
    input_cfg.input_pix_clk = s->pixel_clock;
    input_cfg.d.rdi_cfg.cid = i;
    input_cfg.d.rdi_cfg.frame_based = 1;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_INPUT_CFG, &input_cfg);
    LOG("configure input(%d): %d", i, err);

    // ISP: REQUEST_STREAM
    ss->stream_req.axi_stream_handle = 0;
    if (is_road_cam) {
      ss->stream_req.session_id = 2;
      ss->stream_req.stream_id = /*ISP_META_CHANNEL_BIT | */ISP_NATIVE_BUF_BIT | (1+i);
    } else {
      ss->stream_req.session_id = 3;
      ss->stream_req.stream_id = ISP_NATIVE_BUF_BIT | 1;
    }

    if (i == 0) {
      ss->stream_req.output_format = v4l2_fourcc('R', 'G', '1', '0');
    } else {
      ss->stream_req.output_format = v4l2_fourcc('Q', 'M', 'E', 'T');
    }
    ss->stream_req.stream_src = (msm_vfe_axi_stream_src)(RDI_INTF_0+i);

#ifdef HIGH_FPS
    if (is_road_cam) {
      ss->stream_req.frame_skip_pattern = EVERY_3FRAME;
    }
#endif

    ss->stream_req.frame_base = 1;
    ss->stream_req.buf_divert = 1; //i == 0;

    // setup stream plane. doesn't even matter?
    /*s->stream_req.plane_cfg[0].output_plane_format = Y_PLANE;
    s->stream_req.plane_cfg[0].output_width = s->ci.frame_width;
    s->stream_req.plane_cfg[0].output_height = s->ci.frame_height;
    s->stream_req.plane_cfg[0].output_stride = s->ci.frame_width;
    s->stream_req.plane_cfg[0].output_scan_lines = s->ci.frame_height;
    s->stream_req.plane_cfg[0].rdi_cid = 0;*/

    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_REQUEST_STREAM, &ss->stream_req);
    LOG("isp request stream: %d -> 0x%x", err, ss->stream_req.axi_stream_handle);

    // ISP: REQUEST_BUF
    ss->buf_request.session_id = ss->stream_req.session_id;
    ss->buf_request.stream_id = ss->stream_req.stream_id;
    ss->buf_request.num_buf = FRAME_BUF_COUNT;
    ss->buf_request.buf_type = ISP_PRIVATE_BUF;
    ss->buf_request.handle = 0;
    cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_REQUEST_BUF, &ss->buf_request, "isp request buf");
    LOG("got buf handle: 0x%x", ss->buf_request.handle);

    // ENQUEUE all buffers
    for (int j = 0; j < ss->buf_request.num_buf; j++) {
      ss->qbuf_info[j].handle = ss->buf_request.handle;
      ss->qbuf_info[j].buf_idx = j;
      ss->qbuf_info[j].buffer.num_planes = 1;
      ss->qbuf_info[j].buffer.planes[0].addr = ss->bufs[j].fd;
      ss->qbuf_info[j].buffer.planes[0].length = ss->bufs[j].len;
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &ss->qbuf_info[j]);
    }

    // ISP: UPDATE_STREAM
    struct msm_vfe_axi_stream_update_cmd update_cmd = {};
    update_cmd.num_streams = 1;
    update_cmd.update_info[0].user_stream_id = ss->stream_req.stream_id;
    update_cmd.update_info[0].stream_handle = ss->stream_req.axi_stream_handle;
    update_cmd.update_type = UPDATE_STREAM_ADD_BUFQ;
    cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_UPDATE_STREAM, &update_cmd, "isp update stream");
  }

  LOG("******** START STREAMS ********");

  sub.id = 0;
  sub.type = 0x1ff;
  cam_ioctl(s->isp_fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "isp subscribe");

  // ISP: START_STREAM
  s->stream_cfg.cmd = START_STREAM;
  s->stream_cfg.num_streams = is_road_cam ? 3 : 1;
  for (int i = 0; i < s->stream_cfg.num_streams; i++) {
    s->stream_cfg.stream_handle[i] = s->ss[i].stream_req.axi_stream_handle;
  }
  cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg, "isp start stream");
}

static void road_camera_start(CameraState *s) {
  set_exposure(s, 1.0, 1.0);

  int err = sensor_write_regs(s, start_reg_array, std::size(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);

  int inf_step = 512 - INFINITY_DAC;

  // initial guess
  s->lens_true_pos = 400;

  // reset lens position
  struct msm_actuator_cfg_data actuator_cfg_data = {};
  actuator_cfg_data.cfgtype = CFG_SET_POSITION;
  actuator_cfg_data.cfg.setpos = (struct msm_actuator_set_position_t){
    .number_of_steps = 1,
    .hw_params = (uint32_t)7,
    .pos = {INFINITY_DAC, 0},
    .delay = {0,}
  };
  cam_ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data, "actuator set pos");

  // TODO: confirm this isn't needed
  /*memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = 0,
    .sign_dir = 1,
    .dest_step_pos = inf_step,
    .num_steps = inf_step,
    .curr_lens_pos = 0,
    .ringing_params = &actuator_ringing_params,
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data); // should be ~332 at startup ?
  LOG("init actuator move focus: %d", err);*/
  //actuator_cfg_data.cfg.move.curr_lens_pos;

  s->cur_lens_pos = 0;
  s->cur_step_pos = inf_step;

  actuator_move(s, s->cur_lens_pos);
  LOG("init lens pos: %d", s->cur_lens_pos);
}

void actuator_move(CameraState *s, uint16_t target) {
  // LP3 moves only on even positions. TODO: use proper sensor params

  // focus on infinity assuming phone is perpendicular
  static struct damping_params_t actuator_ringing_params = {
      .damping_step = 1023,
      .damping_delay = 20000,
      .hw_params = 13,
  };

  int step = (target - s->cur_lens_pos) / 2;

  int dest_step_pos = s->cur_step_pos + step;
  dest_step_pos = std::clamp(dest_step_pos, 0, 255);

  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = (int8_t)((step > 0) ? MOVE_NEAR : MOVE_FAR),
    .sign_dir = (int8_t)((step > 0) ? MSM_ACTUATOR_MOVE_SIGNED_NEAR : MSM_ACTUATOR_MOVE_SIGNED_FAR),
    .dest_step_pos = (int16_t)dest_step_pos,
    .num_steps = abs(step),
    .curr_lens_pos = s->cur_lens_pos,
    .ringing_params = &actuator_ringing_params,
  };
  ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);

  s->cur_step_pos = dest_step_pos;
  s->cur_lens_pos = actuator_cfg_data.cfg.move.curr_lens_pos;
  //LOGD("step %d   target: %d  lens pos: %d", dest_step_pos, target, s->cur_lens_pos);
}

static void parse_autofocus(CameraState *s, uint8_t *d) {
  int good_count = 0;
  int16_t max_focus = -32767;
  int avg_focus = 0;

  /*printf("FOCUS: ");
  for (int i = 0; i < 0x10; i++) {
    printf("%2.2X ", d[i]);
  }*/

  for (int i = 0; i < NUM_FOCUS; i++) {
    int doff = i*5+5;
    s->confidence[i] = d[doff];
    // this should just be a 10-bit signed int instead of 11
    // TODO: write it in a nicer way
    int16_t focus_t = (d[doff+1] << 3) | (d[doff+2] >> 5);
    if (focus_t >= 1024) focus_t = -(2048-focus_t);
    s->focus[i] = focus_t;
    //printf("%x->%d ", d[doff], focus_t);
    if (s->confidence[i] > 0x20) {
      good_count++;
      max_focus = std::max(max_focus, s->focus[i]);
      avg_focus += s->focus[i];
    }
  }
  // self recover override
  if (s->self_recover > 1) {
    s->focus_err = 200 * ((s->self_recover % 2 == 0) ? 1:-1); // far for even numbers, close for odd
    s->self_recover -= 2;
    return;
  }

  if (good_count < 4) {
    s->focus_err = nan("");
    return;
  }

  avg_focus /= good_count;

  // outlier rejection
  if (abs(avg_focus - max_focus) > 200) {
    s->focus_err = nan("");
    return;
  }

  s->focus_err = max_focus*1.0;
}

static std::optional<float> get_accel_z(SubMaster *sm) {
  sm->update(0);
  if(sm->updated("sensorEvents")) {
    for (auto event : (*sm)["sensorEvents"].getSensorEvents()) {
      if (event.which() == cereal::SensorEventData::ACCELERATION) {
        if (auto v = event.getAcceleration().getV(); v.size() >= 3)
          return -v[2];
        break;
      }
    }
  }
  return std::nullopt;
}

static void do_autofocus(CameraState *s, SubMaster *sm) {
  float lens_true_pos = s->lens_true_pos.load();
  if (!isnan(s->focus_err)) {
    // learn lens_true_pos
    const float focus_kp = 0.005;
    lens_true_pos -= s->focus_err*focus_kp;
  }

  if (auto accel_z = get_accel_z(sm)) {
    s->last_sag_acc_z = *accel_z;
  }
  const float sag = (s->last_sag_acc_z / 9.8) * 128;
  // stay off the walls
  lens_true_pos = std::clamp(lens_true_pos, float(LP3_AF_DAC_DOWN), float(LP3_AF_DAC_UP));
  int target = std::clamp(lens_true_pos - sag, float(LP3_AF_DAC_DOWN), float(LP3_AF_DAC_UP));
  s->lens_true_pos.store(lens_true_pos);

  /*char debug[4096];
  char *pdebug = debug;
  pdebug += sprintf(pdebug, "focus ");
  for (int i = 0; i < NUM_FOCUS; i++) pdebug += sprintf(pdebug, "%2x(%4d) ", s->confidence[i], s->focus[i]);
  pdebug += sprintf(pdebug, "  err: %7.2f  offset: %6.2f sag: %6.2f lens_true_pos: %6.2f  cur_lens_pos: %4d->%4d", err * focus_kp, offset, sag, s->lens_true_pos, s->cur_lens_pos, target);
  LOGD(debug);*/

  actuator_move(s, target);
}

void camera_autoexposure(CameraState *s, float grey_frac) {
  if (s->camera_num == 0) {
    CameraExpInfo tmp = road_cam_exp.load();
    tmp.op_id++;
    tmp.grey_frac = grey_frac;
    road_cam_exp.store(tmp);
  } else {
    CameraExpInfo tmp = driver_cam_exp.load();
    tmp.op_id++;
    tmp.grey_frac = grey_frac;
    driver_cam_exp.store(tmp);
  }
}

static void driver_camera_start(CameraState *s) {
  set_exposure(s, 1.0, 1.0);
  int err = sensor_write_regs(s, start_reg_array, std::size(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);
}

void cameras_open(MultiCameraState *s) {
  struct msm_ispif_param_data ispif_params = {
    .num = 4,
    .entries = {
      // road camera
      {.vfe_intf = VFE0, .intftype = RDI0, .num_cids = 1, .cids[0] = CID0, .csid = CSID0},
      // driver camera
      {.vfe_intf = VFE1, .intftype = RDI0, .num_cids = 1, .cids[0] = CID0, .csid = CSID2},
      // road camera (focus)
      {.vfe_intf = VFE0, .intftype = RDI1, .num_cids = 1, .cids[0] = CID1, .csid = CSID0},
      // road camera (stats, for AE)
      {.vfe_intf = VFE0, .intftype = RDI2, .num_cids = 1, .cids[0] = CID2, .csid = CSID0},
    },
  };
  s->msmcfg_fd = open("/dev/media0", O_RDWR | O_NONBLOCK);
  assert(s->msmcfg_fd >= 0);

  sensors_init(s);

  s->v4l_fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
  assert(s->v4l_fd >= 0);

  s->ispif_fd = open("/dev/v4l-subdev15", O_RDWR | O_NONBLOCK);
  assert(s->ispif_fd >= 0);

  // ISPIF: stop
  // memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  // ispif_cfg_data.cfg_type = ISPIF_STOP_FRAME_BOUNDARY;
  // ispif_cfg_data.params = ispif_params;
  // err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  // LOG("ispif stop: %d", err);

  LOG("*** open driver camera ***");
  s->driver_cam.ss[0].bufs = s->driver_cam.buf.camera_bufs.get();
  camera_open(&s->driver_cam, false);

  LOG("*** open road camera ***");
  s->road_cam.ss[0].bufs = s->road_cam.buf.camera_bufs.get();
  s->road_cam.ss[1].bufs = s->focus_bufs;
  s->road_cam.ss[2].bufs = s->stats_bufs;
  camera_open(&s->road_cam, true);

  if (getenv("CAMERA_TEST")) {
    cameras_close(s);
    exit(0);
  }

  // ISPIF: set vfe info
  struct ispif_cfg_data ispif_cfg_data = {.cfg_type = ISPIF_SET_VFE_INFO, .vfe_info.num_vfe = 2};
  int err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif set vfe info: %d", err);

  // ISPIF: setup
  ispif_cfg_data = {.cfg_type = ISPIF_INIT, .csid_version = 0x30050000 /* CSID_VERSION_V35*/};
  cam_ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data, "ispif setup");

  ispif_cfg_data = {.cfg_type = ISPIF_CFG, .params = ispif_params};
  cam_ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data, "ispif cfg");

  ispif_cfg_data.cfg_type = ISPIF_START_FRAME_BOUNDARY;
  cam_ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data, "ispif start_frame_boundary");

  driver_camera_start(&s->driver_cam);
  road_camera_start(&s->road_cam);
}


static void camera_close(CameraState *s) {
  // ISP: STOP_STREAM
  s->stream_cfg.cmd = STOP_STREAM;
  cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg, "isp stop stream");

  for (int i = 0; i < 3; i++) {
    StreamState *ss = &s->ss[i];
    if (ss->stream_req.axi_stream_handle != 0) {
      cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_BUF, &ss->buf_request, "isp release buf");

      struct msm_vfe_axi_stream_release_cmd stream_release = {
        .stream_handle = ss->stream_req.axi_stream_handle,
      };
      cam_ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_STREAM, &stream_release, "isp release stream");
    }
  }
}

const char* get_isp_event_name(uint32_t type) {
  switch (type) {
  case ISP_EVENT_REG_UPDATE: return "ISP_EVENT_REG_UPDATE";
  case ISP_EVENT_EPOCH_0: return "ISP_EVENT_EPOCH_0";
  case ISP_EVENT_EPOCH_1: return "ISP_EVENT_EPOCH_1";
  case ISP_EVENT_START_ACK: return "ISP_EVENT_START_ACK";
  case ISP_EVENT_STOP_ACK: return "ISP_EVENT_STOP_ACK";
  case ISP_EVENT_IRQ_VIOLATION: return "ISP_EVENT_IRQ_VIOLATION";
  case ISP_EVENT_STATS_OVERFLOW: return "ISP_EVENT_STATS_OVERFLOW";
  case ISP_EVENT_ERROR: return "ISP_EVENT_ERROR";
  case ISP_EVENT_SOF: return "ISP_EVENT_SOF";
  case ISP_EVENT_EOF: return "ISP_EVENT_EOF";
  case ISP_EVENT_BUF_DONE: return "ISP_EVENT_BUF_DONE";
  case ISP_EVENT_BUF_DIVERT: return "ISP_EVENT_BUF_DIVERT";
  case ISP_EVENT_STATS_NOTIFY: return "ISP_EVENT_STATS_NOTIFY";
  case ISP_EVENT_COMP_STATS_NOTIFY: return "ISP_EVENT_COMP_STATS_NOTIFY";
  case ISP_EVENT_FE_READ_DONE: return "ISP_EVENT_FE_READ_DONE";
  case ISP_EVENT_IOMMU_P_FAULT: return "ISP_EVENT_IOMMU_P_FAULT";
  case ISP_EVENT_HW_FATAL_ERROR: return "ISP_EVENT_HW_FATAL_ERROR";
  case ISP_EVENT_PING_PONG_MISMATCH: return "ISP_EVENT_PING_PONG_MISMATCH";
  case ISP_EVENT_REG_UPDATE_MISSING: return "ISP_EVENT_REG_UPDATE_MISSING";
  case ISP_EVENT_BUF_FATAL_ERROR: return "ISP_EVENT_BUF_FATAL_ERROR";
  case ISP_EVENT_STREAM_UPDATE_DONE: return "ISP_EVENT_STREAM_UPDATE_DONE";
  default: return "unknown";
  }
}

static FrameMetadata get_frame_metadata(CameraState *s, uint32_t frame_id) {
  std::lock_guard lk(s->frame_info_lock);
  for (auto &i : s->frame_metadata) {
    if (i.frame_id == frame_id) {
      return i;
    }
  }
  // should never happen
  return (FrameMetadata){
    .frame_id = (uint32_t)-1,
  };
}

static void ops_thread(MultiCameraState *s) {
  int last_road_cam_op_id = 0;
  int last_driver_cam_op_id = 0;

  CameraExpInfo road_cam_op;
  CameraExpInfo driver_cam_op;

  set_thread_name("camera_settings");
  SubMaster sm({"sensorEvents"});
  while(!do_exit) {
    road_cam_op = road_cam_exp.load();
    if (road_cam_op.op_id != last_road_cam_op_id) {
      do_autoexposure(&s->road_cam, road_cam_op.grey_frac);
      do_autofocus(&s->road_cam, &sm);
      last_road_cam_op_id = road_cam_op.op_id;
    }

    driver_cam_op = driver_cam_exp.load();
    if (driver_cam_op.op_id != last_driver_cam_op_id) {
      do_autoexposure(&s->driver_cam, driver_cam_op.grey_frac);
      last_driver_cam_op_id = driver_cam_op.op_id;
    }

    util::sleep_for(50);
  }
}

static void setup_self_recover(CameraState *c, const uint16_t *lapres, size_t lapres_size) {
  const float lens_true_pos = c->lens_true_pos.load();
  int self_recover = c->self_recover.load();
  if (self_recover < 2 && (lens_true_pos < (LP3_AF_DAC_DOWN + 1) || lens_true_pos > (LP3_AF_DAC_UP - 1)) && is_blur(lapres, lapres_size)) {
    // truly stuck, needs help
    if (--self_recover < -FOCUS_RECOVER_PATIENCE) {
      LOGD("road camera bad state detected. attempting recovery from %.1f, recover state is %d", lens_true_pos, self_recover);
      // parity determined by which end is stuck at
      self_recover = FOCUS_RECOVER_STEPS + (lens_true_pos < LP3_AF_DAC_M ? 1 : 0);
    }
  } else if (self_recover < 2 && (lens_true_pos < (LP3_AF_DAC_M - LP3_AF_DAC_3SIG) || lens_true_pos > (LP3_AF_DAC_M + LP3_AF_DAC_3SIG))) {
    // in suboptimal position with high prob, but may still recover by itself
    if (--self_recover < -(FOCUS_RECOVER_PATIENCE * 3)) {
      self_recover = FOCUS_RECOVER_STEPS / 2 + (lens_true_pos < LP3_AF_DAC_M ? 1 : 0);
    }
  } else if (self_recover < 0) {
    self_recover += 1;  // reset if fine
  }
  c->self_recover.store(self_recover);
}

void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  common_process_driver_camera(s->sm, s->pm, c, cnt);
}

// called by processing_thread
void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  const int roi_id = cnt % std::size(s->lapres);  // rolling roi
  s->lapres[roi_id] = s->lap_conv->Update(b->q, (uint8_t *)b->cur_rgb_buf->addr, roi_id);
  setup_self_recover(c, &s->lapres[0], std::size(s->lapres));

  MessageBuilder msg;
  auto framed = msg.initEvent().initRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  if (env_send_road) {
    framed.setImage(get_frame_image(b));
  }
  framed.setFocusVal(s->road_cam.focus);
  framed.setFocusConf(s->road_cam.confidence);
  framed.setRecoverState(s->road_cam.self_recover);
  framed.setSharpnessScore(s->lapres);
  framed.setTransform(b->yuv_transform.v);
  s->pm->send("roadCameraState", msg);

  if (cnt % 3 == 0) {
    const int x = 290, y = 322, width = 560, height = 314;
    const int skip = 1;
    camera_autoexposure(c, set_exposure_target(b, x, x + width, skip, y, y + height, skip, -1, false, false));
  }
}

void cameras_run(MultiCameraState *s) {
  std::vector<std::thread> threads;
  threads.push_back(std::thread(ops_thread, s));
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));

  CameraState* cameras[2] = {&s->road_cam, &s->driver_cam};

  while (!do_exit) {
    struct pollfd fds[2] = {{.fd = cameras[0]->isp_fd, .events = POLLPRI},
                            {.fd = cameras[1]->isp_fd, .events = POLLPRI}};
    int ret = poll(fds, std::size(fds), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    // process cameras
    for (int i=0; i<2; i++) {
      if (!fds[i].revents) continue;

      CameraState *c = cameras[i];

      struct v4l2_event ev = {};
      ret = ioctl(c->isp_fd, VIDIOC_DQEVENT, &ev);
      const msm_isp_event_data *isp_event_data = (const msm_isp_event_data *)ev.u.data;

      if (ev.type == ISP_EVENT_BUF_DIVERT) {
        const int buf_idx = isp_event_data->u.buf_done.buf_idx;
        const int buffer = (isp_event_data->u.buf_done.stream_id & 0xFFFF) - 1;
        if (buffer == 0) {
          c->buf.camera_bufs_metadata[buf_idx] = get_frame_metadata(c, isp_event_data->frame_id);
          c->buf.queue(buf_idx);
        } else {
          auto &ss = c->ss[buffer];
          if (buffer == 1) {
            parse_autofocus(c, (uint8_t *)(ss.bufs[buf_idx].addr));
          }
          ss.qbuf_info[buf_idx].dirty_buf = 1;
          ioctl(c->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &ss.qbuf_info[buf_idx]);
        }

      } else if (ev.type == ISP_EVENT_EOF) {
        const uint64_t timestamp = (isp_event_data->mono_timestamp.tv_sec * 1000000000ULL + isp_event_data->mono_timestamp.tv_usec * 1000);
        std::lock_guard lk(c->frame_info_lock);
        c->frame_metadata[c->frame_metadata_idx] = (FrameMetadata){
            .frame_id = isp_event_data->frame_id,
            .timestamp_eof = timestamp,
            .frame_length = (uint32_t)c->frame_length,
            .integ_lines = (uint32_t)c->cur_integ_lines,
            .lens_pos = c->cur_lens_pos,
            .lens_sag = c->last_sag_acc_z,
            .lens_err = c->focus_err,
            .lens_true_pos = c->lens_true_pos,
            .gain = c->cur_gain_frac,
            .measured_grey_fraction = c->measured_grey_fraction,
            .target_grey_fraction = c->target_grey_fraction,
            .high_conversion_gain = false,
        };
        c->frame_metadata_idx = (c->frame_metadata_idx + 1) % METADATA_BUF_COUNT;

      } else if (ev.type == ISP_EVENT_ERROR) {
        LOGE("ISP_EVENT_ERROR! err type: 0x%08x", isp_event_data->u.error_info.err_type);
      }
    }
  }

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->road_cam);
  camera_close(&s->driver_cam);
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    s->focus_bufs[i].free();
    s->stats_bufs[i].free();
  }

  delete s->lap_conv;
  delete s->sm;
  delete s->pm;
}
