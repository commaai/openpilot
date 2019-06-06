#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>

#include <linux/media.h>

#include <cutils/properties.h>

#include <pthread.h>
#include <czmq.h>

#include "msmb_isp.h"
#include "msmb_ispif.h"
#include "msmb_camera.h"
#include "msm_cam_sensor.h"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "common/params.h"

#include "cereal/gen/c/log.capnp.h"

#include "sensor_i2c.h"

#include "camera_qcom.h"


// enable this to run the camera at 60fps and sample every third frame
// supposed to reduce 33ms of lag, but no results
//#define HIGH_FPS

#define CAMERA_MSG_AUTOEXPOSE 0

typedef struct CameraMsg {
  int type;
  int camera_num;

  float grey_frac;
} CameraMsg;

extern volatile int do_exit;

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
    .frame_width = 2328,
    .frame_height = 1748,
    .frame_stride = 2912,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = true
  },
  [CAMERA_ID_IMX179] = {
    .frame_width = 3280,
    .frame_height = 2464,
    .frame_stride = 4104,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = false
  },
  [CAMERA_ID_S5K3P8SP] = {
    .frame_width = 2304,
    .frame_height = 1728,
    .frame_stride = 2880,
    .bayer = true,
    .bayer_flip = 1,
    .hdr = false
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
  CameraState *s = cookie;
  // printf("camera_release_buffer %d\n", buf_idx);
  s->ss[0].qbuf_info[buf_idx].dirty_buf = 1;
  ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &s->ss[0].qbuf_info[buf_idx]);
}

static void camera_init(CameraState *s, int camera_id, int camera_num,
                        uint32_t pixel_clock, uint32_t line_length_pclk,
                        unsigned int max_gain, unsigned int fps) {
  memset(s, 0, sizeof(*s));

  s->camera_num = camera_num;
  s->camera_id = camera_id;

  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);
  s->frame_size = s->ci.frame_height * s->ci.frame_stride;

  s->pixel_clock = pixel_clock;
  s->line_length_pclk = line_length_pclk;
  s->max_gain = max_gain;
  s->fps = fps;

  zsock_t *ops_sock = zsock_new_push(">inproc://cameraops");
  assert(ops_sock);
  s->ops_sock = zsock_resolve(ops_sock);

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame",
    camera_release_buffer, s);

  pthread_mutex_init(&s->frame_info_lock, NULL);
}


int sensor_write_regs(CameraState *s, struct msm_camera_i2c_reg_array* arr, size_t size, int data_type) {
  struct msm_camera_i2c_reg_setting out_settings = {
    .reg_setting = arr,
    .size = size,
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = data_type,
    .delay = 0,
  };
  struct sensorb_cfg_data cfg_data = {0};
  cfg_data.cfgtype = CFG_WRITE_I2C_ARRAY;
  cfg_data.cfg.setting = &out_settings;
  return ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &cfg_data);
}

static int imx298_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  int err;

  int analog_gain = min(gain, 448);

  if (gain > 448) {
    s->digital_gain = (512.0/(512-(gain))) / 8.0;
  } else {
    s->digital_gain = 1.0;
  }

  //printf("%5d/%5d %5d %f\n", s->cur_integ_lines, s->cur_frame_length, analog_gain, s->digital_gain);

  int digital_gain = 0x100;

  float white_balance[] = {0.4609375, 1.0, 0.546875};
  //float white_balance[] = {1.0, 1.0, 1.0};

  int digital_gain_gr = digital_gain / white_balance[1];
  int digital_gain_gb = digital_gain / white_balance[1];
  int digital_gain_r = digital_gain / white_balance[0];
  int digital_gain_b = digital_gain / white_balance[2];

  struct msm_camera_i2c_reg_array reg_array[] = {
    // REG_HOLD
    {0x104,0x1,0},
    {0x3002,0x0,0}, // long autoexposure off

    // FRM_LENGTH
    {0x340, frame_length >> 8, 0}, {0x341, frame_length & 0xff, 0},
    // INTEG_TIME aka coarse_int_time_addr aka shutter speed
    {0x202, integ_lines >> 8, 0}, {0x203, integ_lines & 0xff,0},
    // global_gain_addr
    // if you assume 1x gain is 32, 448 is 14x gain, aka 2^14=16384
    {0x204, analog_gain >> 8, 0}, {0x205, analog_gain & 0xff,0},

    // digital gain for colors: gain_greenR, gain_red, gain_blue, gain_greenB
    /*{0x20e, digital_gain_gr >> 8, 0}, {0x20f,digital_gain_gr & 0xFF,0},
    {0x210, digital_gain_r >> 8, 0}, {0x211,digital_gain_r & 0xFF,0},
    {0x212, digital_gain_b >> 8, 0}, {0x213,digital_gain_b & 0xFF,0},
    {0x214, digital_gain_gb >> 8, 0}, {0x215,digital_gain_gb & 0xFF,0},*/

    // REG_HOLD
    {0x104,0x0,0},
  };

  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

static inline int ov8865_get_coarse_gain(int gain) {
  static const int gains[] = {0, 256, 384, 448, 480};
  int i;

  for (i = 1; i < ARRAYSIZE(gains); i++) {
    if (gain >= gains[i - 1] && gain < gains[i])
      break;
  }

  return i - 1;
}

static int ov8865_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  //printf("front camera: %d %d %d\n", gain, integ_lines, frame_length);
  int err, gain_bitmap;
  gain_bitmap = (1 << ov8865_get_coarse_gain(gain)) - 1;
  integ_lines *= 16; // The exposure value in reg is in 16ths of a line
  struct msm_camera_i2c_reg_array reg_array[] = {
    //{0x104,0x1,0},

    // FRM_LENGTH
    {0x380e, frame_length >> 8, 0}, {0x380f, frame_length & 0xff, 0},
    // AEC EXPO
    {0x3500, integ_lines >> 16, 0}, {0x3501, integ_lines >> 8, 0}, {0x3502, integ_lines & 0xff,0},
    // AEC MANUAL
    {0x3503, 0x4, 0},
    // AEC GAIN
    {0x3508, gain_bitmap, 0}, {0x3509, 0xf8, 0},

    //{0x104,0x0,0},
  };
  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

static int imx179_s5k3p8sp_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  //printf("front camera: %d %d %d\n", gain, integ_lines, frame_length);
  int err;

  if (gain > 448) {
    s->digital_gain = (512.0/(512-(gain))) / 8.0;
  } else {
    s->digital_gain = 1.0;
  }

  struct msm_camera_i2c_reg_array reg_array[] = {
    {0x104,0x1,0},

    // FRM_LENGTH
    {0x340, frame_length >> 8, 0}, {0x341, frame_length & 0xff, 0},
    // coarse_int_time
    {0x202, integ_lines >> 8, 0}, {0x203, integ_lines & 0xff,0},
    // global_gain
    {0x204, gain >> 8, 0}, {0x205, gain & 0xff,0},

    {0x104,0x0,0},
  };
  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

void cameras_init(DualCameraState *s) {
  memset(s, 0, sizeof(*s));

  char project_name[1024] = {0};
  property_get("ro.boot.project_name", project_name, "");

  char product_name[1024] = {0};
  property_get("ro.product.name", product_name, "");

  if (strlen(project_name) == 0) {
    LOGD("LePro 3 op system detected");
    s->device = DEVICE_LP3;

    // sensor is flipped in LP3
    // IMAGE_ORIENT = 3
    init_array_imx298[0].reg_data = 3;
    cameras_supported[CAMERA_ID_IMX298].bayer_flip = 3;
  } else if (strcmp(product_name, "OnePlus3") == 0 && strcmp(project_name, "15811") != 0) {
    // no more OP3 support
    s->device = DEVICE_OP3;
    assert(false);
  } else if (strcmp(product_name, "OnePlus3") == 0 && strcmp(project_name, "15811") == 0) {
    // only OP3T support
    s->device = DEVICE_OP3T;
  } else {
    assert(false);
  }

  // 0   = ISO 100
  // 256 = ISO 200
  // 384 = ISO 400
  // 448 = ISO 800
  // 480 = ISO 1600
  // 496 = ISO 3200
  // 504 = ISO 6400, 8x digital gain
  // 508 = ISO 12800, 16x digital gain
  // 510 = ISO 25600, 32x digital gain

  camera_init(&s->rear, CAMERA_ID_IMX298, 0,
              /*pixel_clock=*/600000000, /*line_length_pclk=*/5536,
              /*max_gain=*/510, //0 (ISO 100)- 448 (ISO 800, max analog gain) - 511 (super noisy)
#ifdef HIGH_FPS
              /*fps*/60
#else
              /*fps*/20
#endif
  );
  s->rear.apply_exposure = imx298_apply_exposure;

  if (s->device == DEVICE_OP3T) {
    camera_init(&s->front, CAMERA_ID_S5K3P8SP, 1,
                /*pixel_clock=*/561000000, /*line_length_pclk=*/5120,
                /*max_gain=*/510, 10);
    s->front.apply_exposure = imx179_s5k3p8sp_apply_exposure;
  } else if (s->device == DEVICE_LP3) {
    camera_init(&s->front, CAMERA_ID_OV8865, 1,
                /*pixel_clock=*/251200000, /*line_length_pclk=*/7000,
                /*max_gain=*/510, 10);
    s->front.apply_exposure = ov8865_apply_exposure;
  } else {
    camera_init(&s->front, CAMERA_ID_IMX179, 1,
                /*pixel_clock=*/251200000, /*line_length_pclk=*/3440,
                /*max_gain=*/224, 20);
    s->front.apply_exposure = imx179_s5k3p8sp_apply_exposure;
  }

  // assume the device is upside-down (not anymore)
  s->rear.transform = (mat3){{
     1.0,  0.0, 0.0,
     0.0,  1.0, 0.0,
     0.0,  0.0, 1.0,
  }};

  // probably wrong
  s->front.transform = (mat3){{
     1.0,  0.0, 0.0,
     0.0,  1.0, 0.0,
     0.0,  0.0, 1.0,
  }};

  s->rear.device = s->device;
  s->front.device = s->device;
}

static void set_exposure(CameraState *s, float exposure_frac, float gain_frac) {
  int err = 0;

  unsigned int frame_length = s->pixel_clock / s->line_length_pclk / s->fps;

  unsigned int gain = s->cur_gain;
  unsigned int integ_lines = s->cur_integ_lines;

  if (exposure_frac >= 0) {
    exposure_frac = clamp(exposure_frac, 2.0 / frame_length, 1.0);
    integ_lines = frame_length * exposure_frac;

    // See page 79 of the datasheet, this is the max allowed (-1 for phase adjust)
    integ_lines = min(integ_lines, frame_length-11);
  }

  // done after exposure to not adjust it
  if (s->using_pll) {
    // can adjust frame length by up to +/- 1
    const int PHASE_DEADZONE = 20000;   // 20 us
    int phase_max = 1000000000 / s->fps;
    int phase_diff = s->phase_actual - s->phase_request;
    phase_diff = ((phase_diff + phase_max/2) % phase_max) - phase_max/2;

    if (phase_diff < -PHASE_DEADZONE) {
      frame_length += 1;
    } else if (phase_diff > PHASE_DEADZONE) {
      frame_length -= 1;
    }
  }

  if (gain_frac >= 0) {
    // ISO200 is minimum gain
    gain_frac = clamp(gain_frac, 1.0/64, 1.0);

    // linearize gain response
    // TODO: will be wrong for front camera
    // 0.125 -> 448
    // 0.25  -> 480
    // 0.5   -> 496
    // 1.0   -> 504
    // 512 - 512/(128*gain_frac)
    gain = (s->max_gain/510) * (512 - 512/(256*gain_frac));
  }

  if (gain != s->cur_gain
    || integ_lines != s->cur_integ_lines
    || frame_length != s->cur_frame_length) {

    if (s->apply_exposure) {
      err = s->apply_exposure(s, gain, integ_lines, frame_length);
    }

    if (err == 0) {
      pthread_mutex_lock(&s->frame_info_lock);
      s->cur_gain = gain;
      s->cur_integ_lines = integ_lines;
      s->cur_frame_length = frame_length;
      pthread_mutex_unlock(&s->frame_info_lock);
    }
  }

  if (err == 0) {
    s->cur_exposure_frac = exposure_frac;
    s->cur_gain_frac = gain_frac;
  }

  //LOGD("set exposure: %f %f - %d", exposure_frac, gain_frac, err);
}

static void do_autoexposure(CameraState *s, float grey_frac) {
  const float target_grey = 0.3;

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

void camera_autoexposure(CameraState *s, float grey_frac) {
  CameraMsg msg = {
    .type = CAMERA_MSG_AUTOEXPOSE,
    .camera_num = s->camera_num,
    .grey_frac = grey_frac,
  };

  zmq_send(s->ops_sock, &msg, sizeof(msg), ZMQ_DONTWAIT);
}

static uint8_t* get_eeprom(int eeprom_fd, size_t *out_len) {
  int err;

  struct msm_eeprom_cfg_data cfg = {0};
  cfg.cfgtype = CFG_EEPROM_GET_CAL_DATA;
  err = ioctl(eeprom_fd, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);

  uint32_t num_bytes = cfg.cfg.get_data.num_bytes;
  assert(num_bytes > 100);

  uint8_t* buffer = malloc(num_bytes);
  assert(buffer);
  memset(buffer, 0, num_bytes);

  cfg.cfgtype = CFG_EEPROM_READ_CAL_DATA;
  cfg.cfg.read_data.num_bytes = num_bytes;
  cfg.cfg.read_data.dbuffer = buffer;
  err = ioctl(eeprom_fd, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);

  *out_len = num_bytes;
  return buffer;
}

static void imx298_ois_calibration(int ois_fd, uint8_t* eeprom) {
  int err;

  const int ois_registers[][2] = {
    // == SET_FADJ_PARAM() == (factory adjustment)

    // Set Hall Current DAC
    {0x8230, *(uint16_t*)(eeprom+0x102)}, //_P_30_ADC_CH0 (CURDAT)

    // Set Hall     PreAmp Offset
    {0x8231, *(uint16_t*)(eeprom+0x104)}, //_P_31_ADC_CH1 (HALOFS_X)
    {0x8232, *(uint16_t*)(eeprom+0x106)}, //_P_32_ADC_CH2 (HALOFS_Y)

    // Set Hall-X/Y PostAmp Offset
    {0x841e, *(uint16_t*)(eeprom+0x108)}, //_M_X_H_ofs
    {0x849e, *(uint16_t*)(eeprom+0x10a)}, //_M_Y_H_ofs

    // Set Residual Offset
    {0x8239, *(uint16_t*)(eeprom+0x10c)}, //_P_39_Ch3_VAL_1 (PSTXOF)
    {0x823b, *(uint16_t*)(eeprom+0x10e)}, //_P_3B_Ch3_VAL_3 (PSTYOF)

    // DIGITAL GYRO OFFSET
    {0x8406, *(uint16_t*)(eeprom+0x110)}, //_M_Kgx00
    {0x8486, *(uint16_t*)(eeprom+0x112)}, //_M_Kgy00
    {0x846a, *(uint16_t*)(eeprom+0x120)}, //_M_TMP_X_
    {0x846b, *(uint16_t*)(eeprom+0x122)}, //_M_TMP_Y_

    // HALLSENSE
    // Set Hall Gain
    {0x8446, *(uint16_t*)(eeprom+0x114)}, //_M_KgxHG
    {0x84c6, *(uint16_t*)(eeprom+0x116)}, //_M_KgyHG
    // Set Cross Talk Canceller
    {0x8470, *(uint16_t*)(eeprom+0x124)}, //_M_KgxH0
    {0x8472, *(uint16_t*)(eeprom+0x126)}, //_M_KgyH0

    // LOOPGAIN
    {0x840f, *(uint16_t*)(eeprom+0x118)}, //_M_KgxG
    {0x848f, *(uint16_t*)(eeprom+0x11a)}, //_M_KgyG

    // Position Servo ON ( OIS OFF )
    {0x847f, 0x0c0c}, //_M_EQCTL
  };


  struct msm_ois_cfg_data cfg = {0};
  struct msm_camera_i2c_seq_reg_array ois_reg_settings[ARRAYSIZE(ois_registers)] = {{0}};
  for (int i=0; i<ARRAYSIZE(ois_registers); i++) {
    ois_reg_settings[i].reg_addr = ois_registers[i][0];
    ois_reg_settings[i].reg_data[0] = ois_registers[i][1] & 0xff;
    ois_reg_settings[i].reg_data[1] = (ois_registers[i][1] >> 8) & 0xff;
    ois_reg_settings[i].reg_data_size = 2;
  }
  struct msm_camera_i2c_seq_reg_setting ois_reg_setting = {
    .reg_setting = &ois_reg_settings[0],
    .size = ARRAYSIZE(ois_reg_settings),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .delay = 0,
  };
  cfg.cfgtype = CFG_OIS_I2C_WRITE_SEQ_TABLE;
  cfg.cfg.settings = &ois_reg_setting;
  err = ioctl(ois_fd, VIDIOC_MSM_OIS_CFG, &cfg);
  LOG("ois reg calibration: %d", err);
}




static void sensors_init(DualCameraState *s) {
  int err;

  int sensorinit_fd = -1;
  if (s->device == DEVICE_LP3) {
    sensorinit_fd = open("/dev/v4l-subdev11", O_RDWR | O_NONBLOCK);
  } else {
    sensorinit_fd = open("/dev/v4l-subdev12", O_RDWR | O_NONBLOCK);
  }
  assert(sensorinit_fd >= 0);

  struct sensor_init_cfg_data sensor_init_cfg = {0};

  // init rear sensor

  struct msm_camera_sensor_slave_info slave_info = {0};
  if (s->device == DEVICE_LP3) {
    slave_info = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx298",
      .eeprom_name = "sony_imx298",
      .actuator_name = "dw9800w",
      .ois_name = "",
      .flash_name = "pmic",
      .camera_id = 0,
      .slave_addr = 32,
      .i2c_freq_mode = 1,
      .addr_type = 2,
      .sensor_id_info = {
        .sensor_id_reg_addr = 22,
        .sensor_id = 664,
        .sensor_id_mask = 0,
        .module_id = 9,
        .vcm_id = 6,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 1,
            .seq_val = 5,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 2,
            .delay = 10,
          },
        },
        .size = 7,
        .power_down_setting_a = {
          {
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 1,
            .seq_val = 5,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 6,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = 0,
        .sensor_mount_angle = 90,
      },
      .output_format = 0,
    };
  } else {
    slave_info = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx298",
      .eeprom_name = "sony_imx298",
      .actuator_name = "rohm_bu63165gwl",
      .ois_name = "rohm_bu63165gwl",
      .camera_id = 0,
      .slave_addr = 52,
      .i2c_freq_mode = 2,
      .addr_type = 2,
      .sensor_id_info = {
        .sensor_id_reg_addr = 22,
        .sensor_id = 664,
        .sensor_id_mask = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = 1,
            .seq_val = 6,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 3,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = 2,
            .seq_val = 4,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 2,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 2,
            .delay = 2,
          },
        },
        .size = 9,
        .power_down_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 10,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 4,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 6,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },
        },
        .size_down = 8,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = 0,
        .sensor_mount_angle = 360,
      },
      .output_format = 0,
    };
  }
  slave_info.power_setting_array.power_setting =
    (struct msm_sensor_power_setting *)&slave_info.power_setting_array.power_setting_a[0];
  slave_info.power_setting_array.power_down_setting =
    (struct msm_sensor_power_setting *)&slave_info.power_setting_array.power_down_setting_a[0];
  sensor_init_cfg.cfgtype = CFG_SINIT_PROBE;
  sensor_init_cfg.cfg.setting = &slave_info;
  err = ioctl(sensorinit_fd, VIDIOC_MSM_SENSOR_INIT_CFG, &sensor_init_cfg);
  LOG("sensor init cfg (rear): %d", err);
  assert(err >= 0);


  struct msm_camera_sensor_slave_info slave_info2 = {0};
  if (s->device == DEVICE_LP3) {
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "ov8865_sunny",
      .eeprom_name = "ov8865_plus",
      .actuator_name = "",
      .ois_name = "",
      .flash_name = "",
      .camera_id = 2,
      .slave_addr = 108,
      .i2c_freq_mode = 1,
      .addr_type = 2,
      .sensor_id_info = {
        .sensor_id_reg_addr = 12299,
        .sensor_id = 34917,
        .sensor_id_mask = 0,
        .module_id = 2,
        .vcm_id = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 2,
            .delay = 1,
          },
        },
        .size = 6,
        .power_down_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = 1,
        .sensor_mount_angle = 270,
      },
      .output_format = 0,
    };
  } else if (s->front.camera_id == CAMERA_ID_S5K3P8SP) {
    // init front camera
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "s5k3p8sp",
      .eeprom_name = "s5k3p8sp_m24c64s",
      .actuator_name = "",
      .ois_name = "",
      .camera_id = 1,
      .slave_addr = 32,
      .i2c_freq_mode = 1,
      .addr_type = 2,
      .sensor_id_info = {
        .sensor_id_reg_addr = 0,
        .sensor_id = 12552,
        .sensor_id_mask = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 2,
            .delay = 1,
          },
        },
        .size = 6,
        .power_down_setting_a = {
          {
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = 1,
        .sensor_mount_angle = 270,
      },
      .output_format = 0,
    };
  } else {
    // init front camera
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx179",
      .eeprom_name = "sony_imx179",
      .actuator_name = "",
      .ois_name = "",
      .camera_id = 1,
      .slave_addr = 32,
      .i2c_freq_mode = 1,
      .addr_type = 2,
      .sensor_id_info = {
        .sensor_id_reg_addr = 2,
        .sensor_id = 377,
        .sensor_id_mask = 4095,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 0,
          },
        },
        .size = 5,
        .power_down_setting_a = {
          {
            .seq_type = 0,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 1,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = 2,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = 2,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = 2,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = 1,
        .sensor_mount_angle = 270,
      },
      .output_format = 0,
    };
  }
  slave_info2.power_setting_array.power_setting =
    (struct msm_sensor_power_setting *)&slave_info2.power_setting_array.power_setting_a[0];
  slave_info2.power_setting_array.power_down_setting =
    (struct msm_sensor_power_setting *)&slave_info2.power_setting_array.power_down_setting_a[0];
  sensor_init_cfg.cfgtype = CFG_SINIT_PROBE;
  sensor_init_cfg.cfg.setting = &slave_info2;
  err = ioctl(sensorinit_fd, VIDIOC_MSM_SENSOR_INIT_CFG, &sensor_init_cfg);
  LOG("sensor init cfg (front): %d", err);
  assert(err >= 0);
}

static void camera_open(CameraState *s, bool rear) {
  int err;

  struct sensorb_cfg_data sensorb_cfg_data = {0};
  struct csid_cfg_data csid_cfg_data = {0};
  struct csiphy_cfg_data csiphy_cfg_data = {0};
  struct msm_camera_csiphy_params csiphy_params = {0};
  struct msm_camera_csid_params csid_params = {0};
  struct msm_vfe_input_cfg input_cfg = {0};
  struct msm_vfe_axi_stream_update_cmd update_cmd = {0};
  struct v4l2_event_subscription sub = {0};
  struct ispif_cfg_data ispif_cfg_data = {0};
  struct msm_vfe_cfg_cmd_list cfg_cmd_list = {0};

  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  struct msm_ois_cfg_data ois_cfg_data = {0};

  // open devices
  if (rear) {
    s->csid_fd = open("/dev/v4l-subdev3", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev0", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    if (s->device == DEVICE_LP3) {
      s->sensor_fd = open("/dev/v4l-subdev17", O_RDWR | O_NONBLOCK);
    } else {
      s->sensor_fd = open("/dev/v4l-subdev18", O_RDWR | O_NONBLOCK);
    }
    assert(s->sensor_fd >= 0);
    if (s->device == DEVICE_LP3) {
      s->isp_fd = open("/dev/v4l-subdev13", O_RDWR | O_NONBLOCK);
    } else {
      s->isp_fd = open("/dev/v4l-subdev14", O_RDWR | O_NONBLOCK);
    }
    assert(s->isp_fd >= 0);
    s->eeprom_fd = open("/dev/v4l-subdev8", O_RDWR | O_NONBLOCK);
    assert(s->eeprom_fd >= 0);

    s->actuator_fd = open("/dev/v4l-subdev7", O_RDWR | O_NONBLOCK);
    assert(s->actuator_fd >= 0);

    if (s->device != DEVICE_LP3) {
      s->ois_fd = open("/dev/v4l-subdev10", O_RDWR | O_NONBLOCK);
      assert(s->ois_fd >= 0);
    }
  } else {
    s->csid_fd = open("/dev/v4l-subdev5", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev2", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    if (s->device == DEVICE_LP3) {
      s->sensor_fd = open("/dev/v4l-subdev18", O_RDWR | O_NONBLOCK);
    } else {
      s->sensor_fd = open("/dev/v4l-subdev19", O_RDWR | O_NONBLOCK);
    }
    assert(s->sensor_fd >= 0);
    if (s->device == DEVICE_LP3) {
      s->isp_fd = open("/dev/v4l-subdev14", O_RDWR | O_NONBLOCK);
    } else {
      s->isp_fd = open("/dev/v4l-subdev15", O_RDWR | O_NONBLOCK);
    }
    assert(s->isp_fd >= 0);
    s->eeprom_fd = open("/dev/v4l-subdev9", O_RDWR | O_NONBLOCK);
    assert(s->eeprom_fd >= 0);
  }

  // *** SHUTDOWN ALL ***

  // CSIPHY: release csiphy
  struct msm_camera_csi_lane_params csi_lane_params = {0};
  csi_lane_params.csi_lane_mask = 0x1f;
  csiphy_cfg_data.cfg.csi_lane_params = &csi_lane_params;
  csiphy_cfg_data.cfgtype = CSIPHY_RELEASE;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("release csiphy: %d", err);

  // CSID: release csid
  csid_cfg_data.cfgtype = CSID_RELEASE;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("release csid: %d", err);

  // SENSOR: send power down
  memset(&sensorb_cfg_data, 0, sizeof(sensorb_cfg_data));
  sensorb_cfg_data.cfgtype = CFG_POWER_DOWN;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("sensor power down: %d", err);

  if (rear && s->device != DEVICE_LP3) {
    // ois powerdown
    ois_cfg_data.cfgtype = CFG_OIS_POWERDOWN;
    err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
    LOG("ois powerdown: %d", err);
  }

  // actuator powerdown
  actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERDOWN;
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOG("actuator powerdown: %d", err);

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

  s->eeprom = get_eeprom(s->eeprom_fd, &s->eeprom_size);

  // printf("eeprom:\n");
  // for (int i=0; i<s->eeprom_size; i++) {
  //   printf("%02x", s->eeprom[i]);
  // }
  // printf("\n");

  // CSID: init csid
  csid_cfg_data.cfgtype = CSID_INIT;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("init csid: %d", err);

  // CSIPHY: init csiphy
  memset(&csiphy_cfg_data, 0, sizeof(csiphy_cfg_data));
  csiphy_cfg_data.cfgtype = CSIPHY_INIT;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("init csiphy: %d", err);

  // SENSOR: stop stream
  struct msm_camera_i2c_reg_setting stop_settings = {
    .reg_setting = stop_reg_array,
    .size = ARRAYSIZE(stop_reg_array),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = MSM_CAMERA_I2C_BYTE_DATA,
    .delay = 0
  };
  sensorb_cfg_data.cfgtype = CFG_SET_STOP_STREAM_SETTING;
  sensorb_cfg_data.cfg.setting = &stop_settings;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("stop stream: %d", err);

  // SENSOR: send power up
  memset(&sensorb_cfg_data, 0, sizeof(sensorb_cfg_data));
  sensorb_cfg_data.cfgtype = CFG_POWER_UP;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("sensor power up: %d", err);

  // **** configure the sensor ****

  // SENSOR: send i2c configuration
  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, init_array_imx298, ARRAYSIZE(init_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
  } else if  (s->camera_id == CAMERA_ID_S5K3P8SP) {
    err = sensor_write_regs(s, init_array_s5k3p8sp, ARRAYSIZE(init_array_s5k3p8sp), MSM_CAMERA_I2C_WORD_DATA);
  } else if (s->camera_id == CAMERA_ID_IMX179) {
    err = sensor_write_regs(s, init_array_imx179, ARRAYSIZE(init_array_imx179), MSM_CAMERA_I2C_BYTE_DATA);
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    err = sensor_write_regs(s, init_array_ov8865, ARRAYSIZE(init_array_ov8865), MSM_CAMERA_I2C_BYTE_DATA);
  } else {
    assert(false);
  }
  LOG("sensor init i2c: %d", err);

  if (rear) {
    // init the actuator
    actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERUP;
    err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
    LOG("actuator powerup: %d", err);

    actuator_cfg_data.cfgtype = CFG_ACTUATOR_INIT;
    err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
    LOG("actuator init: %d", err);


    // no OIS in LP3
    if (s->device != DEVICE_LP3) {
      // see sony_imx298_eeprom_format_afdata in libmmcamera_sony_imx298_eeprom.so
      const float far_margin = -0.28;
      uint16_t macro_dac = *(uint16_t*)(s->eeprom + 0x24);
      s->infinity_dac = *(uint16_t*)(s->eeprom + 0x26);
      LOG("macro_dac: %d infinity_dac: %d", macro_dac, s->infinity_dac);

      int dac_range = macro_dac - s->infinity_dac;
      s->infinity_dac += far_margin * dac_range;

      LOG(" -> macro_dac: %d infinity_dac: %d", macro_dac, s->infinity_dac);

      struct msm_actuator_reg_params_t actuator_reg_params[] = {
        {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 240,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 241,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 242,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 243,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        },
      };

      //...
      struct reg_settings_t actuator_init_settings[1] = {0};

      struct region_params_t region_params[] = {
        {
          .step_bound = {512, 0,},
          .code_per_step = 118,
          .qvalue = 128,
        },
      };

      actuator_cfg_data.cfgtype = CFG_SET_ACTUATOR_INFO;
      actuator_cfg_data.cfg.set_info = (struct msm_actuator_set_info_t){
        .actuator_params = {
          .act_type = ACTUATOR_VCM,
          .reg_tbl_size = 4,
          .data_size = 10,
          .init_setting_size = 0,
          .i2c_freq_mode = I2C_CUSTOM_MODE,
          .i2c_addr = 28,
          .i2c_addr_type = MSM_ACTUATOR_BYTE_ADDR,
          .i2c_data_type = MSM_ACTUATOR_BYTE_DATA,
          .reg_tbl_params = &actuator_reg_params[0],
          .init_settings = &actuator_init_settings[0],
          .park_lens = {
            .damping_step = 1023,
            .damping_delay = 15000,
            .hw_params = 58404,
            .max_step = 20,
          }
        },
        .af_tuning_params =   {
          .initial_code = s->infinity_dac,
          .pwd_step = 0,
          .region_size = 1,
          .total_steps = 512,
          .region_params = &region_params[0],
        },
      };
      err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
      LOG("actuator set info: %d", err);

      // power up ois
      ois_cfg_data.cfgtype = CFG_OIS_POWERUP;
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois powerup: %d", err);

      ois_cfg_data.cfgtype = CFG_OIS_INIT;
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois init: %d", err);

      ois_cfg_data.cfgtype = CFG_OIS_CONTROL;
      ois_cfg_data.cfg.set_info.ois_params = (struct msm_ois_params_t){
        // .data_size = 26312,
        .setting_size = 120,
        .i2c_addr = 28,
        .i2c_freq_mode = I2C_CUSTOM_MODE,
        // .i2c_addr_type = wtf
        // .i2c_data_type = wtf
        .settings = &ois_init_settings[0],
      };
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois init settings: %d", err);
    } else {
      // leeco actuator
      // from sniff
      s->infinity_dac = 364;

      struct msm_actuator_reg_params_t actuator_reg_params[] = {
        {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 3,
          .hw_shift = 0,
          .data_type = 9,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        },
      };

      struct reg_settings_t actuator_init_settings[] = {
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=1, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=0, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=2, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },
        { .reg_addr=6, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=64, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },
        { .reg_addr=7, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=113, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },
      };

      struct region_params_t region_params[] = {
        {
          .step_bound = {238, 0,},
          .code_per_step = 235,
          .qvalue = 128,
        },
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
          .park_lens = {
            .damping_step = 1023,
            .damping_delay = 14000,
            .hw_params = 11,
            .max_step = 20,
          }
        },
        .af_tuning_params =   {
          .initial_code = s->infinity_dac,
          .pwd_step = 0,
          .region_size = 1,
          .total_steps = 238,
          .region_params = &region_params[0],
        },
      };

      err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
      LOG("actuator set info: %d", err);
    }
  }

  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, mode_setting_array_imx298, ARRAYSIZE(mode_setting_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
    LOG("sensor setup: %d", err);
  }

  // CSIPHY: configure csiphy
  if (s->camera_id == CAMERA_ID_IMX298) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 14;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 0;
  } else if (s->camera_id == CAMERA_ID_S5K3P8SP) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 24;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 0;
  } else if (s->camera_id == CAMERA_ID_IMX179) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 11;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 2;
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    // guess!
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 24;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 2;
  }
  csiphy_cfg_data.cfgtype = CSIPHY_CFG;
  csiphy_cfg_data.cfg.csiphy_params = &csiphy_params;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("csiphy configure: %d", err);

  // CSID: configure csid
  csid_params.lane_cnt = 4;
  csid_params.lane_assign = 0x4320;
  if (rear) {
    csid_params.phy_sel = 0;
  } else {
    csid_params.phy_sel = 2;
  }
  csid_params.lut_params.num_cid = rear ? 3 : 1;

#define CSI_STATS 0x35
#define CSI_PD 0x36

  csid_params.lut_params.vc_cfg_a[0].cid = 0;
  csid_params.lut_params.vc_cfg_a[0].dt = CSI_RAW10;
  csid_params.lut_params.vc_cfg_a[0].decode_format = CSI_DECODE_10BIT;
  csid_params.lut_params.vc_cfg_a[1].cid = 1;
  csid_params.lut_params.vc_cfg_a[1].dt = CSI_PD;
  csid_params.lut_params.vc_cfg_a[1].decode_format = CSI_DECODE_10BIT;
  csid_params.lut_params.vc_cfg_a[2].cid = 2;
  csid_params.lut_params.vc_cfg_a[2].dt = CSI_STATS;
  csid_params.lut_params.vc_cfg_a[2].decode_format = CSI_DECODE_10BIT;

  csid_params.lut_params.vc_cfg[0] = &csid_params.lut_params.vc_cfg_a[0];
  csid_params.lut_params.vc_cfg[1] = &csid_params.lut_params.vc_cfg_a[1];
  csid_params.lut_params.vc_cfg[2] = &csid_params.lut_params.vc_cfg_a[2];

  csid_cfg_data.cfgtype = CSID_CFG;
  csid_cfg_data.cfg.csid_params = &csid_params;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("csid configure: %d", err);

  // ISP: SMMU_ATTACH
  struct msm_vfe_smmu_attach_cmd smmu_attach_cmd = {
    .security_mode = 0,
    .iommu_attach_mode = IOMMU_ATTACH
  };
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_SMMU_ATTACH, &smmu_attach_cmd);
  LOG("isp smmu attach: %d", err);

  // ******************* STREAM RAW *****************************

  // configure QMET input
  for (int i = 0; i < (rear ? 3 : 1); i++) {
    StreamState *ss = &s->ss[i];

    memset(&input_cfg, 0, sizeof(struct msm_vfe_input_cfg));
    input_cfg.input_src = VFE_RAW_0+i;
    input_cfg.input_pix_clk = s->pixel_clock;
    input_cfg.d.rdi_cfg.cid = i;
    input_cfg.d.rdi_cfg.frame_based = 1;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_INPUT_CFG, &input_cfg);
    LOG("configure input(%d): %d", i, err);

    // ISP: REQUEST_STREAM
    ss->stream_req.axi_stream_handle = 0;
    if (rear) {
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
    ss->stream_req.stream_src = RDI_INTF_0+i;

#ifdef HIGH_FPS
    if (rear) {
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
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_REQUEST_BUF, &ss->buf_request);
    LOG("isp request buf: %d", err);
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
    update_cmd.num_streams = 1;
    update_cmd.update_info[0].user_stream_id = ss->stream_req.stream_id;
    update_cmd.update_info[0].stream_handle = ss->stream_req.axi_stream_handle;
    update_cmd.update_type = UPDATE_STREAM_ADD_BUFQ;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_UPDATE_STREAM, &update_cmd);
    LOG("isp update stream: %d", err);
  }

  LOG("******** START STREAMS ********");

  sub.id = 0;
  sub.type = 0x1ff;
  err = ioctl(s->isp_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  LOG("isp subscribe: %d", err);

  // ISP: START_STREAM
  s->stream_cfg.cmd = START_STREAM;
  s->stream_cfg.num_streams = rear ? 3 : 1;
  for (int i = 0; i < s->stream_cfg.num_streams; i++) {
    s->stream_cfg.stream_handle[i] = s->ss[i].stream_req.axi_stream_handle;
  }
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg);
  LOG("isp start stream: %d", err);
}


static struct damping_params_t actuator_ringing_params = {
  .damping_step = 1023,
  .damping_delay = 15000,
  .hw_params = 0x0000e422,
};

static void rear_start(CameraState *s) {
  int err;

  struct msm_actuator_cfg_data actuator_cfg_data = {0};

  set_exposure(s, 1.0, 1.0);

  err = sensor_write_regs(s, start_reg_array, ARRAYSIZE(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);

  // focus on infinity assuming phone is perpendicular
  int inf_step;

  if (s->device != DEVICE_LP3) {
    imx298_ois_calibration(s->ois_fd, s->eeprom);
    inf_step = 332 - s->infinity_dac;

    // initial guess
    s->lens_true_pos = 300;
  } else {
    // default is OP3, this is for LeEco
    actuator_ringing_params.damping_step = 1023;
    actuator_ringing_params.damping_delay = 20000;
    actuator_ringing_params.hw_params = 13;

    inf_step = 512 - s->infinity_dac;

    // initial guess
    s->lens_true_pos = 400;
  }

  // reset lens position
  memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_SET_POSITION;
  actuator_cfg_data.cfg.setpos = (struct msm_actuator_set_position_t){
    .number_of_steps = 1,
    .hw_params = (s->device != DEVICE_LP3) ? 0x0000e424 : 7,
    .pos = {s->infinity_dac, 0},
    .delay = {0,}
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOG("actuator set pos: %d", err);

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
  int err;

  int step = target - s->cur_lens_pos;
  // LP3 moves only on even positions. TODO: use proper sensor params
  if (s->device == DEVICE_LP3) {
    step /= 2;
  }

  int dest_step_pos = s->cur_step_pos + step;
  dest_step_pos = clamp(dest_step_pos, 0, 255);

  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = (step > 0) ? 0 : 1,
    .sign_dir = (step > 0) ? 1 : -1,
    .dest_step_pos = dest_step_pos,
    .num_steps = abs(step),
    .curr_lens_pos = s->cur_lens_pos,
    .ringing_params = &actuator_ringing_params,
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  //LOGD("actuator move focus: %d", err);

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
    int16_t focus_t = (d[doff+1] << 3) | (d[doff+2] >> 5);
    if (focus_t >= 1024) focus_t = -(2048-focus_t);
    s->focus[i] = focus_t;
    //printf("%x->%d ", d[doff], focus_t);
    if (s->confidence[i] > 0x20) {
      good_count++;
      max_focus = max(max_focus, s->focus[i]);
      avg_focus += s->focus[i];
    }
  }

  //printf("\n");
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

static void do_autofocus(CameraState *s) {
  // params for focus PI controller
  const float focus_kp = 0.005;

  float err = s->focus_err;
  float offset = 0;
  float sag = (s->last_sag_acc_z/9.8) * 128;

  const int dac_up = s->device == DEVICE_LP3? 634:456;
  const int dac_down = s->device == DEVICE_LP3? 366:224;

  if (!isnan(err))  {
    // learn lens_true_pos
    s->lens_true_pos -= err*focus_kp;
  }

  // stay off the walls
  s->lens_true_pos = clamp(s->lens_true_pos, dac_down, dac_up);

  int target = clamp(s->lens_true_pos - sag, dac_down, dac_up);

  /*char debug[4096];
  char *pdebug = debug;
  pdebug += sprintf(pdebug, "focus ");
  for (int i = 0; i < NUM_FOCUS; i++) pdebug += sprintf(pdebug, "%2x(%4d) ", s->confidence[i], s->focus[i]);
  pdebug += sprintf(pdebug, "  err: %7.2f  offset: %6.2f sag: %6.2f lens_true_pos: %6.2f  cur_lens_pos: %4d->%4d", err * focus_kp, offset, sag, s->lens_true_pos, s->cur_lens_pos, target);
  LOGD(debug);*/

  actuator_move(s, target);
}


static void front_start(CameraState *s) {
  int err;

  set_exposure(s, 1.0, 1.0);

  err = sensor_write_regs(s, start_reg_array, ARRAYSIZE(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);
}



void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front) {
  int err;

  struct ispif_cfg_data ispif_cfg_data = {0};

  struct msm_ispif_param_data ispif_params = {0};
  ispif_params.num = 4;
  // rear camera
  ispif_params.entries[0].vfe_intf = 0;
  ispif_params.entries[0].intftype = RDI0;
  ispif_params.entries[0].num_cids = 1;
  ispif_params.entries[0].cids[0] = 0;
  ispif_params.entries[0].csid = 0;
  // front camera
  ispif_params.entries[1].vfe_intf = 1;
  ispif_params.entries[1].intftype = RDI0;
  ispif_params.entries[1].num_cids = 1;
  ispif_params.entries[1].cids[0] = 0;
  ispif_params.entries[1].csid = 2;
  // rear camera (focus)
  ispif_params.entries[2].vfe_intf = 0;
  ispif_params.entries[2].intftype = RDI1;
  ispif_params.entries[2].num_cids = 1;
  ispif_params.entries[2].cids[0] = 1;
  ispif_params.entries[2].csid = 0;
  // rear camera (stats, for AE)
  ispif_params.entries[3].vfe_intf = 0;
  ispif_params.entries[3].intftype = RDI2;
  ispif_params.entries[3].num_cids = 1;
  ispif_params.entries[3].cids[0] = 2;
  ispif_params.entries[3].csid = 0;

  assert(camera_bufs_rear);
  assert(camera_bufs_front);

  int msmcfg_fd = open("/dev/media0", O_RDWR | O_NONBLOCK);
  assert(msmcfg_fd >= 0);

  sensors_init(s);

  int v4l_fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
  assert(v4l_fd >= 0);

  if (s->device == DEVICE_LP3) {
    s->ispif_fd = open("/dev/v4l-subdev15", O_RDWR | O_NONBLOCK);
  } else {
    s->ispif_fd = open("/dev/v4l-subdev16", O_RDWR | O_NONBLOCK);
  }
  assert(s->ispif_fd >= 0);

  // ISPIF: stop
  // memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  // ispif_cfg_data.cfg_type = ISPIF_STOP_FRAME_BOUNDARY;
  // ispif_cfg_data.params = ispif_params;
  // err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  // LOG("ispif stop: %d", err);

  LOG("*** open front ***");
  s->front.ss[0].bufs = camera_bufs_front;
  camera_open(&s->front, false);

  LOG("*** open rear ***");
  s->rear.ss[0].bufs = camera_bufs_rear;
  s->rear.ss[1].bufs = camera_bufs_focus;
  s->rear.ss[2].bufs = camera_bufs_stats;
  camera_open(&s->rear, true);

  if (getenv("CAMERA_TEST")) {
    cameras_close(s);
    exit(0);
  }

  // ISPIF: set vfe info
  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_SET_VFE_INFO;
  ispif_cfg_data.vfe_info.num_vfe = 2;
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif set vfe info: %d", err);

  // ISPIF: setup
  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_INIT;
  ispif_cfg_data.csid_version = 0x30050000; //CSID_VERSION_V35
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif setup: %d", err);

  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_CFG;
  ispif_cfg_data.params = ispif_params;

  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif cfg: %d", err);

  ispif_cfg_data.cfg_type = ISPIF_START_FRAME_BOUNDARY;
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif start_frame_boundary: %d", err);

  front_start(&s->front);
  rear_start(&s->rear);
}


static void camera_close(CameraState *s) {
  int err;

  tbuffer_stop(&s->camera_tb);

  // ISP: STOP_STREAM
  s->stream_cfg.cmd = STOP_STREAM;
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg);
  LOG("isp stop stream: %d", err);

  for (int i = 0; i < 3; i++) {
    StreamState *ss = &s->ss[i];
    if (ss->stream_req.axi_stream_handle != 0) {
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_BUF, &ss->buf_request);
      LOG("isp release buf: %d", err);

      struct msm_vfe_axi_stream_release_cmd stream_release = {
        .stream_handle = ss->stream_req.axi_stream_handle,
      };
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_STREAM, &stream_release);
      LOG("isp release stream: %d", err);
    }
  }

  free(s->eeprom);
}


const char* get_isp_event_name(unsigned int type) {
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
  pthread_mutex_lock(&s->frame_info_lock);
  for (int i=0; i<METADATA_BUF_COUNT; i++) {
    if (s->frame_metadata[i].frame_id == frame_id) {
      pthread_mutex_unlock(&s->frame_info_lock);
      return s->frame_metadata[i];
    }
  }
  pthread_mutex_unlock(&s->frame_info_lock);

  // should never happen
  return (FrameMetadata){
    .frame_id = -1,
  };
}

static bool acceleration_from_sensor_sock(void* sock, float* vs) {
  int err;

  zmq_msg_t msg;
  err = zmq_msg_init(&msg);
  assert(err == 0);

  err = zmq_msg_recv(&msg, sock, 0);
  assert(err >= 0);

  struct capn ctx;
  capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

  cereal_Event_ptr eventp;
  eventp.p = capn_getp(capn_root(&ctx), 0, 1);
  struct cereal_Event eventd;
  cereal_read_Event(&eventd, eventp);

  bool ret = false;

  if (eventd.which == cereal_Event_sensorEvents) {
    cereal_SensorEventData_list lst = eventd.sensorEvents;
    int len = capn_len(lst);
    for (int i=0; i<len; i++) {
      struct cereal_SensorEventData sensord;
      cereal_get_SensorEventData(&sensord, lst, i);

      if (sensord.which == cereal_SensorEventData_acceleration) {
        struct cereal_SensorEventData_SensorVec vecd;
        cereal_read_SensorEventData_SensorVec(&vecd, sensord.acceleration);

        int vlen = capn_len(vecd.v);
        if (vlen < 3) {
          continue; //wtf
        }
        for (int j=0; j<3; j++) {
          vs[j] = capn_to_f32(capn_get32(vecd.v, j));
        }
        ret = true;
        break;
      }
    }
  }

  capn_free(&ctx);
  zmq_msg_close(&msg);

  return ret;
}

static bool gps_time_from_timing_sock(void* sock, uint64_t *mono_time, double* vs) {
  int err;

  zmq_msg_t msg;
  err = zmq_msg_init(&msg);
  assert(err == 0);

  err = zmq_msg_recv(&msg, sock, 0);
  assert(err >= 0);

  struct capn ctx;
  capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

  cereal_Event_ptr eventp;
  eventp.p = capn_getp(capn_root(&ctx), 0, 1);
  struct cereal_Event eventd;
  cereal_read_Event(&eventd, eventp);

  bool ret = false;

  if (eventd.which == cereal_Event_liveLocationTiming) {
    struct cereal_LiveLocationData lld;
    cereal_read_LiveLocationData(&lld, eventd.liveLocationTiming);

    *mono_time = lld.fixMonoTime;
    *vs = lld.timeOfWeek;
    ret = true;
  }

  capn_free(&ctx);
  zmq_msg_close(&msg);

  return ret;
}

static void ops_term() {
  zsock_t *ops_sock = zsock_new_push(">inproc://cameraops");
  assert(ops_sock);

  CameraMsg msg = {.type = -1};
  zmq_send(zsock_resolve(ops_sock), &msg, sizeof(msg), ZMQ_DONTWAIT);

  zsock_destroy(&ops_sock);
}

static void* ops_thread(void* arg) {
  int err;
  DualCameraState *s = (DualCameraState*)arg;

  set_thread_name("camera_settings");

  zsock_t *cameraops = zsock_new_pull("@inproc://cameraops");
  assert(cameraops);

  zsock_t *sensor_sock = zsock_new_sub(">tcp://127.0.0.1:8003", "");
  assert(sensor_sock);

  zsock_t *livelocationtiming_sock = zsock_new_sub(">tcp://127.0.0.1:8049", "");
  assert(livelocationtiming_sock);

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);

  zpoller_t *poller = zpoller_new(cameraops, sensor_sock, livelocationtiming_sock, terminate, NULL);
  assert(poller);

  while (!do_exit) {

    zsock_t *which = (zsock_t*)zpoller_wait(poller, -1);
    if (which == terminate || which == NULL) {
      break;
    }
    void* sockraw = zsock_resolve(which);

    if (which == cameraops) {
      zmq_msg_t msg;
      err = zmq_msg_init(&msg);
      assert(err == 0);

      err = zmq_msg_recv(&msg, sockraw, 0);
      assert(err >= 0);

      CameraMsg cmsg;
      if (zmq_msg_size(&msg) == sizeof(cmsg)) {
        memcpy(&cmsg, zmq_msg_data(&msg), zmq_msg_size(&msg));

        //LOGD("cameraops %d", cmsg.type);

        if (cmsg.type == CAMERA_MSG_AUTOEXPOSE) {
          if (cmsg.camera_num == 0) {
            do_autoexposure(&s->rear, cmsg.grey_frac);
            do_autofocus(&s->rear);
          } else {
            do_autoexposure(&s->front, cmsg.grey_frac);
          }
        } else if (cmsg.type == -1) {
          break;
        }
      }

      zmq_msg_close(&msg);

    } else if (which == sensor_sock) {
      float vs[3] = {0.0};
      bool got_accel = acceleration_from_sensor_sock(sockraw, vs);

      uint64_t ts = nanos_since_boot();
      if (got_accel && ts - s->rear.last_sag_ts > 10000000) { // 10 ms
        s->rear.last_sag_ts = ts;
        s->rear.last_sag_acc_z = -vs[2];
      }
    } else if (which == livelocationtiming_sock) {
      uint64_t mono_time;
      double gps_time;
      if (gps_time_from_timing_sock(sockraw, &mono_time, &gps_time)) {
        s->rear.global_time_offset = (uint64_t)(gps_time*1e9) - mono_time;
        //LOGW("%f %lld  = %lld", gps_time, mono_time, s->rear.global_time_offset);
        s->rear.phase_request = 10000000;
        s->rear.using_pll = true;
      }
    }
  }

  zpoller_destroy(&poller);
  zsock_destroy(&cameraops);
  zsock_destroy(&sensor_sock);
  zsock_destroy(&terminate);

  return NULL;
}

void cameras_run(DualCameraState *s) {
  int err;

  pthread_t ops_thread_handle;
  err = pthread_create(&ops_thread_handle, NULL,
                       ops_thread, s);
  assert(err == 0);

  CameraState* cameras[2] = {&s->rear, &s->front};

  while (!do_exit) {
    struct pollfd fds[2] = {{0}};

    fds[0].fd = cameras[0]->isp_fd;
    fds[0].events = POLLPRI;

    fds[1].fd = cameras[1]->isp_fd;
    fds[1].events = POLLPRI;

    int ret = poll(fds, ARRAYSIZE(fds), 1000);
    if (ret <= 0) {
      LOGE("poll failed (%d)", ret);
      break;
    }

    // process cameras
    for (int i=0; i<2; i++) {
      if (!fds[i].revents) continue;

      CameraState *c = cameras[i];

      struct v4l2_event ev;
      ret = ioctl(c->isp_fd, VIDIOC_DQEVENT, &ev);
      struct msm_isp_event_data *isp_event_data = (struct msm_isp_event_data *)ev.u.data;
      unsigned int event_type = ev.type;

      uint64_t timestamp = (isp_event_data->mono_timestamp.tv_sec*1000000000ULL
                            + isp_event_data->mono_timestamp.tv_usec*1000);

      int buf_idx = isp_event_data->u.buf_done.buf_idx;
      int stream_id = isp_event_data->u.buf_done.stream_id;
      int buffer = (stream_id&0xFFFF) - 1;

      uint64_t t = nanos_since_boot();

      /*if (i == 1) {
        printf("%10.2f: VIDIOC_DQEVENT: %d  type:%X (%s)\n", t*1.0/1e6, ret, event_type, get_isp_event_name(event_type));
      }*/

      // printf("%d: %s\n", i, get_isp_event_name(event_type));

      switch (event_type) {
      case ISP_EVENT_BUF_DIVERT:

        /*if (c->is_samsung) {
          printf("write %d\n", c->frame_size);
          FILE *f = fopen("/tmp/test", "wb");
          fwrite((void*)c->camera_bufs[i].addr, 1, c->frame_size, f);
          fclose(f);
        }*/
        //printf("divert: %d %d %d\n", i, buffer, buf_idx);

        if (buffer == 0) {
          c->camera_bufs_metadata[buf_idx] = get_frame_metadata(c, isp_event_data->frame_id);
          tbuffer_dispatch(&c->camera_tb, buf_idx);
        } else {
          uint8_t *d = c->ss[buffer].bufs[buf_idx].addr;
          if (buffer == 1) {
            parse_autofocus(c, d);
          }
          c->ss[buffer].qbuf_info[buf_idx].dirty_buf = 1;
          ioctl(c->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &c->ss[buffer].qbuf_info[buf_idx]);
        }
        break;
      case ISP_EVENT_EOF:
        // printf("ISP_EVENT_EOF delta %f\n", (t-last_t)/1e6);
        c->last_t = t;

        if (c->using_pll) {
          int mod = ((int)1000000000 / c->fps);
          c->phase_actual = (((timestamp + c->global_time_offset) % mod) + mod) % mod;
          LOGD("phase is %12d request is %12d with offset %lld", c->phase_actual, c->phase_request, c->global_time_offset);
        }

        pthread_mutex_lock(&c->frame_info_lock);
        c->frame_metadata[c->frame_metadata_idx] = (FrameMetadata){
          .frame_id = isp_event_data->frame_id,
          .timestamp_eof = timestamp,
          .frame_length = c->cur_frame_length,
          .integ_lines = c->cur_integ_lines,
          .global_gain = c->cur_gain,
          .lens_pos = c->cur_lens_pos,
          .lens_sag = c->last_sag_acc_z,
          .lens_err = c->focus_err,
          .lens_true_pos = c->lens_true_pos,
        };
        c->frame_metadata_idx = (c->frame_metadata_idx+1)%METADATA_BUF_COUNT;
        pthread_mutex_unlock(&c->frame_info_lock);

        break;
      case ISP_EVENT_ERROR:
        LOGE("ISP_EVENT_ERROR! err type: 0x%08x", isp_event_data->u.error_info.err_type);
        break;
      }
    }
  }

  LOG(" ************** STOPPING **************");

  ops_term();
  err = pthread_join(ops_thread_handle, NULL);
  assert(err == 0);

  cameras_close(s);
}

void cameras_close(DualCameraState *s) {
  camera_close(&s->rear);
  camera_close(&s->front);
}

