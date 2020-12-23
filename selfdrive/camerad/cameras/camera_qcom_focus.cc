#include "camera_qcom_focus.h"

#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <algorithm>
#include <sys/ioctl.h>
#include "common/timing.h"
#include "common/swaglog.h"
#include "sensor_i2c.h"

#include "camera_qcom.h"

#define LP3_AF_DAC_DOWN 366
#define LP3_AF_DAC_UP 634
#define LP3_AF_DAC_M 440
#define LP3_AF_DAC_3SIG 52
#define OP3T_AF_DAC_DOWN 224
#define OP3T_AF_DAC_UP 456
#define OP3T_AF_DAC_M 300
#define OP3T_AF_DAC_3SIG 96

#define FOCUS_RECOVER_PATIENCE 50 // 2.5 seconds of complete blur
#define FOCUS_RECOVER_STEPS 240 // 6 seconds

static void imx298_ois_calibration(int ois_fd_, uint8_t* eeprom) {
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


  struct msm_camera_i2c_seq_reg_array ois_reg_settings[std::size(ois_registers)] = {{0}};
  for (int i=0; i<std::size(ois_registers); i++) {
    ois_reg_settings[i].reg_addr = ois_registers[i][0];
    ois_reg_settings[i].reg_data[0] = ois_registers[i][1] & 0xff;
    ois_reg_settings[i].reg_data[1] = (ois_registers[i][1] >> 8) & 0xff;
    ois_reg_settings[i].reg_data_size = 2;
  }
  struct msm_camera_i2c_seq_reg_setting ois_reg_setting = {
    .reg_setting = &ois_reg_settings[0],
    .size = std::size(ois_reg_settings),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .delay = 0,
  };
  struct msm_ois_cfg_data cfg = {
      .cfgtype = CFG_OIS_I2C_WRITE_SEQ_TABLE,
      .cfg.settings = &ois_reg_setting};
  int err = ioctl(ois_fd_, VIDIOC_MSM_OIS_CFG, &cfg);
  LOGW("ois reg calibration: %d", err);
}

static std::unique_ptr<uint8_t[]> get_eeprom(int eeprom_fd_) {
  struct msm_eeprom_cfg_data cfg = {};
  cfg.cfgtype = CFG_EEPROM_GET_CAL_DATA;
  int err = ioctl(eeprom_fd_, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);

  uint32_t num_bytes = cfg.cfg.get_data.num_bytes;
  assert(num_bytes > 100);

  std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(num_bytes);

  cfg.cfgtype = CFG_EEPROM_READ_CAL_DATA;
  cfg.cfg.read_data.num_bytes = num_bytes;
  cfg.cfg.read_data.dbuffer = buffer.get();
  err = ioctl(eeprom_fd_, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);
  return buffer;
}

void AutoFocus::init_other() {
  eeprom_fd_ = open("/dev/v4l-subdev8", O_RDWR | O_NONBLOCK);
  assert(eeprom_fd_ >= 0);
  eeprom_ = get_eeprom(eeprom_fd_);

  struct msm_ois_cfg_data ois_cfg_data = {};
  ois_fd_ = open("/dev/v4l-subdev10", O_RDWR | O_NONBLOCK);
  assert(ois_fd_ >= 0);
  // ois powerdown
  ois_cfg_data.cfgtype = CFG_OIS_POWERDOWN;
  int err = ioctl(ois_fd_, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
  LOGW("ois powerdown: %d", err);
  // see sony_imx298_eeprom_format_afdata in libmmcamera_sony_imx298_eeprom.so
  const float far_margin = -0.28;
  uint16_t macro_dac = *(uint16_t *)(eeprom_.get() + 0x24);
  infinity_dac_ = *(uint16_t *)(eeprom_.get() + 0x26);
  LOGW("macro_dac: %d infinity_dac_: %d", macro_dac, infinity_dac_);

  int dac_range = macro_dac - infinity_dac_;
  infinity_dac_ += far_margin * dac_range;

  LOGW(" -> macro_dac: %d infinity_dac_: %d", macro_dac, infinity_dac_);

  struct msm_actuator_reg_params_t actuator_reg_params[] = {
    {.reg_write_type = MSM_ACTUATOR_WRITE_DAC, .reg_addr = 240, .data_type = 10, .addr_type = 4},
    {.reg_write_type = MSM_ACTUATOR_WRITE_DAC, .reg_addr = 241, .data_type = 10, .addr_type = 4},
    {.reg_write_type = MSM_ACTUATOR_WRITE_DAC, .reg_addr = 242, .data_type = 10, .addr_type = 4},
    {.reg_write_type = MSM_ACTUATOR_WRITE_DAC, .reg_addr = 243, .data_type = 10, .addr_type = 4},
  };
  struct reg_settings_t actuator_init_settings[1] = {0};

  struct region_params_t region_params[] = {
    {.step_bound = {512, 0,}, .code_per_step = 118, .qvalue = 128}
  };
  
  msm_actuator_cfg_data actuator_cfg_data = {};
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
      .initial_code = (int16_t)infinity_dac_,
      .pwd_step = 0,
      .region_size = 1,
      .total_steps = 512,
      .region_params = &region_params[0],
    },
  };
  err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator set info: %d", err);

  // power up ois
  ois_cfg_data.cfgtype = CFG_OIS_POWERUP;
  err = ioctl(ois_fd_, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
  LOGW("ois powerup: %d", err);

  ois_cfg_data.cfgtype = CFG_OIS_INIT;
  err = ioctl(ois_fd_, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
  LOGW("ois init: %d", err);

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
  err = ioctl(ois_fd_, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
  LOGW("ois init settings: %d", err);

  imx298_ois_calibration(ois_fd_, eeprom_.get());

  cur_step_pos_ = 332 - infinity_dac_;
  // initial guess
  lens_true_pos_ = 300;

  actuator_ringing_params_ = {
      .damping_step = 1023,
      .damping_delay = 15000,
      .hw_params = 0x0000e422,
  };
}

void AutoFocus::init_LP3() {
  // leeco actuator (DW9800W H-Bridge Driver IC)
  // from sniff
  infinity_dac_ = 364;

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
    {.reg_addr = 2, .addr_type = MSM_ACTUATOR_BYTE_ADDR, .reg_data = 1, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0},   // PD = power down
    {.reg_addr = 2, .addr_type = MSM_ACTUATOR_BYTE_ADDR, .reg_data = 0, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2},   // 0 = power up
    {.reg_addr = 2, .addr_type = MSM_ACTUATOR_BYTE_ADDR, .reg_data = 2, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2},   // RING = SAC mode
    {.reg_addr = 6, .addr_type = MSM_ACTUATOR_BYTE_ADDR, .reg_data = 64, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0},  // 0x40 = SAC3 mode
    {.reg_addr = 7, .addr_type = MSM_ACTUATOR_BYTE_ADDR, .reg_data = 113, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0},
    // 0x71 = DIV1 | DIV0 | SACT0 -- Tvib x 1/4 (quarter)
    // SAC Tvib = 6.3 ms + 0.1 ms = 6.4 ms / 4 = 1.6 ms
    // LSC 1-step = 252 + 1*4 = 256 ms / 4 = 64 ms
  };

  struct region_params_t region_params[] = {
    {.step_bound = {238, 0,}, .code_per_step = 235, .qvalue = 128}
  };

  msm_actuator_cfg_data actuator_cfg_data = {};
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
      .initial_code = (int16_t)infinity_dac_,
      .pwd_step = 0,
      .region_size = 1,
      .total_steps = 238,
      .region_params = &region_params[0],
    },
  };

  int err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator set info: %d", err);

  cur_step_pos_ = 512 - infinity_dac_;
  // initial guess
  lens_true_pos_ = 400;

  actuator_ringing_params_ = {
      .damping_step = 1023,
      .damping_delay = 20000,
      .hw_params = 13,
  };
}

void AutoFocus::init(const int device) {
  device_ = device;
  actuator_fd_ = open("/dev/v4l-subdev7", O_RDWR | O_NONBLOCK);
  assert(actuator_fd_ >= 0);

  // actuator powerdown
  msm_actuator_cfg_data actuator_cfg_data = {};
  actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERDOWN;
  int err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator powerdown: %d", err);

  // init the actuator
  actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERUP;
  err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator powerup: %d", err);

  actuator_cfg_data.cfgtype = CFG_ACTUATOR_INIT;
  err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator init: %d", err);

  device_ == DEVICE_LP3 ? init_LP3() : init_other();
}

void AutoFocus::start() {
  // reset lens position
  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_SET_POSITION;
  actuator_cfg_data.cfg.setpos = (struct msm_actuator_set_position_t){
    .number_of_steps = 1,
    .hw_params = (uint32_t)((device_ != DEVICE_LP3) ? 0x0000e424 : 7),
    .pos = {infinity_dac_, 0},
    .delay = {0,}
  };
  int err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOGW("actuator set pos: %d", err);

  // TODO: confirm this isn't needed
  /*memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = 0,
    .sign_dir = 1,
    .dest_step_pos = inf_step,
    .num_steps = inf_step,
    .curr_lens_pos = 0,
    .ringing_params = &actuator_ringing_params_,
  };
  err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data); // should be ~332 at startup ?
  LOGW("init actuator move focus: %d", err);*/
  //actuator_cfg_data.cfg.move.curr_lens_pos;
  actuator_move(cur_lens_pos_);

  LOGW("init lens pos: %d", cur_lens_pos_);
}

void AutoFocus::actuator_move(uint16_t target, bool in_lock) {
  int step = target - cur_lens_pos_;
  // LP3 moves only on even positions. TODO: use proper sensor params
  if (device_ == DEVICE_LP3) {
    step /= 2;
  }

  int dest_step_pos = cur_step_pos_ + step;
  dest_step_pos = std::clamp(dest_step_pos, 0, 255);
  double t1 = millis_since_boot();
  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
      .dir = (int8_t)((step > 0) ? 0 : 1),
      .sign_dir = (int8_t)((step > 0) ? 1 : -1),
      .dest_step_pos = (int16_t)dest_step_pos,
      .num_steps = abs(step),
      .curr_lens_pos = cur_lens_pos_,
      .ringing_params = &actuator_ringing_params_,
  };

  // don't lock ioctl, it's slow
  if (in_lock) {
    mutex_.unlock();
  }
  int err = ioctl(actuator_fd_, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  // LOGD("actuator move focus: %d", err);
  if (in_lock) {
    mutex_.lock();
  }
  cur_step_pos_ = dest_step_pos;
  cur_lens_pos_ = actuator_cfg_data.cfg.move.curr_lens_pos;

  // LOGW("step %d   target: %d  lens pos: %d time:%f", dest_step_pos, target, cur_lens_pos_, millis_since_boot() - t1);
}

void AutoFocus::do_focus(float *accel_z) {
  // params for focus PI controller
  const int dac_up = device_ == DEVICE_LP3? LP3_AF_DAC_UP:OP3T_AF_DAC_UP;
  const int dac_down = device_ == DEVICE_LP3? LP3_AF_DAC_DOWN:OP3T_AF_DAC_DOWN;

  if (accel_z) {
    last_sag_acc_z_ = *accel_z;
  }
  const float sag = (last_sag_acc_z_ / 9.8) * 128;

  mutex_.lock();

  if (!isnan(focus_err_)) {
    // learn lens_true_pos_
    const float focus_kp = 0.005;
    lens_true_pos_ -= focus_err_*focus_kp;
  }

  // stay off the walls
  lens_true_pos_ = std::clamp(lens_true_pos_, float(dac_down), float(dac_up));
  int target = std::clamp(lens_true_pos_ - sag, float(dac_down), float(dac_up));

  /*char debug[4096];
  char *pdebug = debug;
  pdebug += sprintf(pdebug, "focus ");
  for (int i = 0; i < NUM_FOCUS; i++) pdebug += sprintf(pdebug, "%2x(%4d) ", confidence_[i], focus_[i]);
  pdebug += sprintf(pdebug, "  err: %7.2f  offset: %6.2f sag: %6.2f lens_true_pos_: %6.2f  cur_lens_pos_: %4d->%4d", err * focus_kp, offset, sag, s->lens_true_pos_, s->cur_lens_pos_, target);
  LOGD(debug);*/
  actuator_move(target, true);

  mutex_.unlock();
}

void AutoFocus::setup_self_recover(const uint16_t lapres[], const size_t lapres_size) {
  const int dac_down = device_ == DEVICE_LP3 ? LP3_AF_DAC_DOWN : OP3T_AF_DAC_DOWN;
  const int dac_up = device_ == DEVICE_LP3 ? LP3_AF_DAC_UP : OP3T_AF_DAC_UP;
  const int dac_m = device_ == DEVICE_LP3 ? LP3_AF_DAC_M : OP3T_AF_DAC_M;
  const int dac_3sig = device_ == DEVICE_LP3 ? LP3_AF_DAC_3SIG : OP3T_AF_DAC_3SIG;

   std::unique_lock lock(mutex_);

  if (self_recover_ < 2 && (lens_true_pos_ < (dac_down + 1) || lens_true_pos_ > (dac_up - 1)) && is_blur(lapres, lapres_size)) {
    // truly stuck, needs help
    if (--self_recover_ < -FOCUS_RECOVER_PATIENCE) {
      LOGD("rear camera bad state detected. attempting recovery from %.1f, recover state is %d", lens_true_pos_, self_recover_);
      // parity determined by which end is stuck at
      self_recover_ = FOCUS_RECOVER_STEPS + (lens_true_pos_ < dac_m ? 1 : 0);
    }
  } else if (self_recover_ < 2 && (lens_true_pos_ < (dac_m - dac_3sig) || lens_true_pos_ > (dac_m + dac_3sig))) {
    // in suboptimal position with high prob, but may still recover by itself
    if (--self_recover_ < -(FOCUS_RECOVER_PATIENCE * 3)) {
      self_recover_ = FOCUS_RECOVER_STEPS / 2 + (lens_true_pos_ < dac_m ? 1 : 0);
    }
  } else if (self_recover_ < 0) {
    self_recover_ += 1;  // reset if fine
  }
}

void AutoFocus::parse(const uint8_t *d) {
  int good_count = 0;
  int16_t max_focus = -32767;
  int avg_focus = 0;
  
  /*printf("FOCUS: ");
  for (int i = 0; i < 0x10; i++) {
    printf("%2.2X ", d[i]);
  }*/
  std::unique_lock lock(mutex_);

  for (int i = 0; i < NUM_FOCUS; i++) {
    int doff = i*5+5;
    confidence_[i] = d[doff];
    // this should just be a 10-bit signed int instead of 11
    // TODO: write it in a nicer way
    int16_t focus_t = (d[doff+1] << 3) | (d[doff+2] >> 5);
    if (focus_t >= 1024) focus_t = -(2048-focus_t);
    focus_[i] = focus_t;
    //printf("%x->%d ", d[doff], focus_t);
    if (confidence_[i] > 0x20) {
      good_count++;
      max_focus = std::max(max_focus, focus_[i]);
      avg_focus += focus_[i];
    }
  }
  // self recover override
  if (self_recover_ > 1) {
    focus_err_ = 200 * ((self_recover_ % 2 == 0) ? 1:-1); // far for even numbers, close for odd
    self_recover_ -= 2;
    return;
  }

  if (good_count < 4) {
    focus_err_ = nan("");
    return;
  }

  avg_focus /= good_count;

  // outlier rejection
  if (abs(avg_focus - max_focus) > 200) {
    focus_err_ = nan("");
    return;
  }

  focus_err_ = max_focus*1.0;
}

void AutoFocus::fill(cereal::FrameData::Builder fd) {
  std::unique_lock lock(mutex_);
  fd.setFocusVal(focus_);
  fd.setFocusConf(confidence_);
  fd.setRecoverState(self_recover_);
}

void AutoFocus::fill(FrameMetadata &fd) {
  std::unique_lock lock(mutex_);
  fd.lens_pos = cur_lens_pos_;
  fd.lens_sag = last_sag_acc_z_;
  fd.lens_err = focus_err_;
  fd.lens_true_pos = lens_true_pos_;
}
