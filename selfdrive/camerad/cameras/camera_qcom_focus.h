#pragma once

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include <mutex>

#include "common/utilpp.h"
#include "imgproc/utils.h"
#include "msm_cam_sensor.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "camera_common.h"

#define NUM_FOCUS 8

class AutoFocus {
public:
  AutoFocus() = default;
  void init(const int device);
  void start();
  void do_focus(float *accel_z);
  void parse(const uint8_t *d);
  void setup_self_recover(const uint16_t lapres[], const size_t lapres_size);
  void fill(cereal::FrameData::Builder fd);
  void fill(FrameMetadata &fd);

private:
  void actuator_move(uint16_t target, bool in_lock = false);
  void init_LP3();
  void init_other();

  int device_;
  unique_fd actuator_fd_, ois_fd_, eeprom_fd_;
  int self_recover_ = 0;  // af recovery counter, neg is patience, pos is active
  float focus_err_ = 0.;
  float last_sag_acc_z_ = 0.;
  float lens_true_pos_ = 0.;
  uint16_t cur_step_pos_ = 0; 
  uint16_t cur_lens_pos_ = 0;
  int16_t focus_[NUM_FOCUS] = {};
  uint8_t confidence_[NUM_FOCUS] = {};
  uint16_t infinity_dac_ = 0;
  std::unique_ptr<uint8_t[]> eeprom_;
  damping_params_t actuator_ringing_params_ = {};
  std::mutex mutex_;
};
