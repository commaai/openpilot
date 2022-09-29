#pragma once

#include <cassert>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_sensor.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"
#include "system/camerad/cameras/camera_util.h"
#include "system/camerad/cameras/sensor2_i2c.h"

#define CAMERA_ID_IMX298 0
#define CAMERA_ID_IMX179 1
#define CAMERA_ID_S5K3P8SP 2
#define CAMERA_ID_OV8865 3
#define CAMERA_ID_IMX298_FLIPPED 4
#define CAMERA_ID_OV10640 5
#define CAMERA_ID_LGC920 6
#define CAMERA_ID_LGC615 7
#define CAMERA_ID_AR0231 8
#define CAMERA_ID_OX03C10 9

const size_t FRAME_WIDTH = 1928;
const size_t FRAME_HEIGHT = 1208;
const size_t FRAME_STRIDE = 2896;  // for 12 bit output. 1928 * 12 / 8 + 4 (alignment)

typedef struct CameraInfo {
  uint32_t frame_width, frame_height;
  uint32_t frame_stride;
  uint32_t frame_offset = 0;
  uint32_t extra_height = 0;
  int registers_offset = -1;
  int stats_offset = -1;
} CameraInfo;

class MultiCameraState;

class AbstractCamera {
public:
  AbstractCamera() = default;
  virtual ~AbstractCamera(){};
  int open(MultiCameraState *multi_cam_state, int camera_num, bool enabled);
  void close();
  void sensors_start();
  void sensors_i2c(const std::vector<i2c_random_wr_payload> &dat, int op_code);
  void sensors_poke(int request_id);
  void config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset);
  virtual std::vector<i2c_random_wr_payload> getExposureVector(int new_g, bool dc_gain_enabled, int exposure_time, int dc_gain_weight) const = 0;
  virtual void processRegisters(void *addr, cereal::FrameData::Builder &framed) const {}

  int camera_id;
  int camera_num;
  bool enabled;

  int buf0_handle;
  int32_t link_handle;
  int32_t session_handle;
  CameraInfo ci;

  float dc_gain_factor;
  int dc_gain_min_weight;
  int dc_gain_max_weight;
  float dc_gain_on_grey;
  float dc_gain_off_grey;
  float target_grey_factor;
  int exposure_time_min;
  int exposure_time_max;
  int analog_gain_min_idx;
  int analog_gain_rec_idx;
  int analog_gain_max_idx;
  float min_ev, max_ev;
  int registers_offset = 0;
  float sensor_analog_gains[35];

protected:
  int sensors_init();
  virtual int getSlaveAddress(int port) const = 0;

  MultiCameraState *multi_cam_state;
  unique_fd sensor_fd;
  unique_fd csiphy_fd;
  int32_t sensor_dev_handle;
  int32_t isp_dev_handle;
  int32_t csiphy_dev_handle;
  MemoryManager mm;

  camera_sensor_i2c_type i2c_type;
  std::vector<struct i2c_random_wr_payload> start_reg_array;
  std::vector<struct i2c_random_wr_payload> init_array;
  uint32_t in_port_info_dt;
  uint32_t reg_addr;
  uint32_t expected_data;
  uint32_t config_val_low;
};

class CameraAR0231 : public AbstractCamera {
public:
  CameraAR0231();
  ~CameraAR0231() {}
  int getSlaveAddress(int port) const override;
  std::vector<struct i2c_random_wr_payload> getExposureVector(int new_g, bool dc_gain_enabled, int exposure_time, int dc_gain_weight) const override;
  void processRegisters(void *addr, cereal::FrameData::Builder &framed) const override;

private:
  std::map<uint16_t, uint16_t> parseRegisters(uint8_t *data, std::initializer_list<uint16_t> addrs) const;
  std::map<uint16_t, std::pair<int, int>> buildRegisterLut(uint8_t *data) const;

  mutable std::map<uint16_t, std::pair<int, int>> ar0231_register_lut;
};

class CameraOX03C10 : public AbstractCamera {
public:
  CameraOX03C10();
  ~CameraOX03C10() {}
  int getSlaveAddress(int port) const override;
  std::vector<struct i2c_random_wr_payload> getExposureVector(int new_g, bool dc_gain_enabled, int exposure_time, int dc_gain_weight) const override;
};
