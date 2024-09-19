#pragma once

#include <memory>
#include <utility>

#include "media/cam_isp_ife.h"

#include "common/util.h"
#include "system/camerad/cameras/tici.h"
#include "system/camerad/cameras/camera_util.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/sensors/sensor.h"

#define FRAME_BUF_COUNT 4

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

// For use with the Spectra 280 ISP in the SDM845
// https://github.com/commaai/agnos-kernel-sdm845


class SpectraMaster {
public:
  void init();

  unique_fd video0_fd;
  unique_fd cam_sync_fd;
  unique_fd isp_fd;
  int device_iommu = -1;
  int cdm_iommu = -1;
};

class SpectraCamera {
public:
  SpectraCamera(SpectraMaster *master, const CameraConfig &config);
  ~SpectraCamera();

  void camera_open();
  void camera_close();
  void camera_map_bufs();
  void camera_init(VisionIpcServer *v, cl_device_id device_id, cl_context ctx);
  void config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset);

  int clear_req_queue();
  void enqueue_buffer(int i, bool dp);
  void enqueue_req_multi(uint64_t start, int n, bool dp);

  int sensors_init();
  void sensors_start();
  void sensors_poke(int request_id);
  void sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word);

  bool openSensor();
  void configISP();
  void configCSIPHY();
  void linkDevices();

  // *** state ***

  bool open = false;
  bool enabled = true;
  CameraConfig cc;
  std::unique_ptr<const SensorInfo> sensor;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int32_t session_handle = -1;
  int32_t sensor_dev_handle = -1;
  int32_t isp_dev_handle = -1;
  int32_t csiphy_dev_handle = -1;

  int32_t link_handle = -1;

  int buf0_handle = 0;
  int buf_handle[FRAME_BUF_COUNT] = {};
  int sync_objs[FRAME_BUF_COUNT] = {};
  uint64_t request_ids[FRAME_BUF_COUNT] = {};
  uint64_t request_id_last = 0;
  uint64_t frame_id_last = 0;
  uint64_t idx_offset = 0;
  bool skipped = true;

  CameraBuf buf;
  MemoryManager mm;
  SpectraMaster *m;
};
