/**
 * Jetson Camera Implementation
 *
 * This file implements camera capture for Jetson Orin Nano Super with IMX219 sensor
 * using GStreamer C API with nvarguscamerasrc plugin.
 *
 * Key design choices:
 * - GStreamer C API for proper lifecycle management and cleanup
 * - NV12 output directly (no BGR conversion needed)
 * - Thread-per-camera model with shared VisionIpcServer
 * - appsink callbacks for frame processing
 */

#include "system/camerad/cameras/camera_common.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "msgq/visionipc/visionipc_server.h"

// Global exit handler - defined in camera_qcom2.cc but we define our own for standalone builds
ExitHandler do_exit;

namespace {

// Jetson camera configuration - simplified vs QCOM
struct JetsonCameraConfig {
  int camera_num;
  VisionStreamType stream_type;
  const char *publish_name;
  cereal::FrameData::Builder (cereal::Event::Builder::*init_camera_state)();
  bool enabled;
  int sensor_id;  // For nvarguscamerasrc sensor-id parameter
};

// Default configurations
const JetsonCameraConfig JETSON_ROAD_CONFIG = {
  .camera_num = 0,
  .stream_type = VISION_STREAM_ROAD,
  .publish_name = "roadCameraState",
  .init_camera_state = &cereal::Event::Builder::initRoadCameraState,
  .enabled = true,
  .sensor_id = 0,
};

const JetsonCameraConfig JETSON_WIDE_CONFIG = {
  .camera_num = 1,
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .publish_name = "wideRoadCameraState",
  .init_camera_state = &cereal::Event::Builder::initWideRoadCameraState,
  .enabled = false,
  .sensor_id = 1,
};

const JetsonCameraConfig JETSON_DRIVER_CONFIG = {
  .camera_num = 2,
  .stream_type = VISION_STREAM_DRIVER,
  .publish_name = "driverCameraState",
  .init_camera_state = &cereal::Event::Builder::initDriverCameraState,
  .enabled = false,
  .sensor_id = 2,
};

// Helper functions
int getEnvInt(const char *name, int default_val) {
  const char *val = getenv(name);
  return val ? atoi(val) : default_val;
}

std::string getEnvStr(const char *name, const std::string &default_val = "") {
  const char *val = getenv(name);
  return val ? std::string(val) : default_val;
}

// Build GStreamer pipeline string for nvarguscamerasrc
std::string build_gst_pipeline(int sensor_id, int width, int height, int framerate) {
  char buf[1024];
  snprintf(buf, sizeof(buf),
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=NV12 ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false",
    sensor_id, width, height, framerate);
  return std::string(buf);
}

class JetsonCamera {
public:
  JetsonCamera(const JetsonCameraConfig &config, VisionIpcServer *vipc, int width, int height, int fps)
      : config_(config), vipc_server_(vipc), width_(width), height_(height), framerate_(fps) {
    pm_ = std::make_unique<PubMaster>(std::vector<const char*>{config.publish_name});
  }

  ~JetsonCamera() {
    stop();
  }

  bool start() {
    if (!initGStreamer()) {
      return false;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      LOGE("Failed to start pipeline for camera %d", config_.camera_num);
      cleanupGStreamer();
      return false;
    }

    running_ = true;
    LOG("Camera %d (%s) started successfully", config_.camera_num, config_.publish_name);
    return true;
  }

  void stop() {
    running_ = false;
    cleanupGStreamer();
  }

  // Main loop - called from dedicated thread
  void run() {
    while (running_ && !do_exit) {
      // Check for bus messages (errors, warnings, etc.)
      if (bus_) {
        GstMessage *msg = gst_bus_timed_pop(bus_, 100 * GST_MSECOND);
        if (msg) {
          handleBusMessage(msg);
          gst_message_unref(msg);
        }
      } else {
        util::sleep_for(100);
      }
    }

    LOGD("Camera %d thread exiting", config_.camera_num);
  }

  const JetsonCameraConfig &config() const { return config_; }

private:
  bool initGStreamer() {
    // Determine pipeline string
    std::string env_key;
    if (config_.stream_type == VISION_STREAM_ROAD) {
      env_key = "ROAD_CAM";
    } else if (config_.stream_type == VISION_STREAM_WIDE_ROAD) {
      env_key = "WIDE_CAM";
    } else if (config_.stream_type == VISION_STREAM_DRIVER) {
      env_key = "DRIVER_CAM";
    }

    std::string pipeline_str;
    std::string env_val = getEnvStr(env_key.c_str());

    if (!env_val.empty()) {
      // Check if it's just a number (sensor ID) or a full pipeline
      char *endptr;
      long sensor = strtol(env_val.c_str(), &endptr, 10);
      if (*endptr == '\0') {
        // Just a sensor ID
        pipeline_str = build_gst_pipeline(static_cast<int>(sensor), width_, height_, framerate_);
      } else {
        // Custom pipeline string - ensure it has an appsink
        pipeline_str = env_val;
        if (pipeline_str.find("appsink") == std::string::npos) {
          pipeline_str += " ! appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false";
        }
      }
    } else {
      pipeline_str = build_gst_pipeline(config_.sensor_id, width_, height_, framerate_);
    }

    LOG("Starting camera %d with pipeline: %s", config_.camera_num, pipeline_str.c_str());

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
      LOGE("Failed to create pipeline: %s", error->message);
      g_error_free(error);
      return false;
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
      LOGE("Failed to get appsink from pipeline");
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      return false;
    }

    // Configure appsink callbacks
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = JetsonCamera::newSampleCallback;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink_), &callbacks, this, nullptr);

    // Get bus for error handling
    bus_ = gst_element_get_bus(pipeline_);

    return true;
  }

  void cleanupGStreamer() {
    if (pipeline_) {
      LOGD("Cleaning up GStreamer pipeline for camera %d", config_.camera_num);

      // Stop pipeline
      gst_element_set_state(pipeline_, GST_STATE_NULL);

      // Wait for state change to complete (with timeout)
      GstState state;
      GstStateChangeReturn ret = gst_element_get_state(pipeline_, &state, nullptr, 3 * GST_SECOND);
      if (ret == GST_STATE_CHANGE_FAILURE) {
        LOGW("Pipeline state change to NULL failed for camera %d", config_.camera_num);
      }

      if (bus_) {
        gst_object_unref(bus_);
        bus_ = nullptr;
      }

      if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
      }

      gst_object_unref(pipeline_);
      pipeline_ = nullptr;

      LOGD("GStreamer cleanup complete for camera %d", config_.camera_num);
    }
  }

  void handleBusMessage(GstMessage *msg) {
    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *debug = nullptr;
        gst_message_parse_error(msg, &err, &debug);
        LOGE("GStreamer error (cam %d): %s", config_.camera_num, err->message);
        if (debug) {
          LOGD("Debug info: %s", debug);
          g_free(debug);
        }
        g_error_free(err);
        running_ = false;
        break;
      }
      case GST_MESSAGE_WARNING: {
        GError *err = nullptr;
        gchar *debug = nullptr;
        gst_message_parse_warning(msg, &err, &debug);
        LOGW("GStreamer warning (cam %d): %s", config_.camera_num, err->message);
        if (debug) g_free(debug);
        g_error_free(err);
        break;
      }
      case GST_MESSAGE_EOS:
        LOG("End of stream for camera %d", config_.camera_num);
        running_ = false;
        break;
      default:
        break;
    }
  }

  static GstFlowReturn newSampleCallback(GstAppSink *sink, gpointer user_data) {
    JetsonCamera *self = static_cast<JetsonCamera*>(user_data);
    return self->onNewSample(sink);
  }

  GstFlowReturn onNewSample(GstAppSink *sink) {
    if (!running_ || do_exit) {
      return GST_FLOW_EOS;
    }

    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
      return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;

    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    // Get timestamps
    uint64_t timestamp_sof = nanos_since_boot();

    // Get VisionBuf and copy frame data
    VisionBuf *vbuf = vipc_server_->get_buffer(config_.stream_type);
    if (!vbuf) {
      LOGE("Failed to get VisionBuf for camera %d", config_.camera_num);
      gst_buffer_unmap(buffer, &map);
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    // Copy NV12 data: Y plane followed by UV plane
    // NV12 size = width * height * 3 / 2
    size_t expected_size = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (map.size >= expected_size) {
      memcpy(vbuf->addr, map.data, expected_size);
    } else {
      LOGE("Buffer size mismatch: got %zu, expected %zu", map.size, expected_size);
      gst_buffer_unmap(buffer, &map);
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    uint64_t timestamp_eof = nanos_since_boot();

    // Send frame via VisionIPC
    VisionIpcBufExtra extra = {
      .frame_id = frame_id_,
      .timestamp_sof = timestamp_sof,
      .timestamp_eof = timestamp_eof,
    };
    vbuf->set_frame_id(frame_id_);
    vipc_server_->send(vbuf, &extra, false);  // no sync needed - no OpenCL

    // Publish camera state via cereal
    MessageBuilder msg;
    auto framed = (msg.initEvent().*config_.init_camera_state)();
    framed.setFrameId(frame_id_);
    framed.setTimestampEof(timestamp_eof);
    framed.setTimestampSof(timestamp_sof);

    // Set identity transform matrix
    const float transform[] = {1.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 1.0f};
    framed.setTransform(kj::ArrayPtr<const float>(transform, 9));

    pm_->send(config_.publish_name, msg);

    frame_id_++;
    return GST_FLOW_OK;
  }

  JetsonCameraConfig config_;
  VisionIpcServer *vipc_server_;
  std::unique_ptr<PubMaster> pm_;

  // GStreamer components
  GstElement *pipeline_ = nullptr;
  GstElement *appsink_ = nullptr;
  GstBus *bus_ = nullptr;

  // Configuration
  int width_;
  int height_;
  int framerate_;

  // Frame tracking
  uint32_t frame_id_ = 0;

  std::atomic<bool> running_{false};
};

}  // namespace

// Stub for QCOM camerad_thread - not available on Jetson
// This allows the binary to link even if USE_JETSON_CAMERA is not set
// (though it will error at runtime if called)
void camerad_thread() {
  LOGE("QCOM camerad_thread is not available on Jetson!");
  LOGE("Please set USE_JETSON_CAMERA=1 or USE_V4L2_CAMERA=1");
}

// Main entry point for Jetson camera daemon
void jetson_camerad_thread() {
  LOG("-- Starting Jetson camerad");

  // Initialize GStreamer
  gst_init(nullptr, nullptr);

  // Create VisionIpcServer (no OpenCL on Jetson for camera)
  VisionIpcServer vipc("camerad");

  // Get camera dimensions from environment
  int width = getEnvInt("CAMERA_WIDTH", 1280);
  int height = getEnvInt("CAMERA_HEIGHT", 720);
  int fps = getEnvInt("CAMERA_FPS", 20);

  LOG("Camera configuration: %dx%d @ %d fps", width, height, fps);

  // Determine which cameras to enable
  std::vector<JetsonCameraConfig> configs;

  // Road camera - enabled by default unless DISABLE_ROAD is set
  if (!getenv("DISABLE_ROAD")) {
    JetsonCameraConfig road_config = JETSON_ROAD_CONFIG;
    road_config.enabled = true;
    configs.push_back(road_config);
  }

  // Wide camera - enabled if WIDE_CAM is set
  if (getenv("WIDE_CAM")) {
    JetsonCameraConfig wide_config = JETSON_WIDE_CONFIG;
    wide_config.enabled = true;
    configs.push_back(wide_config);
  }

  // Driver camera - enabled if DRIVER_CAM is set
  if (getenv("DRIVER_CAM")) {
    JetsonCameraConfig driver_config = JETSON_DRIVER_CONFIG;
    driver_config.enabled = true;
    configs.push_back(driver_config);
  }

  if (configs.empty()) {
    LOGE("No cameras configured!");
    gst_deinit();
    return;
  }

  // Create VIPC buffers for each camera type
  for (const auto &config : configs) {
    vipc.create_buffers(config.stream_type, VIPC_BUFFER_COUNT, width, height);
    LOG("Created VIPC buffers for %s: %dx%d", config.publish_name, width, height);
  }

  vipc.start_listener();

  // Create camera objects and start threads
  std::vector<std::unique_ptr<JetsonCamera>> cameras;
  std::vector<std::thread> threads;

  for (const auto &config : configs) {
    auto cam = std::make_unique<JetsonCamera>(config, &vipc, width, height, fps);
    if (!cam->start()) {
      LOGE("Failed to start camera %d (%s)", config.camera_num, config.publish_name);
      continue;
    }

    // Start camera thread
    threads.emplace_back(&JetsonCamera::run, cam.get());
    cameras.push_back(std::move(cam));
  }

  if (cameras.empty()) {
    LOGE("No cameras started successfully");
    gst_deinit();
    return;
  }

  LOG("-- Jetson camerad running with %zu camera(s)", cameras.size());

  // Wait for exit signal
  while (!do_exit) {
    util::sleep_for(100);
  }

  LOG("-- Jetson camerad shutting down");

  // Stop all cameras
  for (auto &cam : cameras) {
    cam->stop();
  }

  // Join all threads
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  cameras.clear();

  // GStreamer cleanup
  gst_deinit();

  LOG("-- Jetson camerad shutdown complete");
}
