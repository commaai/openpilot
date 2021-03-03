#pragma once
#include "messaging.hpp"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#define NANOVG_GL3_IMPLEMENTATION
#define nvgCreate nvgCreateGL3
#else
#include <GLES3/gl3.h>
#define NANOVG_GLES3_IMPLEMENTATION
#define nvgCreate nvgCreateGLES3
#endif

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <sstream>

#include "nanovg.h"

#include "common/mat.h"
#include "common/visionimg.h"
#include "common/framebuffer.h"
#include "common/modeldata.h"
#include "common/params.h"
#include "common/glutil.h"
#include "common/transformations/orientation.hpp"
#include "sound.hpp"
#include "visionipc.h"
#include "visionipc_client.h"

#define COLOR_BLACK nvgRGBA(0, 0, 0, 255)
#define COLOR_BLACK_ALPHA(x) nvgRGBA(0, 0, 0, x)
#define COLOR_WHITE nvgRGBA(255, 255, 255, 255)
#define COLOR_WHITE_ALPHA(x) nvgRGBA(255, 255, 255, x)
#define COLOR_YELLOW nvgRGBA(218, 202, 37, 255)
#define COLOR_RED nvgRGBA(201, 34, 49, 255)

#define UI_BUF_COUNT 4

typedef struct Rect {
  int x, y, w, h;
  int centerX() const { return x + w / 2; }
  int centerY() const { return y + h / 2; }
  int right() const { return x + w; }
  int bottom() const { return y + h; }
  bool ptInRect(int px, int py) const {
    return px >= x && px < (x + w) && py >= y && py < (y + h);
  }
} Rect;

const int sbr_w = 300;
const int bdr_s = 30;
const int header_h = 420;
const int footer_h = 280;
const Rect settings_btn = {50, 35, 200, 117};
const Rect home_btn = {60, 1080 - 180 - 40, 180, 180};

const int UI_FREQ = 20;   // Hz

typedef enum NetStatus {
  NET_CONNECTED,
  NET_DISCONNECTED,
  NET_ERROR,
} NetStatus;

typedef enum UIStatus {
  STATUS_OFFROAD,
  STATUS_DISENGAGED,
  STATUS_ENGAGED,
  STATUS_WARNING,
  STATUS_ALERT,
} UIStatus;

static std::map<UIStatus, NVGcolor> bg_colors = {
#ifndef QT_GUI_LIB
  {STATUS_OFFROAD, nvgRGBA(0x07, 0x23, 0x39, 0xf1)},
#else
  {STATUS_OFFROAD, nvgRGBA(0x0, 0x0, 0x0, 0xff)},
#endif
  {STATUS_DISENGAGED, nvgRGBA(0x17, 0x33, 0x49, 0xc8)},
  {STATUS_ENGAGED, nvgRGBA(0x17, 0x86, 0x44, 0xf1)},
  {STATUS_WARNING, nvgRGBA(0xDA, 0x6F, 0x25, 0xf1)},
  {STATUS_ALERT, nvgRGBA(0xC9, 0x22, 0x31, 0xf1)},
};

typedef struct {
  float x, y;
} vertex_data;

typedef struct {
  vertex_data v[TRAJECTORY_SIZE * 2];
  int cnt;
} line_vertices_data;

typedef struct UIScene {

  mat3 view_from_calib;
  bool world_objects_visible;

  bool is_rhd;
  bool driver_view;

  std::string alert_text1;
  std::string alert_text2;
  std::string alert_type;
  float alert_blinking_rate;
  cereal::ControlsState::AlertSize alert_size;

  cereal::PandaState::PandaType pandaType;
  NetStatus athenaStatus;

  cereal::DeviceState::Reader deviceState;
  cereal::RadarState::LeadData::Reader lead_data[2];
  cereal::CarState::Reader car_state;
  cereal::ControlsState::Reader controls_state;
  cereal::DriverState::Reader driver_state;
  cereal::DriverMonitoringState::Reader dmonitoring_state;

  // gps
  int satelliteCount;
  bool gpsOK;

  // modelV2
  float lane_line_probs[4];
  float road_edge_stds[2];
  line_vertices_data track_vertices;
  line_vertices_data lane_line_vertices[4];
  line_vertices_data road_edge_vertices[2];

  // lead
  vertex_data lead_vertices[2];

  float light_sensor, accel_sensor, gyro_sensor;
  bool started, ignition, is_metric, longitudinal_control;
  uint64_t started_frame;
} UIScene;

typedef struct UIState {
  VisionIpcClient * vipc_client;
  VisionIpcClient * vipc_client_front;
  VisionIpcClient * vipc_client_rear;
  VisionBuf * last_frame;

  // framebuffer
  std::unique_ptr<FrameBuffer> fb;
  int fb_w, fb_h;

  // NVG
  NVGcontext *vg;

  // images
  std::map<std::string, int> images;

  SubMaster *sm;

  Sound *sound;
  UIStatus status;
  UIScene scene;
  cereal::UiLayoutState::App active_app;

  // graphics
  std::unique_ptr<GLShader> gl_shader;
  std::unique_ptr<EGLImageTexture> texture[UI_BUF_COUNT];

  GLuint frame_vao[2], frame_vbo[2], frame_ibo[2];
  mat4 rear_frame_mat, front_frame_mat;

  // device state
  bool awake;

  bool sidebar_collapsed;
  Rect video_rect, viz_rect;
  float car_space_transform[6];
} UIState;

void ui_init(UIState *s);
void ui_update(UIState *s);

int write_param_float(float param, const char* param_name, bool persistent_param = false);
template <class T>
int read_param(T* param, const char *param_name, bool persistent_param = false){
  T param_orig = *param;
  char *value;
  size_t sz;

  int result = Params(persistent_param).read_db_value(param_name, &value, &sz);
  if (result == 0){
    std::string s = std::string(value, sz); // value is not null terminated
    free(value);

    // Parse result
    std::istringstream iss(s);
    iss >> *param;

    // Restore original value if parsing failed
    if (iss.fail()) {
      *param = param_orig;
      result = -1;
    }
  }
  return result;
}
