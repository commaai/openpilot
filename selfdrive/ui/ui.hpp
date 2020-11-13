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
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>
#include <sstream>

#include "nanovg.h"

#include "common/mat.h"
#include "common/visionipc.h"
#include "common/visionimg.h"
#include "common/framebuffer.h"
#include "common/modeldata.h"
#include "common/params.h"
#include "sound.hpp"

// TODO: this is also hardcoded in common/transformations/camera.py
// TODO: choose based on frame input size
#ifdef QCOM2
const mat3 intrinsic_matrix = (mat3){{
  2648.0, 0.0, 1928.0/2,
  0.0, 2648.0, 1208.0/2,
  0.0,   0.0,   1.0
}};
#else
const mat3 intrinsic_matrix = (mat3){{
  910., 0., 1164.0/2,
  0., 910., 874.0/2,
  0.,   0.,   1.
}};
#endif

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

const int MODEL_PATH_MAX_VERTICES_CNT = TRAJECTORY_SIZE*2;
const int TRACK_POINTS_MAX_CNT = TRAJECTORY_SIZE*4;

const int SET_SPEED_NA = 255;

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
#ifdef QCOM
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
  vertex_data v[MODEL_PATH_MAX_VERTICES_CNT];
  int cnt;
} line_vertices_data;

typedef struct {
  vertex_data v[TRACK_POINTS_MAX_CNT];
  int cnt;
} track_vertices_data;

typedef struct UIScene {
  UIStatus status = STATUS_OFFROAD;
  bool started = false, is_rhd = false, frontview = false, longitudinal_control = false;

  int satelliteCount = -1;
  NetStatus athenaStatus = NET_DISCONNECTED;
  
  // alert
  float alert_blinking_rate = 0;
  std::string alert_type, alert_text1, alert_text2;
  cereal::ControlsState::AlertSize alert_size = cereal::ControlsState::AlertSize::NONE;

  // controlsState
  float v_cruise = 0., v_ego = 0.;
  bool controls_enabled = false, engageable = false, decel_for_model = false;

  // health
  bool ignition = false;
  cereal::HealthData::HwType hwType = cereal::HealthData::HwType::UNKNOWN;

  // driverState
  float face_position[2] = {};
  bool face_detected = false;

  // thermal
  cereal::ThermalData::ThermalStatus thermal_status;
  cereal::ThermalData::NetworkType network_type;
  cereal::ThermalData::NetworkStrength network_strength;
  std::string battery_status;
  int16_t battery_percent;
  float ambient;

  // sensors
  float light_sensor = 0, accel_sensor = 0, gyro_sensor = 0;

  // paramaters
  bool is_metric = false;

  // modelV2
  float lane_line_probs[4];
  float road_edge_stds[2];
  track_vertices_data track_vertices;
  line_vertices_data lane_line_vertices[4];
  line_vertices_data road_edge_vertices[2];

  // radarState
  struct LeadData{
    bool status;
    float d_rel, v_rel, y_rel;
    vertex_data vd;
  } lead[2] = {};
} UIScene;

struct UIState;
class UIStateThread {
public:
  UIStateThread(UIState *s);
  ~UIStateThread();
  void getScene(UIScene *scene);
  void stop();
  bool car_space_to_full_frame(float in_x, float in_y, float in_z, float *out_x, float *out_y, float margin=0.0);
private:
  enum class UIStateSync {
    kFetch = 0,
    kReady,
  };
  void threadMain();
  void update();
  void updateSockets();
  void update_params();
  void handleAlert();
  void update_model(const cereal::ModelDataV2::Reader &model);
  
  UIState *ui_state;

  SubMaster sm;
  UIScene scene = {};
  uint64_t started_frame = 0;
  std::atomic<bool> run = true;
  mat4 extrinsic_matrix = {};  // Last row is 0 so we can use mat4.

  std::mutex lock;
  UIStateSync state_type = UIStateSync::kFetch;
  std::condition_variable cv;
  std::thread thread;
};

typedef struct UIState {
  UIState() : state_thread(this) {}
  // framebuffer
  FramebufferState *fb;
  int fb_w, fb_h;

  // NVG
  NVGcontext *vg;

  // fonts and images
  int font_sans_regular;
  int font_sans_semibold;
  int font_sans_bold;
  int img_wheel;
  int img_turn;
  int img_face;
  int img_button_settings;
  int img_button_home;
  int img_battery;
  int img_battery_charging;
  int img_network[6];

  SubMaster *sm;

  Sound *sound;
  UIScene scene = {};

  // vision state
  bool vision_connected = false;
  VisionStream stream = {};

  // graphics
  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT];
  EGLImageKHR khr[UI_BUF_COUNT];
  void *priv_hnds[UI_BUF_COUNT];

  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;
  GLuint frame_vao[2], frame_vbo[2], frame_ibo[2];
  mat4 rear_frame_mat, front_frame_mat;

  bool awake = false;

  Rect video_rect = {}, viz_rect = {};
  int ui_viz_ro = 0;

  UIStateThread state_thread;
  // accessed from ui & state thread
  std::atomic<cereal::UiLayoutState::App> active_app = cereal::UiLayoutState::App::HOME;
  std::atomic<bool> sidebar_collapsed = false, world_objects_visible = false;
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
