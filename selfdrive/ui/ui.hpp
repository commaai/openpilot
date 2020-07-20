#pragma once
#include "messaging.hpp"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#define NANOVG_GL3_IMPLEMENTATION
#define nvgCreate nvgCreateGL3
#else
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#define NANOVG_GLES3_IMPLEMENTATION
#define nvgCreate nvgCreateGLES3
#endif
#include <atomic>
#include <pthread.h>
#include "nanovg.h"

#include "common/mat.h"
#include "common/visionipc.h"
#include "common/visionimg.h"
#include "common/framebuffer.h"
#include "common/modeldata.h"
#include "sound.hpp"

#define STATUS_STOPPED 0
#define STATUS_DISENGAGED 1
#define STATUS_ENGAGED 2
#define STATUS_WARNING 3
#define STATUS_ALERT 4

#define NET_CONNECTED 0
#define NET_DISCONNECTED 1
#define NET_ERROR 2

#define COLOR_BLACK nvgRGBA(0, 0, 0, 255)
#define COLOR_BLACK_ALPHA(x) nvgRGBA(0, 0, 0, x)
#define COLOR_WHITE nvgRGBA(255, 255, 255, 255)
#define COLOR_WHITE_ALPHA(x) nvgRGBA(255, 255, 255, x)
#define COLOR_YELLOW nvgRGBA(218, 202, 37, 255)
#define COLOR_RED nvgRGBA(201, 34, 49, 255)
#define COLOR_OCHRE nvgRGBA(218, 111, 37, 255)

#ifndef QCOM
  #define UI_60FPS
#endif

#define UI_BUF_COUNT 4
//#define SHOW_SPEEDLIMIT 1
//#define DEBUG_TURN

const int vwp_w = 1920;
const int vwp_h = 1080;
const int nav_w = 640;
const int nav_ww= 760;
const int sbr_w = 300;
const int bdr_s = 30;
const int box_x = sbr_w+bdr_s;
const int box_y = bdr_s;
const int box_w = vwp_w-sbr_w-(bdr_s*2);
const int box_h = vwp_h-(bdr_s*2);
const int viz_w = vwp_w-(bdr_s*2);
const int ff_xoffset = 32;
const int header_h = 420;
const int footer_h = 280;
const int footer_y = vwp_h-bdr_s-footer_h;
const int settings_btn_h = 117;
const int settings_btn_w = 200;
const int settings_btn_x = 50;
const int settings_btn_y = 35;
const int home_btn_h = 180;
const int home_btn_w = 180;
const int home_btn_x = 60;
const int home_btn_y = vwp_h - home_btn_h - 40;

const int UI_FREQ = 30;   // Hz

const int MODEL_PATH_MAX_VERTICES_CNT = 98;
const int MODEL_LANE_PATH_CNT = 3;
const int TRACK_POINTS_MAX_CNT = 50 * 2;

const int SET_SPEED_NA = 255;

const uint8_t bg_colors[][4] = {
  [STATUS_STOPPED] = {0x07, 0x23, 0x39, 0xff},
  [STATUS_DISENGAGED] = {0x17, 0x33, 0x49, 0xff},
  [STATUS_ENGAGED] = {0x17, 0x86, 0x44, 0xff},
  [STATUS_WARNING] = {0xDA, 0x6F, 0x25, 0xff},
  [STATUS_ALERT] = {0xC9, 0x22, 0x31, 0xff},
};

typedef struct UIScene {
  int frontview;
  int fullview;

  int transformed_width, transformed_height;

  ModelData model;

  float mpc_x[50];
  float mpc_y[50];

  bool world_objects_visible;
  mat4 extrinsic_matrix;      // Last row is 0 so we can use mat4.

  float speedlimit;
  bool speedlimit_valid;

  bool is_rhd;
  bool map_valid;
  bool uilayout_sidebarcollapsed;
  bool uilayout_mapenabled;
  // responsive layout
  int ui_viz_rx;
  int ui_viz_rw;
  int ui_viz_ro;

  int front_box_x, front_box_y, front_box_width, front_box_height;

  std::string alert_text1;
  std::string alert_text2;
  std::string alert_type;
  cereal::ControlsState::AlertSize alert_size;

  // Used to show gps planner status
  bool gps_planner_active;

  cereal::HealthData::HwType hwType;
  int satelliteCount;
  uint8_t athenaStatus;

  cereal::ThermalData::Reader thermal;
  cereal::RadarState::LeadData::Reader lead_data[2];
  cereal::ControlsState::Reader controls_state;
  cereal::DriverState::Reader driver_state;
} UIScene;

typedef struct {
  float x, y;
}vertex_data;

typedef struct {
  vertex_data v[MODEL_PATH_MAX_VERTICES_CNT];
  int cnt;
} model_path_vertices_data;

typedef struct {
  vertex_data v[TRACK_POINTS_MAX_CNT];
  int cnt;
} track_vertices_data;


typedef struct UIState {
  pthread_mutex_t lock;

  // framebuffer
  FramebufferState *fb;
  int fb_w, fb_h;

  // NVG
  NVGcontext *vg;

  // fonts and images
  int font_courbd;
  int font_sans_regular;
  int font_sans_semibold;
  int font_sans_bold;
  int img_wheel;
  int img_turn;
  int img_face;
  int img_map;
  int img_button_settings;
  int img_button_home;
  int img_battery;
  int img_battery_charging;
  int img_network[6];

  // sockets
  SubMaster *sm;
  PubMaster *pm;

  cereal::UiLayoutState::App active_app;

  // vision state
  bool vision_connected;
  bool vision_connect_firstrun;
  int ipc_fd;

  VIPCBuf bufs[UI_BUF_COUNT];
  VIPCBuf front_bufs[UI_BUF_COUNT];
  int cur_vision_idx;
  int cur_vision_front_idx;

  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT];
  EGLImageKHR khr[UI_BUF_COUNT];
  void *priv_hnds[UI_BUF_COUNT];
  GLuint frame_front_texs[UI_BUF_COUNT];
  EGLImageKHR khr_front[UI_BUF_COUNT];
  void *priv_hnds_front[UI_BUF_COUNT];

  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;

  int rgb_width, rgb_height, rgb_stride;
  size_t rgb_buf_len;
  mat4 rgb_transform;

  int rgb_front_width, rgb_front_height, rgb_front_stride;
  size_t rgb_front_buf_len;

  UIScene scene;
  bool awake;

  // timeouts
  int awake_timeout;
  int controls_timeout;
  int speed_lim_off_timeout;
  int is_metric_timeout;
  int longitudinal_control_timeout;
  int limit_set_speed_timeout;
  int hardware_timeout;
  int last_athena_ping_timeout;

  bool controls_seen;

  uint64_t last_athena_ping;
  int status;
  bool is_metric;
  bool longitudinal_control;
  bool limit_set_speed;
  float speed_lim_off;
  bool is_ego_over_limit;
  float alert_blinking_alpha;
  bool alert_blinked;
  bool started;
  bool preview_started;
  bool vision_seen;

  std::atomic<float> light_sensor;

  int touch_fd;

  GLuint frame_vao[2], frame_vbo[2], frame_ibo[2];
  mat4 rear_frame_mat, front_frame_mat;

  model_path_vertices_data model_path_vertices[MODEL_LANE_PATH_CNT * 2];

  track_vertices_data track_vertices[2];

  Sound sound;
} UIState;

// API
void ui_draw_vision_alert(UIState *s, cereal::ControlsState::AlertSize va_size, int va_color,
                          const char* va_text1, const char* va_text2);
void ui_draw(UIState *s);
void ui_draw_sidebar(UIState *s);
void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, float r = 0, int width = 0);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGpaint &paint, float r = 0);
void ui_nvg_init(UIState *s);
