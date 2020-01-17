#ifndef _UI_H
#define _UI_H

#include <GLES3/gl3.h>
#include <EGL/egl.h>

#include "nanovg.h"

#include "common/mat.h"
#include "common/visionipc.h"
#include "common/framebuffer.h"
#include "common/modeldata.h"
#include "messaging.hpp"

#include "cereal/gen/c/log.capnp.h"

#include "sound.hpp"

#define STATUS_STOPPED 0
#define STATUS_DISENGAGED 1
#define STATUS_ENGAGED 2
#define STATUS_WARNING 3
#define STATUS_ALERT 4

#define ALERTSIZE_NONE 0
#define ALERTSIZE_SMALL 1
#define ALERTSIZE_MID 2
#define ALERTSIZE_FULL 3

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
const int header_h = 420;
const int footer_h = 280;
const int footer_y = vwp_h-bdr_s-footer_h;

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

  float v_cruise;
  uint64_t v_cruise_update_ts;
  float v_ego;
  bool decel_for_model;

  float speedlimit;
  bool speedlimit_valid;
  bool map_valid;

  float curvature;
  int engaged;
  bool engageable;
  bool monitoring_active;

  bool uilayout_sidebarcollapsed;
  bool uilayout_mapenabled;
  // responsive layout
  int ui_viz_rx;
  int ui_viz_rw;
  int ui_viz_ro;

  int lead_status;
  float lead_d_rel, lead_y_rel, lead_v_rel;

  int front_box_x, front_box_y, front_box_width, front_box_height;

  uint64_t alert_ts;
  char alert_text1[1024];
  char alert_text2[1024];
  uint8_t alert_size;
  float alert_blinkingrate;

  float awareness_status;

  // Used to show gps planner status
  bool gps_planner_active;
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
  pthread_cond_t bg_cond;

  // framebuffer
  FramebufferState *fb;
  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;

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

  // sockets
  Context *ctx;
  SubSocket *model_sock;
  SubSocket *controlsstate_sock;
  SubSocket *livecalibration_sock;
  SubSocket *radarstate_sock;
  SubSocket *map_data_sock;
  SubSocket *uilayout_sock;
  Poller * poller;

  int active_app;

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

  GLuint line_program;
  GLint line_pos_loc, line_color_loc;
  GLint line_transform_loc;

  int rgb_width, rgb_height, rgb_stride;
  size_t rgb_buf_len;
  mat4 rgb_transform;

  int rgb_front_width, rgb_front_height, rgb_front_stride;
  size_t rgb_front_buf_len;

  UIScene scene;
  bool awake;

  // timeouts
  int awake_timeout;
  int volume_timeout;
  int controls_timeout;
  int alert_sound_timeout;
  int speed_lim_off_timeout;
  int is_metric_timeout;
  int longitudinal_control_timeout;
  int limit_set_speed_timeout;

  bool controls_seen;

  int status;
  bool is_metric;
  bool longitudinal_control;
  bool limit_set_speed;
  float speed_lim_off;
  bool is_ego_over_limit;
  char alert_type[64];
  AudibleAlert alert_sound;
  int alert_size;
  float alert_blinking_alpha;
  bool alert_blinked;

  float light_sensor;

  int touch_fd;

  // Hints for re-calculations and redrawing
  bool model_changed;
  bool livempc_or_radarstate_changed;

  GLuint frame_vao[2], frame_vbo[2], frame_ibo[2];
  mat4 rear_frame_mat, front_frame_mat;

  model_path_vertices_data model_path_vertices[MODEL_LANE_PATH_CNT * 2];

  track_vertices_data track_vertices[2];
} UIState;

// API
void ui_draw_vision_alert(UIState *s, int va_size, int va_color,
                          const char* va_text1, const char* va_text2); 
void ui_draw(UIState *s);
void ui_nvg_init(UIState *s);

#endif
