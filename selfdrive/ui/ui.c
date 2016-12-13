#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>

#include <cutils/properties.h>

#include <GLES3/gl3.h>
#include <EGL/eglext.h>

#include <czmq.h>

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "common/timing.h"
#include "common/util.h"
#include "common/mat.h"

#include "common/framebuffer.h"
#include "common/visionipc.h"
#include "common/modeldata.h"

#include "cereal/gen/c/log.capnp.h"

#include "touch.h"

#define UI_BUF_COUNT 4
typedef struct UIBuf {
  int fd;
  size_t len;
  void* addr;
} UIBuf;

typedef struct UIScene {

  int frontview;

  uint8_t *bgr_ptr;
  int big_box_x, big_box_y, big_box_width, big_box_height;

  int transformed_width, transformed_height;

  uint64_t model_ts;
  ModelData model;

  mat3 big_box_transform; // transformed box -> big box

  float v_cruise;
  float v_ego;
  float angle_steers;
  int engaged;

  int lead_status;
  float lead_d_rel, lead_y_rel, lead_v_rel;

  uint8_t *bgr_front_ptr;
  int front_box_x, front_box_y, front_box_width, front_box_height;

  char alert_text1[1024];
  char alert_text2[1024];

  float awareness_status;
} UIScene;


typedef struct UIState {
  pthread_mutex_t lock;

  TouchState touch;

  FramebufferState *fb;

  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;

  NVGcontext *vg;
  int font;



  zsock_t *model_sock;
  void* model_sock_raw;
  zsock_t *live100_sock;
  void* live100_sock_raw;
  zsock_t *livecalibration_sock;
  void* livecalibration_sock_raw;
  zsock_t *live20_sock;
  void* live20_sock_raw;

  // base ui
  uint64_t last_base_update;
  uint64_t last_rx_bytes;
  uint64_t last_tx_bytes;
  char serial[4096];
  const char* dongle_id;
  char base_text[4096];
  int wifi_enabled;
  int ap_enabled;
  int board_connected;

  // vision state

  bool vision_connected;
  bool vision_connect_firstrun;
  int ipc_fd;

  VisionUIBufs vision_bufs;
  UIBuf bufs[UI_BUF_COUNT];
  UIBuf front_bufs[UI_BUF_COUNT];
  int cur_vision_idx;
  int cur_vision_front_idx;

  GLuint frame_program;

  GLuint frame_tex;
  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;

  GLuint line_program;
  GLint line_pos_loc, line_color_loc;
  GLint line_transform_loc;

  unsigned int rgb_width, rgb_height;
  mat4 rgb_transform;

  unsigned int rgb_front_width, rgb_front_height;
  GLuint frame_front_tex;

  UIScene scene;
  
  bool awake;
  int awake_timeout;
} UIState;

static void set_awake(UIState *s, bool awake) {
  if (awake) {
    // 30 second timeout
    s->awake_timeout = 30;
  }
  if (s->awake != awake) {
    s->awake = awake;

    // TODO: actually turn off the screen and not just the backlight
    FILE *f = fopen("/sys/class/leds/lcd-backlight/brightness", "wb");
    if (f != NULL) {
      if (awake) {
        fprintf(f, "205");
      } else {
        fprintf(f, "0");
      }
      fclose(f);
    }
  }
}

static bool activity_running() {
  return system("dumpsys activity activities | grep mFocusedActivity > /dev/null") == 0;
}

static void start_settings_activity(const char* name) {
  char launch_cmd[1024];
  snprintf(launch_cmd, sizeof(launch_cmd),
           "am start -W --ez :settings:show_fragment_as_subsetting true -n 'com.android.settings/.%s'", name);
  system(launch_cmd);
}

static void wifi_pressed() {
  start_settings_activity("Settings$WifiSettingsActivity");
}
static void ap_pressed() {
  start_settings_activity("Settings$TetherSettingsActivity");
}

static int wifi_enabled(UIState *s) {
  return s->wifi_enabled;
}

static int ap_enabled(UIState *s) {
  return s->ap_enabled;
}

typedef struct Button {
  const char* label;
  int x, y, w, h;
  void (*pressed)(void);
  int (*enabled)(UIState *);
} Button;
static const Button buttons[] = {
  {
    .label = "wifi",
    .x = 400, .y = 730, .w = 250, .h = 250,
    .pressed = wifi_pressed,
    .enabled = wifi_enabled,
  },
  {
    .label = "ap",
    .x = 1300, .y = 730, .w = 250, .h = 250,
    .pressed = ap_pressed,
    .enabled = ap_enabled,
  }
};

// transform from road space into little-box (used for drawing path)
static const mat3 path_transform = {{
   1.29149378e+00, -2.30320967e-01, -3.02391994e+01,
  -1.72449331e-15, -2.12045399e-02,  5.03539175e+01,
  -3.24378996e-17, -1.38821089e-03,  1.06663412e+00,
}};

static const char frame_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_FragColor = texture2D(uTexture, vTexCoord.xy);\n"
  "}\n";

static const char line_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aColor;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vColor = aColor;\n"
  "}\n";

static const char line_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_FragColor = vColor;\n"
  "}\n";

static GLuint load_shader(GLenum shaderType, const char *src) {
  GLint status = 0, len = 0;
  GLuint shader;

  if (!(shader = glCreateShader(shaderType)))
    return 0;

  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

  if (status)
    return shader;

  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len) {
    char *msg = malloc(len);
    if (msg) {
      glGetShaderInfoLog(shader, len, NULL, msg);
      msg[len-1] = 0;
      fprintf(stderr, "error compiling shader:\n%s\n", msg);
      free(msg);
    }
  }
  glDeleteShader(shader);
  return 0;
}

static GLuint load_program(const char *vert_src, const char *frag_src) {
  GLuint vert, frag, prog;
  GLint status = 0, len = 0;

  if (!(vert = load_shader(GL_VERTEX_SHADER, vert_src)))
    return 0;
  if (!(frag = load_shader(GL_FRAGMENT_SHADER, frag_src)))
    goto fail_frag;
  if (!(prog = glCreateProgram()))
    goto fail_prog;

  glAttachShader(prog, vert);
  glAttachShader(prog, frag);
  glLinkProgram(prog);

  glGetProgramiv(prog, GL_LINK_STATUS, &status);
  if (status)
    return prog;

  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
  if (len) {
    char *buf = (char*) malloc(len);
    if (buf) {
      glGetProgramInfoLog(prog, len, NULL, buf);
      buf[len-1] = 0;
      fprintf(stderr, "error linking program:\n%s\n", buf);
      free(buf);
    }
  }
  glDeleteProgram(prog);
fail_prog:
  glDeleteShader(frag);
fail_frag:
  glDeleteShader(vert);
  return 0;
}

static const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

// frame from 4/3 to 16/9 with a 2x zoon
static const mat4 frame_transform = {{
  2*(4./3.)/(16./9.), 0.0, 0.0, 0.0,
               0.0, 2.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0,
}};



static void ui_init(UIState *s) {
  memset(s, 0, sizeof(UIState));

  pthread_mutex_init(&s->lock, NULL);

  // init connections
  s->model_sock = zsock_new_sub(">tcp://127.0.0.1:8009", "");
  assert(s->model_sock);
  s->model_sock_raw = zsock_resolve(s->model_sock);

  s->live100_sock = zsock_new_sub(">tcp://127.0.0.1:8007", "");
  assert(s->live100_sock);
  s->live100_sock_raw = zsock_resolve(s->live100_sock);

  s->livecalibration_sock = zsock_new_sub(">tcp://127.0.0.1:8019", "");
  assert(s->livecalibration_sock);
  s->livecalibration_sock_raw = zsock_resolve(s->livecalibration_sock);

  s->live20_sock = zsock_new_sub(">tcp://127.0.0.1:8012", "");
  assert(s->live20_sock);
  s->live20_sock_raw = zsock_resolve(s->live20_sock);

  s->ipc_fd = -1;

  touch_init(&s->touch);

  // init display
  s->fb = framebuffer_init("ui", 0x00001000,
                           &s->display, &s->surface, &s->fb_w, &s->fb_h);
  assert(s->fb);


  // init base
  property_get("ro.serialno", s->serial, "");

  s->dongle_id = getenv("DONGLE_ID");
  if (!s->dongle_id) s->dongle_id = "(null)";


  // init drawing
  s->vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
  assert(s->vg);
  //s->font = nvgCreateFont(s->vg, "sans-bold", "../assets/Roboto-Bold.ttf");
  s->font = nvgCreateFont(s->vg, "Bold", "../assets/courbd.ttf");
  assert(s->font >= 0);

  // init gl

  s->frame_program = load_program(frame_vertex_shader, frame_fragment_shader);
  assert(s->frame_program);

  s->frame_pos_loc = glGetAttribLocation(s->frame_program, "aPosition");
  s->frame_texcoord_loc = glGetAttribLocation(s->frame_program, "aTexCoord");

  s->frame_texture_loc = glGetUniformLocation(s->frame_program, "uTexture");
  s->frame_transform_loc = glGetUniformLocation(s->frame_program, "uTransform");


  s->line_program = load_program(line_vertex_shader, line_fragment_shader);
  assert(s->line_program);

  s->line_pos_loc = glGetAttribLocation(s->line_program, "aPosition");
  s->line_color_loc = glGetAttribLocation(s->line_program, "aColor");
  s->line_transform_loc = glGetUniformLocation(s->line_program, "uTransform");

  glViewport(0, 0, s->fb_w, s->fb_h);

  glDisable(GL_DEPTH_TEST);

  assert(glGetError() == GL_NO_ERROR);

  // set awake
  set_awake(s, true);
}


static void ui_init_vision(UIState *s, const VisionUIBufs vision_bufs, const int* fds) {
  assert(vision_bufs.num_bufs == UI_BUF_COUNT);
  assert(vision_bufs.num_front_bufs == UI_BUF_COUNT);

  for (int i=0; i<vision_bufs.num_bufs; i++) {
    if (s->bufs[i].addr) {
      munmap(s->bufs[i].addr, vision_bufs.buf_len);
      s->bufs[i].addr = NULL;
      close(s->bufs[i].fd);
    }
    s->bufs[i].fd = fds[i];
    s->bufs[i].len = vision_bufs.buf_len;
    s->bufs[i].addr = mmap(NULL, s->bufs[i].len,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED, s->bufs[i].fd, 0);
    // printf("b %d %p\n", bufs[i].fd, bufs[i].addr);
    assert(s->bufs[i].addr != MAP_FAILED);
  }
  for (int i=0; i<vision_bufs.num_front_bufs; i++) {
    if (s->front_bufs[i].addr) {
      munmap(s->front_bufs[i].addr, vision_bufs.buf_len);
      s->front_bufs[i].addr = NULL;
      close(s->front_bufs[i].fd);
    }
    s->front_bufs[i].fd = fds[vision_bufs.num_bufs + i];
    s->front_bufs[i].len = vision_bufs.front_buf_len;
    s->front_bufs[i].addr = mmap(NULL, s->front_bufs[i].len,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED, s->front_bufs[i].fd, 0);
    // printf("f %d %p\n", front_bufs[i].fd, front_bufs[i].addr);
    assert(s->front_bufs[i].addr != MAP_FAILED);
  }

  s->cur_vision_idx = -1;
  s->cur_vision_front_idx = -1;

  s->scene = (UIScene){
    .frontview = 0,
    .big_box_x = vision_bufs.big_box_x,
    .big_box_y = vision_bufs.big_box_y,
    .big_box_width = vision_bufs.big_box_width,
    .big_box_height = vision_bufs.big_box_height,
    .transformed_width = vision_bufs.transformed_width,
    .transformed_height = vision_bufs.transformed_height,
    .front_box_x = vision_bufs.front_box_x,
    .front_box_y = vision_bufs.front_box_y,
    .front_box_width = vision_bufs.front_box_width,
    .front_box_height = vision_bufs.front_box_height,

    // only used when ran without controls. overwridden by liveCalibration messages.
    .big_box_transform = (mat3){{
      1.16809241e+00,  -3.18601797e-02,   7.42513711e+01,
      7.97437780e-02,   1.09117765e+00,   5.71824220e+01,
      8.67937981e-05,  -7.68221181e-05,   1.00196836e+00,
    }},
  };

  s->vision_bufs = vision_bufs;

  s->rgb_width = vision_bufs.width;
  s->rgb_height = vision_bufs.height;

  s->rgb_front_width = vision_bufs.front_width;
  s->rgb_front_height = vision_bufs.front_height;

  s->rgb_transform = (mat4){{
    2.0/s->rgb_width, 0.0, 0.0, -1.0,
    0.0, 2.0/s->rgb_height, 0.0, -1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};
}

static void ui_update_frame(UIState *s) {
  assert(glGetError() == GL_NO_ERROR);

  UIScene *scene = &s->scene;

  if (scene->frontview && scene->bgr_front_ptr) {
    // load front frame texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s->frame_front_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    s->rgb_front_width, s->rgb_front_height,
                    GL_RGB, GL_UNSIGNED_BYTE, scene->bgr_front_ptr);
  } else if (!scene->frontview && scene->bgr_ptr) {
    // load frame texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s->frame_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    s->rgb_width, s->rgb_height,
                    GL_RGB, GL_UNSIGNED_BYTE, scene->bgr_ptr);
  }

  assert(glGetError() == GL_NO_ERROR);
}

static void draw_rgb_box(UIState *s, int x, int y, int w, int h, uint32_t color) {
  const struct {
    uint32_t x, y, color;
  } verts[] = {
    {x, y, color},
    {x+w, y, color},
    {x+w, y+h, color},
    {x, y+h, color},
    {x, y, color},
  };

  glUseProgram(s->line_program);

  mat4 out_mat = matmul(device_transform,
                        matmul(frame_transform, s->rgb_transform));
  glUniformMatrix4fv(s->line_transform_loc, 1, GL_TRUE, out_mat.v);

  glEnableVertexAttribArray(s->line_pos_loc);
  glVertexAttribPointer(s->line_pos_loc, 2, GL_UNSIGNED_INT, GL_FALSE, sizeof(verts[0]), &verts[0].x);

  glEnableVertexAttribArray(s->line_color_loc);
  glVertexAttribPointer(s->line_color_loc, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(verts[0]), &verts[0].color);

  assert(glGetError() == GL_NO_ERROR);
  glDrawArrays(GL_LINE_STRIP, 0, ARRAYSIZE(verts));
}

static void ui_draw_transformed_box(UIState *s, uint32_t color) {
  const UIScene *scene = &s->scene;

  const mat3 bbt = scene->big_box_transform;

  struct {
    vec3 pos;
    uint32_t color;
  } verts[] = {
    {matvecmul3(bbt, (vec3){{0.0, 0.0, 1.0,}}), color},
    {matvecmul3(bbt, (vec3){{scene->transformed_width, 0.0, 1.0,}}), color},
    {matvecmul3(bbt, (vec3){{scene->transformed_width, scene->transformed_height, 1.0,}}), color},
    {matvecmul3(bbt, (vec3){{0.0, scene->transformed_height, 1.0,}}), color},
    {matvecmul3(bbt, (vec3){{0.0, 0.0, 1.0,}}), color},
  };

  for (int i=0; i<ARRAYSIZE(verts); i++) {
    verts[i].pos.v[0] = scene->big_box_x + verts[i].pos.v[0] / verts[i].pos.v[2];
    verts[i].pos.v[1] = scene->big_box_y + verts[i].pos.v[1] / verts[i].pos.v[2];
    verts[i].pos.v[1] = s->rgb_height - verts[i].pos.v[1];
  }

  glUseProgram(s->line_program);

  mat4 out_mat = matmul(device_transform,
                        matmul(frame_transform, s->rgb_transform));
  glUniformMatrix4fv(s->line_transform_loc, 1, GL_TRUE, out_mat.v);

  glEnableVertexAttribArray(s->line_pos_loc);
  glVertexAttribPointer(s->line_pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(verts[0]), &verts[0].pos.v[0]);

  glEnableVertexAttribArray(s->line_color_loc);
  glVertexAttribPointer(s->line_color_loc, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(verts[0]), &verts[0].color);

  assert(glGetError() == GL_NO_ERROR);
  glDrawArrays(GL_LINE_STRIP, 0, ARRAYSIZE(verts));
}

// TODO: refactor with draw_path
static void draw_cross(UIState *s, float x_in, float y_in, float sz, NVGcolor color) {
  const UIScene *scene = &s->scene;

  const float meter_width = 20;
  const float car_x = 160;
  const float car_y = 570 + meter_width * 8;

  nvgSave(s->vg);

  // path coords are worked out in rgb-box space
  nvgTranslate(s->vg, 240.0f, 0.0);

  // zooom in 2x
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2);
  nvgScale(s->vg, 2.0, 2.0);

  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);

  nvgBeginPath(s->vg);
  nvgStrokeColor(s->vg, color);
  nvgStrokeWidth(s->vg, 5);

  float px = -y_in * meter_width + car_x;
  float py = x_in * -meter_width + car_y;

  vec3 dxy = matvecmul3(path_transform, (vec3){{px, py, 1.0}});
  dxy.v[0] /= dxy.v[2]; dxy.v[1] /= dxy.v[2]; dxy.v[2] = 1.0f; //paranoia
  vec3 bbpos = matvecmul3(scene->big_box_transform, dxy);

  float x = scene->big_box_x + bbpos.v[0]/bbpos.v[2];
  float y = scene->big_box_y + bbpos.v[1]/bbpos.v[2];

  nvgMoveTo(s->vg, x-sz, y);
  nvgLineTo(s->vg, x+sz, y);

  nvgMoveTo(s->vg, x, y-sz);
  nvgLineTo(s->vg, x, y+sz);

  nvgStroke(s->vg);

  nvgRestore(s->vg);
}

static void draw_path(UIState *s, const float* points, float off, NVGcolor color) {
  const UIScene *scene = &s->scene;

  const float meter_width = 20;
  const float car_x = 160;
  const float car_y = 570 + meter_width * 8;

  nvgSave(s->vg);

  // path coords are worked out in rgb-box space
  nvgTranslate(s->vg, 240.0f, 0.0);

  // zooom in 2x
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2);
  nvgScale(s->vg, 2.0, 2.0);

  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);


  nvgBeginPath(s->vg);
  nvgStrokeColor(s->vg, color);
  nvgStrokeWidth(s->vg, 5);

  for (int i=0; i<50; i++) {
    float px = (-points[i] + off) * meter_width + car_x;
    float py = (float)i * -meter_width + car_y;

    vec3 dxy = matvecmul3(path_transform, (vec3){{px, py, 1.0}});
    dxy.v[0] /= dxy.v[2]; dxy.v[1] /= dxy.v[2]; dxy.v[2] = 1.0f; //paranoia
    vec3 bbpos = matvecmul3(scene->big_box_transform, dxy);

    float x = scene->big_box_x + bbpos.v[0]/bbpos.v[2];
    float y = scene->big_box_y + bbpos.v[1]/bbpos.v[2];

    if (i == 0) {
      nvgMoveTo(s->vg, x, y);
    } else {
      nvgLineTo(s->vg, x, y);
    }
  }

  nvgStroke(s->vg);

  nvgRestore(s->vg);
}

static void draw_model_path(UIState *s, const PathData path, NVGcolor color) {
  float var = min(path.std, 0.7);
  draw_path(s, path.points, 0.0, color);
  color.a /= 4;
  draw_path(s, path.points, -var, color);
  draw_path(s, path.points, var, color);
}

static double calc_curvature(float v_ego, float angle_steers) {
  const double deg_to_rad = M_PI / 180.0f;
  const double slip_fator = 0.0014;
  const double steer_ratio = 15.3;
  const double wheel_base = 2.67;

  const double angle_offset = 0.0;

  double angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad;
  double curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego*v_ego));
  return curvature;
}

static void draw_steering(UIState *s, float v_ego, float angle_steers) {
  double curvature = calc_curvature(v_ego, angle_steers);

  float points[50];
  for (int i=0; i<50; i++) {
    float y_actual = i * tan(asin(clamp(i * curvature, -0.999, 0.999))/2.);
    points[i] = y_actual;
  }

  draw_path(s, points, 0.0, nvgRGBA(0, 0, 255, 128));
}

static void draw_frame(UIState *s) {
  // draw frame texture
  const UIScene *scene = &s->scene;

  mat4 out_mat;
  float x1, x2, y1, y2;
  if (s->scene.frontview) {
    out_mat = device_transform; // full 16/9

    // flip horizontally so it looks like a mirror
    x2 = (float)scene->front_box_x / s->rgb_front_width;
    x1 = (float)(scene->front_box_x + scene->front_box_width) / s->rgb_front_width;

    y1 = (float)scene->front_box_y / s->rgb_front_height;
    y2 = (float)(scene->front_box_y + scene->front_box_height) / s->rgb_front_height;
  } else {
    out_mat = matmul(device_transform, frame_transform);

    x1 = 0.0;
    x2 = 1.0;
    y1 = 0.0;
    y2 = 1.0;
  }

  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
    {-1.0, -1.0, x2, y1}, //bl
    {-1.0,  1.0, x2, y2}, //tl
    { 1.0,  1.0, x1, y2}, //tr
    { 1.0, -1.0, x1, y1}, //br
  };

  glActiveTexture(GL_TEXTURE0);
  if (s->scene.frontview) {
    glBindTexture(GL_TEXTURE_2D, s->frame_front_tex);
  } else {
    glBindTexture(GL_TEXTURE_2D, s->frame_tex);
  }

  glUseProgram(s->frame_program);

  glUniform1i(s->frame_texture_loc, 0);
  glUniformMatrix4fv(s->frame_transform_loc, 1, GL_TRUE, out_mat.v);

  glEnableVertexAttribArray(s->frame_pos_loc);
  glVertexAttribPointer(s->frame_pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(frame_coords[0]), frame_coords);

  glEnableVertexAttribArray(s->frame_texcoord_loc);
  glVertexAttribPointer(s->frame_texcoord_loc, 2, GL_FLOAT, GL_FALSE, sizeof(frame_coords[0]), &frame_coords[0][2]);

  assert(glGetError() == GL_NO_ERROR);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, &frame_indicies[0]);
}

static void ui_draw_vision(UIState *s) {
  const UIScene *scene = &s->scene;

  if (scene->engaged) {
    glClearColor(1.0, 0.5, 0.0, 1.0);
  } else {
    glClearColor(0.1, 0.1, 0.1, 1.0);
  }
  glClear(GL_COLOR_BUFFER_BIT);

  draw_frame(s);

  if (!scene->frontview) {
    draw_rgb_box(s, scene->big_box_x, s->rgb_height-scene->big_box_height-scene->big_box_y,
                    scene->big_box_width, scene->big_box_height,
                    0xFF0000FF);

    ui_draw_transformed_box(s, 0xFF00FF00);

    // nvg drawings

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glEnable(GL_CULL_FACE);


    glClear(GL_STENCIL_BUFFER_BIT);

    nvgBeginFrame(s->vg, s->fb_w, s->fb_h, 1.0f);

    draw_steering(s, scene->v_ego, scene->angle_steers);

    // draw paths

    if ((nanos_since_boot() - scene->model_ts) < 1000000000ULL) {
      draw_path(s, scene->model.path.points, 0.0f, nvgRGBA(128, 0, 255, 255));

      draw_model_path(s, scene->model.left_lane, nvgRGBA(0, (int)(255 * scene->model.left_lane.prob), 0, 128));
      draw_model_path(s, scene->model.right_lane, nvgRGBA(0, (int)(255 * scene->model.right_lane.prob), 0, 128));
    }

    // draw speed
    char speed_str[16];
    nvgFontSize(s->vg, 128.0f);

    if (scene->engaged) {
      nvgFillColor(s->vg, nvgRGBA(255,128,0,192));
    } else {
      nvgFillColor(s->vg, nvgRGBA(64,64,64,192));
    }

    if (scene->v_cruise != 255 && scene->v_cruise != 0) {
      // Convert KPH to MPH.
      snprintf(speed_str, sizeof(speed_str), "%3d MPH", (int)(scene->v_cruise * 0.621371 + 0.5));
      nvgTextAlign(s->vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
      nvgText(s->vg, 500, 150, speed_str, NULL);
    }

    nvgFillColor(s->vg, nvgRGBA(255,255,255,192));
    snprintf(speed_str, sizeof(speed_str), "%3d MPH", (int)(scene->v_ego * 2.237 + 0.5));
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
    nvgText(s->vg, 1920-500, 150, speed_str, NULL);

    /*nvgFontSize(s->vg, 64.0f);
    nvgTextAlign(s->vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
    nvgText(s->vg, 100+450-20, 1080-100, "mph", NULL);*/

    if (scene->lead_status) {
      char radar_str[16];
      int lead_v_rel = (int)(2.236 * scene->lead_v_rel);
      snprintf(radar_str, sizeof(radar_str), "%3d m %+d mph", (int)(scene->lead_d_rel), lead_v_rel);
      nvgFontSize(s->vg, 96.0f);
      nvgFillColor(s->vg, nvgRGBA(128,128,0,192));
      nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
      nvgText(s->vg, 1920/2, 150, radar_str, NULL);

      // 2.7 m fudge factor
      draw_cross(s, scene->lead_d_rel + 2.7, scene->lead_y_rel, 15, nvgRGBA(255, 0, 0, 128));
    }


    // draw alert text
    if (strlen(scene->alert_text1) > 0) {
      nvgBeginPath(s->vg);
      nvgRoundedRect(s->vg, 100, 200, 1700, 800, 40);
      nvgFillColor(s->vg, nvgRGBA(10,10,10,220));
      nvgFill(s->vg);

      nvgFontSize(s->vg, 200.0f);
      nvgFillColor(s->vg, nvgRGBA(255,0,0,255));
      nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
      nvgTextBox(s->vg, 100+50, 200+50, 1700-50, scene->alert_text1, NULL);

      if (strlen(scene->alert_text2) > 0) {
        nvgFillColor(s->vg, nvgRGBA(255,255,255,255));
        nvgFontSize(s->vg, 100.0f);
        nvgText(s->vg, 100+1700/2, 200+500, scene->alert_text2, NULL);
      }
    }

    if (scene->awareness_status > 0) {
      nvgBeginPath(s->vg);
      int bar_height = scene->awareness_status*700;
      nvgRect(s->vg, 100, 300+(700-bar_height), 50, bar_height);
      nvgFillColor(s->vg, nvgRGBA(255*(1-scene->awareness_status),255*scene->awareness_status,0,128));
      nvgFill(s->vg);
    }

    nvgEndFrame(s->vg);

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
  }
}

static void ui_draw_base(UIState *s) {
  glClearColor(0.1, 0.1, 0.1, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  nvgBeginFrame(s->vg, s->fb_w, s->fb_h, 1.0f);

  nvgFontSize(s->vg, 96.0f);
  nvgFillColor(s->vg, nvgRGBA(255,255,255,255));
  nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
  nvgTextBox(s->vg, 50, 100, s->fb_w, s->base_text, NULL);

  // draw buttons
  for (int i=0; i<ARRAYSIZE(buttons); i++) {
    const Button *b = &buttons[i];


    nvgBeginPath(s->vg);
    nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 255));
    nvgRoundedRect(s->vg, b->x, b->y, b->w, b->h, 20);
    nvgFill(s->vg);

    if (b->label) {
      if (b->enabled && b->enabled(s)) {
        nvgFillColor(s->vg, nvgRGBA(0, 255, 0, 255));
      } else {
        nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
      }
      nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
      nvgText(s->vg, b->x+b->w/2, b->y+b->h/2, b->label, NULL);
    }

    nvgBeginPath(s->vg);
    nvgStrokeColor(s->vg, nvgRGBA(255, 255, 255, 255));
    nvgStrokeWidth(s->vg, 5);
    nvgRoundedRect(s->vg, b->x, b->y, b->w, b->h, 20);
    nvgStroke(s->vg);
  }

  nvgEndFrame(s->vg);

  glDisable(GL_BLEND);
}

static void ui_draw(UIState *s) {

  if (s->vision_connected) {
    ui_draw_vision(s);
  } else {
    ui_draw_base(s);
  }

  eglSwapBuffers(s->display, s->surface);
  assert(glGetError() == GL_NO_ERROR);
}


static PathData read_path(cereal_ModelData_PathData_ptr pathp) {
  PathData ret = {0};

  struct cereal_ModelData_PathData pathd;
  cereal_read_ModelData_PathData(&pathd, pathp);

  ret.prob = pathd.prob;
  ret.std = pathd.std;

  capn_list32 pointl = pathd.points;
  capn_resolve(&pointl.p);
  for (int i=0; i<50; i++) {
    ret.points[i] = capn_to_f32(capn_get32(pointl, i));
  }

  return ret;
}

static ModelData read_model(cereal_ModelData_ptr modelp) {
  struct cereal_ModelData modeld;
  cereal_read_ModelData(&modeld, modelp);

  ModelData d = {0};

  d.path = read_path(modeld.path);
  d.left_lane = read_path(modeld.leftLane);
  d.right_lane = read_path(modeld.rightLane);

  struct cereal_ModelData_LeadData leadd;
  cereal_read_ModelData_LeadData(&leadd, modeld.lead);
  d.lead = (LeadData){
    .dist = leadd.dist,
    .prob = leadd.prob,
    .std = leadd.std,
  };

  return d;
}

static char* read_file(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long f_len = ftell(f);
  rewind(f);

  char* buf = (char *)malloc(f_len+1);
  assert(buf);
  memset(buf, 0, f_len+1);
  fread(buf, f_len, 1, f);
  fclose(f);

  for (int i=f_len; i>=0; i--) {
    if (buf[i] == '\n') buf[i] = 0;
    else if (buf[i] != 0) break;
  }

  return buf;
}

static int pending_uploads() {
  DIR *dirp = opendir("/sdcard/realdata");
  if (!dirp) return -1;
  int cnt = 0;
  struct dirent *entry = NULL;
  while ((entry = readdir(dirp))) {
    if (entry->d_name[0] != '.') {
      cnt++;
    }
  }
  closedir(dirp);
  return cnt;
}


static void ui_update(UIState *s) {
  int err;

  if (s->vision_connect_firstrun) {
    // cant run this in connector thread because opengl.
    // do this here for now in lieu of a run_on_main_thread event

    // setup frame texture
    glDeleteTextures(1, &s->frame_tex); //silently ignores a 0 texture
    glGenTextures(1, &s->frame_tex);
    glBindTexture(GL_TEXTURE_2D, s->frame_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, s->rgb_width, s->rgb_height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);

    // front
    glDeleteTextures(1, &s->frame_front_tex);
    glGenTextures(1, &s->frame_front_tex);
    glBindTexture(GL_TEXTURE_2D, s->frame_front_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, s->rgb_front_width, s->rgb_front_height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);

    assert(glGetError() == GL_NO_ERROR);

    s->vision_connect_firstrun = false;
  }

  // poll for events
  while (true) {

    zmq_pollitem_t polls[5] = {{0}};
    polls[0].socket = s->live100_sock_raw;
    polls[0].events = ZMQ_POLLIN;
    polls[1].socket = s->livecalibration_sock_raw;
    polls[1].events = ZMQ_POLLIN;
    polls[2].socket = s->model_sock_raw;
    polls[2].events = ZMQ_POLLIN;
    polls[3].socket = s->live20_sock_raw;
    polls[3].events = ZMQ_POLLIN;

    int num_polls = 4;
    if (s->vision_connected) {
      assert(s->ipc_fd >= 0);
      polls[4].fd = s->ipc_fd;
      polls[4].events = ZMQ_POLLIN;
      num_polls++;
    }

    int ret = zmq_poll(polls, num_polls, 0);
    if (ret < 0) {
      printf("poll failed (%d)\n", ret);
      break;
    }
    if (ret == 0) {
      break;
    }

    if (s->vision_connected && polls[4].revents) {
      // vision ipc event
      VisionPacket rp;
      err = vipc_recv(s->ipc_fd, &rp);
      if (err <= 0) {
        printf("vision disconnected\n");
        close(s->ipc_fd);
        s->ipc_fd = -1;
        s->vision_connected = false;
        continue;
      }
      if (rp.type == VISION_UI_ACQUIRE) {
        bool front = rp.d.ui_acq.front;
        int idx = rp.d.ui_acq.idx;
        int release_idx;
        if (front) {
          release_idx = s->cur_vision_front_idx;
        } else {
          release_idx = s->cur_vision_idx;
        }
        if (release_idx >= 0) {
          VisionPacket rep = {
            .type = VISION_UI_RELEASE,
            .d = { .ui_rel = {
              .front = front,
              .idx = release_idx,
            }},
          };
          vipc_send(s->ipc_fd, rep);
        }

        if (front) {
          assert(idx < s->vision_bufs.num_front_bufs);
          s->cur_vision_front_idx = idx;
          s->scene.bgr_front_ptr = s->front_bufs[idx].addr;
        } else {
          assert(idx < s->vision_bufs.num_bufs);
          s->cur_vision_idx = idx;
          s->scene.bgr_ptr = s->bufs[idx].addr;
          // printf("v %d\n", ((uint8_t*)s->bufs[idx].addr)[0]);
        }
        if (front == s->scene.frontview) {
          ui_update_frame(s);
        }

      } else {
        assert(false);
      }
    } else {
      // zmq messages
      void* which = NULL;
      for (int i=0; i<4; i++) {
        if (polls[i].revents) {
          which = polls[i].socket;
          break;
        }
      }
      if (which == NULL) {
        continue;
      }

      zmq_msg_t msg;
      err = zmq_msg_init(&msg);
      assert(err == 0);
      err = zmq_msg_recv(&msg, which, 0);
      assert(err >= 0);


      struct capn ctx;
      capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

      cereal_Event_ptr eventp;
      eventp.p = capn_getp(capn_root(&ctx), 0, 1);
      struct cereal_Event eventd;
      cereal_read_Event(&eventd, eventp);

      if (eventd.which == cereal_Event_live100) {
        struct cereal_Live100Data datad;
        cereal_read_Live100Data(&datad, eventd.live100);

        s->scene.v_cruise = datad.vCruise;
        s->scene.v_ego = datad.vEgo;
        s->scene.angle_steers = datad.angleSteers;
        s->scene.engaged = datad.enabled;
        // printf("recv %f\n", datad.vEgo);

        s->scene.frontview = datad.rearViewCam;
        if (datad.alertText1.str) {
          snprintf(s->scene.alert_text1, sizeof(s->scene.alert_text1), "%s", datad.alertText1.str);
        } else {
          s->scene.alert_text1[0] = '\0';
        }
        if (datad.alertText2.str) {
          snprintf(s->scene.alert_text2, sizeof(s->scene.alert_text2), "%s", datad.alertText2.str);
        } else {
          s->scene.alert_text2[0] = '\0';
        }
        s->scene.awareness_status = datad.awarenessStatus;
      } else if (eventd.which == cereal_Event_live20) {
        struct cereal_Live20Data datad;
        cereal_read_Live20Data(&datad, eventd.live20);
        struct cereal_Live20Data_LeadData leaddatad;
        cereal_read_Live20Data_LeadData(&leaddatad, datad.leadOne);
        s->scene.lead_status = leaddatad.status;
        s->scene.lead_d_rel = leaddatad.dRel;
        s->scene.lead_y_rel = leaddatad.yRel;
        s->scene.lead_v_rel = leaddatad.vRel;
      } else if (eventd.which == cereal_Event_liveCalibration) {
        struct cereal_LiveCalibrationData datad;
        cereal_read_LiveCalibrationData(&datad, eventd.liveCalibration);

        // should we still even have this?

        capn_list32 warpl = datad.warpMatrix;
        capn_resolve(&warpl.p); //is this a bug?
        // pthread_mutex_lock(&s->transform_lock);
        for (int i=0; i<3*3; i++) {
          s->scene.big_box_transform.v[i] = capn_to_f32(capn_get32(warpl, i));
        }
        // pthread_mutex_unlock(&s->transform_lock);

        // printf("recv %f\n", datad.vEgo);
      } else if (eventd.which == cereal_Event_model) {
        s->scene.model_ts = eventd.logMonoTime;
        s->scene.model = read_model(eventd.model);
      }

      capn_free(&ctx);

      zmq_msg_close(&msg);

    }

  }

  // update base ui
  uint64_t ts = nanos_since_boot();
  if (!s->vision_connected && ts - s->last_base_update > 1000000000ULL) {
    char* bat_cap = read_file("/sys/class/power_supply/battery/capacity");
    char* bat_stat = read_file("/sys/class/power_supply/battery/status");

    int tx_rate = 0;
    int rx_rate = 0;
    char* rx_bytes = read_file("/sys/class/net/rmnet_data0/statistics/rx_bytes");
    char* tx_bytes = read_file("/sys/class/net/rmnet_data0/statistics/tx_bytes");
    if (rx_bytes && tx_bytes) {
      uint64_t rx_bytes_n = atoll(rx_bytes);
      rx_rate = rx_bytes_n - s->last_rx_bytes;
      s->last_rx_bytes = rx_bytes_n;

      uint64_t tx_bytes_n = atoll(tx_bytes);
      tx_rate = tx_bytes_n - s->last_tx_bytes;
      s->last_tx_bytes = tx_bytes_n;
    }
    if (rx_bytes) free(rx_bytes);
    if (tx_bytes) free(tx_bytes);

    // TODO: do this properly
    system("git rev-parse --abbrev-ref HEAD > /tmp/git_branch");
    char *git_branch = read_file("/tmp/git_branch");
    system("git rev-parse --short HEAD > /tmp/git_commit");
    char *git_commit = read_file("/tmp/git_commit");

    int pending = pending_uploads();

    // service call wifi 20  # getWifiEnabledState
    // Result: Parcel(00000000 00000003   '........') = enabled
    s->wifi_enabled = !system("service call wifi 20 | grep 00000003 > /dev/null");

    // service call wifi 38  # getWifiApEnabledState
    // Result: Parcel(00000000 0000000d   '........') = enabled
    s->ap_enabled = !system("service call wifi 38 | grep 0000000d > /dev/null");

    s->board_connected = !system("lsusb | grep bbaa > /dev/null");

    snprintf(s->base_text, sizeof(s->base_text),
             "version: %s (%s)\nserial: %s\n dongle id: %s\n battery: %s %s\npending: %d\nrx %.1fkiB/s tx %.1fkiB/s\nboard: %s",
             git_commit, git_branch,
             s->serial, s->dongle_id, bat_cap ? bat_cap : "(null)", bat_stat ? bat_stat : "(null)",
             pending, rx_rate / 1024.0, tx_rate / 1024.0, s->board_connected ? "found" : "NOT FOUND");

    if (bat_cap) free(bat_cap);
    if (bat_stat) free(bat_stat);

    if (git_branch) free(git_branch);
    if (git_commit) free(git_commit);

    s->last_base_update = ts;

    if (s->awake_timeout > 0) {
      s->awake_timeout--;
    } else {
      set_awake(s, false);
    }
  }

  if (s->vision_connected) {
    // always awake if vision is connected
    set_awake(s, true);
  }

  if (!s->vision_connected) {
    // baseui interaction

    int touch_x = -1, touch_y = -1;
    err = touch_poll(&s->touch, &touch_x, &touch_y);
    if (err == 1) {
      if (s->awake) {
        // press buttons
        for (int i=0; i<ARRAYSIZE(buttons); i++) {
          const Button *b = &buttons[i];
          if (touch_x >= b->x && touch_x < b->x+b->w
              && touch_y >= b->y && touch_y < b->y+b->h) {
            if (b->pressed && !activity_running()) {
              b->pressed();
              break;
            }
          }
        }
      } else {
        set_awake(s, true);
      }
    }
  }

}

volatile int do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}


static void* vision_connect_thread(void *args) {
  int err;

  UIState *s = args;
  while (!do_exit) {
    usleep(100000);
    pthread_mutex_lock(&s->lock);
    bool connected = s->vision_connected;
    pthread_mutex_unlock(&s->lock);
    if (connected) continue;

    int fd = vipc_connect();
    if (fd < 0) continue;

    VisionPacket p = {
      .type = VISION_UI_SUBSCRIBE,
    };
    err = vipc_send(fd, p);
    if (err < 0) {
      close(fd);
      continue;
    }

    // printf("init recv\n");
    VisionPacket rp;
    err = vipc_recv(fd, &rp);
    if (err <= 0) {
      close(fd);
      continue;
    }

    assert(rp.type == VISION_UI_BUFS);
    assert(rp.num_fds == rp.d.ui_bufs.num_bufs + rp.d.ui_bufs.num_front_bufs);

    pthread_mutex_lock(&s->lock);
    assert(!s->vision_connected);
    s->ipc_fd = fd;
    ui_init_vision(s, rp.d.ui_bufs, rp.fds);
    s->vision_connected = true;
    s->vision_connect_firstrun = true;
    pthread_mutex_unlock(&s->lock);
  }
  return NULL;
}

int main() {
  int err;

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);

  UIState uistate;
  UIState *s = &uistate;
  ui_init(s);

  pthread_t connect_thread_handle;
  err = pthread_create(&connect_thread_handle, NULL,
                       vision_connect_thread, s);
  assert(err == 0);

  while (!do_exit) {
    pthread_mutex_lock(&s->lock);
    ui_update(s);
    ui_draw(s);
    pthread_mutex_unlock(&s->lock);

    // no simple way to do 30fps vsync with surfaceflinger...
    usleep(30000);
  }

  err = pthread_join(connect_thread_handle, NULL);
  assert(err == 0);

  return 0;
}
