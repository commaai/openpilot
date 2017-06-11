#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>

#include <cutils/properties.h>

#include <GLES3/gl3.h>
#include <EGL/eglext.h>

#include <json.h>
#include <czmq.h>

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "common/timing.h"
#include "common/util.h"
#include "common/mat.h"
#include "common/glutil.h"

#include "common/touch.h"
#include "common/framebuffer.h"
#include "common/visionipc.h"
#include "common/modeldata.h"
#include "common/params.h"

#include "cereal/gen/c/log.capnp.h"

#define UI_BUF_COUNT 4

typedef struct UIScene {
  int frontview;

  uint8_t *bgr_ptr;
  int big_box_x, big_box_y, big_box_width, big_box_height;

  int transformed_width, transformed_height;

  uint64_t model_ts;
  ModelData model;

  bool world_objects_visible;
  // TODO(mgraczyk): Remove and use full frame for everything.
  mat3 warp_matrix;           // transformed box -> big_box.
  mat4 extrinsic_matrix;      // Last row is 0 so we can use mat4.

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
  void *model_sock_raw;
  zsock_t *live100_sock;
  void *live100_sock_raw;
  zsock_t *livecalibration_sock;
  void *livecalibration_sock_raw;
  zsock_t *live20_sock;
  void *live20_sock_raw;

  // vision state
  bool vision_connected;
  bool vision_connect_firstrun;
  int ipc_fd;

  VisionBuf bufs[UI_BUF_COUNT];
  VisionBuf front_bufs[UI_BUF_COUNT];
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

  bool intrinsic_matrix_loaded;
  mat3 intrinsic_matrix;

  UIScene scene;

  bool awake;
  int awake_timeout;

  bool is_metric;
} UIState;

static void set_awake(UIState *s, bool awake) {
  if (awake) {
    // 15 second timeout at 30 fps
    s->awake_timeout = 15*30;
  }
  if (s->awake != awake) {
    s->awake = awake;

    if (awake) {
      printf("awake normal\n");
      framebuffer_set_power(s->fb, HWC_POWER_MODE_NORMAL);

      // can't hurt
      FILE *f = fopen("/sys/class/leds/lcd-backlight/brightness", "wb");
      if (f != NULL) {
        fprintf(f, "205");
        fclose(f);
      }
    } else {
      printf("awake off\n");
      framebuffer_set_power(s->fb, HWC_POWER_MODE_OFF);
    }
  }
}


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
  set_awake(s, true);

  // init drawing
  s->vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
  assert(s->vg);
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
}


// If the intrinsics are in the params entry, this copies them to
// intrinsic_matrix and returns true.  Otherwise returns false.
static bool try_load_intrinsics(mat3 *intrinsic_matrix) {
  char *value;
  const int result =
      read_db_value("/data/params", "CloudCalibration", &value, NULL);

  if (result == 0) {
    JsonNode* calibration_json = json_decode(value);
    free(value);

    JsonNode *intrinsic_json =
        json_find_member(calibration_json, "intrinsic_matrix");

    if (intrinsic_json == NULL || intrinsic_json->tag != JSON_ARRAY) {
      json_delete(calibration_json);
      return false;
    }

    int i = 0;
    JsonNode* json_num; 
    json_foreach(json_num, intrinsic_json) {
      intrinsic_matrix->v[i++] = json_num->number_;
    }
    json_delete(calibration_json);

    return true;
  } else {
    return false;
  }
}


static void ui_init_vision(UIState *s, const VisionStreamBufs back_bufs,
                           int num_back_fds, const int *back_fds,
                           const VisionStreamBufs front_bufs, int num_front_fds,
                           const int *front_fds) {
  const VisionUIInfo ui_info = back_bufs.buf_info.ui_info;

  assert(num_back_fds == UI_BUF_COUNT);
  assert(num_front_fds == UI_BUF_COUNT);

  visionbufs_load(s->bufs, &back_bufs, num_back_fds, back_fds);
  visionbufs_load(s->front_bufs, &front_bufs, num_front_fds, front_fds);

  s->cur_vision_idx = -1;
  s->cur_vision_front_idx = -1;

  s->scene = (UIScene){
      .frontview = 0,
      .big_box_x = ui_info.big_box_x,
      .big_box_y = ui_info.big_box_y,
      .big_box_width = ui_info.big_box_width,
      .big_box_height = ui_info.big_box_height,
      .transformed_width = ui_info.transformed_width,
      .transformed_height = ui_info.transformed_height,
      .front_box_x = ui_info.front_box_x,
      .front_box_y = ui_info.front_box_y,
      .front_box_width = ui_info.front_box_width,
      .front_box_height = ui_info.front_box_height,
      .world_objects_visible = false,  // Invisible until we receive a calibration message.
  };

  s->rgb_width = back_bufs.width;
  s->rgb_height = back_bufs.height;

  s->rgb_front_width = front_bufs.width;
  s->rgb_front_height = front_bufs.height;

  s->rgb_transform = (mat4){{
    2.0/s->rgb_width, 0.0, 0.0, -1.0,
    0.0, 2.0/s->rgb_height, 0.0, -1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};

  char *value;
  const int result = read_db_value("/data/params", "IsMetric", &value, NULL);
  if (result == 0) {
    s->is_metric = value[0] == '1';
    free(value);
  }
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

static void ui_draw_transformed_box(UIState *s, uint32_t color) {
  const UIScene *scene = &s->scene;

  const mat3 bbt = scene->warp_matrix;

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
    verts[i].pos.v[1] = s->rgb_height - (scene->big_box_y +
                                         verts[i].pos.v[1] / verts[i].pos.v[2]);
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

// Projects a point in car to space to the corresponding point in full frame
// image space.
vec3 car_space_to_full_frame(const UIState *s, vec4 car_space_projective) {
  const UIScene *scene = &s->scene;

  // We'll call the car space point p.
  // First project into normalized image coordinates with the extrinsics matrix.
  const vec4 Ep4 = matvecmul(scene->extrinsic_matrix, car_space_projective);

  // The last entry is zero because of how we store E (to use matvecmul).
  const vec3 Ep = {{Ep4.v[0], Ep4.v[1], Ep4.v[2]}};
  const vec3 KEp = matvecmul3(s->intrinsic_matrix, Ep);

  // Project.
  const vec3 p_image = {{KEp.v[0] / KEp.v[2], KEp.v[1] / KEp.v[2], 1.}};
  return p_image;
}


// TODO: refactor with draw_path
static void draw_cross(UIState *s, float x_in, float y_in, float sz, NVGcolor color) {
  const UIScene *scene = &s->scene;

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

  const vec4 p_car_space = (vec4){{x_in, y_in, 0., 1.}};
  const vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);

  float x = p_full_frame.v[0];
  float y = p_full_frame.v[1];
  if (x >= 0 && y >= 0.) {
    nvgMoveTo(s->vg, x-sz, y);
    nvgLineTo(s->vg, x+sz, y);

    nvgMoveTo(s->vg, x, y-sz);
    nvgLineTo(s->vg, x, y+sz);

    nvgStroke(s->vg);
  }

  nvgRestore(s->vg);
}

static void draw_path(UIState *s, const float *points, float off,
                      NVGcolor color) {
  const UIScene *scene = &s->scene;

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
  bool started = false;

  for (int i=0; i<50; i++) {
    float px = (float)i;
    float py = points[i] + off;

    vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);

    float x = p_full_frame.v[0];
    float y = p_full_frame.v[1];
    if (x < 0 || y < 0.) {
      continue;
    }

    if (!started) {
      nvgMoveTo(s->vg, x, y);
      started = true;
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
  double curvature = angle_steers_rad / (steer_ratio * wheel_base *
                                         (1. + slip_fator * v_ego * v_ego));
  return curvature;
}

static void draw_steering(UIState *s, float v_ego, float angle_steers) {
  double curvature = calc_curvature(v_ego, angle_steers);

  float points[50];
  for (int i = 0; i < 50; i++) {
    float y_actual = i * tan(asin(clamp(i * curvature, -0.999, 0.999)) / 2.);
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
  glVertexAttribPointer(s->frame_pos_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), frame_coords);

  glEnableVertexAttribArray(s->frame_texcoord_loc);
  glVertexAttribPointer(s->frame_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), &frame_coords[0][2]);

  assert(glGetError() == GL_NO_ERROR);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, &frame_indicies[0]);
}

// Draw all world space objects.
static void ui_draw_world(UIState *s) {
  const UIScene *scene = &s->scene;
  if (!scene->world_objects_visible) {
    return;
  }

  draw_steering(s, scene->v_ego, scene->angle_steers);

  // draw paths
  if ((nanos_since_boot() - scene->model_ts) < 1000000000ULL) {
    draw_path(s, scene->model.path.points, 0.0f, nvgRGBA(128, 0, 255, 255));

    draw_model_path(
        s, scene->model.left_lane,
        nvgRGBA(0, (int)(255 * scene->model.left_lane.prob), 0, 128));
    draw_model_path(
        s, scene->model.right_lane,
        nvgRGBA(0, (int)(255 * scene->model.right_lane.prob), 0, 128));
  }

  if (scene->lead_status) {
    char radar_str[16];
    if (s->is_metric) {
      int lead_v_rel = (int)(3.6 * scene->lead_v_rel);
      snprintf(radar_str, sizeof(radar_str), "%3d m %+d kph",
               (int)(scene->lead_d_rel), lead_v_rel);
    } else {
      int lead_v_rel = (int)(2.236 * scene->lead_v_rel);
      snprintf(radar_str, sizeof(radar_str), "%3d m %+d mph",
               (int)(scene->lead_d_rel), lead_v_rel);
    }
    nvgFontSize(s->vg, 96.0f);
    nvgFillColor(s->vg, nvgRGBA(128, 128, 0, 192));
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgText(s->vg, 1920 / 2, 150, radar_str, NULL);

    // 2.7 m fudge factor
    draw_cross(s, scene->lead_d_rel + 2.7, scene->lead_y_rel, 15,
               nvgRGBA(255, 0, 0, 128));
  }
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

  // nvg drawings
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glEnable(GL_CULL_FACE);

  glClear(GL_STENCIL_BUFFER_BIT);

  nvgBeginFrame(s->vg, s->fb_w, s->fb_h, 1.0f);

  if (!scene->frontview) {
    ui_draw_transformed_box(s, 0xFF00FF00);
    ui_draw_world(s);

    // draw speed
    char speed_str[16];
    float defaultfontsize = 128.0f;
    float labelfontsize = 65.0f;

    /******************************************
     * Add background rect so it's easier to see in 
     * light background scenes 
     ******************************************/
    // Left side - ACC max speed
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, -15, 0, 535, 175, 30);
    nvgFillColor(s->vg, nvgRGBA(10, 10, 10, 180));
    nvgFill(s->vg);
    /******************************************/

    if (scene->engaged) {
      nvgFillColor(s->vg, nvgRGBA(255, 128, 0, 192));

      // Add label
      nvgFontSize(s->vg, labelfontsize);
      nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
      nvgText(s->vg, 42, 175-30, "ACC: engaged", NULL);
    } else {
      nvgFillColor(s->vg, nvgRGBA(195, 195, 195, 192));

      // Add label
      nvgFontSize(s->vg, labelfontsize);
      nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
      nvgText(s->vg, 42, 175-30, "ACC: disabled", NULL);
    }

    nvgFontSize(s->vg, defaultfontsize);
    if (scene->v_cruise != 255 && scene->v_cruise != 0) {
      if (s->is_metric) {
        snprintf(speed_str, sizeof(speed_str), "%3d KPH",
                 (int)(scene->v_cruise + 0.5));
      } else {
        // Convert KPH to MPH.
        snprintf(speed_str, sizeof(speed_str), "%3d MPH",
                 (int)(scene->v_cruise * 0.621371 + 0.5));
      }
      nvgTextAlign(s->vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
      nvgText(s->vg, 500, 95, speed_str, NULL);
    }

    /******************************************
     * Add background rect so it's easier to see in 
     * light background scenes 
     ******************************************/
    // Right side - Actual speed
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, 1920 - 500, 0, 1920, 175, 20);
    nvgFillColor(s->vg, nvgRGBA(10, 10, 10, 180));
    nvgFill(s->vg);

    // Add label
    nvgFontSize(s->vg, labelfontsize);
    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 192));
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
    nvgText(s->vg, 1920 - 475, 175-30, "Current Speed", NULL);
    /******************************************/

    nvgFontSize(s->vg, defaultfontsize);
    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 192));
    if (s->is_metric) {
      snprintf(speed_str, sizeof(speed_str), "%3d KPH",
               (int)(scene->v_ego * 3.6 + 0.5));
    } else {
      snprintf(speed_str, sizeof(speed_str), "%3d MPH",
               (int)(scene->v_ego * 2.237 + 0.5));
    }
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
    nvgText(s->vg, 1920 - 500, 95, speed_str, NULL);

    /*nvgFontSize(s->vg, 64.0f);
    nvgTextAlign(s->vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
    nvgText(s->vg, 100+450-20, 1080-100, "mph", NULL);*/

    if (scene->awareness_status > 0) {
      nvgBeginPath(s->vg);
      int bar_height = scene->awareness_status * 700;
      nvgRect(s->vg, 100, 300 + (700 - bar_height), 50, bar_height);
      nvgFillColor(s->vg, nvgRGBA(255 * (1 - scene->awareness_status),
                                  255 * scene->awareness_status, 0, 128));
      nvgFill(s->vg);
    }
  }

  // draw alert text
  if (strlen(scene->alert_text1) > 0) {
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, 100, 200, 1700, 800, 40);
    nvgFillColor(s->vg, nvgRGBA(10, 10, 10, 220));
    nvgFill(s->vg);

    nvgFontSize(s->vg, 200.0f);
    nvgFillColor(s->vg, nvgRGBA(255, 0, 0, 255));
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgTextBox(s->vg, 100 + 50, 200 + 50, 1700 - 50, scene->alert_text1,
                NULL);

    if (strlen(scene->alert_text2) > 0) {
      nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
      nvgFontSize(s->vg, 100.0f);
      nvgTextBox(s->vg, 100 + 50, 200 + 550, 1700 - 2*50, scene->alert_text2, NULL);
    }
  }

  nvgEndFrame(s->vg);

  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
}

static void ui_draw_blank(UIState *s) {
  glClearColor(0.1, 0.1, 0.1, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
}

static void ui_draw(UIState *s) {
  if (s->vision_connected) {
    ui_draw_vision(s);
  } else {
    ui_draw_blank(s);
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
  for (int i = 0; i < 50; i++) {
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
      .dist = leadd.dist, .prob = leadd.prob, .std = leadd.std,
  };

  return d;
}

static void ui_update(UIState *s) {
  int err;

  if (!s->intrinsic_matrix_loaded) {
    s->intrinsic_matrix_loaded = try_load_intrinsics(&s->intrinsic_matrix);
  }

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
      if (rp.type == VIPC_STREAM_ACQUIRE) {
        bool front = rp.d.stream_acq.type == VISION_STREAM_UI_FRONT;
        int idx = rp.d.stream_acq.idx;

        int release_idx;
        if (front) {
          release_idx = s->cur_vision_front_idx;
        } else {
          release_idx = s->cur_vision_idx;
        }
        if (release_idx >= 0) {
          VisionPacket rep = {
            .type = VIPC_STREAM_RELEASE,
            .d = { .stream_rel = {
              .type = rp.d.stream_acq.type,
              .idx = release_idx,
            }},
          };
          vipc_send(s->ipc_fd, &rep);
        }

        if (front) {
          assert(idx < UI_BUF_COUNT);
          s->cur_vision_front_idx = idx;
          s->scene.bgr_front_ptr = s->front_bufs[idx].addr;
        } else {
          assert(idx < UI_BUF_COUNT);
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
        s->scene.world_objects_visible = s->intrinsic_matrix_loaded;
        struct cereal_LiveCalibrationData datad;
        cereal_read_LiveCalibrationData(&datad, eventd.liveCalibration);

        // should we still even have this?
        capn_list32 warpl = datad.warpMatrix;
        capn_resolve(&warpl.p);  // is this a bug?
        for (int i = 0; i < 3 * 3; i++) {
          s->scene.warp_matrix.v[i] = capn_to_f32(capn_get32(warpl, i));
        }

        capn_list32 extrinsicl = datad.extrinsicMatrix;
        capn_resolve(&extrinsicl.p);  // is this a bug?
        for (int i = 0; i < 3 * 4; i++) {
          s->scene.extrinsic_matrix.v[i] =
              capn_to_f32(capn_get32(extrinsicl, i));
        }
      } else if (eventd.which == cereal_Event_model) {
        s->scene.model_ts = eventd.logMonoTime;
        s->scene.model = read_model(eventd.model);
      }

      capn_free(&ctx);

      zmq_msg_close(&msg);

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



    VisionPacket p1 = {
      .type = VIPC_STREAM_SUBSCRIBE,
      .d = { .stream_sub = { .type = VISION_STREAM_UI_BACK, .tbuffer = true, }, },
    };
    err = vipc_send(fd, &p1);
    if (err < 0) {
      close(fd);
      continue;
    }
    VisionPacket p2 = {
      .type = VIPC_STREAM_SUBSCRIBE,
      .d = { .stream_sub = { .type = VISION_STREAM_UI_FRONT, .tbuffer = true, }, },
    };
    err = vipc_send(fd, &p2);
    if (err < 0) {
      close(fd);
      continue;
    }

    // printf("init recv\n");
    VisionPacket back_rp;
    err = vipc_recv(fd, &back_rp);
    if (err <= 0) {
      close(fd);
      continue;
    }
    assert(back_rp.type == VIPC_STREAM_BUFS);
    VisionPacket front_rp;
    err = vipc_recv(fd, &front_rp);
    if (err <= 0) {
      close(fd);
      continue;
    }
    assert(front_rp.type == VIPC_STREAM_BUFS);


    pthread_mutex_lock(&s->lock);
    assert(!s->vision_connected);
    s->ipc_fd = fd;

    ui_init_vision(s,
                   back_rp.d.stream_bufs, back_rp.num_fds, back_rp.fds,
                   front_rp.d.stream_bufs, front_rp.num_fds, front_rp.fds);

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
    if (s->awake) {
      pthread_mutex_lock(&s->lock);
      ui_update(s);
      ui_draw(s);
      pthread_mutex_unlock(&s->lock);
    }

    // manage wakefulness
    if (s->awake_timeout > 0) {
      s->awake_timeout--;
    } else {
      set_awake(s, false);
    }

    // always awake if vision is connected
    if (s->vision_connected) {
      set_awake(s, true);
    } else {
      int touch_x = -1, touch_y = -1;
      err = touch_poll(&s->touch, &touch_x, &touch_y);
      if (err == 1) {
        // touch event will still happen :(
        set_awake(s, true);
      }
    }

    // no simple way to do 30fps vsync with surfaceflinger...
    usleep(30000);
  }

  err = pthread_join(connect_thread_handle, NULL);
  assert(err == 0);

  return 0;
}
