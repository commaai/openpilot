#pragma once
#include "messaging.hpp"
#include <assert.h>
#include <cmath>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#define NANOVG_GL3_IMPLEMENTATION
#define nvgCreate nvgCreateGL3
#define nvgDelete nvgDeleteGL3
#else
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#define NANOVG_GLES3_IMPLEMENTATION
#define nvgCreate nvgCreateGLES3
#define nvgDelete nvgDeleteGLES3
#endif
#include "common/framebuffer.h"
#include "common/mat.h"
#include "common/touch.h"
#include "common/visionimg.h"
#include "common/visionipc.h"
#include "nanovg.h"

#define UI_BUF_COUNT 4
const int vwp_w = 1920;
const int vwp_h = 1080;
const int nav_w = 640;
const int nav_ww = 760;
const int sbr_w = 300;
const int bdr_s = 30;
const int box_x = sbr_w + bdr_s;
const int box_y = bdr_s;
const int box_w = vwp_w - sbr_w - (bdr_s * 2);
const int box_h = vwp_h - (bdr_s * 2);
const int viz_w = vwp_w - (bdr_s * 2);

class UIVision {
 public:
  void init(bool front = false);
  void update();
  void draw();
  void swap();
  inline int getRgbWidth() const { return stream.bufs_info.width; }
  inline int getRgbHeight() const { return stream.bufs_info.height; }

  FramebufferState *fb = nullptr;
  int fb_w = 0, fb_h = 0;

 private:
  void initVision();

  VisionStream stream;

  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT] = {};
  EGLImageKHR khr[UI_BUF_COUNT] = {};
  void *priv_hnds[UI_BUF_COUNT] = {};
  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;

  bool vision_connected = false;
  bool front_view = false;
};

void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, float r = 0, int width = 0);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGpaint &paint, float r = 0);
void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, bool active);
void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, NVGcolor color, float img_alpha, int img_y = 0);
