#pragma once
#include <cmath>
#include <assert.h>
#include "messaging.hpp"
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

#include "nanovg.h"
#include "common/framebuffer.h"
#include "common/mat.h"
#include "common/visionimg.h"
#include "common/visionipc.h"
#include "common/touch.h"

#define UI_BUF_COUNT 4
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
class UIVision {
 public:
  void init(bool front=false);
  bool update();
  void draw();
  void swap();
  
  int rgb_width, rgb_height, rgb_stride;
  int ipc_fd = -1;
 
  FramebufferState *fb = nullptr;
  int fb_w=0, fb_h=0;

private:
  void initVision(const VisionPacket& vp);
  void connect();

  VIPCBuf bufs[UI_BUF_COUNT]={};
  int cur_vision_idx = 0;

  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT]={};
  EGLImageKHR khr[UI_BUF_COUNT]={};
  void *priv_hnds[UI_BUF_COUNT]={};
  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;

  bool vision_connected = false;
  
  bool front_view=false;
  size_t rgb_buf_len;
};

void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, float r = 0, int width = 0);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGpaint &paint, float r = 0);
void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, bool active);
void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, NVGcolor color, float img_alpha, int img_y = 0);
