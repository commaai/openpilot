#include <assert.h>
#include <czmq.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "uicommon.hpp"

#ifdef NANOVG_GL3_IMPLEMENTATION
static const char frame_vertex_shader[] =
  "#version 150 core\n"
  "in vec4 aPosition;\n"
  "in vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
  "#version 150 core\n"
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "in vec4 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  colorOut = texture(uTexture, vTexCoord.xy);\n"
  "}\n";
#else
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
#endif

static const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

// frame from 4/3 to box size with a 2x zoom
static const mat4 frame_transform = {{
  2*(4./3.)/((float)viz_w/box_h), 0.0, 0.0, 0.0,
  0.0, 2.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0,
}};

// frame from 4/3 to 16/9 display
static const mat4 full_to_wide_frame_transform = {{
  .75,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

void UIVision::init(bool front) {
  front_view = front;
  fb = framebuffer_init("ui", 0, true, &fb_w, &fb_h);
  assert(fb);

  // init gl
  frame_program = load_program(frame_vertex_shader, frame_fragment_shader);
  assert(frame_program);

  frame_pos_loc = glGetAttribLocation(frame_program, "aPosition");
  frame_texcoord_loc = glGetAttribLocation(frame_program, "aTexCoord");
  frame_texture_loc = glGetUniformLocation(frame_program, "uTexture");
  frame_transform_loc = glGetUniformLocation(frame_program, "uTransform");

  glViewport(0, 0, fb_w, fb_h);
  glDisable(GL_DEPTH_TEST);
  assert(glGetError() == GL_NO_ERROR);

  float x1, x2, y1, y2;
  if (front) {
    // flip horizontally so it looks like a mirror
    x1 = 0.0;
    x2 = 1.0;
    y1 = 1.0;
    y2 = 0.0;
  } else {
    x1 = 1.0;
    x2 = 0.0;
    y1 = 1.0;
    y2 = 0.0;
  }
  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
      {-1.0, -1.0, x2, y1},  //bl
      {-1.0, 1.0, x2, y2},   //tl
      {1.0, 1.0, x1, y2},    //tr
      {1.0, -1.0, x1, y1},   //br
  };

  glGenVertexArrays(1, &frame_vao);
  glBindVertexArray(frame_vao);
  glGenBuffers(1, &frame_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, frame_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(frame_coords), frame_coords, GL_STATIC_DRAW);
  glEnableVertexAttribArray(frame_pos_loc);
  glVertexAttribPointer(frame_pos_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), (const void *)0);
  glEnableVertexAttribArray(frame_texcoord_loc);
  glVertexAttribPointer(frame_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), (const void *)(sizeof(float) * 2));
  glGenBuffers(1, &frame_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frame_ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(frame_indicies), frame_indicies, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  frame_mat = matmul(device_transform, front_view ? full_to_wide_frame_transform : frame_transform);
}

void UIVision::swap(){
  framebuffer_swap(fb);
}    

void UIVision::initVision() {
  assert(stream.num_bufs == UI_BUF_COUNT);
  for (int i = 0; i < UI_BUF_COUNT; i++) {
    if (khr[i] != 0) {
      visionimg_destroy_gl(khr[i], priv_hnds[i]);
      glDeleteTextures(1, &frame_texs[i]);
    }

    VisionImg img = {
        .fd = stream.bufs[i].fd,
        .format = VISIONIMG_FORMAT_RGB24,
        .width = stream.bufs_info.width,
        .height = stream.bufs_info.height,
        .stride = stream.bufs_info.stride,
        .bpp = 3,
        .size = stream.bufs_info.buf_len,
    };
#ifndef QCOM
    priv_hnds[i] = stream.bufs[i].addr;
#endif
    frame_texs[i] = visionimg_to_gl(&img, &khr[i], &priv_hnds[i]);

    glBindTexture(GL_TEXTURE_2D, frame_texs[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
  }
  assert(glGetError() == GL_NO_ERROR);
}

void UIVision::update(volatile sig_atomic_t *do_exit) {
  while (!(*do_exit)) {
    if (!vision_connected) {
      int err = visionstream_init(&stream, front_view ? VISION_STREAM_RGB_FRONT : VISION_STREAM_RGB_BACK, true, nullptr);
      if (err) {
        printf("visionstream connect fail\n");
        usleep(100000);
        continue;
      }
      vision_connected = true;
      initVision();
    }
    VIPCBuf *buf = visionstream_get(&stream, nullptr);
    if (buf == NULL) {
      printf("visionstream get failed\n");
      visionstream_destroy(&stream);
      vision_connected = false;
      continue;
    }

    break;
  }
}

void UIVision::draw() {
  glBindVertexArray(frame_vao);
  glActiveTexture(GL_TEXTURE0);

  if (stream.last_idx  >= 0) {
    glBindTexture(GL_TEXTURE_2D, frame_texs[stream.last_idx]);
    if (!front_view) {
#ifndef QCOM
      // TODO: a better way to do this?
      //printf("%d\n", ((int*)priv_hnds[stream.last_idx])[0]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1164, 874, 0, GL_RGB, GL_UNSIGNED_BYTE, priv_hnds[stream.last_idx]);
#endif
    }
  }

  glUseProgram(frame_program);
  glUniform1i(frame_texture_loc, 0);
  glUniformMatrix4fv(frame_transform_loc, 1, GL_TRUE, frame_mat.v);

  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void*)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
}

void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha){
  nvgBeginPath(vg);
  NVGpaint imgPaint = nvgImagePattern(vg, x, y, w, h, 0, image, alpha);
  nvgRect(vg, x, y, w, h);
  nvgFillPaint(vg, imgPaint);
  nvgFill(vg);
}

void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, float r, int width) {
  nvgBeginPath(vg);
  r > 0? nvgRoundedRect(vg, x, y, w, h, r) : nvgRect(vg, x, y, w, h);
  if (width) {
    nvgStrokeColor(vg, color);
    nvgStrokeWidth(vg, width);
    nvgStroke(vg);
  } else {
    nvgFillColor(vg, color);
    nvgFill(vg);
  }
}

void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGpaint &paint, float r){
  nvgBeginPath(vg);
  r > 0? nvgRoundedRect(vg, x, y, w, h, r) : nvgRect(vg, x, y, w, h);
  nvgFillPaint(vg, paint);
  nvgFill(vg);
}

void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, NVGcolor color, float img_alpha, int img_y) {
  const int img_size = size * 1.5;
  nvgBeginPath(vg);
  nvgCircle(vg, x, y + (bdr_s * 1.5), size);
  nvgFillColor(vg, color);
  nvgFill(vg);
  ui_draw_image(vg, x - (img_size / 2), img_y ? img_y : y - (size / 4), img_size, img_size, image, img_alpha);
}

void ui_draw_circle_image(NVGcontext *vg, float x, float y, int size, int image, bool active) {
  float bg_alpha = active ? 0.3f : 0.1f;
  float img_alpha = active ? 1.0f : 0.15f;
  ui_draw_circle_image(vg, x, y, size, image, nvgRGBA(0, 0, 0, (255 * bg_alpha)), img_alpha);
}
