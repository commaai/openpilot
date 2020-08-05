#include <assert.h>
#include <czmq.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "uicommon.hpp"

#ifndef QCOM
  #define UI_60FPS
#endif

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
  ipc_fd = -1;
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

static int vision_subscribe(int fd, VisionPacket *rp, VisionStreamType type) {
  int err;
  LOGW("vision_subscribe type:%d", type);

  VisionPacket p1 = {
    .type = VIPC_STREAM_SUBSCRIBE,
    .d = { .stream_sub = { .type = type, .tbuffer = true, }, },
  };
  err = vipc_send(fd, &p1);
  if (err < 0) {
    close(fd);
    return 0;
  }

  do {
    err = vipc_recv(fd, rp);
    if (err <= 0) {
      close(fd);
      return 0;
    }

    // release what we aren't ready for yet
    if (rp->type == VIPC_STREAM_ACQUIRE) {
      VisionPacket rep = {
        .type = VIPC_STREAM_RELEASE,
        .d = { .stream_rel = {
          .type = rp->d.stream_acq.type,
          .idx = rp->d.stream_acq.idx,
        }},
      };
      vipc_send(fd, &rep);
    }
  } while (rp->type != VIPC_STREAM_BUFS || rp->d.stream_bufs.type != type);

  return 1;
}


void UIVision::vision_connect() {
  bool do_exit = false;
  while (!do_exit) {
    int fd = vipc_connect();
    if (fd < 0) {
      usleep(100000);
      continue;
    }

    VisionPacket vp;
    if (!vision_subscribe(fd, &vp, front_view ? VISION_STREAM_RGB_FRONT : VISION_STREAM_RGB_BACK)) continue;
    ipc_fd = fd;
    ui_init_vision(vp);

    vision_connected = true;
    break;
  }
}

void UIVision::swap(){
  framebuffer_swap(fb);
}    

void UIVision::ui_init_vision(const VisionPacket& vp) {
  assert(vp.num_fds == UI_BUF_COUNT);
  const VisionStreamBufs& vs_bufs = vp.d.stream_bufs;
  vipc_bufs_load(bufs, &vs_bufs, vp.num_fds, vp.fds);

  cur_vision_idx = -1;

  rgb_width = vs_bufs.width;
  rgb_height = vs_bufs.height;
  rgb_stride = vs_bufs.stride;
  rgb_buf_len = vs_bufs.buf_len;

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    if (khr[i] != 0) {
      visionimg_destroy_gl(khr[i], priv_hnds[i]);
      glDeleteTextures(1, &frame_texs[i]);
    }

    VisionImg img = {
        .fd = bufs[i].fd,
        .format = VISIONIMG_FORMAT_RGB24,
        .width = rgb_width,
        .height = rgb_height,
        .stride = rgb_stride,
        .bpp = 3,
        .size = rgb_buf_len,
    };
#ifndef QCOM
    priv_hnds[i] = bufs[i].addr;
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


bool UIVision::ui_update() {
  if (!vision_connected) {
      vision_connect();
  }
  zmq_pollitem_t polls[1] = {{0}};
  // Take an rgb image from visiond if there is one
  while (true) {
    if (ipc_fd < 0) {
      // TODO: rethink this, for now it should only trigger on PC
      LOGW("vision disconnected by other thread");
      vision_connected = false;
      return false;
    }
    polls[0].fd = ipc_fd;
    polls[0].events = ZMQ_POLLIN;
#ifdef UI_60FPS
    // uses more CPU in both UI and surfaceflinger
    // 16% / 21%
    int ret = zmq_poll(polls, 1, 1);
#else
    // 9% / 13% = a 14% savings
    int ret = zmq_poll(polls, 1, 1000);
#endif
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;

      LOGE("poll failed (%d - %d)", ret, errno);
      close(ipc_fd);
      ipc_fd = -1;
      vision_connected = false;
      return false;
    } else if (ret == 0) {
      break;
    }
    // vision ipc event
    VisionPacket rp;
    int err = vipc_recv(ipc_fd, &rp);
    if (err <= 0) {
      LOGW("vision disconnected");
      close(ipc_fd);
      ipc_fd = -1;
      vision_connected = false;
      return false;
    }
    if (rp.type == VIPC_STREAM_ACQUIRE) {
      int idx = rp.d.stream_acq.idx;

      int release_idx = cur_vision_idx;
      if (release_idx >= 0) {
        VisionPacket rep = {
            .type = VIPC_STREAM_RELEASE,
            .d = {.stream_rel = {
                      .type = rp.d.stream_acq.type,
                      .idx = release_idx,
                  }},
        };
        vipc_send(ipc_fd, &rep);
      }

      assert(idx < UI_BUF_COUNT);
      cur_vision_idx = idx;
    } else {
      assert(false);
    }
    break;
  }
  return true;
}

void UIVision::draw_frame() {
  glBindVertexArray(frame_vao);
  glActiveTexture(GL_TEXTURE0);
  if (cur_vision_idx >= 0) {
    glBindTexture(GL_TEXTURE_2D, frame_texs[cur_vision_idx]);
    if (!front_view) {
#ifndef QCOM
      // TODO: a better way to do this?
      //printf("%d\n", ((int*)s->priv_hnds[s->cur_vision_idx])[0]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1164, 874, 0, GL_RGB, GL_UNSIGNED_BYTE, priv_hnds[cur_vision_idx]);
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
