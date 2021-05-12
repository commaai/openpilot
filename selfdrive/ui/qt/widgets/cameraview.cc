
#include "selfdrive/ui/qt/widgets/cameraview.h"

#include <iostream>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/qt_window.h"


static const char frame_vertex_shader[] =
#ifdef NANOVG_GL3_IMPLEMENTATION
  "#version 150 core\n"
#else
  "#version 300 es\n"
#endif
  "in vec4 aPosition;\n"
  "in vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
#ifdef NANOVG_GL3_IMPLEMENTATION
  "#version 150 core\n"
#else
  "#version 300 es\n"
#endif
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "in vec4 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  colorOut = texture(uTexture, vTexCoord.xy);\n"
#ifdef QCOM
  "  vec3 dz = vec3(0.0627f, 0.0627f, 0.0627f);\n"
  "  colorOut.rgb = ((vec3(1.0f, 1.0f, 1.0f) - dz) * colorOut.rgb / vec3(1.0f, 1.0f, 1.0f)) + dz;\n"
#endif
  "}\n";

static const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

static mat4 get_driver_view_transform() {
  const float driver_view_ratio = 1.333;
  mat4 transform;
  if (Hardware::TICI()) {
    // from dmonitoring.cc
    const int full_width_tici = 1928;
    const int full_height_tici = 1208;
    const int adapt_width_tici = 668;
    const int crop_x_offset = 32;
    const int crop_y_offset = -196;
    const float yscale = full_height_tici * driver_view_ratio / adapt_width_tici;
    const float xscale = yscale*(1080-2*bdr_s)/(2160-2*bdr_s)*full_width_tici/full_height_tici;
    transform = (mat4){{
      xscale,  0.0, 0.0, xscale*crop_x_offset/full_width_tici*2,
      0.0,  yscale, 0.0, yscale*crop_y_offset/full_height_tici*2,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  
  } else {
     // frame from 4/3 to 16/9 display
    transform = (mat4){{
      driver_view_ratio*(1080-2*bdr_s)/(1920-2*bdr_s),  0.0, 0.0, 0.0,
      0.0,  1.0, 0.0, 0.0,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  }
  return transform;
}

CameraViewWidget::CameraViewWidget(QWidget *parent) : QOpenGLWidget(parent) {

}

CameraViewWidget::~CameraViewWidget() {
  makeCurrent();
  doneCurrent();
}

void CameraViewWidget::initializeGL() {
  initializeOpenGLFunctions();

  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
 
  UIState *s = &QUIState::ui_state;
  // init gl
  if (!gl_shader) {
    gl_shader = std::make_unique<GLShader>(frame_vertex_shader, frame_fragment_shader);
  }
  GLint frame_pos_loc = glGetAttribLocation(gl_shader->prog, "aPosition");
  GLint frame_texcoord_loc = glGetAttribLocation(gl_shader->prog, "aTexCoord");

  auto [x1, x2, y1, y2] = s->scene.driver_view ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
    {-1.0, -1.0, x2, y1}, //bl
    {-1.0,  1.0, x2, y2}, //tl
    { 1.0,  1.0, x1, y2}, //tr
    { 1.0, -1.0, x1, y1}, //br
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

  if (s->scene.driver_view) {
    frame_mat = matmul(device_transform, get_driver_view_transform());
  } else {
    auto intrinsic_matrix = s->wide_camera ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
    float zx = s->zoom * 2 * intrinsic_matrix.v[2] / s->video_rect.w;
    float zy = s->zoom * 2 * intrinsic_matrix.v[5] / s->video_rect.h;

    const mat4 frame_transform = {{
      zx, 0.0, 0.0, 0.0,
      0.0, zy, 0.0, -y_offset / s->video_rect.h * 2,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
    }};
    frame_mat = matmul(device_transform, frame_transform);
  }
}

void CameraViewWidget::paintGL() {
  UIState *s = &QUIState::ui_state;
  s->viz_rect = Rect{bdr_s, bdr_s, width() - 2 * bdr_s, height() - 2 * bdr_s};
  const NVGcolor color = bg_colors[s->status];
  glClearColor(color.r, color.g, color.b, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glEnable(GL_SCISSOR_TEST);
  glViewport(s->video_rect.x, s->video_rect.y, s->video_rect.w, s->video_rect.h);
  glScissor(s->viz_rect.x, s->viz_rect.y, s->viz_rect.w, s->viz_rect.h);

  glBindVertexArray(frame_vao);
  glActiveTexture(GL_TEXTURE0);

  if (s->last_frame) {
    printf("paint here %d %d %d %d****\n", s->viz_rect.w, s->viz_rect.h, s->video_rect.w, s->video_rect.h);  
    glBindTexture(GL_TEXTURE_2D, s->texture[s->last_frame->idx]->frame_tex);
    if (!Hardware::EON()) {
      // this is handled in ion on QCOM
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s->last_frame->width, s->last_frame->height,
                   0, GL_RGB, GL_UNSIGNED_BYTE, s->last_frame->addr);
    }
  }

  glUseProgram(gl_shader->prog);
  glUniform1i(gl_shader->getUniformLocation("uTexture"), 0);
  glUniformMatrix4fv(gl_shader->getUniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);

  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);


  glDisable(GL_SCISSOR_TEST);

  glViewport(0, 0, s->fb_w, s->fb_h);
}

// class RoadView

RoadView::RoadView(QWidget* parent) : CameraViewWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
}

void RoadView::initializeGL() {
  ui_nvg_init(&QUIState::ui_state);
  CameraViewWidget::initializeGL();
    
  prev_draw_t = millis_since_boot();
}

void RoadView::update(const UIState &s) {
  // Connecting to visionIPC requires opengl to be current
  if (s.vipc_client->connected){
    makeCurrent();
  }
  repaint();
}

void RoadView::paintGL() {
  CameraViewWidget::paintGL();
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66 && !QUIState::ui_state.scene.driver_view) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}


// class DriverView

void DriverView::paintGL() {
  CameraViewWidget::paintGL();
}

