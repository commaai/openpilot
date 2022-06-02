#include "selfdrive/ui/qt/widgets/cameraview.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#include <QOpenGLBuffer>
#include <QOffscreenSurface>

namespace {

const char frame_vertex_shader[] =
#ifdef __APPLE__
  "#version 330 core\n"
#else
  "#version 300 es\n"
#endif
  "layout(location = 0) in vec4 aPosition;\n"
  "layout(location = 1) in vec2 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec2 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

const char frame_fragment_shader[] =
#ifdef __APPLE__
  "#version 330 core\n"
#else
  "#version 300 es\n"
  "precision mediump float;\n"
#endif
  "uniform sampler2D uTextureY;\n"
  "uniform sampler2D uTextureUV;\n"
  "in vec2 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  float y = texture(uTextureY, vTexCoord).r;\n"
  "  vec2 uv = texture(uTextureUV, vTexCoord).rg - 0.5;\n"
  "  float r = y + 1.402 * uv.y;\n"
  "  float g = y - 0.344 * uv.x - 0.714 * uv.y;\n"
  "  float b = y + 1.772 * uv.x;\n"
  "  colorOut = vec4(r, g, b, 1.0);\n"
  "}\n";

const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

mat4 get_driver_view_transform(int screen_width, int screen_height, int stream_width, int stream_height) {
  const float driver_view_ratio = 2.0;
  const float yscale = stream_height * driver_view_ratio / stream_width;
  const float xscale = yscale*screen_height/screen_width*stream_width/stream_height;
  mat4 transform = (mat4){{
    xscale,  0.0, 0.0, 0.0,
    0.0,  yscale, 0.0, 0.0,
    0.0,  0.0, 1.0, 0.0,
    0.0,  0.0, 0.0, 1.0,
  }};
  return transform;
}

mat4 get_fit_view_transform(float widget_aspect_ratio, float frame_aspect_ratio) {
  float zx = 1, zy = 1;
  if (frame_aspect_ratio > widget_aspect_ratio) {
    zy = widget_aspect_ratio / frame_aspect_ratio;
  } else {
    zx = frame_aspect_ratio / widget_aspect_ratio;
  }

  const mat4 frame_transform = {{
    zx, 0.0, 0.0, 0.0,
    0.0, zy, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};
  return frame_transform;
}

} // namespace

CameraViewWidget::CameraViewWidget(std::string stream_name, VisionStreamType type, bool zoom, QWidget* parent) :
                                   stream_name(stream_name), stream_type(type), zoomed_view(zoom), QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  connect(this, &CameraViewWidget::vipcThreadConnected, this, &CameraViewWidget::vipcConnected, Qt::BlockingQueuedConnection);
  connect(this, &CameraViewWidget::vipcThreadFrameReceived, this, &CameraViewWidget::vipcFrameReceived);
}

CameraViewWidget::~CameraViewWidget() {
  makeCurrent();
  if (isValid()) {
    glDeleteVertexArrays(1, &frame_vao);
    glDeleteBuffers(1, &frame_vbo);
    glDeleteBuffers(1, &frame_ibo);
    glDeleteBuffers(3, textures);
  }
  doneCurrent();
}

void CameraViewWidget::initializeGL() {
  initializeOpenGLFunctions();

  program = std::make_unique<QOpenGLShaderProgram>(context());
  bool ret = program->addShaderFromSourceCode(QOpenGLShader::Vertex, frame_vertex_shader);
  assert(ret);
  ret = program->addShaderFromSourceCode(QOpenGLShader::Fragment, frame_fragment_shader);
  assert(ret);

  program->link();
  GLint frame_pos_loc = program->attributeLocation("aPosition");
  GLint frame_texcoord_loc = program->attributeLocation("aTexCoord");

  auto [x1, x2, y1, y2] = stream_type == VISION_STREAM_DRIVER ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
    {-1.0, -1.0, x2, y1}, // bl
    {-1.0,  1.0, x2, y2}, // tl
    { 1.0,  1.0, x1, y2}, // tr
    { 1.0, -1.0, x1, y1}, // br
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

  glGenTextures(3, textures);
  glUseProgram(program->programId());
  glUniform1i(program->uniformLocation("uTextureY"), 0);
  glUniform1i(program->uniformLocation("uTextureUV"), 1);
}

void CameraViewWidget::showEvent(QShowEvent *event) {
  frames.clear();
  if (!vipc_thread) {
    vipc_thread = new QThread();
    connect(vipc_thread, &QThread::started, [=]() { vipcThread(); });
    connect(vipc_thread, &QThread::finished, vipc_thread, &QObject::deleteLater);
    vipc_thread->start();
  }
}

void CameraViewWidget::hideEvent(QHideEvent *event) {
  if (vipc_thread) {
    vipc_thread->requestInterruption();
    vipc_thread->quit();
    vipc_thread->wait();
    vipc_thread = nullptr;
  }
}

void CameraViewWidget::updateFrameMat(int w, int h) {
  if (zoomed_view) {
    if (stream_type == VISION_STREAM_DRIVER) {
      frame_mat = matmul(device_transform, get_driver_view_transform(w, h, stream_width, stream_height));
    } else {
      auto intrinsic_matrix = stream_type == VISION_STREAM_WIDE_ROAD ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
      float zoom = ZOOM / intrinsic_matrix.v[0];
      if (stream_type == VISION_STREAM_WIDE_ROAD) {
        zoom *= 0.5;
      }
      float zx = zoom * 2 * intrinsic_matrix.v[2] / width();
      float zy = zoom * 2 * intrinsic_matrix.v[5] / height();

      const mat4 frame_transform = {{
        zx, 0.0, 0.0, 0.0,
        0.0, zy, 0.0, -y_offset / height() * 2,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
      }};
      frame_mat = matmul(device_transform, frame_transform);
    }
  } else if (stream_width > 0 && stream_height > 0) {
    // fit frame to widget size
    float widget_aspect_ratio = (float)width() / height();
    float frame_aspect_ratio = (float)stream_width  / stream_height;
    frame_mat = matmul(device_transform, get_fit_view_transform(widget_aspect_ratio, frame_aspect_ratio));
  }
}

void CameraViewWidget::paintGL() {
  glClearColor(bg.redF(), bg.greenF(), bg.blueF(), bg.alphaF());
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  if (frames.empty()) return;

  int frame_idx;
  for (frame_idx = 0; frame_idx < frames.size() - 1; frame_idx++) {
    if (frames[frame_idx].first == draw_frame_id) break;
  }
  VisionBuf *frame = frames[frame_idx].second;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride);
  glViewport(0, 0, width(), height());
  glBindVertexArray(frame_vao);

  glUseProgram(program->programId());

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width, stream_height, GL_RED, GL_UNSIGNED_BYTE, frame->y);
  assert(glGetError() == GL_NO_ERROR);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride/2);

  glActiveTexture(GL_TEXTURE0 + 1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width/2, stream_height/2, GL_RG, GL_UNSIGNED_BYTE, frame->uv);
  assert(glGetError() == GL_NO_ERROR);

  glUniformMatrix4fv(program->uniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);
  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

void CameraViewWidget::vipcConnected(VisionIpcClient *vipc_client) {
  makeCurrent();
  frames.clear();
  stream_width = vipc_client->buffers[0].width;
  stream_height = vipc_client->buffers[0].height;
  stream_stride = vipc_client->buffers[0].stride;

  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, stream_width, stream_height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  assert(glGetError() == GL_NO_ERROR);

  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, stream_width/2, stream_height/2, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);
  assert(glGetError() == GL_NO_ERROR);

  updateFrameMat(width(), height());
}

void CameraViewWidget::vipcFrameReceived(VisionBuf *buf, uint32_t frame_id) {
  frames.push_back(std::make_pair(frame_id, buf));
  while (frames.size() > FRAME_BUFFER_SIZE) {
    frames.pop_front();
  }
  update();
}

void CameraViewWidget::vipcThread() {
  VisionStreamType cur_stream_type = stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra meta_main = {0};

  while (!QThread::currentThread()->isInterruptionRequested()) {
    if (!vipc_client || cur_stream_type != stream_type) {
      cur_stream_type = stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream_type, false));
    }

    if (!vipc_client->connected) {
      if (!vipc_client->connect(false)) {
        QThread::msleep(100);
        continue;
      }
      emit vipcThreadConnected(vipc_client.get());
    }

    if (VisionBuf *buf = vipc_client->recv(&meta_main, 1000)) {
      emit vipcThreadFrameReceived(buf, meta_main.frame_id);
    }
  }
}
