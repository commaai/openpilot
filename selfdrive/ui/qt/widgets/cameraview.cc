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

const char frame_fragment_shader[] =
#ifdef __APPLE__
  "#version 150 core\n"
#else
  "#version 300 es\n"
  "precision mediump float;\n"
#endif
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

const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

mat4 get_driver_view_transform(int screen_width, int screen_height, int stream_width, int stream_height) {
  const float driver_view_ratio = 1.333;
  mat4 transform;
  if (stream_width == TICI_CAM_WIDTH) {
    const float yscale = stream_height * driver_view_ratio / tici_dm_crop::width;
    const float xscale = yscale*screen_height/screen_width*stream_width/stream_height;
    transform = (mat4){{
      xscale,  0.0, 0.0, xscale*tici_dm_crop::x_offset/stream_width*2,
      0.0,  yscale, 0.0, yscale*tici_dm_crop::y_offset/stream_height*2,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  } else {
    // frame from 4/3 to 16/9 display
    transform = (mat4){{
      driver_view_ratio * screen_height / screen_width,  0.0, 0.0, 0.0,
      0.0,  1.0, 0.0, 0.0,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  }
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
}

CameraViewWidget::~CameraViewWidget() {
  makeCurrent();
  if (isValid()) {
    glDeleteVertexArrays(1, &frame_vao);
    glDeleteBuffers(1, &frame_vbo);
    glDeleteBuffers(1, &frame_ibo);
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

  auto [x1, x2, y1, y2] = stream_type == VISION_STREAM_RGB_FRONT ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
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
}

void CameraViewWidget::showEvent(QShowEvent *event) {
  latest_texture_id = -1;
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
    if (stream_type == VISION_STREAM_RGB_FRONT) {
      frame_mat = matmul(device_transform, get_driver_view_transform(w, h, stream_width, stream_height));
    } else {
      auto intrinsic_matrix = stream_type == VISION_STREAM_RGB_WIDE ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
      float zoom = ZOOM / intrinsic_matrix.v[0];
      if (stream_type == VISION_STREAM_RGB_WIDE) {
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

  std::lock_guard lk(lock);

  if (latest_texture_id == -1) return;

  glViewport(0, 0, width(), height());
  // sync with the PBO
  if (wait_fence) {
    wait_fence->wait();
  }

  glBindVertexArray(frame_vao);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture[latest_texture_id]->frame_tex);

  glUseProgram(program->programId());
  glUniform1i(program->uniformLocation("uTexture"), 0);
  glUniformMatrix4fv(program->uniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);

  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
}

void CameraViewWidget::vipcConnected(VisionIpcClient *vipc_client) {
  makeCurrent();
  for (int i = 0; i < vipc_client->num_buffers; i++) {
    texture[i].reset(new EGLImageTexture(&vipc_client->buffers[i]));

    glBindTexture(GL_TEXTURE_2D, texture[i]->frame_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    assert(glGetError() == GL_NO_ERROR);
  }
  latest_texture_id = -1;
  stream_width = vipc_client->buffers[0].width;
  stream_height = vipc_client->buffers[0].height;
  updateFrameMat(width(), height());
}

void CameraViewWidget::vipcThread() {
  VisionStreamType cur_stream_type = stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;

  std::unique_ptr<QOpenGLContext> ctx;
  std::unique_ptr<QOffscreenSurface> surface;
  std::unique_ptr<QOpenGLBuffer> gl_buffer;

  if (!Hardware::EON()) {
    ctx = std::make_unique<QOpenGLContext>();
    ctx->setFormat(context()->format());
    ctx->setShareContext(context());
    ctx->create();
    assert(ctx->isValid());

    surface = std::make_unique<QOffscreenSurface>();
    surface->setFormat(ctx->format());
    surface->create();
    ctx->makeCurrent(surface.get());
    assert(QOpenGLContext::currentContext() == ctx.get());
    initializeOpenGLFunctions();
  }

  while (!QThread::currentThread()->isInterruptionRequested()) {
    if (!vipc_client || cur_stream_type != stream_type) {
      cur_stream_type = stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream_type, true));
    }

    if (!vipc_client->connected) {
      if (!vipc_client->connect(false)) {
        QThread::msleep(100);
        continue;
      }

      if (!Hardware::EON()) {
        gl_buffer.reset(new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer));
        gl_buffer->create();
        gl_buffer->bind();
        gl_buffer->setUsagePattern(QOpenGLBuffer::StreamDraw);
        gl_buffer->allocate(vipc_client->buffers[0].len);
      }

      emit vipcThreadConnected(vipc_client.get());
    }

    if (VisionBuf *buf = vipc_client->recv(nullptr, 1000)) {
      {
        std::lock_guard lk(lock);
        if (!Hardware::EON()) {
          void *texture_buffer = gl_buffer->map(QOpenGLBuffer::WriteOnly);
          memcpy(texture_buffer, buf->addr, buf->len);
          gl_buffer->unmap();

          // copy pixels from PBO to texture object
          glBindTexture(GL_TEXTURE_2D, texture[buf->idx]->frame_tex);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf->width, buf->height, GL_RGB, GL_UNSIGNED_BYTE, 0);
          glBindTexture(GL_TEXTURE_2D, 0);
          assert(glGetError() == GL_NO_ERROR);

          wait_fence.reset(new WaitFence());
        }
        latest_texture_id = buf->idx;
      }
      // Schedule update. update() will be invoked on the gui thread.
      QMetaObject::invokeMethod(this, "update");

      // TODO: remove later, it's only connected by DriverView.
      emit vipcThreadFrameReceived(buf);
    }
  }
}
