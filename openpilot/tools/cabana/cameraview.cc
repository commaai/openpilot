#include "tools/cabana/cameraview.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#include <QApplication>

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

} // namespace

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type, QWidget* parent) :
                          stream_name(stream_name), active_stream_type(type), requested_stream_type(type), QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  qRegisterMetaType<std::set<VisionStreamType>>("availableStreams");
  QObject::connect(this, &CameraWidget::vipcThreadConnected, this, &CameraWidget::vipcConnected, Qt::BlockingQueuedConnection);
  QObject::connect(this, &CameraWidget::vipcThreadFrameReceived, this, &CameraWidget::vipcFrameReceived, Qt::QueuedConnection);
  QObject::connect(this, &CameraWidget::vipcAvailableStreamsUpdated, this, &CameraWidget::availableStreamsUpdated, Qt::QueuedConnection);
  QObject::connect(QApplication::instance(), &QCoreApplication::aboutToQuit, this, &CameraWidget::stopVipcThread);
}

CameraWidget::~CameraWidget() {
  makeCurrent();
  stopVipcThread();
  if (isValid()) {
    glDeleteVertexArrays(1, &frame_vao);
    glDeleteBuffers(1, &frame_vbo);
    glDeleteBuffers(1, &frame_ibo);
    glDeleteTextures(2, textures);
    shader_program_.reset();
  }
  doneCurrent();
}

void CameraWidget::initializeGL() {
  initializeOpenGLFunctions();

  shader_program_ = std::make_unique<QOpenGLShaderProgram>(context());
  shader_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, frame_vertex_shader);
  shader_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, frame_fragment_shader);
  shader_program_->link();

  GLint frame_pos_loc = shader_program_->attributeLocation("aPosition");
  GLint frame_texcoord_loc = shader_program_->attributeLocation("aTexCoord");

  auto [x1, x2, y1, y2] = requested_stream_type == VISION_STREAM_DRIVER ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
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

  glGenTextures(2, textures);

  shader_program_->bind();
  shader_program_->setUniformValue("uTextureY", 0);
  shader_program_->setUniformValue("uTextureUV", 1);
  shader_program_->release();
}

void CameraWidget::showEvent(QShowEvent *event) {
  if (!vipc_thread) {
    clearFrames();
    vipc_thread = new QThread();
    connect(vipc_thread, &QThread::started, [=]() { vipcThread(); });
    connect(vipc_thread, &QThread::finished, vipc_thread, &QObject::deleteLater);
    vipc_thread->start();
  }
}

void CameraWidget::stopVipcThread() {
  makeCurrent();
  if (vipc_thread) {
    vipc_thread->requestInterruption();
    vipc_thread->quit();
    vipc_thread->wait();
    vipc_thread = nullptr;
  }
}

void CameraWidget::availableStreamsUpdated(std::set<VisionStreamType> streams) {
  available_streams = streams;
}

void CameraWidget::paintGL() {
  glClearColor(bg.redF(), bg.greenF(), bg.blueF(), bg.alphaF());
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  std::lock_guard lk(frame_lock);
  if (!current_frame_) return;

  // Scale for aspect ratio
  float widget_ratio = (float)width() / height();
  float frame_ratio = (float)stream_width / stream_height;
  float scale_x = std::min(frame_ratio / widget_ratio, 1.0f);
  float scale_y = std::min(widget_ratio / frame_ratio, 1.0f);

  glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio());

  shader_program_->bind();
  QMatrix4x4 transform;
  transform.scale(scale_x, scale_y, 1.0f);
  shader_program_->setUniformValue("uTransform", transform);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width, stream_height, GL_RED, GL_UNSIGNED_BYTE, current_frame_->y);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride/2);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width/2, stream_height/2, GL_RG, GL_UNSIGNED_BYTE, current_frame_->uv);

  glBindVertexArray(frame_vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, nullptr);
  glBindVertexArray(0);

  // Reset both texture units
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

  shader_program_->release();
}

void CameraWidget::vipcConnected(VisionIpcClient *vipc_client) {
  makeCurrent();
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
}

void CameraWidget::vipcFrameReceived() {
  update();
}

void CameraWidget::vipcThread() {
  VisionStreamType cur_stream = requested_stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra frame_meta = {};

  while (!QThread::currentThread()->isInterruptionRequested()) {
    if (!vipc_client || cur_stream != requested_stream_type) {
      clearFrames();
      qDebug().nospace() << "connecting to stream " << requested_stream_type << ", was connected to " << cur_stream;
      cur_stream = requested_stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream, false));
    }
    active_stream_type = cur_stream;

    if (!vipc_client->connected) {
      clearFrames();
      auto streams = VisionIpcClient::getAvailableStreams(stream_name, false);
      if (streams.empty()) {
        QThread::msleep(100);
        continue;
      }
      emit vipcAvailableStreamsUpdated(streams);

      if (!vipc_client->connect(false)) {
        QThread::msleep(100);
        continue;
      }
      emit vipcThreadConnected(vipc_client.get());
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      {
        std::lock_guard lk(frame_lock);
        current_frame_ = buf;
        frame_meta_ = frame_meta;
      }
      emit vipcThreadFrameReceived();
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  current_frame_ = nullptr;
  available_streams.clear();
}
