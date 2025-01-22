#include "selfdrive/ui/qt/widgets/cameraview.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#include <cmath>
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
#ifdef QCOM2
  "#version 300 es\n"
  "#extension GL_OES_EGL_image_external_essl3 : enable\n"
  "precision mediump float;\n"
  "uniform samplerExternalOES uTexture;\n"
  "in vec2 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  colorOut = texture(uTexture, vTexCoord);\n"
  // gamma to improve worst case visibility when dark
  "  colorOut.rgb = pow(colorOut.rgb, vec3(1.0/1.28));\n"
  "}\n";
#else
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
#endif

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
#ifndef QCOM2
    glDeleteTextures(2, textures);
#endif
  }
  doneCurrent();
}

// Qt uses device-independent pixels, depending on platform this may be
// different to what OpenGL uses
int CameraWidget::glWidth() {
    return width() * devicePixelRatio();
}

int CameraWidget::glHeight() {
  return height() * devicePixelRatio();
}

void CameraWidget::initializeGL() {
  initializeOpenGLFunctions();

  program = std::make_unique<QOpenGLShaderProgram>(context());
  bool ret = program->addShaderFromSourceCode(QOpenGLShader::Vertex, frame_vertex_shader);
  assert(ret);
  ret = program->addShaderFromSourceCode(QOpenGLShader::Fragment, frame_fragment_shader);
  assert(ret);

  program->link();
  GLint frame_pos_loc = program->attributeLocation("aPosition");
  GLint frame_texcoord_loc = program->attributeLocation("aTexCoord");

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

  glUseProgram(program->programId());

#ifdef QCOM2
  glUniform1i(program->uniformLocation("uTexture"), 0);
#else
  glGenTextures(2, textures);
  glUniform1i(program->uniformLocation("uTextureY"), 0);
  glUniform1i(program->uniformLocation("uTextureUV"), 1);
#endif
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

#ifdef QCOM2
  EGLDisplay egl_display = eglGetCurrentDisplay();
  assert(egl_display != EGL_NO_DISPLAY);
  for (auto &pair : egl_images) {
    eglDestroyImageKHR(egl_display, pair.second);
    assert(eglGetError() == EGL_SUCCESS);
  }
  egl_images.clear();
#endif
}

void CameraWidget::availableStreamsUpdated(std::set<VisionStreamType> streams) {
  available_streams = streams;
}

mat4 CameraWidget::calcFrameMatrix() {
  // Scale the frame to fit the widget while maintaining the aspect ratio.
  float widget_aspect_ratio = (float)width() / height();
  float frame_aspect_ratio = (float)stream_width / stream_height;
  float zx = std::min(frame_aspect_ratio / widget_aspect_ratio, 1.0f);
  float zy = std::min(widget_aspect_ratio / frame_aspect_ratio, 1.0f);

  return mat4{{
    zx, 0.0, 0.0, 0.0,
    0.0, zy, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};
}

void CameraWidget::paintGL() {
  glClearColor(bg.redF(), bg.greenF(), bg.blueF(), bg.alphaF());
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  std::lock_guard lk(frame_lock);
  if (frames.empty()) return;

  int frame_idx = frames.size() - 1;

  // Always draw latest frame until sync logic is more stable
  // for (frame_idx = 0; frame_idx < frames.size() - 1; frame_idx++) {
  //   if (frames[frame_idx].first == draw_frame_id) break;
  // }

  // Log duplicate/dropped frames
  if (frames[frame_idx].first == prev_frame_id) {
    qDebug() << "Drawing same frame twice" << frames[frame_idx].first;
  } else if (frames[frame_idx].first != prev_frame_id + 1) {
    qDebug() << "Skipped frame" << frames[frame_idx].first;
  }
  prev_frame_id = frames[frame_idx].first;
  VisionBuf *frame = frames[frame_idx].second;
  assert(frame != nullptr);

  auto frame_mat = calcFrameMatrix();

  glViewport(0, 0, glWidth(), glHeight());
  glBindVertexArray(frame_vao);
  glUseProgram(program->programId());
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

#ifdef QCOM2
  // no frame copy
  glActiveTexture(GL_TEXTURE0);
  glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, egl_images[frame->idx]);
  assert(glGetError() == GL_NO_ERROR);
#else
  // fallback to copy
  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width, stream_height, GL_RED, GL_UNSIGNED_BYTE, frame->y);
  assert(glGetError() == GL_NO_ERROR);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, stream_stride/2);
  glActiveTexture(GL_TEXTURE0 + 1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stream_width/2, stream_height/2, GL_RG, GL_UNSIGNED_BYTE, frame->uv);
  assert(glGetError() == GL_NO_ERROR);
#endif

  glUniformMatrix4fv(program->uniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

void CameraWidget::vipcConnected(VisionIpcClient *vipc_client) {
  makeCurrent();
  stream_width = vipc_client->buffers[0].width;
  stream_height = vipc_client->buffers[0].height;
  stream_stride = vipc_client->buffers[0].stride;

#ifdef QCOM2
  EGLDisplay egl_display = eglGetCurrentDisplay();
  assert(egl_display != EGL_NO_DISPLAY);
  for (auto &pair : egl_images) {
    eglDestroyImageKHR(egl_display, pair.second);
  }
  egl_images.clear();

  for (int i = 0; i < vipc_client->num_buffers; i++) {  // import buffers into OpenGL
    int fd = dup(vipc_client->buffers[i].fd);  // eglDestroyImageKHR will close, so duplicate
    EGLint img_attrs[] = {
      EGL_WIDTH, (int)vipc_client->buffers[i].width,
      EGL_HEIGHT, (int)vipc_client->buffers[i].height,
      EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_NV12,
      EGL_DMA_BUF_PLANE0_FD_EXT, fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
      EGL_DMA_BUF_PLANE0_PITCH_EXT, (int)vipc_client->buffers[i].stride,
      EGL_DMA_BUF_PLANE1_FD_EXT, fd,
      EGL_DMA_BUF_PLANE1_OFFSET_EXT, (int)vipc_client->buffers[i].uv_offset,
      EGL_DMA_BUF_PLANE1_PITCH_EXT, (int)vipc_client->buffers[i].stride,
      EGL_NONE
    };
    egl_images[i] = eglCreateImageKHR(egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, 0, img_attrs);
    assert(eglGetError() == EGL_SUCCESS);
  }
#else
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
#endif
}

void CameraWidget::vipcFrameReceived() {
  update();
}

void CameraWidget::vipcThread() {
  VisionStreamType cur_stream = requested_stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra meta_main = {0};

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

    if (VisionBuf *buf = vipc_client->recv(&meta_main, 1000)) {
      {
        std::lock_guard lk(frame_lock);
        frames.push_back(std::make_pair(meta_main.frame_id, buf));
        while (frames.size() > FRAME_BUFFER_SIZE) {
          frames.pop_front();
        }
      }
      emit vipcThreadFrameReceived();
    } else {
      if (!isVisible()) {
        vipc_client->connected = false;
      }
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  frames.clear();
  available_streams.clear();
}
