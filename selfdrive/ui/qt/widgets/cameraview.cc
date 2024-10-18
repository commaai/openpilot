#include "selfdrive/ui/qt/widgets/cameraview.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

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

CameraWidget::CameraWidget(const std::string &stream_name, VisionStreamType type, QWidget *parent)
    : stream_name(stream_name), QOpenGLWidget(parent) {
  qRegisterMetaType<std::set<VisionStreamType>>("availableStreams");
  vipc_client = std::make_unique<VisionIpcClient>(stream_name, type, false);
}

CameraWidget::~CameraWidget() {
  makeCurrent();
  if (isValid()) {
    glDeleteVertexArrays(1, &frame_vao);
    glDeleteBuffers(1, &frame_vbo);
    glDeleteBuffers(1, &frame_ibo);
    glDeleteTextures(2, textures);
  }
  doneCurrent();
  clearEGLImages();
}

void CameraWidget::clearEGLImages() {
#ifdef QCOM2
  EGLDisplay egl_display = eglGetCurrentDisplay();
  for (auto &pair : egl_images) {
    eglDestroyImageKHR(egl_display, pair.second);
  }
  egl_images.clear();
#endif
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

  auto [x1, x2, y1, y2] = vipc_client->type == VISION_STREAM_DRIVER ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
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

  auto frame_mat = calcFrameMatrix();
  auto frame = receiveFrame();
  if (!frame) return;

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

void CameraWidget::vipcConnected() {
  makeCurrent();
  stream_width = vipc_client->buffers[0].width;
  stream_height = vipc_client->buffers[0].height;
  stream_stride = vipc_client->buffers[0].stride;

#ifdef QCOM2
  clearEGLImages();
  EGLDisplay egl_display = eglGetCurrentDisplay();
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

void CameraWidget::setStreamType(VisionStreamType type) {
  if (type != vipc_client->type) {
    qDebug().nospace() << "connecting to stream " << type;
    vipc_client.reset(new VisionIpcClient(stream_name, type, false));
  }
}

VisionBuf *CameraWidget::receiveFrame() {
  if (!vipc_client->connected) {
    vision_buf = nullptr;
    available_streams = VisionIpcClient::getAvailableStreams(stream_name, false);
    if (available_streams.empty() || !vipc_client->connect(false)) {
      return nullptr;
    }
    emit vipcAvailableStreamsUpdated(available_streams);
    vipcConnected();
  }

  if (auto frame = vipc_client->recv(nullptr, 0)) {
    vision_buf = frame;
  }
  return vision_buf;
}
