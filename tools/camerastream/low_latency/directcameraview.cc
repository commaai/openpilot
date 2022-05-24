#include <GLES3/gl3.h>
#include "cereal/messaging/messaging.h"

#include "directcameraview.h"

extern "C" {
CUresult cuGraphicsGLRegisterImage ( CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int  Flags );
}

DirectCameraViewWidget::DirectCameraViewWidget(const char* stream_name, const char* addr) :
  stream_name(stream_name), addr(addr) {

  connect(this, &DirectCameraViewWidget::vipcThreadFrameReceived, this, &DirectCameraViewWidget::vipcFrameReceived);
}

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
  "uniform sampler2D uTextureU;\n"
  "uniform sampler2D uTextureV;\n"
  "in vec2 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  float y = texture(uTextureY, vTexCoord).r;\n"
  "  float u = texture(uTextureU, vTexCoord).r - 0.5;\n"
  "  float v = texture(uTextureV, vTexCoord).r - 0.5;\n"
  "  float r = y + 1.402 * v;\n"
  "  float g = y - 0.344 * u - 0.714 * v;\n"
  "  float b = y + 1.772 * u;\n"
  "  colorOut = vec4(r, g, b, 1.0);\n"
  "}\n";


void DirectCameraViewWidget::showEvent(QShowEvent *event) {
  if (!vipc_thread) {
    vipc_thread = new QThread();
    connect(vipc_thread, &QThread::started, [=]() { vipcThread(); });
    connect(vipc_thread, &QThread::finished, vipc_thread, &QObject::deleteLater);
    vipc_thread->start();
  }
}

void DirectCameraViewWidget::initializeGL() {
  initializeOpenGLFunctions();

  program = std::make_unique<QOpenGLShaderProgram>(context());
  bool ret = program->addShaderFromSourceCode(QOpenGLShader::Vertex, frame_vertex_shader);
  assert(ret);
  ret = program->addShaderFromSourceCode(QOpenGLShader::Fragment, frame_fragment_shader);
  assert(ret);

  program->link();
  GLint frame_pos_loc = program->attributeLocation("aPosition");
  GLint frame_texcoord_loc = program->attributeLocation("aTexCoord");

  auto [x1, x2, y1, y2] = std::tuple(1.f, 0.f, 1.f, 0.f);
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
  glUniform1i(program->uniformLocation("uTextureU"), 1);
  glUniform1i(program->uniformLocation("uTextureV"), 2);

  // link textures
  decoder.reset(new SimpleDecoder);
  /*CHECK(cuGraphicsGLRegisterImage(&res[0], textures[0], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
  CHECK(cuGraphicsGLRegisterImage(&res[1], textures[1], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
  CHECK(cuGraphicsGLRegisterImage(&res[2], textures[2], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));*/
}

void DirectCameraViewWidget::paintGL() {
  printf("paintGL\n");
}

void DirectCameraViewWidget::vipcFrameReceived(unsigned char *dat, int len) {
  printf("frame loaded %d\n", len);
  CUdeviceptr dpSrcFrame = decoder->decode(dat, len);
  
  CHECK(cuGraphicsMapResources(3, res, NULL));
  for (int plane = 0; plane < 1; plane++) {
    CUarray texture_array;
    CHECK(cuGraphicsSubResourceGetMappedArray(&texture_array, res[plane], 0, 0));

    // copy in the data
    CUDA_MEMCPY2D cu2d = {0};
    cu2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu2d.srcDevice = dpSrcFrame;
    cu2d.srcPitch = 2048;

    cu2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cu2d.dstArray = texture_array;

    cu2d.WidthInBytes = 2048;

    if (plane == 1) {
      cu2d.Height = 1216/2;
      cu2d.srcY = 1216;
    } else {
      cu2d.Height = 1216;
    }

    CHECK(cuMemcpy2D(&cu2d));
  }
  CHECK(cuGraphicsUnmapResources(3, res, NULL));
  decoder->free_frame();
  free(dat);
  update();

}

void DirectCameraViewWidget::vipcThread() {
  setenv("ZMQ", "1", 1);
  Context * c = Context::create();
  auto sub_sock = SubSocket::create(c, stream_name, addr);

  bool seen_header = false;
  while (!QThread::currentThread()->isInterruptionRequested()) {
    Message *msg = sub_sock->receive();
    capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
    auto event = cmsg.getRoot<cereal::Event>();
    auto edata = event.getRoadEncodeData();

    if (!seen_header) {
      auto header = edata.getHeader();
      auto data = edata.getData();
      if (header.size() > 0) {
        printf("got header\n");
        seen_header = true;
        unsigned char *snd = (unsigned char *)malloc(header.size()+data.size());
        memcpy(snd, header.begin(), header.size());
        memcpy(snd+header.size(), data.begin(), data.size());
        emit vipcFrameReceived(snd, header.size()+data.size());
      }
    } else {
      auto data = edata.getData();
      unsigned char *snd = (unsigned char *)malloc(data.size());
      memcpy(snd, data.begin(), data.size());
      emit vipcThreadFrameReceived(snd, data.size());
    }
  }
}