#include <stdio.h>
#include <SDL.h>

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <cudaGL.h>
#include <assert.h>
#include <GL/glut.h>

#include "cereal/messaging/messaging.h"
#include "common/timing.h"

//#define SLOW

/*
clang++ compressed_low_latency.cc -I/usr/local/cuda/include -I/home/batman/Downloads/Video_Codec_SDK_11.1.5/Interface -o compressed_low_latency -lcuda -lnvcuvid -I/home/batman/openpilot /home/batman/openpilot/cereal/libmessaging.a -lzmq -lcapnp -lkj -I/usr/include/SDL2 -lSDL2 -lOpenGL
*/

#define CHECK(x) {CUresult ret = x; \
  if (ret != CUDA_SUCCESS) { \
    const char *str; \
    cuGetErrorName(ret, &str); \
    printf("error: %s\n", str); \
  } \
  assert(ret == CUDA_SUCCESS); \
}

CUdevice cuDevice;
CUcontext cuContext;
CUvideoctxlock m_ctxLock;
CUvideodecoder m_hDecoder = NULL;

SDL_Window *window;
SDL_Surface *window_surface;
SDL_Renderer* renderer;
SDL_Texture* texture;

uint64_t st;
bool parsed;

CUgraphicsResource res[2];

int HandleVideoSequence(void *junk, CUVIDEOFORMAT *pVideoFormat) {
  int nDecodeSurface = 20;
  printf("HandleVideoSequence %dx%d %d-%d %d-%d\n",
    pVideoFormat->coded_width, pVideoFormat->coded_height,
    pVideoFormat->display_area.left, pVideoFormat->display_area.right,
    pVideoFormat->display_area.top, pVideoFormat->display_area.bottom);
  CHECK(cuvidCtxLockCreate(&m_ctxLock, cuContext));
  CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
  videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
  videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
  videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
  videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
  videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
  videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
  videoDecodeCreateInfo.vidLock = m_ctxLock;
  videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
  videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
  videoDecodeCreateInfo.ulMaxWidth = videoDecodeCreateInfo.ulWidth;
  videoDecodeCreateInfo.ulMaxHeight = videoDecodeCreateInfo.ulHeight;
  videoDecodeCreateInfo.ulTargetWidth = videoDecodeCreateInfo.ulWidth;
  videoDecodeCreateInfo.ulTargetHeight = videoDecodeCreateInfo.ulHeight;
  CHECK(cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo))
  return nDecodeSurface;
}

int HandlePictureDecode(void *junk, CUVIDPICPARAMS *pPicParams) {
  CHECK(cuvidDecodePicture(m_hDecoder, pPicParams));
  CUVIDPROCPARAMS videoProcessingParameters = { 0 };
  CUdeviceptr dpSrcFrame = 0;
  unsigned int nSrcPitch = 0;
  CHECK(cuvidMapVideoFrame(m_hDecoder, pPicParams->CurrPicIdx, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

  uint64_t ct1 = nanos_since_boot();

#ifdef SLOW
  void *pixels;
  int pitch;
  SDL_LockTexture(texture, NULL, &pixels, &pitch);
  assert(pitch == nSrcPitch);
  cuMemcpy((CUdeviceptr)pixels, dpSrcFrame, 2048*1216*3/2);
  SDL_UnlockTexture(texture);
#else
  CHECK(cuGraphicsMapResources(2, res, NULL));

  for (int plane = 0; plane < 2; plane++) {
    CUarray texture_array;
    CHECK(cuGraphicsSubResourceGetMappedArray(&texture_array, res[plane], 0, 0));

    /*CUDA_ARRAY_DESCRIPTOR pArrayDescriptor;
    CHECK(cuArrayGetDescriptor(&pArrayDescriptor, texture_array));
    printf("dst array is %p = %ldx%ld %d\n", texture_array,
      pArrayDescriptor.Width, pArrayDescriptor.Height, pArrayDescriptor.NumChannels);*/

    // copy in the data
    CUDA_MEMCPY2D cu2d = {0};
    cu2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu2d.srcDevice = dpSrcFrame;
    cu2d.srcPitch = nSrcPitch;

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

  // unmap
  //CHECK(cuArrayDestroy(texture_array));
  CHECK(cuGraphicsUnmapResources(2, res, NULL));
#endif

  uint64_t ct2 = nanos_since_boot();

  SDL_Rect screen{0, 0, 1928, 1208};
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, &screen, NULL);
  SDL_RenderPresent(renderer);

  uint64_t et = nanos_since_boot()-st;
  printf("pc latency: %.2f ms (copy %.2f ms)\n", et/1e6, (ct2-ct1)/1e6);

  CHECK(cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
  parsed = true;
  return 1;
}

int main() {
  // init cuda
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&cuDevice, 0));
  CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  SDL_Init(SDL_INIT_VIDEO);
  SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
  window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1928, 1208, 0);
  assert(window != NULL);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  assert(renderer != NULL);
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_NV12,
    SDL_TEXTUREACCESS_STREAMING, 2048, 1216);
  assert(texture != NULL);

  // init cuvid
  CUvideoparser m_hParser = NULL;
  CUVIDPARSERPARAMS videoParserParameters = { };
  videoParserParameters.CodecType = cudaVideoCodec_HEVC;
  videoParserParameters.ulMaxNumDecodeSurfaces = 1;
  videoParserParameters.ulClockRate = 0;
  videoParserParameters.ulMaxDisplayDelay = 0;
  videoParserParameters.pUserData = NULL;
  videoParserParameters.pfnSequenceCallback = HandleVideoSequence;
  videoParserParameters.pfnDecodePicture = HandlePictureDecode;
  videoParserParameters.pfnDisplayPicture = NULL;
  videoParserParameters.pfnGetOperatingPoint = NULL;
  CHECK(cuvidCreateVideoParser(&m_hParser, &videoParserParameters));

  // get opengl texture number
  SDL_GL_BindTexture(texture, NULL, NULL);
  GLint whichID;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &whichID);
  SDL_GL_UnbindTexture(texture);

  // link the texture to CUDA
  CHECK(cuGraphicsGLRegisterImage(&res[0], whichID,
    GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
  CHECK(cuGraphicsGLRegisterImage(&res[1], whichID+1,
    GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

  setenv("ZMQ", "1", 1);
  Context * c = Context::create();
  SubSocket *sub_sock = SubSocket::create(c, "roadEncodeData", "192.168.3.188");
  bool seen_header = false;
  while (1) {
    Message *msg = sub_sock->receive();
    capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
    auto event = cmsg.getRoot<cereal::Event>();
    auto edata = event.getRoadEncodeData();
    if (!seen_header) {
      auto header = edata.getHeader();
      if (header.size() > 0) {
        printf("got header\n");
        seen_header = true;
        CUVIDSOURCEDATAPACKET packet = { 0 };
        packet.payload = header.begin();
        packet.payload_size = header.size();
        packet.flags = 0;
        CHECK(cuvidParseVideoData(m_hParser, &packet));
      } else {
        continue;
      }
    }

    auto data = edata.getData();
    CUVIDSOURCEDATAPACKET packet = { 0 };
    packet.payload = data.begin();
    packet.payload_size = data.size();
    packet.flags = CUVID_PKT_ENDOFPICTURE;

    st = nanos_since_boot();
    parsed = false;
    CHECK(cuvidParseVideoData(m_hParser, &packet));
    assert(parsed);

    delete msg;
  }
}
