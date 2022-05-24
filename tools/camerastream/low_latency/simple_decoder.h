#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>

#include "common/timing.h"

#define CHECK(x) {CUresult ret = x; \
  if (ret != CUDA_SUCCESS) { \
    const char *str; \
    cuGetErrorName(ret, &str); \
    printf("error: %s\n", str); \
  } \
  assert(ret == CUDA_SUCCESS); \
}

class SimpleDecoder {
public:
  SimpleDecoder();

  CUdeviceptr decode(const unsigned char *dat, int len, bool is_header=false);
  void free_frame();

private:
  int HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat);
  int HandlePictureDecode(CUVIDPICPARAMS *pPicParams);

  static int HandleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat) noexcept {
    return ((SimpleDecoder *)pUserData)->HandleVideoSequence(pVideoFormat);
  }

  static int HandlePictureDecodeProc(void *pUserData, CUVIDPICPARAMS *pPicParams) noexcept {
    return ((SimpleDecoder *)pUserData)->HandlePictureDecode(pPicParams);
  }

  CUdeviceptr dpSrcFrame = 0;

  CUdevice cuDevice = 0;
  CUcontext cuContext = 0;
  CUvideoctxlock m_ctxLock = 0;
  CUvideodecoder m_hDecoder = NULL;
  CUvideoparser m_hParser = NULL;

  bool parsed = false;
};

void SimpleDecoder::free_frame() {
  CHECK(cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
  dpSrcFrame = 0;
}

CUdeviceptr SimpleDecoder::decode(const unsigned char *dat, int len, bool is_header) {
  CUVIDSOURCEDATAPACKET packet = { 0 };
  packet.payload = dat;
  packet.payload_size = len;
  packet.flags = is_header ? 0 : CUVID_PKT_ENDOFPICTURE;
  parsed = false;
  CHECK(cuvidParseVideoData(m_hParser, &packet));
  if (is_header) return 0;
  assert(parsed);
  return dpSrcFrame;
}

SimpleDecoder::SimpleDecoder() {
  // init cuda
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&cuDevice, 0));
  CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  // init cuvid
  CUVIDPARSERPARAMS videoParserParameters = { };
  videoParserParameters.CodecType = cudaVideoCodec_HEVC;
  videoParserParameters.ulMaxNumDecodeSurfaces = 1;
  videoParserParameters.ulClockRate = 0;
  videoParserParameters.ulMaxDisplayDelay = 0;
  videoParserParameters.pUserData = this;
  videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
  videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
  videoParserParameters.pfnDisplayPicture = NULL;
  videoParserParameters.pfnGetOperatingPoint = NULL;
  CHECK(cuvidCreateVideoParser(&m_hParser, &videoParserParameters));
}

int SimpleDecoder::HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat) {
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

int SimpleDecoder::HandlePictureDecode(CUVIDPICPARAMS *pPicParams) {
  CHECK(cuvidDecodePicture(m_hDecoder, pPicParams));
  if (dpSrcFrame) free_frame();

  unsigned int nSrcPitch = 0;
  CUVIDPROCPARAMS videoProcessingParameters = { 0 };
  CHECK(cuvidMapVideoFrame(m_hDecoder, pPicParams->CurrPicIdx, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

  parsed = true;
  return 1;


  /*

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

    CUDA_ARRAY_DESCRIPTOR pArrayDescriptor;
    CHECK(cuArrayGetDescriptor(&pArrayDescriptor, texture_array));
    printf("dst array is %p = %ldx%ld %d\n", texture_array,
      pArrayDescriptor.Width, pArrayDescriptor.Height, pArrayDescriptor.NumChannels);

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



  */
}
