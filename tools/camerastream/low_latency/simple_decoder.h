#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>

#include "common/timing.h"

#define CHECK(x) {CUresult __ret = x; \
  if (__ret != CUDA_SUCCESS) { \
    const char *__str; \
    cuGetErrorName(__ret, &__str); \
    printf("error: %s\n", __str); \
  } \
  assert(__ret == CUDA_SUCCESS); \
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
