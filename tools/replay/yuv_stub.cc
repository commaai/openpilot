// Stub for libyuv's I420ToNV12 to satisfy linker when building
// the Cython extension without the non-PIC vendored libyuv.a.
// This code path is only reached when video frames are decoded
// (i.e. not with --no-vipc).
#include <cstdint>
#include <cstdlib>

extern "C" int I420ToNV12(const uint8_t*, int, const uint8_t*, int,
                          const uint8_t*, int, uint8_t*, int,
                          uint8_t*, int, int, int) {
  abort();  // should never be called in --no-vipc mode
}
