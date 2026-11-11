// zstd + bz2 decompression for the browser: one call per file, output grows until the stream ends
#include <stdint.h>
#include <stdlib.h>
#include "zstd.h"
#include "bzlib.h"

#define EXPORT __attribute__((used, visibility("default")))

static uint8_t *out;
static size_t out_len;

static int grow(size_t *cap) {
  *cap *= 2;
  uint8_t *p = realloc(out, *cap);
  if (!p) return 0;
  out = p;
  return 1;
}

EXPORT uint8_t *result(void) { return out; }
EXPORT size_t result_len(void) { return out_len; }
EXPORT void release(void) { free(out); out = NULL; out_len = 0; }

// returns 0 on success; the output stays valid until release()
EXPORT int zstd(const uint8_t *src, size_t n) {
  release();
  size_t cap = n * 4 + (1 << 16);
  out = malloc(cap);
  ZSTD_DStream *ds = ZSTD_createDStream();
  if (!out || !ds) return -1;
  ZSTD_inBuffer in = {src, n, 0};
  int err = 0;
  while (in.pos < in.size) {
    if (out_len == cap && !grow(&cap)) { err = -1; break; }
    ZSTD_outBuffer ob = {out, cap, out_len};
    size_t r = ZSTD_decompressStream(ds, &ob, &in);
    out_len = ob.pos;
    if (ZSTD_isError(r)) { err = -2; break; }
    if (r == 0 && in.pos == in.size) break;
    if (out_len < cap && in.pos == in.size) break;  // truncated stream: keep what decoded
  }
  ZSTD_freeDStream(ds);
  return err;
}

EXPORT int bz2(const uint8_t *src, size_t n) {
  release();
  size_t cap = n * 8 + (1 << 16);
  out = malloc(cap);
  bz_stream s = {0};
  if (!out || BZ2_bzDecompressInit(&s, 0, 0) != BZ_OK) return -1;
  s.next_in = (char *)src;
  s.avail_in = n;
  int err = 0;
  for (;;) {
    if (out_len == cap && !grow(&cap)) { err = -1; break; }
    s.next_out = (char *)out + out_len;
    s.avail_out = cap - out_len;
    int r = BZ2_bzDecompress(&s);
    out_len = cap - s.avail_out;
    if (r == BZ_STREAM_END) {
      // concatenated streams (pbzip2) start over
      if (s.avail_in == 0) break;
      BZ2_bzDecompressEnd(&s);
      if (BZ2_bzDecompressInit(&s, 0, 0) != BZ_OK) { err = -1; break; }
    } else if (r != BZ_OK) { err = -2; break; }
    else if (s.avail_in == 0 && s.avail_out) break;  // truncated
  }
  BZ2_bzDecompressEnd(&s);
  return err;
}

void bz_internal_error(int code) { (void)code; abort(); }
