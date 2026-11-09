// Software rasterizer and KMS output for the comma four UI, see cpu_renderer.py
// Pixels are premultiplied RGBA8 (DRM ABGR8888), colors are passed in straight alpha.
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <drm/drm.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_mode.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) MIN(MAX(x, lo), hi)

static struct {
  uint8_t *pixels;
  int width, height, stride;
  int x0, y0, x1, y1;  // clip
} T;

void cr_target(uint8_t *pixels, int width, int height, int stride) {
  T.pixels = pixels;
  T.width = width;
  T.height = height;
  T.stride = stride;
  T.x0 = T.y0 = 0;
  T.x1 = width;
  T.y1 = height;
}

void cr_clip(int x0, int y0, int x1, int y1) {
  T.x0 = CLAMP(x0, 0, T.width);
  T.y0 = CLAMP(y0, 0, T.height);
  T.x1 = CLAMP(x1, T.x0, T.width);
  T.y1 = CLAMP(y1, T.y0, T.height);
}

static inline uint32_t *row(int y) { return (uint32_t *)(T.pixels + (size_t)y * T.stride); }

// scale all four 8-bit channels by a/255
static inline uint32_t mul(uint32_t c, uint32_t a) {
  uint32_t rb = (c & 0x00ff00ff) * a + 0x00800080;
  uint32_t ga = ((c >> 8) & 0x00ff00ff) * a + 0x00800080;
  rb = ((rb + ((rb >> 8) & 0x00ff00ff)) >> 8) & 0x00ff00ff;
  ga = (ga + ((ga >> 8) & 0x00ff00ff)) & 0xff00ff00;
  return rb | ga;
}

static inline uint32_t premultiply(uint32_t c) { return (mul(c | 0xff000000, c >> 24) & 0x00ffffff) | (c & 0xff000000); }
static inline uint32_t over(uint32_t dst, uint32_t src) { return src + mul(dst, 255 - (src >> 24)); }

static inline uint32_t mul4(uint32_t a, uint32_t b) {
  uint32_t r = 0;
  for (int s = 0; s < 32; s += 8) {
    uint32_t x = ((a >> s) & 255) * ((b >> s) & 255) + 128;
    r |= ((x + (x >> 8)) >> 8) << s;
  }
  return r;
}

static inline uint32_t lerp(uint32_t a, uint32_t b, uint32_t t) { return mul(a, 255 - t) + mul(b, t); }

static void span(uint32_t *p, int n, uint32_t c) {
  if ((c >> 24) == 255) {
    for (int i = 0; i < n; i++) p[i] = c;
  } else if (c) {
    for (int i = 0; i < n; i++) p[i] = over(p[i], c);
  }
}

// pixels whose centers are in [a, b)
static inline int px0(float a) { return (int)ceilf(a - 0.5f); }

void cr_clear(uint32_t color) {
  uint32_t c = premultiply(color);
  for (int y = 0; y < T.height; y++) {
    uint32_t *p = row(y);
    for (int x = 0; x < T.width; x++) p[x] = c;
  }
}

// axis aligned rectangle, corners are bilinearly interpolated
void cr_rect(float x0, float y0, float x1, float y1, uint32_t tl, uint32_t tr, uint32_t bl, uint32_t br) {
  int ix0 = MAX(px0(x0), T.x0), ix1 = MIN(px0(x1), T.x1);
  int iy0 = MAX(px0(y0), T.y0), iy1 = MIN(px0(y1), T.y1);
  if (ix0 >= ix1 || iy0 >= iy1) return;
  tl = premultiply(tl); tr = premultiply(tr); bl = premultiply(bl); br = premultiply(br);
  for (int y = iy0; y < iy1; y++) {
    uint32_t t = (uint32_t)CLAMP((y + 0.5f - y0) / (y1 - y0) * 255.0f + 0.5f, 0.0f, 255.0f);
    uint32_t l = lerp(tl, bl, t), r = lerp(tr, br, t);
    uint32_t *p = row(y);
    if (l == r) {
      span(p + ix0, ix1 - ix0, l);
      continue;
    }
    for (int x = ix0; x < ix1; x++) {
      uint32_t s = (uint32_t)CLAMP((x + 0.5f - x0) / (x1 - x0) * 255.0f + 0.5f, 0.0f, 255.0f);
      p[x] = over(p[x], lerp(l, r, s));
    }
  }
}

// signed distance to a rounded box with half size (hw, hh) and corner radius r
static inline float rbox(float dx, float dy, float hw, float hh, float r) {
  float qx = fabsf(dx) - hw + r, qy = fabsf(dy) - hh + r;
  float ox = MAX(qx, 0.0f), oy = MAX(qy, 0.0f);
  return sqrtf(ox * ox + oy * oy) + MIN(MAX(qx, qy), 0.0f) - r;
}

// half width of the row of a rounded box where its distance is <= k
static inline float rbox_extent(float dy, float hw, float hh, float r, float k) {
  float qy = fabsf(dy) - hh + r, m = k + r;
  if (qy > m) return -1.0f;
  return hw - r + (qy > 0 ? sqrtf(m * m - qy * qy) : m);
}

// filled rounded rectangle, or its outline when thickness > 0. antialiased
void cr_rounded_rect(float x, float y, float w, float h, float r, float thickness, uint32_t color) {
  float hw = w / 2, hh = h / 2, cx = x + hw, cy = y + hh;
  r = CLAMP(r, 0.0f, MIN(hw, hh));
  uint32_t c = premultiply(color);
  int iy0 = MAX((int)floorf(y), T.y0), iy1 = MIN((int)ceilf(y + h), T.y1);
  for (int iy = iy0; iy < iy1; iy++) {
    float dy = iy + 0.5f - cy;
    float outer = rbox_extent(dy, hw, hh, r, 0.5f);
    if (outer < 0) continue;
    float solid = rbox_extent(dy, hw, hh, r, -0.5f);
    float inner = thickness > 0 ? rbox_extent(dy, hw, hh, r, 0.5f - thickness) : -1.0f;
    float hole = thickness > 0 ? rbox_extent(dy, hw, hh, r, -thickness - 0.5f) : -1.0f;
    // [s0, s1) is fully covered except for [i0, i1), of which [h0, h1) is empty. the rest is antialiased
    int x0 = MAX((int)floorf(cx - outer), T.x0), x1 = MIN((int)ceilf(cx + outer), T.x1);
    int s0 = solid < 0 ? x1 : CLAMP(px0(cx - solid), x0, x1), s1 = solid < 0 ? x1 : CLAMP(px0(cx + solid), s0, x1);
    int i0 = inner < 0 ? s1 : CLAMP(px0(cx - inner), s0, s1), i1 = inner < 0 ? s1 : CLAMP(px0(cx + inner), i0, s1);
    int h0 = hole < 0 ? i0 : CLAMP(px0(cx - hole), i0, i1), h1 = hole < 0 ? i0 : CLAMP(px0(cx + hole), h0, i1);
    uint32_t *p = row(iy);
    span(p + s0, i0 - s0, c);
    span(p + i1, s1 - i1, c);
    const int edges[4][2] = {{x0, s0}, {i0, h0}, {h1, i1}, {s1, x1}};
    for (int e = 0; e < 4; e++) {
      for (int ix = edges[e][0]; ix < edges[e][1]; ix++) {
        float d = rbox(ix + 0.5f - cx, dy, hw, hh, r);
        float a = CLAMP(0.5f - d, 0.0f, 1.0f);
        if (thickness > 0) a *= CLAMP(d + thickness + 0.5f, 0.0f, 1.0f);
        if (a > 0) p[ix] = over(p[ix], mul(c, (uint32_t)(a * 255.0f + 0.5f)));
      }
    }
  }
}

// ring between two radii from angle a0 to a1 (degrees, clockwise from +x), colored from inner to outer radius
void cr_ring(float cx, float cy, float r0, float r1, float a0, float a1, uint32_t inner, uint32_t outer) {
  inner = premultiply(inner);
  outer = premultiply(outer);
  int full = a1 - a0 >= 360.0f;
  float start = fmodf(a0, 360.0f);
  if (start < 0) start += 360.0f;
  int iy0 = MAX((int)floorf(cy - r1), T.y0), iy1 = MIN((int)ceilf(cy + r1), T.y1);
  int ix0 = MAX((int)floorf(cx - r1), T.x0), ix1 = MIN((int)ceilf(cx + r1), T.x1);
  for (int iy = iy0; iy < iy1; iy++) {
    uint32_t *p = row(iy);
    float dy = iy + 0.5f - cy;
    for (int ix = ix0; ix < ix1; ix++) {
      float dx = ix + 0.5f - cx, d = sqrtf(dx * dx + dy * dy);
      float a = CLAMP(r1 - d + 0.5f, 0.0f, 1.0f) * (r0 > 0 ? CLAMP(d - r0 + 0.5f, 0.0f, 1.0f) : 1.0f);
      if (a <= 0) continue;
      if (!full) {
        float angle = fmodf(atan2f(dy, dx) * 57.29578f - start + 720.0f, 360.0f);
        if (angle > a1 - a0) continue;
      }
      uint32_t c = inner == outer ? inner : lerp(inner, outer, (uint32_t)CLAMP((d - r0) / (r1 - r0) * 255.0f, 0.0f, 255.0f));
      p[ix] = over(p[ix], mul(c, (uint32_t)(a * 255.0f + 0.5f)));
    }
  }
}

static void triangle(const float *a, const float *b, const float *c, uint32_t ca, uint32_t cb, uint32_t cc) {
  float area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  if (fabsf(area) < 1e-6f) return;
  if (area < 0) {  // make counterclockwise in screen space
    const float *t = b; b = c; c = t;
    uint32_t tc = cb; cb = cc; cc = tc;
    area = -area;
  }
  // barycentric weight of vertex k is (wa[k] * x + wy[k] * y + wc[k]) / area, inside where all are >= 0
  const float *v[3] = {a, b, c};
  float wa[3], wy[3], wc[3];
  for (int k = 0; k < 3; k++) {
    const float *p = v[(k + 1) % 3], *q = v[(k + 2) % 3];
    wa[k] = p[1] - q[1];
    wy[k] = q[0] - p[0];
    wc[k] = -wy[k] * p[1] - wa[k] * p[0];
  }
  // so each channel is cx[s] * x + cy[s] * y + cz[s]
  uint32_t col[3] = {ca, cb, cc};
  float cx[4] = {0}, cy[4] = {0}, cz[4] = {0.5f, 0.5f, 0.5f, 0.5f};  // + 0.5 rounds
  for (int s = 0; s < 4; s++) {
    for (int k = 0; k < 3; k++) {
      float cv = ((col[k] >> (8 * s)) & 255) / area;
      cx[s] += wa[k] * cv;
      cy[s] += wy[k] * cv;
      cz[s] += wc[k] * cv;
    }
  }
  int flat = ca == cb && cb == cc;
  int iy0 = MAX(px0(MIN(MIN(a[1], b[1]), c[1])), T.y0), iy1 = MIN(px0(MAX(MAX(a[1], b[1]), c[1])) + 1, T.y1);
  for (int iy = iy0; iy < iy1; iy++) {
    float py = iy + 0.5f, lo = T.x0, hi = T.x1;
    for (int k = 0; k < 3; k++) {
      float wb = wy[k] * py + wc[k];
      if (wa[k] > 0) lo = MAX(lo, -wb / wa[k]);
      else if (wa[k] < 0) hi = MIN(hi, -wb / wa[k]);
      else if (wb < 0 || (wb == 0 && wy[k] < 0)) hi = lo;  // centers on a bottom edge belong to the triangle below
    }
    int x0 = MAX(px0(lo), T.x0), x1 = MIN(px0(hi), T.x1);
    if (x0 >= x1) continue;
    uint32_t *p = row(iy);
    if (flat) {
      span(p + x0, x1 - x0, ca);
      continue;
    }
    float ch[4];
    for (int s = 0; s < 4; s++) ch[s] = cx[s] * (x0 + 0.5f) + cy[s] * py + cz[s];
    for (int x = x0; x < x1; x++) {
      float k = x - x0;
      uint32_t px = 0;
      for (int s = 0; s < 4; s++) px |= (uint32_t)CLAMP(ch[s] + cx[s] * k, 0.0f, 255.0f) << (8 * s);
      p[x] = over(p[x], px);
    }
  }
}

// triangles from n points: mode 0 = list, 1 = strip, 2 = fan. one color per point, or a single color
void cr_triangles(const float *xy, int n, const uint32_t *colors, int ncolors, int mode) {
  if (n < 3 || ncolors < 1) return;
  uint32_t c[n];
  for (int i = 0; i < n; i++) c[i] = premultiply(colors[ncolors == n ? i : 0]);
  for (int i = 2; i < n; i += mode == 0 ? 3 : 1) {
    int i0 = mode == 2 ? 0 : i - 2, i1 = i - 1;
    triangle(&xy[2 * i0], &xy[2 * i1], &xy[2 * i], c[i0], c[i1], c[i]);
  }
}

// texel at (x, y), or the bilinear mix of (xa, ya) to (xb, yb) with weights wx, wy out of 256, times the tint
static inline uint32_t texel(const uint8_t *src, int sstride, int a8, int xa, int xb, int ya, int yb, uint32_t wx, uint32_t wy,
                             uint32_t tint) {
  const uint8_t *ra = src + (size_t)ya * sstride, *rb = src + (size_t)yb * sstride;
  if (a8) {
    uint32_t a = ra[xa];
    if (wx | wy) {
      uint32_t top = ra[xa] * (256 - wx) + ra[xb] * wx, bot = rb[xa] * (256 - wx) + rb[xb] * wx;
      a = (top * (256 - wy) + bot * wy) >> 16;
    }
    return mul(tint, a);
  }
  uint32_t c = *(const uint32_t *)(ra + xa * 4);
  if (wx | wy) {
    uint32_t ta = lerp(c, *(const uint32_t *)(ra + xb * 4), wx * 255 >> 8);
    uint32_t tb = lerp(*(const uint32_t *)(rb + xa * 4), *(const uint32_t *)(rb + xb * 4), wx * 255 >> 8);
    c = lerp(ta, tb, wy * 255 >> 8);
  }
  return tint == 0xffffffff ? c : mul4(c, tint);
}

// texel coordinate u in [lo, hi) of a texture n wide to the two texels and weight sampling it
static inline int taps(float u, float lo, float hi, int n, int bilinear, int *a, int *b, uint32_t *w) {
  if (u < lo || u >= hi) return 0;
  if (bilinear) {
    float f = u - 0.5f;
    int i = (int)floorf(f);
    *w = (uint32_t)((f - i) * 256.0f);
    *a = CLAMP(i, 0, n - 1);
    *b = CLAMP(i + 1, 0, n - 1);
  } else {
    *a = *b = CLAMP((int)u, 0, n - 1);
    *w = 0;
  }
  return 1;
}

// raylib's DrawTexturePro: source rect of src drawn into dest rect rotated around origin.
// src is premultiplied RGBA, or an alpha mask when a8. negative source size flips
void cr_blit(const uint8_t *src, int sw, int sh, int sstride, int a8, float sx, float sy, float src_w, float src_h,
             float dx, float dy, float dw, float dh, float ox, float oy, float rotation, uint32_t tint, int bilinear) {
  if (dw <= 0 || dh <= 0 || src_w == 0 || src_h == 0 || sw <= 0 || sh <= 0) return;
  uint32_t t = premultiply(tint);
  float cs = cosf(rotation * 0.01745329f), sn = sinf(rotation * 0.01745329f);
  // dest corners
  float xs[4], ys[4];
  for (int i = 0; i < 4; i++) {
    float lx = (i & 1 ? dw : 0) - ox, ly = (i & 2 ? dh : 0) - oy;
    xs[i] = dx + lx * cs - ly * sn;
    ys[i] = dy + lx * sn + ly * cs;
  }
  int ix0 = MAX(px0(MIN(MIN(xs[0], xs[1]), MIN(xs[2], xs[3]))), T.x0);
  int ix1 = MIN(px0(MAX(MAX(xs[0], xs[1]), MAX(xs[2], xs[3]))), T.x1);
  int iy0 = MAX(px0(MIN(MIN(ys[0], ys[1]), MIN(ys[2], ys[3]))), T.y0);
  int iy1 = MIN(px0(MAX(MAX(ys[0], ys[1]), MAX(ys[2], ys[3]))), T.y1);
  if (ix0 >= ix1 || iy0 >= iy1) return;
  // screen -> source texel: u = bu + x*u0x + y*u0y
  float su = src_w / dw, sv = src_h / dh;
  float u0x = cs * su, u0y = sn * su, v0x = -sn * sv, v0y = cs * sv;
  float bu = (src_w < 0 ? sx - src_w : sx) + ox * su, bv = (src_h < 0 ? sy - src_h : sy) + oy * sv;
  float hu = sx + fabsf(src_w), hv = sy + fabsf(src_h);
  int xa, xb, ya, yb;
  uint32_t wx, wy;
  if (rotation == 0) {  // texel columns and rows are separable, and u increases along the row
    int n = 0, cxa[ix1 - ix0], cxb[ix1 - ix0];
    uint32_t cwx[ix1 - ix0], buf[ix1 - ix0];
    for (int ix = ix0; ix < ix1; ix++) {
      if (taps(bu + (ix + 0.5f - dx) * u0x, sx, hu, sw, bilinear, &cxa[n], &cxb[n], &cwx[n])) {
        if (n == 0) ix0 = ix;
        n++;
      } else if (n) {
        break;
      }
    }
    int unscaled = !bilinear;
    for (int i = 1; i < n && unscaled; i++) unscaled = cxa[i] == cxa[0] + i;
    for (int iy = iy0; iy < iy1 && n; iy++) {
      if (!taps(bv + (iy + 0.5f - dy) * v0y, sy, hv, sh, bilinear, &ya, &yb, &wy)) continue;
      const uint8_t *s = src + (size_t)ya * sstride;
      uint32_t *p = row(iy) + ix0;
      if (unscaled && a8) {
        for (int i = 0; i < n; i++) p[i] = over(p[i], mul(t, s[cxa[0] + i]));
      } else if (unscaled && t == 0xffffffff) {
        for (int i = 0; i < n; i++) p[i] = over(p[i], ((const uint32_t *)s)[cxa[0] + i]);
      } else {
        for (int i = 0; i < n; i++) buf[i] = texel(src, sstride, a8, cxa[i], cxb[i], ya, yb, cwx[i], wy, t);
        for (int i = 0; i < n; i++) p[i] = over(p[i], buf[i]);
      }
    }
    return;
  }
  for (int iy = iy0; iy < iy1; iy++) {
    uint32_t *p = row(iy);
    float py = iy + 0.5f - dy;
    for (int ix = ix0; ix < ix1; ix++) {
      float px = ix + 0.5f - dx;
      if (taps(bu + px * u0x + py * u0y, sx, hu, sw, bilinear, &xa, &xb, &wx) &&
          taps(bv + px * v0x + py * v0y, sy, hv, sh, bilinear, &ya, &yb, &wy)) {
        p[ix] = over(p[ix], texel(src, sstride, a8, xa, xb, ya, yb, wx, wy, t));
      }
    }
  }
}

// NV12 frame drawn opaque into dest rect, sampled at nearest and converted like the GPU's NV12 import (limited range BT.601),
// then mixed with its luma by saturation/256 and each channel mapped through curve, as the frame shaders do. like on the GPU,
// channels can be outside [0, 255] until the curve, which has 1280 entries for -512 to 767
// the result is reused until new_frame or the parameters change
static struct {
  uint32_t *px;
  size_t cap;
  float params[4];
  const void *ptrs[2];
  int ints[8];
} NV;

// frame row pixels lo to hi, lo even. a chroma pair is shared by two luma pixels.
// mixing with the luma is scaling the chroma, as the conversion keeps the luma. with saturation <= 256, channels are -277 to 536
static void nv12_row(uint32_t *restrict p, const uint8_t *restrict yr, const uint8_t *restrict uvr, int lo, int hi,
                     const uint8_t *restrict curve, int saturation) {
  for (int x = lo; x <= hi; x += 2) {
    int u = (uvr[x] - 128) * saturation, v = (uvr[x + 1] - 128) * saturation;
    int dr = (409 * v) >> 16, dg = (100 * u + 208 * v) >> 16, db = (516 * u) >> 16;
    for (int k = 0; k < 2; k++) {
      int l = ((yr[x + k] - 16) * 298) >> 8;
      const uint8_t *c = curve + 512 + l;
      p[x + k] = 0xff000000 | (uint32_t)c[db] << 16 | (uint32_t)c[-dg] << 8 | c[dr];
    }
  }
}

void cr_nv12(const uint8_t *y_plane, const uint8_t *uv_plane, int stride, int fw, int fh, float dx, float dy, float dw, float dh,
             int flip_x, const uint8_t *curve, int saturation, int new_frame) {
  int ix0 = MAX(px0(dx), T.x0), ix1 = MIN(px0(dx + dw), T.x1);
  int iy0 = MAX(px0(dy), T.y0), iy1 = MIN(px0(dy + dh), T.y1);
  if (ix0 >= ix1 || iy0 >= iy1) return;
  int n = ix1 - ix0;
  float params[4] = {dx, dy, dw, dh};
  const void *ptrs[2] = {y_plane, curve};
  saturation = CLAMP(saturation, 0, 256);
  int ints[8] = {ix0, ix1, iy0, iy1, fw, fh, flip_x, saturation};
  int convert = new_frame || memcmp(params, NV.params, sizeof(params)) || memcmp(ptrs, NV.ptrs, sizeof(ptrs)) ||
                memcmp(ints, NV.ints, sizeof(ints));
  if (convert) {
    size_t size = (size_t)n * (iy1 - iy0);
    if (size > NV.cap) {
      free(NV.px);
      NV.px = malloc(size * 4);
      NV.cap = NV.px ? size : 0;
      if (!NV.px) return;
    }
    memcpy(NV.params, params, sizeof(params));
    memcpy(NV.ptrs, ptrs, sizeof(ptrs));
    memcpy(NV.ints, ints, sizeof(ints));
  }
  int cols[n];
  for (int i = 0; i < n && convert; i++) {
    int fx = CLAMP((int)((ix0 + i + 0.5f - dx) * fw / dw), 0, fw - 1);
    cols[i] = flip_x ? fw - 1 - fx : fx;
  }
  int lo = convert ? MIN(cols[0], cols[n - 1]) & ~1 : 0, hi = convert ? MAX(cols[0], cols[n - 1]) : 0;  // sampled frame columns
  uint32_t src[hi + 2];  // in whole chroma pairs
  for (int iy = iy0, prev = -1; iy < iy1; iy++) {
    uint32_t *p = NV.px + (size_t)(iy - iy0) * n;
    if (convert) {
      int fy = CLAMP((int)((iy + 0.5f - dy) * fh / dh), 0, fh - 1);
      if (fy == prev) {  // upscaled, same frame row as the last one
        memcpy(p, p - n, (size_t)n * 4);
      } else {
        // converted at the frame's resolution, then scaled
        nv12_row(src, y_plane + (size_t)fy * stride, uv_plane + (size_t)(fy / 2) * stride, lo, hi, curve, saturation);
        for (int i = 0; i < n; i++) p[i] = src[cols[i]];
      }
      prev = fy;
    }
    memcpy(row(iy) + ix0, p, (size_t)n * 4);
  }
}

// ***** KMS output *****
// the display's rotator turns the landscape target into one of two portrait panel buffers, which is then flipped to

#if __has_include(<linux/msm_ion.h>)
#include <fcntl.h>
#include <linux/ion.h>
#include <linux/msm_ion.h>
#include <linux/videodev2.h>

struct ion_buf {
  int fd;
  ion_user_handle_t handle;
  uint8_t *map;
  size_t size;
};

static struct {
  int fd, own, ion, rot;  // the master fd from magic, and our own for the buffers so they go away with the process
  uint32_t crtc, connector;
  struct drm_mode_modeinfo mode;
  struct ion_buf target, panel[2];
  uint32_t fb[2];
  int back, crtc_set;
} D = {.fd = -1, .own = -1, .ion = -1, .rot = -1};

static int ion_alloc(struct ion_buf *b, size_t size, int cached) {
  struct ion_allocation_data alloc = {.len = size, .align = 4096, .heap_id_mask = 1 << ION_IOMMU_HEAP_ID,
                                      .flags = cached ? ION_FLAG_CACHED : 0};
  if (ioctl(D.ion, ION_IOC_ALLOC, &alloc)) return -errno;
  b->handle = alloc.handle;
  struct ion_fd_data share = {.handle = alloc.handle};
  if (ioctl(D.ion, ION_IOC_SHARE, &share)) return -errno;
  b->fd = share.fd;
  b->size = size;  // set with the fd
  b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
  return b->map == MAP_FAILED ? (b->map = NULL, -errno) : 0;
}

static void ion_free(struct ion_buf *b) {
  if (b->map) munmap(b->map, b->size);
  if (b->size) close(b->fd);
  if (b->handle) ioctl(D.ion, ION_IOC_FREE, &(struct ion_handle_data){.handle = b->handle});
  memset(b, 0, sizeof(*b));
}

static int rot_queue(int type, int index, const struct ion_buf *b) {
  // the rotator takes the dmabuf fd as the user pointer
  struct v4l2_buffer buf = {.type = type, .memory = V4L2_MEMORY_USERPTR, .index = index, .m.userptr = b->fd,
                            .length = b->size, .bytesused = b->size};
  return ioctl(D.rot, VIDIOC_QBUF, &buf) ? -errno : 0;
}

static int rot_wait(int type) {
  struct v4l2_buffer buf = {.type = type, .memory = V4L2_MEMORY_USERPTR};
  if (ioctl(D.rot, VIDIOC_DQBUF, &buf)) return -errno;
  return buf.flags & V4L2_BUF_FLAG_ERROR ? -EIO : 0;
}

void cr_display_close(void);

// open the panel and the rotator into it
int cr_display_init(int fd, int canonical_zero) {
  uint32_t crtcs[8] = {0}, connectors[8] = {0};
  struct drm_mode_card_res res = {.crtc_id_ptr = (uintptr_t)crtcs, .connector_id_ptr = (uintptr_t)connectors,
                                  .count_crtcs = 8, .count_connectors = 8};
  if (ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &res) || !res.count_crtcs) return -errno;
  for (uint32_t i = 0; i < MIN(res.count_connectors, 8); i++) {
    struct drm_mode_modeinfo modes[16];
    struct drm_mode_get_connector conn = {.connector_id = connectors[i], .modes_ptr = (uintptr_t)modes, .count_modes = 16};
    if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &conn) == 0 && conn.connection == 1 && conn.count_modes > 0) {
      D.connector = connectors[i];
      D.mode = modes[0];
      break;
    }
  }
  if (!D.connector) return -ENODEV;
  D.fd = fd;
  D.crtc = crtcs[0];
  D.own = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  D.ion = open("/dev/ion", O_RDWR);
  D.rot = open("/dev/video2", O_RDWR);  // sde_rotator, a v4l2 mem2mem device
  int pw = D.mode.hdisplay, ph = D.mode.vdisplay;
  struct v4l2_control rotate = {.id = V4L2_CID_ROTATE, .value = canonical_zero ? 270 : 90};
  // the v4l2 and DRM fourccs are the same
  struct v4l2_format in = {.type = V4L2_BUF_TYPE_VIDEO_OUTPUT, .fmt.pix = {.width = ph, .height = pw, .pixelformat = DRM_FORMAT_ABGR8888}};
  struct v4l2_format out = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .fmt.pix = {.width = pw, .height = ph, .pixelformat = DRM_FORMAT_ABGR8888}};
  struct v4l2_requestbuffers reqs[2] = {{.count = 1, .type = V4L2_BUF_TYPE_VIDEO_OUTPUT, .memory = V4L2_MEMORY_USERPTR},
                                        {.count = 2, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_USERPTR}};
  struct drm_auth auth;
  int ret = D.own < 0 || ioctl(D.own, DRM_IOCTL_GET_MAGIC, &auth) || ioctl(fd, DRM_IOCTL_AUTH_MAGIC, &auth) ||
            D.ion < 0 || D.rot < 0 || ioctl(D.rot, VIDIOC_S_CTRL, &rotate) || ioctl(D.rot, VIDIOC_S_FMT, &in) ||
            ioctl(D.rot, VIDIOC_S_FMT, &out) ? -errno : 0;
  if (!ret && in.fmt.pix.bytesperline != (uint32_t)ph * 4) ret = -EINVAL;
  // the target is read back while drawing, so it's cached and cleaned before each rotation
  if (!ret) ret = ion_alloc(&D.target, in.fmt.pix.sizeimage, 1);
  for (int i = 0; i < 2 && !ret; i++) {
    ret = ion_alloc(&D.panel[i], out.fmt.pix.sizeimage, 0);
    struct drm_prime_handle prime = {.fd = D.panel[i].fd};
    if (!ret && ioctl(D.own, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime)) ret = -errno;
    struct drm_mode_fb_cmd2 fb = {.width = pw, .height = ph, .pixel_format = DRM_FORMAT_ABGR8888, .handles = {prime.handle},
                                  .pitches = {out.fmt.pix.bytesperline}};
    if (!ret && ioctl(D.own, DRM_IOCTL_MODE_ADDFB2, &fb)) ret = -errno;
    D.fb[i] = fb.fb_id;
  }
  for (int i = 0; i < 2 && !ret; i++) {
    if (ioctl(D.rot, VIDIOC_REQBUFS, &reqs[i]) || ioctl(D.rot, VIDIOC_STREAMON, &reqs[i].type)) ret = -errno;
  }
  if (ret) cr_display_close();
  return ret;
}

// the buffer to draw into, if a width x height target is the panel's size
uint8_t *cr_display_target(int width, int height) {
  return width == D.mode.vdisplay && height == D.mode.hdisplay ? D.target.map : NULL;
}

// set a property of the display's CRTC to a blob of size bytes, or clear it when data is NULL
int cr_display_blob(const char *name, const void *data, uint32_t size) {
  uint32_t ids[128];
  uint64_t values[128];
  struct drm_mode_obj_get_properties props = {.props_ptr = (uintptr_t)ids, .prop_values_ptr = (uintptr_t)values, .count_props = 128,
                                              .obj_id = D.crtc, .obj_type = DRM_MODE_OBJECT_CRTC};
  if (ioctl(D.fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &props)) return -errno;
  for (uint32_t i = 0; i < MIN(props.count_props, 128); i++) {
    struct drm_mode_get_property prop = {.prop_id = ids[i]};
    if (ioctl(D.fd, DRM_IOCTL_MODE_GETPROPERTY, &prop) || strcmp(prop.name, name)) continue;
    struct drm_mode_create_blob blob = {.data = (uintptr_t)data, .length = size};
    if (data && ioctl(D.fd, DRM_IOCTL_MODE_CREATEPROPBLOB, &blob)) return -errno;
    struct drm_mode_obj_set_property set = {.value = data ? blob.blob_id : 0, .prop_id = ids[i], .obj_id = D.crtc,
                                            .obj_type = DRM_MODE_OBJECT_CRTC};
    int ret = ioctl(D.fd, DRM_IOCTL_MODE_OBJ_SETPROPERTY, &set) ? -errno : 0;
    if (data) ioctl(D.fd, DRM_IOCTL_MODE_DESTROYPROPBLOB, &(struct drm_mode_destroy_blob){.blob_id = blob.blob_id});
    return ret;
  }
  return -ENOENT;
}

// show the target on the panel at the next vblank
int cr_display_present(void) {
  if (T.pixels != D.target.map) return -EINVAL;
  struct ion_flush_data flush = {.handle = D.target.handle, .vaddr = D.target.map, .length = D.target.size};
  struct ion_custom_data clean = {.cmd = ION_IOC_CLEAN_CACHES, .arg = (uintptr_t)&flush};
  if (ioctl(D.ion, ION_IOC_CUSTOM, &clean)) return -errno;
  int ret = rot_queue(V4L2_BUF_TYPE_VIDEO_OUTPUT, 0, &D.target);
  if (!ret) ret = rot_queue(V4L2_BUF_TYPE_VIDEO_CAPTURE, D.back, &D.panel[D.back]);
  if (!ret) ret = rot_wait(V4L2_BUF_TYPE_VIDEO_CAPTURE);
  if (!ret) ret = rot_wait(V4L2_BUF_TYPE_VIDEO_OUTPUT);
  if (ret) {  // restarting the queues returns any buffers still in them
    for (int type = V4L2_BUF_TYPE_VIDEO_CAPTURE; type <= V4L2_BUF_TYPE_VIDEO_OUTPUT; type++) {
      ioctl(D.rot, VIDIOC_STREAMOFF, &type);
      ioctl(D.rot, VIDIOC_STREAMON, &type);
    }
    return ret;
  }
  // like raylib: flip, or set the CRTC again if another DRM client disabled it, then wait for the vblank
  uint32_t fb = D.fb[D.back];
  struct drm_mode_crtc_page_flip flip = {.crtc_id = D.crtc, .fb_id = fb};
  if (!D.crtc_set || ioctl(D.fd, DRM_IOCTL_MODE_PAGE_FLIP, &flip)) {
    struct drm_mode_crtc set = {.crtc_id = D.crtc, .fb_id = fb, .set_connectors_ptr = (uintptr_t)&D.connector,
                                .count_connectors = 1, .mode = D.mode, .mode_valid = 1};
    D.crtc_set = ioctl(D.fd, DRM_IOCTL_MODE_SETCRTC, &set) == 0;
    if (!D.crtc_set) return -errno;
  }
  union drm_wait_vblank vblank = {.request = {.type = _DRM_VBLANK_RELATIVE, .sequence = 1}};
  ioctl(D.fd, DRM_IOCTL_WAIT_VBLANK, &vblank);
  D.back ^= 1;
  return 0;
}

void cr_display_close(void) {
  if (D.rot >= 0) close(D.rot);
  if (D.own >= 0) close(D.own);  // and with it the framebuffers
  for (int i = 0; i < 2; i++) ion_free(&D.panel[i]);
  ion_free(&D.target);
  if (D.ion >= 0) close(D.ion);
  memset(&D, 0, sizeof(D));
  D.fd = D.own = D.ion = D.rot = -1;
}
#else
int cr_display_init(int fd, int canonical_zero) { return -ENODEV; }
uint8_t *cr_display_target(int width, int height) { return NULL; }
int cr_display_blob(const char *name, const void *data, uint32_t size) { return -ENODEV; }
int cr_display_present(void) { return -ENODEV; }
void cr_display_close(void) {}
#endif
