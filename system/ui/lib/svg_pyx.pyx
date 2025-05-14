# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import pyray as rl
from libc.stdlib cimport malloc


cdef extern from "nanosvg/nanosvg.h":
  ctypedef struct NSVGimage:
    float width
    float height
  NSVGimage* nsvgParseFromFile(const char* filename, const char* units, float dpi)
  void nsvgDelete(NSVGimage* image)


cdef extern from "nanosvg/nanosvgrast.h":
  ctypedef struct NSVGrasterizer:
    pass
  NSVGrasterizer* nsvgCreateRasterizer()
  void nsvgRasterize(NSVGrasterizer* r,
                     NSVGimage* image, float tx, float ty, float scale,
				             unsigned char* dst, int w, int h, int stride)
  void nsvgDeleteRasterizer(NSVGrasterizer*)


def load_image_svg(image_path: str, width: int, height: int):
  cdef:
    NSVGimage* svg_img
    NSVGrasterizer* rast
    unsigned char* img_data
    int w = width
    int h = height
    float scale_w, scale_h, scale
    int offset_x = 0
    int offset_y = 0

  svg_img = nsvgParseFromFile(image_path.encode("utf-8"), b"px", 96.0)
  if not svg_img:
    raise RuntimeError(f"nsvgParse failed for {image_path!r}")

  if width == 0:
    w = <int> svg_img.width
  if height == 0:
    h = <int> svg_img.height

  scale_w = width / svg_img.width if width != 0 else 1.0
  scale_h = height / svg_img.height if height != 0 else 1.0
  scale = min(scale_w, scale_h)

  if scale_h > scale_w:
    offset_y = <int> ((height - svg_img.height * scale) / 2.0)
  else:
    offset_x = <int> ((width - svg_img.width * scale) / 2.0)

  rast = nsvgCreateRasterizer()
  img_data = <unsigned char *> malloc(w * h * 4)
  if not img_data:
    nsvgDelete(svg_img)
    nsvgDeleteRasterizer(rast)
    raise MemoryError("malloc() failed for SVG raster buffer")

  nsvgRasterize(rast,
                svg_img,
                offset_x, offset_y, scale,
                img_data, w, h, w * 4)

  nsvgDelete(svg_img)
  nsvgDeleteRasterizer(rast)

  return rl.Image(data=img_data, width=w, height=h, mipmaps=1, format=rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
