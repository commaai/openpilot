#include "system/loggerd/encoder/jpeg_encoder.h"

#include <cassert>
#include <cstring>

JpegEncoder::JpegEncoder(const std::string &pusblish_name, int width, int height)
    : publish_name(pusblish_name), thumbnail_width(width), thumbnail_height(height) {
  yuv_buffer.resize((thumbnail_width * ((thumbnail_height + 15) & ~15) * 3) / 2);
  pm = std::make_unique<PubMaster>(std::vector{pusblish_name.c_str()});
}

JpegEncoder::~JpegEncoder() {
  if (out_buffer) {
    free(out_buffer);
  }
}

void JpegEncoder::pushThumbnail(VisionBuf *buf, const VisionIpcBufExtra &extra) {
  generateThumbnail(buf->y, buf->uv, buf->width, buf->height, buf->stride);

  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initThumbnail();
  thumbnaild.setFrameId(extra.frame_id);
  thumbnaild.setTimestampEof(extra.timestamp_eof);
  thumbnaild.setThumbnail({out_buffer, out_size});

  pm->send(publish_name.c_str(), msg);
}

void JpegEncoder::generateThumbnail(const uint8_t *y_addr, const uint8_t *uv_addr, int width, int height, int stride) {
  int downscale = width / thumbnail_width;
  assert(downscale * thumbnail_height == height);

  // make the buffer big enough. jpeg_write_raw_data requires 16-pixels aligned height to be used.
  uint8_t *y_plane = yuv_buffer.data();
  uint8_t *u_plane = y_plane + thumbnail_width * thumbnail_height;
  uint8_t *v_plane = u_plane + (thumbnail_width * thumbnail_height) / 4;
  {
    // subsampled conversion from nv12 to yuv
    for (int hy = 0; hy < thumbnail_height / 2; hy++) {
      for (int hx = 0; hx < thumbnail_width / 2; hx++) {
        int ix = hx * downscale + (downscale - 1) / 2;
        int iy = hy * downscale + (downscale - 1) / 2;
        y_plane[(hy * 2 + 0) * thumbnail_width + (hx * 2 + 0)] = y_addr[(iy * 2 + 0) * stride + ix * 2 + 0];
        y_plane[(hy * 2 + 0) * thumbnail_width + (hx * 2 + 1)] = y_addr[(iy * 2 + 0) * stride + ix * 2 + 1];
        y_plane[(hy * 2 + 1) * thumbnail_width + (hx * 2 + 0)] = y_addr[(iy * 2 + 1) * stride + ix * 2 + 0];
        y_plane[(hy * 2 + 1) * thumbnail_width + (hx * 2 + 1)] = y_addr[(iy * 2 + 1) * stride + ix * 2 + 1];
        u_plane[hy * thumbnail_width / 2 + hx] = uv_addr[iy * stride + ix * 2 + 0];
        v_plane[hy * thumbnail_width / 2 + hx] = uv_addr[iy * stride + ix * 2 + 1];
      }
    }
  }

  compressToJpeg(y_plane, u_plane, v_plane);
}

void JpegEncoder::compressToJpeg(uint8_t *y_plane, uint8_t *u_plane, uint8_t *v_plane) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  if (out_buffer) {
    free(out_buffer);
    out_buffer = nullptr;
    out_size = 0;
  }
  jpeg_mem_dest(&cinfo, &out_buffer, &out_size);

  cinfo.image_width = thumbnail_width;
  cinfo.image_height = thumbnail_height;
  cinfo.input_components = 3;

  jpeg_set_defaults(&cinfo);
  jpeg_set_colorspace(&cinfo, JCS_YCbCr);
  // configure sampling factors for yuv420.
  cinfo.comp_info[0].h_samp_factor = 2;  // Y
  cinfo.comp_info[0].v_samp_factor = 2;
  cinfo.comp_info[1].h_samp_factor = 1;  // U
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;  // V
  cinfo.comp_info[2].v_samp_factor = 1;
  cinfo.raw_data_in = TRUE;

  jpeg_set_quality(&cinfo, 50, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW y[16], u[8], v[8];
  JSAMPARRAY planes[3]{y, u, v};

  for (int line = 0; line < cinfo.image_height; line += 16) {
    for (int i = 0; i < 16; ++i) {
      y[i] = y_plane + (line + i) * cinfo.image_width;
      if (i % 2 == 0) {
        int offset = (cinfo.image_width / 2) * ((i + line) / 2);
        u[i / 2] = u_plane + offset;
        v[i / 2] = v_plane + offset;
      }
    }
    jpeg_write_raw_data(&cinfo, planes, 16);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}
