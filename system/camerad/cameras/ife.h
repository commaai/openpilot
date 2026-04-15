#pragma once

#include <algorithm>

#include "cdm.h"

#include "system/camerad/cameras/hw.h"
#include "system/camerad/sensors/sensor.h"

namespace {

constexpr uint32_t IFE_ABF_RNR_RSQUARE_SHIFT = 10;

uint32_t build_ife_abf_cfg() {
  return (3U << 26) | (3U << 24) | (3U << 20) |
         (3U << 18) | (2U << 16) | (1U << 12) |
         (2U << 8) | (2U << 4) |
         (1U << 3) | (1U << 2) | 1U;
}

uint32_t build_ife_abf_curve_offset(const SensorInfo *s) {
  return std::min<uint32_t>(0x7fU, (s->black_level >> 1) + 4U);
}

std::vector<uint32_t> build_ife_abf_rnr_cfg(const SensorInfo *s) {
  const uint32_t bx = s->frame_width / 2;
  const uint32_t by = s->frame_height / 2;
  const uint32_t init_rsquare = bx * bx + by * by;
  const uint32_t scaled_rsquare = std::max<uint32_t>(1U, init_rsquare >> IFE_ABF_RNR_RSQUARE_SHIFT);
  const uint32_t anchor0 = std::max<uint32_t>(1U, scaled_rsquare / 4);
  const uint32_t anchor1 = std::max<uint32_t>(anchor0 + 1, scaled_rsquare / 2);
  const uint32_t anchor2 = std::max<uint32_t>(anchor1 + 1, (scaled_rsquare * 3) / 4);
  const uint32_t anchor3 = std::min<uint32_t>(0xfffU, std::max<uint32_t>(anchor2 + 1, scaled_rsquare));

  return {
    (by << 16) | bx,
    init_rsquare & 0x0fffffffU,
    (anchor1 << 16) | anchor0,
    (anchor3 << 16) | anchor2,
    (3U << 16) | (0U << 8) | 8U,
    (3U << 16) | (0U << 8) | 8U,
    (3U << 16) | (0U << 8) | 8U,
    8U,
    (2U << 16) | (0U << 8) | 1U,
    (2U << 16) | (0U << 8) | 1U,
    (2U << 16) | (0U << 8) | 1U,
    1U,
    IFE_ABF_RNR_RSQUARE_SHIFT,
  };
}

std::vector<uint32_t> build_ife_abf_bpc_cfg(const SensorInfo *s) {
  const uint32_t black_level = std::min<uint32_t>(0xfffU, s->black_level);
  const uint32_t offset = std::min<uint32_t>(0xfffU, black_level + 48U);

  return {
    (offset << 16) | (4U << 8) | 31U,
    (black_level << 8) | (5U << 4) | 2U,
  };
}

std::vector<uint32_t> build_ife_abf_noise_preserve_cfg(const SensorInfo *s) {
  const uint32_t anchor_lo = std::min<uint32_t>(0x3ffU, s->black_level + 16U);

  return {
    (32U << 16) | anchor_lo,
    (2U << 24) | (32U << 12) | 32U,
    (2U << 24) | (32U << 12) | 40U,
  };
}

int build_ife_abf(uint8_t *dst, const SensorInfo *s) {
  uint8_t *start = dst;
  const uint32_t curve_offset = build_ife_abf_curve_offset(s);

  dst += write_cont(dst, 0x5e8, {
    build_ife_abf_cfg(),
  });
  dst += write_cont(dst, 0x5f4, {
    curve_offset,
    curve_offset,
    curve_offset,
    curve_offset,
  });
  dst += write_cont(dst, 0x604, build_ife_abf_rnr_cfg(s));
  dst += write_cont(dst, 0x638, build_ife_abf_bpc_cfg(s));
  dst += write_cont(dst, 0x640, build_ife_abf_noise_preserve_cfg(s));

  return dst - start;
}

}  // namespace

int build_common_ife_bps(uint8_t *dst, const CameraConfig cam, const SensorInfo *s, std::vector<uint32_t> &patches, bool ife) {
  uint8_t *start = dst;

  /*
    Common between IFE and BPS.
  */

  // IFE -> BPS addresses
  /*
  std::map<uint32_t, uint32_t> addrs = {
    {0xf30, 0x3468},
  };
  */

  // YUV
  dst += write_cont(dst, ife ? 0xf30 : 0x3468, {
    0x00680208,
    0x00000108,
    0x00400000,
    0x03ff0000,
    0x01c01ed8,
    0x00001f68,
    0x02000000,
    0x03ff0000,
    0x1fb81e88,
    0x000001c0,
    0x02000000,
    0x03ff0000,
  });

  return dst - start;
}

int build_update(uint8_t *dst, const CameraConfig cam, const SensorInfo *s, std::vector<uint32_t> &patches) {
  uint8_t *start = dst;

  // init sequence
  dst += write_random(dst, {
    0x2c, 0xffffffff,
    0x30, 0xffffffff,
    0x34, 0xffffffff,
    0x38, 0xffffffff,
    0x3c, 0xffffffff,
  });

  // demux cfg
  dst += write_cont(dst, 0x560, {
    0x00000001,
    0x04440444,
    0x04450445,
    0x04440444,
    0x04450445,
    0x000000ca,
    0x0000009c,
  });

  // white balance
  dst += write_cont(dst, 0x6fc, {
    0x00800080,
    0x00000080,
    0x00000000,
    0x00000000,
  });

  // module config/enables (e.g. enable debayer, white balance, etc.)
  dst += write_cont(dst, 0x40, {
    0x00000c06 |
    ((uint32_t)(cam.vignetting_correction) << 8) |
    (1 << 7),
  });
  dst += write_cont(dst, 0x44, {
    0x00000000,
  });
  dst += write_cont(dst, 0x48, {
    (1 << 3) | (1 << 1),
  });
  dst += write_cont(dst, 0x4c, {
    0x00000019,
  });
  dst += write_cont(dst, 0xf00, {
    0x00000000,
  });

  // cropping
  dst += write_cont(dst, 0xe0c, {
    0x00000e00,
  });
  dst += write_cont(dst, 0xe2c, {
    0x00000e00,
  });

  // black level scale + offset
  dst += write_cont(dst, 0x6b0, {
    ((uint32_t)(1 << 11) << 0xf) | (s->black_level << (14 - s->bits_per_pixel)),
    0x0,
    0x0,
  });

  return dst - start;
}


int build_initial_config(uint8_t *dst, const CameraConfig cam, const SensorInfo *s, std::vector<uint32_t> &patches, uint32_t out_width, uint32_t out_height) {
  uint8_t *start = dst;

  // start with the every frame config
  dst += build_update(dst, cam, s, patches);

  uint64_t addr;

  dst += build_ife_abf(dst, s);

  // setup
  dst += write_cont(dst, 0x478, {
    0x00000004,
    0x004000c0,
  });
  dst += write_cont(dst, 0x488, {
    0x00000000,
    0x00000000,
    0x00000f0f,
  });
  dst += write_cont(dst, 0x49c, {
    0x00000001,
  });
  dst += write_cont(dst, 0xce4, {
    0x00000000,
    0x00000000,
  });

  // linearization
  dst += write_cont(dst, 0x4dc, {
    0x00000000,
  });
  dst += write_cont(dst, 0x4e0, s->linearization_pts);
  dst += write_cont(dst, 0x4f0, s->linearization_pts);
  dst += write_cont(dst, 0x500, s->linearization_pts);
  dst += write_cont(dst, 0x510, s->linearization_pts);
  // TODO: this is DMI64 in the dump, does that matter?
  dst += write_dmi(dst, &addr, s->linearization_lut.size()*sizeof(uint32_t), 0xc24, 9);
  patches.push_back(addr - (uint64_t)start);

  // vignetting correction
  dst += write_cont(dst, 0x6bc, {
    0x0b3c0000,
    0x00670067,
    0xd3b1300c,
    0x13b1300c,
  });
  dst += write_cont(dst, 0x6d8, {
    0xec4e4000,
    0x0100c003,
  });
  dst += write_dmi(dst, &addr, s->vignetting_lut.size()*sizeof(uint32_t), 0xc24, 14); // GRR
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, s->vignetting_lut.size()*sizeof(uint32_t), 0xc24, 15); // GBB
  patches.push_back(addr - (uint64_t)start);

  // debayer
  dst += write_cont(dst, 0x6f8, {
    0x00000100,
  });
  dst += write_cont(dst, 0x71c, {
    0x00008000,
    0x08000066,
  });

  // color correction
  dst += write_cont(dst, 0x760, s->color_correct_matrix);

  // gamma
  dst += write_cont(dst, 0x798, {
    0x00000000,
  });
  dst += write_dmi(dst, &addr, s->gamma_lut_rgb.size()*sizeof(uint32_t), 0xc24, 26);  // G
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, s->gamma_lut_rgb.size()*sizeof(uint32_t), 0xc24, 28);  // B
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, s->gamma_lut_rgb.size()*sizeof(uint32_t), 0xc24, 30);  // R
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, s->noise_std_lut.size()*sizeof(uint32_t), 0xc24, 12);
  patches.push_back(addr - (uint64_t)start);

  // output size/scaling
  dst += write_cont(dst, 0xa3c, {
    0x00000003,
    ((out_width - 1) << 16) | (s->frame_width - 1),
    0x30036666,
    0x00000000,
    0x00000000,
    s->frame_width - 1,
    ((out_height - 1) << 16) | (s->frame_height - 1),
    0x30036666,
    0x00000000,
    0x00000000,
    s->frame_height - 1,
  });
  dst += write_cont(dst, 0xa68, {
    0x00000003,
    ((out_width / 2 - 1) << 16) | (s->frame_width - 1),
    0x3006cccc,
    0x00000000,
    0x00000000,
    s->frame_width - 1,
    ((out_height / 2 - 1) << 16) | (s->frame_height - 1),
    0x3006cccc,
    0x00000000,
    0x00000000,
    s->frame_height - 1,
  });

  // cropping
  dst += write_cont(dst, 0xe10, {
    out_height - 1,
    out_width - 1,
  });
  dst += write_cont(dst, 0xe30, {
    out_height / 2 - 1,
    out_width - 1,
  });
  dst += write_cont(dst, 0xe18, {
    0x0ff00000,
    0x00000016,
  });
  dst += write_cont(dst, 0xe38, {
    0x0ff00000,
    0x00000017,
  });

  dst += build_common_ife_bps(dst, cam, s, patches, true);

  return dst - start;
}

