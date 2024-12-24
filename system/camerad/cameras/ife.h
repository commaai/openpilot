#include "cdm.h"

#include "system/camerad/cameras/tici.h"
#include "system/camerad/sensors/sensor.h"


int build_update(uint8_t *dst, const SensorInfo *s, std::vector<uint32_t> &patches) {
  uint8_t *start = dst;

  dst += write_random(dst, {
    0x2c, 0xffffffff,
    0x30, 0xffffffff,
    0x34, 0xffffffff,
    0x38, 0xffffffff,
    0x3c, 0xffffffff,
  });

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

  dst += write_cont(dst, 0x40, {
    0x00000c06, // (1<<8) to enable vignetting correction
  });

  dst += write_cont(dst, 0x48, {
    (1 << 3) | (1 << 1),
  });

  dst += write_cont(dst, 0x4c, {
    0x00000019,
  });

  dst += write_cont(dst, 0xe0c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xe2c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0x44, {
    0x00000000,
  });

  dst += write_cont(dst, 0xaac, {
    0x00000000,
  });

  dst += write_cont(dst, 0xf00, {
    0x00000000,
  });

  // black level scale + offset
  dst += write_cont(dst, 0x6b0, {
    ((uint32_t)(1 << 11) << 0xf) | (s->black_level << (14 - s->bits_per_pixel)),
    0x0,
    0x0,
  });

  return dst - start;
}


int build_initial_config(uint8_t *dst, const SensorInfo *s, std::vector<uint32_t> &patches) {
  uint8_t *start = dst;

  // start with the every frame config
  dst += build_update(dst, s, patches);

  uint64_t addr;

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
  dst += write_dmi(dst, &addr, 288, 0xc24, 9);
  patches.push_back(addr - (uint64_t)start);
  /* TODO
  cdm_dmi_cmd_t 248
    .length = 287
    .reserved = 33
    .cmd = 11
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 9
  */

  // vignetting correction
  dst += write_cont(dst, 0x6bc, {
    0x0b3c0000,
    0x00670067,
    0xd3b1300c,
    0x13b1300c,
    0x00670067,
    0xd3b1300c,
    0x13b1300c,
    0xec4e4000,
    0x0100c003,
    0xec4e4000,
    0x0100c003,
  });
  dst += write_dmi(dst, &addr, 884, 0xc24, 14);
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, 884, 0xc24, 15);
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
  dst += write_dmi(dst, &addr, 256, 0xc24, 26);  // G
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, 256, 0xc24, 28);  // B
  patches.push_back(addr - (uint64_t)start);
  dst += write_dmi(dst, &addr, 256, 0xc24, 30);  // R
  patches.push_back(addr - (uint64_t)start);

  // YUV
  dst += write_cont(dst, 0xf30, {
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

  // TODO: remove this
  dst += write_cont(dst, 0xa3c, {
    0x00000003,
    ((s->frame_width - 1) << 16) | (s->frame_width - 1),
    0x30036666,
    0x00000000,
    0x00000000,
    s->frame_width - 1,
    ((s->frame_height - 1) << 16) | (s->frame_height - 1),
    0x30036666,
    0x00000000,
    0x00000000,
    s->frame_height - 1,
  });
  dst += write_cont(dst, 0xa68, {
    0x00000003,
    ((s->frame_width/2 - 1) << 16) | (s->frame_width - 1),
    0x3006cccc,
    0x00000000,
    0x00000000,
    s->frame_width - 1,
    ((s->frame_height/2 - 1) << 16) | (s->frame_height - 1),
    0x3006cccc,
    0x00000000,
    0x00000000,
    s->frame_height - 1,
  });

  // cropping
  dst += write_cont(dst, 0xe10, {
    s->frame_height - 1,
    s->frame_width - 1,
  });
  dst += write_cont(dst, 0xe30, {
    s->frame_height/2 - 1,
    s->frame_width - 1,
  });
  dst += write_cont(dst, 0xe18, {
    0x0ff00000,
    0x00000016,
  });
  dst += write_cont(dst, 0xe38, {
    0x0ff00000,
    0x00000017,
  });

  return dst - start;
}


