#include "cdm.h"

int build_initial_config(uint8_t *dst) {
  uint8_t *start = dst;

  // constants, some kind of HW quirk?
  dst += write_random(dst, {
    0x2c, 0xffffffff,
    0x30, 0xffffffff,
    0x34, 0xffffffff,
    0x38, 0xffffffff,
    0x3c, 0xffffffff,
  });

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

  dst += write_cont(dst, 0x4dc, {
    0x00000000,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
  });
  /* TODO
  cdm_dmi_cmd_t 248
    .length = 287
    .reserved = 33
    .cmd = 11
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 9
  */

  dst += write_cont(dst, 0x560, {
    0x00000001,
    0x04440444,
    0x04450445,
    0x04440444,
    0x04450445,
    0x000000ca,
    0x0000009c,
  });

  dst += write_cont(dst, 0x5e8, {
    0x06363005,
  });

  dst += write_cont(dst, 0x5f4, {
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x3b3839a0,
    0x003f8040,
    0x00000000,
    0x00000000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00000009,
    0x00400808,
    0x00000044,
    0x004000a0,
    0x0a0d00a6,
    0x0a0d00a6,
  });
  /* TODO
  cdm_dmi_cmd_t 392
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 12
  */

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
  /* TODO
  cdm_dmi_cmd_t 444
    .length = 883
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 14
  */
  /* TODO
  cdm_dmi_cmd_t 444
    .length = 883
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 15
  */

  dst += write_cont(dst, 0x6fc, {
    0x00bf0080,
    0x00000106,
    0x00000000,
    0x00000000,
  });

  dst += write_cont(dst, 0x6f8, {
    0x00000100,
  });

  dst += write_cont(dst, 0x71c, {
    0x00008000,
    0x08000066,
  });

  dst += write_cont(dst, 0x760, {
    0x00800080,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00800080,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00800080,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
  });

  dst += write_cont(dst, 0x794, {
    0x00000000,
  });
  /* TODO
  cdm_dmi_cmd_t 568
    .length = 511
    .reserved = 33
    .cmd = 11
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 24
  */

  dst += write_cont(dst, 0x798, {
    0x00000000,
  });
  /* TODO
  cdm_dmi_cmd_t 580
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 26
  */
  /* TODO
  cdm_dmi_cmd_t 580
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 28
  */
  /* TODO
  cdm_dmi_cmd_t 580
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 0
    .DMIAddr = 3108
    .DMISel = 30
  */

  dst += write_cont(dst, 0xf30, {
    0x00750259,
    0x00000132,
    0x00000000,
    0x03ff0000,
    0x01fe1eae,
    0x00001f54,
    0x02000000,
    0x03ff0000,
    0x1fad1e55,
    0x000001fe,
    0x02000000,
    0x03ff0000,
  });

  dst += write_cont(dst, 0xa3c, {
    0x00000003,
    0x07870787,
    0x30036666,
    0x00000000,
    0x00000000,
    0x00000787,
    0x04b704b7,
    0x30036666,
    0x00000000,
    0x00000000,
    0x000004b7,
  });

  dst += write_cont(dst, 0xa68, {
    0x00000003,
    0x03c30787,
    0x3006cccc,
    0x00000000,
    0x00000000,
    0x00000787,
    0x025b04b7,
    0x3006cccc,
    0x00000000,
    0x00000000,
    0x00000787,
  });


  dst += write_cont(dst, 0xe10, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xe30, {
    0x0000025b,
    0x00000787,
  });

  dst += write_cont(dst, 0xe18, {
    0x0ff00000,
    0x00000016,
  });

  dst += write_cont(dst, 0xe38, {
    0x0ff00000,
    0x00000017,
  });

  dst += write_cont(dst, 0xd84, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xda4, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xd60, {
    0x04380300,
    0x09016c7d,
    0x021c0300,
  });

  dst += write_cont(dst, 0xd98, {
    0x0ff00000,
    0x00000016,
  });

  dst += write_cont(dst, 0xdb8, {
    0x0ff00000,
    0x00000017,
  });

  dst += write_cont(dst, 0xd6c, {
    0x00000300,
  });

  dst += write_cont(dst, 0xd70, {
    0x010e0f00,
    0x09016c7d,
    0x00870f00,
  });

  dst += write_cont(dst, 0xd7c, {
    0x00000f00,
  });

  dst += write_cont(dst, 0x40, {
    0x00000586,
  });

  dst += write_cont(dst, 0x48, {
    0x0000000e,
  });

  dst += write_cont(dst, 0x4c, {
    0x00000019,
  });

  dst += write_cont(dst, 0xe4c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe6c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe0c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xe2c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xd8c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdac, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdcc, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdec, {
    0x00000000,
  });

  dst += write_cont(dst, 0x44, {
    0x00000000,
  });

  dst += write_cont(dst, 0xaac, {
    0x00000040,
  });

  dst += write_cont(dst, 0xf00, {
    0x00000000,
  });

  //hexdump(start, dst - start);
  return dst - start;
}

int build_first_update(uint8_t *dst) {
  uint8_t *start = dst;

  dst += write_random(dst, {
    0x2c, 0xffffffff,
    0x30, 0xffffffff,
    0x34, 0xffffffff,
    0x38, 0xffffffff,
    0x3c, 0xffffffff,
  });

  dst += write_cont(dst, 0x4dc, {
    0x00000001,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
    0x04050b84,
    0x13031a82,
    0x22022981,
    0x3100387f,
    0x04010b80,
    0x13001a80,
    0x2200297f,
    0x30ff387f,
  });
  /* TODO
  cdm_dmi_cmd_t 184
    .length = 287
    .reserved = 33
    .cmd = 11
    .addr = 832
    .DMIAddr = 3108
    .DMISel = 10
  */

  dst += write_cont(dst, 0x560, {
    0x00000001,
    0x04440444,
    0x04450445,
    0x04440444,
    0x04450445,
    0x000000ca,
    0x0000009c,
  });

  dst += write_cont(dst, 0x5c4, {
    0x00000000,
    0x00001000,
    0x00001000,
    0x00001000,
    0x00001000,
    0x00800080,
    0x00802040,
    0x00000000,
  });

  dst += write_cont(dst, 0x5e8, {
    0x06363007,
  });

  dst += write_cont(dst, 0x5f4, {
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x3b3839a0,
    0x003f8040,
    0x00000000,
    0x00000000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00078000,
    0x00000009,
    0x00400808,
    0x00000044,
    0x004000a0,
    0x0a0d00a6,
    0x0a0d00a6,
  });
  /* TODO
  cdm_dmi_cmd_t 368
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 5344
    .DMIAddr = 3108
    .DMISel = 13
  */

  dst += write_cont(dst, 0x6fc, {
    0x00800080,
    0x00000080,
    0x00000000,
    0x00000000,
  });

  dst += write_cont(dst, 0x6f8, {
    0x00000100,
  });

  dst += write_cont(dst, 0x71c, {
    0x00008000,
    0x08000066,
  });

  dst += write_cont(dst, 0x794, {
    0x00000001,
  });
  /* TODO
  cdm_dmi_cmd_t 432
    .length = 511
    .reserved = 33
    .cmd = 11
    .addr = 832
    .DMIAddr = 3108
    .DMISel = 25
  */

  dst += write_cont(dst, 0x798, {
    0x00000007,
  });
  /* TODO
  cdm_dmi_cmd_t 444
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 5344
    .DMIAddr = 3108
    .DMISel = 27
  */
  /* TODO
  cdm_dmi_cmd_t 444
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 5344
    .DMIAddr = 3108
    .DMISel = 29
  */
  /* TODO
  cdm_dmi_cmd_t 444
    .length = 255
    .reserved = 33
    .cmd = 10
    .addr = 5344
    .DMIAddr = 3108
    .DMISel = 31
  */

  dst += write_cont(dst, 0xd84, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xda4, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xd6c, {
    0x00000300,
  });

  dst += write_cont(dst, 0xd70, {
    0x02640f00,
    0x09016c7d,
    0x01320f00,
  });

  dst += write_cont(dst, 0xd7c, {
    0x00000f00,
  });

  dst += write_cont(dst, 0x40, {
    0x00000444,
  });

  dst += write_cont(dst, 0x48, {
    0x00000000,
  });

  dst += write_cont(dst, 0x4c, {
    0x00000019,
  });

  dst += write_cont(dst, 0xe4c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe6c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe0c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xe2c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xd8c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdac, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdcc, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdec, {
    0x00000000,
  });

  dst += write_cont(dst, 0x44, {
    0x00000000,
  });

  dst += write_cont(dst, 0xaac, {
    0x00000040,
  });

  dst += write_cont(dst, 0xf00, {
    0x00000000,
  });

  return dst - start;
}

int build_update(uint8_t *dst) {
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

  dst += write_cont(dst, 0x6fc, {
    0x00800080,
    0x00000080,
    0x00000000,
    0x00000000,
  });

  dst += write_cont(dst, 0xd84, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xda4, {
    0x000004b7,
    0x00000787,
  });

  dst += write_cont(dst, 0xd6c, {
    0x00000300,
  });

  dst += write_cont(dst, 0xd70, {
    0x02640f00,
    0x09016c7d,
    0x01320f00,
  });

  dst += write_cont(dst, 0xd7c, {
    0x00000f00,
  });

  dst += write_cont(dst, 0x40, {
    0x00000444,
  });

  dst += write_cont(dst, 0x48, {
    0x00000000,
  });

  dst += write_cont(dst, 0x4c, {
    0x00000019,
  });

  dst += write_cont(dst, 0xe4c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe6c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xe0c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xe2c, {
    0x00000e00,
  });

  dst += write_cont(dst, 0xd8c, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdac, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdcc, {
    0x00000000,
  });

  dst += write_cont(dst, 0xdec, {
    0x00000000,
  });

  dst += write_cont(dst, 0x44, {
    0x00000000,
  });

  dst += write_cont(dst, 0xaac, {
    0x00000040,
  });

  dst += write_cont(dst, 0xf00, {
    0x00000000,
  });

  return dst - start;
}
