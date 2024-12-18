#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>

// our helpers
int write_random(uint8_t *dst, const std::vector<uint32_t> &vals);
int write_cont(uint8_t *dst, uint32_t reg, const std::vector<uint32_t> &vals);
int write_dmi(uint8_t *dst, uint64_t *addr, uint32_t length, uint32_t dmi_addr, uint8_t sel);

// from drivers/media/platform/msm/camera/cam_cdm/cam_cdm_util.{c,h}

enum cam_cdm_command {
  CAM_CDM_CMD_UNUSED = 0x0,
  CAM_CDM_CMD_DMI = 0x1,
  CAM_CDM_CMD_NOT_DEFINED = 0x2,
  CAM_CDM_CMD_REG_CONT = 0x3,
  CAM_CDM_CMD_REG_RANDOM = 0x4,
  CAM_CDM_CMD_BUFF_INDIRECT = 0x5,
  CAM_CDM_CMD_GEN_IRQ = 0x6,
  CAM_CDM_CMD_WAIT_EVENT = 0x7,
  CAM_CDM_CMD_CHANGE_BASE = 0x8,
  CAM_CDM_CMD_PERF_CTRL = 0x9,
  CAM_CDM_CMD_DMI_32 = 0xa,
  CAM_CDM_CMD_DMI_64 = 0xb,
  CAM_CDM_CMD_PRIVATE_BASE = 0xc,
  CAM_CDM_CMD_SWD_DMI_32 = (CAM_CDM_CMD_PRIVATE_BASE + 0x64),
  CAM_CDM_CMD_SWD_DMI_64 = (CAM_CDM_CMD_PRIVATE_BASE + 0x65),
  CAM_CDM_CMD_PRIVATE_BASE_MAX = 0x7F
};

/**
 * struct cdm_regrandom_cmd - Definition for CDM random register command.
 * @count: Number of register writes
 * @reserved: reserved bits
 * @cmd: Command ID (CDMCmd)
 */
struct cdm_regrandom_cmd {
  unsigned int count    : 16;
  unsigned int reserved : 8;
  unsigned int cmd      : 8;
} __attribute__((__packed__));

/**
 * struct cdm_regcontinuous_cmd - Definition for a CDM register range command.
 * @count: Number of register writes
 * @reserved0: reserved bits
 * @cmd: Command ID (CDMCmd)
 * @offset: Start address of the range of registers
 * @reserved1: reserved bits
 */
struct cdm_regcontinuous_cmd {
  unsigned int count     : 16;
  unsigned int reserved0 : 8;
  unsigned int cmd       : 8;
  unsigned int offset    : 24;
  unsigned int reserved1 : 8;
} __attribute__((__packed__));

/**
 * struct cdm_dmi_cmd - Definition for a CDM DMI command.
 * @length: Number of bytes in LUT - 1
 * @reserved: reserved bits
 * @cmd: Command ID (CDMCmd)
 * @addr: Address of the LUT in memory
 * @DMIAddr: Address of the target DMI config register
 * @DMISel: DMI identifier
 */
struct cdm_dmi_cmd {
  unsigned int length   : 16;
  unsigned int reserved : 8;
  unsigned int cmd      : 8;
  unsigned int addr;
  unsigned int DMIAddr  : 24;
  unsigned int DMISel   : 8;
} __attribute__((__packed__));
