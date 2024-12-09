#include "cdm.h"
#include "stddef.h"

int write_dmi(uint8_t *dst, uint64_t *addr, uint32_t length, uint32_t dmi_addr, uint8_t sel) {
  struct cdm_dmi_cmd *cmd = (struct cdm_dmi_cmd*)dst;
  cmd->cmd = CAM_CDM_CMD_DMI_32;
  cmd->length = length - 1;
  cmd->reserved = 0;
  cmd->addr = 0; // gets patched in
  cmd->DMIAddr = dmi_addr;
  cmd->DMISel = sel;

  *addr = (uint64_t)(dst + offsetof(struct cdm_dmi_cmd, addr));
  return sizeof(struct cdm_dmi_cmd);
}

int write_cont(uint8_t *dst, uint32_t reg, std::vector<uint32_t> vals) {
  struct cdm_regcontinuous_cmd *cmd = (struct cdm_regcontinuous_cmd*)dst;
  cmd->cmd = CAM_CDM_CMD_REG_CONT;
  cmd->count = vals.size();
  cmd->offset = reg;
  cmd->reserved0 = 0;
  cmd->reserved1 = 0;

  uint32_t *vd = (uint32_t*)(dst + sizeof(struct cdm_regcontinuous_cmd));
  for (int i = 0; i < vals.size(); i++) {
    *vd = vals[i];
    vd++;
  }

  return sizeof(struct cdm_regcontinuous_cmd) + vals.size()*sizeof(uint32_t);
}

int write_random(uint8_t *dst, std::vector<uint32_t> vals) {
  struct cdm_regrandom_cmd *cmd = (struct cdm_regrandom_cmd*)dst;
  cmd->cmd = CAM_CDM_CMD_REG_RANDOM;
  cmd->count = vals.size() / 2;
  cmd->reserved = 0;

  uint32_t *vd = (uint32_t*)(dst + sizeof(struct cdm_regrandom_cmd));
  for (int i = 0; i < vals.size(); i++) {
    *vd = vals[i];
    vd++;
  }

  return sizeof(struct cdm_regrandom_cmd) + vals.size()*sizeof(uint32_t);
}
