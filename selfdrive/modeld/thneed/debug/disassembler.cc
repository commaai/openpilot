#include "debug/include/adreno_pm4types.h"
#define REG_A5XX_TPL1_CS_TEX_CONST_LO        0x0000e760
#define REG_A5XX_TPL1_CS_TEX_SAMP_LO         0x0000e75c
#define REG_A5XX_SP_CS_CTRL_REG0             0x0000e5f0

std::map<int, std::string> regs = {
  {0x0000e760, "REG_A5XX_TPL1_CS_TEX_CONST_LO"},
  {0x0000e75c, "REG_A5XX_TPL1_CS_TEX_SAMP_LO"},
  {0x00000e06, "REG_A5XX_HLSQ_MODE_CNTL"},
  {0x00000e91, "REG_A5XX_UCHE_CACHE_INVALIDATE_MIN_LO"},
  {0x00000ec2, "REG_A5XX_SP_MODE_CNTL"},
  {0x0000e580, "REG_A5XX_SP_SP_CNTL"},
  {0x0000e5f0, "REG_A5XX_SP_CS_CTRL_REG0"},
  {0x0000e796, "REG_A5XX_HLSQ_CS_CNTL"},
  {0x0000e784, "REG_A5XX_HLSQ_CONTROL_0_REG"},
  {0x0000e7b0, "REG_A5XX_HLSQ_CS_NDRANGE_0"},
  {0x0000e7b9, "REG_A5XX_HLSQ_CS_KERNEL_GROUP_X"},
  {0x00000cdd, "REG_A5XX_VSC_RESOLVE_CNTL"},
};

std::map<int, std::string> ops = {
  {33, "CP_REG_RMW"},
  {62, "CP_REG_TO_MEM"},
  {49, "CP_RUN_OPENCL"},
  {16, "CP_NOP"},
  {38, "CP_WAIT_FOR_IDLE"},
  {110, "CP_COMPUTE_CHECKPOINT"},
  {48, "CP_LOAD_STATE"},
};

void CachedCommand::disassemble() {
  uint32_t *src = (uint32_t *)cmds[1].gpuaddr;
  int len = cmds[1].size/4;
  printf("disassemble %p %d\n", src, len);

  int i = 0;
  while (i < len) {
    int pktsize;
    int pkttype = -1;

    if (pkt_is_type0(src[i])) {
      pkttype = 0;
      pktsize = type0_pkt_size(src[i]);
    } else if (pkt_is_type3(src[i])) {
      pkttype = 3;
      pktsize = type3_pkt_size(src[i]);
    } else if (pkt_is_type4(src[i])) {
      pkttype = 4;
      pktsize = type4_pkt_size(src[i]);
    } else if (pkt_is_type7(src[i])) {
      pkttype = 7;
      pktsize = type7_pkt_size(src[i]);
    }
    printf("%3d: type:%d size:%d ", i, pkttype, pktsize);

    if (pkttype == 7) {
      int op = cp_type7_opcode(src[i]);
      if (ops.find(op) != ops.end()) {
        printf("%-40s ", ops[op].c_str());
      } else {
        printf("op:  %4d ", op);
      }
    }

    if (pkttype == 4) {
      int reg = cp_type4_base_index_one_reg_wr(src[i]);
      if (regs.find(reg) != regs.end()) {
        printf("%-40s ", regs[reg].c_str());
      } else {
        printf("reg: %4x ", reg);
      }
    }

    for (int j = 0; j < pktsize+1; j++) {
      printf("%8.8X ", src[i+j]);
    }
    printf("\n");

    uint64_t addr;
    if (pkttype == 7) {
      switch (cp_type7_opcode(src[i])) {
        case CP_LOAD_STATE:
          int dst_off = src[i+1] & 0x1FFF;
          int state_src = (src[i+1] >> 16) & 3;
          int state_block = (src[i+1] >> 18) & 7;
          int state_type = src[i+2] & 3;
          int num_unit = (src[i+1] & 0xffc00000) >> 22;
          printf("  dst_off: %x  state_src: %d  state_block: %d  state_type: %d  num_unit: %d\n",
              dst_off, state_src, state_block, state_type, num_unit);
          addr = (uint64_t)(src[i+2] & 0xfffffffc) | ((uint64_t)(src[i+3]) << 32);
          if (state_block == 5 && state_type == 0) {
            if (!(addr&0xFFF)) {
              int len = 0x1000;
              if (num_unit >= 32) len += 0x1000;
              //hexdump((uint32_t *)addr, len);
              char fn[0x100];
              snprintf(fn, sizeof(fn), "/tmp/0x%lx.shader", addr);
              printf("dumping %s\n", fn);
              FILE *f = fopen(fn, "wb");
              // groups of 16 instructions
              fwrite((void*)addr, 1, len, f);
              fclose(f);
            }
          }
          break;
      }
    }

    /*if (pkttype == 4) {
      switch (cp_type4_base_index_one_reg_wr(src[i])) {
        case REG_A5XX_SP_CS_CTRL_REG0:
          addr = (uint64_t)(src[i+4] & 0xfffffffc) | ((uint64_t)(src[i+5]) << 32);
          hexdump((uint32_t *)addr, 0x1000);
          break;
      }
    }*/

    /*if (pkttype == 4 && cp_type4_base_index_one_reg_wr(src[i]) == REG_A5XX_TPL1_CS_TEX_CONST_LO) {
      uint64_t addr = (uint64_t)(src[i+1] & 0xffffffff) | ((uint64_t)(src[i+2]) << 32);
      hexdump((uint32_t *)addr, 0x40);
    }

    if (pkttype == 4 && cp_type4_base_index_one_reg_wr(src[i]) == REG_A5XX_TPL1_CS_TEX_SAMP_LO) {
      uint64_t addr = (uint64_t)(src[i+1] & 0xffffffff) | ((uint64_t)(src[i+2]) << 32);
      hexdump((uint32_t *)addr, 0x40);
    }*/

    if (pkttype == -1) break;
    i += (1+pktsize);
  }
  assert(i == len);
}
