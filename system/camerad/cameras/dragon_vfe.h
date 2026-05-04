#pragma once

#include <linux/ioctl.h>
#include <linux/types.h>

struct vfe_reg_write {
  __u32 offset;
  __u32 value;
};

struct vfe_write_regs_cmd {
  __u64 regs;
  __u32 count;
  __u32 pad;
};

struct vfe_dmi_cmd {
  __u32 dmi_cfg_offset;
  __u8 ram_select;
  __u8 pad[3];
  __u32 count;
  __u64 data;
};

struct vfe_set_buf_cmd {
  __u32 wm_index;
  __u32 stride;
  __u64 iova;
  __u32 frame_inc;
  __u32 pad;
};

#define VFE_IOC_MAGIC '#'

#define VFE_WRITE_REGS _IOW(VFE_IOC_MAGIC, 1, struct vfe_write_regs_cmd)
#define VFE_WRITE_DMI _IOW(VFE_IOC_MAGIC, 2, struct vfe_dmi_cmd)
#define VFE_SET_BUF _IOW(VFE_IOC_MAGIC, 5, struct vfe_set_buf_cmd)
#define VFE_REG_UPDATE _IO(VFE_IOC_MAGIC, 6)
#define VFE_WAIT_SOF _IO(VFE_IOC_MAGIC, 9)
