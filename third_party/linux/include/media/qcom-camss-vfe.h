/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Qualcomm Camera Subsystem VFE userspace register control interface.
 *
 * Allows userspace (camerad) to program ISP registers, upload DMI LUTs,
 * map DMA-BUF buffers into the VFE IOMMU, and control frame output.
 * The kernel driver handles power, clocks, IOMMU, and interrupt delivery.
 */
#ifndef __UAPI_QCOM_CAMSS_VFE_H
#define __UAPI_QCOM_CAMSS_VFE_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define VFE_REG_SPACE_SIZE	0x3000

struct vfe_reg_write {
	__u32 offset;
	__u32 value;
};

struct vfe_write_regs_cmd {
	__u64 regs;	/* userspace pointer to struct vfe_reg_write[] */
	__u32 count;
	__u32 pad;
};

struct vfe_dmi_cmd {
	__u32 dmi_cfg_offset;	/* DMI config register (e.g. 0xc24 on VFE170) */
	__u8  ram_select;	/* DMI table selector */
	__u8  pad[3];
	__u32 count;		/* number of 32-bit entries */
	__u64 data;		/* userspace pointer to __u32[] */
};

struct vfe_map_buf_cmd {
	__s32 fd;		/* input: DMA-BUF fd */
	__u32 pad;
	__u64 iova;		/* output: IOVA in VFE SMMU */
	__u64 size;		/* output: mapped size */
};

struct vfe_unmap_buf_cmd {
	__u64 iova;
};

struct vfe_set_buf_cmd {
	__u32 wm_index;		/* write master index */
	__u32 stride;		/* bytes per line */
	__u64 iova;		/* buffer IOVA */
	__u32 frame_inc;	/* frame size for auto-increment */
	__u32 pad;
};

#define VFE_IOC_MAGIC		'#'

#define VFE_WRITE_REGS		_IOW(VFE_IOC_MAGIC, 1, struct vfe_write_regs_cmd)
#define VFE_WRITE_DMI		_IOW(VFE_IOC_MAGIC, 2, struct vfe_dmi_cmd)
#define VFE_MAP_BUF		_IOWR(VFE_IOC_MAGIC, 3, struct vfe_map_buf_cmd)
#define VFE_UNMAP_BUF		_IOW(VFE_IOC_MAGIC, 4, struct vfe_unmap_buf_cmd)
#define VFE_SET_BUF		_IOW(VFE_IOC_MAGIC, 5, struct vfe_set_buf_cmd)
#define VFE_REG_UPDATE		_IO(VFE_IOC_MAGIC, 6)
#define VFE_START		_IO(VFE_IOC_MAGIC, 7)
#define VFE_STOP		_IO(VFE_IOC_MAGIC, 8)

/*
 * Sensor register write interface.
 * Used on sensor V4L2 subdev fd via VIDIOC_DEFAULT.
 */
struct sensor_reg_write {
	__u16 addr;
	__u16 data;
};

struct sensor_write_regs_cmd {
	__u64 regs;		/* userspace pointer to struct sensor_reg_write[] */
	__u32 count;
	__u8  data_width;	/* 1 = byte, 2 = word */
	__u8  pad[3];
};

#define SENSOR_IOC_MAGIC	'S'

#define SENSOR_WRITE_REGS	_IOW(SENSOR_IOC_MAGIC, 1, struct sensor_write_regs_cmd)

#endif /* __UAPI_QCOM_CAMSS_VFE_H */
