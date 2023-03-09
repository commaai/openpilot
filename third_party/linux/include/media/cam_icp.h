#ifndef __UAPI_CAM_ICP_H__
#define __UAPI_CAM_ICP_H__

#include "cam_defs.h"

/* icp, ipe, bps, cdm(ipe/bps) are used in querycap */
#define CAM_ICP_DEV_TYPE_A5      1
#define CAM_ICP_DEV_TYPE_IPE     2
#define CAM_ICP_DEV_TYPE_BPS     3
#define CAM_ICP_DEV_TYPE_IPE_CDM 4
#define CAM_ICP_DEV_TYPE_BPS_CDM 5
#define CAM_ICP_DEV_TYPE_MAX     5

/* definitions needed for icp aquire device */
#define CAM_ICP_RES_TYPE_BPS        1
#define CAM_ICP_RES_TYPE_IPE_RT     2
#define CAM_ICP_RES_TYPE_IPE        3
#define CAM_ICP_RES_TYPE_MAX        4

/* packet opcode types */
#define CAM_ICP_OPCODE_IPE_UPDATE 0
#define CAM_ICP_OPCODE_BPS_UPDATE 1

/* IPE input port resource type */
#define CAM_ICP_IPE_INPUT_IMAGE_FULL            0x0
#define CAM_ICP_IPE_INPUT_IMAGE_DS4             0x1
#define CAM_ICP_IPE_INPUT_IMAGE_DS16            0x2
#define CAM_ICP_IPE_INPUT_IMAGE_DS64            0x3
#define CAM_ICP_IPE_INPUT_IMAGE_FULL_REF        0x4
#define CAM_ICP_IPE_INPUT_IMAGE_DS4_REF         0x5
#define CAM_ICP_IPE_INPUT_IMAGE_DS16_REF        0x6
#define CAM_ICP_IPE_INPUT_IMAGE_DS64_REF        0x7

/* IPE output port resource type */
#define CAM_ICP_IPE_OUTPUT_IMAGE_DISPLAY        0x8
#define CAM_ICP_IPE_OUTPUT_IMAGE_VIDEO          0x9
#define CAM_ICP_IPE_OUTPUT_IMAGE_FULL_REF       0xA
#define CAM_ICP_IPE_OUTPUT_IMAGE_DS4_REF        0xB
#define CAM_ICP_IPE_OUTPUT_IMAGE_DS16_REF       0xC
#define CAM_ICP_IPE_OUTPUT_IMAGE_DS64_REF       0xD

#define CAM_ICP_IPE_IMAGE_MAX                   0xE

/* BPS input port resource type */
#define CAM_ICP_BPS_INPUT_IMAGE                 0x0

/* BPS output port resource type */
#define CAM_ICP_BPS_OUTPUT_IMAGE_FULL           0x1
#define CAM_ICP_BPS_OUTPUT_IMAGE_DS4            0x2
#define CAM_ICP_BPS_OUTPUT_IMAGE_DS16           0x3
#define CAM_ICP_BPS_OUTPUT_IMAGE_DS64           0x4
#define CAM_ICP_BPS_OUTPUT_IMAGE_STATS_BG       0x5
#define CAM_ICP_BPS_OUTPUT_IMAGE_STATS_BHIST    0x6
#define CAM_ICP_BPS_OUTPUT_IMAGE_REG1           0x7
#define CAM_ICP_BPS_OUTPUT_IMAGE_REG2           0x8

#define CAM_ICP_BPS_IO_IMAGES_MAX               0x9

/* Command meta types */
#define CAM_ICP_CMD_META_GENERIC_BLOB           0x1

/* Generic blob types */
#define CAM_ICP_CMD_GENERIC_BLOB_CLK            0x1
#define CAM_ICP_CMD_GENERIC_BLOB_CFG_IO         0x2

/**
 * struct cam_icp_clk_bw_request
 *
 * @budget_ns: Time required to process frame
 * @frame_cycles: Frame cycles needed to process the frame
 * @rt_flag: Flag to indicate real time stream
 * @uncompressed_bw: Bandwidth required to process frame
 * @compressed_bw: Compressed bandwidth to process frame
 */
struct cam_icp_clk_bw_request {
	uint64_t budget_ns;
	uint32_t frame_cycles;
	uint32_t rt_flag;
	uint64_t uncompressed_bw;
	uint64_t compressed_bw;
};

/**
 * struct cam_icp_dev_ver - Device information for particular hw type
 *
 * This is used to get device version info of
 * ICP, IPE, BPS and CDM related IPE and BPS from firmware
 * and use this info in CAM_QUERY_CAP IOCTL
 *
 * @dev_type: hardware type for the cap info(icp, ipe, bps, cdm(ipe/bps))
 * @reserved: reserved field
 * @hw_ver: major, minor and incr values of a device version
 */
struct cam_icp_dev_ver {
	uint32_t dev_type;
	uint32_t reserved;
	struct cam_hw_version hw_ver;
};

/**
 * struct cam_icp_ver - ICP version info
 *
 * This strcuture is used for fw and api version
 * this is used to get firmware version and api version from firmware
 * and use this info in CAM_QUERY_CAP IOCTL
 *
 * @major: FW version major
 * @minor: FW version minor
 * @revision: FW version increment
 */
struct cam_icp_ver {
	uint32_t major;
	uint32_t minor;
	uint32_t revision;
	uint32_t reserved;
};

/**
 * struct cam_icp_query_cap_cmd - ICP query device capability payload
 *
 * @dev_iommu_handle: icp iommu handles for secure/non secure modes
 * @cdm_iommu_handle: iommu handles for secure/non secure modes
 * @fw_version: firmware version info
 * @api_version: api version info
 * @num_ipe: number of ipes
 * @num_bps: number of bps
 * @dev_ver: returned device capability array
 */
struct cam_icp_query_cap_cmd {
	struct cam_iommu_handle dev_iommu_handle;
	struct cam_iommu_handle cdm_iommu_handle;
	struct cam_icp_ver fw_version;
	struct cam_icp_ver api_version;
	uint32_t num_ipe;
	uint32_t num_bps;
	struct cam_icp_dev_ver dev_ver[CAM_ICP_DEV_TYPE_MAX];
};

/**
 * struct cam_icp_res_info - ICP output resource info
 *
 * @format: format of the resource
 * @width:  width in pixels
 * @height: height in lines
 * @fps:  fps
 */
struct cam_icp_res_info {
	uint32_t format;
	uint32_t width;
	uint32_t height;
	uint32_t fps;
};

/**
 * struct cam_icp_acquire_dev_info - An ICP device info
 *
 * @scratch_mem_size: Output param - size of scratch memory
 * @dev_type: device type (IPE_RT/IPE_NON_RT/BPS)
 * @io_config_cmd_size: size of IO config command
 * @io_config_cmd_handle: IO config command for each acquire
 * @secure_mode: camera mode (secure/non secure)
 * @chain_info: chaining info of FW device handles
 * @in_res: resource info used for clock and bandwidth calculation
 * @num_out_res: number of output resources
 * @out_res: output resource
 */
struct cam_icp_acquire_dev_info {
	uint32_t scratch_mem_size;
	uint32_t dev_type;
	uint32_t io_config_cmd_size;
	int32_t  io_config_cmd_handle;
	uint32_t secure_mode;
	int32_t chain_info;
	struct cam_icp_res_info in_res;
	uint32_t num_out_res;
	struct cam_icp_res_info out_res[1];
} __attribute__((__packed__));

#endif /* __UAPI_CAM_ICP_H__ */
