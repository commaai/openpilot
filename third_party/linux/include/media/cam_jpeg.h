#ifndef __UAPI_CAM_JPEG_H__
#define __UAPI_CAM_JPEG_H__

#include "cam_defs.h"

/* enc, dma, cdm(enc/dma) are used in querycap */
#define CAM_JPEG_DEV_TYPE_ENC      0
#define CAM_JPEG_DEV_TYPE_DMA      1
#define CAM_JPEG_DEV_TYPE_MAX      2

#define CAM_JPEG_NUM_DEV_PER_RES_MAX      1

/* definitions needed for jpeg aquire device */
#define CAM_JPEG_RES_TYPE_ENC        0
#define CAM_JPEG_RES_TYPE_DMA        1
#define CAM_JPEG_RES_TYPE_MAX        2

/* packet opcode types */
#define CAM_JPEG_OPCODE_ENC_UPDATE 0
#define CAM_JPEG_OPCODE_DMA_UPDATE 1

/* ENC input port resource type */
#define CAM_JPEG_ENC_INPUT_IMAGE                 0x0

/* ENC output port resource type */
#define CAM_JPEG_ENC_OUTPUT_IMAGE                0x1

#define CAM_JPEG_ENC_IO_IMAGES_MAX               0x2

/* DMA input port resource type */
#define CAM_JPEG_DMA_INPUT_IMAGE                 0x0

/* DMA output port resource type */
#define CAM_JPEG_DMA_OUTPUT_IMAGE                0x1

#define CAM_JPEG_DMA_IO_IMAGES_MAX               0x2

#define CAM_JPEG_IMAGE_MAX                       0x2

/**
 * struct cam_jpeg_dev_ver - Device information for particular hw type
 *
 * This is used to get device version info of JPEG ENC, JPEG DMA
 * from hardware and use this info in CAM_QUERY_CAP IOCTL
 *
 * @size : Size of struct passed
 * @dev_type: Hardware type for the cap info(jpeg enc, jpeg dma)
 * @hw_ver: Major, minor and incr values of a device version
 */
struct cam_jpeg_dev_ver {
	uint32_t size;
	uint32_t dev_type;
	struct cam_hw_version hw_ver;
};

/**
 * struct cam_jpeg_query_cap_cmd - JPEG query device capability payload
 *
 * @dev_iommu_handle: Jpeg iommu handles for secure/non secure
 *      modes
 * @cdm_iommu_handle: Iommu handles for secure/non secure modes
 * @num_enc: Number of encoder
 * @num_dma: Number of dma
 * @dev_ver: Returned device capability array
 */
struct cam_jpeg_query_cap_cmd {
	struct cam_iommu_handle dev_iommu_handle;
	struct cam_iommu_handle cdm_iommu_handle;
	uint32_t num_enc;
	uint32_t num_dma;
	struct cam_jpeg_dev_ver dev_ver[CAM_JPEG_DEV_TYPE_MAX];
};

/**
 * struct cam_jpeg_res_info - JPEG output resource info
 *
 * @format: Format of the resource
 * @width:  Width in pixels
 * @height: Height in lines
 * @fps:  Fps
 */
struct cam_jpeg_res_info {
	uint32_t format;
	uint32_t width;
	uint32_t height;
	uint32_t fps;
};

/**
 * struct cam_jpeg_acquire_dev_info - An JPEG device info
 *
 * @dev_type: Device type (ENC/DMA)
 * @reserved: Reserved Bytes
 * @in_res: In resource info
 * @in_res: Iut resource info
 */
struct cam_jpeg_acquire_dev_info {
	uint32_t dev_type;
	uint32_t reserved;
	struct cam_jpeg_res_info in_res;
	struct cam_jpeg_res_info out_res;
};

/**
 * struct cam_jpeg_config_inout_param_info - JPEG Config time
 *     input output params
 *
 * @clk_index: Input Param- clock selection index.(-1 default)
 * @output_size: Output Param - jpeg encode/dma output size in
 *     bytes
 */
struct cam_jpeg_config_inout_param_info {
	int32_t clk_index;
	int32_t output_size;
};

#endif /* __UAPI_CAM_JPEG_H__ */
