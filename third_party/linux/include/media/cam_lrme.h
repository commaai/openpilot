#ifndef __UAPI_CAM_LRME_H__
#define __UAPI_CAM_LRME_H__

#include "cam_defs.h"

/* LRME Resource Types */

enum CAM_LRME_IO_TYPE {
	CAM_LRME_IO_TYPE_TAR,
	CAM_LRME_IO_TYPE_REF,
	CAM_LRME_IO_TYPE_RES,
	CAM_LRME_IO_TYPE_DS2,
};

#define CAM_LRME_INPUT_PORT_TYPE_TAR (1 << 0)
#define CAM_LRME_INPUT_PORT_TYPE_REF (1 << 1)

#define CAM_LRME_OUTPUT_PORT_TYPE_DS2 (1 << 0)
#define CAM_LRME_OUTPUT_PORT_TYPE_RES (1 << 1)

#define CAM_LRME_DEV_MAX 1


struct cam_lrme_hw_version {
	uint32_t gen;
	uint32_t rev;
	uint32_t step;
};

struct cam_lrme_dev_cap {
	struct cam_lrme_hw_version clc_hw_version;
	struct cam_lrme_hw_version bus_rd_hw_version;
	struct cam_lrme_hw_version bus_wr_hw_version;
	struct cam_lrme_hw_version top_hw_version;
	struct cam_lrme_hw_version top_titan_version;
};

/**
 * struct cam_lrme_query_cap_cmd - LRME query device capability payload
 *
 * @dev_iommu_handle: LRME iommu handles for secure/non secure
 *      modes
 * @cdm_iommu_handle: Iommu handles for secure/non secure modes
 * @num_devices: number of hardware devices
 * @dev_caps: Returned device capability array
 */
struct cam_lrme_query_cap_cmd {
	struct cam_iommu_handle device_iommu;
	struct cam_iommu_handle cdm_iommu;
	uint32_t num_devices;
	struct cam_lrme_dev_cap dev_caps[CAM_LRME_DEV_MAX];
};

struct cam_lrme_soc_info {
	uint64_t clock_rate;
	uint64_t bandwidth;
	uint64_t reserved[4];
};

struct cam_lrme_acquire_args {
	struct cam_lrme_soc_info lrme_soc_info;
};

#endif /* __UAPI_CAM_LRME_H__ */

