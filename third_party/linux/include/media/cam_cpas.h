#ifndef __UAPI_CAM_CPAS_H__
#define __UAPI_CAM_CPAS_H__

#include "cam_defs.h"

#define CAM_FAMILY_CAMERA_SS     1
#define CAM_FAMILY_CPAS_SS       2

/**
 * struct cam_cpas_query_cap - CPAS query device capability payload
 *
 * @camera_family     : Camera family type
 * @reserved          : Reserved field for alignment
 * @camera_version    : Camera platform version
 * @cpas_version      : Camera CPAS version within camera platform
 *
 */
struct cam_cpas_query_cap {
	uint32_t                 camera_family;
	uint32_t                 reserved;
	struct cam_hw_version    camera_version;
	struct cam_hw_version    cpas_version;
};

#endif /* __UAPI_CAM_CPAS_H__ */
