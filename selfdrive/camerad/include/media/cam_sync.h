#ifndef __UAPI_CAM_SYNC_H__
#define __UAPI_CAM_SYNC_H__

#include <linux/videodev2.h>
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/media.h>

#define CAM_SYNC_DEVICE_NAME                     "cam_sync_device"

/* V4L event which user space will subscribe to */
#define CAM_SYNC_V4L_EVENT                       (V4L2_EVENT_PRIVATE_START + 0)

/* Specific event ids to get notified in user space */
#define CAM_SYNC_V4L_EVENT_ID_CB_TRIG            0

/* Size of opaque payload sent to kernel for safekeeping until signal time */
#define CAM_SYNC_USER_PAYLOAD_SIZE               2

/* Device type for sync device needed for device discovery */
#define CAM_SYNC_DEVICE_TYPE                     (MEDIA_ENT_F_OLD_BASE)

#define CAM_SYNC_GET_PAYLOAD_PTR(ev, type)       \
	(type *)((char *)ev.u.data + sizeof(struct cam_sync_ev_header))

#define CAM_SYNC_GET_HEADER_PTR(ev)              \
	((struct cam_sync_ev_header *)ev.u.data)

#define CAM_SYNC_STATE_INVALID                   0
#define CAM_SYNC_STATE_ACTIVE                    1
#define CAM_SYNC_STATE_SIGNALED_SUCCESS          2
#define CAM_SYNC_STATE_SIGNALED_ERROR            3

/**
 * struct cam_sync_ev_header - Event header for sync event notification
 *
 * @sync_obj: Sync object
 * @status:   Status of the object
 */
struct cam_sync_ev_header {
	int32_t sync_obj;
	int32_t status;
};

/**
 * struct cam_sync_info - Sync object creation information
 *
 * @name:       Optional string representation of the sync object
 * @sync_obj:   Sync object returned after creation in kernel
 */
struct cam_sync_info {
	char name[64];
	int32_t sync_obj;
};

/**
 * struct cam_sync_signal - Sync object signaling struct
 *
 * @sync_obj:   Sync object to be signaled
 * @sync_state: State of the sync object to which it should be signaled
 */
struct cam_sync_signal {
	int32_t sync_obj;
	uint32_t sync_state;
};

/**
 * struct cam_sync_merge - Merge information for sync objects
 *
 * @sync_objs:  Pointer to sync objects
 * @num_objs:   Number of objects in the array
 * @merged:     Merged sync object
 */
struct cam_sync_merge {
	__u64 sync_objs;
	uint32_t num_objs;
	int32_t merged;
};

/**
 * struct cam_sync_userpayload_info - Payload info from user space
 *
 * @sync_obj:   Sync object for which payload has to be registered for
 * @reserved:   Reserved
 * @payload:    Pointer to user payload
 */
struct cam_sync_userpayload_info {
	int32_t sync_obj;
	uint32_t reserved;
	__u64 payload[CAM_SYNC_USER_PAYLOAD_SIZE];
};

/**
 * struct cam_sync_wait - Sync object wait information
 *
 * @sync_obj:   Sync object to wait on
 * @reserved:   Reserved
 * @timeout_ms: Timeout in milliseconds
 */
struct cam_sync_wait {
	int32_t sync_obj;
	uint32_t reserved;
	uint64_t timeout_ms;
};

/**
 * struct cam_private_ioctl_arg - Sync driver ioctl argument
 *
 * @id:         IOCTL command id
 * @size:       Size of command payload
 * @result:     Result of command execution
 * @reserved:   Reserved
 * @ioctl_ptr:  Pointer to user data
 */
struct cam_private_ioctl_arg {
	__u32 id;
	__u32 size;
	__u32 result;
	__u32 reserved;
	__u64 ioctl_ptr;
};

#define CAM_PRIVATE_IOCTL_CMD \
	_IOWR('V', BASE_VIDIOC_PRIVATE, struct cam_private_ioctl_arg)

#define CAM_SYNC_CREATE                          0
#define CAM_SYNC_DESTROY                         1
#define CAM_SYNC_SIGNAL                          2
#define CAM_SYNC_MERGE                           3
#define CAM_SYNC_REGISTER_PAYLOAD                4
#define CAM_SYNC_DEREGISTER_PAYLOAD              5
#define CAM_SYNC_WAIT                            6

#endif /* __UAPI_CAM_SYNC_H__ */
