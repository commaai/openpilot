#ifndef __UAPI_LINUX_CAM_REQ_MGR_H
#define __UAPI_LINUX_CAM_REQ_MGR_H

#include <linux/videodev2.h>
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/media.h>
#include <media/cam_defs.h>

#define CAM_REQ_MGR_VNODE_NAME "cam-req-mgr-devnode"

#define CAM_DEVICE_TYPE_BASE      (MEDIA_ENT_F_OLD_BASE)
#define CAM_VNODE_DEVICE_TYPE     (CAM_DEVICE_TYPE_BASE)
#define CAM_SENSOR_DEVICE_TYPE    (CAM_DEVICE_TYPE_BASE + 1)
#define CAM_IFE_DEVICE_TYPE       (CAM_DEVICE_TYPE_BASE + 2)
#define CAM_ICP_DEVICE_TYPE       (CAM_DEVICE_TYPE_BASE + 3)
#define CAM_LRME_DEVICE_TYPE      (CAM_DEVICE_TYPE_BASE + 4)
#define CAM_JPEG_DEVICE_TYPE      (CAM_DEVICE_TYPE_BASE + 5)
#define CAM_FD_DEVICE_TYPE        (CAM_DEVICE_TYPE_BASE + 6)
#define CAM_CPAS_DEVICE_TYPE      (CAM_DEVICE_TYPE_BASE + 7)
#define CAM_CSIPHY_DEVICE_TYPE    (CAM_DEVICE_TYPE_BASE + 8)
#define CAM_ACTUATOR_DEVICE_TYPE  (CAM_DEVICE_TYPE_BASE + 9)
#define CAM_CCI_DEVICE_TYPE       (CAM_DEVICE_TYPE_BASE + 10)
#define CAM_FLASH_DEVICE_TYPE     (CAM_DEVICE_TYPE_BASE + 11)
#define CAM_EEPROM_DEVICE_TYPE    (CAM_DEVICE_TYPE_BASE + 12)
#define CAM_OIS_DEVICE_TYPE       (CAM_DEVICE_TYPE_BASE + 13)

/* cam_req_mgr hdl info */
#define CAM_REQ_MGR_HDL_IDX_POS           8
#define CAM_REQ_MGR_HDL_IDX_MASK          ((1 << CAM_REQ_MGR_HDL_IDX_POS) - 1)
#define CAM_REQ_MGR_GET_HDL_IDX(hdl)      (hdl & CAM_REQ_MGR_HDL_IDX_MASK)

/**
 * Max handles supported by cam_req_mgr
 * It includes both session and device handles
 */
#define CAM_REQ_MGR_MAX_HANDLES           64
#define MAX_LINKS_PER_SESSION             2

/* V4L event type which user space will subscribe to */
#define V4L_EVENT_CAM_REQ_MGR_EVENT       (V4L2_EVENT_PRIVATE_START + 0)

/* Specific event ids to get notified in user space */
#define V4L_EVENT_CAM_REQ_MGR_SOF            0
#define V4L_EVENT_CAM_REQ_MGR_ERROR          1
#define V4L_EVENT_CAM_REQ_MGR_SOF_BOOT_TS    2

/* SOF Event status */
#define CAM_REQ_MGR_SOF_EVENT_SUCCESS           0
#define CAM_REQ_MGR_SOF_EVENT_ERROR             1

/* Link control operations */
#define CAM_REQ_MGR_LINK_ACTIVATE               0
#define CAM_REQ_MGR_LINK_DEACTIVATE             1

/**
 * Request Manager : flush_type
 * @CAM_REQ_MGR_FLUSH_TYPE_ALL: Req mgr will remove all the pending
 * requests from input/processing queue.
 * @CAM_REQ_MGR_FLUSH_TYPE_CANCEL_REQ: Req mgr will remove only particular
 * request id from input/processing queue.
 * @CAM_REQ_MGR_FLUSH_TYPE_MAX: Max number of the flush type
 * @opcode: CAM_REQ_MGR_FLUSH_REQ
 */
#define CAM_REQ_MGR_FLUSH_TYPE_ALL          0
#define CAM_REQ_MGR_FLUSH_TYPE_CANCEL_REQ   1
#define CAM_REQ_MGR_FLUSH_TYPE_MAX          2

/**
 * Request Manager : Sync Mode type
 * @CAM_REQ_MGR_SYNC_MODE_NO_SYNC: Req mgr will apply non-sync mode for this
 * request.
 * @CAM_REQ_MGR_SYNC_MODE_SYNC: Req mgr will apply sync mode for this request.
 */
#define CAM_REQ_MGR_SYNC_MODE_NO_SYNC   0
#define CAM_REQ_MGR_SYNC_MODE_SYNC      1

/**
 * struct cam_req_mgr_event_data
 * @session_hdl: session handle
 * @link_hdl: link handle
 * @frame_id: frame id
 * @reserved: reserved for 64 bit aligngment
 * @req_id: request id
 * @tv_sec: timestamp in seconds
 * @tv_usec: timestamp in micro seconds
 */
struct cam_req_mgr_event_data {
	int32_t   session_hdl;
	int32_t   link_hdl;
	int32_t   frame_id;
	int32_t   reserved;
	int64_t   req_id;
	uint64_t  tv_sec;
	uint64_t  tv_usec;
};

/**
 * struct cam_req_mgr_session_info
 * @session_hdl: In/Output param - session_handle
 * @opcode1: CAM_REQ_MGR_CREATE_SESSION
 * @opcode2: CAM_REQ_MGR_DESTROY_SESSION
 */
struct cam_req_mgr_session_info {
	int32_t session_hdl;
	int32_t reserved;
};

/**
 * struct cam_req_mgr_link_info
 * @session_hdl: Input param - Identifier for CSL session
 * @num_devices: Input Param - Num of devices to be linked
 * @dev_hdls: Input param - List of device handles to be linked
 * @link_hdl: Output Param -Identifier for link
 * @opcode: CAM_REQ_MGR_LINK
 */
struct cam_req_mgr_link_info {
	int32_t session_hdl;
	uint32_t num_devices;
	int32_t dev_hdls[CAM_REQ_MGR_MAX_HANDLES];
	int32_t link_hdl;
};

/**
 * struct cam_req_mgr_unlink_info
 * @session_hdl: input param - session handle
 * @link_hdl: input param - link handle
 * @opcode: CAM_REQ_MGR_UNLINK
 */
struct cam_req_mgr_unlink_info {
	int32_t session_hdl;
	int32_t link_hdl;
};

/**
 * struct cam_req_mgr_flush_info
 * @brief: User can tell drivers to flush a particular request id or
 * flush all requests from its pending processing queue. Flush is a
 * blocking call and driver shall ensure all requests are flushed
 * before returning.
 * @session_hdl: Input param - Identifier for CSL session
 * @link_hdl: Input Param -Identifier for link
 * @flush_type: User can cancel a particular req id or can flush
 * all requests in queue
 * @reserved: reserved for 64 bit aligngment
 * @req_id: field is valid only if flush type is cancel request
 * for flush all this field value is not considered.
 * @opcode: CAM_REQ_MGR_FLUSH_REQ
 */
struct cam_req_mgr_flush_info {
	int32_t session_hdl;
	int32_t link_hdl;
	uint32_t flush_type;
	uint32_t reserved;
	int64_t req_id;
};

/** struct cam_req_mgr_sched_info
 * @session_hdl: Input param - Identifier for CSL session
 * @link_hdl: Input Param -Identifier for link
 * inluding itself.
 * @bubble_enable: Input Param - Cam req mgr will do bubble recovery if this
 * flag is set.
 * @sync_mode: Type of Sync mode for this request
 * @req_id: Input Param - Request Id from which all requests will be flushed
 */
struct cam_req_mgr_sched_request {
	int32_t session_hdl;
	int32_t link_hdl;
	int32_t bubble_enable;
	int32_t sync_mode;
	int64_t req_id;
};

/**
 * struct cam_req_mgr_sync_mode
 * @session_hdl:         Input param - Identifier for CSL session
 * @sync_mode:           Input Param - Type of sync mode
 * @num_links:           Input Param - Num of links in sync mode (Valid only
 *                             when sync_mode is one of SYNC enabled modes)
 * @link_hdls:           Input Param - Array of link handles to be in sync mode
 *                             (Valid only when sync_mode is one of SYNC
 *                             enabled modes)
 * @master_link_hdl:     Input Param - To dictate which link's SOF drives system
 *                             (Valid only when sync_mode is one of SYNC
 *                             enabled modes)
 *
 * @opcode: CAM_REQ_MGR_SYNC_MODE
 */
struct cam_req_mgr_sync_mode {
	int32_t session_hdl;
	int32_t sync_mode;
	int32_t num_links;
	int32_t link_hdls[MAX_LINKS_PER_SESSION];
	int32_t master_link_hdl;
	int32_t reserved;
};

/**
 * struct cam_req_mgr_link_control
 * @ops:                 Link operations: activate/deactive
 * @session_hdl:         Input param - Identifier for CSL session
 * @num_links:           Input Param - Num of links
 * @reserved:            reserved field
 * @link_hdls:           Input Param - Links to be activated/deactivated
 *
 * @opcode: CAM_REQ_MGR_LINK_CONTROL
 */
struct cam_req_mgr_link_control {
	int32_t ops;
	int32_t session_hdl;
	int32_t num_links;
	int32_t reserved;
	int32_t link_hdls[MAX_LINKS_PER_SESSION];
};

/**
 * cam_req_mgr specific opcode ids
 */
#define CAM_REQ_MGR_CREATE_DEV_NODES            (CAM_COMMON_OPCODE_MAX + 1)
#define CAM_REQ_MGR_CREATE_SESSION              (CAM_COMMON_OPCODE_MAX + 2)
#define CAM_REQ_MGR_DESTROY_SESSION             (CAM_COMMON_OPCODE_MAX + 3)
#define CAM_REQ_MGR_LINK                        (CAM_COMMON_OPCODE_MAX + 4)
#define CAM_REQ_MGR_UNLINK                      (CAM_COMMON_OPCODE_MAX + 5)
#define CAM_REQ_MGR_SCHED_REQ                   (CAM_COMMON_OPCODE_MAX + 6)
#define CAM_REQ_MGR_FLUSH_REQ                   (CAM_COMMON_OPCODE_MAX + 7)
#define CAM_REQ_MGR_SYNC_MODE                   (CAM_COMMON_OPCODE_MAX + 8)
#define CAM_REQ_MGR_ALLOC_BUF                   (CAM_COMMON_OPCODE_MAX + 9)
#define CAM_REQ_MGR_MAP_BUF                     (CAM_COMMON_OPCODE_MAX + 10)
#define CAM_REQ_MGR_RELEASE_BUF                 (CAM_COMMON_OPCODE_MAX + 11)
#define CAM_REQ_MGR_CACHE_OPS                   (CAM_COMMON_OPCODE_MAX + 12)
#define CAM_REQ_MGR_LINK_CONTROL                (CAM_COMMON_OPCODE_MAX + 13)
/* end of cam_req_mgr opcodes */

#define CAM_MEM_FLAG_HW_READ_WRITE              (1<<0)
#define CAM_MEM_FLAG_HW_READ_ONLY               (1<<1)
#define CAM_MEM_FLAG_HW_WRITE_ONLY              (1<<2)
#define CAM_MEM_FLAG_KMD_ACCESS                 (1<<3)
#define CAM_MEM_FLAG_UMD_ACCESS                 (1<<4)
#define CAM_MEM_FLAG_PROTECTED_MODE             (1<<5)
#define CAM_MEM_FLAG_CMD_BUF_TYPE               (1<<6)
#define CAM_MEM_FLAG_PIXEL_BUF_TYPE             (1<<7)
#define CAM_MEM_FLAG_STATS_BUF_TYPE             (1<<8)
#define CAM_MEM_FLAG_PACKET_BUF_TYPE            (1<<9)
#define CAM_MEM_FLAG_CACHE                      (1<<10)
#define CAM_MEM_FLAG_HW_SHARED_ACCESS           (1<<11)

#define CAM_MEM_MMU_MAX_HANDLE                  16

/* Maximum allowed buffers in existence */
#define CAM_MEM_BUFQ_MAX                        1024

#define CAM_MEM_MGR_SECURE_BIT_POS              15
#define CAM_MEM_MGR_HDL_IDX_SIZE                15
#define CAM_MEM_MGR_HDL_FD_SIZE                 16
#define CAM_MEM_MGR_HDL_IDX_END_POS             16
#define CAM_MEM_MGR_HDL_FD_END_POS              32

#define CAM_MEM_MGR_HDL_IDX_MASK      ((1 << CAM_MEM_MGR_HDL_IDX_SIZE) - 1)

#define GET_MEM_HANDLE(idx, fd) \
	((idx & CAM_MEM_MGR_HDL_IDX_MASK) | \
	(fd << (CAM_MEM_MGR_HDL_FD_END_POS - CAM_MEM_MGR_HDL_FD_SIZE))) \

#define GET_FD_FROM_HANDLE(hdl) \
	(hdl >> (CAM_MEM_MGR_HDL_FD_END_POS - CAM_MEM_MGR_HDL_FD_SIZE)) \

#define CAM_MEM_MGR_GET_HDL_IDX(hdl) (hdl & CAM_MEM_MGR_HDL_IDX_MASK)

#define CAM_MEM_MGR_SET_SECURE_HDL(hdl, flag) \
	((flag) ? (hdl |= (1 << CAM_MEM_MGR_SECURE_BIT_POS)) : \
	((hdl) &= ~(1 << CAM_MEM_MGR_SECURE_BIT_POS)))

#define CAM_MEM_MGR_IS_SECURE_HDL(hdl) \
	(((hdl) & \
	(1<<CAM_MEM_MGR_SECURE_BIT_POS)) >> CAM_MEM_MGR_SECURE_BIT_POS)

/**
 * memory allocation type
 */
#define CAM_MEM_DMA_NONE                        0
#define CAM_MEM_DMA_BIDIRECTIONAL               1
#define CAM_MEM_DMA_TO_DEVICE                   2
#define CAM_MEM_DMA_FROM_DEVICE                 3


/**
 * memory cache operation
 */
#define CAM_MEM_CLEAN_CACHE                     1
#define CAM_MEM_INV_CACHE                       2
#define CAM_MEM_CLEAN_INV_CACHE                 3


/**
 * struct cam_mem_alloc_out_params
 * @buf_handle: buffer handle
 * @fd: output buffer file descriptor
 * @vaddr: virtual address pointer
 */
struct cam_mem_alloc_out_params {
	uint32_t buf_handle;
	int32_t fd;
	uint64_t vaddr;
};

/**
 * struct cam_mem_map_out_params
 * @buf_handle: buffer handle
 * @reserved: reserved for future
 * @vaddr: virtual address pointer
 */
struct cam_mem_map_out_params {
	uint32_t buf_handle;
	uint32_t reserved;
	uint64_t vaddr;
};

/**
 * struct cam_mem_mgr_alloc_cmd
 * @len: size of buffer to allocate
 * @align: alignment of the buffer
 * @mmu_hdls: array of mmu handles
 * @num_hdl: number of handles
 * @flags: flags of the buffer
 * @out: out params
 */
/* CAM_REQ_MGR_ALLOC_BUF */
struct cam_mem_mgr_alloc_cmd {
	uint64_t len;
	uint64_t align;
	int32_t mmu_hdls[CAM_MEM_MMU_MAX_HANDLE];
	uint32_t num_hdl;
	uint32_t flags;
	struct cam_mem_alloc_out_params out;
};

/**
 * struct cam_mem_mgr_map_cmd
 * @mmu_hdls: array of mmu handles
 * @num_hdl: number of handles
 * @flags: flags of the buffer
 * @fd: output buffer file descriptor
 * @reserved: reserved field
 * @out: out params
 */

/* CAM_REQ_MGR_MAP_BUF */
struct cam_mem_mgr_map_cmd {
	int32_t mmu_hdls[CAM_MEM_MMU_MAX_HANDLE];
	uint32_t num_hdl;
	uint32_t flags;
	int32_t fd;
	uint32_t reserved;
	struct cam_mem_map_out_params out;
};

/**
 * struct cam_mem_mgr_map_cmd
 * @buf_handle: buffer handle
 * @reserved: reserved field
 */
/* CAM_REQ_MGR_RELEASE_BUF */
struct cam_mem_mgr_release_cmd {
	int32_t buf_handle;
	uint32_t reserved;
};

/**
 * struct cam_mem_mgr_map_cmd
 * @buf_handle: buffer handle
 * @ops: cache operations
 */
/* CAM_REQ_MGR_CACHE_OPS */
struct cam_mem_cache_ops_cmd {
	int32_t buf_handle;
	uint32_t mem_cache_ops;
};

/**
 * Request Manager : error message type
 * @CAM_REQ_MGR_ERROR_TYPE_DEVICE: Device error message, fatal to session
 * @CAM_REQ_MGR_ERROR_TYPE_REQUEST: Error on a single request, not fatal
 * @CAM_REQ_MGR_ERROR_TYPE_BUFFER: Buffer was not filled, not fatal
 */
#define CAM_REQ_MGR_ERROR_TYPE_DEVICE           0
#define CAM_REQ_MGR_ERROR_TYPE_REQUEST          1
#define CAM_REQ_MGR_ERROR_TYPE_BUFFER           2

/**
 * struct cam_req_mgr_error_msg
 * @error_type: type of error
 * @request_id: request id of frame
 * @device_hdl: device handle
 * @linke_hdl: link_hdl
 * @resource_size: size of the resource
 */
struct cam_req_mgr_error_msg {
	uint32_t error_type;
	uint32_t request_id;
	int32_t device_hdl;
	int32_t link_hdl;
	uint64_t resource_size;
};

/**
 * struct cam_req_mgr_frame_msg
 * @request_id: request id of the frame
 * @frame_id: frame id of the frame
 * @timestamp: timestamp of the frame
 * @link_hdl: link handle associated with this message
 * @sof_status: sof status success or fail
 */
struct cam_req_mgr_frame_msg {
	uint64_t request_id;
	uint64_t frame_id;
	uint64_t timestamp;
	int32_t  link_hdl;
	uint32_t sof_status;
};

/**
 * struct cam_req_mgr_message
 * @session_hdl: session to which the frame belongs to
 * @reserved: reserved field
 * @u: union which can either be error or frame message
 */
struct cam_req_mgr_message {
	int32_t session_hdl;
	int32_t reserved;
	union {
		struct cam_req_mgr_error_msg err_msg;
		struct cam_req_mgr_frame_msg frame_msg;
	} u;
};
#endif /* __UAPI_LINUX_CAM_REQ_MGR_H */
