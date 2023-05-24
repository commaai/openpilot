#ifndef __UAPI_CAM_DEFS_H__
#define __UAPI_CAM_DEFS_H__

#include <linux/videodev2.h>
#include <linux/types.h>
#include <linux/ioctl.h>


/* camera op codes */
#define CAM_COMMON_OPCODE_BASE                  0x100
#define CAM_QUERY_CAP                           (CAM_COMMON_OPCODE_BASE + 0x1)
#define CAM_ACQUIRE_DEV                         (CAM_COMMON_OPCODE_BASE + 0x2)
#define CAM_START_DEV                           (CAM_COMMON_OPCODE_BASE + 0x3)
#define CAM_STOP_DEV                            (CAM_COMMON_OPCODE_BASE + 0x4)
#define CAM_CONFIG_DEV                          (CAM_COMMON_OPCODE_BASE + 0x5)
#define CAM_RELEASE_DEV                         (CAM_COMMON_OPCODE_BASE + 0x6)
#define CAM_SD_SHUTDOWN                         (CAM_COMMON_OPCODE_BASE + 0x7)
#define CAM_FLUSH_REQ                           (CAM_COMMON_OPCODE_BASE + 0x8)
#define CAM_COMMON_OPCODE_MAX                   (CAM_COMMON_OPCODE_BASE + 0x9)

#define CAM_EXT_OPCODE_BASE                     0x200
#define CAM_CONFIG_DEV_EXTERNAL                 (CAM_EXT_OPCODE_BASE + 0x1)

/* camera handle type */
#define CAM_HANDLE_USER_POINTER                 1
#define CAM_HANDLE_MEM_HANDLE                   2

/* Generic Blob CmdBuffer header properties */
#define CAM_GENERIC_BLOB_CMDBUFFER_SIZE_MASK    0xFFFFFF00
#define CAM_GENERIC_BLOB_CMDBUFFER_SIZE_SHIFT   8
#define CAM_GENERIC_BLOB_CMDBUFFER_TYPE_MASK    0xFF
#define CAM_GENERIC_BLOB_CMDBUFFER_TYPE_SHIFT   0

/* Command Buffer Types */
#define CAM_CMD_BUF_DMI                     0x1
#define CAM_CMD_BUF_DMI16                   0x2
#define CAM_CMD_BUF_DMI32                   0x3
#define CAM_CMD_BUF_DMI64                   0x4
#define CAM_CMD_BUF_DIRECT                  0x5
#define CAM_CMD_BUF_INDIRECT                0x6
#define CAM_CMD_BUF_I2C                     0x7
#define CAM_CMD_BUF_FW                      0x8
#define CAM_CMD_BUF_GENERIC                 0x9
#define CAM_CMD_BUF_LEGACY                  0xA

/**
 * enum flush_type_t - Identifies the various flush types
 *
 * @CAM_FLUSH_TYPE_REQ:    Flush specific request
 * @CAM_FLUSH_TYPE_ALL:    Flush all requests belonging to a context
 * @CAM_FLUSH_TYPE_MAX:    Max enum to validate flush type
 *
 */
enum flush_type_t {
	CAM_FLUSH_TYPE_REQ,
	CAM_FLUSH_TYPE_ALL,
	CAM_FLUSH_TYPE_MAX
};

/**
 * struct cam_control - Structure used by ioctl control for camera
 *
 * @op_code:            This is the op code for camera control
 * @size:               Control command size
 * @handle_type:        User pointer or shared memory handle
 * @reserved:           Reserved field for 64 bit alignment
 * @handle:             Control command payload
 */
struct cam_control {
	uint32_t        op_code;
	uint32_t        size;
	uint32_t        handle_type;
	uint32_t        reserved;
	uint64_t        handle;
};

/* camera IOCTL */
#define VIDIOC_CAM_CONTROL \
	_IOWR('V', BASE_VIDIOC_PRIVATE, struct cam_control)

/**
 * struct cam_hw_version - Structure for HW version of camera devices
 *
 * @major    : Hardware version major
 * @minor    : Hardware version minor
 * @incr     : Hardware version increment
 * @reserved : Reserved for 64 bit aligngment
 */
struct cam_hw_version {
	uint32_t major;
	uint32_t minor;
	uint32_t incr;
	uint32_t reserved;
};

/**
 * struct cam_iommu_handle - Structure for IOMMU handles of camera hw devices
 *
 * @non_secure: Device Non Secure IOMMU handle
 * @secure:     Device Secure IOMMU handle
 *
 */
struct cam_iommu_handle {
	int32_t non_secure;
	int32_t secure;
};

/* camera secure mode */
#define CAM_SECURE_MODE_NON_SECURE             0
#define CAM_SECURE_MODE_SECURE                 1

/* Camera Format Type */
#define CAM_FORMAT_BASE                         0
#define CAM_FORMAT_MIPI_RAW_6                   1
#define CAM_FORMAT_MIPI_RAW_8                   2
#define CAM_FORMAT_MIPI_RAW_10                  3
#define CAM_FORMAT_MIPI_RAW_12                  4
#define CAM_FORMAT_MIPI_RAW_14                  5
#define CAM_FORMAT_MIPI_RAW_16                  6
#define CAM_FORMAT_MIPI_RAW_20                  7
#define CAM_FORMAT_QTI_RAW_8                    8
#define CAM_FORMAT_QTI_RAW_10                   9
#define CAM_FORMAT_QTI_RAW_12                   10
#define CAM_FORMAT_QTI_RAW_14                   11
#define CAM_FORMAT_PLAIN8                       12
#define CAM_FORMAT_PLAIN16_8                    13
#define CAM_FORMAT_PLAIN16_10                   14
#define CAM_FORMAT_PLAIN16_12                   15
#define CAM_FORMAT_PLAIN16_14                   16
#define CAM_FORMAT_PLAIN16_16                   17
#define CAM_FORMAT_PLAIN32_20                   18
#define CAM_FORMAT_PLAIN64                      19
#define CAM_FORMAT_PLAIN128                     20
#define CAM_FORMAT_ARGB                         21
#define CAM_FORMAT_ARGB_10                      22
#define CAM_FORMAT_ARGB_12                      23
#define CAM_FORMAT_ARGB_14                      24
#define CAM_FORMAT_DPCM_10_6_10                 25
#define CAM_FORMAT_DPCM_10_8_10                 26
#define CAM_FORMAT_DPCM_12_6_12                 27
#define CAM_FORMAT_DPCM_12_8_12                 28
#define CAM_FORMAT_DPCM_14_8_14                 29
#define CAM_FORMAT_DPCM_14_10_14                30
#define CAM_FORMAT_NV21                         31
#define CAM_FORMAT_NV12                         32
#define CAM_FORMAT_TP10                         33
#define CAM_FORMAT_YUV422                       34
#define CAM_FORMAT_PD8                          35
#define CAM_FORMAT_PD10                         36
#define CAM_FORMAT_UBWC_NV12                    37
#define CAM_FORMAT_UBWC_NV12_4R                 38
#define CAM_FORMAT_UBWC_TP10                    39
#define CAM_FORMAT_UBWC_P010                    40
#define CAM_FORMAT_PLAIN8_SWAP                  41
#define CAM_FORMAT_PLAIN8_10                    42
#define CAM_FORMAT_PLAIN8_10_SWAP               43
#define CAM_FORMAT_YV12                         44
#define CAM_FORMAT_Y_ONLY                       45
#define CAM_FORMAT_MAX                          46

/* camera rotaion */
#define CAM_ROTATE_CW_0_DEGREE                  0
#define CAM_ROTATE_CW_90_DEGREE                 1
#define CAM_RORATE_CW_180_DEGREE                2
#define CAM_ROTATE_CW_270_DEGREE                3

/* camera Color Space */
#define CAM_COLOR_SPACE_BASE                    0
#define CAM_COLOR_SPACE_BT601_FULL              1
#define CAM_COLOR_SPACE_BT601625                2
#define CAM_COLOR_SPACE_BT601525                3
#define CAM_COLOR_SPACE_BT709                   4
#define CAM_COLOR_SPACE_DEPTH                   5
#define CAM_COLOR_SPACE_MAX                     6

/* camera buffer direction */
#define CAM_BUF_INPUT                           1
#define CAM_BUF_OUTPUT                          2
#define CAM_BUF_IN_OUT                          3

/* camera packet device Type */
#define CAM_PACKET_DEV_BASE                     0
#define CAM_PACKET_DEV_IMG_SENSOR               1
#define CAM_PACKET_DEV_ACTUATOR                 2
#define CAM_PACKET_DEV_COMPANION                3
#define CAM_PACKET_DEV_EEPOM                    4
#define CAM_PACKET_DEV_CSIPHY                   5
#define CAM_PACKET_DEV_OIS                      6
#define CAM_PACKET_DEV_FLASH                    7
#define CAM_PACKET_DEV_FD                       8
#define CAM_PACKET_DEV_JPEG_ENC                 9
#define CAM_PACKET_DEV_JPEG_DEC                 10
#define CAM_PACKET_DEV_VFE                      11
#define CAM_PACKET_DEV_CPP                      12
#define CAM_PACKET_DEV_CSID                     13
#define CAM_PACKET_DEV_ISPIF                    14
#define CAM_PACKET_DEV_IFE                      15
#define CAM_PACKET_DEV_ICP                      16
#define CAM_PACKET_DEV_LRME                     17
#define CAM_PACKET_DEV_MAX                      18


/* constants */
#define CAM_PACKET_MAX_PLANES                   3

/**
 * struct cam_plane_cfg - Plane configuration info
 *
 * @width:                      Plane width in pixels
 * @height:                     Plane height in lines
 * @plane_stride:               Plane stride in pixel
 * @slice_height:               Slice height in line (not used by ISP)
 * @meta_stride:                UBWC metadata stride
 * @meta_size:                  UBWC metadata plane size
 * @meta_offset:                UBWC metadata offset
 * @packer_config:              UBWC packer config
 * @mode_config:                UBWC mode config
 * @tile_config:                UBWC tile config
 * @h_init:                     UBWC horizontal initial coordinate in pixels
 * @v_init:                     UBWC vertical initial coordinate in lines
 *
 */
struct cam_plane_cfg {
	uint32_t                width;
	uint32_t                height;
	uint32_t                plane_stride;
	uint32_t                slice_height;
	uint32_t                meta_stride;
	uint32_t                meta_size;
	uint32_t                meta_offset;
	uint32_t                packer_config;
	uint32_t                mode_config;
	uint32_t                tile_config;
	uint32_t                h_init;
	uint32_t                v_init;
};

/**
 * struct cam_cmd_buf_desc - Command buffer descriptor
 *
 * @mem_handle:                 Command buffer handle
 * @offset:                     Command start offset
 * @size:                       Size of the command buffer in bytes
 * @length:                     Used memory in command buffer in bytes
 * @type:                       Type of the command buffer
 * @meta_data:                  Data type for private command buffer
 *                              Between UMD and KMD
 *
 */
struct cam_cmd_buf_desc {
	int32_t                 mem_handle;
	uint32_t                offset;
	uint32_t                size;
	uint32_t                length;
	uint32_t                type;
	uint32_t                meta_data;
};

/**
 * struct cam_buf_io_cfg - Buffer io configuration for buffers
 *
 * @mem_handle:                 Mem_handle array for the buffers.
 * @offsets:                    Offsets for each planes in the buffer
 * @planes:                     Per plane information
 * @width:                      Main plane width in pixel
 * @height:                     Main plane height in lines
 * @format:                     Format of the buffer
 * @color_space:                Color space for the buffer
 * @color_pattern:              Color pattern in the buffer
 * @bpp:                        Bit per pixel
 * @rotation:                   Rotation information for the buffer
 * @resource_type:              Resource type associated with the buffer
 * @fence:                      Fence handle
 * @early_fence:                Fence handle for early signal
 * @aux_cmd_buf:                An auxiliary command buffer that may be
 *                              used for programming the IO
 * @direction:                  Direction of the config
 * @batch_size:                 Batch size in HFR mode
 * @subsample_pattern:          Subsample pattern. Used in HFR mode. It
 *                              should be consistent with batchSize and
 *                              CAMIF programming.
 * @subsample_period:           Subsample period. Used in HFR mode. It
 *                              should be consistent with batchSize and
 *                              CAMIF programming.
 * @framedrop_pattern:          Framedrop pattern
 * @framedrop_period:           Framedrop period
 * @flag:                       Flags for extra information
 * @direction:                  Buffer direction: input or output
 * @padding:                    Padding for the structure
 *
 */
struct cam_buf_io_cfg {
	int32_t                         mem_handle[CAM_PACKET_MAX_PLANES];
	uint32_t                        offsets[CAM_PACKET_MAX_PLANES];
	struct cam_plane_cfg            planes[CAM_PACKET_MAX_PLANES];
	uint32_t                        format;
	uint32_t                        color_space;
	uint32_t                        color_pattern;
	uint32_t                        bpp;
	uint32_t                        rotation;
	uint32_t                        resource_type;
	int32_t                         fence;
	int32_t                         early_fence;
	struct cam_cmd_buf_desc         aux_cmd_buf;
	uint32_t                        direction;
	uint32_t                        batch_size;
	uint32_t                        subsample_pattern;
	uint32_t                        subsample_period;
	uint32_t                        framedrop_pattern;
	uint32_t                        framedrop_period;
	uint32_t                        flag;
	uint32_t                        padding;
};

/**
 * struct cam_packet_header - Camera packet header
 *
 * @op_code:                    Camera packet opcode
 * @size:                       Size of the camera packet in bytes
 * @request_id:                 Request id for this camera packet
 * @flags:                      Flags for the camera packet
 * @padding:                    Padding
 *
 */
struct cam_packet_header {
	uint32_t                op_code;
	uint32_t                size;
	uint64_t                request_id;
	uint32_t                flags;
	uint32_t                padding;
};

/**
 * struct cam_patch_desc - Patch structure
 *
 * @dst_buf_hdl:                Memory handle for the dest buffer
 * @dst_offset:                 Offset byte in the dest buffer
 * @src_buf_hdl:                Memory handle for the source buffer
 * @src_offset:                 Offset byte in the source buffer
 *
 */
struct cam_patch_desc {
	int32_t                 dst_buf_hdl;
	uint32_t                dst_offset;
	int32_t                 src_buf_hdl;
	uint32_t                src_offset;
};

/**
 * struct cam_packet - Camera packet structure
 *
 * @header:                     Camera packet header
 * @cmd_buf_offset:             Command buffer start offset
 * @num_cmd_buf:                Number of the command buffer in the packet
 * @io_config_offset:           Buffer io configuration start offset
 * @num_io_configs:             Number of the buffer io configurations
 * @patch_offset:               Patch offset for the patch structure
 * @num_patches:                Number of the patch structure
 * @kmd_cmd_buf_index:          Command buffer index which contains extra
 *                              space for the KMD buffer
 * @kmd_cmd_buf_offset:         Offset from the beginning of the command
 *                              buffer for KMD usage.
 * @payload:                    Camera packet payload
 *
 */
struct cam_packet {
	struct cam_packet_header        header;
	uint32_t                        cmd_buf_offset;
	uint32_t                        num_cmd_buf;
	uint32_t                        io_configs_offset;
	uint32_t                        num_io_configs;
	uint32_t                        patch_offset;
	uint32_t                        num_patches;
	uint32_t                        kmd_cmd_buf_index;
	uint32_t                        kmd_cmd_buf_offset;
	uint64_t                        payload[1];

};

/**
 * struct cam_release_dev_cmd - Control payload for release devices
 *
 * @session_handle:             Session handle for the release
 * @dev_handle:                 Device handle for the release
 */
struct cam_release_dev_cmd {
	int32_t                 session_handle;
	int32_t                 dev_handle;
};

/**
 * struct cam_start_stop_dev_cmd - Control payload for start/stop device
 *
 * @session_handle:             Session handle for the start/stop command
 * @dev_handle:                 Device handle for the start/stop command
 *
 */
struct cam_start_stop_dev_cmd {
	int32_t                 session_handle;
	int32_t                 dev_handle;
};

/**
 * struct cam_config_dev_cmd - Command payload for configure device
 *
 * @session_handle:             Session handle for the command
 * @dev_handle:                 Device handle for the command
 * @offset:                     Offset byte in the packet handle.
 * @packet_handle:              Packet memory handle for the actual packet:
 *                              struct cam_packet.
 *
 */
struct cam_config_dev_cmd {
	int32_t                 session_handle;
	int32_t                 dev_handle;
	uint64_t                offset;
	uint64_t                packet_handle;
};

/**
 * struct cam_query_cap_cmd - Payload for query device capability
 *
 * @size:               Handle size
 * @handle_type:        User pointer or shared memory handle
 * @caps_handle:        Device specific query command payload
 *
 */
struct cam_query_cap_cmd {
	uint32_t        size;
	uint32_t        handle_type;
	uint64_t        caps_handle;
};

/**
 * struct cam_acquire_dev_cmd - Control payload for acquire devices
 *
 * @session_handle:     Session handle for the acquire command
 * @dev_handle:         Device handle to be returned
 * @handle_type:        Resource handle type:
 *                      1 = user pointer, 2 = mem handle
 * @num_resources:      Number of the resources to be acquired
 * @resources_hdl:      Resource handle that refers to the actual
 *                      resource array. Each item in this
 *                      array is device specific resource structure
 *
 */
struct cam_acquire_dev_cmd {
	int32_t         session_handle;
	int32_t         dev_handle;
	uint32_t        handle_type;
	uint32_t        num_resources;
	uint64_t        resource_hdl;
};

/**
 * struct cam_flush_dev_cmd - Control payload for flush devices
 *
 * @version:           Version
 * @session_handle:    Session handle for the acquire command
 * @dev_handle:        Device handle to be returned
 * @flush_type:        Flush type:
 *                     0 = flush specific request
 *                     1 = flush all
 * @reserved:          Reserved for 64 bit aligngment
 * @req_id:            Request id that needs to cancel
 *
 */
struct cam_flush_dev_cmd {
	uint64_t       version;
	int32_t        session_handle;
	int32_t        dev_handle;
	uint32_t       flush_type;
	uint32_t       reserved;
	int64_t        req_id;
};

#endif /* __UAPI_CAM_DEFS_H__ */
