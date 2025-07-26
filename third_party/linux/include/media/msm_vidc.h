#ifndef __MSM_VIDC_H__
#define __MSM_VIDC_H__

#include <linux/types.h>

#define MSM_VIDC_HAL_INTERLACE_COLOR_FORMAT_NV12	0x2
#define MSM_VIDC_HAL_INTERLACE_COLOR_FORMAT_NV12_UBWC	0x8002
#define MSM_VIDC_4x_1 0x1
#define MSM_VIDC_EXTRADATA_FRAME_QP_ADV 0x1

struct msm_vidc_extradata_header {
	unsigned int size;
	unsigned int:32; /** Keeping binary compatibility */
	unsigned int:32; /* with firmware and OpenMAX IL **/
	unsigned int type; /* msm_vidc_extradata_type */
	unsigned int data_size;
	unsigned char data[1];
};

struct msm_vidc_interlace_payload {
	unsigned int format;
	unsigned int color_format;
};

struct msm_vidc_framerate_payload {
	unsigned int frame_rate;
};

struct msm_vidc_ts_payload {
	unsigned int timestamp_lo;
	unsigned int timestamp_hi;
};

struct msm_vidc_concealmb_payload {
	unsigned int num_mbs;
};

struct msm_vidc_recoverysei_payload {
	unsigned int flags;
};

struct msm_vidc_aspect_ratio_payload {
	unsigned int size;
	unsigned int version;
	unsigned int port_index;
	unsigned int aspect_width;
	unsigned int aspect_height;
};

struct msm_vidc_mpeg2_seqdisp_payload {
	unsigned int video_format;
	unsigned int color_descp;
	unsigned int color_primaries;
	unsigned int transfer_char;
	unsigned int matrix_coeffs;
	unsigned int disp_width;
	unsigned int disp_height;
};

struct msm_vidc_vc1_seqdisp_payload {
	unsigned int prog_seg_format;
	unsigned int uv_sampl_fmt;
	unsigned int color_format;
	unsigned int color_primaries;
	unsigned int transfer_char;
	unsigned int matrix_coeffs;
	unsigned int aspect_ratio;
	unsigned int aspect_horiz;
	unsigned int aspect_vert;
};

struct msm_vidc_input_crop_payload {
	unsigned int size;
	unsigned int version;
	unsigned int port_index;
	unsigned int left;
	unsigned int top;
	unsigned int width;
	unsigned int height;
};

struct msm_vidc_misr_info {
	unsigned int misr_dpb_luma;
	unsigned int misr_dpb_chroma;
	unsigned int misr_opb_luma;
	unsigned int misr_opb_chroma;
};
struct msm_vidc_output_crop_payload {
	unsigned int size;
	unsigned int version;
	unsigned int port_index;
	unsigned int left;
	unsigned int top;
	unsigned int display_width;
	unsigned int display_height;
	unsigned int width;
	unsigned int height;
	unsigned int frame_num;
	unsigned int bit_depth_y;
	unsigned int bit_depth_c;
	struct msm_vidc_misr_info misr_info[2];
};


struct msm_vidc_digital_zoom_payload {
	unsigned int size;
	unsigned int version;
	unsigned int port_index;
	unsigned int zoom_width;
	unsigned int zoom_height;
};

struct msm_vidc_extradata_index {
	unsigned int type;
	union {
		struct msm_vidc_input_crop_payload input_crop;
		struct msm_vidc_digital_zoom_payload digital_zoom;
		struct msm_vidc_aspect_ratio_payload aspect_ratio;
	};
};

struct msm_vidc_panscan_window {
	unsigned int panscan_height_offset;
	unsigned int panscan_width_offset;
	unsigned int panscan_window_width;
	unsigned int panscan_window_height;
};

struct msm_vidc_panscan_window_payload {
	unsigned int num_panscan_windows;
	struct msm_vidc_panscan_window wnd[1];
};

struct msm_vidc_stream_userdata_payload {
	unsigned int type;
	unsigned int data[1];
};

struct msm_vidc_frame_qp_payload {
	unsigned int frame_qp;
	unsigned int qp_sum;
	unsigned int skip_qp_sum;
	unsigned int skip_num_blocks;
	unsigned int total_num_blocks;
};

struct msm_vidc_frame_bits_info_payload {
	unsigned int frame_bits;
	unsigned int header_bits;
};

struct msm_vidc_s3d_frame_packing_payload {
	unsigned int fpa_id;
	unsigned int cancel_flag;
	unsigned int fpa_type;
	unsigned int quin_cunx_flag;
	unsigned int content_interprtation_type;
	unsigned int spatial_flipping_flag;
	unsigned int frame0_flipped_flag;
	unsigned int field_views_flag;
	unsigned int current_frame_is_frame0_flag;
	unsigned int frame0_self_contained_flag;
	unsigned int frame1_self_contained_flag;
	unsigned int frame0_graid_pos_x;
	unsigned int frame0_graid_pos_y;
	unsigned int frame1_graid_pos_x;
	unsigned int frame1_graid_pos_y;
	unsigned int fpa_reserved_byte;
	unsigned int fpa_repetition_period;
	unsigned int fpa_extension_flag;
};

struct msm_vidc_vqzip_sei_payload {
	unsigned int size;
	unsigned int data[1];
};

struct msm_vidc_ubwc_cr_stats_info {
	unsigned int stats_tile_32;
	unsigned int stats_tile_64;
	unsigned int stats_tile_96;
	unsigned int stats_tile_128;
	unsigned int stats_tile_160;
	unsigned int stats_tile_192;
	unsigned int stats_tile_256;
};

struct msm_vidc_yuv_stats_payload {
	unsigned int frame_qp;
	unsigned int texture;
	unsigned int luma_in_q16;
	unsigned int frame_difference;
};

struct msm_vidc_vpx_colorspace_payload {
	unsigned int color_space;
	unsigned int yuv_range_flag;
	unsigned int sumsampling_x;
	unsigned int sumsampling_y;
};

struct msm_vidc_roi_qp_payload {
	int upper_qp_offset;
	int lower_qp_offset;
	unsigned int b_roi_info;
	int mbi_info_size;
	unsigned int data[1];
};

struct msm_vidc_mastering_display_colour_sei_payload {
	unsigned int nDisplayPrimariesX[3];
	unsigned int nDisplayPrimariesY[3];
	unsigned int nWhitePointX;
	unsigned int nWhitePointY;
	unsigned int nMaxDisplayMasteringLuminance;
	unsigned int nMinDisplayMasteringLuminance;
};

struct msm_vidc_content_light_level_sei_payload {
	unsigned int nMaxContentLight;
	unsigned int nMaxPicAverageLight;
};

struct msm_vidc_vui_display_info_payload {
	unsigned int video_signal_present_flag;
	unsigned int video_format;
	unsigned int bit_depth_y;
	unsigned int bit_depth_c;
	unsigned int video_full_range_flag;
	unsigned int color_description_present_flag;
	unsigned int color_primaries;
	unsigned int transfer_characteristics;
	unsigned int matrix_coefficients;
	unsigned int chroma_location_info_present_flag;
	unsigned int chroma_format_idc;
	unsigned int separate_color_plane_flag;
	unsigned int chroma_sample_loc_type_top_field;
	unsigned int chroma_sample_loc_type_bottom_field;
};

enum msm_vidc_extradata_type {
	MSM_VIDC_EXTRADATA_NONE = 0x00000000,
	MSM_VIDC_EXTRADATA_MB_QUANTIZATION = 0x00000001,
	MSM_VIDC_EXTRADATA_INTERLACE_VIDEO = 0x00000002,
	MSM_VIDC_EXTRADATA_VC1_FRAMEDISP = 0x00000003,
	MSM_VIDC_EXTRADATA_VC1_SEQDISP = 0x00000004,
	MSM_VIDC_EXTRADATA_TIMESTAMP = 0x00000005,
	MSM_VIDC_EXTRADATA_S3D_FRAME_PACKING = 0x00000006,
	MSM_VIDC_EXTRADATA_FRAME_RATE = 0x00000007,
	MSM_VIDC_EXTRADATA_PANSCAN_WINDOW = 0x00000008,
	MSM_VIDC_EXTRADATA_RECOVERY_POINT_SEI = 0x00000009,
	MSM_VIDC_EXTRADATA_MPEG2_SEQDISP = 0x0000000D,
	MSM_VIDC_EXTRADATA_STREAM_USERDATA = 0x0000000E,
	MSM_VIDC_EXTRADATA_FRAME_QP = 0x0000000F,
	MSM_VIDC_EXTRADATA_FRAME_BITS_INFO = 0x00000010,
	MSM_VIDC_EXTRADATA_VQZIP_SEI = 0x00000011,
	MSM_VIDC_EXTRADATA_ROI_QP = 0x00000013,
#define MSM_VIDC_EXTRADATA_VPX_COLORSPACE_INFO \
	MSM_VIDC_EXTRADATA_VPX_COLORSPACE_INFO
	MSM_VIDC_EXTRADATA_VPX_COLORSPACE_INFO = 0x00000014,
#define MSM_VIDC_EXTRADATA_MASTERING_DISPLAY_COLOUR_SEI \
	MSM_VIDC_EXTRADATA_MASTERING_DISPLAY_COLOUR_SEI
	MSM_VIDC_EXTRADATA_MASTERING_DISPLAY_COLOUR_SEI = 0x00000015,
#define MSM_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI \
	MSM_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI
	MSM_VIDC_EXTRADATA_CONTENT_LIGHT_LEVEL_SEI = 0x00000016,
#define MSM_VIDC_EXTRADATA_PQ_INFO \
	MSM_VIDC_EXTRADATA_PQ_INFO
	MSM_VIDC_EXTRADATA_PQ_INFO = 0x00000017,
#define MSM_VIDC_EXTRADATA_COLOUR_REMAPPING_INFO_SEI \
	MSM_VIDC_EXTRADATA_COLOUR_REMAPPING_INFO_SEI
	MSM_VIDC_EXTRADATA_COLOUR_REMAPPING_INFO_SEI = 0x00000018,
#define MSM_VIDC_EXTRADATA_UBWC_CR_STAT_INFO \
	MSM_VIDC_EXTRADATA_UBWC_CR_STAT_INFO
	MSM_VIDC_EXTRADATA_UBWC_CR_STAT_INFO = 0x00000019,
	MSM_VIDC_EXTRADATA_INPUT_CROP = 0x0700000E,
#define MSM_VIDC_EXTRADATA_OUTPUT_CROP \
	MSM_VIDC_EXTRADATA_OUTPUT_CROP
	MSM_VIDC_EXTRADATA_OUTPUT_CROP = 0x0700000F,
	MSM_VIDC_EXTRADATA_DIGITAL_ZOOM = 0x07000010,
	MSM_VIDC_EXTRADATA_MULTISLICE_INFO = 0x7F100000,
	MSM_VIDC_EXTRADATA_NUM_CONCEALED_MB = 0x7F100001,
	MSM_VIDC_EXTRADATA_INDEX = 0x7F100002,
	MSM_VIDC_EXTRADATA_ASPECT_RATIO = 0x7F100003,
	MSM_VIDC_EXTRADATA_METADATA_LTR = 0x7F100004,
	MSM_VIDC_EXTRADATA_METADATA_FILLER = 0x7FE00002,
	MSM_VIDC_EXTRADATA_METADATA_MBI = 0x7F100005,
#define MSM_VIDC_EXTRADATA_VUI_DISPLAY_INFO \
	MSM_VIDC_EXTRADATA_VUI_DISPLAY_INFO
	MSM_VIDC_EXTRADATA_VUI_DISPLAY_INFO = 0x7F100006,
	MSM_VIDC_EXTRADATA_YUVSTATS_INFO = 0x7F100007,
};
enum msm_vidc_interlace_type {
	MSM_VIDC_INTERLACE_FRAME_PROGRESSIVE = 0x01,
	MSM_VIDC_INTERLACE_INTERLEAVE_FRAME_TOPFIELDFIRST = 0x02,
	MSM_VIDC_INTERLACE_INTERLEAVE_FRAME_BOTTOMFIELDFIRST = 0x04,
	MSM_VIDC_INTERLACE_FRAME_TOPFIELDFIRST = 0x08,
	MSM_VIDC_INTERLACE_FRAME_BOTTOMFIELDFIRST = 0x10,
#define MSM_VIDC_INTERLACE_FRAME_MBAFF \
	MSM_VIDC_INTERLACE_FRAME_MBAFF
	MSM_VIDC_INTERLACE_FRAME_MBAFF = 0x20,
};

/* enum msm_vidc_framepack_type */
#define MSM_VIDC_FRAMEPACK_CHECKERBOARD 0x00
#define MSM_VIDC_FRAMEPACK_COLUMN_INTERLEAVE 0x01
#define MSM_VIDC_FRAMEPACK_ROW_INTERLEAVE 0x02
#define MSM_VIDC_FRAMEPACK_SIDE_BY_SIDE 0x03
#define MSM_VIDC_FRAMEPACK_TOP_BOTTOM 0x04
#define MSM_VIDC_FRAMEPACK_TEMPORAL_INTERLEAVE 0x05

enum msm_vidc_recovery_sei {
	MSM_VIDC_FRAME_RECONSTRUCTION_INCORRECT = 0x0,
	MSM_VIDC_FRAME_RECONSTRUCTION_CORRECT = 0x01,
	MSM_VIDC_FRAME_RECONSTRUCTION_APPROXIMATELY_CORRECT = 0x02,
};
enum msm_vidc_userdata_type {
	MSM_VIDC_USERDATA_TYPE_FRAME = 0x1,
	MSM_VIDC_USERDATA_TYPE_TOP_FIELD = 0x2,
	MSM_VIDC_USERDATA_TYPE_BOTTOM_FIELD = 0x3,
};

/* See colour_primaries of ISO/IEC 14496 for significance */
enum msm_vidc_h264_color_primaries_values {
	MSM_VIDC_RESERVED_1 = 0,
	MSM_VIDC_BT709_5 = 1,
	MSM_VIDC_UNSPECIFIED = 2,
	MSM_VIDC_RESERVED_2 = 3,
	MSM_VIDC_BT470_6_M = 4,
	MSM_VIDC_BT601_6_625 = 5,
	MSM_VIDC_BT470_6_BG = MSM_VIDC_BT601_6_625,
	MSM_VIDC_BT601_6_525 = 6,
	MSM_VIDC_SMPTE_240M = 7,
	MSM_VIDC_GENERIC_FILM = 8,
	MSM_VIDC_BT2020 = 9,
};

enum msm_vidc_vp9_color_primaries_values {
	MSM_VIDC_CS_UNKNOWN,
	MSM_VIDC_CS_BT_601,
	MSM_VIDC_CS_BT_709,
	MSM_VIDC_CS_SMPTE_170,
	MSM_VIDC_CS_SMPTE_240,
	MSM_VIDC_CS_BT_2020,
	MSM_VIDC_CS_RESERVED,
	MSM_VIDC_CS_RGB,
};

enum msm_vidc_h264_matrix_coeff_values {
	MSM_VIDC_MATRIX_RGB = 0,
	MSM_VIDC_MATRIX_BT_709_5 = 1,
	MSM_VIDC_MATRIX_UNSPECIFIED = 2,
	MSM_VIDC_MATRIX_RESERVED = 3,
	MSM_VIDC_MATRIX_FCC_47 = 4,
	MSM_VIDC_MATRIX_601_6_625 = 5,
	MSM_VIDC_MATRIX_BT470_BG = MSM_VIDC_MATRIX_601_6_625,
	MSM_VIDC_MATRIX_601_6_525 = 6,
	MSM_VIDC_MATRIX_SMPTE_170M = MSM_VIDC_MATRIX_601_6_525,
	MSM_VIDC_MATRIX_SMPTE_240M = 7,
	MSM_VIDC_MATRIX_Y_CG_CO = 8,
	MSM_VIDC_MATRIX_BT_2020 = 9,
	MSM_VIDC_MATRIX_BT_2020_CONST = 10,
};

enum msm_vidc_h264_transfer_chars_values {
	MSM_VIDC_TRANSFER_RESERVED_1 = 0,
	MSM_VIDC_TRANSFER_BT709_5 = 1,
	MSM_VIDC_TRANSFER_UNSPECIFIED = 2,
	MSM_VIDC_TRANSFER_RESERVED_2 = 3,
	MSM_VIDC_TRANSFER_BT_470_6_M = 4,
	MSM_VIDC_TRANSFER_BT_470_6_BG = 5,
	MSM_VIDC_TRANSFER_601_6_625 = 6,
	MSM_VIDC_TRANSFER_601_6_525 = MSM_VIDC_TRANSFER_601_6_625,
	MSM_VIDC_TRANSFER_SMPTE_240M = 7,
	MSM_VIDC_TRANSFER_LINEAR = 8,
	MSM_VIDC_TRANSFER_LOG_100_1 = 9,
	MSM_VIDC_TRANSFER_LOG_100_SQRT10_1 = 10,
	MSM_VIDC_TRANSFER_IEC_61966 = 11,
	MSM_VIDC_TRANSFER_BT_1361 = 12,
	MSM_VIDC_TRANSFER_SRGB = 13,
	MSM_VIDC_TRANSFER_BT_2020_10 = 14,
	MSM_VIDC_TRANSFER_BT_2020_12 = 15,
#define MSM_VIDC_TRANSFER_SMPTE_ST2084 \
	MSM_VIDC_TRANSFER_SMPTE_ST2084
	MSM_VIDC_TRANSFER_SMPTE_ST2084 = 16,
#define MSM_VIDC_TRANSFER_SMPTE_ST428_1 \
	MSM_VIDC_TRANSFER_SMPTE_ST428_1
	MSM_VIDC_TRANSFER_SMPTE_ST428_1 = 17,
#define MSM_VIDC_TRANSFER_HLG \
	MSM_VIDC_TRANSFER_HLG
	MSM_VIDC_TRANSFER_HLG = 18,
};

enum msm_vidc_pixel_depth {
	MSM_VIDC_BIT_DEPTH_8,
	MSM_VIDC_BIT_DEPTH_10,
	MSM_VIDC_BIT_DEPTH_UNSUPPORTED = 0XFFFFFFFF,
};

enum msm_vidc_video_format {
	MSM_VIDC_COMPONENT,
	MSM_VIDC_PAL,
	MSM_VIDC_NTSC,
	MSM_VIDC_SECAM,
	MSM_VIDC_MAC,
	MSM_VIDC_UNSPECIFIED_FORMAT,
	MSM_VIDC_RESERVED_1_FORMAT,
	MSM_VIDC_RESERVED_2_FORMAT,
};

enum msm_vidc_color_desc_flag {
	MSM_VIDC_COLOR_DESC_NOT_PRESENT,
	MSM_VIDC_COLOR_DESC_PRESENT,
};

/*enum msm_vidc_pic_struct */
#define MSM_VIDC_PIC_STRUCT_MAYBE_INTERLACED 0x0
#define MSM_VIDC_PIC_STRUCT_PROGRESSIVE 0x1
#define MSM_VIDC_PIC_STRUCT_UNKNOWN 0XFFFFFFFF
/*default when layer ID isn't specified*/
#define MSM_VIDC_ALL_LAYER_ID 0xFF

#endif
