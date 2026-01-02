# mypy: ignore-errors
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
enum_HEVCNALUnitType = CEnum(ctypes.c_uint32)
HEVC_NAL_TRAIL_N = enum_HEVCNALUnitType.define('HEVC_NAL_TRAIL_N', 0)
HEVC_NAL_TRAIL_R = enum_HEVCNALUnitType.define('HEVC_NAL_TRAIL_R', 1)
HEVC_NAL_TSA_N = enum_HEVCNALUnitType.define('HEVC_NAL_TSA_N', 2)
HEVC_NAL_TSA_R = enum_HEVCNALUnitType.define('HEVC_NAL_TSA_R', 3)
HEVC_NAL_STSA_N = enum_HEVCNALUnitType.define('HEVC_NAL_STSA_N', 4)
HEVC_NAL_STSA_R = enum_HEVCNALUnitType.define('HEVC_NAL_STSA_R', 5)
HEVC_NAL_RADL_N = enum_HEVCNALUnitType.define('HEVC_NAL_RADL_N', 6)
HEVC_NAL_RADL_R = enum_HEVCNALUnitType.define('HEVC_NAL_RADL_R', 7)
HEVC_NAL_RASL_N = enum_HEVCNALUnitType.define('HEVC_NAL_RASL_N', 8)
HEVC_NAL_RASL_R = enum_HEVCNALUnitType.define('HEVC_NAL_RASL_R', 9)
HEVC_NAL_VCL_N10 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N10', 10)
HEVC_NAL_VCL_R11 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R11', 11)
HEVC_NAL_VCL_N12 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N12', 12)
HEVC_NAL_VCL_R13 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R13', 13)
HEVC_NAL_VCL_N14 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N14', 14)
HEVC_NAL_VCL_R15 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R15', 15)
HEVC_NAL_BLA_W_LP = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_W_LP', 16)
HEVC_NAL_BLA_W_RADL = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_W_RADL', 17)
HEVC_NAL_BLA_N_LP = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_N_LP', 18)
HEVC_NAL_IDR_W_RADL = enum_HEVCNALUnitType.define('HEVC_NAL_IDR_W_RADL', 19)
HEVC_NAL_IDR_N_LP = enum_HEVCNALUnitType.define('HEVC_NAL_IDR_N_LP', 20)
HEVC_NAL_CRA_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_CRA_NUT', 21)
HEVC_NAL_RSV_IRAP_VCL22 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_IRAP_VCL22', 22)
HEVC_NAL_RSV_IRAP_VCL23 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_IRAP_VCL23', 23)
HEVC_NAL_RSV_VCL24 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL24', 24)
HEVC_NAL_RSV_VCL25 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL25', 25)
HEVC_NAL_RSV_VCL26 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL26', 26)
HEVC_NAL_RSV_VCL27 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL27', 27)
HEVC_NAL_RSV_VCL28 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL28', 28)
HEVC_NAL_RSV_VCL29 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL29', 29)
HEVC_NAL_RSV_VCL30 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL30', 30)
HEVC_NAL_RSV_VCL31 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL31', 31)
HEVC_NAL_VPS = enum_HEVCNALUnitType.define('HEVC_NAL_VPS', 32)
HEVC_NAL_SPS = enum_HEVCNALUnitType.define('HEVC_NAL_SPS', 33)
HEVC_NAL_PPS = enum_HEVCNALUnitType.define('HEVC_NAL_PPS', 34)
HEVC_NAL_AUD = enum_HEVCNALUnitType.define('HEVC_NAL_AUD', 35)
HEVC_NAL_EOS_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_EOS_NUT', 36)
HEVC_NAL_EOB_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_EOB_NUT', 37)
HEVC_NAL_FD_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_FD_NUT', 38)
HEVC_NAL_SEI_PREFIX = enum_HEVCNALUnitType.define('HEVC_NAL_SEI_PREFIX', 39)
HEVC_NAL_SEI_SUFFIX = enum_HEVCNALUnitType.define('HEVC_NAL_SEI_SUFFIX', 40)
HEVC_NAL_RSV_NVCL41 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL41', 41)
HEVC_NAL_RSV_NVCL42 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL42', 42)
HEVC_NAL_RSV_NVCL43 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL43', 43)
HEVC_NAL_RSV_NVCL44 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL44', 44)
HEVC_NAL_RSV_NVCL45 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL45', 45)
HEVC_NAL_RSV_NVCL46 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL46', 46)
HEVC_NAL_RSV_NVCL47 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL47', 47)
HEVC_NAL_UNSPEC48 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC48', 48)
HEVC_NAL_UNSPEC49 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC49', 49)
HEVC_NAL_UNSPEC50 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC50', 50)
HEVC_NAL_UNSPEC51 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC51', 51)
HEVC_NAL_UNSPEC52 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC52', 52)
HEVC_NAL_UNSPEC53 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC53', 53)
HEVC_NAL_UNSPEC54 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC54', 54)
HEVC_NAL_UNSPEC55 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC55', 55)
HEVC_NAL_UNSPEC56 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC56', 56)
HEVC_NAL_UNSPEC57 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC57', 57)
HEVC_NAL_UNSPEC58 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC58', 58)
HEVC_NAL_UNSPEC59 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC59', 59)
HEVC_NAL_UNSPEC60 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC60', 60)
HEVC_NAL_UNSPEC61 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC61', 61)
HEVC_NAL_UNSPEC62 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC62', 62)
HEVC_NAL_UNSPEC63 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC63', 63)

enum_HEVCSliceType = CEnum(ctypes.c_uint32)
HEVC_SLICE_B = enum_HEVCSliceType.define('HEVC_SLICE_B', 0)
HEVC_SLICE_P = enum_HEVCSliceType.define('HEVC_SLICE_P', 1)
HEVC_SLICE_I = enum_HEVCSliceType.define('HEVC_SLICE_I', 2)

_anonenum0 = CEnum(ctypes.c_uint32)
HEVC_MAX_LAYERS = _anonenum0.define('HEVC_MAX_LAYERS', 63)
HEVC_MAX_SUB_LAYERS = _anonenum0.define('HEVC_MAX_SUB_LAYERS', 7)
HEVC_MAX_LAYER_SETS = _anonenum0.define('HEVC_MAX_LAYER_SETS', 1024)
HEVC_MAX_LAYER_ID = _anonenum0.define('HEVC_MAX_LAYER_ID', 63)
HEVC_MAX_NUH_LAYER_ID = _anonenum0.define('HEVC_MAX_NUH_LAYER_ID', 62)
HEVC_MAX_VPS_COUNT = _anonenum0.define('HEVC_MAX_VPS_COUNT', 16)
HEVC_MAX_SPS_COUNT = _anonenum0.define('HEVC_MAX_SPS_COUNT', 16)
HEVC_MAX_PPS_COUNT = _anonenum0.define('HEVC_MAX_PPS_COUNT', 64)
HEVC_MAX_DPB_SIZE = _anonenum0.define('HEVC_MAX_DPB_SIZE', 16)
HEVC_MAX_REFS = _anonenum0.define('HEVC_MAX_REFS', 16)
HEVC_MAX_SHORT_TERM_REF_PIC_SETS = _anonenum0.define('HEVC_MAX_SHORT_TERM_REF_PIC_SETS', 64)
HEVC_MAX_LONG_TERM_REF_PICS = _anonenum0.define('HEVC_MAX_LONG_TERM_REF_PICS', 32)
HEVC_MIN_LOG2_CTB_SIZE = _anonenum0.define('HEVC_MIN_LOG2_CTB_SIZE', 4)
HEVC_MAX_LOG2_CTB_SIZE = _anonenum0.define('HEVC_MAX_LOG2_CTB_SIZE', 6)
HEVC_MAX_CPB_CNT = _anonenum0.define('HEVC_MAX_CPB_CNT', 32)
HEVC_MAX_LUMA_PS = _anonenum0.define('HEVC_MAX_LUMA_PS', 35651584)
HEVC_MAX_WIDTH = _anonenum0.define('HEVC_MAX_WIDTH', 16888)
HEVC_MAX_HEIGHT = _anonenum0.define('HEVC_MAX_HEIGHT', 16888)
HEVC_MAX_TILE_ROWS = _anonenum0.define('HEVC_MAX_TILE_ROWS', 22)
HEVC_MAX_TILE_COLUMNS = _anonenum0.define('HEVC_MAX_TILE_COLUMNS', 20)
HEVC_MAX_SLICE_SEGMENTS = _anonenum0.define('HEVC_MAX_SLICE_SEGMENTS', 600)
HEVC_MAX_ENTRY_POINT_OFFSETS = _anonenum0.define('HEVC_MAX_ENTRY_POINT_OFFSETS', 2700)
HEVC_MAX_PALETTE_PREDICTOR_SIZE = _anonenum0.define('HEVC_MAX_PALETTE_PREDICTOR_SIZE', 128)

enum_HEVCScalabilityMask = CEnum(ctypes.c_uint32)
HEVC_SCALABILITY_DEPTH = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_DEPTH', 32768)
HEVC_SCALABILITY_MULTIVIEW = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_MULTIVIEW', 16384)
HEVC_SCALABILITY_SPATIAL = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_SPATIAL', 8192)
HEVC_SCALABILITY_AUXILIARY = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_AUXILIARY', 4096)
HEVC_SCALABILITY_MASK_MAX = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_MASK_MAX', 65535)

enum_HEVCAuxId = CEnum(ctypes.c_uint32)
HEVC_AUX_ALPHA = enum_HEVCAuxId.define('HEVC_AUX_ALPHA', 1)
HEVC_AUX_DEPTH = enum_HEVCAuxId.define('HEVC_AUX_DEPTH', 2)

class struct_H265RawNALUnitHeader(Struct): pass
uint8_t = ctypes.c_ubyte
struct_H265RawNALUnitHeader._fields_ = [
  ('nal_unit_type', uint8_t),
  ('nuh_layer_id', uint8_t),
  ('nuh_temporal_id_plus1', uint8_t),
]
H265RawNALUnitHeader = struct_H265RawNALUnitHeader
class struct_H265RawProfileTierLevel(Struct): pass
struct_H265RawProfileTierLevel._fields_ = [
  ('general_profile_space', uint8_t),
  ('general_tier_flag', uint8_t),
  ('general_profile_idc', uint8_t),
  ('general_profile_compatibility_flag', (uint8_t * 32)),
  ('general_progressive_source_flag', uint8_t),
  ('general_interlaced_source_flag', uint8_t),
  ('general_non_packed_constraint_flag', uint8_t),
  ('general_frame_only_constraint_flag', uint8_t),
  ('general_max_12bit_constraint_flag', uint8_t),
  ('general_max_10bit_constraint_flag', uint8_t),
  ('general_max_8bit_constraint_flag', uint8_t),
  ('general_max_422chroma_constraint_flag', uint8_t),
  ('general_max_420chroma_constraint_flag', uint8_t),
  ('general_max_monochrome_constraint_flag', uint8_t),
  ('general_intra_constraint_flag', uint8_t),
  ('general_one_picture_only_constraint_flag', uint8_t),
  ('general_lower_bit_rate_constraint_flag', uint8_t),
  ('general_max_14bit_constraint_flag', uint8_t),
  ('general_inbld_flag', uint8_t),
  ('general_level_idc', uint8_t),
  ('sub_layer_profile_present_flag', (uint8_t * 7)),
  ('sub_layer_level_present_flag', (uint8_t * 7)),
  ('sub_layer_profile_space', (uint8_t * 7)),
  ('sub_layer_tier_flag', (uint8_t * 7)),
  ('sub_layer_profile_idc', (uint8_t * 7)),
  ('sub_layer_profile_compatibility_flag', ((uint8_t * 32) * 7)),
  ('sub_layer_progressive_source_flag', (uint8_t * 7)),
  ('sub_layer_interlaced_source_flag', (uint8_t * 7)),
  ('sub_layer_non_packed_constraint_flag', (uint8_t * 7)),
  ('sub_layer_frame_only_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_12bit_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_10bit_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_8bit_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_422chroma_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_420chroma_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_monochrome_constraint_flag', (uint8_t * 7)),
  ('sub_layer_intra_constraint_flag', (uint8_t * 7)),
  ('sub_layer_one_picture_only_constraint_flag', (uint8_t * 7)),
  ('sub_layer_lower_bit_rate_constraint_flag', (uint8_t * 7)),
  ('sub_layer_max_14bit_constraint_flag', (uint8_t * 7)),
  ('sub_layer_inbld_flag', (uint8_t * 7)),
  ('sub_layer_level_idc', (uint8_t * 7)),
]
H265RawProfileTierLevel = struct_H265RawProfileTierLevel
class struct_H265RawSubLayerHRDParameters(Struct): pass
uint32_t = ctypes.c_uint32
struct_H265RawSubLayerHRDParameters._fields_ = [
  ('bit_rate_value_minus1', (uint32_t * 32)),
  ('cpb_size_value_minus1', (uint32_t * 32)),
  ('cpb_size_du_value_minus1', (uint32_t * 32)),
  ('bit_rate_du_value_minus1', (uint32_t * 32)),
  ('cbr_flag', (uint8_t * 32)),
]
H265RawSubLayerHRDParameters = struct_H265RawSubLayerHRDParameters
class struct_H265RawHRDParameters(Struct): pass
uint16_t = ctypes.c_uint16
struct_H265RawHRDParameters._fields_ = [
  ('nal_hrd_parameters_present_flag', uint8_t),
  ('vcl_hrd_parameters_present_flag', uint8_t),
  ('sub_pic_hrd_params_present_flag', uint8_t),
  ('tick_divisor_minus2', uint8_t),
  ('du_cpb_removal_delay_increment_length_minus1', uint8_t),
  ('sub_pic_cpb_params_in_pic_timing_sei_flag', uint8_t),
  ('dpb_output_delay_du_length_minus1', uint8_t),
  ('bit_rate_scale', uint8_t),
  ('cpb_size_scale', uint8_t),
  ('cpb_size_du_scale', uint8_t),
  ('initial_cpb_removal_delay_length_minus1', uint8_t),
  ('au_cpb_removal_delay_length_minus1', uint8_t),
  ('dpb_output_delay_length_minus1', uint8_t),
  ('fixed_pic_rate_general_flag', (uint8_t * 7)),
  ('fixed_pic_rate_within_cvs_flag', (uint8_t * 7)),
  ('elemental_duration_in_tc_minus1', (uint16_t * 7)),
  ('low_delay_hrd_flag', (uint8_t * 7)),
  ('cpb_cnt_minus1', (uint8_t * 7)),
  ('nal_sub_layer_hrd_parameters', (H265RawSubLayerHRDParameters * 7)),
  ('vcl_sub_layer_hrd_parameters', (H265RawSubLayerHRDParameters * 7)),
]
H265RawHRDParameters = struct_H265RawHRDParameters
class struct_H265RawVUI(Struct): pass
struct_H265RawVUI._fields_ = [
  ('aspect_ratio_info_present_flag', uint8_t),
  ('aspect_ratio_idc', uint8_t),
  ('sar_width', uint16_t),
  ('sar_height', uint16_t),
  ('overscan_info_present_flag', uint8_t),
  ('overscan_appropriate_flag', uint8_t),
  ('video_signal_type_present_flag', uint8_t),
  ('video_format', uint8_t),
  ('video_full_range_flag', uint8_t),
  ('colour_description_present_flag', uint8_t),
  ('colour_primaries', uint8_t),
  ('transfer_characteristics', uint8_t),
  ('matrix_coefficients', uint8_t),
  ('chroma_loc_info_present_flag', uint8_t),
  ('chroma_sample_loc_type_top_field', uint8_t),
  ('chroma_sample_loc_type_bottom_field', uint8_t),
  ('neutral_chroma_indication_flag', uint8_t),
  ('field_seq_flag', uint8_t),
  ('frame_field_info_present_flag', uint8_t),
  ('default_display_window_flag', uint8_t),
  ('def_disp_win_left_offset', uint16_t),
  ('def_disp_win_right_offset', uint16_t),
  ('def_disp_win_top_offset', uint16_t),
  ('def_disp_win_bottom_offset', uint16_t),
  ('vui_timing_info_present_flag', uint8_t),
  ('vui_num_units_in_tick', uint32_t),
  ('vui_time_scale', uint32_t),
  ('vui_poc_proportional_to_timing_flag', uint8_t),
  ('vui_num_ticks_poc_diff_one_minus1', uint32_t),
  ('vui_hrd_parameters_present_flag', uint8_t),
  ('hrd_parameters', H265RawHRDParameters),
  ('bitstream_restriction_flag', uint8_t),
  ('tiles_fixed_structure_flag', uint8_t),
  ('motion_vectors_over_pic_boundaries_flag', uint8_t),
  ('restricted_ref_pic_lists_flag', uint8_t),
  ('min_spatial_segmentation_idc', uint16_t),
  ('max_bytes_per_pic_denom', uint8_t),
  ('max_bits_per_min_cu_denom', uint8_t),
  ('log2_max_mv_length_horizontal', uint8_t),
  ('log2_max_mv_length_vertical', uint8_t),
]
H265RawVUI = struct_H265RawVUI
class struct_H265RawExtensionData(Struct): pass
H265RawExtensionData = struct_H265RawExtensionData
class struct_H265RawVPS(Struct): pass
H265RawVPS = struct_H265RawVPS
class struct_H265RawSTRefPicSet(Struct): pass
struct_H265RawSTRefPicSet._fields_ = [
  ('inter_ref_pic_set_prediction_flag', uint8_t),
  ('delta_idx_minus1', uint8_t),
  ('delta_rps_sign', uint8_t),
  ('abs_delta_rps_minus1', uint16_t),
  ('used_by_curr_pic_flag', (uint8_t * 16)),
  ('use_delta_flag', (uint8_t * 16)),
  ('num_negative_pics', uint8_t),
  ('num_positive_pics', uint8_t),
  ('delta_poc_s0_minus1', (uint16_t * 16)),
  ('used_by_curr_pic_s0_flag', (uint8_t * 16)),
  ('delta_poc_s1_minus1', (uint16_t * 16)),
  ('used_by_curr_pic_s1_flag', (uint8_t * 16)),
]
H265RawSTRefPicSet = struct_H265RawSTRefPicSet
class struct_H265RawScalingList(Struct): pass
int16_t = ctypes.c_int16
int8_t = ctypes.c_byte
struct_H265RawScalingList._fields_ = [
  ('scaling_list_pred_mode_flag', ((uint8_t * 6) * 4)),
  ('scaling_list_pred_matrix_id_delta', ((uint8_t * 6) * 4)),
  ('scaling_list_dc_coef_minus8', ((int16_t * 6) * 4)),
  ('scaling_list_delta_coeff', (((int8_t * 64) * 6) * 4)),
]
H265RawScalingList = struct_H265RawScalingList
class struct_H265RawSPS(Struct): pass
H265RawSPS = struct_H265RawSPS
class struct_H265RawPPS(Struct): pass
H265RawPPS = struct_H265RawPPS
class struct_H265RawAUD(Struct): pass
struct_H265RawAUD._fields_ = [
  ('nal_unit_header', H265RawNALUnitHeader),
  ('pic_type', uint8_t),
]
H265RawAUD = struct_H265RawAUD
class struct_H265RawSliceHeader(Struct): pass
struct_H265RawSliceHeader._fields_ = [
  ('nal_unit_header', H265RawNALUnitHeader),
  ('first_slice_segment_in_pic_flag', uint8_t),
  ('no_output_of_prior_pics_flag', uint8_t),
  ('slice_pic_parameter_set_id', uint8_t),
  ('dependent_slice_segment_flag', uint8_t),
  ('slice_segment_address', uint16_t),
  ('slice_reserved_flag', (uint8_t * 8)),
  ('slice_type', uint8_t),
  ('pic_output_flag', uint8_t),
  ('colour_plane_id', uint8_t),
  ('slice_pic_order_cnt_lsb', uint16_t),
  ('short_term_ref_pic_set_sps_flag', uint8_t),
  ('short_term_ref_pic_set', H265RawSTRefPicSet),
  ('short_term_ref_pic_set_idx', uint8_t),
  ('num_long_term_sps', uint8_t),
  ('num_long_term_pics', uint8_t),
  ('lt_idx_sps', (uint8_t * 16)),
  ('poc_lsb_lt', (uint8_t * 16)),
  ('used_by_curr_pic_lt_flag', (uint8_t * 16)),
  ('delta_poc_msb_present_flag', (uint8_t * 16)),
  ('delta_poc_msb_cycle_lt', (uint32_t * 16)),
  ('slice_temporal_mvp_enabled_flag', uint8_t),
  ('slice_sao_luma_flag', uint8_t),
  ('slice_sao_chroma_flag', uint8_t),
  ('num_ref_idx_active_override_flag', uint8_t),
  ('num_ref_idx_l0_active_minus1', uint8_t),
  ('num_ref_idx_l1_active_minus1', uint8_t),
  ('ref_pic_list_modification_flag_l0', uint8_t),
  ('list_entry_l0', (uint8_t * 16)),
  ('ref_pic_list_modification_flag_l1', uint8_t),
  ('list_entry_l1', (uint8_t * 16)),
  ('mvd_l1_zero_flag', uint8_t),
  ('cabac_init_flag', uint8_t),
  ('collocated_from_l0_flag', uint8_t),
  ('collocated_ref_idx', uint8_t),
  ('luma_log2_weight_denom', uint8_t),
  ('delta_chroma_log2_weight_denom', int8_t),
  ('luma_weight_l0_flag', (uint8_t * 16)),
  ('chroma_weight_l0_flag', (uint8_t * 16)),
  ('delta_luma_weight_l0', (int8_t * 16)),
  ('luma_offset_l0', (int16_t * 16)),
  ('delta_chroma_weight_l0', ((int8_t * 2) * 16)),
  ('chroma_offset_l0', ((int16_t * 2) * 16)),
  ('luma_weight_l1_flag', (uint8_t * 16)),
  ('chroma_weight_l1_flag', (uint8_t * 16)),
  ('delta_luma_weight_l1', (int8_t * 16)),
  ('luma_offset_l1', (int16_t * 16)),
  ('delta_chroma_weight_l1', ((int8_t * 2) * 16)),
  ('chroma_offset_l1', ((int16_t * 2) * 16)),
  ('five_minus_max_num_merge_cand', uint8_t),
  ('use_integer_mv_flag', uint8_t),
  ('slice_qp_delta', int8_t),
  ('slice_cb_qp_offset', int8_t),
  ('slice_cr_qp_offset', int8_t),
  ('slice_act_y_qp_offset', int8_t),
  ('slice_act_cb_qp_offset', int8_t),
  ('slice_act_cr_qp_offset', int8_t),
  ('cu_chroma_qp_offset_enabled_flag', uint8_t),
  ('deblocking_filter_override_flag', uint8_t),
  ('slice_deblocking_filter_disabled_flag', uint8_t),
  ('slice_beta_offset_div2', int8_t),
  ('slice_tc_offset_div2', int8_t),
  ('slice_loop_filter_across_slices_enabled_flag', uint8_t),
  ('num_entry_point_offsets', uint16_t),
  ('offset_len_minus1', uint8_t),
  ('entry_point_offset_minus1', (uint32_t * 2700)),
  ('slice_segment_header_extension_length', uint16_t),
  ('slice_segment_header_extension_data_byte', (uint8_t * 256)),
]
H265RawSliceHeader = struct_H265RawSliceHeader
class struct_H265RawSlice(Struct): pass
H265RawSlice = struct_H265RawSlice
class struct_H265RawSEIBufferingPeriod(Struct): pass
struct_H265RawSEIBufferingPeriod._fields_ = [
  ('bp_seq_parameter_set_id', uint8_t),
  ('irap_cpb_params_present_flag', uint8_t),
  ('cpb_delay_offset', uint32_t),
  ('dpb_delay_offset', uint32_t),
  ('concatenation_flag', uint8_t),
  ('au_cpb_removal_delay_delta_minus1', uint32_t),
  ('nal_initial_cpb_removal_delay', (uint32_t * 32)),
  ('nal_initial_cpb_removal_offset', (uint32_t * 32)),
  ('nal_initial_alt_cpb_removal_delay', (uint32_t * 32)),
  ('nal_initial_alt_cpb_removal_offset', (uint32_t * 32)),
  ('vcl_initial_cpb_removal_delay', (uint32_t * 32)),
  ('vcl_initial_cpb_removal_offset', (uint32_t * 32)),
  ('vcl_initial_alt_cpb_removal_delay', (uint32_t * 32)),
  ('vcl_initial_alt_cpb_removal_offset', (uint32_t * 32)),
  ('use_alt_cpb_params_flag', uint8_t),
]
H265RawSEIBufferingPeriod = struct_H265RawSEIBufferingPeriod
class struct_H265RawSEIPicTiming(Struct): pass
struct_H265RawSEIPicTiming._fields_ = [
  ('pic_struct', uint8_t),
  ('source_scan_type', uint8_t),
  ('duplicate_flag', uint8_t),
  ('au_cpb_removal_delay_minus1', uint32_t),
  ('pic_dpb_output_delay', uint32_t),
  ('pic_dpb_output_du_delay', uint32_t),
  ('num_decoding_units_minus1', uint16_t),
  ('du_common_cpb_removal_delay_flag', uint8_t),
  ('du_common_cpb_removal_delay_increment_minus1', uint32_t),
  ('num_nalus_in_du_minus1', (uint16_t * 600)),
  ('du_cpb_removal_delay_increment_minus1', (uint32_t * 600)),
]
H265RawSEIPicTiming = struct_H265RawSEIPicTiming
class struct_H265RawSEIPanScanRect(Struct): pass
int32_t = ctypes.c_int32
struct_H265RawSEIPanScanRect._fields_ = [
  ('pan_scan_rect_id', uint32_t),
  ('pan_scan_rect_cancel_flag', uint8_t),
  ('pan_scan_cnt_minus1', uint8_t),
  ('pan_scan_rect_left_offset', (int32_t * 3)),
  ('pan_scan_rect_right_offset', (int32_t * 3)),
  ('pan_scan_rect_top_offset', (int32_t * 3)),
  ('pan_scan_rect_bottom_offset', (int32_t * 3)),
  ('pan_scan_rect_persistence_flag', uint16_t),
]
H265RawSEIPanScanRect = struct_H265RawSEIPanScanRect
class struct_H265RawSEIRecoveryPoint(Struct): pass
struct_H265RawSEIRecoveryPoint._fields_ = [
  ('recovery_poc_cnt', int16_t),
  ('exact_match_flag', uint8_t),
  ('broken_link_flag', uint8_t),
]
H265RawSEIRecoveryPoint = struct_H265RawSEIRecoveryPoint
class struct_H265RawFilmGrainCharacteristics(Struct): pass
struct_H265RawFilmGrainCharacteristics._fields_ = [
  ('film_grain_characteristics_cancel_flag', uint8_t),
  ('film_grain_model_id', uint8_t),
  ('separate_colour_description_present_flag', uint8_t),
  ('film_grain_bit_depth_luma_minus8', uint8_t),
  ('film_grain_bit_depth_chroma_minus8', uint8_t),
  ('film_grain_full_range_flag', uint8_t),
  ('film_grain_colour_primaries', uint8_t),
  ('film_grain_transfer_characteristics', uint8_t),
  ('film_grain_matrix_coeffs', uint8_t),
  ('blending_mode_id', uint8_t),
  ('log2_scale_factor', uint8_t),
  ('comp_model_present_flag', (uint8_t * 3)),
  ('num_intensity_intervals_minus1', (uint8_t * 3)),
  ('num_model_values_minus1', (uint8_t * 3)),
  ('intensity_interval_lower_bound', ((uint8_t * 256) * 3)),
  ('intensity_interval_upper_bound', ((uint8_t * 256) * 3)),
  ('comp_model_value', (((int16_t * 6) * 256) * 3)),
  ('film_grain_characteristics_persistence_flag', uint8_t),
]
H265RawFilmGrainCharacteristics = struct_H265RawFilmGrainCharacteristics
class struct_H265RawSEIDisplayOrientation(Struct): pass
struct_H265RawSEIDisplayOrientation._fields_ = [
  ('display_orientation_cancel_flag', uint8_t),
  ('hor_flip', uint8_t),
  ('ver_flip', uint8_t),
  ('anticlockwise_rotation', uint16_t),
  ('display_orientation_repetition_period', uint16_t),
  ('display_orientation_persistence_flag', uint8_t),
]
H265RawSEIDisplayOrientation = struct_H265RawSEIDisplayOrientation
class struct_H265RawSEIActiveParameterSets(Struct): pass
struct_H265RawSEIActiveParameterSets._fields_ = [
  ('active_video_parameter_set_id', uint8_t),
  ('self_contained_cvs_flag', uint8_t),
  ('no_parameter_set_update_flag', uint8_t),
  ('num_sps_ids_minus1', uint8_t),
  ('active_seq_parameter_set_id', (uint8_t * 16)),
  ('layer_sps_idx', (uint8_t * 63)),
]
H265RawSEIActiveParameterSets = struct_H265RawSEIActiveParameterSets
class struct_H265RawSEIDecodedPictureHash(Struct): pass
struct_H265RawSEIDecodedPictureHash._fields_ = [
  ('hash_type', uint8_t),
  ('picture_md5', ((uint8_t * 16) * 3)),
  ('picture_crc', (uint16_t * 3)),
  ('picture_checksum', (uint32_t * 3)),
]
H265RawSEIDecodedPictureHash = struct_H265RawSEIDecodedPictureHash
class struct_H265RawSEITimeCode(Struct): pass
struct_H265RawSEITimeCode._fields_ = [
  ('num_clock_ts', uint8_t),
  ('clock_timestamp_flag', (uint8_t * 3)),
  ('units_field_based_flag', (uint8_t * 3)),
  ('counting_type', (uint8_t * 3)),
  ('full_timestamp_flag', (uint8_t * 3)),
  ('discontinuity_flag', (uint8_t * 3)),
  ('cnt_dropped_flag', (uint8_t * 3)),
  ('n_frames', (uint16_t * 3)),
  ('seconds_value', (uint8_t * 3)),
  ('minutes_value', (uint8_t * 3)),
  ('hours_value', (uint8_t * 3)),
  ('seconds_flag', (uint8_t * 3)),
  ('minutes_flag', (uint8_t * 3)),
  ('hours_flag', (uint8_t * 3)),
  ('time_offset_length', (uint8_t * 3)),
  ('time_offset_value', (int32_t * 3)),
]
H265RawSEITimeCode = struct_H265RawSEITimeCode
class struct_H265RawSEIAlphaChannelInfo(Struct): pass
struct_H265RawSEIAlphaChannelInfo._fields_ = [
  ('alpha_channel_cancel_flag', uint8_t),
  ('alpha_channel_use_idc', uint8_t),
  ('alpha_channel_bit_depth_minus8', uint8_t),
  ('alpha_transparent_value', uint16_t),
  ('alpha_opaque_value', uint16_t),
  ('alpha_channel_incr_flag', uint8_t),
  ('alpha_channel_clip_flag', uint8_t),
  ('alpha_channel_clip_type_flag', uint8_t),
]
H265RawSEIAlphaChannelInfo = struct_H265RawSEIAlphaChannelInfo
class struct_H265RawSEI3DReferenceDisplaysInfo(Struct): pass
struct_H265RawSEI3DReferenceDisplaysInfo._fields_ = [
  ('prec_ref_display_width', uint8_t),
  ('ref_viewing_distance_flag', uint8_t),
  ('prec_ref_viewing_dist', uint8_t),
  ('num_ref_displays_minus1', uint8_t),
  ('left_view_id', (uint16_t * 32)),
  ('right_view_id', (uint16_t * 32)),
  ('exponent_ref_display_width', (uint8_t * 32)),
  ('mantissa_ref_display_width', (uint8_t * 32)),
  ('exponent_ref_viewing_distance', (uint8_t * 32)),
  ('mantissa_ref_viewing_distance', (uint8_t * 32)),
  ('additional_shift_present_flag', (uint8_t * 32)),
  ('num_sample_shift_plus512', (uint16_t * 32)),
  ('three_dimensional_reference_displays_extension_flag', uint8_t),
]
H265RawSEI3DReferenceDisplaysInfo = struct_H265RawSEI3DReferenceDisplaysInfo
class struct_H265RawSEI(Struct): pass
class struct_SEIRawMessageList(Struct): pass
SEIRawMessageList = struct_SEIRawMessageList
class struct_SEIRawMessage(Struct): pass
SEIRawMessage = struct_SEIRawMessage
size_t = ctypes.c_uint64
struct_SEIRawMessage._fields_ = [
  ('payload_type', uint32_t),
  ('payload_size', uint32_t),
  ('payload', ctypes.c_void_p),
  ('payload_ref', ctypes.c_void_p),
  ('extension_data', ctypes.POINTER(uint8_t)),
  ('extension_bit_length', size_t),
]
struct_SEIRawMessageList._fields_ = [
  ('messages', ctypes.POINTER(SEIRawMessage)),
  ('nb_messages', ctypes.c_int32),
  ('nb_messages_allocated', ctypes.c_int32),
]
struct_H265RawSEI._fields_ = [
  ('nal_unit_header', H265RawNALUnitHeader),
  ('message_list', SEIRawMessageList),
]
H265RawSEI = struct_H265RawSEI
class struct_H265RawFiller(Struct): pass
struct_H265RawFiller._fields_ = [
  ('nal_unit_header', H265RawNALUnitHeader),
  ('filler_size', uint32_t),
]
H265RawFiller = struct_H265RawFiller
class struct_CodedBitstreamH265Context(Struct): pass
CodedBitstreamH265Context = struct_CodedBitstreamH265Context
