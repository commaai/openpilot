# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
import gzip, base64
dll = DLL('mesa', ['tinymesa_cpu', 'tinymesa'])
class struct_u_printf_info(Struct): pass
u_printf_info = struct_u_printf_info
uint32_t = ctypes.c_uint32
try: nir_debug = uint32_t.in_dll(dll, 'nir_debug')
except (ValueError,AttributeError): pass
try: nir_debug_print_shader = (ctypes.c_bool * 15).in_dll(dll, 'nir_debug_print_shader')
except (ValueError,AttributeError): pass
nir_component_mask_t = ctypes.c_uint16
try: (nir_process_debug_variable:=dll.nir_process_debug_variable).restype, nir_process_debug_variable.argtypes = None, []
except AttributeError: pass

try: (nir_component_mask_can_reinterpret:=dll.nir_component_mask_can_reinterpret).restype, nir_component_mask_can_reinterpret.argtypes = ctypes.c_bool, [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_component_mask_reinterpret:=dll.nir_component_mask_reinterpret).restype, nir_component_mask_reinterpret.argtypes = nir_component_mask_t, [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

class struct_nir_state_slot(Struct): pass
gl_state_index16 = ctypes.c_int16
struct_nir_state_slot._fields_ = [
  ('tokens', (gl_state_index16 * 4)),
]
nir_state_slot = struct_nir_state_slot
nir_rounding_mode = CEnum(ctypes.c_uint32)
nir_rounding_mode_undef = nir_rounding_mode.define('nir_rounding_mode_undef', 0)
nir_rounding_mode_rtne = nir_rounding_mode.define('nir_rounding_mode_rtne', 1)
nir_rounding_mode_ru = nir_rounding_mode.define('nir_rounding_mode_ru', 2)
nir_rounding_mode_rd = nir_rounding_mode.define('nir_rounding_mode_rd', 3)
nir_rounding_mode_rtz = nir_rounding_mode.define('nir_rounding_mode_rtz', 4)

nir_ray_query_value = CEnum(ctypes.c_uint32)
nir_ray_query_value_intersection_type = nir_ray_query_value.define('nir_ray_query_value_intersection_type', 0)
nir_ray_query_value_intersection_t = nir_ray_query_value.define('nir_ray_query_value_intersection_t', 1)
nir_ray_query_value_intersection_instance_custom_index = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_custom_index', 2)
nir_ray_query_value_intersection_instance_id = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_id', 3)
nir_ray_query_value_intersection_instance_sbt_index = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_sbt_index', 4)
nir_ray_query_value_intersection_geometry_index = nir_ray_query_value.define('nir_ray_query_value_intersection_geometry_index', 5)
nir_ray_query_value_intersection_primitive_index = nir_ray_query_value.define('nir_ray_query_value_intersection_primitive_index', 6)
nir_ray_query_value_intersection_barycentrics = nir_ray_query_value.define('nir_ray_query_value_intersection_barycentrics', 7)
nir_ray_query_value_intersection_front_face = nir_ray_query_value.define('nir_ray_query_value_intersection_front_face', 8)
nir_ray_query_value_intersection_object_ray_direction = nir_ray_query_value.define('nir_ray_query_value_intersection_object_ray_direction', 9)
nir_ray_query_value_intersection_object_ray_origin = nir_ray_query_value.define('nir_ray_query_value_intersection_object_ray_origin', 10)
nir_ray_query_value_intersection_object_to_world = nir_ray_query_value.define('nir_ray_query_value_intersection_object_to_world', 11)
nir_ray_query_value_intersection_world_to_object = nir_ray_query_value.define('nir_ray_query_value_intersection_world_to_object', 12)
nir_ray_query_value_intersection_candidate_aabb_opaque = nir_ray_query_value.define('nir_ray_query_value_intersection_candidate_aabb_opaque', 13)
nir_ray_query_value_tmin = nir_ray_query_value.define('nir_ray_query_value_tmin', 14)
nir_ray_query_value_flags = nir_ray_query_value.define('nir_ray_query_value_flags', 15)
nir_ray_query_value_world_ray_direction = nir_ray_query_value.define('nir_ray_query_value_world_ray_direction', 16)
nir_ray_query_value_world_ray_origin = nir_ray_query_value.define('nir_ray_query_value_world_ray_origin', 17)
nir_ray_query_value_intersection_triangle_vertex_positions = nir_ray_query_value.define('nir_ray_query_value_intersection_triangle_vertex_positions', 18)

nir_resource_data_intel = CEnum(ctypes.c_uint32)
nir_resource_intel_bindless = nir_resource_data_intel.define('nir_resource_intel_bindless', 1)
nir_resource_intel_pushable = nir_resource_data_intel.define('nir_resource_intel_pushable', 2)
nir_resource_intel_sampler = nir_resource_data_intel.define('nir_resource_intel_sampler', 4)
nir_resource_intel_non_uniform = nir_resource_data_intel.define('nir_resource_intel_non_uniform', 8)
nir_resource_intel_sampler_embedded = nir_resource_data_intel.define('nir_resource_intel_sampler_embedded', 16)

nir_preamble_class = CEnum(ctypes.c_uint32)
nir_preamble_class_general = nir_preamble_class.define('nir_preamble_class_general', 0)
nir_preamble_class_image = nir_preamble_class.define('nir_preamble_class_image', 1)
nir_preamble_num_classes = nir_preamble_class.define('nir_preamble_num_classes', 2)

nir_cmat_signed = CEnum(ctypes.c_uint32)
NIR_CMAT_A_SIGNED = nir_cmat_signed.define('NIR_CMAT_A_SIGNED', 1)
NIR_CMAT_B_SIGNED = nir_cmat_signed.define('NIR_CMAT_B_SIGNED', 2)
NIR_CMAT_C_SIGNED = nir_cmat_signed.define('NIR_CMAT_C_SIGNED', 4)
NIR_CMAT_RESULT_SIGNED = nir_cmat_signed.define('NIR_CMAT_RESULT_SIGNED', 8)

class nir_const_value(ctypes.Union): pass
int8_t = ctypes.c_byte
uint8_t = ctypes.c_ubyte
int16_t = ctypes.c_int16
uint16_t = ctypes.c_uint16
int32_t = ctypes.c_int32
int64_t = ctypes.c_int64
uint64_t = ctypes.c_uint64
nir_const_value._fields_ = [
  ('b', ctypes.c_bool),
  ('f32', ctypes.c_float),
  ('f64', ctypes.c_double),
  ('i8', int8_t),
  ('u8', uint8_t),
  ('i16', int16_t),
  ('u16', uint16_t),
  ('i32', int32_t),
  ('u32', uint32_t),
  ('i64', int64_t),
  ('u64', uint64_t),
]
try: (nir_const_value_for_float:=dll.nir_const_value_for_float).restype, nir_const_value_for_float.argtypes = nir_const_value, [ctypes.c_double, ctypes.c_uint32]
except AttributeError: pass

try: (nir_const_value_as_float:=dll.nir_const_value_as_float).restype, nir_const_value_as_float.argtypes = ctypes.c_double, [nir_const_value, ctypes.c_uint32]
except AttributeError: pass

class struct_nir_constant(Struct): pass
nir_constant = struct_nir_constant
struct_nir_constant._fields_ = [
  ('values', (nir_const_value * 16)),
  ('is_null_constant', ctypes.c_bool),
  ('num_elements', ctypes.c_uint32),
  ('elements', ctypes.POINTER(ctypes.POINTER(nir_constant))),
]
nir_depth_layout = CEnum(ctypes.c_uint32)
nir_depth_layout_none = nir_depth_layout.define('nir_depth_layout_none', 0)
nir_depth_layout_any = nir_depth_layout.define('nir_depth_layout_any', 1)
nir_depth_layout_greater = nir_depth_layout.define('nir_depth_layout_greater', 2)
nir_depth_layout_less = nir_depth_layout.define('nir_depth_layout_less', 3)
nir_depth_layout_unchanged = nir_depth_layout.define('nir_depth_layout_unchanged', 4)

nir_var_declaration_type = CEnum(ctypes.c_uint32)
nir_var_declared_normally = nir_var_declaration_type.define('nir_var_declared_normally', 0)
nir_var_declared_implicitly = nir_var_declaration_type.define('nir_var_declared_implicitly', 1)
nir_var_hidden = nir_var_declaration_type.define('nir_var_hidden', 2)

class struct_nir_variable_data(Struct): pass
class struct_nir_variable_data_0(ctypes.Union): pass
class struct_nir_variable_data_0_image(Struct): pass
enum_pipe_format = CEnum(ctypes.c_uint32)
PIPE_FORMAT_NONE = enum_pipe_format.define('PIPE_FORMAT_NONE', 0)
PIPE_FORMAT_R64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64_UINT', 1)
PIPE_FORMAT_R64G64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64_UINT', 2)
PIPE_FORMAT_R64G64B64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_UINT', 3)
PIPE_FORMAT_R64G64B64A64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_UINT', 4)
PIPE_FORMAT_R64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64_SINT', 5)
PIPE_FORMAT_R64G64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64_SINT', 6)
PIPE_FORMAT_R64G64B64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_SINT', 7)
PIPE_FORMAT_R64G64B64A64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_SINT', 8)
PIPE_FORMAT_R64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64_FLOAT', 9)
PIPE_FORMAT_R64G64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64_FLOAT', 10)
PIPE_FORMAT_R64G64B64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_FLOAT', 11)
PIPE_FORMAT_R64G64B64A64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_FLOAT', 12)
PIPE_FORMAT_R32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32_FLOAT', 13)
PIPE_FORMAT_R32G32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32_FLOAT', 14)
PIPE_FORMAT_R32G32B32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_FLOAT', 15)
PIPE_FORMAT_R32G32B32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_FLOAT', 16)
PIPE_FORMAT_R32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32_UNORM', 17)
PIPE_FORMAT_R32G32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32_UNORM', 18)
PIPE_FORMAT_R32G32B32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_UNORM', 19)
PIPE_FORMAT_R32G32B32A32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_UNORM', 20)
PIPE_FORMAT_R32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32_USCALED', 21)
PIPE_FORMAT_R32G32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32_USCALED', 22)
PIPE_FORMAT_R32G32B32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_USCALED', 23)
PIPE_FORMAT_R32G32B32A32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_USCALED', 24)
PIPE_FORMAT_R32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32_SNORM', 25)
PIPE_FORMAT_R32G32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32_SNORM', 26)
PIPE_FORMAT_R32G32B32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SNORM', 27)
PIPE_FORMAT_R32G32B32A32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SNORM', 28)
PIPE_FORMAT_R32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32_SSCALED', 29)
PIPE_FORMAT_R32G32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32_SSCALED', 30)
PIPE_FORMAT_R32G32B32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SSCALED', 31)
PIPE_FORMAT_R32G32B32A32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SSCALED', 32)
PIPE_FORMAT_R16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16_UNORM', 33)
PIPE_FORMAT_R16G16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16_UNORM', 34)
PIPE_FORMAT_R16G16B16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_UNORM', 35)
PIPE_FORMAT_R16G16B16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_UNORM', 36)
PIPE_FORMAT_R16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16_USCALED', 37)
PIPE_FORMAT_R16G16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16_USCALED', 38)
PIPE_FORMAT_R16G16B16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_USCALED', 39)
PIPE_FORMAT_R16G16B16A16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_USCALED', 40)
PIPE_FORMAT_R16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16_SNORM', 41)
PIPE_FORMAT_R16G16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16_SNORM', 42)
PIPE_FORMAT_R16G16B16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SNORM', 43)
PIPE_FORMAT_R16G16B16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SNORM', 44)
PIPE_FORMAT_R16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16_SSCALED', 45)
PIPE_FORMAT_R16G16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16_SSCALED', 46)
PIPE_FORMAT_R16G16B16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SSCALED', 47)
PIPE_FORMAT_R16G16B16A16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SSCALED', 48)
PIPE_FORMAT_R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_UNORM', 49)
PIPE_FORMAT_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_UNORM', 50)
PIPE_FORMAT_R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_UNORM', 51)
PIPE_FORMAT_B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_UNORM', 52)
PIPE_FORMAT_R8G8B8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_UNORM', 53)
PIPE_FORMAT_B8G8R8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_UNORM', 54)
PIPE_FORMAT_R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8_USCALED', 55)
PIPE_FORMAT_R8G8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8_USCALED', 56)
PIPE_FORMAT_R8G8B8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_USCALED', 57)
PIPE_FORMAT_B8G8R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_USCALED', 58)
PIPE_FORMAT_R8G8B8A8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_USCALED', 59)
PIPE_FORMAT_B8G8R8A8_USCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_USCALED', 60)
PIPE_FORMAT_A8B8G8R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_USCALED', 61)
PIPE_FORMAT_R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8_SNORM', 62)
PIPE_FORMAT_R8G8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_SNORM', 63)
PIPE_FORMAT_R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SNORM', 64)
PIPE_FORMAT_B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SNORM', 65)
PIPE_FORMAT_R8G8B8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SNORM', 66)
PIPE_FORMAT_B8G8R8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SNORM', 67)
PIPE_FORMAT_R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8_SSCALED', 68)
PIPE_FORMAT_R8G8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8_SSCALED', 69)
PIPE_FORMAT_R8G8B8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SSCALED', 70)
PIPE_FORMAT_B8G8R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SSCALED', 71)
PIPE_FORMAT_R8G8B8A8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SSCALED', 72)
PIPE_FORMAT_B8G8R8A8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SSCALED', 73)
PIPE_FORMAT_A8B8G8R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SSCALED', 74)
PIPE_FORMAT_A8R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_UNORM', 75)
PIPE_FORMAT_R32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32_FIXED', 76)
PIPE_FORMAT_R32G32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32_FIXED', 77)
PIPE_FORMAT_R32G32B32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_FIXED', 78)
PIPE_FORMAT_R32G32B32A32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_FIXED', 79)
PIPE_FORMAT_R16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16_FLOAT', 80)
PIPE_FORMAT_R16G16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16_FLOAT', 81)
PIPE_FORMAT_R16G16B16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_FLOAT', 82)
PIPE_FORMAT_R16G16B16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_FLOAT', 83)
PIPE_FORMAT_R8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8_UINT', 84)
PIPE_FORMAT_R8G8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8_UINT', 85)
PIPE_FORMAT_R8G8B8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_UINT', 86)
PIPE_FORMAT_B8G8R8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_UINT', 87)
PIPE_FORMAT_R8G8B8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_UINT', 88)
PIPE_FORMAT_B8G8R8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_UINT', 89)
PIPE_FORMAT_R8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8_SINT', 90)
PIPE_FORMAT_R8G8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8_SINT', 91)
PIPE_FORMAT_R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SINT', 92)
PIPE_FORMAT_B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SINT', 93)
PIPE_FORMAT_R8G8B8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SINT', 94)
PIPE_FORMAT_B8G8R8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SINT', 95)
PIPE_FORMAT_R16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16_UINT', 96)
PIPE_FORMAT_R16G16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16_UINT', 97)
PIPE_FORMAT_R16G16B16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_UINT', 98)
PIPE_FORMAT_R16G16B16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_UINT', 99)
PIPE_FORMAT_R16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16_SINT', 100)
PIPE_FORMAT_R16G16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16_SINT', 101)
PIPE_FORMAT_R16G16B16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SINT', 102)
PIPE_FORMAT_R16G16B16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SINT', 103)
PIPE_FORMAT_R32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32_UINT', 104)
PIPE_FORMAT_R32G32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32_UINT', 105)
PIPE_FORMAT_R32G32B32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_UINT', 106)
PIPE_FORMAT_R32G32B32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_UINT', 107)
PIPE_FORMAT_R32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32_SINT', 108)
PIPE_FORMAT_R32G32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32_SINT', 109)
PIPE_FORMAT_R32G32B32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SINT', 110)
PIPE_FORMAT_R32G32B32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SINT', 111)
PIPE_FORMAT_R10G10B10A2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_UNORM', 112)
PIPE_FORMAT_R10G10B10A2_SNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SNORM', 113)
PIPE_FORMAT_R10G10B10A2_USCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_USCALED', 114)
PIPE_FORMAT_R10G10B10A2_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SSCALED', 115)
PIPE_FORMAT_B10G10R10A2_UNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_UNORM', 116)
PIPE_FORMAT_B10G10R10A2_SNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SNORM', 117)
PIPE_FORMAT_B10G10R10A2_USCALED = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_USCALED', 118)
PIPE_FORMAT_B10G10R10A2_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SSCALED', 119)
PIPE_FORMAT_R11G11B10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R11G11B10_FLOAT', 120)
PIPE_FORMAT_R10G10B10A2_UINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_UINT', 121)
PIPE_FORMAT_R10G10B10A2_SINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SINT', 122)
PIPE_FORMAT_B10G10R10A2_UINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_UINT', 123)
PIPE_FORMAT_B10G10R10A2_SINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SINT', 124)
PIPE_FORMAT_B8G8R8X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_UNORM', 125)
PIPE_FORMAT_X8B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_UNORM', 126)
PIPE_FORMAT_X8R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_UNORM', 127)
PIPE_FORMAT_B5G5R5A1_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G5R5A1_UNORM', 128)
PIPE_FORMAT_R4G4B4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4G4B4A4_UNORM', 129)
PIPE_FORMAT_B4G4R4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_B4G4R4A4_UNORM', 130)
PIPE_FORMAT_R5G6B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_UNORM', 131)
PIPE_FORMAT_B5G6R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_UNORM', 132)
PIPE_FORMAT_L8_UNORM = enum_pipe_format.define('PIPE_FORMAT_L8_UNORM', 133)
PIPE_FORMAT_A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8_UNORM', 134)
PIPE_FORMAT_I8_UNORM = enum_pipe_format.define('PIPE_FORMAT_I8_UNORM', 135)
PIPE_FORMAT_L8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_L8A8_UNORM', 136)
PIPE_FORMAT_L16_UNORM = enum_pipe_format.define('PIPE_FORMAT_L16_UNORM', 137)
PIPE_FORMAT_UYVY = enum_pipe_format.define('PIPE_FORMAT_UYVY', 138)
PIPE_FORMAT_VYUY = enum_pipe_format.define('PIPE_FORMAT_VYUY', 139)
PIPE_FORMAT_YUYV = enum_pipe_format.define('PIPE_FORMAT_YUYV', 140)
PIPE_FORMAT_YVYU = enum_pipe_format.define('PIPE_FORMAT_YVYU', 141)
PIPE_FORMAT_Z16_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z16_UNORM', 142)
PIPE_FORMAT_Z16_UNORM_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_Z16_UNORM_S8_UINT', 143)
PIPE_FORMAT_Z32_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z32_UNORM', 144)
PIPE_FORMAT_Z32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_Z32_FLOAT', 145)
PIPE_FORMAT_Z24_UNORM_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_Z24_UNORM_S8_UINT', 146)
PIPE_FORMAT_S8_UINT_Z24_UNORM = enum_pipe_format.define('PIPE_FORMAT_S8_UINT_Z24_UNORM', 147)
PIPE_FORMAT_Z24X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z24X8_UNORM', 148)
PIPE_FORMAT_X8Z24_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8Z24_UNORM', 149)
PIPE_FORMAT_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_S8_UINT', 150)
PIPE_FORMAT_L8_SRGB = enum_pipe_format.define('PIPE_FORMAT_L8_SRGB', 151)
PIPE_FORMAT_R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8_SRGB', 152)
PIPE_FORMAT_L8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_L8A8_SRGB', 153)
PIPE_FORMAT_R8G8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8_SRGB', 154)
PIPE_FORMAT_R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SRGB', 155)
PIPE_FORMAT_B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SRGB', 156)
PIPE_FORMAT_A8B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SRGB', 157)
PIPE_FORMAT_X8B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SRGB', 158)
PIPE_FORMAT_B8G8R8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SRGB', 159)
PIPE_FORMAT_B8G8R8X8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SRGB', 160)
PIPE_FORMAT_A8R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SRGB', 161)
PIPE_FORMAT_X8R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SRGB', 162)
PIPE_FORMAT_R8G8B8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SRGB', 163)
PIPE_FORMAT_DXT1_RGB = enum_pipe_format.define('PIPE_FORMAT_DXT1_RGB', 164)
PIPE_FORMAT_DXT1_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT1_RGBA', 165)
PIPE_FORMAT_DXT3_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT3_RGBA', 166)
PIPE_FORMAT_DXT5_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT5_RGBA', 167)
PIPE_FORMAT_DXT1_SRGB = enum_pipe_format.define('PIPE_FORMAT_DXT1_SRGB', 168)
PIPE_FORMAT_DXT1_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT1_SRGBA', 169)
PIPE_FORMAT_DXT3_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT3_SRGBA', 170)
PIPE_FORMAT_DXT5_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT5_SRGBA', 171)
PIPE_FORMAT_RGTC1_UNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC1_UNORM', 172)
PIPE_FORMAT_RGTC1_SNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC1_SNORM', 173)
PIPE_FORMAT_RGTC2_UNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC2_UNORM', 174)
PIPE_FORMAT_RGTC2_SNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC2_SNORM', 175)
PIPE_FORMAT_R8G8_B8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_B8G8_UNORM', 176)
PIPE_FORMAT_G8R8_G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_G8B8_UNORM', 177)
PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM', 178)
PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM', 179)
PIPE_FORMAT_X6R10_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6R10_UNORM', 180)
PIPE_FORMAT_X6R10X6G10_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6R10X6G10_UNORM', 181)
PIPE_FORMAT_X4R12_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4R12_UNORM', 182)
PIPE_FORMAT_X4R12X4G12_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4R12X4G12_UNORM', 183)
PIPE_FORMAT_R8SG8SB8UX8U_NORM = enum_pipe_format.define('PIPE_FORMAT_R8SG8SB8UX8U_NORM', 184)
PIPE_FORMAT_R5SG5SB6U_NORM = enum_pipe_format.define('PIPE_FORMAT_R5SG5SB6U_NORM', 185)
PIPE_FORMAT_A8B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_UNORM', 186)
PIPE_FORMAT_B5G5R5X1_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G5R5X1_UNORM', 187)
PIPE_FORMAT_R9G9B9E5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R9G9B9E5_FLOAT', 188)
PIPE_FORMAT_Z32_FLOAT_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_Z32_FLOAT_S8X24_UINT', 189)
PIPE_FORMAT_R1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R1_UNORM', 190)
PIPE_FORMAT_R10G10B10X2_USCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_USCALED', 191)
PIPE_FORMAT_R10G10B10X2_SNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_SNORM', 192)
PIPE_FORMAT_L4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_L4A4_UNORM', 193)
PIPE_FORMAT_A2R10G10B10_UNORM = enum_pipe_format.define('PIPE_FORMAT_A2R10G10B10_UNORM', 194)
PIPE_FORMAT_A2B10G10R10_UNORM = enum_pipe_format.define('PIPE_FORMAT_A2B10G10R10_UNORM', 195)
PIPE_FORMAT_R10SG10SB10SA2U_NORM = enum_pipe_format.define('PIPE_FORMAT_R10SG10SB10SA2U_NORM', 196)
PIPE_FORMAT_R8G8Bx_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8Bx_SNORM', 197)
PIPE_FORMAT_R8G8B8X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_UNORM', 198)
PIPE_FORMAT_B4G4R4X4_UNORM = enum_pipe_format.define('PIPE_FORMAT_B4G4R4X4_UNORM', 199)
PIPE_FORMAT_X24S8_UINT = enum_pipe_format.define('PIPE_FORMAT_X24S8_UINT', 200)
PIPE_FORMAT_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_S8X24_UINT', 201)
PIPE_FORMAT_X32_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_X32_S8X24_UINT', 202)
PIPE_FORMAT_R3G3B2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R3G3B2_UNORM', 203)
PIPE_FORMAT_B2G3R3_UNORM = enum_pipe_format.define('PIPE_FORMAT_B2G3R3_UNORM', 204)
PIPE_FORMAT_L16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_L16A16_UNORM', 205)
PIPE_FORMAT_A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_A16_UNORM', 206)
PIPE_FORMAT_I16_UNORM = enum_pipe_format.define('PIPE_FORMAT_I16_UNORM', 207)
PIPE_FORMAT_LATC1_UNORM = enum_pipe_format.define('PIPE_FORMAT_LATC1_UNORM', 208)
PIPE_FORMAT_LATC1_SNORM = enum_pipe_format.define('PIPE_FORMAT_LATC1_SNORM', 209)
PIPE_FORMAT_LATC2_UNORM = enum_pipe_format.define('PIPE_FORMAT_LATC2_UNORM', 210)
PIPE_FORMAT_LATC2_SNORM = enum_pipe_format.define('PIPE_FORMAT_LATC2_SNORM', 211)
PIPE_FORMAT_A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8_SNORM', 212)
PIPE_FORMAT_L8_SNORM = enum_pipe_format.define('PIPE_FORMAT_L8_SNORM', 213)
PIPE_FORMAT_L8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_L8A8_SNORM', 214)
PIPE_FORMAT_I8_SNORM = enum_pipe_format.define('PIPE_FORMAT_I8_SNORM', 215)
PIPE_FORMAT_A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_A16_SNORM', 216)
PIPE_FORMAT_L16_SNORM = enum_pipe_format.define('PIPE_FORMAT_L16_SNORM', 217)
PIPE_FORMAT_L16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_L16A16_SNORM', 218)
PIPE_FORMAT_I16_SNORM = enum_pipe_format.define('PIPE_FORMAT_I16_SNORM', 219)
PIPE_FORMAT_A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_A16_FLOAT', 220)
PIPE_FORMAT_L16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L16_FLOAT', 221)
PIPE_FORMAT_L16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L16A16_FLOAT', 222)
PIPE_FORMAT_I16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_I16_FLOAT', 223)
PIPE_FORMAT_A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_A32_FLOAT', 224)
PIPE_FORMAT_L32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L32_FLOAT', 225)
PIPE_FORMAT_L32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L32A32_FLOAT', 226)
PIPE_FORMAT_I32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_I32_FLOAT', 227)
PIPE_FORMAT_YV12 = enum_pipe_format.define('PIPE_FORMAT_YV12', 228)
PIPE_FORMAT_YV16 = enum_pipe_format.define('PIPE_FORMAT_YV16', 229)
PIPE_FORMAT_IYUV = enum_pipe_format.define('PIPE_FORMAT_IYUV', 230)
PIPE_FORMAT_NV12 = enum_pipe_format.define('PIPE_FORMAT_NV12', 231)
PIPE_FORMAT_NV21 = enum_pipe_format.define('PIPE_FORMAT_NV21', 232)
PIPE_FORMAT_NV16 = enum_pipe_format.define('PIPE_FORMAT_NV16', 233)
PIPE_FORMAT_NV15 = enum_pipe_format.define('PIPE_FORMAT_NV15', 234)
PIPE_FORMAT_NV20 = enum_pipe_format.define('PIPE_FORMAT_NV20', 235)
PIPE_FORMAT_Y8_400_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_400_UNORM', 236)
PIPE_FORMAT_Y8_U8_V8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_422_UNORM', 237)
PIPE_FORMAT_Y8_U8_V8_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_444_UNORM', 238)
PIPE_FORMAT_Y8_U8_V8_440_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_440_UNORM', 239)
PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM', 240)
PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM', 241)
PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM', 242)
PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM', 243)
PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM', 244)
PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM', 245)
PIPE_FORMAT_Y16_U16_V16_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_420_UNORM', 246)
PIPE_FORMAT_Y16_U16_V16_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_422_UNORM', 247)
PIPE_FORMAT_Y16_U16V16_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16V16_422_UNORM', 248)
PIPE_FORMAT_Y16_U16_V16_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_444_UNORM', 249)
PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED', 250)
PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED', 251)
PIPE_FORMAT_A4R4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4R4_UNORM', 252)
PIPE_FORMAT_R4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4A4_UNORM', 253)
PIPE_FORMAT_R8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8A8_UNORM', 254)
PIPE_FORMAT_A8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8_UNORM', 255)
PIPE_FORMAT_A8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8_UINT', 256)
PIPE_FORMAT_I8_UINT = enum_pipe_format.define('PIPE_FORMAT_I8_UINT', 257)
PIPE_FORMAT_L8_UINT = enum_pipe_format.define('PIPE_FORMAT_L8_UINT', 258)
PIPE_FORMAT_L8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_L8A8_UINT', 259)
PIPE_FORMAT_A8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8_SINT', 260)
PIPE_FORMAT_I8_SINT = enum_pipe_format.define('PIPE_FORMAT_I8_SINT', 261)
PIPE_FORMAT_L8_SINT = enum_pipe_format.define('PIPE_FORMAT_L8_SINT', 262)
PIPE_FORMAT_L8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_L8A8_SINT', 263)
PIPE_FORMAT_A16_UINT = enum_pipe_format.define('PIPE_FORMAT_A16_UINT', 264)
PIPE_FORMAT_I16_UINT = enum_pipe_format.define('PIPE_FORMAT_I16_UINT', 265)
PIPE_FORMAT_L16_UINT = enum_pipe_format.define('PIPE_FORMAT_L16_UINT', 266)
PIPE_FORMAT_L16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_L16A16_UINT', 267)
PIPE_FORMAT_A16_SINT = enum_pipe_format.define('PIPE_FORMAT_A16_SINT', 268)
PIPE_FORMAT_I16_SINT = enum_pipe_format.define('PIPE_FORMAT_I16_SINT', 269)
PIPE_FORMAT_L16_SINT = enum_pipe_format.define('PIPE_FORMAT_L16_SINT', 270)
PIPE_FORMAT_L16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_L16A16_SINT', 271)
PIPE_FORMAT_A32_UINT = enum_pipe_format.define('PIPE_FORMAT_A32_UINT', 272)
PIPE_FORMAT_I32_UINT = enum_pipe_format.define('PIPE_FORMAT_I32_UINT', 273)
PIPE_FORMAT_L32_UINT = enum_pipe_format.define('PIPE_FORMAT_L32_UINT', 274)
PIPE_FORMAT_L32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_L32A32_UINT', 275)
PIPE_FORMAT_A32_SINT = enum_pipe_format.define('PIPE_FORMAT_A32_SINT', 276)
PIPE_FORMAT_I32_SINT = enum_pipe_format.define('PIPE_FORMAT_I32_SINT', 277)
PIPE_FORMAT_L32_SINT = enum_pipe_format.define('PIPE_FORMAT_L32_SINT', 278)
PIPE_FORMAT_L32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_L32A32_SINT', 279)
PIPE_FORMAT_A8R8G8B8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_UINT', 280)
PIPE_FORMAT_A8B8G8R8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_UINT', 281)
PIPE_FORMAT_A2R10G10B10_UINT = enum_pipe_format.define('PIPE_FORMAT_A2R10G10B10_UINT', 282)
PIPE_FORMAT_A2B10G10R10_UINT = enum_pipe_format.define('PIPE_FORMAT_A2B10G10R10_UINT', 283)
PIPE_FORMAT_R5G6B5_UINT = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_UINT', 284)
PIPE_FORMAT_B5G6R5_UINT = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_UINT', 285)
PIPE_FORMAT_R5G5B5A1_UINT = enum_pipe_format.define('PIPE_FORMAT_R5G5B5A1_UINT', 286)
PIPE_FORMAT_B5G5R5A1_UINT = enum_pipe_format.define('PIPE_FORMAT_B5G5R5A1_UINT', 287)
PIPE_FORMAT_A1R5G5B5_UINT = enum_pipe_format.define('PIPE_FORMAT_A1R5G5B5_UINT', 288)
PIPE_FORMAT_A1B5G5R5_UINT = enum_pipe_format.define('PIPE_FORMAT_A1B5G5R5_UINT', 289)
PIPE_FORMAT_R4G4B4A4_UINT = enum_pipe_format.define('PIPE_FORMAT_R4G4B4A4_UINT', 290)
PIPE_FORMAT_B4G4R4A4_UINT = enum_pipe_format.define('PIPE_FORMAT_B4G4R4A4_UINT', 291)
PIPE_FORMAT_A4R4G4B4_UINT = enum_pipe_format.define('PIPE_FORMAT_A4R4G4B4_UINT', 292)
PIPE_FORMAT_A4B4G4R4_UINT = enum_pipe_format.define('PIPE_FORMAT_A4B4G4R4_UINT', 293)
PIPE_FORMAT_R3G3B2_UINT = enum_pipe_format.define('PIPE_FORMAT_R3G3B2_UINT', 294)
PIPE_FORMAT_B2G3R3_UINT = enum_pipe_format.define('PIPE_FORMAT_B2G3R3_UINT', 295)
PIPE_FORMAT_ETC1_RGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC1_RGB8', 296)
PIPE_FORMAT_R8G8_R8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_R8B8_UNORM', 297)
PIPE_FORMAT_R8B8_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8B8_R8G8_UNORM', 298)
PIPE_FORMAT_G8R8_B8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_B8R8_UNORM', 299)
PIPE_FORMAT_B8R8_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8R8_G8R8_UNORM', 300)
PIPE_FORMAT_G8B8_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8B8_G8R8_UNORM', 301)
PIPE_FORMAT_B8G8_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8_R8G8_UNORM', 302)
PIPE_FORMAT_R8G8B8X8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SNORM', 303)
PIPE_FORMAT_R8G8B8X8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SRGB', 304)
PIPE_FORMAT_R8G8B8X8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_UINT', 305)
PIPE_FORMAT_R8G8B8X8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SINT', 306)
PIPE_FORMAT_B10G10R10X2_UNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_UNORM', 307)
PIPE_FORMAT_R16G16B16X16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_UNORM', 308)
PIPE_FORMAT_R16G16B16X16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_SNORM', 309)
PIPE_FORMAT_R16G16B16X16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_FLOAT', 310)
PIPE_FORMAT_R16G16B16X16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_UINT', 311)
PIPE_FORMAT_R16G16B16X16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_SINT', 312)
PIPE_FORMAT_R32G32B32X32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_FLOAT', 313)
PIPE_FORMAT_R32G32B32X32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_UINT', 314)
PIPE_FORMAT_R32G32B32X32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_SINT', 315)
PIPE_FORMAT_R8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8A8_SNORM', 316)
PIPE_FORMAT_R16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16A16_UNORM', 317)
PIPE_FORMAT_R16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16A16_SNORM', 318)
PIPE_FORMAT_R16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16A16_FLOAT', 319)
PIPE_FORMAT_R32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32A32_FLOAT', 320)
PIPE_FORMAT_R8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8A8_UINT', 321)
PIPE_FORMAT_R8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8A8_SINT', 322)
PIPE_FORMAT_R16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16A16_UINT', 323)
PIPE_FORMAT_R16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16A16_SINT', 324)
PIPE_FORMAT_R32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32A32_UINT', 325)
PIPE_FORMAT_R32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32A32_SINT', 326)
PIPE_FORMAT_B5G6R5_SRGB = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_SRGB', 327)
PIPE_FORMAT_BPTC_RGBA_UNORM = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGBA_UNORM', 328)
PIPE_FORMAT_BPTC_SRGBA = enum_pipe_format.define('PIPE_FORMAT_BPTC_SRGBA', 329)
PIPE_FORMAT_BPTC_RGB_FLOAT = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGB_FLOAT', 330)
PIPE_FORMAT_BPTC_RGB_UFLOAT = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGB_UFLOAT', 331)
PIPE_FORMAT_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_UNORM', 332)
PIPE_FORMAT_G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_SNORM', 333)
PIPE_FORMAT_G16R16_UNORM = enum_pipe_format.define('PIPE_FORMAT_G16R16_UNORM', 334)
PIPE_FORMAT_G16R16_SNORM = enum_pipe_format.define('PIPE_FORMAT_G16R16_SNORM', 335)
PIPE_FORMAT_A8B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SNORM', 336)
PIPE_FORMAT_X8B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SNORM', 337)
PIPE_FORMAT_ETC2_RGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGB8', 338)
PIPE_FORMAT_ETC2_SRGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGB8', 339)
PIPE_FORMAT_ETC2_RGB8A1 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGB8A1', 340)
PIPE_FORMAT_ETC2_SRGB8A1 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGB8A1', 341)
PIPE_FORMAT_ETC2_RGBA8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGBA8', 342)
PIPE_FORMAT_ETC2_SRGBA8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGBA8', 343)
PIPE_FORMAT_ETC2_R11_UNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_R11_UNORM', 344)
PIPE_FORMAT_ETC2_R11_SNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_R11_SNORM', 345)
PIPE_FORMAT_ETC2_RG11_UNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_RG11_UNORM', 346)
PIPE_FORMAT_ETC2_RG11_SNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_RG11_SNORM', 347)
PIPE_FORMAT_ASTC_4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4', 348)
PIPE_FORMAT_ASTC_5x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4', 349)
PIPE_FORMAT_ASTC_5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5', 350)
PIPE_FORMAT_ASTC_6x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5', 351)
PIPE_FORMAT_ASTC_6x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6', 352)
PIPE_FORMAT_ASTC_8x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5', 353)
PIPE_FORMAT_ASTC_8x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6', 354)
PIPE_FORMAT_ASTC_8x8 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8', 355)
PIPE_FORMAT_ASTC_10x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5', 356)
PIPE_FORMAT_ASTC_10x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6', 357)
PIPE_FORMAT_ASTC_10x8 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8', 358)
PIPE_FORMAT_ASTC_10x10 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10', 359)
PIPE_FORMAT_ASTC_12x10 = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10', 360)
PIPE_FORMAT_ASTC_12x12 = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12', 361)
PIPE_FORMAT_ASTC_4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4_SRGB', 362)
PIPE_FORMAT_ASTC_5x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4_SRGB', 363)
PIPE_FORMAT_ASTC_5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5_SRGB', 364)
PIPE_FORMAT_ASTC_6x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5_SRGB', 365)
PIPE_FORMAT_ASTC_6x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6_SRGB', 366)
PIPE_FORMAT_ASTC_8x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5_SRGB', 367)
PIPE_FORMAT_ASTC_8x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6_SRGB', 368)
PIPE_FORMAT_ASTC_8x8_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8_SRGB', 369)
PIPE_FORMAT_ASTC_10x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5_SRGB', 370)
PIPE_FORMAT_ASTC_10x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6_SRGB', 371)
PIPE_FORMAT_ASTC_10x8_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8_SRGB', 372)
PIPE_FORMAT_ASTC_10x10_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10_SRGB', 373)
PIPE_FORMAT_ASTC_12x10_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10_SRGB', 374)
PIPE_FORMAT_ASTC_12x12_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12_SRGB', 375)
PIPE_FORMAT_ASTC_3x3x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_3x3x3', 376)
PIPE_FORMAT_ASTC_4x3x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x3x3', 377)
PIPE_FORMAT_ASTC_4x4x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x3', 378)
PIPE_FORMAT_ASTC_4x4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x4', 379)
PIPE_FORMAT_ASTC_5x4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4x4', 380)
PIPE_FORMAT_ASTC_5x5x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x4', 381)
PIPE_FORMAT_ASTC_5x5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x5', 382)
PIPE_FORMAT_ASTC_6x5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5x5', 383)
PIPE_FORMAT_ASTC_6x6x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x5', 384)
PIPE_FORMAT_ASTC_6x6x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x6', 385)
PIPE_FORMAT_ASTC_3x3x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_3x3x3_SRGB', 386)
PIPE_FORMAT_ASTC_4x3x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x3x3_SRGB', 387)
PIPE_FORMAT_ASTC_4x4x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x3_SRGB', 388)
PIPE_FORMAT_ASTC_4x4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x4_SRGB', 389)
PIPE_FORMAT_ASTC_5x4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4x4_SRGB', 390)
PIPE_FORMAT_ASTC_5x5x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x4_SRGB', 391)
PIPE_FORMAT_ASTC_5x5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x5_SRGB', 392)
PIPE_FORMAT_ASTC_6x5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5x5_SRGB', 393)
PIPE_FORMAT_ASTC_6x6x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x5_SRGB', 394)
PIPE_FORMAT_ASTC_6x6x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x6_SRGB', 395)
PIPE_FORMAT_ASTC_4x4_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4_FLOAT', 396)
PIPE_FORMAT_ASTC_5x4_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4_FLOAT', 397)
PIPE_FORMAT_ASTC_5x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5_FLOAT', 398)
PIPE_FORMAT_ASTC_6x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5_FLOAT', 399)
PIPE_FORMAT_ASTC_6x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6_FLOAT', 400)
PIPE_FORMAT_ASTC_8x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5_FLOAT', 401)
PIPE_FORMAT_ASTC_8x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6_FLOAT', 402)
PIPE_FORMAT_ASTC_8x8_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8_FLOAT', 403)
PIPE_FORMAT_ASTC_10x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5_FLOAT', 404)
PIPE_FORMAT_ASTC_10x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6_FLOAT', 405)
PIPE_FORMAT_ASTC_10x8_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8_FLOAT', 406)
PIPE_FORMAT_ASTC_10x10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10_FLOAT', 407)
PIPE_FORMAT_ASTC_12x10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10_FLOAT', 408)
PIPE_FORMAT_ASTC_12x12_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12_FLOAT', 409)
PIPE_FORMAT_FXT1_RGB = enum_pipe_format.define('PIPE_FORMAT_FXT1_RGB', 410)
PIPE_FORMAT_FXT1_RGBA = enum_pipe_format.define('PIPE_FORMAT_FXT1_RGBA', 411)
PIPE_FORMAT_P010 = enum_pipe_format.define('PIPE_FORMAT_P010', 412)
PIPE_FORMAT_P012 = enum_pipe_format.define('PIPE_FORMAT_P012', 413)
PIPE_FORMAT_P016 = enum_pipe_format.define('PIPE_FORMAT_P016', 414)
PIPE_FORMAT_P030 = enum_pipe_format.define('PIPE_FORMAT_P030', 415)
PIPE_FORMAT_Y210 = enum_pipe_format.define('PIPE_FORMAT_Y210', 416)
PIPE_FORMAT_Y212 = enum_pipe_format.define('PIPE_FORMAT_Y212', 417)
PIPE_FORMAT_Y216 = enum_pipe_format.define('PIPE_FORMAT_Y216', 418)
PIPE_FORMAT_Y410 = enum_pipe_format.define('PIPE_FORMAT_Y410', 419)
PIPE_FORMAT_Y412 = enum_pipe_format.define('PIPE_FORMAT_Y412', 420)
PIPE_FORMAT_Y416 = enum_pipe_format.define('PIPE_FORMAT_Y416', 421)
PIPE_FORMAT_R10G10B10X2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_UNORM', 422)
PIPE_FORMAT_A1R5G5B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_A1R5G5B5_UNORM', 423)
PIPE_FORMAT_A1B5G5R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_A1B5G5R5_UNORM', 424)
PIPE_FORMAT_X1B5G5R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_X1B5G5R5_UNORM', 425)
PIPE_FORMAT_R5G5B5A1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G5B5A1_UNORM', 426)
PIPE_FORMAT_A4R4G4B4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4R4G4B4_UNORM', 427)
PIPE_FORMAT_A4B4G4R4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4B4G4R4_UNORM', 428)
PIPE_FORMAT_G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_G8R8_SINT', 429)
PIPE_FORMAT_A8B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SINT', 430)
PIPE_FORMAT_X8B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SINT', 431)
PIPE_FORMAT_ATC_RGB = enum_pipe_format.define('PIPE_FORMAT_ATC_RGB', 432)
PIPE_FORMAT_ATC_RGBA_EXPLICIT = enum_pipe_format.define('PIPE_FORMAT_ATC_RGBA_EXPLICIT', 433)
PIPE_FORMAT_ATC_RGBA_INTERPOLATED = enum_pipe_format.define('PIPE_FORMAT_ATC_RGBA_INTERPOLATED', 434)
PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = enum_pipe_format.define('PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8', 435)
PIPE_FORMAT_AYUV = enum_pipe_format.define('PIPE_FORMAT_AYUV', 436)
PIPE_FORMAT_XYUV = enum_pipe_format.define('PIPE_FORMAT_XYUV', 437)
PIPE_FORMAT_R8G8B8_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_420_UNORM_PACKED', 438)
PIPE_FORMAT_R8_G8B8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8B8_420_UNORM', 439)
PIPE_FORMAT_R8_B8G8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8G8_420_UNORM', 440)
PIPE_FORMAT_G8_B8R8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8R8_420_UNORM', 441)
PIPE_FORMAT_R10G10B10_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10_420_UNORM_PACKED', 442)
PIPE_FORMAT_R10_G10B10_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10_G10B10_420_UNORM', 443)
PIPE_FORMAT_R10_G10B10_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10_G10B10_422_UNORM', 444)
PIPE_FORMAT_R8_G8_B8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8_B8_420_UNORM', 445)
PIPE_FORMAT_R8_B8_G8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8_G8_420_UNORM', 446)
PIPE_FORMAT_G8_B8_R8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8_R8_420_UNORM', 447)
PIPE_FORMAT_R8_G8B8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8B8_422_UNORM', 448)
PIPE_FORMAT_R8_B8G8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8G8_422_UNORM', 449)
PIPE_FORMAT_G8_B8R8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8R8_422_UNORM', 450)
PIPE_FORMAT_R8_G8_B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8_B8_UNORM', 451)
PIPE_FORMAT_Y8_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_UNORM', 452)
PIPE_FORMAT_B8G8R8X8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SNORM', 453)
PIPE_FORMAT_B8G8R8X8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_UINT', 454)
PIPE_FORMAT_B8G8R8X8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SINT', 455)
PIPE_FORMAT_A8R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SNORM', 456)
PIPE_FORMAT_A8R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SINT', 457)
PIPE_FORMAT_X8R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SNORM', 458)
PIPE_FORMAT_X8R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SINT', 459)
PIPE_FORMAT_R5G5B5X1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G5B5X1_UNORM', 460)
PIPE_FORMAT_X1R5G5B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_X1R5G5B5_UNORM', 461)
PIPE_FORMAT_R4G4B4X4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4G4B4X4_UNORM', 462)
PIPE_FORMAT_B10G10R10X2_SNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_SNORM', 463)
PIPE_FORMAT_R5G6B5_SRGB = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_SRGB', 464)
PIPE_FORMAT_R10G10B10X2_SINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_SINT', 465)
PIPE_FORMAT_B10G10R10X2_SINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_SINT', 466)
PIPE_FORMAT_G16R16_SINT = enum_pipe_format.define('PIPE_FORMAT_G16R16_SINT', 467)
PIPE_FORMAT_COUNT = enum_pipe_format.define('PIPE_FORMAT_COUNT', 468)

struct_nir_variable_data_0_image._fields_ = [
  ('format', enum_pipe_format),
]
class struct_nir_variable_data_0_sampler(Struct): pass
struct_nir_variable_data_0_sampler._fields_ = [
  ('is_inline_sampler', ctypes.c_uint32,1),
  ('addressing_mode', ctypes.c_uint32,3),
  ('normalized_coordinates', ctypes.c_uint32,1),
  ('filter_mode', ctypes.c_uint32,1),
]
class struct_nir_variable_data_0_xfb(Struct): pass
struct_nir_variable_data_0_xfb._fields_ = [
  ('buffer', uint16_t,2),
  ('stride', uint16_t),
]
struct_nir_variable_data_0._fields_ = [
  ('image', struct_nir_variable_data_0_image),
  ('sampler', struct_nir_variable_data_0_sampler),
  ('xfb', struct_nir_variable_data_0_xfb),
]
struct_nir_variable_data._anonymous_ = ['_0']
struct_nir_variable_data._fields_ = [
  ('mode', ctypes.c_uint32,21),
  ('read_only', ctypes.c_uint32,1),
  ('centroid', ctypes.c_uint32,1),
  ('sample', ctypes.c_uint32,1),
  ('patch', ctypes.c_uint32,1),
  ('invariant', ctypes.c_uint32,1),
  ('explicit_invariant', ctypes.c_uint32,1),
  ('ray_query', ctypes.c_uint32,1),
  ('precision', ctypes.c_uint32,2),
  ('assigned', ctypes.c_uint32,1),
  ('cannot_coalesce', ctypes.c_uint32,1),
  ('always_active_io', ctypes.c_uint32,1),
  ('interpolation', ctypes.c_uint32,3),
  ('location_frac', ctypes.c_uint32,2),
  ('compact', ctypes.c_uint32,1),
  ('fb_fetch_output', ctypes.c_uint32,1),
  ('bindless', ctypes.c_uint32,1),
  ('explicit_binding', ctypes.c_uint32,1),
  ('explicit_location', ctypes.c_uint32,1),
  ('implicit_sized_array', ctypes.c_uint32,1),
  ('max_array_access', ctypes.c_int32),
  ('has_initializer', ctypes.c_uint32,1),
  ('is_implicit_initializer', ctypes.c_uint32,1),
  ('is_xfb', ctypes.c_uint32,1),
  ('is_xfb_only', ctypes.c_uint32,1),
  ('explicit_xfb_buffer', ctypes.c_uint32,1),
  ('explicit_xfb_stride', ctypes.c_uint32,1),
  ('explicit_offset', ctypes.c_uint32,1),
  ('matrix_layout', ctypes.c_uint32,2),
  ('from_named_ifc_block', ctypes.c_uint32,1),
  ('from_ssbo_unsized_array', ctypes.c_uint32,1),
  ('must_be_shader_input', ctypes.c_uint32,1),
  ('used', ctypes.c_uint32,1),
  ('how_declared', ctypes.c_uint32,2),
  ('per_view', ctypes.c_uint32,1),
  ('per_primitive', ctypes.c_uint32,1),
  ('per_vertex', ctypes.c_uint32,1),
  ('aliased_shared_memory', ctypes.c_uint32,1),
  ('depth_layout', ctypes.c_uint32,3),
  ('stream', ctypes.c_uint32,9),
  ('access', ctypes.c_uint32,9),
  ('descriptor_set', ctypes.c_uint32,5),
  ('index', ctypes.c_uint32),
  ('binding', ctypes.c_uint32),
  ('location', ctypes.c_int32),
  ('alignment', ctypes.c_uint32),
  ('driver_location', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
  ('_0', struct_nir_variable_data_0),
  ('node_name', ctypes.POINTER(ctypes.c_char)),
]
nir_variable_data = struct_nir_variable_data
class struct_nir_variable(Struct): pass
class struct_exec_node(Struct): pass
struct_exec_node._fields_ = [
  ('next', ctypes.POINTER(struct_exec_node)),
  ('prev', ctypes.POINTER(struct_exec_node)),
]
class struct_glsl_type(Struct): pass
enum_glsl_base_type = CEnum(ctypes.c_uint32)
GLSL_TYPE_UINT = enum_glsl_base_type.define('GLSL_TYPE_UINT', 0)
GLSL_TYPE_INT = enum_glsl_base_type.define('GLSL_TYPE_INT', 1)
GLSL_TYPE_FLOAT = enum_glsl_base_type.define('GLSL_TYPE_FLOAT', 2)
GLSL_TYPE_FLOAT16 = enum_glsl_base_type.define('GLSL_TYPE_FLOAT16', 3)
GLSL_TYPE_BFLOAT16 = enum_glsl_base_type.define('GLSL_TYPE_BFLOAT16', 4)
GLSL_TYPE_FLOAT_E4M3FN = enum_glsl_base_type.define('GLSL_TYPE_FLOAT_E4M3FN', 5)
GLSL_TYPE_FLOAT_E5M2 = enum_glsl_base_type.define('GLSL_TYPE_FLOAT_E5M2', 6)
GLSL_TYPE_DOUBLE = enum_glsl_base_type.define('GLSL_TYPE_DOUBLE', 7)
GLSL_TYPE_UINT8 = enum_glsl_base_type.define('GLSL_TYPE_UINT8', 8)
GLSL_TYPE_INT8 = enum_glsl_base_type.define('GLSL_TYPE_INT8', 9)
GLSL_TYPE_UINT16 = enum_glsl_base_type.define('GLSL_TYPE_UINT16', 10)
GLSL_TYPE_INT16 = enum_glsl_base_type.define('GLSL_TYPE_INT16', 11)
GLSL_TYPE_UINT64 = enum_glsl_base_type.define('GLSL_TYPE_UINT64', 12)
GLSL_TYPE_INT64 = enum_glsl_base_type.define('GLSL_TYPE_INT64', 13)
GLSL_TYPE_BOOL = enum_glsl_base_type.define('GLSL_TYPE_BOOL', 14)
GLSL_TYPE_COOPERATIVE_MATRIX = enum_glsl_base_type.define('GLSL_TYPE_COOPERATIVE_MATRIX', 15)
GLSL_TYPE_SAMPLER = enum_glsl_base_type.define('GLSL_TYPE_SAMPLER', 16)
GLSL_TYPE_TEXTURE = enum_glsl_base_type.define('GLSL_TYPE_TEXTURE', 17)
GLSL_TYPE_IMAGE = enum_glsl_base_type.define('GLSL_TYPE_IMAGE', 18)
GLSL_TYPE_ATOMIC_UINT = enum_glsl_base_type.define('GLSL_TYPE_ATOMIC_UINT', 19)
GLSL_TYPE_STRUCT = enum_glsl_base_type.define('GLSL_TYPE_STRUCT', 20)
GLSL_TYPE_INTERFACE = enum_glsl_base_type.define('GLSL_TYPE_INTERFACE', 21)
GLSL_TYPE_ARRAY = enum_glsl_base_type.define('GLSL_TYPE_ARRAY', 22)
GLSL_TYPE_VOID = enum_glsl_base_type.define('GLSL_TYPE_VOID', 23)
GLSL_TYPE_SUBROUTINE = enum_glsl_base_type.define('GLSL_TYPE_SUBROUTINE', 24)
GLSL_TYPE_ERROR = enum_glsl_base_type.define('GLSL_TYPE_ERROR', 25)

class struct_glsl_cmat_description(Struct): pass
struct_glsl_cmat_description._fields_ = [
  ('element_type', uint8_t,5),
  ('scope', uint8_t,3),
  ('rows', uint8_t),
  ('cols', uint8_t),
  ('use', uint8_t),
]
uintptr_t = ctypes.c_uint64
class struct_glsl_type_fields(ctypes.Union): pass
glsl_type = struct_glsl_type
class struct_glsl_struct_field(Struct): pass
glsl_struct_field = struct_glsl_struct_field
class struct_glsl_struct_field_0(ctypes.Union): pass
class struct_glsl_struct_field_0_0(Struct): pass
struct_glsl_struct_field_0_0._fields_ = [
  ('interpolation', ctypes.c_uint32,3),
  ('centroid', ctypes.c_uint32,1),
  ('sample', ctypes.c_uint32,1),
  ('matrix_layout', ctypes.c_uint32,2),
  ('patch', ctypes.c_uint32,1),
  ('precision', ctypes.c_uint32,2),
  ('memory_read_only', ctypes.c_uint32,1),
  ('memory_write_only', ctypes.c_uint32,1),
  ('memory_coherent', ctypes.c_uint32,1),
  ('memory_volatile', ctypes.c_uint32,1),
  ('memory_restrict', ctypes.c_uint32,1),
  ('explicit_xfb_buffer', ctypes.c_uint32,1),
  ('implicit_sized_array', ctypes.c_uint32,1),
]
struct_glsl_struct_field_0._anonymous_ = ['_0']
struct_glsl_struct_field_0._fields_ = [
  ('_0', struct_glsl_struct_field_0_0),
  ('flags', ctypes.c_uint32),
]
struct_glsl_struct_field._anonymous_ = ['_0']
struct_glsl_struct_field._fields_ = [
  ('type', ctypes.POINTER(glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('location', ctypes.c_int32),
  ('component', ctypes.c_int32),
  ('offset', ctypes.c_int32),
  ('xfb_buffer', ctypes.c_int32),
  ('xfb_stride', ctypes.c_int32),
  ('image_format', enum_pipe_format),
  ('_0', struct_glsl_struct_field_0),
]
struct_glsl_type_fields._fields_ = [
  ('array', ctypes.POINTER(glsl_type)),
  ('structure', ctypes.POINTER(glsl_struct_field)),
]
struct_glsl_type._fields_ = [
  ('gl_type', uint32_t),
  ('base_type', enum_glsl_base_type,8),
  ('sampled_type', enum_glsl_base_type,8),
  ('sampler_dimensionality', ctypes.c_uint32,4),
  ('sampler_shadow', ctypes.c_uint32,1),
  ('sampler_array', ctypes.c_uint32,1),
  ('interface_packing', ctypes.c_uint32,2),
  ('interface_row_major', ctypes.c_uint32,1),
  ('cmat_desc', struct_glsl_cmat_description),
  ('packed', ctypes.c_uint32,1),
  ('has_builtin_name', ctypes.c_uint32,1),
  ('vector_elements', uint8_t),
  ('matrix_columns', uint8_t),
  ('length', ctypes.c_uint32),
  ('name_id', uintptr_t),
  ('explicit_stride', ctypes.c_uint32),
  ('explicit_alignment', ctypes.c_uint32),
  ('fields', struct_glsl_type_fields),
]
nir_variable = struct_nir_variable
struct_nir_variable._fields_ = [
  ('node', struct_exec_node),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('data', struct_nir_variable_data),
  ('index', ctypes.c_uint32),
  ('num_members', uint16_t),
  ('max_ifc_array_access', ctypes.POINTER(ctypes.c_int32)),
  ('num_state_slots', uint16_t),
  ('state_slots', ctypes.POINTER(nir_state_slot)),
  ('constant_initializer', ctypes.POINTER(nir_constant)),
  ('pointer_initializer', ctypes.POINTER(nir_variable)),
  ('interface_type', ctypes.POINTER(struct_glsl_type)),
  ('members', ctypes.POINTER(nir_variable_data)),
]
nir_instr_type = CEnum(ctypes.c_ubyte)
nir_instr_type_alu = nir_instr_type.define('nir_instr_type_alu', 0)
nir_instr_type_deref = nir_instr_type.define('nir_instr_type_deref', 1)
nir_instr_type_call = nir_instr_type.define('nir_instr_type_call', 2)
nir_instr_type_tex = nir_instr_type.define('nir_instr_type_tex', 3)
nir_instr_type_intrinsic = nir_instr_type.define('nir_instr_type_intrinsic', 4)
nir_instr_type_load_const = nir_instr_type.define('nir_instr_type_load_const', 5)
nir_instr_type_jump = nir_instr_type.define('nir_instr_type_jump', 6)
nir_instr_type_undef = nir_instr_type.define('nir_instr_type_undef', 7)
nir_instr_type_phi = nir_instr_type.define('nir_instr_type_phi', 8)
nir_instr_type_parallel_copy = nir_instr_type.define('nir_instr_type_parallel_copy', 9)

class struct_nir_instr(Struct): pass
class struct_nir_block(Struct): pass
nir_block = struct_nir_block
class struct_nir_cf_node(Struct): pass
nir_cf_node = struct_nir_cf_node
nir_cf_node_type = CEnum(ctypes.c_uint32)
nir_cf_node_block = nir_cf_node_type.define('nir_cf_node_block', 0)
nir_cf_node_if = nir_cf_node_type.define('nir_cf_node_if', 1)
nir_cf_node_loop = nir_cf_node_type.define('nir_cf_node_loop', 2)
nir_cf_node_function = nir_cf_node_type.define('nir_cf_node_function', 3)

nir_cf_node = struct_nir_cf_node
struct_nir_cf_node._fields_ = [
  ('node', struct_exec_node),
  ('type', nir_cf_node_type),
  ('parent', ctypes.POINTER(nir_cf_node)),
]
class struct_exec_list(Struct): pass
struct_exec_list._fields_ = [
  ('head_sentinel', struct_exec_node),
  ('tail_sentinel', struct_exec_node),
]
nir_block = struct_nir_block
class struct_set(Struct): pass
class struct_set_entry(Struct): pass
struct_set_entry._fields_ = [
  ('hash', uint32_t),
  ('key', ctypes.c_void_p),
]
struct_set._fields_ = [
  ('mem_ctx', ctypes.c_void_p),
  ('table', ctypes.POINTER(struct_set_entry)),
  ('key_hash_function', ctypes.CFUNCTYPE(uint32_t, ctypes.c_void_p)),
  ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)),
  ('size', uint32_t),
  ('rehash', uint32_t),
  ('size_magic', uint64_t),
  ('rehash_magic', uint64_t),
  ('max_entries', uint32_t),
  ('size_index', uint32_t),
  ('entries', uint32_t),
  ('deleted_entries', uint32_t),
]
struct_nir_block._fields_ = [
  ('cf_node', nir_cf_node),
  ('instr_list', struct_exec_list),
  ('index', ctypes.c_uint32),
  ('divergent', ctypes.c_bool),
  ('successors', (ctypes.POINTER(nir_block) * 2)),
  ('predecessors', ctypes.POINTER(struct_set)),
  ('imm_dom', ctypes.POINTER(nir_block)),
  ('num_dom_children', ctypes.c_uint32),
  ('dom_children', ctypes.POINTER(ctypes.POINTER(nir_block))),
  ('dom_frontier', ctypes.POINTER(struct_set)),
  ('dom_pre_index', uint32_t),
  ('dom_post_index', uint32_t),
  ('start_ip', uint32_t),
  ('end_ip', uint32_t),
  ('live_in', ctypes.POINTER(ctypes.c_uint32)),
  ('live_out', ctypes.POINTER(ctypes.c_uint32)),
]
struct_nir_instr._fields_ = [
  ('node', struct_exec_node),
  ('block', ctypes.POINTER(nir_block)),
  ('type', nir_instr_type),
  ('pass_flags', uint8_t),
  ('has_debug_info', ctypes.c_bool),
  ('index', uint32_t),
]
nir_instr = struct_nir_instr
class struct_nir_def(Struct): pass
class struct_list_head(Struct): pass
struct_list_head._fields_ = [
  ('prev', ctypes.POINTER(struct_list_head)),
  ('next', ctypes.POINTER(struct_list_head)),
]
struct_nir_def._fields_ = [
  ('parent_instr', ctypes.POINTER(nir_instr)),
  ('uses', struct_list_head),
  ('index', ctypes.c_uint32),
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('divergent', ctypes.c_bool),
  ('loop_invariant', ctypes.c_bool),
]
nir_def = struct_nir_def
class struct_nir_src(Struct): pass
struct_nir_src._fields_ = [
  ('_parent', uintptr_t),
  ('use_link', struct_list_head),
  ('ssa', ctypes.POINTER(nir_def)),
]
nir_src = struct_nir_src
try: (nir_src_is_divergent:=dll.nir_src_is_divergent).restype, nir_src_is_divergent.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_src)]
except AttributeError: pass

class struct_nir_alu_src(Struct): pass
struct_nir_alu_src._fields_ = [
  ('src', nir_src),
  ('swizzle', (uint8_t * 16)),
]
nir_alu_src = struct_nir_alu_src
nir_alu_type = CEnum(ctypes.c_ubyte)
nir_type_invalid = nir_alu_type.define('nir_type_invalid', 0)
nir_type_int = nir_alu_type.define('nir_type_int', 2)
nir_type_uint = nir_alu_type.define('nir_type_uint', 4)
nir_type_bool = nir_alu_type.define('nir_type_bool', 6)
nir_type_float = nir_alu_type.define('nir_type_float', 128)
nir_type_bool1 = nir_alu_type.define('nir_type_bool1', 7)
nir_type_bool8 = nir_alu_type.define('nir_type_bool8', 14)
nir_type_bool16 = nir_alu_type.define('nir_type_bool16', 22)
nir_type_bool32 = nir_alu_type.define('nir_type_bool32', 38)
nir_type_int1 = nir_alu_type.define('nir_type_int1', 3)
nir_type_int8 = nir_alu_type.define('nir_type_int8', 10)
nir_type_int16 = nir_alu_type.define('nir_type_int16', 18)
nir_type_int32 = nir_alu_type.define('nir_type_int32', 34)
nir_type_int64 = nir_alu_type.define('nir_type_int64', 66)
nir_type_uint1 = nir_alu_type.define('nir_type_uint1', 5)
nir_type_uint8 = nir_alu_type.define('nir_type_uint8', 12)
nir_type_uint16 = nir_alu_type.define('nir_type_uint16', 20)
nir_type_uint32 = nir_alu_type.define('nir_type_uint32', 36)
nir_type_uint64 = nir_alu_type.define('nir_type_uint64', 68)
nir_type_float16 = nir_alu_type.define('nir_type_float16', 144)
nir_type_float32 = nir_alu_type.define('nir_type_float32', 160)
nir_type_float64 = nir_alu_type.define('nir_type_float64', 192)

try: (nir_get_nir_type_for_glsl_base_type:=dll.nir_get_nir_type_for_glsl_base_type).restype, nir_get_nir_type_for_glsl_base_type.argtypes = nir_alu_type, [enum_glsl_base_type]
except AttributeError: pass

try: (nir_get_glsl_base_type_for_nir_type:=dll.nir_get_glsl_base_type_for_nir_type).restype, nir_get_glsl_base_type_for_nir_type.argtypes = enum_glsl_base_type, [nir_alu_type]
except AttributeError: pass

nir_op = CEnum(ctypes.c_uint32)
nir_op_alignbyte_amd = nir_op.define('nir_op_alignbyte_amd', 0)
nir_op_amul = nir_op.define('nir_op_amul', 1)
nir_op_andg_ir3 = nir_op.define('nir_op_andg_ir3', 2)
nir_op_b16all_fequal16 = nir_op.define('nir_op_b16all_fequal16', 3)
nir_op_b16all_fequal2 = nir_op.define('nir_op_b16all_fequal2', 4)
nir_op_b16all_fequal3 = nir_op.define('nir_op_b16all_fequal3', 5)
nir_op_b16all_fequal4 = nir_op.define('nir_op_b16all_fequal4', 6)
nir_op_b16all_fequal5 = nir_op.define('nir_op_b16all_fequal5', 7)
nir_op_b16all_fequal8 = nir_op.define('nir_op_b16all_fequal8', 8)
nir_op_b16all_iequal16 = nir_op.define('nir_op_b16all_iequal16', 9)
nir_op_b16all_iequal2 = nir_op.define('nir_op_b16all_iequal2', 10)
nir_op_b16all_iequal3 = nir_op.define('nir_op_b16all_iequal3', 11)
nir_op_b16all_iequal4 = nir_op.define('nir_op_b16all_iequal4', 12)
nir_op_b16all_iequal5 = nir_op.define('nir_op_b16all_iequal5', 13)
nir_op_b16all_iequal8 = nir_op.define('nir_op_b16all_iequal8', 14)
nir_op_b16any_fnequal16 = nir_op.define('nir_op_b16any_fnequal16', 15)
nir_op_b16any_fnequal2 = nir_op.define('nir_op_b16any_fnequal2', 16)
nir_op_b16any_fnequal3 = nir_op.define('nir_op_b16any_fnequal3', 17)
nir_op_b16any_fnequal4 = nir_op.define('nir_op_b16any_fnequal4', 18)
nir_op_b16any_fnequal5 = nir_op.define('nir_op_b16any_fnequal5', 19)
nir_op_b16any_fnequal8 = nir_op.define('nir_op_b16any_fnequal8', 20)
nir_op_b16any_inequal16 = nir_op.define('nir_op_b16any_inequal16', 21)
nir_op_b16any_inequal2 = nir_op.define('nir_op_b16any_inequal2', 22)
nir_op_b16any_inequal3 = nir_op.define('nir_op_b16any_inequal3', 23)
nir_op_b16any_inequal4 = nir_op.define('nir_op_b16any_inequal4', 24)
nir_op_b16any_inequal5 = nir_op.define('nir_op_b16any_inequal5', 25)
nir_op_b16any_inequal8 = nir_op.define('nir_op_b16any_inequal8', 26)
nir_op_b16csel = nir_op.define('nir_op_b16csel', 27)
nir_op_b2b1 = nir_op.define('nir_op_b2b1', 28)
nir_op_b2b16 = nir_op.define('nir_op_b2b16', 29)
nir_op_b2b32 = nir_op.define('nir_op_b2b32', 30)
nir_op_b2b8 = nir_op.define('nir_op_b2b8', 31)
nir_op_b2f16 = nir_op.define('nir_op_b2f16', 32)
nir_op_b2f32 = nir_op.define('nir_op_b2f32', 33)
nir_op_b2f64 = nir_op.define('nir_op_b2f64', 34)
nir_op_b2i1 = nir_op.define('nir_op_b2i1', 35)
nir_op_b2i16 = nir_op.define('nir_op_b2i16', 36)
nir_op_b2i32 = nir_op.define('nir_op_b2i32', 37)
nir_op_b2i64 = nir_op.define('nir_op_b2i64', 38)
nir_op_b2i8 = nir_op.define('nir_op_b2i8', 39)
nir_op_b32all_fequal16 = nir_op.define('nir_op_b32all_fequal16', 40)
nir_op_b32all_fequal2 = nir_op.define('nir_op_b32all_fequal2', 41)
nir_op_b32all_fequal3 = nir_op.define('nir_op_b32all_fequal3', 42)
nir_op_b32all_fequal4 = nir_op.define('nir_op_b32all_fequal4', 43)
nir_op_b32all_fequal5 = nir_op.define('nir_op_b32all_fequal5', 44)
nir_op_b32all_fequal8 = nir_op.define('nir_op_b32all_fequal8', 45)
nir_op_b32all_iequal16 = nir_op.define('nir_op_b32all_iequal16', 46)
nir_op_b32all_iequal2 = nir_op.define('nir_op_b32all_iequal2', 47)
nir_op_b32all_iequal3 = nir_op.define('nir_op_b32all_iequal3', 48)
nir_op_b32all_iequal4 = nir_op.define('nir_op_b32all_iequal4', 49)
nir_op_b32all_iequal5 = nir_op.define('nir_op_b32all_iequal5', 50)
nir_op_b32all_iequal8 = nir_op.define('nir_op_b32all_iequal8', 51)
nir_op_b32any_fnequal16 = nir_op.define('nir_op_b32any_fnequal16', 52)
nir_op_b32any_fnequal2 = nir_op.define('nir_op_b32any_fnequal2', 53)
nir_op_b32any_fnequal3 = nir_op.define('nir_op_b32any_fnequal3', 54)
nir_op_b32any_fnequal4 = nir_op.define('nir_op_b32any_fnequal4', 55)
nir_op_b32any_fnequal5 = nir_op.define('nir_op_b32any_fnequal5', 56)
nir_op_b32any_fnequal8 = nir_op.define('nir_op_b32any_fnequal8', 57)
nir_op_b32any_inequal16 = nir_op.define('nir_op_b32any_inequal16', 58)
nir_op_b32any_inequal2 = nir_op.define('nir_op_b32any_inequal2', 59)
nir_op_b32any_inequal3 = nir_op.define('nir_op_b32any_inequal3', 60)
nir_op_b32any_inequal4 = nir_op.define('nir_op_b32any_inequal4', 61)
nir_op_b32any_inequal5 = nir_op.define('nir_op_b32any_inequal5', 62)
nir_op_b32any_inequal8 = nir_op.define('nir_op_b32any_inequal8', 63)
nir_op_b32csel = nir_op.define('nir_op_b32csel', 64)
nir_op_b32fcsel_mdg = nir_op.define('nir_op_b32fcsel_mdg', 65)
nir_op_b8all_fequal16 = nir_op.define('nir_op_b8all_fequal16', 66)
nir_op_b8all_fequal2 = nir_op.define('nir_op_b8all_fequal2', 67)
nir_op_b8all_fequal3 = nir_op.define('nir_op_b8all_fequal3', 68)
nir_op_b8all_fequal4 = nir_op.define('nir_op_b8all_fequal4', 69)
nir_op_b8all_fequal5 = nir_op.define('nir_op_b8all_fequal5', 70)
nir_op_b8all_fequal8 = nir_op.define('nir_op_b8all_fequal8', 71)
nir_op_b8all_iequal16 = nir_op.define('nir_op_b8all_iequal16', 72)
nir_op_b8all_iequal2 = nir_op.define('nir_op_b8all_iequal2', 73)
nir_op_b8all_iequal3 = nir_op.define('nir_op_b8all_iequal3', 74)
nir_op_b8all_iequal4 = nir_op.define('nir_op_b8all_iequal4', 75)
nir_op_b8all_iequal5 = nir_op.define('nir_op_b8all_iequal5', 76)
nir_op_b8all_iequal8 = nir_op.define('nir_op_b8all_iequal8', 77)
nir_op_b8any_fnequal16 = nir_op.define('nir_op_b8any_fnequal16', 78)
nir_op_b8any_fnequal2 = nir_op.define('nir_op_b8any_fnequal2', 79)
nir_op_b8any_fnequal3 = nir_op.define('nir_op_b8any_fnequal3', 80)
nir_op_b8any_fnequal4 = nir_op.define('nir_op_b8any_fnequal4', 81)
nir_op_b8any_fnequal5 = nir_op.define('nir_op_b8any_fnequal5', 82)
nir_op_b8any_fnequal8 = nir_op.define('nir_op_b8any_fnequal8', 83)
nir_op_b8any_inequal16 = nir_op.define('nir_op_b8any_inequal16', 84)
nir_op_b8any_inequal2 = nir_op.define('nir_op_b8any_inequal2', 85)
nir_op_b8any_inequal3 = nir_op.define('nir_op_b8any_inequal3', 86)
nir_op_b8any_inequal4 = nir_op.define('nir_op_b8any_inequal4', 87)
nir_op_b8any_inequal5 = nir_op.define('nir_op_b8any_inequal5', 88)
nir_op_b8any_inequal8 = nir_op.define('nir_op_b8any_inequal8', 89)
nir_op_b8csel = nir_op.define('nir_op_b8csel', 90)
nir_op_ball_fequal16 = nir_op.define('nir_op_ball_fequal16', 91)
nir_op_ball_fequal2 = nir_op.define('nir_op_ball_fequal2', 92)
nir_op_ball_fequal3 = nir_op.define('nir_op_ball_fequal3', 93)
nir_op_ball_fequal4 = nir_op.define('nir_op_ball_fequal4', 94)
nir_op_ball_fequal5 = nir_op.define('nir_op_ball_fequal5', 95)
nir_op_ball_fequal8 = nir_op.define('nir_op_ball_fequal8', 96)
nir_op_ball_iequal16 = nir_op.define('nir_op_ball_iequal16', 97)
nir_op_ball_iequal2 = nir_op.define('nir_op_ball_iequal2', 98)
nir_op_ball_iequal3 = nir_op.define('nir_op_ball_iequal3', 99)
nir_op_ball_iequal4 = nir_op.define('nir_op_ball_iequal4', 100)
nir_op_ball_iequal5 = nir_op.define('nir_op_ball_iequal5', 101)
nir_op_ball_iequal8 = nir_op.define('nir_op_ball_iequal8', 102)
nir_op_bany_fnequal16 = nir_op.define('nir_op_bany_fnequal16', 103)
nir_op_bany_fnequal2 = nir_op.define('nir_op_bany_fnequal2', 104)
nir_op_bany_fnequal3 = nir_op.define('nir_op_bany_fnequal3', 105)
nir_op_bany_fnequal4 = nir_op.define('nir_op_bany_fnequal4', 106)
nir_op_bany_fnequal5 = nir_op.define('nir_op_bany_fnequal5', 107)
nir_op_bany_fnequal8 = nir_op.define('nir_op_bany_fnequal8', 108)
nir_op_bany_inequal16 = nir_op.define('nir_op_bany_inequal16', 109)
nir_op_bany_inequal2 = nir_op.define('nir_op_bany_inequal2', 110)
nir_op_bany_inequal3 = nir_op.define('nir_op_bany_inequal3', 111)
nir_op_bany_inequal4 = nir_op.define('nir_op_bany_inequal4', 112)
nir_op_bany_inequal5 = nir_op.define('nir_op_bany_inequal5', 113)
nir_op_bany_inequal8 = nir_op.define('nir_op_bany_inequal8', 114)
nir_op_bcsel = nir_op.define('nir_op_bcsel', 115)
nir_op_bf2f = nir_op.define('nir_op_bf2f', 116)
nir_op_bfdot16 = nir_op.define('nir_op_bfdot16', 117)
nir_op_bfdot2 = nir_op.define('nir_op_bfdot2', 118)
nir_op_bfdot2_bfadd = nir_op.define('nir_op_bfdot2_bfadd', 119)
nir_op_bfdot3 = nir_op.define('nir_op_bfdot3', 120)
nir_op_bfdot4 = nir_op.define('nir_op_bfdot4', 121)
nir_op_bfdot5 = nir_op.define('nir_op_bfdot5', 122)
nir_op_bfdot8 = nir_op.define('nir_op_bfdot8', 123)
nir_op_bffma = nir_op.define('nir_op_bffma', 124)
nir_op_bfi = nir_op.define('nir_op_bfi', 125)
nir_op_bfm = nir_op.define('nir_op_bfm', 126)
nir_op_bfmul = nir_op.define('nir_op_bfmul', 127)
nir_op_bit_count = nir_op.define('nir_op_bit_count', 128)
nir_op_bitfield_insert = nir_op.define('nir_op_bitfield_insert', 129)
nir_op_bitfield_reverse = nir_op.define('nir_op_bitfield_reverse', 130)
nir_op_bitfield_select = nir_op.define('nir_op_bitfield_select', 131)
nir_op_bitnz = nir_op.define('nir_op_bitnz', 132)
nir_op_bitnz16 = nir_op.define('nir_op_bitnz16', 133)
nir_op_bitnz32 = nir_op.define('nir_op_bitnz32', 134)
nir_op_bitnz8 = nir_op.define('nir_op_bitnz8', 135)
nir_op_bitz = nir_op.define('nir_op_bitz', 136)
nir_op_bitz16 = nir_op.define('nir_op_bitz16', 137)
nir_op_bitz32 = nir_op.define('nir_op_bitz32', 138)
nir_op_bitz8 = nir_op.define('nir_op_bitz8', 139)
nir_op_bounds_agx = nir_op.define('nir_op_bounds_agx', 140)
nir_op_byte_perm_amd = nir_op.define('nir_op_byte_perm_amd', 141)
nir_op_cube_amd = nir_op.define('nir_op_cube_amd', 142)
nir_op_e4m3fn2f = nir_op.define('nir_op_e4m3fn2f', 143)
nir_op_e5m22f = nir_op.define('nir_op_e5m22f', 144)
nir_op_extr_agx = nir_op.define('nir_op_extr_agx', 145)
nir_op_extract_i16 = nir_op.define('nir_op_extract_i16', 146)
nir_op_extract_i8 = nir_op.define('nir_op_extract_i8', 147)
nir_op_extract_u16 = nir_op.define('nir_op_extract_u16', 148)
nir_op_extract_u8 = nir_op.define('nir_op_extract_u8', 149)
nir_op_f2bf = nir_op.define('nir_op_f2bf', 150)
nir_op_f2e4m3fn = nir_op.define('nir_op_f2e4m3fn', 151)
nir_op_f2e4m3fn_sat = nir_op.define('nir_op_f2e4m3fn_sat', 152)
nir_op_f2e4m3fn_satfn = nir_op.define('nir_op_f2e4m3fn_satfn', 153)
nir_op_f2e5m2 = nir_op.define('nir_op_f2e5m2', 154)
nir_op_f2e5m2_sat = nir_op.define('nir_op_f2e5m2_sat', 155)
nir_op_f2f16 = nir_op.define('nir_op_f2f16', 156)
nir_op_f2f16_rtne = nir_op.define('nir_op_f2f16_rtne', 157)
nir_op_f2f16_rtz = nir_op.define('nir_op_f2f16_rtz', 158)
nir_op_f2f32 = nir_op.define('nir_op_f2f32', 159)
nir_op_f2f64 = nir_op.define('nir_op_f2f64', 160)
nir_op_f2fmp = nir_op.define('nir_op_f2fmp', 161)
nir_op_f2i1 = nir_op.define('nir_op_f2i1', 162)
nir_op_f2i16 = nir_op.define('nir_op_f2i16', 163)
nir_op_f2i32 = nir_op.define('nir_op_f2i32', 164)
nir_op_f2i64 = nir_op.define('nir_op_f2i64', 165)
nir_op_f2i8 = nir_op.define('nir_op_f2i8', 166)
nir_op_f2imp = nir_op.define('nir_op_f2imp', 167)
nir_op_f2snorm_16_v3d = nir_op.define('nir_op_f2snorm_16_v3d', 168)
nir_op_f2u1 = nir_op.define('nir_op_f2u1', 169)
nir_op_f2u16 = nir_op.define('nir_op_f2u16', 170)
nir_op_f2u32 = nir_op.define('nir_op_f2u32', 171)
nir_op_f2u64 = nir_op.define('nir_op_f2u64', 172)
nir_op_f2u8 = nir_op.define('nir_op_f2u8', 173)
nir_op_f2ump = nir_op.define('nir_op_f2ump', 174)
nir_op_f2unorm_16_v3d = nir_op.define('nir_op_f2unorm_16_v3d', 175)
nir_op_fabs = nir_op.define('nir_op_fabs', 176)
nir_op_fadd = nir_op.define('nir_op_fadd', 177)
nir_op_fall_equal16 = nir_op.define('nir_op_fall_equal16', 178)
nir_op_fall_equal2 = nir_op.define('nir_op_fall_equal2', 179)
nir_op_fall_equal3 = nir_op.define('nir_op_fall_equal3', 180)
nir_op_fall_equal4 = nir_op.define('nir_op_fall_equal4', 181)
nir_op_fall_equal5 = nir_op.define('nir_op_fall_equal5', 182)
nir_op_fall_equal8 = nir_op.define('nir_op_fall_equal8', 183)
nir_op_fany_nequal16 = nir_op.define('nir_op_fany_nequal16', 184)
nir_op_fany_nequal2 = nir_op.define('nir_op_fany_nequal2', 185)
nir_op_fany_nequal3 = nir_op.define('nir_op_fany_nequal3', 186)
nir_op_fany_nequal4 = nir_op.define('nir_op_fany_nequal4', 187)
nir_op_fany_nequal5 = nir_op.define('nir_op_fany_nequal5', 188)
nir_op_fany_nequal8 = nir_op.define('nir_op_fany_nequal8', 189)
nir_op_fceil = nir_op.define('nir_op_fceil', 190)
nir_op_fclamp_pos = nir_op.define('nir_op_fclamp_pos', 191)
nir_op_fcos = nir_op.define('nir_op_fcos', 192)
nir_op_fcos_amd = nir_op.define('nir_op_fcos_amd', 193)
nir_op_fcos_mdg = nir_op.define('nir_op_fcos_mdg', 194)
nir_op_fcsel = nir_op.define('nir_op_fcsel', 195)
nir_op_fcsel_ge = nir_op.define('nir_op_fcsel_ge', 196)
nir_op_fcsel_gt = nir_op.define('nir_op_fcsel_gt', 197)
nir_op_fdiv = nir_op.define('nir_op_fdiv', 198)
nir_op_fdot16 = nir_op.define('nir_op_fdot16', 199)
nir_op_fdot16_replicated = nir_op.define('nir_op_fdot16_replicated', 200)
nir_op_fdot2 = nir_op.define('nir_op_fdot2', 201)
nir_op_fdot2_replicated = nir_op.define('nir_op_fdot2_replicated', 202)
nir_op_fdot3 = nir_op.define('nir_op_fdot3', 203)
nir_op_fdot3_replicated = nir_op.define('nir_op_fdot3_replicated', 204)
nir_op_fdot4 = nir_op.define('nir_op_fdot4', 205)
nir_op_fdot4_replicated = nir_op.define('nir_op_fdot4_replicated', 206)
nir_op_fdot5 = nir_op.define('nir_op_fdot5', 207)
nir_op_fdot5_replicated = nir_op.define('nir_op_fdot5_replicated', 208)
nir_op_fdot8 = nir_op.define('nir_op_fdot8', 209)
nir_op_fdot8_replicated = nir_op.define('nir_op_fdot8_replicated', 210)
nir_op_fdph = nir_op.define('nir_op_fdph', 211)
nir_op_fdph_replicated = nir_op.define('nir_op_fdph_replicated', 212)
nir_op_feq = nir_op.define('nir_op_feq', 213)
nir_op_feq16 = nir_op.define('nir_op_feq16', 214)
nir_op_feq32 = nir_op.define('nir_op_feq32', 215)
nir_op_feq8 = nir_op.define('nir_op_feq8', 216)
nir_op_fequ = nir_op.define('nir_op_fequ', 217)
nir_op_fequ16 = nir_op.define('nir_op_fequ16', 218)
nir_op_fequ32 = nir_op.define('nir_op_fequ32', 219)
nir_op_fequ8 = nir_op.define('nir_op_fequ8', 220)
nir_op_fexp2 = nir_op.define('nir_op_fexp2', 221)
nir_op_ffloor = nir_op.define('nir_op_ffloor', 222)
nir_op_ffma = nir_op.define('nir_op_ffma', 223)
nir_op_ffmaz = nir_op.define('nir_op_ffmaz', 224)
nir_op_ffract = nir_op.define('nir_op_ffract', 225)
nir_op_fge = nir_op.define('nir_op_fge', 226)
nir_op_fge16 = nir_op.define('nir_op_fge16', 227)
nir_op_fge32 = nir_op.define('nir_op_fge32', 228)
nir_op_fge8 = nir_op.define('nir_op_fge8', 229)
nir_op_fgeu = nir_op.define('nir_op_fgeu', 230)
nir_op_fgeu16 = nir_op.define('nir_op_fgeu16', 231)
nir_op_fgeu32 = nir_op.define('nir_op_fgeu32', 232)
nir_op_fgeu8 = nir_op.define('nir_op_fgeu8', 233)
nir_op_find_lsb = nir_op.define('nir_op_find_lsb', 234)
nir_op_fisfinite = nir_op.define('nir_op_fisfinite', 235)
nir_op_fisfinite32 = nir_op.define('nir_op_fisfinite32', 236)
nir_op_fisnormal = nir_op.define('nir_op_fisnormal', 237)
nir_op_flog2 = nir_op.define('nir_op_flog2', 238)
nir_op_flrp = nir_op.define('nir_op_flrp', 239)
nir_op_flt = nir_op.define('nir_op_flt', 240)
nir_op_flt16 = nir_op.define('nir_op_flt16', 241)
nir_op_flt32 = nir_op.define('nir_op_flt32', 242)
nir_op_flt8 = nir_op.define('nir_op_flt8', 243)
nir_op_fltu = nir_op.define('nir_op_fltu', 244)
nir_op_fltu16 = nir_op.define('nir_op_fltu16', 245)
nir_op_fltu32 = nir_op.define('nir_op_fltu32', 246)
nir_op_fltu8 = nir_op.define('nir_op_fltu8', 247)
nir_op_fmax = nir_op.define('nir_op_fmax', 248)
nir_op_fmax_agx = nir_op.define('nir_op_fmax_agx', 249)
nir_op_fmin = nir_op.define('nir_op_fmin', 250)
nir_op_fmin_agx = nir_op.define('nir_op_fmin_agx', 251)
nir_op_fmod = nir_op.define('nir_op_fmod', 252)
nir_op_fmul = nir_op.define('nir_op_fmul', 253)
nir_op_fmulz = nir_op.define('nir_op_fmulz', 254)
nir_op_fneg = nir_op.define('nir_op_fneg', 255)
nir_op_fneo = nir_op.define('nir_op_fneo', 256)
nir_op_fneo16 = nir_op.define('nir_op_fneo16', 257)
nir_op_fneo32 = nir_op.define('nir_op_fneo32', 258)
nir_op_fneo8 = nir_op.define('nir_op_fneo8', 259)
nir_op_fneu = nir_op.define('nir_op_fneu', 260)
nir_op_fneu16 = nir_op.define('nir_op_fneu16', 261)
nir_op_fneu32 = nir_op.define('nir_op_fneu32', 262)
nir_op_fneu8 = nir_op.define('nir_op_fneu8', 263)
nir_op_ford = nir_op.define('nir_op_ford', 264)
nir_op_ford16 = nir_op.define('nir_op_ford16', 265)
nir_op_ford32 = nir_op.define('nir_op_ford32', 266)
nir_op_ford8 = nir_op.define('nir_op_ford8', 267)
nir_op_fpow = nir_op.define('nir_op_fpow', 268)
nir_op_fquantize2f16 = nir_op.define('nir_op_fquantize2f16', 269)
nir_op_frcp = nir_op.define('nir_op_frcp', 270)
nir_op_frem = nir_op.define('nir_op_frem', 271)
nir_op_frexp_exp = nir_op.define('nir_op_frexp_exp', 272)
nir_op_frexp_sig = nir_op.define('nir_op_frexp_sig', 273)
nir_op_fround_even = nir_op.define('nir_op_fround_even', 274)
nir_op_frsq = nir_op.define('nir_op_frsq', 275)
nir_op_fsat = nir_op.define('nir_op_fsat', 276)
nir_op_fsat_signed = nir_op.define('nir_op_fsat_signed', 277)
nir_op_fsign = nir_op.define('nir_op_fsign', 278)
nir_op_fsin = nir_op.define('nir_op_fsin', 279)
nir_op_fsin_agx = nir_op.define('nir_op_fsin_agx', 280)
nir_op_fsin_amd = nir_op.define('nir_op_fsin_amd', 281)
nir_op_fsin_mdg = nir_op.define('nir_op_fsin_mdg', 282)
nir_op_fsqrt = nir_op.define('nir_op_fsqrt', 283)
nir_op_fsub = nir_op.define('nir_op_fsub', 284)
nir_op_fsum2 = nir_op.define('nir_op_fsum2', 285)
nir_op_fsum3 = nir_op.define('nir_op_fsum3', 286)
nir_op_fsum4 = nir_op.define('nir_op_fsum4', 287)
nir_op_ftrunc = nir_op.define('nir_op_ftrunc', 288)
nir_op_funord = nir_op.define('nir_op_funord', 289)
nir_op_funord16 = nir_op.define('nir_op_funord16', 290)
nir_op_funord32 = nir_op.define('nir_op_funord32', 291)
nir_op_funord8 = nir_op.define('nir_op_funord8', 292)
nir_op_i2f16 = nir_op.define('nir_op_i2f16', 293)
nir_op_i2f32 = nir_op.define('nir_op_i2f32', 294)
nir_op_i2f64 = nir_op.define('nir_op_i2f64', 295)
nir_op_i2fmp = nir_op.define('nir_op_i2fmp', 296)
nir_op_i2i1 = nir_op.define('nir_op_i2i1', 297)
nir_op_i2i16 = nir_op.define('nir_op_i2i16', 298)
nir_op_i2i32 = nir_op.define('nir_op_i2i32', 299)
nir_op_i2i64 = nir_op.define('nir_op_i2i64', 300)
nir_op_i2i8 = nir_op.define('nir_op_i2i8', 301)
nir_op_i2imp = nir_op.define('nir_op_i2imp', 302)
nir_op_i32csel_ge = nir_op.define('nir_op_i32csel_ge', 303)
nir_op_i32csel_gt = nir_op.define('nir_op_i32csel_gt', 304)
nir_op_iabs = nir_op.define('nir_op_iabs', 305)
nir_op_iadd = nir_op.define('nir_op_iadd', 306)
nir_op_iadd3 = nir_op.define('nir_op_iadd3', 307)
nir_op_iadd_sat = nir_op.define('nir_op_iadd_sat', 308)
nir_op_iand = nir_op.define('nir_op_iand', 309)
nir_op_ibfe = nir_op.define('nir_op_ibfe', 310)
nir_op_ibitfield_extract = nir_op.define('nir_op_ibitfield_extract', 311)
nir_op_icsel_eqz = nir_op.define('nir_op_icsel_eqz', 312)
nir_op_idiv = nir_op.define('nir_op_idiv', 313)
nir_op_ieq = nir_op.define('nir_op_ieq', 314)
nir_op_ieq16 = nir_op.define('nir_op_ieq16', 315)
nir_op_ieq32 = nir_op.define('nir_op_ieq32', 316)
nir_op_ieq8 = nir_op.define('nir_op_ieq8', 317)
nir_op_ifind_msb = nir_op.define('nir_op_ifind_msb', 318)
nir_op_ifind_msb_rev = nir_op.define('nir_op_ifind_msb_rev', 319)
nir_op_ige = nir_op.define('nir_op_ige', 320)
nir_op_ige16 = nir_op.define('nir_op_ige16', 321)
nir_op_ige32 = nir_op.define('nir_op_ige32', 322)
nir_op_ige8 = nir_op.define('nir_op_ige8', 323)
nir_op_ihadd = nir_op.define('nir_op_ihadd', 324)
nir_op_ilea_agx = nir_op.define('nir_op_ilea_agx', 325)
nir_op_ilt = nir_op.define('nir_op_ilt', 326)
nir_op_ilt16 = nir_op.define('nir_op_ilt16', 327)
nir_op_ilt32 = nir_op.define('nir_op_ilt32', 328)
nir_op_ilt8 = nir_op.define('nir_op_ilt8', 329)
nir_op_imad = nir_op.define('nir_op_imad', 330)
nir_op_imad24_ir3 = nir_op.define('nir_op_imad24_ir3', 331)
nir_op_imadsh_mix16 = nir_op.define('nir_op_imadsh_mix16', 332)
nir_op_imadshl_agx = nir_op.define('nir_op_imadshl_agx', 333)
nir_op_imax = nir_op.define('nir_op_imax', 334)
nir_op_imin = nir_op.define('nir_op_imin', 335)
nir_op_imod = nir_op.define('nir_op_imod', 336)
nir_op_imsubshl_agx = nir_op.define('nir_op_imsubshl_agx', 337)
nir_op_imul = nir_op.define('nir_op_imul', 338)
nir_op_imul24 = nir_op.define('nir_op_imul24', 339)
nir_op_imul24_relaxed = nir_op.define('nir_op_imul24_relaxed', 340)
nir_op_imul_2x32_64 = nir_op.define('nir_op_imul_2x32_64', 341)
nir_op_imul_32x16 = nir_op.define('nir_op_imul_32x16', 342)
nir_op_imul_high = nir_op.define('nir_op_imul_high', 343)
nir_op_ine = nir_op.define('nir_op_ine', 344)
nir_op_ine16 = nir_op.define('nir_op_ine16', 345)
nir_op_ine32 = nir_op.define('nir_op_ine32', 346)
nir_op_ine8 = nir_op.define('nir_op_ine8', 347)
nir_op_ineg = nir_op.define('nir_op_ineg', 348)
nir_op_inot = nir_op.define('nir_op_inot', 349)
nir_op_insert_u16 = nir_op.define('nir_op_insert_u16', 350)
nir_op_insert_u8 = nir_op.define('nir_op_insert_u8', 351)
nir_op_interleave_agx = nir_op.define('nir_op_interleave_agx', 352)
nir_op_ior = nir_op.define('nir_op_ior', 353)
nir_op_irem = nir_op.define('nir_op_irem', 354)
nir_op_irhadd = nir_op.define('nir_op_irhadd', 355)
nir_op_ishl = nir_op.define('nir_op_ishl', 356)
nir_op_ishr = nir_op.define('nir_op_ishr', 357)
nir_op_isign = nir_op.define('nir_op_isign', 358)
nir_op_isub = nir_op.define('nir_op_isub', 359)
nir_op_isub_sat = nir_op.define('nir_op_isub_sat', 360)
nir_op_ixor = nir_op.define('nir_op_ixor', 361)
nir_op_ldexp = nir_op.define('nir_op_ldexp', 362)
nir_op_ldexp16_pan = nir_op.define('nir_op_ldexp16_pan', 363)
nir_op_lea_nv = nir_op.define('nir_op_lea_nv', 364)
nir_op_mov = nir_op.define('nir_op_mov', 365)
nir_op_mqsad_4x8 = nir_op.define('nir_op_mqsad_4x8', 366)
nir_op_msad_4x8 = nir_op.define('nir_op_msad_4x8', 367)
nir_op_pack_2x16_to_snorm_2x8_v3d = nir_op.define('nir_op_pack_2x16_to_snorm_2x8_v3d', 368)
nir_op_pack_2x16_to_unorm_10_2_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_10_2_v3d', 369)
nir_op_pack_2x16_to_unorm_2x10_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_2x10_v3d', 370)
nir_op_pack_2x16_to_unorm_2x8_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_2x8_v3d', 371)
nir_op_pack_2x32_to_2x16_v3d = nir_op.define('nir_op_pack_2x32_to_2x16_v3d', 372)
nir_op_pack_32_2x16 = nir_op.define('nir_op_pack_32_2x16', 373)
nir_op_pack_32_2x16_split = nir_op.define('nir_op_pack_32_2x16_split', 374)
nir_op_pack_32_4x8 = nir_op.define('nir_op_pack_32_4x8', 375)
nir_op_pack_32_4x8_split = nir_op.define('nir_op_pack_32_4x8_split', 376)
nir_op_pack_32_to_r11g11b10_v3d = nir_op.define('nir_op_pack_32_to_r11g11b10_v3d', 377)
nir_op_pack_4x16_to_4x8_v3d = nir_op.define('nir_op_pack_4x16_to_4x8_v3d', 378)
nir_op_pack_64_2x32 = nir_op.define('nir_op_pack_64_2x32', 379)
nir_op_pack_64_2x32_split = nir_op.define('nir_op_pack_64_2x32_split', 380)
nir_op_pack_64_4x16 = nir_op.define('nir_op_pack_64_4x16', 381)
nir_op_pack_double_2x32_dxil = nir_op.define('nir_op_pack_double_2x32_dxil', 382)
nir_op_pack_half_2x16 = nir_op.define('nir_op_pack_half_2x16', 383)
nir_op_pack_half_2x16_rtz_split = nir_op.define('nir_op_pack_half_2x16_rtz_split', 384)
nir_op_pack_half_2x16_split = nir_op.define('nir_op_pack_half_2x16_split', 385)
nir_op_pack_sint_2x16 = nir_op.define('nir_op_pack_sint_2x16', 386)
nir_op_pack_snorm_2x16 = nir_op.define('nir_op_pack_snorm_2x16', 387)
nir_op_pack_snorm_4x8 = nir_op.define('nir_op_pack_snorm_4x8', 388)
nir_op_pack_uint_2x16 = nir_op.define('nir_op_pack_uint_2x16', 389)
nir_op_pack_uint_32_to_r10g10b10a2_v3d = nir_op.define('nir_op_pack_uint_32_to_r10g10b10a2_v3d', 390)
nir_op_pack_unorm_2x16 = nir_op.define('nir_op_pack_unorm_2x16', 391)
nir_op_pack_unorm_4x8 = nir_op.define('nir_op_pack_unorm_4x8', 392)
nir_op_pack_uvec2_to_uint = nir_op.define('nir_op_pack_uvec2_to_uint', 393)
nir_op_pack_uvec4_to_uint = nir_op.define('nir_op_pack_uvec4_to_uint', 394)
nir_op_prmt_nv = nir_op.define('nir_op_prmt_nv', 395)
nir_op_sdot_2x16_iadd = nir_op.define('nir_op_sdot_2x16_iadd', 396)
nir_op_sdot_2x16_iadd_sat = nir_op.define('nir_op_sdot_2x16_iadd_sat', 397)
nir_op_sdot_4x8_iadd = nir_op.define('nir_op_sdot_4x8_iadd', 398)
nir_op_sdot_4x8_iadd_sat = nir_op.define('nir_op_sdot_4x8_iadd_sat', 399)
nir_op_seq = nir_op.define('nir_op_seq', 400)
nir_op_sge = nir_op.define('nir_op_sge', 401)
nir_op_shfr = nir_op.define('nir_op_shfr', 402)
nir_op_shlg_ir3 = nir_op.define('nir_op_shlg_ir3', 403)
nir_op_shlm_ir3 = nir_op.define('nir_op_shlm_ir3', 404)
nir_op_shrg_ir3 = nir_op.define('nir_op_shrg_ir3', 405)
nir_op_shrm_ir3 = nir_op.define('nir_op_shrm_ir3', 406)
nir_op_slt = nir_op.define('nir_op_slt', 407)
nir_op_sne = nir_op.define('nir_op_sne', 408)
nir_op_sudot_4x8_iadd = nir_op.define('nir_op_sudot_4x8_iadd', 409)
nir_op_sudot_4x8_iadd_sat = nir_op.define('nir_op_sudot_4x8_iadd_sat', 410)
nir_op_u2f16 = nir_op.define('nir_op_u2f16', 411)
nir_op_u2f32 = nir_op.define('nir_op_u2f32', 412)
nir_op_u2f64 = nir_op.define('nir_op_u2f64', 413)
nir_op_u2fmp = nir_op.define('nir_op_u2fmp', 414)
nir_op_u2u1 = nir_op.define('nir_op_u2u1', 415)
nir_op_u2u16 = nir_op.define('nir_op_u2u16', 416)
nir_op_u2u32 = nir_op.define('nir_op_u2u32', 417)
nir_op_u2u64 = nir_op.define('nir_op_u2u64', 418)
nir_op_u2u8 = nir_op.define('nir_op_u2u8', 419)
nir_op_uabs_isub = nir_op.define('nir_op_uabs_isub', 420)
nir_op_uabs_usub = nir_op.define('nir_op_uabs_usub', 421)
nir_op_uadd_carry = nir_op.define('nir_op_uadd_carry', 422)
nir_op_uadd_sat = nir_op.define('nir_op_uadd_sat', 423)
nir_op_ubfe = nir_op.define('nir_op_ubfe', 424)
nir_op_ubitfield_extract = nir_op.define('nir_op_ubitfield_extract', 425)
nir_op_uclz = nir_op.define('nir_op_uclz', 426)
nir_op_udiv = nir_op.define('nir_op_udiv', 427)
nir_op_udiv_aligned_4 = nir_op.define('nir_op_udiv_aligned_4', 428)
nir_op_udot_2x16_uadd = nir_op.define('nir_op_udot_2x16_uadd', 429)
nir_op_udot_2x16_uadd_sat = nir_op.define('nir_op_udot_2x16_uadd_sat', 430)
nir_op_udot_4x8_uadd = nir_op.define('nir_op_udot_4x8_uadd', 431)
nir_op_udot_4x8_uadd_sat = nir_op.define('nir_op_udot_4x8_uadd_sat', 432)
nir_op_ufind_msb = nir_op.define('nir_op_ufind_msb', 433)
nir_op_ufind_msb_rev = nir_op.define('nir_op_ufind_msb_rev', 434)
nir_op_uge = nir_op.define('nir_op_uge', 435)
nir_op_uge16 = nir_op.define('nir_op_uge16', 436)
nir_op_uge32 = nir_op.define('nir_op_uge32', 437)
nir_op_uge8 = nir_op.define('nir_op_uge8', 438)
nir_op_uhadd = nir_op.define('nir_op_uhadd', 439)
nir_op_ulea_agx = nir_op.define('nir_op_ulea_agx', 440)
nir_op_ult = nir_op.define('nir_op_ult', 441)
nir_op_ult16 = nir_op.define('nir_op_ult16', 442)
nir_op_ult32 = nir_op.define('nir_op_ult32', 443)
nir_op_ult8 = nir_op.define('nir_op_ult8', 444)
nir_op_umad24 = nir_op.define('nir_op_umad24', 445)
nir_op_umad24_relaxed = nir_op.define('nir_op_umad24_relaxed', 446)
nir_op_umax = nir_op.define('nir_op_umax', 447)
nir_op_umax_4x8_vc4 = nir_op.define('nir_op_umax_4x8_vc4', 448)
nir_op_umin = nir_op.define('nir_op_umin', 449)
nir_op_umin_4x8_vc4 = nir_op.define('nir_op_umin_4x8_vc4', 450)
nir_op_umod = nir_op.define('nir_op_umod', 451)
nir_op_umul24 = nir_op.define('nir_op_umul24', 452)
nir_op_umul24_relaxed = nir_op.define('nir_op_umul24_relaxed', 453)
nir_op_umul_2x32_64 = nir_op.define('nir_op_umul_2x32_64', 454)
nir_op_umul_32x16 = nir_op.define('nir_op_umul_32x16', 455)
nir_op_umul_high = nir_op.define('nir_op_umul_high', 456)
nir_op_umul_low = nir_op.define('nir_op_umul_low', 457)
nir_op_umul_unorm_4x8_vc4 = nir_op.define('nir_op_umul_unorm_4x8_vc4', 458)
nir_op_unpack_32_2x16 = nir_op.define('nir_op_unpack_32_2x16', 459)
nir_op_unpack_32_2x16_split_x = nir_op.define('nir_op_unpack_32_2x16_split_x', 460)
nir_op_unpack_32_2x16_split_y = nir_op.define('nir_op_unpack_32_2x16_split_y', 461)
nir_op_unpack_32_4x8 = nir_op.define('nir_op_unpack_32_4x8', 462)
nir_op_unpack_64_2x32 = nir_op.define('nir_op_unpack_64_2x32', 463)
nir_op_unpack_64_2x32_split_x = nir_op.define('nir_op_unpack_64_2x32_split_x', 464)
nir_op_unpack_64_2x32_split_y = nir_op.define('nir_op_unpack_64_2x32_split_y', 465)
nir_op_unpack_64_4x16 = nir_op.define('nir_op_unpack_64_4x16', 466)
nir_op_unpack_double_2x32_dxil = nir_op.define('nir_op_unpack_double_2x32_dxil', 467)
nir_op_unpack_half_2x16 = nir_op.define('nir_op_unpack_half_2x16', 468)
nir_op_unpack_half_2x16_split_x = nir_op.define('nir_op_unpack_half_2x16_split_x', 469)
nir_op_unpack_half_2x16_split_y = nir_op.define('nir_op_unpack_half_2x16_split_y', 470)
nir_op_unpack_snorm_2x16 = nir_op.define('nir_op_unpack_snorm_2x16', 471)
nir_op_unpack_snorm_4x8 = nir_op.define('nir_op_unpack_snorm_4x8', 472)
nir_op_unpack_unorm_2x16 = nir_op.define('nir_op_unpack_unorm_2x16', 473)
nir_op_unpack_unorm_4x8 = nir_op.define('nir_op_unpack_unorm_4x8', 474)
nir_op_urhadd = nir_op.define('nir_op_urhadd', 475)
nir_op_urol = nir_op.define('nir_op_urol', 476)
nir_op_uror = nir_op.define('nir_op_uror', 477)
nir_op_usadd_4x8_vc4 = nir_op.define('nir_op_usadd_4x8_vc4', 478)
nir_op_ushr = nir_op.define('nir_op_ushr', 479)
nir_op_ussub_4x8_vc4 = nir_op.define('nir_op_ussub_4x8_vc4', 480)
nir_op_usub_borrow = nir_op.define('nir_op_usub_borrow', 481)
nir_op_usub_sat = nir_op.define('nir_op_usub_sat', 482)
nir_op_vec16 = nir_op.define('nir_op_vec16', 483)
nir_op_vec2 = nir_op.define('nir_op_vec2', 484)
nir_op_vec3 = nir_op.define('nir_op_vec3', 485)
nir_op_vec4 = nir_op.define('nir_op_vec4', 486)
nir_op_vec5 = nir_op.define('nir_op_vec5', 487)
nir_op_vec8 = nir_op.define('nir_op_vec8', 488)
nir_last_opcode = nir_op.define('nir_last_opcode', 488)
nir_num_opcodes = nir_op.define('nir_num_opcodes', 489)

try: (nir_type_conversion_op:=dll.nir_type_conversion_op).restype, nir_type_conversion_op.argtypes = nir_op, [nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError: pass

nir_atomic_op = CEnum(ctypes.c_uint32)
nir_atomic_op_iadd = nir_atomic_op.define('nir_atomic_op_iadd', 0)
nir_atomic_op_imin = nir_atomic_op.define('nir_atomic_op_imin', 1)
nir_atomic_op_umin = nir_atomic_op.define('nir_atomic_op_umin', 2)
nir_atomic_op_imax = nir_atomic_op.define('nir_atomic_op_imax', 3)
nir_atomic_op_umax = nir_atomic_op.define('nir_atomic_op_umax', 4)
nir_atomic_op_iand = nir_atomic_op.define('nir_atomic_op_iand', 5)
nir_atomic_op_ior = nir_atomic_op.define('nir_atomic_op_ior', 6)
nir_atomic_op_ixor = nir_atomic_op.define('nir_atomic_op_ixor', 7)
nir_atomic_op_xchg = nir_atomic_op.define('nir_atomic_op_xchg', 8)
nir_atomic_op_fadd = nir_atomic_op.define('nir_atomic_op_fadd', 9)
nir_atomic_op_fmin = nir_atomic_op.define('nir_atomic_op_fmin', 10)
nir_atomic_op_fmax = nir_atomic_op.define('nir_atomic_op_fmax', 11)
nir_atomic_op_cmpxchg = nir_atomic_op.define('nir_atomic_op_cmpxchg', 12)
nir_atomic_op_fcmpxchg = nir_atomic_op.define('nir_atomic_op_fcmpxchg', 13)
nir_atomic_op_inc_wrap = nir_atomic_op.define('nir_atomic_op_inc_wrap', 14)
nir_atomic_op_dec_wrap = nir_atomic_op.define('nir_atomic_op_dec_wrap', 15)
nir_atomic_op_ordered_add_gfx12_amd = nir_atomic_op.define('nir_atomic_op_ordered_add_gfx12_amd', 16)

try: (nir_atomic_op_to_alu:=dll.nir_atomic_op_to_alu).restype, nir_atomic_op_to_alu.argtypes = nir_op, [nir_atomic_op]
except AttributeError: pass

try: (nir_op_vec:=dll.nir_op_vec).restype, nir_op_vec.argtypes = nir_op, [ctypes.c_uint32]
except AttributeError: pass

try: (nir_op_is_vec:=dll.nir_op_is_vec).restype, nir_op_is_vec.argtypes = ctypes.c_bool, [nir_op]
except AttributeError: pass

nir_op_algebraic_property = CEnum(ctypes.c_uint32)
NIR_OP_IS_2SRC_COMMUTATIVE = nir_op_algebraic_property.define('NIR_OP_IS_2SRC_COMMUTATIVE', 1)
NIR_OP_IS_ASSOCIATIVE = nir_op_algebraic_property.define('NIR_OP_IS_ASSOCIATIVE', 2)
NIR_OP_IS_SELECTION = nir_op_algebraic_property.define('NIR_OP_IS_SELECTION', 4)

class struct_nir_op_info(Struct): pass
struct_nir_op_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('num_inputs', uint8_t),
  ('output_size', uint8_t),
  ('output_type', nir_alu_type),
  ('input_sizes', (uint8_t * 16)),
  ('input_types', (nir_alu_type * 16)),
  ('algebraic_properties', nir_op_algebraic_property),
  ('is_conversion', ctypes.c_bool),
]
nir_op_info = struct_nir_op_info
try: nir_op_infos = (nir_op_info * 489).in_dll(dll, 'nir_op_infos')
except (ValueError,AttributeError): pass
class struct_nir_alu_instr(Struct): pass
struct_nir_alu_instr._fields_ = [
  ('instr', nir_instr),
  ('op', nir_op),
  ('exact', ctypes.c_bool,1),
  ('no_signed_wrap', ctypes.c_bool,1),
  ('no_unsigned_wrap', ctypes.c_bool,1),
  ('fp_fast_math', uint32_t,9),
  ('def', nir_def),
  ('src', (nir_alu_src * 0)),
]
nir_alu_instr = struct_nir_alu_instr
try: (nir_alu_src_copy:=dll.nir_alu_src_copy).restype, nir_alu_src_copy.argtypes = None, [ctypes.POINTER(nir_alu_src), ctypes.POINTER(nir_alu_src)]
except AttributeError: pass

try: (nir_alu_instr_src_read_mask:=dll.nir_alu_instr_src_read_mask).restype, nir_alu_instr_src_read_mask.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_ssa_alu_instr_src_components:=dll.nir_ssa_alu_instr_src_components).restype, nir_ssa_alu_instr_src_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_instr_is_comparison:=dll.nir_alu_instr_is_comparison).restype, nir_alu_instr_is_comparison.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_const_value_negative_equal:=dll.nir_const_value_negative_equal).restype, nir_const_value_negative_equal.argtypes = ctypes.c_bool, [nir_const_value, nir_const_value, nir_alu_type]
except AttributeError: pass

try: (nir_alu_srcs_equal:=dll.nir_alu_srcs_equal).restype, nir_alu_srcs_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_srcs_negative_equal_typed:=dll.nir_alu_srcs_negative_equal_typed).restype, nir_alu_srcs_negative_equal_typed.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32, nir_alu_type]
except AttributeError: pass

try: (nir_alu_srcs_negative_equal:=dll.nir_alu_srcs_negative_equal).restype, nir_alu_srcs_negative_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_src_is_trivial_ssa:=dll.nir_alu_src_is_trivial_ssa).restype, nir_alu_src_is_trivial_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

nir_deref_type = CEnum(ctypes.c_uint32)
nir_deref_type_var = nir_deref_type.define('nir_deref_type_var', 0)
nir_deref_type_array = nir_deref_type.define('nir_deref_type_array', 1)
nir_deref_type_array_wildcard = nir_deref_type.define('nir_deref_type_array_wildcard', 2)
nir_deref_type_ptr_as_array = nir_deref_type.define('nir_deref_type_ptr_as_array', 3)
nir_deref_type_struct = nir_deref_type.define('nir_deref_type_struct', 4)
nir_deref_type_cast = nir_deref_type.define('nir_deref_type_cast', 5)

class struct_nir_deref_instr(Struct): pass
nir_variable_mode = CEnum(ctypes.c_uint32)
nir_var_system_value = nir_variable_mode.define('nir_var_system_value', 1)
nir_var_uniform = nir_variable_mode.define('nir_var_uniform', 2)
nir_var_shader_in = nir_variable_mode.define('nir_var_shader_in', 4)
nir_var_shader_out = nir_variable_mode.define('nir_var_shader_out', 8)
nir_var_image = nir_variable_mode.define('nir_var_image', 16)
nir_var_shader_call_data = nir_variable_mode.define('nir_var_shader_call_data', 32)
nir_var_ray_hit_attrib = nir_variable_mode.define('nir_var_ray_hit_attrib', 64)
nir_var_mem_ubo = nir_variable_mode.define('nir_var_mem_ubo', 128)
nir_var_mem_push_const = nir_variable_mode.define('nir_var_mem_push_const', 256)
nir_var_mem_ssbo = nir_variable_mode.define('nir_var_mem_ssbo', 512)
nir_var_mem_constant = nir_variable_mode.define('nir_var_mem_constant', 1024)
nir_var_mem_task_payload = nir_variable_mode.define('nir_var_mem_task_payload', 2048)
nir_var_mem_node_payload = nir_variable_mode.define('nir_var_mem_node_payload', 4096)
nir_var_mem_node_payload_in = nir_variable_mode.define('nir_var_mem_node_payload_in', 8192)
nir_var_function_in = nir_variable_mode.define('nir_var_function_in', 16384)
nir_var_function_out = nir_variable_mode.define('nir_var_function_out', 32768)
nir_var_function_inout = nir_variable_mode.define('nir_var_function_inout', 65536)
nir_var_shader_temp = nir_variable_mode.define('nir_var_shader_temp', 131072)
nir_var_function_temp = nir_variable_mode.define('nir_var_function_temp', 262144)
nir_var_mem_shared = nir_variable_mode.define('nir_var_mem_shared', 524288)
nir_var_mem_global = nir_variable_mode.define('nir_var_mem_global', 1048576)
nir_var_mem_generic = nir_variable_mode.define('nir_var_mem_generic', 1966080)
nir_var_read_only_modes = nir_variable_mode.define('nir_var_read_only_modes', 1159)
nir_var_vec_indexable_modes = nir_variable_mode.define('nir_var_vec_indexable_modes', 1969033)
nir_num_variable_modes = nir_variable_mode.define('nir_num_variable_modes', 21)
nir_var_all = nir_variable_mode.define('nir_var_all', 2097151)

class struct_nir_deref_instr_0(ctypes.Union): pass
struct_nir_deref_instr_0._fields_ = [
  ('var', ctypes.POINTER(nir_variable)),
  ('parent', nir_src),
]
class struct_nir_deref_instr_1(ctypes.Union): pass
class struct_nir_deref_instr_1_arr(Struct): pass
struct_nir_deref_instr_1_arr._fields_ = [
  ('index', nir_src),
  ('in_bounds', ctypes.c_bool),
]
class struct_nir_deref_instr_1_strct(Struct): pass
struct_nir_deref_instr_1_strct._fields_ = [
  ('index', ctypes.c_uint32),
]
class struct_nir_deref_instr_1_cast(Struct): pass
struct_nir_deref_instr_1_cast._fields_ = [
  ('ptr_stride', ctypes.c_uint32),
  ('align_mul', ctypes.c_uint32),
  ('align_offset', ctypes.c_uint32),
]
struct_nir_deref_instr_1._fields_ = [
  ('arr', struct_nir_deref_instr_1_arr),
  ('strct', struct_nir_deref_instr_1_strct),
  ('cast', struct_nir_deref_instr_1_cast),
]
struct_nir_deref_instr._anonymous_ = ['_0', '_1']
struct_nir_deref_instr._fields_ = [
  ('instr', nir_instr),
  ('deref_type', nir_deref_type),
  ('modes', nir_variable_mode),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('_0', struct_nir_deref_instr_0),
  ('_1', struct_nir_deref_instr_1),
  ('def', nir_def),
]
nir_deref_instr = struct_nir_deref_instr
try: (nir_deref_cast_is_trivial:=dll.nir_deref_cast_is_trivial).restype, nir_deref_cast_is_trivial.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_has_indirect:=dll.nir_deref_instr_has_indirect).restype, nir_deref_instr_has_indirect.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_is_known_out_of_bounds:=dll.nir_deref_instr_is_known_out_of_bounds).restype, nir_deref_instr_is_known_out_of_bounds.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

nir_deref_instr_has_complex_use_options = CEnum(ctypes.c_uint32)
nir_deref_instr_has_complex_use_allow_memcpy_src = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_memcpy_src', 1)
nir_deref_instr_has_complex_use_allow_memcpy_dst = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_memcpy_dst', 2)
nir_deref_instr_has_complex_use_allow_atomics = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_atomics', 4)

try: (nir_deref_instr_has_complex_use:=dll.nir_deref_instr_has_complex_use).restype, nir_deref_instr_has_complex_use.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr), nir_deref_instr_has_complex_use_options]
except AttributeError: pass

try: (nir_deref_instr_remove_if_unused:=dll.nir_deref_instr_remove_if_unused).restype, nir_deref_instr_remove_if_unused.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_array_stride:=dll.nir_deref_instr_array_stride).restype, nir_deref_instr_array_stride.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

class struct_nir_call_instr(Struct): pass
class struct_nir_function(Struct): pass
nir_function = struct_nir_function
class struct_nir_shader(Struct): pass
nir_shader = struct_nir_shader
class struct_gc_ctx(Struct): pass
gc_ctx = struct_gc_ctx
class struct_nir_shader_compiler_options(Struct): pass
nir_shader_compiler_options = struct_nir_shader_compiler_options
nir_instr_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
nir_lower_int64_options = CEnum(ctypes.c_uint32)
nir_lower_imul64 = nir_lower_int64_options.define('nir_lower_imul64', 1)
nir_lower_isign64 = nir_lower_int64_options.define('nir_lower_isign64', 2)
nir_lower_divmod64 = nir_lower_int64_options.define('nir_lower_divmod64', 4)
nir_lower_imul_high64 = nir_lower_int64_options.define('nir_lower_imul_high64', 8)
nir_lower_bcsel64 = nir_lower_int64_options.define('nir_lower_bcsel64', 16)
nir_lower_icmp64 = nir_lower_int64_options.define('nir_lower_icmp64', 32)
nir_lower_iadd64 = nir_lower_int64_options.define('nir_lower_iadd64', 64)
nir_lower_iabs64 = nir_lower_int64_options.define('nir_lower_iabs64', 128)
nir_lower_ineg64 = nir_lower_int64_options.define('nir_lower_ineg64', 256)
nir_lower_logic64 = nir_lower_int64_options.define('nir_lower_logic64', 512)
nir_lower_minmax64 = nir_lower_int64_options.define('nir_lower_minmax64', 1024)
nir_lower_shift64 = nir_lower_int64_options.define('nir_lower_shift64', 2048)
nir_lower_imul_2x32_64 = nir_lower_int64_options.define('nir_lower_imul_2x32_64', 4096)
nir_lower_extract64 = nir_lower_int64_options.define('nir_lower_extract64', 8192)
nir_lower_ufind_msb64 = nir_lower_int64_options.define('nir_lower_ufind_msb64', 16384)
nir_lower_bit_count64 = nir_lower_int64_options.define('nir_lower_bit_count64', 32768)
nir_lower_subgroup_shuffle64 = nir_lower_int64_options.define('nir_lower_subgroup_shuffle64', 65536)
nir_lower_scan_reduce_bitwise64 = nir_lower_int64_options.define('nir_lower_scan_reduce_bitwise64', 131072)
nir_lower_scan_reduce_iadd64 = nir_lower_int64_options.define('nir_lower_scan_reduce_iadd64', 262144)
nir_lower_vote_ieq64 = nir_lower_int64_options.define('nir_lower_vote_ieq64', 524288)
nir_lower_usub_sat64 = nir_lower_int64_options.define('nir_lower_usub_sat64', 1048576)
nir_lower_iadd_sat64 = nir_lower_int64_options.define('nir_lower_iadd_sat64', 2097152)
nir_lower_find_lsb64 = nir_lower_int64_options.define('nir_lower_find_lsb64', 4194304)
nir_lower_conv64 = nir_lower_int64_options.define('nir_lower_conv64', 8388608)
nir_lower_uadd_sat64 = nir_lower_int64_options.define('nir_lower_uadd_sat64', 16777216)
nir_lower_iadd3_64 = nir_lower_int64_options.define('nir_lower_iadd3_64', 33554432)
nir_lower_bitfield_reverse64 = nir_lower_int64_options.define('nir_lower_bitfield_reverse64', 67108864)
nir_lower_bitfield_extract64 = nir_lower_int64_options.define('nir_lower_bitfield_extract64', 134217728)

nir_lower_doubles_options = CEnum(ctypes.c_uint32)
nir_lower_drcp = nir_lower_doubles_options.define('nir_lower_drcp', 1)
nir_lower_dsqrt = nir_lower_doubles_options.define('nir_lower_dsqrt', 2)
nir_lower_drsq = nir_lower_doubles_options.define('nir_lower_drsq', 4)
nir_lower_dtrunc = nir_lower_doubles_options.define('nir_lower_dtrunc', 8)
nir_lower_dfloor = nir_lower_doubles_options.define('nir_lower_dfloor', 16)
nir_lower_dceil = nir_lower_doubles_options.define('nir_lower_dceil', 32)
nir_lower_dfract = nir_lower_doubles_options.define('nir_lower_dfract', 64)
nir_lower_dround_even = nir_lower_doubles_options.define('nir_lower_dround_even', 128)
nir_lower_dmod = nir_lower_doubles_options.define('nir_lower_dmod', 256)
nir_lower_dsub = nir_lower_doubles_options.define('nir_lower_dsub', 512)
nir_lower_ddiv = nir_lower_doubles_options.define('nir_lower_ddiv', 1024)
nir_lower_dsign = nir_lower_doubles_options.define('nir_lower_dsign', 2048)
nir_lower_dminmax = nir_lower_doubles_options.define('nir_lower_dminmax', 4096)
nir_lower_dsat = nir_lower_doubles_options.define('nir_lower_dsat', 8192)
nir_lower_fp64_full_software = nir_lower_doubles_options.define('nir_lower_fp64_full_software', 16384)

nir_divergence_options = CEnum(ctypes.c_uint32)
nir_divergence_single_prim_per_subgroup = nir_divergence_options.define('nir_divergence_single_prim_per_subgroup', 1)
nir_divergence_single_patch_per_tcs_subgroup = nir_divergence_options.define('nir_divergence_single_patch_per_tcs_subgroup', 2)
nir_divergence_single_patch_per_tes_subgroup = nir_divergence_options.define('nir_divergence_single_patch_per_tes_subgroup', 4)
nir_divergence_view_index_uniform = nir_divergence_options.define('nir_divergence_view_index_uniform', 8)
nir_divergence_single_frag_shading_rate_per_subgroup = nir_divergence_options.define('nir_divergence_single_frag_shading_rate_per_subgroup', 16)
nir_divergence_multiple_workgroup_per_compute_subgroup = nir_divergence_options.define('nir_divergence_multiple_workgroup_per_compute_subgroup', 32)
nir_divergence_shader_record_ptr_uniform = nir_divergence_options.define('nir_divergence_shader_record_ptr_uniform', 64)
nir_divergence_uniform_load_tears = nir_divergence_options.define('nir_divergence_uniform_load_tears', 128)
nir_divergence_ignore_undef_if_phi_srcs = nir_divergence_options.define('nir_divergence_ignore_undef_if_phi_srcs', 256)

nir_io_options = CEnum(ctypes.c_uint32)
nir_io_has_flexible_input_interpolation_except_flat = nir_io_options.define('nir_io_has_flexible_input_interpolation_except_flat', 1)
nir_io_dont_use_pos_for_non_fs_varyings = nir_io_options.define('nir_io_dont_use_pos_for_non_fs_varyings', 2)
nir_io_16bit_input_output_support = nir_io_options.define('nir_io_16bit_input_output_support', 4)
nir_io_mediump_is_32bit = nir_io_options.define('nir_io_mediump_is_32bit', 8)
nir_io_prefer_scalar_fs_inputs = nir_io_options.define('nir_io_prefer_scalar_fs_inputs', 16)
nir_io_mix_convergent_flat_with_interpolated = nir_io_options.define('nir_io_mix_convergent_flat_with_interpolated', 32)
nir_io_vectorizer_ignores_types = nir_io_options.define('nir_io_vectorizer_ignores_types', 64)
nir_io_always_interpolate_convergent_fs_inputs = nir_io_options.define('nir_io_always_interpolate_convergent_fs_inputs', 128)
nir_io_compaction_rotates_color_channels = nir_io_options.define('nir_io_compaction_rotates_color_channels', 256)
nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups = nir_io_options.define('nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups', 512)
nir_io_radv_intrinsic_component_workaround = nir_io_options.define('nir_io_radv_intrinsic_component_workaround', 1024)
nir_io_has_intrinsics = nir_io_options.define('nir_io_has_intrinsics', 65536)
nir_io_separate_clip_cull_distance_arrays = nir_io_options.define('nir_io_separate_clip_cull_distance_arrays', 131072)

struct_nir_shader_compiler_options._fields_ = [
  ('lower_fdiv', ctypes.c_bool),
  ('lower_ffma16', ctypes.c_bool),
  ('lower_ffma32', ctypes.c_bool),
  ('lower_ffma64', ctypes.c_bool),
  ('fuse_ffma16', ctypes.c_bool),
  ('fuse_ffma32', ctypes.c_bool),
  ('fuse_ffma64', ctypes.c_bool),
  ('lower_flrp16', ctypes.c_bool),
  ('lower_flrp32', ctypes.c_bool),
  ('lower_flrp64', ctypes.c_bool),
  ('lower_fpow', ctypes.c_bool),
  ('lower_fsat', ctypes.c_bool),
  ('lower_fsqrt', ctypes.c_bool),
  ('lower_sincos', ctypes.c_bool),
  ('lower_fmod', ctypes.c_bool),
  ('lower_bitfield_extract8', ctypes.c_bool),
  ('lower_bitfield_extract16', ctypes.c_bool),
  ('lower_bitfield_extract', ctypes.c_bool),
  ('lower_bitfield_insert', ctypes.c_bool),
  ('lower_bitfield_reverse', ctypes.c_bool),
  ('lower_bit_count', ctypes.c_bool),
  ('lower_ifind_msb', ctypes.c_bool),
  ('lower_ufind_msb', ctypes.c_bool),
  ('lower_find_lsb', ctypes.c_bool),
  ('lower_uadd_carry', ctypes.c_bool),
  ('lower_usub_borrow', ctypes.c_bool),
  ('lower_mul_high', ctypes.c_bool),
  ('lower_mul_high16', ctypes.c_bool),
  ('lower_fneg', ctypes.c_bool),
  ('lower_ineg', ctypes.c_bool),
  ('lower_fisnormal', ctypes.c_bool),
  ('lower_scmp', ctypes.c_bool),
  ('lower_vector_cmp', ctypes.c_bool),
  ('lower_bitops', ctypes.c_bool),
  ('lower_isign', ctypes.c_bool),
  ('lower_fsign', ctypes.c_bool),
  ('lower_iabs', ctypes.c_bool),
  ('lower_umax', ctypes.c_bool),
  ('lower_umin', ctypes.c_bool),
  ('lower_fminmax_signed_zero', ctypes.c_bool),
  ('lower_fdph', ctypes.c_bool),
  ('fdot_replicates', ctypes.c_bool),
  ('lower_ffloor', ctypes.c_bool),
  ('lower_ffract', ctypes.c_bool),
  ('lower_fceil', ctypes.c_bool),
  ('lower_ftrunc', ctypes.c_bool),
  ('lower_fround_even', ctypes.c_bool),
  ('lower_ldexp', ctypes.c_bool),
  ('lower_pack_half_2x16', ctypes.c_bool),
  ('lower_pack_unorm_2x16', ctypes.c_bool),
  ('lower_pack_snorm_2x16', ctypes.c_bool),
  ('lower_pack_unorm_4x8', ctypes.c_bool),
  ('lower_pack_snorm_4x8', ctypes.c_bool),
  ('lower_pack_64_2x32', ctypes.c_bool),
  ('lower_pack_64_4x16', ctypes.c_bool),
  ('lower_pack_32_2x16', ctypes.c_bool),
  ('lower_pack_64_2x32_split', ctypes.c_bool),
  ('lower_pack_32_2x16_split', ctypes.c_bool),
  ('lower_unpack_half_2x16', ctypes.c_bool),
  ('lower_unpack_unorm_2x16', ctypes.c_bool),
  ('lower_unpack_snorm_2x16', ctypes.c_bool),
  ('lower_unpack_unorm_4x8', ctypes.c_bool),
  ('lower_unpack_snorm_4x8', ctypes.c_bool),
  ('lower_unpack_64_2x32_split', ctypes.c_bool),
  ('lower_unpack_32_2x16_split', ctypes.c_bool),
  ('lower_pack_split', ctypes.c_bool),
  ('lower_extract_byte', ctypes.c_bool),
  ('lower_extract_word', ctypes.c_bool),
  ('lower_insert_byte', ctypes.c_bool),
  ('lower_insert_word', ctypes.c_bool),
  ('vertex_id_zero_based', ctypes.c_bool),
  ('lower_base_vertex', ctypes.c_bool),
  ('instance_id_includes_base_index', ctypes.c_bool),
  ('lower_helper_invocation', ctypes.c_bool),
  ('optimize_sample_mask_in', ctypes.c_bool),
  ('optimize_load_front_face_fsign', ctypes.c_bool),
  ('optimize_quad_vote_to_reduce', ctypes.c_bool),
  ('lower_cs_local_index_to_id', ctypes.c_bool),
  ('lower_cs_local_id_to_index', ctypes.c_bool),
  ('has_cs_global_id', ctypes.c_bool),
  ('lower_device_index_to_zero', ctypes.c_bool),
  ('lower_wpos_pntc', ctypes.c_bool),
  ('lower_hadd', ctypes.c_bool),
  ('lower_hadd64', ctypes.c_bool),
  ('lower_uadd_sat', ctypes.c_bool),
  ('lower_usub_sat', ctypes.c_bool),
  ('lower_iadd_sat', ctypes.c_bool),
  ('lower_mul_32x16', ctypes.c_bool),
  ('lower_bfloat16_conversions', ctypes.c_bool),
  ('vectorize_tess_levels', ctypes.c_bool),
  ('lower_to_scalar', ctypes.c_bool),
  ('lower_to_scalar_filter', nir_instr_filter_cb),
  ('vectorize_vec2_16bit', ctypes.c_bool),
  ('unify_interfaces', ctypes.c_bool),
  ('lower_interpolate_at', ctypes.c_bool),
  ('lower_mul_2x32_64', ctypes.c_bool),
  ('has_rotate8', ctypes.c_bool),
  ('has_rotate16', ctypes.c_bool),
  ('has_rotate32', ctypes.c_bool),
  ('has_shfr32', ctypes.c_bool),
  ('has_iadd3', ctypes.c_bool),
  ('has_amul', ctypes.c_bool),
  ('has_imul24', ctypes.c_bool),
  ('has_umul24', ctypes.c_bool),
  ('has_mul24_relaxed', ctypes.c_bool),
  ('has_imad32', ctypes.c_bool),
  ('has_umad24', ctypes.c_bool),
  ('has_fused_comp_and_csel', ctypes.c_bool),
  ('has_icsel_eqz64', ctypes.c_bool),
  ('has_icsel_eqz32', ctypes.c_bool),
  ('has_icsel_eqz16', ctypes.c_bool),
  ('has_fneo_fcmpu', ctypes.c_bool),
  ('has_ford_funord', ctypes.c_bool),
  ('has_fsub', ctypes.c_bool),
  ('has_isub', ctypes.c_bool),
  ('has_pack_32_4x8', ctypes.c_bool),
  ('has_texture_scaling', ctypes.c_bool),
  ('has_sdot_4x8', ctypes.c_bool),
  ('has_udot_4x8', ctypes.c_bool),
  ('has_sudot_4x8', ctypes.c_bool),
  ('has_sdot_4x8_sat', ctypes.c_bool),
  ('has_udot_4x8_sat', ctypes.c_bool),
  ('has_sudot_4x8_sat', ctypes.c_bool),
  ('has_dot_2x16', ctypes.c_bool),
  ('has_bfdot2_bfadd', ctypes.c_bool),
  ('has_fmulz', ctypes.c_bool),
  ('has_fmulz_no_denorms', ctypes.c_bool),
  ('has_find_msb_rev', ctypes.c_bool),
  ('has_pack_half_2x16_rtz', ctypes.c_bool),
  ('has_bit_test', ctypes.c_bool),
  ('has_bfe', ctypes.c_bool),
  ('has_bfm', ctypes.c_bool),
  ('has_bfi', ctypes.c_bool),
  ('has_bitfield_select', ctypes.c_bool),
  ('has_uclz', ctypes.c_bool),
  ('has_msad', ctypes.c_bool),
  ('has_f2e4m3fn_satfn', ctypes.c_bool),
  ('has_load_global_bounded', ctypes.c_bool),
  ('intel_vec4', ctypes.c_bool),
  ('avoid_ternary_with_two_constants', ctypes.c_bool),
  ('support_8bit_alu', ctypes.c_bool),
  ('support_16bit_alu', ctypes.c_bool),
  ('max_unroll_iterations', ctypes.c_uint32),
  ('max_unroll_iterations_aggressive', ctypes.c_uint32),
  ('max_unroll_iterations_fp64', ctypes.c_uint32),
  ('lower_uniforms_to_ubo', ctypes.c_bool),
  ('force_indirect_unrolling_sampler', ctypes.c_bool),
  ('no_integers', ctypes.c_bool),
  ('force_indirect_unrolling', nir_variable_mode),
  ('driver_functions', ctypes.c_bool),
  ('late_lower_int64', ctypes.c_bool),
  ('lower_int64_options', nir_lower_int64_options),
  ('lower_doubles_options', nir_lower_doubles_options),
  ('divergence_analysis_options', nir_divergence_options),
  ('support_indirect_inputs', uint8_t),
  ('support_indirect_outputs', uint8_t),
  ('lower_image_offset_to_range_base', ctypes.c_bool),
  ('lower_atomic_offset_to_range_base', ctypes.c_bool),
  ('preserve_mediump', ctypes.c_bool),
  ('lower_fquantize2f16', ctypes.c_bool),
  ('force_f2f16_rtz', ctypes.c_bool),
  ('lower_layer_fs_input_to_sysval', ctypes.c_bool),
  ('compact_arrays', ctypes.c_bool),
  ('discard_is_demote', ctypes.c_bool),
  ('has_ddx_intrinsics', ctypes.c_bool),
  ('scalarize_ddx', ctypes.c_bool),
  ('per_view_unique_driver_locations', ctypes.c_bool),
  ('compact_view_index', ctypes.c_bool),
  ('io_options', nir_io_options),
  ('skip_lower_packing_ops', ctypes.c_uint32),
  ('lower_mediump_io', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_shader))),
  ('varying_expression_max_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader))),
  ('varying_estimate_instr_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr))),
  ('max_varying_expression_cost', ctypes.c_uint32),
]
class struct_shader_info(Struct): pass
blake3_hash = (ctypes.c_ubyte * 32)
enum_pipe_shader_type = CEnum(ctypes.c_int32)
MESA_SHADER_NONE = enum_pipe_shader_type.define('MESA_SHADER_NONE', -1)
MESA_SHADER_VERTEX = enum_pipe_shader_type.define('MESA_SHADER_VERTEX', 0)
PIPE_SHADER_VERTEX = enum_pipe_shader_type.define('PIPE_SHADER_VERTEX', 0)
MESA_SHADER_TESS_CTRL = enum_pipe_shader_type.define('MESA_SHADER_TESS_CTRL', 1)
PIPE_SHADER_TESS_CTRL = enum_pipe_shader_type.define('PIPE_SHADER_TESS_CTRL', 1)
MESA_SHADER_TESS_EVAL = enum_pipe_shader_type.define('MESA_SHADER_TESS_EVAL', 2)
PIPE_SHADER_TESS_EVAL = enum_pipe_shader_type.define('PIPE_SHADER_TESS_EVAL', 2)
MESA_SHADER_GEOMETRY = enum_pipe_shader_type.define('MESA_SHADER_GEOMETRY', 3)
PIPE_SHADER_GEOMETRY = enum_pipe_shader_type.define('PIPE_SHADER_GEOMETRY', 3)
MESA_SHADER_FRAGMENT = enum_pipe_shader_type.define('MESA_SHADER_FRAGMENT', 4)
PIPE_SHADER_FRAGMENT = enum_pipe_shader_type.define('PIPE_SHADER_FRAGMENT', 4)
MESA_SHADER_COMPUTE = enum_pipe_shader_type.define('MESA_SHADER_COMPUTE', 5)
PIPE_SHADER_COMPUTE = enum_pipe_shader_type.define('PIPE_SHADER_COMPUTE', 5)
PIPE_SHADER_TYPES = enum_pipe_shader_type.define('PIPE_SHADER_TYPES', 6)
MESA_SHADER_TASK = enum_pipe_shader_type.define('MESA_SHADER_TASK', 6)
PIPE_SHADER_TASK = enum_pipe_shader_type.define('PIPE_SHADER_TASK', 6)
MESA_SHADER_MESH = enum_pipe_shader_type.define('MESA_SHADER_MESH', 7)
PIPE_SHADER_MESH = enum_pipe_shader_type.define('PIPE_SHADER_MESH', 7)
PIPE_SHADER_MESH_TYPES = enum_pipe_shader_type.define('PIPE_SHADER_MESH_TYPES', 8)
MESA_SHADER_RAYGEN = enum_pipe_shader_type.define('MESA_SHADER_RAYGEN', 8)
MESA_SHADER_ANY_HIT = enum_pipe_shader_type.define('MESA_SHADER_ANY_HIT', 9)
MESA_SHADER_CLOSEST_HIT = enum_pipe_shader_type.define('MESA_SHADER_CLOSEST_HIT', 10)
MESA_SHADER_MISS = enum_pipe_shader_type.define('MESA_SHADER_MISS', 11)
MESA_SHADER_INTERSECTION = enum_pipe_shader_type.define('MESA_SHADER_INTERSECTION', 12)
MESA_SHADER_CALLABLE = enum_pipe_shader_type.define('MESA_SHADER_CALLABLE', 13)
MESA_SHADER_KERNEL = enum_pipe_shader_type.define('MESA_SHADER_KERNEL', 14)

gl_shader_stage = enum_pipe_shader_type
enum_gl_subgroup_size = CEnum(ctypes.c_ubyte)
SUBGROUP_SIZE_VARYING = enum_gl_subgroup_size.define('SUBGROUP_SIZE_VARYING', 0)
SUBGROUP_SIZE_UNIFORM = enum_gl_subgroup_size.define('SUBGROUP_SIZE_UNIFORM', 1)
SUBGROUP_SIZE_API_CONSTANT = enum_gl_subgroup_size.define('SUBGROUP_SIZE_API_CONSTANT', 2)
SUBGROUP_SIZE_FULL_SUBGROUPS = enum_gl_subgroup_size.define('SUBGROUP_SIZE_FULL_SUBGROUPS', 3)
SUBGROUP_SIZE_REQUIRE_4 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_4', 4)
SUBGROUP_SIZE_REQUIRE_8 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_8', 8)
SUBGROUP_SIZE_REQUIRE_16 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_16', 16)
SUBGROUP_SIZE_REQUIRE_32 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_32', 32)
SUBGROUP_SIZE_REQUIRE_64 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_64', 64)
SUBGROUP_SIZE_REQUIRE_128 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_128', 128)

enum_gl_derivative_group = CEnum(ctypes.c_uint32)
DERIVATIVE_GROUP_NONE = enum_gl_derivative_group.define('DERIVATIVE_GROUP_NONE', 0)
DERIVATIVE_GROUP_QUADS = enum_gl_derivative_group.define('DERIVATIVE_GROUP_QUADS', 1)
DERIVATIVE_GROUP_LINEAR = enum_gl_derivative_group.define('DERIVATIVE_GROUP_LINEAR', 2)

class struct_shader_info_0(ctypes.Union): pass
class struct_shader_info_0_vs(Struct): pass
struct_shader_info_0_vs._fields_ = [
  ('double_inputs', uint64_t),
  ('blit_sgprs_amd', uint8_t,4),
  ('tes_agx', ctypes.c_bool,1),
  ('window_space_position', ctypes.c_bool,1),
  ('needs_edge_flag', ctypes.c_bool,1),
]
class struct_shader_info_0_gs(Struct): pass
enum_mesa_prim = CEnum(ctypes.c_ubyte)
MESA_PRIM_POINTS = enum_mesa_prim.define('MESA_PRIM_POINTS', 0)
MESA_PRIM_LINES = enum_mesa_prim.define('MESA_PRIM_LINES', 1)
MESA_PRIM_LINE_LOOP = enum_mesa_prim.define('MESA_PRIM_LINE_LOOP', 2)
MESA_PRIM_LINE_STRIP = enum_mesa_prim.define('MESA_PRIM_LINE_STRIP', 3)
MESA_PRIM_TRIANGLES = enum_mesa_prim.define('MESA_PRIM_TRIANGLES', 4)
MESA_PRIM_TRIANGLE_STRIP = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_STRIP', 5)
MESA_PRIM_TRIANGLE_FAN = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_FAN', 6)
MESA_PRIM_QUADS = enum_mesa_prim.define('MESA_PRIM_QUADS', 7)
MESA_PRIM_QUAD_STRIP = enum_mesa_prim.define('MESA_PRIM_QUAD_STRIP', 8)
MESA_PRIM_POLYGON = enum_mesa_prim.define('MESA_PRIM_POLYGON', 9)
MESA_PRIM_LINES_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_LINES_ADJACENCY', 10)
MESA_PRIM_LINE_STRIP_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_LINE_STRIP_ADJACENCY', 11)
MESA_PRIM_TRIANGLES_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_TRIANGLES_ADJACENCY', 12)
MESA_PRIM_TRIANGLE_STRIP_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_STRIP_ADJACENCY', 13)
MESA_PRIM_PATCHES = enum_mesa_prim.define('MESA_PRIM_PATCHES', 14)
MESA_PRIM_MAX = enum_mesa_prim.define('MESA_PRIM_MAX', 14)
MESA_PRIM_COUNT = enum_mesa_prim.define('MESA_PRIM_COUNT', 15)
MESA_PRIM_UNKNOWN = enum_mesa_prim.define('MESA_PRIM_UNKNOWN', 28)

struct_shader_info_0_gs._fields_ = [
  ('output_primitive', enum_mesa_prim),
  ('input_primitive', enum_mesa_prim),
  ('vertices_out', uint16_t),
  ('invocations', uint8_t),
  ('vertices_in', uint8_t,3),
  ('uses_end_primitive', ctypes.c_bool,1),
  ('active_stream_mask', uint8_t,4),
]
class struct_shader_info_0_fs(Struct): pass
enum_gl_frag_depth_layout = CEnum(ctypes.c_uint32)
FRAG_DEPTH_LAYOUT_NONE = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_NONE', 0)
FRAG_DEPTH_LAYOUT_ANY = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_ANY', 1)
FRAG_DEPTH_LAYOUT_GREATER = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_GREATER', 2)
FRAG_DEPTH_LAYOUT_LESS = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_LESS', 3)
FRAG_DEPTH_LAYOUT_UNCHANGED = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_UNCHANGED', 4)

enum_gl_frag_stencil_layout = CEnum(ctypes.c_uint32)
FRAG_STENCIL_LAYOUT_NONE = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_NONE', 0)
FRAG_STENCIL_LAYOUT_ANY = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_ANY', 1)
FRAG_STENCIL_LAYOUT_GREATER = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_GREATER', 2)
FRAG_STENCIL_LAYOUT_LESS = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_LESS', 3)
FRAG_STENCIL_LAYOUT_UNCHANGED = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_UNCHANGED', 4)

struct_shader_info_0_fs._fields_ = [
  ('uses_discard', ctypes.c_bool,1),
  ('uses_fbfetch_output', ctypes.c_bool,1),
  ('fbfetch_coherent', ctypes.c_bool,1),
  ('color_is_dual_source', ctypes.c_bool,1),
  ('require_full_quads', ctypes.c_bool,1),
  ('quad_derivatives', ctypes.c_bool,1),
  ('needs_coarse_quad_helper_invocations', ctypes.c_bool,1),
  ('needs_full_quad_helper_invocations', ctypes.c_bool,1),
  ('uses_sample_qualifier', ctypes.c_bool,1),
  ('uses_sample_shading', ctypes.c_bool,1),
  ('early_fragment_tests', ctypes.c_bool,1),
  ('inner_coverage', ctypes.c_bool,1),
  ('post_depth_coverage', ctypes.c_bool,1),
  ('pixel_center_integer', ctypes.c_bool,1),
  ('origin_upper_left', ctypes.c_bool,1),
  ('pixel_interlock_ordered', ctypes.c_bool,1),
  ('pixel_interlock_unordered', ctypes.c_bool,1),
  ('sample_interlock_ordered', ctypes.c_bool,1),
  ('sample_interlock_unordered', ctypes.c_bool,1),
  ('untyped_color_outputs', ctypes.c_bool,1),
  ('depth_layout', enum_gl_frag_depth_layout,3),
  ('color0_interp', ctypes.c_uint32,3),
  ('color0_sample', ctypes.c_bool,1),
  ('color0_centroid', ctypes.c_bool,1),
  ('color1_interp', ctypes.c_uint32,3),
  ('color1_sample', ctypes.c_bool,1),
  ('color1_centroid', ctypes.c_bool,1),
  ('advanced_blend_modes', ctypes.c_uint32),
  ('early_and_late_fragment_tests', ctypes.c_bool,1),
  ('stencil_front_layout', enum_gl_frag_stencil_layout,3),
  ('stencil_back_layout', enum_gl_frag_stencil_layout,3),
]
class struct_shader_info_0_cs(Struct): pass
struct_shader_info_0_cs._fields_ = [
  ('workgroup_size_hint', (uint16_t * 3)),
  ('user_data_components_amd', uint8_t,4),
  ('has_variable_shared_mem', ctypes.c_bool,1),
  ('has_cooperative_matrix', ctypes.c_bool,1),
  ('image_block_size_per_thread_agx', uint8_t),
  ('ptr_size', ctypes.c_uint32),
  ('shader_index', uint32_t),
  ('node_payloads_size', uint32_t),
  ('workgroup_count', (uint32_t * 3)),
]
class struct_shader_info_0_tess(Struct): pass
enum_tess_primitive_mode = CEnum(ctypes.c_uint32)
TESS_PRIMITIVE_UNSPECIFIED = enum_tess_primitive_mode.define('TESS_PRIMITIVE_UNSPECIFIED', 0)
TESS_PRIMITIVE_TRIANGLES = enum_tess_primitive_mode.define('TESS_PRIMITIVE_TRIANGLES', 1)
TESS_PRIMITIVE_QUADS = enum_tess_primitive_mode.define('TESS_PRIMITIVE_QUADS', 2)
TESS_PRIMITIVE_ISOLINES = enum_tess_primitive_mode.define('TESS_PRIMITIVE_ISOLINES', 3)

struct_shader_info_0_tess._fields_ = [
  ('_primitive_mode', enum_tess_primitive_mode),
  ('tcs_vertices_out', uint8_t),
  ('spacing', ctypes.c_uint32,2),
  ('ccw', ctypes.c_bool,1),
  ('point_mode', ctypes.c_bool,1),
  ('tcs_same_invocation_inputs_read', uint64_t),
  ('tcs_cross_invocation_inputs_read', uint64_t),
  ('tcs_cross_invocation_outputs_read', uint64_t),
  ('tcs_cross_invocation_outputs_written', uint64_t),
  ('tcs_outputs_read_by_tes', uint64_t),
  ('tcs_patch_outputs_read_by_tes', uint32_t),
  ('tcs_outputs_read_by_tes_16bit', uint16_t),
]
class struct_shader_info_0_mesh(Struct): pass
struct_shader_info_0_mesh._fields_ = [
  ('ms_cross_invocation_output_access', uint64_t),
  ('ts_mesh_dispatch_dimensions', (uint32_t * 3)),
  ('max_vertices_out', uint16_t),
  ('max_primitives_out', uint16_t),
  ('primitive_type', enum_mesa_prim),
  ('nv', ctypes.c_bool),
]
struct_shader_info_0._fields_ = [
  ('vs', struct_shader_info_0_vs),
  ('gs', struct_shader_info_0_gs),
  ('fs', struct_shader_info_0_fs),
  ('cs', struct_shader_info_0_cs),
  ('tess', struct_shader_info_0_tess),
  ('mesh', struct_shader_info_0_mesh),
]
struct_shader_info._anonymous_ = ['_0']
struct_shader_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('label', ctypes.POINTER(ctypes.c_char)),
  ('internal', ctypes.c_bool),
  ('source_blake3', blake3_hash),
  ('stage', gl_shader_stage,8),
  ('prev_stage', gl_shader_stage,8),
  ('next_stage', gl_shader_stage,8),
  ('prev_stage_has_xfb', ctypes.c_bool),
  ('num_textures', uint8_t),
  ('num_ubos', uint8_t),
  ('num_abos', uint8_t),
  ('num_ssbos', uint8_t),
  ('num_images', uint8_t),
  ('inputs_read', uint64_t),
  ('dual_slot_inputs', uint64_t),
  ('outputs_written', uint64_t),
  ('outputs_read', uint64_t),
  ('system_values_read', (ctypes.c_uint32 * 4)),
  ('per_primitive_inputs', uint64_t),
  ('per_primitive_outputs', uint64_t),
  ('per_view_outputs', uint64_t),
  ('view_mask', uint32_t),
  ('inputs_read_16bit', uint16_t),
  ('outputs_written_16bit', uint16_t),
  ('outputs_read_16bit', uint16_t),
  ('inputs_read_indirectly_16bit', uint16_t),
  ('outputs_read_indirectly_16bit', uint16_t),
  ('outputs_written_indirectly_16bit', uint16_t),
  ('patch_inputs_read', uint32_t),
  ('patch_outputs_written', uint32_t),
  ('patch_outputs_read', uint32_t),
  ('inputs_read_indirectly', uint64_t),
  ('outputs_read_indirectly', uint64_t),
  ('outputs_written_indirectly', uint64_t),
  ('patch_inputs_read_indirectly', uint32_t),
  ('patch_outputs_read_indirectly', uint32_t),
  ('patch_outputs_written_indirectly', uint32_t),
  ('textures_used', (ctypes.c_uint32 * 4)),
  ('textures_used_by_txf', (ctypes.c_uint32 * 4)),
  ('samplers_used', (ctypes.c_uint32 * 1)),
  ('images_used', (ctypes.c_uint32 * 2)),
  ('image_buffers', (ctypes.c_uint32 * 2)),
  ('msaa_images', (ctypes.c_uint32 * 2)),
  ('float_controls_execution_mode', uint32_t),
  ('shared_size', ctypes.c_uint32),
  ('task_payload_size', ctypes.c_uint32),
  ('ray_queries', ctypes.c_uint32),
  ('workgroup_size', (uint16_t * 3)),
  ('subgroup_size', enum_gl_subgroup_size),
  ('num_subgroups', uint8_t),
  ('uses_wide_subgroup_intrinsics', ctypes.c_bool),
  ('xfb_stride', (uint8_t * 4)),
  ('inlinable_uniform_dw_offsets', (uint16_t * 4)),
  ('num_inlinable_uniforms', uint8_t,4),
  ('clip_distance_array_size', uint8_t,4),
  ('cull_distance_array_size', uint8_t,4),
  ('uses_texture_gather', ctypes.c_bool,1),
  ('uses_resource_info_query', ctypes.c_bool,1),
  ('bit_sizes_float', uint8_t),
  ('bit_sizes_int', uint8_t),
  ('first_ubo_is_default_ubo', ctypes.c_bool,1),
  ('separate_shader', ctypes.c_bool,1),
  ('has_transform_feedback_varyings', ctypes.c_bool,1),
  ('flrp_lowered', ctypes.c_bool,1),
  ('io_lowered', ctypes.c_bool,1),
  ('var_copies_lowered', ctypes.c_bool,1),
  ('writes_memory', ctypes.c_bool,1),
  ('layer_viewport_relative', ctypes.c_bool,1),
  ('uses_control_barrier', ctypes.c_bool,1),
  ('uses_memory_barrier', ctypes.c_bool,1),
  ('uses_bindless', ctypes.c_bool,1),
  ('shared_memory_explicit_layout', ctypes.c_bool,1),
  ('zero_initialize_shared_memory', ctypes.c_bool,1),
  ('workgroup_size_variable', ctypes.c_bool,1),
  ('uses_printf', ctypes.c_bool,1),
  ('maximally_reconverges', ctypes.c_bool,1),
  ('use_aco_amd', ctypes.c_bool,1),
  ('use_lowered_image_to_global', ctypes.c_bool,1),
  ('use_legacy_math_rules', ctypes.c_bool),
  ('derivative_group', enum_gl_derivative_group,2),
  ('_0', struct_shader_info_0),
]
class struct_nir_xfb_info(Struct): pass
nir_xfb_info = struct_nir_xfb_info
struct_nir_shader._fields_ = [
  ('gctx', ctypes.POINTER(gc_ctx)),
  ('variables', struct_exec_list),
  ('options', ctypes.POINTER(nir_shader_compiler_options)),
  ('info', struct_shader_info),
  ('functions', struct_exec_list),
  ('num_inputs', ctypes.c_uint32),
  ('num_uniforms', ctypes.c_uint32),
  ('num_outputs', ctypes.c_uint32),
  ('global_mem_size', ctypes.c_uint32),
  ('scratch_size', ctypes.c_uint32),
  ('constant_data', ctypes.c_void_p),
  ('constant_data_size', ctypes.c_uint32),
  ('xfb_info', ctypes.POINTER(nir_xfb_info)),
  ('printf_info_count', ctypes.c_uint32),
  ('printf_info', ctypes.POINTER(u_printf_info)),
  ('has_debug_info', ctypes.c_bool),
]
class struct_nir_parameter(Struct): pass
nir_parameter = struct_nir_parameter
struct_nir_parameter._fields_ = [
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('is_return', ctypes.c_bool),
  ('implicit_conversion_prohibited', ctypes.c_bool),
  ('is_uniform', ctypes.c_bool),
  ('mode', nir_variable_mode),
  ('driver_attributes', uint32_t),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
]
class struct_nir_function_impl(Struct): pass
nir_function_impl = struct_nir_function_impl
nir_function = struct_nir_function
nir_metadata = CEnum(ctypes.c_int32)
nir_metadata_none = nir_metadata.define('nir_metadata_none', 0)
nir_metadata_block_index = nir_metadata.define('nir_metadata_block_index', 1)
nir_metadata_dominance = nir_metadata.define('nir_metadata_dominance', 2)
nir_metadata_live_defs = nir_metadata.define('nir_metadata_live_defs', 4)
nir_metadata_not_properly_reset = nir_metadata.define('nir_metadata_not_properly_reset', 8)
nir_metadata_loop_analysis = nir_metadata.define('nir_metadata_loop_analysis', 16)
nir_metadata_instr_index = nir_metadata.define('nir_metadata_instr_index', 32)
nir_metadata_divergence = nir_metadata.define('nir_metadata_divergence', 64)
nir_metadata_control_flow = nir_metadata.define('nir_metadata_control_flow', 3)
nir_metadata_all = nir_metadata.define('nir_metadata_all', -9)

struct_nir_function_impl._fields_ = [
  ('cf_node', nir_cf_node),
  ('function', ctypes.POINTER(nir_function)),
  ('preamble', ctypes.POINTER(nir_function)),
  ('body', struct_exec_list),
  ('end_block', ctypes.POINTER(nir_block)),
  ('locals', struct_exec_list),
  ('ssa_alloc', ctypes.c_uint32),
  ('num_blocks', ctypes.c_uint32),
  ('structured', ctypes.c_bool),
  ('valid_metadata', nir_metadata),
  ('loop_analysis_indirect_mask', nir_variable_mode),
  ('loop_analysis_force_unroll_sampler_indirect', ctypes.c_bool),
]
struct_nir_function._fields_ = [
  ('node', struct_exec_node),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('shader', ctypes.POINTER(nir_shader)),
  ('num_params', ctypes.c_uint32),
  ('params', ctypes.POINTER(nir_parameter)),
  ('impl', ctypes.POINTER(nir_function_impl)),
  ('driver_attributes', uint32_t),
  ('is_entrypoint', ctypes.c_bool),
  ('is_exported', ctypes.c_bool),
  ('is_preamble', ctypes.c_bool),
  ('should_inline', ctypes.c_bool),
  ('dont_inline', ctypes.c_bool),
  ('workgroup_size', (ctypes.c_uint32 * 3)),
  ('is_subroutine', ctypes.c_bool),
  ('is_tmp_globals_wrapper', ctypes.c_bool),
  ('num_subroutine_types', ctypes.c_int32),
  ('subroutine_types', ctypes.POINTER(ctypes.POINTER(struct_glsl_type))),
  ('subroutine_index', ctypes.c_int32),
  ('pass_flags', uint32_t),
]
struct_nir_call_instr._fields_ = [
  ('instr', nir_instr),
  ('callee', ctypes.POINTER(nir_function)),
  ('indirect_callee', nir_src),
  ('num_params', ctypes.c_uint32),
  ('params', (nir_src * 0)),
]
nir_call_instr = struct_nir_call_instr
class struct_nir_intrinsic_instr(Struct): pass
nir_intrinsic_op = CEnum(ctypes.c_uint32)
nir_intrinsic_accept_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_accept_ray_intersection', 0)
nir_intrinsic_addr_mode_is = nir_intrinsic_op.define('nir_intrinsic_addr_mode_is', 1)
nir_intrinsic_al2p_nv = nir_intrinsic_op.define('nir_intrinsic_al2p_nv', 2)
nir_intrinsic_ald_nv = nir_intrinsic_op.define('nir_intrinsic_ald_nv', 3)
nir_intrinsic_alpha_to_coverage = nir_intrinsic_op.define('nir_intrinsic_alpha_to_coverage', 4)
nir_intrinsic_as_uniform = nir_intrinsic_op.define('nir_intrinsic_as_uniform', 5)
nir_intrinsic_ast_nv = nir_intrinsic_op.define('nir_intrinsic_ast_nv', 6)
nir_intrinsic_atomic_add_gen_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_gen_prim_count_amd', 7)
nir_intrinsic_atomic_add_gs_emit_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_gs_emit_prim_count_amd', 8)
nir_intrinsic_atomic_add_shader_invocation_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_shader_invocation_count_amd', 9)
nir_intrinsic_atomic_add_xfb_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_xfb_prim_count_amd', 10)
nir_intrinsic_atomic_counter_add = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_add', 11)
nir_intrinsic_atomic_counter_add_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_add_deref', 12)
nir_intrinsic_atomic_counter_and = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_and', 13)
nir_intrinsic_atomic_counter_and_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_and_deref', 14)
nir_intrinsic_atomic_counter_comp_swap = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_comp_swap', 15)
nir_intrinsic_atomic_counter_comp_swap_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_comp_swap_deref', 16)
nir_intrinsic_atomic_counter_exchange = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_exchange', 17)
nir_intrinsic_atomic_counter_exchange_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_exchange_deref', 18)
nir_intrinsic_atomic_counter_inc = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_inc', 19)
nir_intrinsic_atomic_counter_inc_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_inc_deref', 20)
nir_intrinsic_atomic_counter_max = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_max', 21)
nir_intrinsic_atomic_counter_max_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_max_deref', 22)
nir_intrinsic_atomic_counter_min = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_min', 23)
nir_intrinsic_atomic_counter_min_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_min_deref', 24)
nir_intrinsic_atomic_counter_or = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_or', 25)
nir_intrinsic_atomic_counter_or_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_or_deref', 26)
nir_intrinsic_atomic_counter_post_dec = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_post_dec', 27)
nir_intrinsic_atomic_counter_post_dec_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_post_dec_deref', 28)
nir_intrinsic_atomic_counter_pre_dec = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_pre_dec', 29)
nir_intrinsic_atomic_counter_pre_dec_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_pre_dec_deref', 30)
nir_intrinsic_atomic_counter_read = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_read', 31)
nir_intrinsic_atomic_counter_read_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_read_deref', 32)
nir_intrinsic_atomic_counter_xor = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_xor', 33)
nir_intrinsic_atomic_counter_xor_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_xor_deref', 34)
nir_intrinsic_ballot = nir_intrinsic_op.define('nir_intrinsic_ballot', 35)
nir_intrinsic_ballot_bit_count_exclusive = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_exclusive', 36)
nir_intrinsic_ballot_bit_count_inclusive = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_inclusive', 37)
nir_intrinsic_ballot_bit_count_reduce = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_reduce', 38)
nir_intrinsic_ballot_bitfield_extract = nir_intrinsic_op.define('nir_intrinsic_ballot_bitfield_extract', 39)
nir_intrinsic_ballot_find_lsb = nir_intrinsic_op.define('nir_intrinsic_ballot_find_lsb', 40)
nir_intrinsic_ballot_find_msb = nir_intrinsic_op.define('nir_intrinsic_ballot_find_msb', 41)
nir_intrinsic_ballot_relaxed = nir_intrinsic_op.define('nir_intrinsic_ballot_relaxed', 42)
nir_intrinsic_bar_break_nv = nir_intrinsic_op.define('nir_intrinsic_bar_break_nv', 43)
nir_intrinsic_bar_set_nv = nir_intrinsic_op.define('nir_intrinsic_bar_set_nv', 44)
nir_intrinsic_bar_sync_nv = nir_intrinsic_op.define('nir_intrinsic_bar_sync_nv', 45)
nir_intrinsic_barrier = nir_intrinsic_op.define('nir_intrinsic_barrier', 46)
nir_intrinsic_begin_invocation_interlock = nir_intrinsic_op.define('nir_intrinsic_begin_invocation_interlock', 47)
nir_intrinsic_bindgen_return = nir_intrinsic_op.define('nir_intrinsic_bindgen_return', 48)
nir_intrinsic_bindless_image_agx = nir_intrinsic_op.define('nir_intrinsic_bindless_image_agx', 49)
nir_intrinsic_bindless_image_atomic = nir_intrinsic_op.define('nir_intrinsic_bindless_image_atomic', 50)
nir_intrinsic_bindless_image_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_bindless_image_atomic_swap', 51)
nir_intrinsic_bindless_image_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_bindless_image_descriptor_amd', 52)
nir_intrinsic_bindless_image_format = nir_intrinsic_op.define('nir_intrinsic_bindless_image_format', 53)
nir_intrinsic_bindless_image_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_bindless_image_fragment_mask_load_amd', 54)
nir_intrinsic_bindless_image_levels = nir_intrinsic_op.define('nir_intrinsic_bindless_image_levels', 55)
nir_intrinsic_bindless_image_load = nir_intrinsic_op.define('nir_intrinsic_bindless_image_load', 56)
nir_intrinsic_bindless_image_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_bindless_image_load_raw_intel', 57)
nir_intrinsic_bindless_image_order = nir_intrinsic_op.define('nir_intrinsic_bindless_image_order', 58)
nir_intrinsic_bindless_image_samples = nir_intrinsic_op.define('nir_intrinsic_bindless_image_samples', 59)
nir_intrinsic_bindless_image_samples_identical = nir_intrinsic_op.define('nir_intrinsic_bindless_image_samples_identical', 60)
nir_intrinsic_bindless_image_size = nir_intrinsic_op.define('nir_intrinsic_bindless_image_size', 61)
nir_intrinsic_bindless_image_sparse_load = nir_intrinsic_op.define('nir_intrinsic_bindless_image_sparse_load', 62)
nir_intrinsic_bindless_image_store = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store', 63)
nir_intrinsic_bindless_image_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store_block_agx', 64)
nir_intrinsic_bindless_image_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store_raw_intel', 65)
nir_intrinsic_bindless_image_texel_address = nir_intrinsic_op.define('nir_intrinsic_bindless_image_texel_address', 66)
nir_intrinsic_bindless_resource_ir3 = nir_intrinsic_op.define('nir_intrinsic_bindless_resource_ir3', 67)
nir_intrinsic_brcst_active_ir3 = nir_intrinsic_op.define('nir_intrinsic_brcst_active_ir3', 68)
nir_intrinsic_btd_retire_intel = nir_intrinsic_op.define('nir_intrinsic_btd_retire_intel', 69)
nir_intrinsic_btd_spawn_intel = nir_intrinsic_op.define('nir_intrinsic_btd_spawn_intel', 70)
nir_intrinsic_btd_stack_push_intel = nir_intrinsic_op.define('nir_intrinsic_btd_stack_push_intel', 71)
nir_intrinsic_bvh64_intersect_ray_amd = nir_intrinsic_op.define('nir_intrinsic_bvh64_intersect_ray_amd', 72)
nir_intrinsic_bvh8_intersect_ray_amd = nir_intrinsic_op.define('nir_intrinsic_bvh8_intersect_ray_amd', 73)
nir_intrinsic_bvh_stack_rtn_amd = nir_intrinsic_op.define('nir_intrinsic_bvh_stack_rtn_amd', 74)
nir_intrinsic_cmat_binary_op = nir_intrinsic_op.define('nir_intrinsic_cmat_binary_op', 75)
nir_intrinsic_cmat_bitcast = nir_intrinsic_op.define('nir_intrinsic_cmat_bitcast', 76)
nir_intrinsic_cmat_construct = nir_intrinsic_op.define('nir_intrinsic_cmat_construct', 77)
nir_intrinsic_cmat_convert = nir_intrinsic_op.define('nir_intrinsic_cmat_convert', 78)
nir_intrinsic_cmat_copy = nir_intrinsic_op.define('nir_intrinsic_cmat_copy', 79)
nir_intrinsic_cmat_extract = nir_intrinsic_op.define('nir_intrinsic_cmat_extract', 80)
nir_intrinsic_cmat_insert = nir_intrinsic_op.define('nir_intrinsic_cmat_insert', 81)
nir_intrinsic_cmat_length = nir_intrinsic_op.define('nir_intrinsic_cmat_length', 82)
nir_intrinsic_cmat_load = nir_intrinsic_op.define('nir_intrinsic_cmat_load', 83)
nir_intrinsic_cmat_muladd = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd', 84)
nir_intrinsic_cmat_muladd_amd = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd_amd', 85)
nir_intrinsic_cmat_muladd_nv = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd_nv', 86)
nir_intrinsic_cmat_scalar_op = nir_intrinsic_op.define('nir_intrinsic_cmat_scalar_op', 87)
nir_intrinsic_cmat_store = nir_intrinsic_op.define('nir_intrinsic_cmat_store', 88)
nir_intrinsic_cmat_transpose = nir_intrinsic_op.define('nir_intrinsic_cmat_transpose', 89)
nir_intrinsic_cmat_unary_op = nir_intrinsic_op.define('nir_intrinsic_cmat_unary_op', 90)
nir_intrinsic_convert_alu_types = nir_intrinsic_op.define('nir_intrinsic_convert_alu_types', 91)
nir_intrinsic_convert_cmat_intel = nir_intrinsic_op.define('nir_intrinsic_convert_cmat_intel', 92)
nir_intrinsic_copy_deref = nir_intrinsic_op.define('nir_intrinsic_copy_deref', 93)
nir_intrinsic_copy_fs_outputs_nv = nir_intrinsic_op.define('nir_intrinsic_copy_fs_outputs_nv', 94)
nir_intrinsic_copy_global_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_global_to_uniform_ir3', 95)
nir_intrinsic_copy_push_const_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_push_const_to_uniform_ir3', 96)
nir_intrinsic_copy_ubo_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_ubo_to_uniform_ir3', 97)
nir_intrinsic_ddx = nir_intrinsic_op.define('nir_intrinsic_ddx', 98)
nir_intrinsic_ddx_coarse = nir_intrinsic_op.define('nir_intrinsic_ddx_coarse', 99)
nir_intrinsic_ddx_fine = nir_intrinsic_op.define('nir_intrinsic_ddx_fine', 100)
nir_intrinsic_ddy = nir_intrinsic_op.define('nir_intrinsic_ddy', 101)
nir_intrinsic_ddy_coarse = nir_intrinsic_op.define('nir_intrinsic_ddy_coarse', 102)
nir_intrinsic_ddy_fine = nir_intrinsic_op.define('nir_intrinsic_ddy_fine', 103)
nir_intrinsic_debug_break = nir_intrinsic_op.define('nir_intrinsic_debug_break', 104)
nir_intrinsic_decl_reg = nir_intrinsic_op.define('nir_intrinsic_decl_reg', 105)
nir_intrinsic_demote = nir_intrinsic_op.define('nir_intrinsic_demote', 106)
nir_intrinsic_demote_if = nir_intrinsic_op.define('nir_intrinsic_demote_if', 107)
nir_intrinsic_demote_samples = nir_intrinsic_op.define('nir_intrinsic_demote_samples', 108)
nir_intrinsic_deref_atomic = nir_intrinsic_op.define('nir_intrinsic_deref_atomic', 109)
nir_intrinsic_deref_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_deref_atomic_swap', 110)
nir_intrinsic_deref_buffer_array_length = nir_intrinsic_op.define('nir_intrinsic_deref_buffer_array_length', 111)
nir_intrinsic_deref_implicit_array_length = nir_intrinsic_op.define('nir_intrinsic_deref_implicit_array_length', 112)
nir_intrinsic_deref_mode_is = nir_intrinsic_op.define('nir_intrinsic_deref_mode_is', 113)
nir_intrinsic_deref_texture_src = nir_intrinsic_op.define('nir_intrinsic_deref_texture_src', 114)
nir_intrinsic_doorbell_agx = nir_intrinsic_op.define('nir_intrinsic_doorbell_agx', 115)
nir_intrinsic_dpas_intel = nir_intrinsic_op.define('nir_intrinsic_dpas_intel', 116)
nir_intrinsic_dpp16_shift_amd = nir_intrinsic_op.define('nir_intrinsic_dpp16_shift_amd', 117)
nir_intrinsic_elect = nir_intrinsic_op.define('nir_intrinsic_elect', 118)
nir_intrinsic_elect_any_ir3 = nir_intrinsic_op.define('nir_intrinsic_elect_any_ir3', 119)
nir_intrinsic_emit_primitive_poly = nir_intrinsic_op.define('nir_intrinsic_emit_primitive_poly', 120)
nir_intrinsic_emit_vertex = nir_intrinsic_op.define('nir_intrinsic_emit_vertex', 121)
nir_intrinsic_emit_vertex_nv = nir_intrinsic_op.define('nir_intrinsic_emit_vertex_nv', 122)
nir_intrinsic_emit_vertex_with_counter = nir_intrinsic_op.define('nir_intrinsic_emit_vertex_with_counter', 123)
nir_intrinsic_end_invocation_interlock = nir_intrinsic_op.define('nir_intrinsic_end_invocation_interlock', 124)
nir_intrinsic_end_primitive = nir_intrinsic_op.define('nir_intrinsic_end_primitive', 125)
nir_intrinsic_end_primitive_nv = nir_intrinsic_op.define('nir_intrinsic_end_primitive_nv', 126)
nir_intrinsic_end_primitive_with_counter = nir_intrinsic_op.define('nir_intrinsic_end_primitive_with_counter', 127)
nir_intrinsic_enqueue_node_payloads = nir_intrinsic_op.define('nir_intrinsic_enqueue_node_payloads', 128)
nir_intrinsic_exclusive_scan = nir_intrinsic_op.define('nir_intrinsic_exclusive_scan', 129)
nir_intrinsic_exclusive_scan_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_exclusive_scan_clusters_ir3', 130)
nir_intrinsic_execute_callable = nir_intrinsic_op.define('nir_intrinsic_execute_callable', 131)
nir_intrinsic_execute_closest_hit_amd = nir_intrinsic_op.define('nir_intrinsic_execute_closest_hit_amd', 132)
nir_intrinsic_execute_miss_amd = nir_intrinsic_op.define('nir_intrinsic_execute_miss_amd', 133)
nir_intrinsic_export_agx = nir_intrinsic_op.define('nir_intrinsic_export_agx', 134)
nir_intrinsic_export_amd = nir_intrinsic_op.define('nir_intrinsic_export_amd', 135)
nir_intrinsic_export_dual_src_blend_amd = nir_intrinsic_op.define('nir_intrinsic_export_dual_src_blend_amd', 136)
nir_intrinsic_export_row_amd = nir_intrinsic_op.define('nir_intrinsic_export_row_amd', 137)
nir_intrinsic_fence_helper_exit_agx = nir_intrinsic_op.define('nir_intrinsic_fence_helper_exit_agx', 138)
nir_intrinsic_fence_mem_to_tex_agx = nir_intrinsic_op.define('nir_intrinsic_fence_mem_to_tex_agx', 139)
nir_intrinsic_fence_pbe_to_tex_agx = nir_intrinsic_op.define('nir_intrinsic_fence_pbe_to_tex_agx', 140)
nir_intrinsic_fence_pbe_to_tex_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_fence_pbe_to_tex_pixel_agx', 141)
nir_intrinsic_final_primitive_nv = nir_intrinsic_op.define('nir_intrinsic_final_primitive_nv', 142)
nir_intrinsic_finalize_incoming_node_payload = nir_intrinsic_op.define('nir_intrinsic_finalize_incoming_node_payload', 143)
nir_intrinsic_first_invocation = nir_intrinsic_op.define('nir_intrinsic_first_invocation', 144)
nir_intrinsic_fs_out_nv = nir_intrinsic_op.define('nir_intrinsic_fs_out_nv', 145)
nir_intrinsic_gds_atomic_add_amd = nir_intrinsic_op.define('nir_intrinsic_gds_atomic_add_amd', 146)
nir_intrinsic_get_ssbo_size = nir_intrinsic_op.define('nir_intrinsic_get_ssbo_size', 147)
nir_intrinsic_get_ubo_size = nir_intrinsic_op.define('nir_intrinsic_get_ubo_size', 148)
nir_intrinsic_global_atomic = nir_intrinsic_op.define('nir_intrinsic_global_atomic', 149)
nir_intrinsic_global_atomic_2x32 = nir_intrinsic_op.define('nir_intrinsic_global_atomic_2x32', 150)
nir_intrinsic_global_atomic_agx = nir_intrinsic_op.define('nir_intrinsic_global_atomic_agx', 151)
nir_intrinsic_global_atomic_amd = nir_intrinsic_op.define('nir_intrinsic_global_atomic_amd', 152)
nir_intrinsic_global_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap', 153)
nir_intrinsic_global_atomic_swap_2x32 = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_2x32', 154)
nir_intrinsic_global_atomic_swap_agx = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_agx', 155)
nir_intrinsic_global_atomic_swap_amd = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_amd', 156)
nir_intrinsic_ignore_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_ignore_ray_intersection', 157)
nir_intrinsic_imadsp_nv = nir_intrinsic_op.define('nir_intrinsic_imadsp_nv', 158)
nir_intrinsic_image_atomic = nir_intrinsic_op.define('nir_intrinsic_image_atomic', 159)
nir_intrinsic_image_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_image_atomic_swap', 160)
nir_intrinsic_image_deref_atomic = nir_intrinsic_op.define('nir_intrinsic_image_deref_atomic', 161)
nir_intrinsic_image_deref_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_image_deref_atomic_swap', 162)
nir_intrinsic_image_deref_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_image_deref_descriptor_amd', 163)
nir_intrinsic_image_deref_format = nir_intrinsic_op.define('nir_intrinsic_image_deref_format', 164)
nir_intrinsic_image_deref_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_image_deref_fragment_mask_load_amd', 165)
nir_intrinsic_image_deref_levels = nir_intrinsic_op.define('nir_intrinsic_image_deref_levels', 166)
nir_intrinsic_image_deref_load = nir_intrinsic_op.define('nir_intrinsic_image_deref_load', 167)
nir_intrinsic_image_deref_load_info_nv = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_info_nv', 168)
nir_intrinsic_image_deref_load_param_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_param_intel', 169)
nir_intrinsic_image_deref_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_raw_intel', 170)
nir_intrinsic_image_deref_order = nir_intrinsic_op.define('nir_intrinsic_image_deref_order', 171)
nir_intrinsic_image_deref_samples = nir_intrinsic_op.define('nir_intrinsic_image_deref_samples', 172)
nir_intrinsic_image_deref_samples_identical = nir_intrinsic_op.define('nir_intrinsic_image_deref_samples_identical', 173)
nir_intrinsic_image_deref_size = nir_intrinsic_op.define('nir_intrinsic_image_deref_size', 174)
nir_intrinsic_image_deref_sparse_load = nir_intrinsic_op.define('nir_intrinsic_image_deref_sparse_load', 175)
nir_intrinsic_image_deref_store = nir_intrinsic_op.define('nir_intrinsic_image_deref_store', 176)
nir_intrinsic_image_deref_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_image_deref_store_block_agx', 177)
nir_intrinsic_image_deref_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_store_raw_intel', 178)
nir_intrinsic_image_deref_texel_address = nir_intrinsic_op.define('nir_intrinsic_image_deref_texel_address', 179)
nir_intrinsic_image_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_image_descriptor_amd', 180)
nir_intrinsic_image_format = nir_intrinsic_op.define('nir_intrinsic_image_format', 181)
nir_intrinsic_image_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_image_fragment_mask_load_amd', 182)
nir_intrinsic_image_levels = nir_intrinsic_op.define('nir_intrinsic_image_levels', 183)
nir_intrinsic_image_load = nir_intrinsic_op.define('nir_intrinsic_image_load', 184)
nir_intrinsic_image_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_load_raw_intel', 185)
nir_intrinsic_image_order = nir_intrinsic_op.define('nir_intrinsic_image_order', 186)
nir_intrinsic_image_samples = nir_intrinsic_op.define('nir_intrinsic_image_samples', 187)
nir_intrinsic_image_samples_identical = nir_intrinsic_op.define('nir_intrinsic_image_samples_identical', 188)
nir_intrinsic_image_size = nir_intrinsic_op.define('nir_intrinsic_image_size', 189)
nir_intrinsic_image_sparse_load = nir_intrinsic_op.define('nir_intrinsic_image_sparse_load', 190)
nir_intrinsic_image_store = nir_intrinsic_op.define('nir_intrinsic_image_store', 191)
nir_intrinsic_image_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_image_store_block_agx', 192)
nir_intrinsic_image_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_store_raw_intel', 193)
nir_intrinsic_image_texel_address = nir_intrinsic_op.define('nir_intrinsic_image_texel_address', 194)
nir_intrinsic_inclusive_scan = nir_intrinsic_op.define('nir_intrinsic_inclusive_scan', 195)
nir_intrinsic_inclusive_scan_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_inclusive_scan_clusters_ir3', 196)
nir_intrinsic_initialize_node_payloads = nir_intrinsic_op.define('nir_intrinsic_initialize_node_payloads', 197)
nir_intrinsic_interp_deref_at_centroid = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_centroid', 198)
nir_intrinsic_interp_deref_at_offset = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_offset', 199)
nir_intrinsic_interp_deref_at_sample = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_sample', 200)
nir_intrinsic_interp_deref_at_vertex = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_vertex', 201)
nir_intrinsic_inverse_ballot = nir_intrinsic_op.define('nir_intrinsic_inverse_ballot', 202)
nir_intrinsic_ipa_nv = nir_intrinsic_op.define('nir_intrinsic_ipa_nv', 203)
nir_intrinsic_is_helper_invocation = nir_intrinsic_op.define('nir_intrinsic_is_helper_invocation', 204)
nir_intrinsic_is_sparse_resident_zink = nir_intrinsic_op.define('nir_intrinsic_is_sparse_resident_zink', 205)
nir_intrinsic_is_sparse_texels_resident = nir_intrinsic_op.define('nir_intrinsic_is_sparse_texels_resident', 206)
nir_intrinsic_is_subgroup_invocation_lt_amd = nir_intrinsic_op.define('nir_intrinsic_is_subgroup_invocation_lt_amd', 207)
nir_intrinsic_isberd_nv = nir_intrinsic_op.define('nir_intrinsic_isberd_nv', 208)
nir_intrinsic_lane_permute_16_amd = nir_intrinsic_op.define('nir_intrinsic_lane_permute_16_amd', 209)
nir_intrinsic_last_invocation = nir_intrinsic_op.define('nir_intrinsic_last_invocation', 210)
nir_intrinsic_launch_mesh_workgroups = nir_intrinsic_op.define('nir_intrinsic_launch_mesh_workgroups', 211)
nir_intrinsic_launch_mesh_workgroups_with_payload_deref = nir_intrinsic_op.define('nir_intrinsic_launch_mesh_workgroups_with_payload_deref', 212)
nir_intrinsic_ldc_nv = nir_intrinsic_op.define('nir_intrinsic_ldc_nv', 213)
nir_intrinsic_ldcx_nv = nir_intrinsic_op.define('nir_intrinsic_ldcx_nv', 214)
nir_intrinsic_ldtram_nv = nir_intrinsic_op.define('nir_intrinsic_ldtram_nv', 215)
nir_intrinsic_load_aa_line_width = nir_intrinsic_op.define('nir_intrinsic_load_aa_line_width', 216)
nir_intrinsic_load_accel_struct_amd = nir_intrinsic_op.define('nir_intrinsic_load_accel_struct_amd', 217)
nir_intrinsic_load_active_samples_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_samples_agx', 218)
nir_intrinsic_load_active_subgroup_count_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_subgroup_count_agx', 219)
nir_intrinsic_load_active_subgroup_invocation_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_subgroup_invocation_agx', 220)
nir_intrinsic_load_agx = nir_intrinsic_op.define('nir_intrinsic_load_agx', 221)
nir_intrinsic_load_alpha_reference_amd = nir_intrinsic_op.define('nir_intrinsic_load_alpha_reference_amd', 222)
nir_intrinsic_load_api_sample_mask_agx = nir_intrinsic_op.define('nir_intrinsic_load_api_sample_mask_agx', 223)
nir_intrinsic_load_attrib_clamp_agx = nir_intrinsic_op.define('nir_intrinsic_load_attrib_clamp_agx', 224)
nir_intrinsic_load_attribute_pan = nir_intrinsic_op.define('nir_intrinsic_load_attribute_pan', 225)
nir_intrinsic_load_back_face_agx = nir_intrinsic_op.define('nir_intrinsic_load_back_face_agx', 226)
nir_intrinsic_load_barycentric_at_offset = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_offset', 227)
nir_intrinsic_load_barycentric_at_offset_nv = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_offset_nv', 228)
nir_intrinsic_load_barycentric_at_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_sample', 229)
nir_intrinsic_load_barycentric_centroid = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_centroid', 230)
nir_intrinsic_load_barycentric_coord_at_offset = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_at_offset', 231)
nir_intrinsic_load_barycentric_coord_at_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_at_sample', 232)
nir_intrinsic_load_barycentric_coord_centroid = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_centroid', 233)
nir_intrinsic_load_barycentric_coord_pixel = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_pixel', 234)
nir_intrinsic_load_barycentric_coord_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_sample', 235)
nir_intrinsic_load_barycentric_model = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_model', 236)
nir_intrinsic_load_barycentric_optimize_amd = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_optimize_amd', 237)
nir_intrinsic_load_barycentric_pixel = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_pixel', 238)
nir_intrinsic_load_barycentric_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_sample', 239)
nir_intrinsic_load_base_global_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_base_global_invocation_id', 240)
nir_intrinsic_load_base_instance = nir_intrinsic_op.define('nir_intrinsic_load_base_instance', 241)
nir_intrinsic_load_base_vertex = nir_intrinsic_op.define('nir_intrinsic_load_base_vertex', 242)
nir_intrinsic_load_base_workgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_base_workgroup_id', 243)
nir_intrinsic_load_blend_const_color_a_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_a_float', 244)
nir_intrinsic_load_blend_const_color_aaaa8888_unorm = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_aaaa8888_unorm', 245)
nir_intrinsic_load_blend_const_color_b_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_b_float', 246)
nir_intrinsic_load_blend_const_color_g_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_g_float', 247)
nir_intrinsic_load_blend_const_color_r_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_r_float', 248)
nir_intrinsic_load_blend_const_color_rgba = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_rgba', 249)
nir_intrinsic_load_blend_const_color_rgba8888_unorm = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_rgba8888_unorm', 250)
nir_intrinsic_load_btd_global_arg_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_global_arg_addr_intel', 251)
nir_intrinsic_load_btd_local_arg_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_local_arg_addr_intel', 252)
nir_intrinsic_load_btd_resume_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_resume_sbt_addr_intel', 253)
nir_intrinsic_load_btd_shader_type_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_shader_type_intel', 254)
nir_intrinsic_load_btd_stack_id_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_stack_id_intel', 255)
nir_intrinsic_load_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_buffer_amd', 256)
nir_intrinsic_load_callable_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_callable_sbt_addr_intel', 257)
nir_intrinsic_load_callable_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_callable_sbt_stride_intel', 258)
nir_intrinsic_load_clamp_vertex_color_amd = nir_intrinsic_op.define('nir_intrinsic_load_clamp_vertex_color_amd', 259)
nir_intrinsic_load_clip_half_line_width_amd = nir_intrinsic_op.define('nir_intrinsic_load_clip_half_line_width_amd', 260)
nir_intrinsic_load_clip_z_coeff_agx = nir_intrinsic_op.define('nir_intrinsic_load_clip_z_coeff_agx', 261)
nir_intrinsic_load_coalesced_input_count = nir_intrinsic_op.define('nir_intrinsic_load_coalesced_input_count', 262)
nir_intrinsic_load_coefficients_agx = nir_intrinsic_op.define('nir_intrinsic_load_coefficients_agx', 263)
nir_intrinsic_load_color0 = nir_intrinsic_op.define('nir_intrinsic_load_color0', 264)
nir_intrinsic_load_color1 = nir_intrinsic_op.define('nir_intrinsic_load_color1', 265)
nir_intrinsic_load_const_buf_base_addr_lvp = nir_intrinsic_op.define('nir_intrinsic_load_const_buf_base_addr_lvp', 266)
nir_intrinsic_load_const_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_const_ir3', 267)
nir_intrinsic_load_constant = nir_intrinsic_op.define('nir_intrinsic_load_constant', 268)
nir_intrinsic_load_constant_agx = nir_intrinsic_op.define('nir_intrinsic_load_constant_agx', 269)
nir_intrinsic_load_constant_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_constant_base_ptr', 270)
nir_intrinsic_load_converted_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_converted_output_pan', 271)
nir_intrinsic_load_core_id_agx = nir_intrinsic_op.define('nir_intrinsic_load_core_id_agx', 272)
nir_intrinsic_load_cull_any_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_any_enabled_amd', 273)
nir_intrinsic_load_cull_back_face_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_back_face_enabled_amd', 274)
nir_intrinsic_load_cull_ccw_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_ccw_amd', 275)
nir_intrinsic_load_cull_front_face_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_front_face_enabled_amd', 276)
nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd', 277)
nir_intrinsic_load_cull_mask = nir_intrinsic_op.define('nir_intrinsic_load_cull_mask', 278)
nir_intrinsic_load_cull_mask_and_flags_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_mask_and_flags_amd', 279)
nir_intrinsic_load_cull_small_line_precision_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_line_precision_amd', 280)
nir_intrinsic_load_cull_small_lines_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_lines_enabled_amd', 281)
nir_intrinsic_load_cull_small_triangle_precision_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_triangle_precision_amd', 282)
nir_intrinsic_load_cull_small_triangles_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_triangles_enabled_amd', 283)
nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd', 284)
nir_intrinsic_load_debug_log_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_debug_log_desc_amd', 285)
nir_intrinsic_load_depth_never_agx = nir_intrinsic_op.define('nir_intrinsic_load_depth_never_agx', 286)
nir_intrinsic_load_deref = nir_intrinsic_op.define('nir_intrinsic_load_deref', 287)
nir_intrinsic_load_deref_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_deref_block_intel', 288)
nir_intrinsic_load_draw_id = nir_intrinsic_op.define('nir_intrinsic_load_draw_id', 289)
nir_intrinsic_load_esgs_vertex_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_esgs_vertex_stride_amd', 290)
nir_intrinsic_load_exported_agx = nir_intrinsic_op.define('nir_intrinsic_load_exported_agx', 291)
nir_intrinsic_load_fb_layers_v3d = nir_intrinsic_op.define('nir_intrinsic_load_fb_layers_v3d', 292)
nir_intrinsic_load_fbfetch_image_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_fbfetch_image_desc_amd', 293)
nir_intrinsic_load_fbfetch_image_fmask_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_fbfetch_image_fmask_desc_amd', 294)
nir_intrinsic_load_fep_w_v3d = nir_intrinsic_op.define('nir_intrinsic_load_fep_w_v3d', 295)
nir_intrinsic_load_first_vertex = nir_intrinsic_op.define('nir_intrinsic_load_first_vertex', 296)
nir_intrinsic_load_fixed_point_size_agx = nir_intrinsic_op.define('nir_intrinsic_load_fixed_point_size_agx', 297)
nir_intrinsic_load_flat_mask = nir_intrinsic_op.define('nir_intrinsic_load_flat_mask', 298)
nir_intrinsic_load_force_vrs_rates_amd = nir_intrinsic_op.define('nir_intrinsic_load_force_vrs_rates_amd', 299)
nir_intrinsic_load_frag_coord = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord', 300)
nir_intrinsic_load_frag_coord_unscaled_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_unscaled_ir3', 301)
nir_intrinsic_load_frag_coord_w = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_w', 302)
nir_intrinsic_load_frag_coord_z = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_z', 303)
nir_intrinsic_load_frag_coord_zw_pan = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_zw_pan', 304)
nir_intrinsic_load_frag_invocation_count = nir_intrinsic_op.define('nir_intrinsic_load_frag_invocation_count', 305)
nir_intrinsic_load_frag_offset_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_offset_ir3', 306)
nir_intrinsic_load_frag_shading_rate = nir_intrinsic_op.define('nir_intrinsic_load_frag_shading_rate', 307)
nir_intrinsic_load_frag_size = nir_intrinsic_op.define('nir_intrinsic_load_frag_size', 308)
nir_intrinsic_load_frag_size_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_size_ir3', 309)
nir_intrinsic_load_from_texture_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_from_texture_handle_agx', 310)
nir_intrinsic_load_front_face = nir_intrinsic_op.define('nir_intrinsic_load_front_face', 311)
nir_intrinsic_load_front_face_fsign = nir_intrinsic_op.define('nir_intrinsic_load_front_face_fsign', 312)
nir_intrinsic_load_fs_input_interp_deltas = nir_intrinsic_op.define('nir_intrinsic_load_fs_input_interp_deltas', 313)
nir_intrinsic_load_fs_msaa_intel = nir_intrinsic_op.define('nir_intrinsic_load_fs_msaa_intel', 314)
nir_intrinsic_load_fully_covered = nir_intrinsic_op.define('nir_intrinsic_load_fully_covered', 315)
nir_intrinsic_load_geometry_param_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_geometry_param_buffer_poly', 316)
nir_intrinsic_load_global = nir_intrinsic_op.define('nir_intrinsic_load_global', 317)
nir_intrinsic_load_global_2x32 = nir_intrinsic_op.define('nir_intrinsic_load_global_2x32', 318)
nir_intrinsic_load_global_amd = nir_intrinsic_op.define('nir_intrinsic_load_global_amd', 319)
nir_intrinsic_load_global_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_global_base_ptr', 320)
nir_intrinsic_load_global_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_global_block_intel', 321)
nir_intrinsic_load_global_bounded = nir_intrinsic_op.define('nir_intrinsic_load_global_bounded', 322)
nir_intrinsic_load_global_constant = nir_intrinsic_op.define('nir_intrinsic_load_global_constant', 323)
nir_intrinsic_load_global_constant_bounded = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_bounded', 324)
nir_intrinsic_load_global_constant_offset = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_offset', 325)
nir_intrinsic_load_global_constant_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_uniform_block_intel', 326)
nir_intrinsic_load_global_etna = nir_intrinsic_op.define('nir_intrinsic_load_global_etna', 327)
nir_intrinsic_load_global_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_global_invocation_id', 328)
nir_intrinsic_load_global_invocation_index = nir_intrinsic_op.define('nir_intrinsic_load_global_invocation_index', 329)
nir_intrinsic_load_global_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_global_ir3', 330)
nir_intrinsic_load_global_size = nir_intrinsic_op.define('nir_intrinsic_load_global_size', 331)
nir_intrinsic_load_gs_header_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_gs_header_ir3', 332)
nir_intrinsic_load_gs_vertex_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_gs_vertex_offset_amd', 333)
nir_intrinsic_load_gs_wave_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_gs_wave_id_amd', 334)
nir_intrinsic_load_helper_arg_hi_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_arg_hi_agx', 335)
nir_intrinsic_load_helper_arg_lo_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_arg_lo_agx', 336)
nir_intrinsic_load_helper_invocation = nir_intrinsic_op.define('nir_intrinsic_load_helper_invocation', 337)
nir_intrinsic_load_helper_op_id_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_op_id_agx', 338)
nir_intrinsic_load_hit_attrib_amd = nir_intrinsic_op.define('nir_intrinsic_load_hit_attrib_amd', 339)
nir_intrinsic_load_hs_out_patch_data_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_hs_out_patch_data_offset_amd', 340)
nir_intrinsic_load_hs_patch_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_hs_patch_stride_ir3', 341)
nir_intrinsic_load_initial_edgeflags_amd = nir_intrinsic_op.define('nir_intrinsic_load_initial_edgeflags_amd', 342)
nir_intrinsic_load_inline_data_intel = nir_intrinsic_op.define('nir_intrinsic_load_inline_data_intel', 343)
nir_intrinsic_load_input = nir_intrinsic_op.define('nir_intrinsic_load_input', 344)
nir_intrinsic_load_input_assembly_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_input_assembly_buffer_poly', 345)
nir_intrinsic_load_input_attachment_conv_pan = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_conv_pan', 346)
nir_intrinsic_load_input_attachment_coord = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_coord', 347)
nir_intrinsic_load_input_attachment_target_pan = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_target_pan', 348)
nir_intrinsic_load_input_topology_poly = nir_intrinsic_op.define('nir_intrinsic_load_input_topology_poly', 349)
nir_intrinsic_load_input_vertex = nir_intrinsic_op.define('nir_intrinsic_load_input_vertex', 350)
nir_intrinsic_load_instance_id = nir_intrinsic_op.define('nir_intrinsic_load_instance_id', 351)
nir_intrinsic_load_interpolated_input = nir_intrinsic_op.define('nir_intrinsic_load_interpolated_input', 352)
nir_intrinsic_load_intersection_opaque_amd = nir_intrinsic_op.define('nir_intrinsic_load_intersection_opaque_amd', 353)
nir_intrinsic_load_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_invocation_id', 354)
nir_intrinsic_load_is_first_fan_agx = nir_intrinsic_op.define('nir_intrinsic_load_is_first_fan_agx', 355)
nir_intrinsic_load_is_indexed_draw = nir_intrinsic_op.define('nir_intrinsic_load_is_indexed_draw', 356)
nir_intrinsic_load_kernel_input = nir_intrinsic_op.define('nir_intrinsic_load_kernel_input', 357)
nir_intrinsic_load_layer_id = nir_intrinsic_op.define('nir_intrinsic_load_layer_id', 358)
nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd = nir_intrinsic_op.define('nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd', 359)
nir_intrinsic_load_leaf_opaque_intel = nir_intrinsic_op.define('nir_intrinsic_load_leaf_opaque_intel', 360)
nir_intrinsic_load_leaf_procedural_intel = nir_intrinsic_op.define('nir_intrinsic_load_leaf_procedural_intel', 361)
nir_intrinsic_load_line_coord = nir_intrinsic_op.define('nir_intrinsic_load_line_coord', 362)
nir_intrinsic_load_line_width = nir_intrinsic_op.define('nir_intrinsic_load_line_width', 363)
nir_intrinsic_load_local_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_local_invocation_id', 364)
nir_intrinsic_load_local_invocation_index = nir_intrinsic_op.define('nir_intrinsic_load_local_invocation_index', 365)
nir_intrinsic_load_local_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_load_local_pixel_agx', 366)
nir_intrinsic_load_local_shared_r600 = nir_intrinsic_op.define('nir_intrinsic_load_local_shared_r600', 367)
nir_intrinsic_load_lshs_vertex_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_lshs_vertex_stride_amd', 368)
nir_intrinsic_load_max_polygon_intel = nir_intrinsic_op.define('nir_intrinsic_load_max_polygon_intel', 369)
nir_intrinsic_load_merged_wave_info_amd = nir_intrinsic_op.define('nir_intrinsic_load_merged_wave_info_amd', 370)
nir_intrinsic_load_mesh_view_count = nir_intrinsic_op.define('nir_intrinsic_load_mesh_view_count', 371)
nir_intrinsic_load_mesh_view_indices = nir_intrinsic_op.define('nir_intrinsic_load_mesh_view_indices', 372)
nir_intrinsic_load_multisampled_pan = nir_intrinsic_op.define('nir_intrinsic_load_multisampled_pan', 373)
nir_intrinsic_load_noperspective_varyings_pan = nir_intrinsic_op.define('nir_intrinsic_load_noperspective_varyings_pan', 374)
nir_intrinsic_load_num_subgroups = nir_intrinsic_op.define('nir_intrinsic_load_num_subgroups', 375)
nir_intrinsic_load_num_vertices = nir_intrinsic_op.define('nir_intrinsic_load_num_vertices', 376)
nir_intrinsic_load_num_vertices_per_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_num_vertices_per_primitive_amd', 377)
nir_intrinsic_load_num_workgroups = nir_intrinsic_op.define('nir_intrinsic_load_num_workgroups', 378)
nir_intrinsic_load_ordered_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_ordered_id_amd', 379)
nir_intrinsic_load_output = nir_intrinsic_op.define('nir_intrinsic_load_output', 380)
nir_intrinsic_load_packed_passthrough_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_packed_passthrough_primitive_amd', 381)
nir_intrinsic_load_param = nir_intrinsic_op.define('nir_intrinsic_load_param', 382)
nir_intrinsic_load_patch_vertices_in = nir_intrinsic_op.define('nir_intrinsic_load_patch_vertices_in', 383)
nir_intrinsic_load_per_primitive_input = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_input', 384)
nir_intrinsic_load_per_primitive_output = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_output', 385)
nir_intrinsic_load_per_primitive_remap_intel = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_remap_intel', 386)
nir_intrinsic_load_per_vertex_input = nir_intrinsic_op.define('nir_intrinsic_load_per_vertex_input', 387)
nir_intrinsic_load_per_vertex_output = nir_intrinsic_op.define('nir_intrinsic_load_per_vertex_output', 388)
nir_intrinsic_load_per_view_output = nir_intrinsic_op.define('nir_intrinsic_load_per_view_output', 389)
nir_intrinsic_load_persp_center_rhw_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_persp_center_rhw_ir3', 390)
nir_intrinsic_load_pipeline_stat_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_pipeline_stat_query_enabled_amd', 391)
nir_intrinsic_load_pixel_coord = nir_intrinsic_op.define('nir_intrinsic_load_pixel_coord', 392)
nir_intrinsic_load_point_coord = nir_intrinsic_op.define('nir_intrinsic_load_point_coord', 393)
nir_intrinsic_load_point_coord_maybe_flipped = nir_intrinsic_op.define('nir_intrinsic_load_point_coord_maybe_flipped', 394)
nir_intrinsic_load_poly_line_smooth_enabled = nir_intrinsic_op.define('nir_intrinsic_load_poly_line_smooth_enabled', 395)
nir_intrinsic_load_polygon_stipple_agx = nir_intrinsic_op.define('nir_intrinsic_load_polygon_stipple_agx', 396)
nir_intrinsic_load_polygon_stipple_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_polygon_stipple_buffer_amd', 397)
nir_intrinsic_load_preamble = nir_intrinsic_op.define('nir_intrinsic_load_preamble', 398)
nir_intrinsic_load_prim_gen_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_prim_gen_query_enabled_amd', 399)
nir_intrinsic_load_prim_xfb_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_prim_xfb_query_enabled_amd', 400)
nir_intrinsic_load_primitive_id = nir_intrinsic_op.define('nir_intrinsic_load_primitive_id', 401)
nir_intrinsic_load_primitive_location_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_primitive_location_ir3', 402)
nir_intrinsic_load_printf_buffer_address = nir_intrinsic_op.define('nir_intrinsic_load_printf_buffer_address', 403)
nir_intrinsic_load_printf_buffer_size = nir_intrinsic_op.define('nir_intrinsic_load_printf_buffer_size', 404)
nir_intrinsic_load_provoking_last = nir_intrinsic_op.define('nir_intrinsic_load_provoking_last', 405)
nir_intrinsic_load_provoking_vtx_amd = nir_intrinsic_op.define('nir_intrinsic_load_provoking_vtx_amd', 406)
nir_intrinsic_load_provoking_vtx_in_prim_amd = nir_intrinsic_op.define('nir_intrinsic_load_provoking_vtx_in_prim_amd', 407)
nir_intrinsic_load_push_constant = nir_intrinsic_op.define('nir_intrinsic_load_push_constant', 408)
nir_intrinsic_load_push_constant_zink = nir_intrinsic_op.define('nir_intrinsic_load_push_constant_zink', 409)
nir_intrinsic_load_r600_indirect_per_vertex_input = nir_intrinsic_op.define('nir_intrinsic_load_r600_indirect_per_vertex_input', 410)
nir_intrinsic_load_rasterization_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_primitive_amd', 411)
nir_intrinsic_load_rasterization_samples_amd = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_samples_amd', 412)
nir_intrinsic_load_rasterization_stream = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_stream', 413)
nir_intrinsic_load_raw_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_output_pan', 414)
nir_intrinsic_load_raw_vertex_id_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_vertex_id_pan', 415)
nir_intrinsic_load_raw_vertex_offset_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_vertex_offset_pan', 416)
nir_intrinsic_load_ray_base_mem_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_base_mem_addr_intel', 417)
nir_intrinsic_load_ray_flags = nir_intrinsic_op.define('nir_intrinsic_load_ray_flags', 418)
nir_intrinsic_load_ray_geometry_index = nir_intrinsic_op.define('nir_intrinsic_load_ray_geometry_index', 419)
nir_intrinsic_load_ray_hit_kind = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_kind', 420)
nir_intrinsic_load_ray_hit_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_sbt_addr_intel', 421)
nir_intrinsic_load_ray_hit_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_sbt_stride_intel', 422)
nir_intrinsic_load_ray_hw_stack_size_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hw_stack_size_intel', 423)
nir_intrinsic_load_ray_instance_custom_index = nir_intrinsic_op.define('nir_intrinsic_load_ray_instance_custom_index', 424)
nir_intrinsic_load_ray_launch_id = nir_intrinsic_op.define('nir_intrinsic_load_ray_launch_id', 425)
nir_intrinsic_load_ray_launch_size = nir_intrinsic_op.define('nir_intrinsic_load_ray_launch_size', 426)
nir_intrinsic_load_ray_miss_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_miss_sbt_addr_intel', 427)
nir_intrinsic_load_ray_miss_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_miss_sbt_stride_intel', 428)
nir_intrinsic_load_ray_num_dss_rt_stacks_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_num_dss_rt_stacks_intel', 429)
nir_intrinsic_load_ray_object_direction = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_direction', 430)
nir_intrinsic_load_ray_object_origin = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_origin', 431)
nir_intrinsic_load_ray_object_to_world = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_to_world', 432)
nir_intrinsic_load_ray_query_global_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_query_global_intel', 433)
nir_intrinsic_load_ray_sw_stack_size_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_sw_stack_size_intel', 434)
nir_intrinsic_load_ray_t_max = nir_intrinsic_op.define('nir_intrinsic_load_ray_t_max', 435)
nir_intrinsic_load_ray_t_min = nir_intrinsic_op.define('nir_intrinsic_load_ray_t_min', 436)
nir_intrinsic_load_ray_tracing_stack_base_lvp = nir_intrinsic_op.define('nir_intrinsic_load_ray_tracing_stack_base_lvp', 437)
nir_intrinsic_load_ray_triangle_vertex_positions = nir_intrinsic_op.define('nir_intrinsic_load_ray_triangle_vertex_positions', 438)
nir_intrinsic_load_ray_world_direction = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_direction', 439)
nir_intrinsic_load_ray_world_origin = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_origin', 440)
nir_intrinsic_load_ray_world_to_object = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_to_object', 441)
nir_intrinsic_load_readonly_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_readonly_output_pan', 442)
nir_intrinsic_load_reg = nir_intrinsic_op.define('nir_intrinsic_load_reg', 443)
nir_intrinsic_load_reg_indirect = nir_intrinsic_op.define('nir_intrinsic_load_reg_indirect', 444)
nir_intrinsic_load_rel_patch_id_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_rel_patch_id_ir3', 445)
nir_intrinsic_load_reloc_const_intel = nir_intrinsic_op.define('nir_intrinsic_load_reloc_const_intel', 446)
nir_intrinsic_load_resume_shader_address_amd = nir_intrinsic_op.define('nir_intrinsic_load_resume_shader_address_amd', 447)
nir_intrinsic_load_ring_attr_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_attr_amd', 448)
nir_intrinsic_load_ring_attr_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_attr_offset_amd', 449)
nir_intrinsic_load_ring_es2gs_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_es2gs_offset_amd', 450)
nir_intrinsic_load_ring_esgs_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_esgs_amd', 451)
nir_intrinsic_load_ring_gs2vs_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_gs2vs_offset_amd', 452)
nir_intrinsic_load_ring_gsvs_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_gsvs_amd', 453)
nir_intrinsic_load_ring_mesh_scratch_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_mesh_scratch_amd', 454)
nir_intrinsic_load_ring_mesh_scratch_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_mesh_scratch_offset_amd', 455)
nir_intrinsic_load_ring_task_draw_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_task_draw_amd', 456)
nir_intrinsic_load_ring_task_payload_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_task_payload_amd', 457)
nir_intrinsic_load_ring_tess_factors_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_factors_amd', 458)
nir_intrinsic_load_ring_tess_factors_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_factors_offset_amd', 459)
nir_intrinsic_load_ring_tess_offchip_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_offchip_amd', 460)
nir_intrinsic_load_ring_tess_offchip_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_offchip_offset_amd', 461)
nir_intrinsic_load_root_agx = nir_intrinsic_op.define('nir_intrinsic_load_root_agx', 462)
nir_intrinsic_load_rt_arg_scratch_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_rt_arg_scratch_offset_amd', 463)
nir_intrinsic_load_rt_conversion_pan = nir_intrinsic_op.define('nir_intrinsic_load_rt_conversion_pan', 464)
nir_intrinsic_load_sample_id = nir_intrinsic_op.define('nir_intrinsic_load_sample_id', 465)
nir_intrinsic_load_sample_id_no_per_sample = nir_intrinsic_op.define('nir_intrinsic_load_sample_id_no_per_sample', 466)
nir_intrinsic_load_sample_mask = nir_intrinsic_op.define('nir_intrinsic_load_sample_mask', 467)
nir_intrinsic_load_sample_mask_in = nir_intrinsic_op.define('nir_intrinsic_load_sample_mask_in', 468)
nir_intrinsic_load_sample_pos = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos', 469)
nir_intrinsic_load_sample_pos_from_id = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos_from_id', 470)
nir_intrinsic_load_sample_pos_or_center = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos_or_center', 471)
nir_intrinsic_load_sample_positions_agx = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_agx', 472)
nir_intrinsic_load_sample_positions_amd = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_amd', 473)
nir_intrinsic_load_sample_positions_pan = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_pan', 474)
nir_intrinsic_load_sampler_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_sampler_handle_agx', 475)
nir_intrinsic_load_sampler_lod_parameters = nir_intrinsic_op.define('nir_intrinsic_load_sampler_lod_parameters', 476)
nir_intrinsic_load_samples_log2_agx = nir_intrinsic_op.define('nir_intrinsic_load_samples_log2_agx', 477)
nir_intrinsic_load_sbt_base_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_base_amd', 478)
nir_intrinsic_load_sbt_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_offset_amd', 479)
nir_intrinsic_load_sbt_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_stride_amd', 480)
nir_intrinsic_load_scalar_arg_amd = nir_intrinsic_op.define('nir_intrinsic_load_scalar_arg_amd', 481)
nir_intrinsic_load_scratch = nir_intrinsic_op.define('nir_intrinsic_load_scratch', 482)
nir_intrinsic_load_scratch_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_scratch_base_ptr', 483)
nir_intrinsic_load_shader_call_data_offset_lvp = nir_intrinsic_op.define('nir_intrinsic_load_shader_call_data_offset_lvp', 484)
nir_intrinsic_load_shader_index = nir_intrinsic_op.define('nir_intrinsic_load_shader_index', 485)
nir_intrinsic_load_shader_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_shader_output_pan', 486)
nir_intrinsic_load_shader_part_tests_zs_agx = nir_intrinsic_op.define('nir_intrinsic_load_shader_part_tests_zs_agx', 487)
nir_intrinsic_load_shader_record_ptr = nir_intrinsic_op.define('nir_intrinsic_load_shader_record_ptr', 488)
nir_intrinsic_load_shared = nir_intrinsic_op.define('nir_intrinsic_load_shared', 489)
nir_intrinsic_load_shared2_amd = nir_intrinsic_op.define('nir_intrinsic_load_shared2_amd', 490)
nir_intrinsic_load_shared_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_shared_base_ptr', 491)
nir_intrinsic_load_shared_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_shared_block_intel', 492)
nir_intrinsic_load_shared_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_shared_ir3', 493)
nir_intrinsic_load_shared_lock_nv = nir_intrinsic_op.define('nir_intrinsic_load_shared_lock_nv', 494)
nir_intrinsic_load_shared_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_shared_uniform_block_intel', 495)
nir_intrinsic_load_simd_width_intel = nir_intrinsic_op.define('nir_intrinsic_load_simd_width_intel', 496)
nir_intrinsic_load_sm_count_nv = nir_intrinsic_op.define('nir_intrinsic_load_sm_count_nv', 497)
nir_intrinsic_load_sm_id_nv = nir_intrinsic_op.define('nir_intrinsic_load_sm_id_nv', 498)
nir_intrinsic_load_smem_amd = nir_intrinsic_op.define('nir_intrinsic_load_smem_amd', 499)
nir_intrinsic_load_ssbo = nir_intrinsic_op.define('nir_intrinsic_load_ssbo', 500)
nir_intrinsic_load_ssbo_address = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_address', 501)
nir_intrinsic_load_ssbo_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_block_intel', 502)
nir_intrinsic_load_ssbo_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_intel', 503)
nir_intrinsic_load_ssbo_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_ir3', 504)
nir_intrinsic_load_ssbo_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_uniform_block_intel', 505)
nir_intrinsic_load_stack = nir_intrinsic_op.define('nir_intrinsic_load_stack', 506)
nir_intrinsic_load_stat_query_address_agx = nir_intrinsic_op.define('nir_intrinsic_load_stat_query_address_agx', 507)
nir_intrinsic_load_streamout_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_buffer_amd', 508)
nir_intrinsic_load_streamout_config_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_config_amd', 509)
nir_intrinsic_load_streamout_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_offset_amd', 510)
nir_intrinsic_load_streamout_write_index_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_write_index_amd', 511)
nir_intrinsic_load_subgroup_eq_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_eq_mask', 512)
nir_intrinsic_load_subgroup_ge_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_ge_mask', 513)
nir_intrinsic_load_subgroup_gt_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_gt_mask', 514)
nir_intrinsic_load_subgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_id', 515)
nir_intrinsic_load_subgroup_id_shift_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_id_shift_ir3', 516)
nir_intrinsic_load_subgroup_invocation = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_invocation', 517)
nir_intrinsic_load_subgroup_le_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_le_mask', 518)
nir_intrinsic_load_subgroup_lt_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_lt_mask', 519)
nir_intrinsic_load_subgroup_size = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_size', 520)
nir_intrinsic_load_sysval_agx = nir_intrinsic_op.define('nir_intrinsic_load_sysval_agx', 521)
nir_intrinsic_load_sysval_nv = nir_intrinsic_op.define('nir_intrinsic_load_sysval_nv', 522)
nir_intrinsic_load_task_payload = nir_intrinsic_op.define('nir_intrinsic_load_task_payload', 523)
nir_intrinsic_load_task_ring_entry_amd = nir_intrinsic_op.define('nir_intrinsic_load_task_ring_entry_amd', 524)
nir_intrinsic_load_tcs_header_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_header_ir3', 525)
nir_intrinsic_load_tcs_in_param_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_in_param_base_r600', 526)
nir_intrinsic_load_tcs_mem_attrib_stride = nir_intrinsic_op.define('nir_intrinsic_load_tcs_mem_attrib_stride', 527)
nir_intrinsic_load_tcs_num_patches_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_num_patches_amd', 528)
nir_intrinsic_load_tcs_out_param_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_out_param_base_r600', 529)
nir_intrinsic_load_tcs_primitive_mode_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_primitive_mode_amd', 530)
nir_intrinsic_load_tcs_rel_patch_id_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_rel_patch_id_r600', 531)
nir_intrinsic_load_tcs_tess_factor_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_tess_factor_base_r600', 532)
nir_intrinsic_load_tcs_tess_levels_to_tes_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_tess_levels_to_tes_amd', 533)
nir_intrinsic_load_tess_coord = nir_intrinsic_op.define('nir_intrinsic_load_tess_coord', 534)
nir_intrinsic_load_tess_coord_xy = nir_intrinsic_op.define('nir_intrinsic_load_tess_coord_xy', 535)
nir_intrinsic_load_tess_factor_base_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tess_factor_base_ir3', 536)
nir_intrinsic_load_tess_level_inner = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_inner', 537)
nir_intrinsic_load_tess_level_inner_default = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_inner_default', 538)
nir_intrinsic_load_tess_level_outer = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_outer', 539)
nir_intrinsic_load_tess_level_outer_default = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_outer_default', 540)
nir_intrinsic_load_tess_param_base_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tess_param_base_ir3', 541)
nir_intrinsic_load_tess_param_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_tess_param_buffer_poly', 542)
nir_intrinsic_load_tess_rel_patch_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_tess_rel_patch_id_amd', 543)
nir_intrinsic_load_tex_sprite_mask_agx = nir_intrinsic_op.define('nir_intrinsic_load_tex_sprite_mask_agx', 544)
nir_intrinsic_load_texture_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_texture_handle_agx', 545)
nir_intrinsic_load_texture_scale = nir_intrinsic_op.define('nir_intrinsic_load_texture_scale', 546)
nir_intrinsic_load_texture_size_etna = nir_intrinsic_op.define('nir_intrinsic_load_texture_size_etna', 547)
nir_intrinsic_load_tlb_color_brcm = nir_intrinsic_op.define('nir_intrinsic_load_tlb_color_brcm', 548)
nir_intrinsic_load_topology_id_intel = nir_intrinsic_op.define('nir_intrinsic_load_topology_id_intel', 549)
nir_intrinsic_load_typed_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_typed_buffer_amd', 550)
nir_intrinsic_load_uav_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_uav_ir3', 551)
nir_intrinsic_load_ubo = nir_intrinsic_op.define('nir_intrinsic_load_ubo', 552)
nir_intrinsic_load_ubo_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ubo_uniform_block_intel', 553)
nir_intrinsic_load_ubo_vec4 = nir_intrinsic_op.define('nir_intrinsic_load_ubo_vec4', 554)
nir_intrinsic_load_uniform = nir_intrinsic_op.define('nir_intrinsic_load_uniform', 555)
nir_intrinsic_load_user_clip_plane = nir_intrinsic_op.define('nir_intrinsic_load_user_clip_plane', 556)
nir_intrinsic_load_user_data_amd = nir_intrinsic_op.define('nir_intrinsic_load_user_data_amd', 557)
nir_intrinsic_load_uvs_index_agx = nir_intrinsic_op.define('nir_intrinsic_load_uvs_index_agx', 558)
nir_intrinsic_load_vbo_base_agx = nir_intrinsic_op.define('nir_intrinsic_load_vbo_base_agx', 559)
nir_intrinsic_load_vector_arg_amd = nir_intrinsic_op.define('nir_intrinsic_load_vector_arg_amd', 560)
nir_intrinsic_load_vertex_id = nir_intrinsic_op.define('nir_intrinsic_load_vertex_id', 561)
nir_intrinsic_load_vertex_id_zero_base = nir_intrinsic_op.define('nir_intrinsic_load_vertex_id_zero_base', 562)
nir_intrinsic_load_view_index = nir_intrinsic_op.define('nir_intrinsic_load_view_index', 563)
nir_intrinsic_load_viewport_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_offset', 564)
nir_intrinsic_load_viewport_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_scale', 565)
nir_intrinsic_load_viewport_x_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_x_offset', 566)
nir_intrinsic_load_viewport_x_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_x_scale', 567)
nir_intrinsic_load_viewport_y_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_y_offset', 568)
nir_intrinsic_load_viewport_y_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_y_scale', 569)
nir_intrinsic_load_viewport_z_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_z_offset', 570)
nir_intrinsic_load_viewport_z_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_z_scale', 571)
nir_intrinsic_load_vs_output_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_vs_output_buffer_poly', 572)
nir_intrinsic_load_vs_outputs_poly = nir_intrinsic_op.define('nir_intrinsic_load_vs_outputs_poly', 573)
nir_intrinsic_load_vs_primitive_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_vs_primitive_stride_ir3', 574)
nir_intrinsic_load_vs_vertex_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_vs_vertex_stride_ir3', 575)
nir_intrinsic_load_vulkan_descriptor = nir_intrinsic_op.define('nir_intrinsic_load_vulkan_descriptor', 576)
nir_intrinsic_load_warp_id_nv = nir_intrinsic_op.define('nir_intrinsic_load_warp_id_nv', 577)
nir_intrinsic_load_warps_per_sm_nv = nir_intrinsic_op.define('nir_intrinsic_load_warps_per_sm_nv', 578)
nir_intrinsic_load_work_dim = nir_intrinsic_op.define('nir_intrinsic_load_work_dim', 579)
nir_intrinsic_load_workgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_id', 580)
nir_intrinsic_load_workgroup_index = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_index', 581)
nir_intrinsic_load_workgroup_num_input_primitives_amd = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_num_input_primitives_amd', 582)
nir_intrinsic_load_workgroup_num_input_vertices_amd = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_num_input_vertices_amd', 583)
nir_intrinsic_load_workgroup_size = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_size', 584)
nir_intrinsic_load_xfb_address = nir_intrinsic_op.define('nir_intrinsic_load_xfb_address', 585)
nir_intrinsic_load_xfb_index_buffer = nir_intrinsic_op.define('nir_intrinsic_load_xfb_index_buffer', 586)
nir_intrinsic_load_xfb_size = nir_intrinsic_op.define('nir_intrinsic_load_xfb_size', 587)
nir_intrinsic_load_xfb_state_address_gfx12_amd = nir_intrinsic_op.define('nir_intrinsic_load_xfb_state_address_gfx12_amd', 588)
nir_intrinsic_masked_swizzle_amd = nir_intrinsic_op.define('nir_intrinsic_masked_swizzle_amd', 589)
nir_intrinsic_mbcnt_amd = nir_intrinsic_op.define('nir_intrinsic_mbcnt_amd', 590)
nir_intrinsic_memcpy_deref = nir_intrinsic_op.define('nir_intrinsic_memcpy_deref', 591)
nir_intrinsic_nop = nir_intrinsic_op.define('nir_intrinsic_nop', 592)
nir_intrinsic_nop_amd = nir_intrinsic_op.define('nir_intrinsic_nop_amd', 593)
nir_intrinsic_optimization_barrier_sgpr_amd = nir_intrinsic_op.define('nir_intrinsic_optimization_barrier_sgpr_amd', 594)
nir_intrinsic_optimization_barrier_vgpr_amd = nir_intrinsic_op.define('nir_intrinsic_optimization_barrier_vgpr_amd', 595)
nir_intrinsic_ordered_add_loop_gfx12_amd = nir_intrinsic_op.define('nir_intrinsic_ordered_add_loop_gfx12_amd', 596)
nir_intrinsic_ordered_xfb_counter_add_gfx11_amd = nir_intrinsic_op.define('nir_intrinsic_ordered_xfb_counter_add_gfx11_amd', 597)
nir_intrinsic_overwrite_tes_arguments_amd = nir_intrinsic_op.define('nir_intrinsic_overwrite_tes_arguments_amd', 598)
nir_intrinsic_overwrite_vs_arguments_amd = nir_intrinsic_op.define('nir_intrinsic_overwrite_vs_arguments_amd', 599)
nir_intrinsic_pin_cx_handle_nv = nir_intrinsic_op.define('nir_intrinsic_pin_cx_handle_nv', 600)
nir_intrinsic_preamble_end_ir3 = nir_intrinsic_op.define('nir_intrinsic_preamble_end_ir3', 601)
nir_intrinsic_preamble_start_ir3 = nir_intrinsic_op.define('nir_intrinsic_preamble_start_ir3', 602)
nir_intrinsic_prefetch_sam_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_sam_ir3', 603)
nir_intrinsic_prefetch_tex_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_tex_ir3', 604)
nir_intrinsic_prefetch_ubo_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_ubo_ir3', 605)
nir_intrinsic_printf = nir_intrinsic_op.define('nir_intrinsic_printf', 606)
nir_intrinsic_printf_abort = nir_intrinsic_op.define('nir_intrinsic_printf_abort', 607)
nir_intrinsic_quad_ballot_agx = nir_intrinsic_op.define('nir_intrinsic_quad_ballot_agx', 608)
nir_intrinsic_quad_broadcast = nir_intrinsic_op.define('nir_intrinsic_quad_broadcast', 609)
nir_intrinsic_quad_swap_diagonal = nir_intrinsic_op.define('nir_intrinsic_quad_swap_diagonal', 610)
nir_intrinsic_quad_swap_horizontal = nir_intrinsic_op.define('nir_intrinsic_quad_swap_horizontal', 611)
nir_intrinsic_quad_swap_vertical = nir_intrinsic_op.define('nir_intrinsic_quad_swap_vertical', 612)
nir_intrinsic_quad_swizzle_amd = nir_intrinsic_op.define('nir_intrinsic_quad_swizzle_amd', 613)
nir_intrinsic_quad_vote_all = nir_intrinsic_op.define('nir_intrinsic_quad_vote_all', 614)
nir_intrinsic_quad_vote_any = nir_intrinsic_op.define('nir_intrinsic_quad_vote_any', 615)
nir_intrinsic_r600_indirect_vertex_at_index = nir_intrinsic_op.define('nir_intrinsic_r600_indirect_vertex_at_index', 616)
nir_intrinsic_ray_intersection_ir3 = nir_intrinsic_op.define('nir_intrinsic_ray_intersection_ir3', 617)
nir_intrinsic_read_attribute_payload_intel = nir_intrinsic_op.define('nir_intrinsic_read_attribute_payload_intel', 618)
nir_intrinsic_read_first_invocation = nir_intrinsic_op.define('nir_intrinsic_read_first_invocation', 619)
nir_intrinsic_read_getlast_ir3 = nir_intrinsic_op.define('nir_intrinsic_read_getlast_ir3', 620)
nir_intrinsic_read_invocation = nir_intrinsic_op.define('nir_intrinsic_read_invocation', 621)
nir_intrinsic_read_invocation_cond_ir3 = nir_intrinsic_op.define('nir_intrinsic_read_invocation_cond_ir3', 622)
nir_intrinsic_reduce = nir_intrinsic_op.define('nir_intrinsic_reduce', 623)
nir_intrinsic_reduce_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_reduce_clusters_ir3', 624)
nir_intrinsic_report_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_report_ray_intersection', 625)
nir_intrinsic_resource_intel = nir_intrinsic_op.define('nir_intrinsic_resource_intel', 626)
nir_intrinsic_rotate = nir_intrinsic_op.define('nir_intrinsic_rotate', 627)
nir_intrinsic_rq_confirm_intersection = nir_intrinsic_op.define('nir_intrinsic_rq_confirm_intersection', 628)
nir_intrinsic_rq_generate_intersection = nir_intrinsic_op.define('nir_intrinsic_rq_generate_intersection', 629)
nir_intrinsic_rq_initialize = nir_intrinsic_op.define('nir_intrinsic_rq_initialize', 630)
nir_intrinsic_rq_load = nir_intrinsic_op.define('nir_intrinsic_rq_load', 631)
nir_intrinsic_rq_proceed = nir_intrinsic_op.define('nir_intrinsic_rq_proceed', 632)
nir_intrinsic_rq_terminate = nir_intrinsic_op.define('nir_intrinsic_rq_terminate', 633)
nir_intrinsic_rt_execute_callable = nir_intrinsic_op.define('nir_intrinsic_rt_execute_callable', 634)
nir_intrinsic_rt_resume = nir_intrinsic_op.define('nir_intrinsic_rt_resume', 635)
nir_intrinsic_rt_return_amd = nir_intrinsic_op.define('nir_intrinsic_rt_return_amd', 636)
nir_intrinsic_rt_trace_ray = nir_intrinsic_op.define('nir_intrinsic_rt_trace_ray', 637)
nir_intrinsic_sample_mask_agx = nir_intrinsic_op.define('nir_intrinsic_sample_mask_agx', 638)
nir_intrinsic_select_vertex_poly = nir_intrinsic_op.define('nir_intrinsic_select_vertex_poly', 639)
nir_intrinsic_sendmsg_amd = nir_intrinsic_op.define('nir_intrinsic_sendmsg_amd', 640)
nir_intrinsic_set_vertex_and_primitive_count = nir_intrinsic_op.define('nir_intrinsic_set_vertex_and_primitive_count', 641)
nir_intrinsic_shader_clock = nir_intrinsic_op.define('nir_intrinsic_shader_clock', 642)
nir_intrinsic_shared_append_amd = nir_intrinsic_op.define('nir_intrinsic_shared_append_amd', 643)
nir_intrinsic_shared_atomic = nir_intrinsic_op.define('nir_intrinsic_shared_atomic', 644)
nir_intrinsic_shared_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_shared_atomic_swap', 645)
nir_intrinsic_shared_consume_amd = nir_intrinsic_op.define('nir_intrinsic_shared_consume_amd', 646)
nir_intrinsic_shuffle = nir_intrinsic_op.define('nir_intrinsic_shuffle', 647)
nir_intrinsic_shuffle_down = nir_intrinsic_op.define('nir_intrinsic_shuffle_down', 648)
nir_intrinsic_shuffle_down_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_down_uniform_ir3', 649)
nir_intrinsic_shuffle_up = nir_intrinsic_op.define('nir_intrinsic_shuffle_up', 650)
nir_intrinsic_shuffle_up_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_up_uniform_ir3', 651)
nir_intrinsic_shuffle_xor = nir_intrinsic_op.define('nir_intrinsic_shuffle_xor', 652)
nir_intrinsic_shuffle_xor_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_xor_uniform_ir3', 653)
nir_intrinsic_sleep_amd = nir_intrinsic_op.define('nir_intrinsic_sleep_amd', 654)
nir_intrinsic_sparse_residency_code_and = nir_intrinsic_op.define('nir_intrinsic_sparse_residency_code_and', 655)
nir_intrinsic_ssa_bar_nv = nir_intrinsic_op.define('nir_intrinsic_ssa_bar_nv', 656)
nir_intrinsic_ssbo_atomic = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic', 657)
nir_intrinsic_ssbo_atomic_ir3 = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_ir3', 658)
nir_intrinsic_ssbo_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_swap', 659)
nir_intrinsic_ssbo_atomic_swap_ir3 = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_swap_ir3', 660)
nir_intrinsic_stack_map_agx = nir_intrinsic_op.define('nir_intrinsic_stack_map_agx', 661)
nir_intrinsic_stack_unmap_agx = nir_intrinsic_op.define('nir_intrinsic_stack_unmap_agx', 662)
nir_intrinsic_store_agx = nir_intrinsic_op.define('nir_intrinsic_store_agx', 663)
nir_intrinsic_store_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_store_buffer_amd', 664)
nir_intrinsic_store_combined_output_pan = nir_intrinsic_op.define('nir_intrinsic_store_combined_output_pan', 665)
nir_intrinsic_store_const_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_const_ir3', 666)
nir_intrinsic_store_deref = nir_intrinsic_op.define('nir_intrinsic_store_deref', 667)
nir_intrinsic_store_deref_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_deref_block_intel', 668)
nir_intrinsic_store_global = nir_intrinsic_op.define('nir_intrinsic_store_global', 669)
nir_intrinsic_store_global_2x32 = nir_intrinsic_op.define('nir_intrinsic_store_global_2x32', 670)
nir_intrinsic_store_global_amd = nir_intrinsic_op.define('nir_intrinsic_store_global_amd', 671)
nir_intrinsic_store_global_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_global_block_intel', 672)
nir_intrinsic_store_global_etna = nir_intrinsic_op.define('nir_intrinsic_store_global_etna', 673)
nir_intrinsic_store_global_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_global_ir3', 674)
nir_intrinsic_store_hit_attrib_amd = nir_intrinsic_op.define('nir_intrinsic_store_hit_attrib_amd', 675)
nir_intrinsic_store_local_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_store_local_pixel_agx', 676)
nir_intrinsic_store_local_shared_r600 = nir_intrinsic_op.define('nir_intrinsic_store_local_shared_r600', 677)
nir_intrinsic_store_output = nir_intrinsic_op.define('nir_intrinsic_store_output', 678)
nir_intrinsic_store_per_primitive_output = nir_intrinsic_op.define('nir_intrinsic_store_per_primitive_output', 679)
nir_intrinsic_store_per_primitive_payload_intel = nir_intrinsic_op.define('nir_intrinsic_store_per_primitive_payload_intel', 680)
nir_intrinsic_store_per_vertex_output = nir_intrinsic_op.define('nir_intrinsic_store_per_vertex_output', 681)
nir_intrinsic_store_per_view_output = nir_intrinsic_op.define('nir_intrinsic_store_per_view_output', 682)
nir_intrinsic_store_preamble = nir_intrinsic_op.define('nir_intrinsic_store_preamble', 683)
nir_intrinsic_store_raw_output_pan = nir_intrinsic_op.define('nir_intrinsic_store_raw_output_pan', 684)
nir_intrinsic_store_reg = nir_intrinsic_op.define('nir_intrinsic_store_reg', 685)
nir_intrinsic_store_reg_indirect = nir_intrinsic_op.define('nir_intrinsic_store_reg_indirect', 686)
nir_intrinsic_store_scalar_arg_amd = nir_intrinsic_op.define('nir_intrinsic_store_scalar_arg_amd', 687)
nir_intrinsic_store_scratch = nir_intrinsic_op.define('nir_intrinsic_store_scratch', 688)
nir_intrinsic_store_shared = nir_intrinsic_op.define('nir_intrinsic_store_shared', 689)
nir_intrinsic_store_shared2_amd = nir_intrinsic_op.define('nir_intrinsic_store_shared2_amd', 690)
nir_intrinsic_store_shared_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_shared_block_intel', 691)
nir_intrinsic_store_shared_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_shared_ir3', 692)
nir_intrinsic_store_shared_unlock_nv = nir_intrinsic_op.define('nir_intrinsic_store_shared_unlock_nv', 693)
nir_intrinsic_store_ssbo = nir_intrinsic_op.define('nir_intrinsic_store_ssbo', 694)
nir_intrinsic_store_ssbo_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_block_intel', 695)
nir_intrinsic_store_ssbo_intel = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_intel', 696)
nir_intrinsic_store_ssbo_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_ir3', 697)
nir_intrinsic_store_stack = nir_intrinsic_op.define('nir_intrinsic_store_stack', 698)
nir_intrinsic_store_task_payload = nir_intrinsic_op.define('nir_intrinsic_store_task_payload', 699)
nir_intrinsic_store_tf_r600 = nir_intrinsic_op.define('nir_intrinsic_store_tf_r600', 700)
nir_intrinsic_store_tlb_sample_color_v3d = nir_intrinsic_op.define('nir_intrinsic_store_tlb_sample_color_v3d', 701)
nir_intrinsic_store_uvs_agx = nir_intrinsic_op.define('nir_intrinsic_store_uvs_agx', 702)
nir_intrinsic_store_vector_arg_amd = nir_intrinsic_op.define('nir_intrinsic_store_vector_arg_amd', 703)
nir_intrinsic_store_zs_agx = nir_intrinsic_op.define('nir_intrinsic_store_zs_agx', 704)
nir_intrinsic_strict_wqm_coord_amd = nir_intrinsic_op.define('nir_intrinsic_strict_wqm_coord_amd', 705)
nir_intrinsic_subfm_nv = nir_intrinsic_op.define('nir_intrinsic_subfm_nv', 706)
nir_intrinsic_suclamp_nv = nir_intrinsic_op.define('nir_intrinsic_suclamp_nv', 707)
nir_intrinsic_sueau_nv = nir_intrinsic_op.define('nir_intrinsic_sueau_nv', 708)
nir_intrinsic_suldga_nv = nir_intrinsic_op.define('nir_intrinsic_suldga_nv', 709)
nir_intrinsic_sustga_nv = nir_intrinsic_op.define('nir_intrinsic_sustga_nv', 710)
nir_intrinsic_task_payload_atomic = nir_intrinsic_op.define('nir_intrinsic_task_payload_atomic', 711)
nir_intrinsic_task_payload_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_task_payload_atomic_swap', 712)
nir_intrinsic_terminate = nir_intrinsic_op.define('nir_intrinsic_terminate', 713)
nir_intrinsic_terminate_if = nir_intrinsic_op.define('nir_intrinsic_terminate_if', 714)
nir_intrinsic_terminate_ray = nir_intrinsic_op.define('nir_intrinsic_terminate_ray', 715)
nir_intrinsic_trace_ray = nir_intrinsic_op.define('nir_intrinsic_trace_ray', 716)
nir_intrinsic_trace_ray_intel = nir_intrinsic_op.define('nir_intrinsic_trace_ray_intel', 717)
nir_intrinsic_unit_test_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_amd', 718)
nir_intrinsic_unit_test_divergent_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_divergent_amd', 719)
nir_intrinsic_unit_test_uniform_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_uniform_amd', 720)
nir_intrinsic_unpin_cx_handle_nv = nir_intrinsic_op.define('nir_intrinsic_unpin_cx_handle_nv', 721)
nir_intrinsic_use = nir_intrinsic_op.define('nir_intrinsic_use', 722)
nir_intrinsic_vild_nv = nir_intrinsic_op.define('nir_intrinsic_vild_nv', 723)
nir_intrinsic_vote_all = nir_intrinsic_op.define('nir_intrinsic_vote_all', 724)
nir_intrinsic_vote_any = nir_intrinsic_op.define('nir_intrinsic_vote_any', 725)
nir_intrinsic_vote_feq = nir_intrinsic_op.define('nir_intrinsic_vote_feq', 726)
nir_intrinsic_vote_ieq = nir_intrinsic_op.define('nir_intrinsic_vote_ieq', 727)
nir_intrinsic_vulkan_resource_index = nir_intrinsic_op.define('nir_intrinsic_vulkan_resource_index', 728)
nir_intrinsic_vulkan_resource_reindex = nir_intrinsic_op.define('nir_intrinsic_vulkan_resource_reindex', 729)
nir_intrinsic_write_invocation_amd = nir_intrinsic_op.define('nir_intrinsic_write_invocation_amd', 730)
nir_intrinsic_xfb_counter_sub_gfx11_amd = nir_intrinsic_op.define('nir_intrinsic_xfb_counter_sub_gfx11_amd', 731)
nir_last_intrinsic = nir_intrinsic_op.define('nir_last_intrinsic', 731)
nir_num_intrinsics = nir_intrinsic_op.define('nir_num_intrinsics', 732)

struct_nir_intrinsic_instr._fields_ = [
  ('instr', nir_instr),
  ('intrinsic', nir_intrinsic_op),
  ('def', nir_def),
  ('num_components', uint8_t),
  ('const_index', (ctypes.c_int32 * 8)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('src', (nir_src * 0)),
]
nir_intrinsic_instr = struct_nir_intrinsic_instr
nir_memory_semantics = CEnum(ctypes.c_uint32)
NIR_MEMORY_ACQUIRE = nir_memory_semantics.define('NIR_MEMORY_ACQUIRE', 1)
NIR_MEMORY_RELEASE = nir_memory_semantics.define('NIR_MEMORY_RELEASE', 2)
NIR_MEMORY_ACQ_REL = nir_memory_semantics.define('NIR_MEMORY_ACQ_REL', 3)
NIR_MEMORY_MAKE_AVAILABLE = nir_memory_semantics.define('NIR_MEMORY_MAKE_AVAILABLE', 4)
NIR_MEMORY_MAKE_VISIBLE = nir_memory_semantics.define('NIR_MEMORY_MAKE_VISIBLE', 8)

nir_intrinsic_semantic_flag = CEnum(ctypes.c_uint32)
NIR_INTRINSIC_CAN_ELIMINATE = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_CAN_ELIMINATE', 1)
NIR_INTRINSIC_CAN_REORDER = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_CAN_REORDER', 2)
NIR_INTRINSIC_SUBGROUP = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_SUBGROUP', 4)
NIR_INTRINSIC_QUADGROUP = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_QUADGROUP', 8)

class struct_nir_io_semantics(Struct): pass
struct_nir_io_semantics._fields_ = [
  ('location', ctypes.c_uint32,7),
  ('num_slots', ctypes.c_uint32,6),
  ('dual_source_blend_index', ctypes.c_uint32,1),
  ('fb_fetch_output', ctypes.c_uint32,1),
  ('fb_fetch_output_coherent', ctypes.c_uint32,1),
  ('gs_streams', ctypes.c_uint32,8),
  ('medium_precision', ctypes.c_uint32,1),
  ('per_view', ctypes.c_uint32,1),
  ('high_16bits', ctypes.c_uint32,1),
  ('high_dvec2', ctypes.c_uint32,1),
  ('no_varying', ctypes.c_uint32,1),
  ('no_sysval_output', ctypes.c_uint32,1),
  ('interp_explicit_strict', ctypes.c_uint32,1),
  ('_pad', ctypes.c_uint32,1),
]
nir_io_semantics = struct_nir_io_semantics
class struct_nir_io_xfb(Struct): pass
class struct_nir_io_xfb_out(Struct): pass
struct_nir_io_xfb_out._fields_ = [
  ('num_components', uint8_t,4),
  ('buffer', uint8_t,4),
  ('offset', uint8_t),
]
struct_nir_io_xfb._fields_ = [
  ('out', (struct_nir_io_xfb_out * 2)),
]
nir_io_xfb = struct_nir_io_xfb
try: (nir_instr_xfb_write_mask:=dll.nir_instr_xfb_write_mask).restype, nir_instr_xfb_write_mask.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

class struct_nir_intrinsic_info(Struct): pass
struct_nir_intrinsic_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('num_srcs', uint8_t),
  ('src_components', (int8_t * 11)),
  ('has_dest', ctypes.c_bool),
  ('dest_components', uint8_t),
  ('dest_bit_sizes', uint8_t),
  ('bit_size_src', int8_t),
  ('num_indices', uint8_t),
  ('indices', (uint8_t * 8)),
  ('index_map', (uint8_t * 75)),
  ('flags', nir_intrinsic_semantic_flag),
]
nir_intrinsic_info = struct_nir_intrinsic_info
try: nir_intrinsic_infos = (nir_intrinsic_info * 732).in_dll(dll, 'nir_intrinsic_infos')
except (ValueError,AttributeError): pass
try: (nir_intrinsic_src_components:=dll.nir_intrinsic_src_components).restype, nir_intrinsic_src_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_dest_components:=dll.nir_intrinsic_dest_components).restype, nir_intrinsic_dest_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_instr_src_type:=dll.nir_intrinsic_instr_src_type).restype, nir_intrinsic_instr_src_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_instr_dest_type:=dll.nir_intrinsic_instr_dest_type).restype, nir_intrinsic_instr_dest_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_copy_const_indices:=dll.nir_intrinsic_copy_const_indices).restype, nir_intrinsic_copy_const_indices.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_image_intrinsic_coord_components:=dll.nir_image_intrinsic_coord_components).restype, nir_image_intrinsic_coord_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_rewrite_image_intrinsic:=dll.nir_rewrite_image_intrinsic).restype, nir_rewrite_image_intrinsic.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_def), ctypes.c_bool]
except AttributeError: pass

try: (nir_intrinsic_can_reorder:=dll.nir_intrinsic_can_reorder).restype, nir_intrinsic_can_reorder.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_writes_external_memory:=dll.nir_intrinsic_writes_external_memory).restype, nir_intrinsic_writes_external_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

enum_nir_tex_src_type = CEnum(ctypes.c_uint32)
nir_tex_src_coord = enum_nir_tex_src_type.define('nir_tex_src_coord', 0)
nir_tex_src_projector = enum_nir_tex_src_type.define('nir_tex_src_projector', 1)
nir_tex_src_comparator = enum_nir_tex_src_type.define('nir_tex_src_comparator', 2)
nir_tex_src_offset = enum_nir_tex_src_type.define('nir_tex_src_offset', 3)
nir_tex_src_bias = enum_nir_tex_src_type.define('nir_tex_src_bias', 4)
nir_tex_src_lod = enum_nir_tex_src_type.define('nir_tex_src_lod', 5)
nir_tex_src_min_lod = enum_nir_tex_src_type.define('nir_tex_src_min_lod', 6)
nir_tex_src_lod_bias_min_agx = enum_nir_tex_src_type.define('nir_tex_src_lod_bias_min_agx', 7)
nir_tex_src_ms_index = enum_nir_tex_src_type.define('nir_tex_src_ms_index', 8)
nir_tex_src_ms_mcs_intel = enum_nir_tex_src_type.define('nir_tex_src_ms_mcs_intel', 9)
nir_tex_src_ddx = enum_nir_tex_src_type.define('nir_tex_src_ddx', 10)
nir_tex_src_ddy = enum_nir_tex_src_type.define('nir_tex_src_ddy', 11)
nir_tex_src_texture_deref = enum_nir_tex_src_type.define('nir_tex_src_texture_deref', 12)
nir_tex_src_sampler_deref = enum_nir_tex_src_type.define('nir_tex_src_sampler_deref', 13)
nir_tex_src_texture_offset = enum_nir_tex_src_type.define('nir_tex_src_texture_offset', 14)
nir_tex_src_sampler_offset = enum_nir_tex_src_type.define('nir_tex_src_sampler_offset', 15)
nir_tex_src_texture_handle = enum_nir_tex_src_type.define('nir_tex_src_texture_handle', 16)
nir_tex_src_sampler_handle = enum_nir_tex_src_type.define('nir_tex_src_sampler_handle', 17)
nir_tex_src_sampler_deref_intrinsic = enum_nir_tex_src_type.define('nir_tex_src_sampler_deref_intrinsic', 18)
nir_tex_src_texture_deref_intrinsic = enum_nir_tex_src_type.define('nir_tex_src_texture_deref_intrinsic', 19)
nir_tex_src_plane = enum_nir_tex_src_type.define('nir_tex_src_plane', 20)
nir_tex_src_backend1 = enum_nir_tex_src_type.define('nir_tex_src_backend1', 21)
nir_tex_src_backend2 = enum_nir_tex_src_type.define('nir_tex_src_backend2', 22)
nir_num_tex_src_types = enum_nir_tex_src_type.define('nir_num_tex_src_types', 23)

nir_tex_src_type = enum_nir_tex_src_type
class struct_nir_tex_src(Struct): pass
struct_nir_tex_src._fields_ = [
  ('src', nir_src),
  ('src_type', nir_tex_src_type),
]
nir_tex_src = struct_nir_tex_src
enum_nir_texop = CEnum(ctypes.c_uint32)
nir_texop_tex = enum_nir_texop.define('nir_texop_tex', 0)
nir_texop_txb = enum_nir_texop.define('nir_texop_txb', 1)
nir_texop_txl = enum_nir_texop.define('nir_texop_txl', 2)
nir_texop_txd = enum_nir_texop.define('nir_texop_txd', 3)
nir_texop_txf = enum_nir_texop.define('nir_texop_txf', 4)
nir_texop_txf_ms = enum_nir_texop.define('nir_texop_txf_ms', 5)
nir_texop_txf_ms_fb = enum_nir_texop.define('nir_texop_txf_ms_fb', 6)
nir_texop_txf_ms_mcs_intel = enum_nir_texop.define('nir_texop_txf_ms_mcs_intel', 7)
nir_texop_txs = enum_nir_texop.define('nir_texop_txs', 8)
nir_texop_lod = enum_nir_texop.define('nir_texop_lod', 9)
nir_texop_tg4 = enum_nir_texop.define('nir_texop_tg4', 10)
nir_texop_query_levels = enum_nir_texop.define('nir_texop_query_levels', 11)
nir_texop_texture_samples = enum_nir_texop.define('nir_texop_texture_samples', 12)
nir_texop_samples_identical = enum_nir_texop.define('nir_texop_samples_identical', 13)
nir_texop_tex_prefetch = enum_nir_texop.define('nir_texop_tex_prefetch', 14)
nir_texop_lod_bias = enum_nir_texop.define('nir_texop_lod_bias', 15)
nir_texop_fragment_fetch_amd = enum_nir_texop.define('nir_texop_fragment_fetch_amd', 16)
nir_texop_fragment_mask_fetch_amd = enum_nir_texop.define('nir_texop_fragment_mask_fetch_amd', 17)
nir_texop_descriptor_amd = enum_nir_texop.define('nir_texop_descriptor_amd', 18)
nir_texop_sampler_descriptor_amd = enum_nir_texop.define('nir_texop_sampler_descriptor_amd', 19)
nir_texop_image_min_lod_agx = enum_nir_texop.define('nir_texop_image_min_lod_agx', 20)
nir_texop_has_custom_border_color_agx = enum_nir_texop.define('nir_texop_has_custom_border_color_agx', 21)
nir_texop_custom_border_color_agx = enum_nir_texop.define('nir_texop_custom_border_color_agx', 22)
nir_texop_hdr_dim_nv = enum_nir_texop.define('nir_texop_hdr_dim_nv', 23)
nir_texop_tex_type_nv = enum_nir_texop.define('nir_texop_tex_type_nv', 24)

nir_texop = enum_nir_texop
class struct_nir_tex_instr(Struct): pass
enum_glsl_sampler_dim = CEnum(ctypes.c_uint32)
GLSL_SAMPLER_DIM_1D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_1D', 0)
GLSL_SAMPLER_DIM_2D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_2D', 1)
GLSL_SAMPLER_DIM_3D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_3D', 2)
GLSL_SAMPLER_DIM_CUBE = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_CUBE', 3)
GLSL_SAMPLER_DIM_RECT = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_RECT', 4)
GLSL_SAMPLER_DIM_BUF = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_BUF', 5)
GLSL_SAMPLER_DIM_EXTERNAL = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_EXTERNAL', 6)
GLSL_SAMPLER_DIM_MS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_MS', 7)
GLSL_SAMPLER_DIM_SUBPASS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_SUBPASS', 8)
GLSL_SAMPLER_DIM_SUBPASS_MS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_SUBPASS_MS', 9)

struct_nir_tex_instr._fields_ = [
  ('instr', nir_instr),
  ('sampler_dim', enum_glsl_sampler_dim),
  ('dest_type', nir_alu_type),
  ('op', nir_texop),
  ('def', nir_def),
  ('src', ctypes.POINTER(nir_tex_src)),
  ('num_srcs', ctypes.c_uint32),
  ('coord_components', ctypes.c_uint32),
  ('is_array', ctypes.c_bool),
  ('is_shadow', ctypes.c_bool),
  ('is_new_style_shadow', ctypes.c_bool),
  ('is_sparse', ctypes.c_bool),
  ('component', ctypes.c_uint32,2),
  ('array_is_lowered_cube', ctypes.c_uint32,1),
  ('is_gather_implicit_lod', ctypes.c_uint32,1),
  ('skip_helpers', ctypes.c_uint32,1),
  ('tg4_offsets', ((int8_t * 2) * 4)),
  ('texture_non_uniform', ctypes.c_bool),
  ('sampler_non_uniform', ctypes.c_bool),
  ('offset_non_uniform', ctypes.c_bool),
  ('texture_index', ctypes.c_uint32),
  ('sampler_index', ctypes.c_uint32),
  ('backend_flags', uint32_t),
]
nir_tex_instr = struct_nir_tex_instr
try: (nir_tex_instr_need_sampler:=dll.nir_tex_instr_need_sampler).restype, nir_tex_instr_need_sampler.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_result_size:=dll.nir_tex_instr_result_size).restype, nir_tex_instr_result_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_is_query:=dll.nir_tex_instr_is_query).restype, nir_tex_instr_is_query.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_has_implicit_derivative:=dll.nir_tex_instr_has_implicit_derivative).restype, nir_tex_instr_has_implicit_derivative.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_src_type:=dll.nir_tex_instr_src_type).restype, nir_tex_instr_src_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_src_size:=dll.nir_tex_instr_src_size).restype, nir_tex_instr_src_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_add_src:=dll.nir_tex_instr_add_src).restype, nir_tex_instr_add_src.argtypes = None, [ctypes.POINTER(nir_tex_instr), nir_tex_src_type, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_tex_instr_remove_src:=dll.nir_tex_instr_remove_src).restype, nir_tex_instr_remove_src.argtypes = None, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_has_explicit_tg4_offsets:=dll.nir_tex_instr_has_explicit_tg4_offsets).restype, nir_tex_instr_has_explicit_tg4_offsets.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

class struct_nir_load_const_instr(Struct): pass
struct_nir_load_const_instr._fields_ = [
  ('instr', nir_instr),
  ('def', nir_def),
  ('value', (nir_const_value * 0)),
]
nir_load_const_instr = struct_nir_load_const_instr
nir_jump_type = CEnum(ctypes.c_uint32)
nir_jump_return = nir_jump_type.define('nir_jump_return', 0)
nir_jump_halt = nir_jump_type.define('nir_jump_halt', 1)
nir_jump_break = nir_jump_type.define('nir_jump_break', 2)
nir_jump_continue = nir_jump_type.define('nir_jump_continue', 3)
nir_jump_goto = nir_jump_type.define('nir_jump_goto', 4)
nir_jump_goto_if = nir_jump_type.define('nir_jump_goto_if', 5)

class struct_nir_jump_instr(Struct): pass
struct_nir_jump_instr._fields_ = [
  ('instr', nir_instr),
  ('type', nir_jump_type),
  ('condition', nir_src),
  ('target', ctypes.POINTER(nir_block)),
  ('else_target', ctypes.POINTER(nir_block)),
]
nir_jump_instr = struct_nir_jump_instr
class struct_nir_undef_instr(Struct): pass
struct_nir_undef_instr._fields_ = [
  ('instr', nir_instr),
  ('def', nir_def),
]
nir_undef_instr = struct_nir_undef_instr
class struct_nir_phi_src(Struct): pass
struct_nir_phi_src._fields_ = [
  ('node', struct_exec_node),
  ('pred', ctypes.POINTER(nir_block)),
  ('src', nir_src),
]
nir_phi_src = struct_nir_phi_src
class struct_nir_phi_instr(Struct): pass
struct_nir_phi_instr._fields_ = [
  ('instr', nir_instr),
  ('srcs', struct_exec_list),
  ('def', nir_def),
]
nir_phi_instr = struct_nir_phi_instr
class struct_nir_parallel_copy_entry(Struct): pass
class struct_nir_parallel_copy_entry_dest(ctypes.Union): pass
struct_nir_parallel_copy_entry_dest._fields_ = [
  ('def', nir_def),
  ('reg', nir_src),
]
struct_nir_parallel_copy_entry._fields_ = [
  ('node', struct_exec_node),
  ('src_is_reg', ctypes.c_bool),
  ('dest_is_reg', ctypes.c_bool),
  ('src', nir_src),
  ('dest', struct_nir_parallel_copy_entry_dest),
]
nir_parallel_copy_entry = struct_nir_parallel_copy_entry
class struct_nir_parallel_copy_instr(Struct): pass
struct_nir_parallel_copy_instr._fields_ = [
  ('instr', nir_instr),
  ('entries', struct_exec_list),
]
nir_parallel_copy_instr = struct_nir_parallel_copy_instr
class struct_nir_instr_debug_info(Struct): pass
struct_nir_instr_debug_info._fields_ = [
  ('filename', ctypes.POINTER(ctypes.c_char)),
  ('line', uint32_t),
  ('column', uint32_t),
  ('spirv_offset', uint32_t),
  ('nir_line', uint32_t),
  ('variable_name', ctypes.POINTER(ctypes.c_char)),
  ('instr', nir_instr),
]
nir_instr_debug_info = struct_nir_instr_debug_info
class struct_nir_scalar(Struct): pass
struct_nir_scalar._fields_ = [
  ('def', ctypes.POINTER(nir_def)),
  ('comp', ctypes.c_uint32),
]
nir_scalar = struct_nir_scalar
try: (nir_scalar_chase_movs:=dll.nir_scalar_chase_movs).restype, nir_scalar_chase_movs.argtypes = nir_scalar, [nir_scalar]
except AttributeError: pass

class struct_nir_binding(Struct): pass
struct_nir_binding._fields_ = [
  ('success', ctypes.c_bool),
  ('var', ctypes.POINTER(nir_variable)),
  ('desc_set', ctypes.c_uint32),
  ('binding', ctypes.c_uint32),
  ('num_indices', ctypes.c_uint32),
  ('indices', (nir_src * 4)),
  ('read_first_invocation', ctypes.c_bool),
]
nir_binding = struct_nir_binding
try: (nir_chase_binding:=dll.nir_chase_binding).restype, nir_chase_binding.argtypes = nir_binding, [nir_src]
except AttributeError: pass

try: (nir_get_binding_variable:=dll.nir_get_binding_variable).restype, nir_get_binding_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_binding]
except AttributeError: pass

try: (nir_block_contains_work:=dll.nir_block_contains_work).restype, nir_block_contains_work.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

nir_selection_control = CEnum(ctypes.c_uint32)
nir_selection_control_none = nir_selection_control.define('nir_selection_control_none', 0)
nir_selection_control_flatten = nir_selection_control.define('nir_selection_control_flatten', 1)
nir_selection_control_dont_flatten = nir_selection_control.define('nir_selection_control_dont_flatten', 2)
nir_selection_control_divergent_always_taken = nir_selection_control.define('nir_selection_control_divergent_always_taken', 3)

class struct_nir_if(Struct): pass
struct_nir_if._fields_ = [
  ('cf_node', nir_cf_node),
  ('condition', nir_src),
  ('control', nir_selection_control),
  ('then_list', struct_exec_list),
  ('else_list', struct_exec_list),
]
nir_if = struct_nir_if
class struct_nir_loop_terminator(Struct): pass
struct_nir_loop_terminator._fields_ = [
  ('nif', ctypes.POINTER(nir_if)),
  ('conditional_instr', ctypes.POINTER(nir_instr)),
  ('break_block', ctypes.POINTER(nir_block)),
  ('continue_from_block', ctypes.POINTER(nir_block)),
  ('continue_from_then', ctypes.c_bool),
  ('induction_rhs', ctypes.c_bool),
  ('exact_trip_count_unknown', ctypes.c_bool),
  ('loop_terminator_link', struct_list_head),
]
nir_loop_terminator = struct_nir_loop_terminator
class struct_nir_loop_induction_variable(Struct): pass
struct_nir_loop_induction_variable._fields_ = [
  ('basis', ctypes.POINTER(nir_def)),
  ('def', ctypes.POINTER(nir_def)),
  ('init_src', ctypes.POINTER(nir_src)),
  ('update_src', ctypes.POINTER(nir_alu_src)),
]
nir_loop_induction_variable = struct_nir_loop_induction_variable
class struct_nir_loop_info(Struct): pass
class struct_hash_table(Struct): pass
class struct_hash_entry(Struct): pass
struct_hash_entry._fields_ = [
  ('hash', uint32_t),
  ('key', ctypes.c_void_p),
  ('data', ctypes.c_void_p),
]
struct_hash_table._fields_ = [
  ('table', ctypes.POINTER(struct_hash_entry)),
  ('key_hash_function', ctypes.CFUNCTYPE(uint32_t, ctypes.c_void_p)),
  ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)),
  ('deleted_key', ctypes.c_void_p),
  ('size', uint32_t),
  ('rehash', uint32_t),
  ('size_magic', uint64_t),
  ('rehash_magic', uint64_t),
  ('max_entries', uint32_t),
  ('size_index', uint32_t),
  ('entries', uint32_t),
  ('deleted_entries', uint32_t),
]
struct_nir_loop_info._fields_ = [
  ('instr_cost', ctypes.c_uint32),
  ('has_soft_fp64', ctypes.c_bool),
  ('guessed_trip_count', ctypes.c_uint32),
  ('max_trip_count', ctypes.c_uint32),
  ('exact_trip_count_known', ctypes.c_bool),
  ('force_unroll', ctypes.c_bool),
  ('complex_loop', ctypes.c_bool),
  ('limiting_terminator', ctypes.POINTER(nir_loop_terminator)),
  ('loop_terminator_list', struct_list_head),
  ('induction_vars', ctypes.POINTER(struct_hash_table)),
]
nir_loop_info = struct_nir_loop_info
nir_loop_control = CEnum(ctypes.c_uint32)
nir_loop_control_none = nir_loop_control.define('nir_loop_control_none', 0)
nir_loop_control_unroll = nir_loop_control.define('nir_loop_control_unroll', 1)
nir_loop_control_dont_unroll = nir_loop_control.define('nir_loop_control_dont_unroll', 2)

class struct_nir_loop(Struct): pass
struct_nir_loop._fields_ = [
  ('cf_node', nir_cf_node),
  ('body', struct_exec_list),
  ('continue_list', struct_exec_list),
  ('info', ctypes.POINTER(nir_loop_info)),
  ('control', nir_loop_control),
  ('partially_unrolled', ctypes.c_bool),
  ('divergent_continue', ctypes.c_bool),
  ('divergent_break', ctypes.c_bool),
]
nir_loop = struct_nir_loop
nir_intrin_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
nir_vectorize_cb = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
try: (nir_remove_non_entrypoints:=dll.nir_remove_non_entrypoints).restype, nir_remove_non_entrypoints.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_non_exported:=dll.nir_remove_non_exported).restype, nir_remove_non_exported.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_entrypoints:=dll.nir_remove_entrypoints).restype, nir_remove_entrypoints.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_is_exported:=dll.nir_fixup_is_exported).restype, nir_fixup_is_exported.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

shader_info = struct_shader_info
try: (nir_shader_create:=dll.nir_shader_create).restype, nir_shader_create.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, gl_shader_stage, ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(shader_info)]
except AttributeError: pass

try: (nir_shader_add_variable:=dll.nir_shader_add_variable).restype, nir_shader_add_variable.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_variable)]
except AttributeError: pass

try: (nir_variable_create:=dll.nir_variable_create).restype, nir_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_local_variable_create:=dll.nir_local_variable_create).restype, nir_local_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_state_variable_create:=dll.nir_state_variable_create).restype, nir_state_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char), (gl_state_index16 * 4)]
except AttributeError: pass

try: (nir_get_variable_with_location:=dll.nir_get_variable_with_location).restype, nir_get_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError: pass

try: (nir_create_variable_with_location:=dll.nir_create_variable_with_location).restype, nir_create_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError: pass

try: (nir_find_variable_with_location:=dll.nir_find_variable_with_location).restype, nir_find_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError: pass

try: (nir_find_variable_with_driver_location:=dll.nir_find_variable_with_driver_location).restype, nir_find_variable_with_driver_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError: pass

try: (nir_find_state_variable:=dll.nir_find_state_variable).restype, nir_find_state_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), (gl_state_index16 * 4)]
except AttributeError: pass

try: (nir_find_sampler_variable_with_tex_index:=dll.nir_find_sampler_variable_with_tex_index).restype, nir_find_sampler_variable_with_tex_index.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_sort_variables_with_modes:=dll.nir_sort_variables_with_modes).restype, nir_sort_variables_with_modes.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(nir_variable), ctypes.POINTER(nir_variable)), nir_variable_mode]
except AttributeError: pass

try: (nir_function_create:=dll.nir_function_create).restype, nir_function_create.argtypes = ctypes.POINTER(nir_function), [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_function_impl_create:=dll.nir_function_impl_create).restype, nir_function_impl_create.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_function_impl_create_bare:=dll.nir_function_impl_create_bare).restype, nir_function_impl_create_bare.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_block_create:=dll.nir_block_create).restype, nir_block_create.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_if_create:=dll.nir_if_create).restype, nir_if_create.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_loop_create:=dll.nir_loop_create).restype, nir_loop_create.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_cf_node_get_function:=dll.nir_cf_node_get_function).restype, nir_cf_node_get_function.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_metadata_require:=dll.nir_metadata_require).restype, nir_metadata_require.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_metadata]
except AttributeError: pass

try: (nir_shader_preserve_all_metadata:=dll.nir_shader_preserve_all_metadata).restype, nir_shader_preserve_all_metadata.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_invalidate:=dll.nir_metadata_invalidate).restype, nir_metadata_invalidate.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_progress:=dll.nir_progress).restype, nir_progress.argtypes = ctypes.c_bool, [ctypes.c_bool, ctypes.POINTER(nir_function_impl), nir_metadata]
except AttributeError: pass

try: (nir_alu_instr_create:=dll.nir_alu_instr_create).restype, nir_alu_instr_create.argtypes = ctypes.POINTER(nir_alu_instr), [ctypes.POINTER(nir_shader), nir_op]
except AttributeError: pass

try: (nir_deref_instr_create:=dll.nir_deref_instr_create).restype, nir_deref_instr_create.argtypes = ctypes.POINTER(nir_deref_instr), [ctypes.POINTER(nir_shader), nir_deref_type]
except AttributeError: pass

try: (nir_jump_instr_create:=dll.nir_jump_instr_create).restype, nir_jump_instr_create.argtypes = ctypes.POINTER(nir_jump_instr), [ctypes.POINTER(nir_shader), nir_jump_type]
except AttributeError: pass

try: (nir_load_const_instr_create:=dll.nir_load_const_instr_create).restype, nir_load_const_instr_create.argtypes = ctypes.POINTER(nir_load_const_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_instr_create:=dll.nir_intrinsic_instr_create).restype, nir_intrinsic_instr_create.argtypes = ctypes.POINTER(nir_intrinsic_instr), [ctypes.POINTER(nir_shader), nir_intrinsic_op]
except AttributeError: pass

try: (nir_call_instr_create:=dll.nir_call_instr_create).restype, nir_call_instr_create.argtypes = ctypes.POINTER(nir_call_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_tex_instr_create:=dll.nir_tex_instr_create).restype, nir_tex_instr_create.argtypes = ctypes.POINTER(nir_tex_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_phi_instr_create:=dll.nir_phi_instr_create).restype, nir_phi_instr_create.argtypes = ctypes.POINTER(nir_phi_instr), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_phi_instr_add_src:=dll.nir_phi_instr_add_src).restype, nir_phi_instr_add_src.argtypes = ctypes.POINTER(nir_phi_src), [ctypes.POINTER(nir_phi_instr), ctypes.POINTER(nir_block), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_parallel_copy_instr_create:=dll.nir_parallel_copy_instr_create).restype, nir_parallel_copy_instr_create.argtypes = ctypes.POINTER(nir_parallel_copy_instr), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_undef_instr_create:=dll.nir_undef_instr_create).restype, nir_undef_instr_create.argtypes = ctypes.POINTER(nir_undef_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_binop_identity:=dll.nir_alu_binop_identity).restype, nir_alu_binop_identity.argtypes = nir_const_value, [nir_op, ctypes.c_uint32]
except AttributeError: pass

nir_cursor_option = CEnum(ctypes.c_uint32)
nir_cursor_before_block = nir_cursor_option.define('nir_cursor_before_block', 0)
nir_cursor_after_block = nir_cursor_option.define('nir_cursor_after_block', 1)
nir_cursor_before_instr = nir_cursor_option.define('nir_cursor_before_instr', 2)
nir_cursor_after_instr = nir_cursor_option.define('nir_cursor_after_instr', 3)

class struct_nir_cursor(Struct): pass
class struct_nir_cursor_0(ctypes.Union): pass
struct_nir_cursor_0._fields_ = [
  ('block', ctypes.POINTER(nir_block)),
  ('instr', ctypes.POINTER(nir_instr)),
]
struct_nir_cursor._anonymous_ = ['_0']
struct_nir_cursor._fields_ = [
  ('option', nir_cursor_option),
  ('_0', struct_nir_cursor_0),
]
nir_cursor = struct_nir_cursor
try: (nir_cursors_equal:=dll.nir_cursors_equal).restype, nir_cursors_equal.argtypes = ctypes.c_bool, [nir_cursor, nir_cursor]
except AttributeError: pass

try: (nir_instr_insert:=dll.nir_instr_insert).restype, nir_instr_insert.argtypes = None, [nir_cursor, ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_move:=dll.nir_instr_move).restype, nir_instr_move.argtypes = ctypes.c_bool, [nir_cursor, ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_remove_v:=dll.nir_instr_remove_v).restype, nir_instr_remove_v.argtypes = None, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_free:=dll.nir_instr_free).restype, nir_instr_free.argtypes = None, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_free_list:=dll.nir_instr_free_list).restype, nir_instr_free_list.argtypes = None, [ctypes.POINTER(struct_exec_list)]
except AttributeError: pass

try: (nir_instr_free_and_dce:=dll.nir_instr_free_and_dce).restype, nir_instr_free_and_dce.argtypes = nir_cursor, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_def:=dll.nir_instr_def).restype, nir_instr_def.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_instr)]
except AttributeError: pass

nir_foreach_def_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_def), ctypes.c_void_p)
nir_foreach_src_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_src), ctypes.c_void_p)
try: (nir_foreach_phi_src_leaving_block:=dll.nir_foreach_phi_src_leaving_block).restype, nir_foreach_phi_src_leaving_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), nir_foreach_src_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_src_as_const_value:=dll.nir_src_as_const_value).restype, nir_src_as_const_value.argtypes = ctypes.POINTER(nir_const_value), [nir_src]
except AttributeError: pass

try: (nir_src_as_string:=dll.nir_src_as_string).restype, nir_src_as_string.argtypes = ctypes.POINTER(ctypes.c_char), [nir_src]
except AttributeError: pass

try: (nir_src_is_always_uniform:=dll.nir_src_is_always_uniform).restype, nir_src_is_always_uniform.argtypes = ctypes.c_bool, [nir_src]
except AttributeError: pass

try: (nir_srcs_equal:=dll.nir_srcs_equal).restype, nir_srcs_equal.argtypes = ctypes.c_bool, [nir_src, nir_src]
except AttributeError: pass

try: (nir_instrs_equal:=dll.nir_instrs_equal).restype, nir_instrs_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_src_get_block:=dll.nir_src_get_block).restype, nir_src_get_block.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_init_src:=dll.nir_instr_init_src).restype, nir_instr_init_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_instr_clear_src:=dll.nir_instr_clear_src).restype, nir_instr_clear_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_move_src:=dll.nir_instr_move_src).restype, nir_instr_move_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src), ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_is_before:=dll.nir_instr_is_before).restype, nir_instr_is_before.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_def_init:=dll.nir_def_init).restype, nir_def_init.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_def_rewrite_uses:=dll.nir_def_rewrite_uses).restype, nir_def_rewrite_uses.argtypes = None, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_rewrite_uses_src:=dll.nir_def_rewrite_uses_src).restype, nir_def_rewrite_uses_src.argtypes = None, [ctypes.POINTER(nir_def), nir_src]
except AttributeError: pass

try: (nir_def_rewrite_uses_after:=dll.nir_def_rewrite_uses_after).restype, nir_def_rewrite_uses_after.argtypes = None, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_src_components_read:=dll.nir_src_components_read).restype, nir_src_components_read.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_def_components_read:=dll.nir_def_components_read).restype, nir_def_components_read.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_all_uses_are_fsat:=dll.nir_def_all_uses_are_fsat).restype, nir_def_all_uses_are_fsat.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_all_uses_ignore_sign_bit:=dll.nir_def_all_uses_ignore_sign_bit).restype, nir_def_all_uses_ignore_sign_bit.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_sort_unstructured_blocks:=dll.nir_sort_unstructured_blocks).restype, nir_sort_unstructured_blocks.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_block_unstructured_next:=dll.nir_block_unstructured_next).restype, nir_block_unstructured_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_unstructured_start_block:=dll.nir_unstructured_start_block).restype, nir_unstructured_start_block.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_block_cf_tree_next:=dll.nir_block_cf_tree_next).restype, nir_block_cf_tree_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_cf_tree_prev:=dll.nir_block_cf_tree_prev).restype, nir_block_cf_tree_prev.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_first:=dll.nir_cf_node_cf_tree_first).restype, nir_cf_node_cf_tree_first.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_last:=dll.nir_cf_node_cf_tree_last).restype, nir_cf_node_cf_tree_last.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_next:=dll.nir_cf_node_cf_tree_next).restype, nir_cf_node_cf_tree_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_prev:=dll.nir_cf_node_cf_tree_prev).restype, nir_cf_node_cf_tree_prev.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_block_get_following_if:=dll.nir_block_get_following_if).restype, nir_block_get_following_if.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_get_following_loop:=dll.nir_block_get_following_loop).restype, nir_block_get_following_loop.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_get_predecessors_sorted:=dll.nir_block_get_predecessors_sorted).restype, nir_block_get_predecessors_sorted.argtypes = ctypes.POINTER(ctypes.POINTER(nir_block)), [ctypes.POINTER(nir_block), ctypes.c_void_p]
except AttributeError: pass

try: (nir_index_ssa_defs:=dll.nir_index_ssa_defs).restype, nir_index_ssa_defs.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_index_instrs:=dll.nir_index_instrs).restype, nir_index_instrs.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_index_blocks:=dll.nir_index_blocks).restype, nir_index_blocks.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_shader_clear_pass_flags:=dll.nir_shader_clear_pass_flags).restype, nir_shader_clear_pass_flags.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_index_vars:=dll.nir_shader_index_vars).restype, nir_shader_index_vars.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_function_impl_index_vars:=dll.nir_function_impl_index_vars).restype, nir_function_impl_index_vars.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

class struct__IO_FILE(Struct): pass
FILE = struct__IO_FILE
class struct__IO_marker(Struct): pass
__off_t = ctypes.c_int64
_IO_lock_t = None
__off64_t = ctypes.c_int64
class struct__IO_codecvt(Struct): pass
class struct__IO_wide_data(Struct): pass
size_t = ctypes.c_uint64
struct__IO_FILE._fields_ = [
  ('_flags', ctypes.c_int32),
  ('_IO_read_ptr', ctypes.POINTER(ctypes.c_char)),
  ('_IO_read_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_read_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_ptr', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_buf_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_buf_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_save_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_backup_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_save_end', ctypes.POINTER(ctypes.c_char)),
  ('_markers', ctypes.POINTER(struct__IO_marker)),
  ('_chain', ctypes.POINTER(struct__IO_FILE)),
  ('_fileno', ctypes.c_int32),
  ('_flags2', ctypes.c_int32),
  ('_old_offset', ctypes.c_int64),
  ('_cur_column', ctypes.c_uint16),
  ('_vtable_offset', ctypes.c_byte),
  ('_shortbuf', (ctypes.c_char * 1)),
  ('_lock', ctypes.POINTER(_IO_lock_t)),
  ('_offset', ctypes.c_int64),
  ('_codecvt', ctypes.POINTER(struct__IO_codecvt)),
  ('_wide_data', ctypes.POINTER(struct__IO_wide_data)),
  ('_freeres_list', ctypes.POINTER(struct__IO_FILE)),
  ('_freeres_buf', ctypes.c_void_p),
  ('__pad5', size_t),
  ('_mode', ctypes.c_int32),
  ('_unused2', (ctypes.c_char * 20)),
]
try: (nir_print_shader:=dll.nir_print_shader).restype, nir_print_shader.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_function_body:=dll.nir_print_function_body).restype, nir_print_function_body.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_shader_annotated:=dll.nir_print_shader_annotated).restype, nir_print_shader_annotated.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_print_instr:=dll.nir_print_instr).restype, nir_print_instr.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_deref:=dll.nir_print_deref).restype, nir_print_deref.argtypes = None, [ctypes.POINTER(nir_deref_instr), ctypes.POINTER(FILE)]
except AttributeError: pass

enum_mesa_log_level = CEnum(ctypes.c_uint32)
MESA_LOG_ERROR = enum_mesa_log_level.define('MESA_LOG_ERROR', 0)
MESA_LOG_WARN = enum_mesa_log_level.define('MESA_LOG_WARN', 1)
MESA_LOG_INFO = enum_mesa_log_level.define('MESA_LOG_INFO', 2)
MESA_LOG_DEBUG = enum_mesa_log_level.define('MESA_LOG_DEBUG', 3)

try: (nir_log_shader_annotated_tagged:=dll.nir_log_shader_annotated_tagged).restype, nir_log_shader_annotated_tagged.argtypes = None, [enum_mesa_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_shader_as_str:=dll.nir_shader_as_str).restype, nir_shader_as_str.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_as_str_annotated:=dll.nir_shader_as_str_annotated).restype, nir_shader_as_str_annotated.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), ctypes.c_void_p]
except AttributeError: pass

try: (nir_instr_as_str:=dll.nir_instr_as_str).restype, nir_instr_as_str.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_instr), ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_gather_debug_info:=dll.nir_shader_gather_debug_info).restype, nir_shader_gather_debug_info.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char), uint32_t]
except AttributeError: pass

try: (nir_instr_clone:=dll.nir_instr_clone).restype, nir_instr_clone.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_clone_deep:=dll.nir_instr_clone_deep).restype, nir_instr_clone_deep.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_instr), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_alu_instr_clone:=dll.nir_alu_instr_clone).restype, nir_alu_instr_clone.argtypes = ctypes.POINTER(nir_alu_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_shader_clone:=dll.nir_shader_clone).restype, nir_shader_clone.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_function_clone:=dll.nir_function_clone).restype, nir_function_clone.argtypes = ctypes.POINTER(nir_function), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_function_impl_clone:=dll.nir_function_impl_clone).restype, nir_function_impl_clone.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_function_impl_clone_remap_globals:=dll.nir_function_impl_clone_remap_globals).restype, nir_function_impl_clone_remap_globals.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_constant_clone:=dll.nir_constant_clone).restype, nir_constant_clone.argtypes = ctypes.POINTER(nir_constant), [ctypes.POINTER(nir_constant), ctypes.POINTER(nir_variable)]
except AttributeError: pass

try: (nir_variable_clone:=dll.nir_variable_clone).restype, nir_variable_clone.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_variable), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_replace:=dll.nir_shader_replace).restype, nir_shader_replace.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_serialize_deserialize:=dll.nir_shader_serialize_deserialize).restype, nir_shader_serialize_deserialize.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_validate_shader:=dll.nir_validate_shader).restype, nir_validate_shader.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_validate_ssa_dominance:=dll.nir_validate_ssa_dominance).restype, nir_validate_ssa_dominance.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_metadata_set_validation_flag:=dll.nir_metadata_set_validation_flag).restype, nir_metadata_set_validation_flag.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_check_validation_flag:=dll.nir_metadata_check_validation_flag).restype, nir_metadata_check_validation_flag.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_require_all:=dll.nir_metadata_require_all).restype, nir_metadata_require_all.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_instr_writemask_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_uint32, ctypes.c_void_p)
class struct_nir_builder(Struct): pass
nir_lower_instr_cb = ctypes.CFUNCTYPE(ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
try: (nir_function_impl_lower_instructions:=dll.nir_function_impl_lower_instructions).restype, nir_function_impl_lower_instructions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_lower_instructions:=dll.nir_shader_lower_instructions).restype, nir_shader_lower_instructions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_calc_dominance_impl:=dll.nir_calc_dominance_impl).restype, nir_calc_dominance_impl.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_calc_dominance:=dll.nir_calc_dominance).restype, nir_calc_dominance.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_dominance_lca:=dll.nir_dominance_lca).restype, nir_dominance_lca.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block), ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_dominates:=dll.nir_block_dominates).restype, nir_block_dominates.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_is_unreachable:=dll.nir_block_is_unreachable).restype, nir_block_is_unreachable.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_dump_dom_tree_impl:=dll.nir_dump_dom_tree_impl).restype, nir_dump_dom_tree_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_tree:=dll.nir_dump_dom_tree).restype, nir_dump_dom_tree.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_frontier_impl:=dll.nir_dump_dom_frontier_impl).restype, nir_dump_dom_frontier_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_frontier:=dll.nir_dump_dom_frontier).restype, nir_dump_dom_frontier.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_cfg_impl:=dll.nir_dump_cfg_impl).restype, nir_dump_cfg_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_cfg:=dll.nir_dump_cfg).restype, nir_dump_cfg.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_gs_count_vertices_and_primitives:=dll.nir_gs_count_vertices_and_primitives).restype, nir_gs_count_vertices_and_primitives.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError: pass

nir_load_grouping = CEnum(ctypes.c_uint32)
nir_group_all = nir_load_grouping.define('nir_group_all', 0)
nir_group_same_resource_only = nir_load_grouping.define('nir_group_same_resource_only', 1)

try: (nir_group_loads:=dll.nir_group_loads).restype, nir_group_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_load_grouping, ctypes.c_uint32]
except AttributeError: pass

try: (nir_shrink_vec_array_vars:=dll.nir_shrink_vec_array_vars).restype, nir_shrink_vec_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_split_array_vars:=dll.nir_split_array_vars).restype, nir_split_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_split_var_copies:=dll.nir_split_var_copies).restype, nir_split_var_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_split_per_member_structs:=dll.nir_split_per_member_structs).restype, nir_split_per_member_structs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_split_struct_vars:=dll.nir_split_struct_vars).restype, nir_split_struct_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_returns_impl:=dll.nir_lower_returns_impl).restype, nir_lower_returns_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_returns:=dll.nir_lower_returns).restype, nir_lower_returns.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_builder = struct_nir_builder
try: (nir_inline_function_impl:=dll.nir_inline_function_impl).restype, nir_inline_function_impl.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_function_impl), ctypes.POINTER(ctypes.POINTER(nir_def)), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_inline_functions:=dll.nir_inline_functions).restype, nir_inline_functions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_cleanup_functions:=dll.nir_cleanup_functions).restype, nir_cleanup_functions.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_shader_functions:=dll.nir_link_shader_functions).restype, nir_link_shader_functions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_calls_to_builtins:=dll.nir_lower_calls_to_builtins).restype, nir_lower_calls_to_builtins.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_find_inlinable_uniforms:=dll.nir_find_inlinable_uniforms).restype, nir_find_inlinable_uniforms.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_inline_uniforms:=dll.nir_inline_uniforms).restype, nir_inline_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(uint32_t), ctypes.POINTER(uint16_t)]
except AttributeError: pass

try: (nir_collect_src_uniforms:=dll.nir_collect_src_uniforms).restype, nir_collect_src_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_src), ctypes.c_int32, ctypes.POINTER(uint32_t), ctypes.POINTER(uint8_t), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_add_inlinable_uniforms:=dll.nir_add_inlinable_uniforms).restype, nir_add_inlinable_uniforms.argtypes = None, [ctypes.POINTER(nir_src), ctypes.POINTER(nir_loop_info), ctypes.POINTER(uint32_t), ctypes.POINTER(uint8_t), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_propagate_invariant:=dll.nir_propagate_invariant).restype, nir_propagate_invariant.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_var_copy_instr:=dll.nir_lower_var_copy_instr).restype, nir_lower_var_copy_instr.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_deref_copy_instr:=dll.nir_lower_deref_copy_instr).restype, nir_lower_deref_copy_instr.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_lower_var_copies:=dll.nir_lower_var_copies).restype, nir_lower_var_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_memcpy:=dll.nir_opt_memcpy).restype, nir_opt_memcpy.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_memcpy:=dll.nir_lower_memcpy).restype, nir_lower_memcpy.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_deref_modes:=dll.nir_fixup_deref_modes).restype, nir_fixup_deref_modes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_deref_types:=dll.nir_fixup_deref_types).restype, nir_fixup_deref_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_global_vars_to_local:=dll.nir_lower_global_vars_to_local).restype, nir_lower_global_vars_to_local.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_constant_to_temp:=dll.nir_lower_constant_to_temp).restype, nir_lower_constant_to_temp.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_array_deref_of_vec_options = CEnum(ctypes.c_uint32)
nir_lower_direct_array_deref_of_vec_load = nir_lower_array_deref_of_vec_options.define('nir_lower_direct_array_deref_of_vec_load', 1)
nir_lower_indirect_array_deref_of_vec_load = nir_lower_array_deref_of_vec_options.define('nir_lower_indirect_array_deref_of_vec_load', 2)
nir_lower_direct_array_deref_of_vec_store = nir_lower_array_deref_of_vec_options.define('nir_lower_direct_array_deref_of_vec_store', 4)
nir_lower_indirect_array_deref_of_vec_store = nir_lower_array_deref_of_vec_options.define('nir_lower_indirect_array_deref_of_vec_store', 8)

try: (nir_lower_array_deref_of_vec:=dll.nir_lower_array_deref_of_vec).restype, nir_lower_array_deref_of_vec.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_variable)), nir_lower_array_deref_of_vec_options]
except AttributeError: pass

try: (nir_lower_indirect_derefs:=dll.nir_lower_indirect_derefs).restype, nir_lower_indirect_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, uint32_t]
except AttributeError: pass

try: (nir_lower_indirect_var_derefs:=dll.nir_lower_indirect_var_derefs).restype, nir_lower_indirect_var_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_set)]
except AttributeError: pass

try: (nir_lower_locals_to_regs:=dll.nir_lower_locals_to_regs).restype, nir_lower_locals_to_regs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint8_t]
except AttributeError: pass

try: (nir_lower_io_vars_to_temporaries:=dll.nir_lower_io_vars_to_temporaries).restype, nir_lower_io_vars_to_temporaries.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

glsl_type_size_align_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
try: (nir_lower_vars_to_scratch:=dll.nir_lower_vars_to_scratch).restype, nir_lower_vars_to_scratch.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, glsl_type_size_align_func, glsl_type_size_align_func]
except AttributeError: pass

try: (nir_lower_scratch_to_var:=dll.nir_lower_scratch_to_var).restype, nir_lower_scratch_to_var.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_halfz:=dll.nir_lower_clip_halfz).restype, nir_lower_clip_halfz.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_gather_info:=dll.nir_shader_gather_info).restype, nir_shader_gather_info.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_gather_types:=dll.nir_gather_types).restype, nir_gather_types.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_remove_unused_varyings:=dll.nir_remove_unused_varyings).restype, nir_remove_unused_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_unused_io_vars:=dll.nir_remove_unused_io_vars).restype, nir_remove_unused_io_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(uint64_t), ctypes.POINTER(uint64_t)]
except AttributeError: pass

try: (nir_compact_varyings:=dll.nir_compact_varyings).restype, nir_compact_varyings.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_link_xfb_varyings:=dll.nir_link_xfb_varyings).restype, nir_link_xfb_varyings.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_opt_varyings:=dll.nir_link_opt_varyings).restype, nir_link_opt_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_varying_precision:=dll.nir_link_varying_precision).restype, nir_link_varying_precision.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_clone_uniform_variable:=dll.nir_clone_uniform_variable).restype, nir_clone_uniform_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_variable), ctypes.c_bool]
except AttributeError: pass

try: (nir_clone_deref_instr:=dll.nir_clone_deref_instr).restype, nir_clone_deref_instr.argtypes = ctypes.POINTER(nir_deref_instr), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_variable), ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

nir_opt_varyings_progress = CEnum(ctypes.c_uint32)
nir_progress_producer = nir_opt_varyings_progress.define('nir_progress_producer', 1)
nir_progress_consumer = nir_opt_varyings_progress.define('nir_progress_consumer', 2)

try: (nir_opt_varyings:=dll.nir_opt_varyings).restype, nir_opt_varyings.argtypes = nir_opt_varyings_progress, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

gl_varying_slot = CEnum(ctypes.c_uint32)
VARYING_SLOT_POS = gl_varying_slot.define('VARYING_SLOT_POS', 0)
VARYING_SLOT_COL0 = gl_varying_slot.define('VARYING_SLOT_COL0', 1)
VARYING_SLOT_COL1 = gl_varying_slot.define('VARYING_SLOT_COL1', 2)
VARYING_SLOT_FOGC = gl_varying_slot.define('VARYING_SLOT_FOGC', 3)
VARYING_SLOT_TEX0 = gl_varying_slot.define('VARYING_SLOT_TEX0', 4)
VARYING_SLOT_TEX1 = gl_varying_slot.define('VARYING_SLOT_TEX1', 5)
VARYING_SLOT_TEX2 = gl_varying_slot.define('VARYING_SLOT_TEX2', 6)
VARYING_SLOT_TEX3 = gl_varying_slot.define('VARYING_SLOT_TEX3', 7)
VARYING_SLOT_TEX4 = gl_varying_slot.define('VARYING_SLOT_TEX4', 8)
VARYING_SLOT_TEX5 = gl_varying_slot.define('VARYING_SLOT_TEX5', 9)
VARYING_SLOT_TEX6 = gl_varying_slot.define('VARYING_SLOT_TEX6', 10)
VARYING_SLOT_TEX7 = gl_varying_slot.define('VARYING_SLOT_TEX7', 11)
VARYING_SLOT_PSIZ = gl_varying_slot.define('VARYING_SLOT_PSIZ', 12)
VARYING_SLOT_BFC0 = gl_varying_slot.define('VARYING_SLOT_BFC0', 13)
VARYING_SLOT_BFC1 = gl_varying_slot.define('VARYING_SLOT_BFC1', 14)
VARYING_SLOT_EDGE = gl_varying_slot.define('VARYING_SLOT_EDGE', 15)
VARYING_SLOT_CLIP_VERTEX = gl_varying_slot.define('VARYING_SLOT_CLIP_VERTEX', 16)
VARYING_SLOT_CLIP_DIST0 = gl_varying_slot.define('VARYING_SLOT_CLIP_DIST0', 17)
VARYING_SLOT_CLIP_DIST1 = gl_varying_slot.define('VARYING_SLOT_CLIP_DIST1', 18)
VARYING_SLOT_CULL_DIST0 = gl_varying_slot.define('VARYING_SLOT_CULL_DIST0', 19)
VARYING_SLOT_CULL_DIST1 = gl_varying_slot.define('VARYING_SLOT_CULL_DIST1', 20)
VARYING_SLOT_PRIMITIVE_ID = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_ID', 21)
VARYING_SLOT_LAYER = gl_varying_slot.define('VARYING_SLOT_LAYER', 22)
VARYING_SLOT_VIEWPORT = gl_varying_slot.define('VARYING_SLOT_VIEWPORT', 23)
VARYING_SLOT_FACE = gl_varying_slot.define('VARYING_SLOT_FACE', 24)
VARYING_SLOT_PNTC = gl_varying_slot.define('VARYING_SLOT_PNTC', 25)
VARYING_SLOT_TESS_LEVEL_OUTER = gl_varying_slot.define('VARYING_SLOT_TESS_LEVEL_OUTER', 26)
VARYING_SLOT_TESS_LEVEL_INNER = gl_varying_slot.define('VARYING_SLOT_TESS_LEVEL_INNER', 27)
VARYING_SLOT_BOUNDING_BOX0 = gl_varying_slot.define('VARYING_SLOT_BOUNDING_BOX0', 28)
VARYING_SLOT_BOUNDING_BOX1 = gl_varying_slot.define('VARYING_SLOT_BOUNDING_BOX1', 29)
VARYING_SLOT_VIEW_INDEX = gl_varying_slot.define('VARYING_SLOT_VIEW_INDEX', 30)
VARYING_SLOT_VIEWPORT_MASK = gl_varying_slot.define('VARYING_SLOT_VIEWPORT_MASK', 31)
VARYING_SLOT_PRIMITIVE_SHADING_RATE = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_SHADING_RATE', 24)
VARYING_SLOT_PRIMITIVE_COUNT = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_COUNT', 26)
VARYING_SLOT_PRIMITIVE_INDICES = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_INDICES', 27)
VARYING_SLOT_TASK_COUNT = gl_varying_slot.define('VARYING_SLOT_TASK_COUNT', 28)
VARYING_SLOT_CULL_PRIMITIVE = gl_varying_slot.define('VARYING_SLOT_CULL_PRIMITIVE', 28)
VARYING_SLOT_VAR0 = gl_varying_slot.define('VARYING_SLOT_VAR0', 32)
VARYING_SLOT_VAR1 = gl_varying_slot.define('VARYING_SLOT_VAR1', 33)
VARYING_SLOT_VAR2 = gl_varying_slot.define('VARYING_SLOT_VAR2', 34)
VARYING_SLOT_VAR3 = gl_varying_slot.define('VARYING_SLOT_VAR3', 35)
VARYING_SLOT_VAR4 = gl_varying_slot.define('VARYING_SLOT_VAR4', 36)
VARYING_SLOT_VAR5 = gl_varying_slot.define('VARYING_SLOT_VAR5', 37)
VARYING_SLOT_VAR6 = gl_varying_slot.define('VARYING_SLOT_VAR6', 38)
VARYING_SLOT_VAR7 = gl_varying_slot.define('VARYING_SLOT_VAR7', 39)
VARYING_SLOT_VAR8 = gl_varying_slot.define('VARYING_SLOT_VAR8', 40)
VARYING_SLOT_VAR9 = gl_varying_slot.define('VARYING_SLOT_VAR9', 41)
VARYING_SLOT_VAR10 = gl_varying_slot.define('VARYING_SLOT_VAR10', 42)
VARYING_SLOT_VAR11 = gl_varying_slot.define('VARYING_SLOT_VAR11', 43)
VARYING_SLOT_VAR12 = gl_varying_slot.define('VARYING_SLOT_VAR12', 44)
VARYING_SLOT_VAR13 = gl_varying_slot.define('VARYING_SLOT_VAR13', 45)
VARYING_SLOT_VAR14 = gl_varying_slot.define('VARYING_SLOT_VAR14', 46)
VARYING_SLOT_VAR15 = gl_varying_slot.define('VARYING_SLOT_VAR15', 47)
VARYING_SLOT_VAR16 = gl_varying_slot.define('VARYING_SLOT_VAR16', 48)
VARYING_SLOT_VAR17 = gl_varying_slot.define('VARYING_SLOT_VAR17', 49)
VARYING_SLOT_VAR18 = gl_varying_slot.define('VARYING_SLOT_VAR18', 50)
VARYING_SLOT_VAR19 = gl_varying_slot.define('VARYING_SLOT_VAR19', 51)
VARYING_SLOT_VAR20 = gl_varying_slot.define('VARYING_SLOT_VAR20', 52)
VARYING_SLOT_VAR21 = gl_varying_slot.define('VARYING_SLOT_VAR21', 53)
VARYING_SLOT_VAR22 = gl_varying_slot.define('VARYING_SLOT_VAR22', 54)
VARYING_SLOT_VAR23 = gl_varying_slot.define('VARYING_SLOT_VAR23', 55)
VARYING_SLOT_VAR24 = gl_varying_slot.define('VARYING_SLOT_VAR24', 56)
VARYING_SLOT_VAR25 = gl_varying_slot.define('VARYING_SLOT_VAR25', 57)
VARYING_SLOT_VAR26 = gl_varying_slot.define('VARYING_SLOT_VAR26', 58)
VARYING_SLOT_VAR27 = gl_varying_slot.define('VARYING_SLOT_VAR27', 59)
VARYING_SLOT_VAR28 = gl_varying_slot.define('VARYING_SLOT_VAR28', 60)
VARYING_SLOT_VAR29 = gl_varying_slot.define('VARYING_SLOT_VAR29', 61)
VARYING_SLOT_VAR30 = gl_varying_slot.define('VARYING_SLOT_VAR30', 62)
VARYING_SLOT_VAR31 = gl_varying_slot.define('VARYING_SLOT_VAR31', 63)
VARYING_SLOT_PATCH0 = gl_varying_slot.define('VARYING_SLOT_PATCH0', 64)
VARYING_SLOT_PATCH1 = gl_varying_slot.define('VARYING_SLOT_PATCH1', 65)
VARYING_SLOT_PATCH2 = gl_varying_slot.define('VARYING_SLOT_PATCH2', 66)
VARYING_SLOT_PATCH3 = gl_varying_slot.define('VARYING_SLOT_PATCH3', 67)
VARYING_SLOT_PATCH4 = gl_varying_slot.define('VARYING_SLOT_PATCH4', 68)
VARYING_SLOT_PATCH5 = gl_varying_slot.define('VARYING_SLOT_PATCH5', 69)
VARYING_SLOT_PATCH6 = gl_varying_slot.define('VARYING_SLOT_PATCH6', 70)
VARYING_SLOT_PATCH7 = gl_varying_slot.define('VARYING_SLOT_PATCH7', 71)
VARYING_SLOT_PATCH8 = gl_varying_slot.define('VARYING_SLOT_PATCH8', 72)
VARYING_SLOT_PATCH9 = gl_varying_slot.define('VARYING_SLOT_PATCH9', 73)
VARYING_SLOT_PATCH10 = gl_varying_slot.define('VARYING_SLOT_PATCH10', 74)
VARYING_SLOT_PATCH11 = gl_varying_slot.define('VARYING_SLOT_PATCH11', 75)
VARYING_SLOT_PATCH12 = gl_varying_slot.define('VARYING_SLOT_PATCH12', 76)
VARYING_SLOT_PATCH13 = gl_varying_slot.define('VARYING_SLOT_PATCH13', 77)
VARYING_SLOT_PATCH14 = gl_varying_slot.define('VARYING_SLOT_PATCH14', 78)
VARYING_SLOT_PATCH15 = gl_varying_slot.define('VARYING_SLOT_PATCH15', 79)
VARYING_SLOT_PATCH16 = gl_varying_slot.define('VARYING_SLOT_PATCH16', 80)
VARYING_SLOT_PATCH17 = gl_varying_slot.define('VARYING_SLOT_PATCH17', 81)
VARYING_SLOT_PATCH18 = gl_varying_slot.define('VARYING_SLOT_PATCH18', 82)
VARYING_SLOT_PATCH19 = gl_varying_slot.define('VARYING_SLOT_PATCH19', 83)
VARYING_SLOT_PATCH20 = gl_varying_slot.define('VARYING_SLOT_PATCH20', 84)
VARYING_SLOT_PATCH21 = gl_varying_slot.define('VARYING_SLOT_PATCH21', 85)
VARYING_SLOT_PATCH22 = gl_varying_slot.define('VARYING_SLOT_PATCH22', 86)
VARYING_SLOT_PATCH23 = gl_varying_slot.define('VARYING_SLOT_PATCH23', 87)
VARYING_SLOT_PATCH24 = gl_varying_slot.define('VARYING_SLOT_PATCH24', 88)
VARYING_SLOT_PATCH25 = gl_varying_slot.define('VARYING_SLOT_PATCH25', 89)
VARYING_SLOT_PATCH26 = gl_varying_slot.define('VARYING_SLOT_PATCH26', 90)
VARYING_SLOT_PATCH27 = gl_varying_slot.define('VARYING_SLOT_PATCH27', 91)
VARYING_SLOT_PATCH28 = gl_varying_slot.define('VARYING_SLOT_PATCH28', 92)
VARYING_SLOT_PATCH29 = gl_varying_slot.define('VARYING_SLOT_PATCH29', 93)
VARYING_SLOT_PATCH30 = gl_varying_slot.define('VARYING_SLOT_PATCH30', 94)
VARYING_SLOT_PATCH31 = gl_varying_slot.define('VARYING_SLOT_PATCH31', 95)
VARYING_SLOT_VAR0_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR0_16BIT', 96)
VARYING_SLOT_VAR1_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR1_16BIT', 97)
VARYING_SLOT_VAR2_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR2_16BIT', 98)
VARYING_SLOT_VAR3_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR3_16BIT', 99)
VARYING_SLOT_VAR4_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR4_16BIT', 100)
VARYING_SLOT_VAR5_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR5_16BIT', 101)
VARYING_SLOT_VAR6_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR6_16BIT', 102)
VARYING_SLOT_VAR7_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR7_16BIT', 103)
VARYING_SLOT_VAR8_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR8_16BIT', 104)
VARYING_SLOT_VAR9_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR9_16BIT', 105)
VARYING_SLOT_VAR10_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR10_16BIT', 106)
VARYING_SLOT_VAR11_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR11_16BIT', 107)
VARYING_SLOT_VAR12_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR12_16BIT', 108)
VARYING_SLOT_VAR13_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR13_16BIT', 109)
VARYING_SLOT_VAR14_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR14_16BIT', 110)
VARYING_SLOT_VAR15_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR15_16BIT', 111)
NUM_TOTAL_VARYING_SLOTS = gl_varying_slot.define('NUM_TOTAL_VARYING_SLOTS', 112)

try: (nir_slot_is_sysval_output:=dll.nir_slot_is_sysval_output).restype, nir_slot_is_sysval_output.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_slot_is_varying:=dll.nir_slot_is_varying).restype, nir_slot_is_varying.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_slot_is_sysval_output_and_varying:=dll.nir_slot_is_sysval_output_and_varying).restype, nir_slot_is_sysval_output_and_varying.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_remove_varying:=dll.nir_remove_varying).restype, nir_remove_varying.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr), gl_shader_stage]
except AttributeError: pass

try: (nir_remove_sysval_output:=dll.nir_remove_sysval_output).restype, nir_remove_sysval_output.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr), gl_shader_stage]
except AttributeError: pass

try: (nir_lower_amul:=dll.nir_lower_amul).restype, nir_lower_amul.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool)]
except AttributeError: pass

try: (nir_lower_ubo_vec4:=dll.nir_lower_ubo_vec4).restype, nir_lower_ubo_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_sort_variables_by_location:=dll.nir_sort_variables_by_location).restype, nir_sort_variables_by_location.argtypes = None, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_assign_io_var_locations:=dll.nir_assign_io_var_locations).restype, nir_assign_io_var_locations.argtypes = None, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint32), gl_shader_stage]
except AttributeError: pass

try: (nir_opt_clip_cull_const:=dll.nir_opt_clip_cull_const).restype, nir_opt_clip_cull_const.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_io_options = CEnum(ctypes.c_uint32)
nir_lower_io_lower_64bit_to_32 = nir_lower_io_options.define('nir_lower_io_lower_64bit_to_32', 1)
nir_lower_io_lower_64bit_float_to_32 = nir_lower_io_options.define('nir_lower_io_lower_64bit_float_to_32', 2)
nir_lower_io_lower_64bit_to_32_new = nir_lower_io_options.define('nir_lower_io_lower_64bit_to_32_new', 4)
nir_lower_io_use_interpolated_input_intrinsics = nir_lower_io_options.define('nir_lower_io_use_interpolated_input_intrinsics', 8)

try: (nir_lower_io:=dll.nir_lower_io).restype, nir_lower_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool), nir_lower_io_options]
except AttributeError: pass

try: (nir_io_add_const_offset_to_base:=dll.nir_io_add_const_offset_to_base).restype, nir_io_add_const_offset_to_base.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_io_passes:=dll.nir_lower_io_passes).restype, nir_lower_io_passes.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_io_add_intrinsic_xfb_info:=dll.nir_io_add_intrinsic_xfb_info).restype, nir_io_add_intrinsic_xfb_info.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_indirect_loads:=dll.nir_lower_io_indirect_loads).restype, nir_lower_io_indirect_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_vars_to_explicit_types:=dll.nir_lower_vars_to_explicit_types).restype, nir_lower_vars_to_explicit_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, glsl_type_size_align_func]
except AttributeError: pass

try: (nir_gather_explicit_io_initializers:=dll.nir_gather_explicit_io_initializers).restype, nir_gather_explicit_io_initializers.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.c_void_p, size_t, nir_variable_mode]
except AttributeError: pass

try: (nir_lower_vec3_to_vec4:=dll.nir_lower_vec3_to_vec4).restype, nir_lower_vec3_to_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

nir_address_format = CEnum(ctypes.c_uint32)
nir_address_format_32bit_global = nir_address_format.define('nir_address_format_32bit_global', 0)
nir_address_format_64bit_global = nir_address_format.define('nir_address_format_64bit_global', 1)
nir_address_format_2x32bit_global = nir_address_format.define('nir_address_format_2x32bit_global', 2)
nir_address_format_64bit_global_32bit_offset = nir_address_format.define('nir_address_format_64bit_global_32bit_offset', 3)
nir_address_format_64bit_bounded_global = nir_address_format.define('nir_address_format_64bit_bounded_global', 4)
nir_address_format_32bit_index_offset = nir_address_format.define('nir_address_format_32bit_index_offset', 5)
nir_address_format_32bit_index_offset_pack64 = nir_address_format.define('nir_address_format_32bit_index_offset_pack64', 6)
nir_address_format_vec2_index_32bit_offset = nir_address_format.define('nir_address_format_vec2_index_32bit_offset', 7)
nir_address_format_62bit_generic = nir_address_format.define('nir_address_format_62bit_generic', 8)
nir_address_format_32bit_offset = nir_address_format.define('nir_address_format_32bit_offset', 9)
nir_address_format_32bit_offset_as_64bit = nir_address_format.define('nir_address_format_32bit_offset_as_64bit', 10)
nir_address_format_logical = nir_address_format.define('nir_address_format_logical', 11)

try: (nir_address_format_bit_size:=dll.nir_address_format_bit_size).restype, nir_address_format_bit_size.argtypes = ctypes.c_uint32, [nir_address_format]
except AttributeError: pass

try: (nir_address_format_num_components:=dll.nir_address_format_num_components).restype, nir_address_format_num_components.argtypes = ctypes.c_uint32, [nir_address_format]
except AttributeError: pass

try: (nir_address_format_null_value:=dll.nir_address_format_null_value).restype, nir_address_format_null_value.argtypes = ctypes.POINTER(nir_const_value), [nir_address_format]
except AttributeError: pass

try: (nir_build_addr_iadd:=dll.nir_build_addr_iadd).restype, nir_build_addr_iadd.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_address_format, nir_variable_mode, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_addr_iadd_imm:=dll.nir_build_addr_iadd_imm).restype, nir_build_addr_iadd_imm.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_address_format, nir_variable_mode, int64_t]
except AttributeError: pass

try: (nir_build_addr_ieq:=dll.nir_build_addr_ieq).restype, nir_build_addr_ieq.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_build_addr_isub:=dll.nir_build_addr_isub).restype, nir_build_addr_isub.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_explicit_io_address_from_deref:=dll.nir_explicit_io_address_from_deref).restype, nir_explicit_io_address_from_deref.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_deref_instr), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_get_explicit_deref_align:=dll.nir_get_explicit_deref_align).restype, nir_get_explicit_deref_align.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr), ctypes.c_bool, ctypes.POINTER(uint32_t), ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (nir_lower_explicit_io_instr:=dll.nir_lower_explicit_io_instr).restype, nir_lower_explicit_io_instr.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_lower_explicit_io:=dll.nir_lower_explicit_io).restype, nir_lower_explicit_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, nir_address_format]
except AttributeError: pass

nir_mem_access_shift_method = CEnum(ctypes.c_uint32)
nir_mem_access_shift_method_scalar = nir_mem_access_shift_method.define('nir_mem_access_shift_method_scalar', 0)
nir_mem_access_shift_method_shift64 = nir_mem_access_shift_method.define('nir_mem_access_shift_method_shift64', 1)
nir_mem_access_shift_method_bytealign_amd = nir_mem_access_shift_method.define('nir_mem_access_shift_method_bytealign_amd', 2)

class struct_nir_mem_access_size_align(Struct): pass
struct_nir_mem_access_size_align._fields_ = [
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('align', uint16_t),
  ('shift', nir_mem_access_shift_method),
]
nir_mem_access_size_align = struct_nir_mem_access_size_align
enum_gl_access_qualifier = CEnum(ctypes.c_uint32)
ACCESS_COHERENT = enum_gl_access_qualifier.define('ACCESS_COHERENT', 1)
ACCESS_RESTRICT = enum_gl_access_qualifier.define('ACCESS_RESTRICT', 2)
ACCESS_VOLATILE = enum_gl_access_qualifier.define('ACCESS_VOLATILE', 4)
ACCESS_NON_READABLE = enum_gl_access_qualifier.define('ACCESS_NON_READABLE', 8)
ACCESS_NON_WRITEABLE = enum_gl_access_qualifier.define('ACCESS_NON_WRITEABLE', 16)
ACCESS_NON_UNIFORM = enum_gl_access_qualifier.define('ACCESS_NON_UNIFORM', 32)
ACCESS_CAN_REORDER = enum_gl_access_qualifier.define('ACCESS_CAN_REORDER', 64)
ACCESS_NON_TEMPORAL = enum_gl_access_qualifier.define('ACCESS_NON_TEMPORAL', 128)
ACCESS_INCLUDE_HELPERS = enum_gl_access_qualifier.define('ACCESS_INCLUDE_HELPERS', 256)
ACCESS_IS_SWIZZLED_AMD = enum_gl_access_qualifier.define('ACCESS_IS_SWIZZLED_AMD', 512)
ACCESS_USES_FORMAT_AMD = enum_gl_access_qualifier.define('ACCESS_USES_FORMAT_AMD', 1024)
ACCESS_FMASK_LOWERED_AMD = enum_gl_access_qualifier.define('ACCESS_FMASK_LOWERED_AMD', 2048)
ACCESS_CAN_SPECULATE = enum_gl_access_qualifier.define('ACCESS_CAN_SPECULATE', 4096)
ACCESS_CP_GE_COHERENT_AMD = enum_gl_access_qualifier.define('ACCESS_CP_GE_COHERENT_AMD', 8192)
ACCESS_IN_BOUNDS = enum_gl_access_qualifier.define('ACCESS_IN_BOUNDS', 16384)
ACCESS_KEEP_SCALAR = enum_gl_access_qualifier.define('ACCESS_KEEP_SCALAR', 32768)
ACCESS_SMEM_AMD = enum_gl_access_qualifier.define('ACCESS_SMEM_AMD', 65536)

nir_lower_mem_access_bit_sizes_cb = ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, enum_gl_access_qualifier, ctypes.c_void_p)
class struct_nir_lower_mem_access_bit_sizes_options(Struct): pass
struct_nir_lower_mem_access_bit_sizes_options._fields_ = [
  ('callback', nir_lower_mem_access_bit_sizes_cb),
  ('modes', nir_variable_mode),
  ('may_lower_unaligned_stores_to_atomics', ctypes.c_bool),
  ('cb_data', ctypes.c_void_p),
]
nir_lower_mem_access_bit_sizes_options = struct_nir_lower_mem_access_bit_sizes_options
try: (nir_lower_mem_access_bit_sizes:=dll.nir_lower_mem_access_bit_sizes).restype, nir_lower_mem_access_bit_sizes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_mem_access_bit_sizes_options)]
except AttributeError: pass

try: (nir_lower_robust_access:=dll.nir_lower_robust_access).restype, nir_lower_robust_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_intrin_filter_cb, ctypes.c_void_p]
except AttributeError: pass

nir_should_vectorize_mem_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
class struct_nir_load_store_vectorize_options(Struct): pass
struct_nir_load_store_vectorize_options._fields_ = [
  ('callback', nir_should_vectorize_mem_func),
  ('modes', nir_variable_mode),
  ('robust_modes', nir_variable_mode),
  ('cb_data', ctypes.c_void_p),
  ('has_shared2_amd', ctypes.c_bool),
]
nir_load_store_vectorize_options = struct_nir_load_store_vectorize_options
try: (nir_opt_load_store_vectorize:=dll.nir_opt_load_store_vectorize).restype, nir_opt_load_store_vectorize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_load_store_vectorize_options)]
except AttributeError: pass

try: (nir_opt_load_store_update_alignments:=dll.nir_opt_load_store_update_alignments).restype, nir_opt_load_store_update_alignments.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_shader_calls_should_remat_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
class struct_nir_lower_shader_calls_options(Struct): pass
struct_nir_lower_shader_calls_options._fields_ = [
  ('address_format', nir_address_format),
  ('stack_alignment', ctypes.c_uint32),
  ('localized_loads', ctypes.c_bool),
  ('vectorizer_callback', nir_should_vectorize_mem_func),
  ('vectorizer_data', ctypes.c_void_p),
  ('should_remat_callback', nir_lower_shader_calls_should_remat_func),
  ('should_remat_data', ctypes.c_void_p),
]
nir_lower_shader_calls_options = struct_nir_lower_shader_calls_options
try: (nir_lower_shader_calls:=dll.nir_lower_shader_calls).restype, nir_lower_shader_calls.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_shader_calls_options), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(nir_shader))), ctypes.POINTER(uint32_t), ctypes.c_void_p]
except AttributeError: pass

try: (nir_get_io_offset_src_number:=dll.nir_get_io_offset_src_number).restype, nir_get_io_offset_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_index_src_number:=dll.nir_get_io_index_src_number).restype, nir_get_io_index_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_arrayed_index_src_number:=dll.nir_get_io_arrayed_index_src_number).restype, nir_get_io_arrayed_index_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_offset_src:=dll.nir_get_io_offset_src).restype, nir_get_io_offset_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_index_src:=dll.nir_get_io_index_src).restype, nir_get_io_index_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_arrayed_index_src:=dll.nir_get_io_arrayed_index_src).restype, nir_get_io_arrayed_index_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_shader_call_payload_src:=dll.nir_get_shader_call_payload_src).restype, nir_get_shader_call_payload_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_is_output_load:=dll.nir_is_output_load).restype, nir_is_output_load.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_is_arrayed_io:=dll.nir_is_arrayed_io).restype, nir_is_arrayed_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_variable), gl_shader_stage]
except AttributeError: pass

try: (nir_lower_reg_intrinsics_to_ssa_impl:=dll.nir_lower_reg_intrinsics_to_ssa_impl).restype, nir_lower_reg_intrinsics_to_ssa_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_reg_intrinsics_to_ssa:=dll.nir_lower_reg_intrinsics_to_ssa).restype, nir_lower_reg_intrinsics_to_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_vars_to_ssa:=dll.nir_lower_vars_to_ssa).restype, nir_lower_vars_to_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_dead_derefs:=dll.nir_remove_dead_derefs).restype, nir_remove_dead_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_dead_derefs_impl:=dll.nir_remove_dead_derefs_impl).restype, nir_remove_dead_derefs_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

class struct_nir_remove_dead_variables_options(Struct): pass
struct_nir_remove_dead_variables_options._fields_ = [
  ('can_remove_var', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_variable), ctypes.c_void_p)),
  ('can_remove_var_data', ctypes.c_void_p),
]
nir_remove_dead_variables_options = struct_nir_remove_dead_variables_options
try: (nir_remove_dead_variables:=dll.nir_remove_dead_variables).restype, nir_remove_dead_variables.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(nir_remove_dead_variables_options)]
except AttributeError: pass

try: (nir_lower_variable_initializers:=dll.nir_lower_variable_initializers).restype, nir_lower_variable_initializers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_zero_initialize_shared_memory:=dll.nir_zero_initialize_shared_memory).restype, nir_zero_initialize_shared_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_clear_shared_memory:=dll.nir_clear_shared_memory).restype, nir_clear_shared_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

nir_opt_move_to_top_options = CEnum(ctypes.c_uint32)
nir_move_to_entry_block_only = nir_opt_move_to_top_options.define('nir_move_to_entry_block_only', 1)
nir_move_to_top_input_loads = nir_opt_move_to_top_options.define('nir_move_to_top_input_loads', 2)
nir_move_to_top_load_smem_amd = nir_opt_move_to_top_options.define('nir_move_to_top_load_smem_amd', 4)

try: (nir_opt_move_to_top:=dll.nir_opt_move_to_top).restype, nir_opt_move_to_top.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_opt_move_to_top_options]
except AttributeError: pass

try: (nir_move_vec_src_uses_to_dest:=dll.nir_move_vec_src_uses_to_dest).restype, nir_move_vec_src_uses_to_dest.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_move_output_stores_to_end:=dll.nir_move_output_stores_to_end).restype, nir_move_output_stores_to_end.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_vec_to_regs:=dll.nir_lower_vec_to_regs).restype, nir_lower_vec_to_regs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_writemask_filter_cb, ctypes.c_void_p]
except AttributeError: pass

enum_compare_func = CEnum(ctypes.c_uint32)
COMPARE_FUNC_NEVER = enum_compare_func.define('COMPARE_FUNC_NEVER', 0)
COMPARE_FUNC_LESS = enum_compare_func.define('COMPARE_FUNC_LESS', 1)
COMPARE_FUNC_EQUAL = enum_compare_func.define('COMPARE_FUNC_EQUAL', 2)
COMPARE_FUNC_LEQUAL = enum_compare_func.define('COMPARE_FUNC_LEQUAL', 3)
COMPARE_FUNC_GREATER = enum_compare_func.define('COMPARE_FUNC_GREATER', 4)
COMPARE_FUNC_NOTEQUAL = enum_compare_func.define('COMPARE_FUNC_NOTEQUAL', 5)
COMPARE_FUNC_GEQUAL = enum_compare_func.define('COMPARE_FUNC_GEQUAL', 6)
COMPARE_FUNC_ALWAYS = enum_compare_func.define('COMPARE_FUNC_ALWAYS', 7)

try: (nir_lower_alpha_test:=dll.nir_lower_alpha_test).restype, nir_lower_alpha_test.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), enum_compare_func, ctypes.c_bool, ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

try: (nir_lower_alpha_to_coverage:=dll.nir_lower_alpha_to_coverage).restype, nir_lower_alpha_to_coverage.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint8_t, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_alpha_to_one:=dll.nir_lower_alpha_to_one).restype, nir_lower_alpha_to_one.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu:=dll.nir_lower_alu).restype, nir_lower_alu.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_flrp:=dll.nir_lower_flrp).restype, nir_lower_flrp.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

try: (nir_scale_fdiv:=dll.nir_scale_fdiv).restype, nir_scale_fdiv.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu_to_scalar:=dll.nir_lower_alu_to_scalar).restype, nir_lower_alu_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_alu_width:=dll.nir_lower_alu_width).restype, nir_lower_alu_width.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_alu_vec8_16_srcs:=dll.nir_lower_alu_vec8_16_srcs).restype, nir_lower_alu_vec8_16_srcs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_bool_to_bitsize:=dll.nir_lower_bool_to_bitsize).restype, nir_lower_bool_to_bitsize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_bool_to_float:=dll.nir_lower_bool_to_float).restype, nir_lower_bool_to_float.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_bool_to_int32:=dll.nir_lower_bool_to_int32).restype, nir_lower_bool_to_int32.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_simplify_convert_alu_types:=dll.nir_opt_simplify_convert_alu_types).restype, nir_opt_simplify_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_const_arrays_to_uniforms:=dll.nir_lower_const_arrays_to_uniforms).restype, nir_lower_const_arrays_to_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_convert_alu_types:=dll.nir_lower_convert_alu_types).restype, nir_lower_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_intrinsic_instr))]
except AttributeError: pass

try: (nir_lower_constant_convert_alu_types:=dll.nir_lower_constant_convert_alu_types).restype, nir_lower_constant_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu_conversion_to_intrinsic:=dll.nir_lower_alu_conversion_to_intrinsic).restype, nir_lower_alu_conversion_to_intrinsic.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int_to_float:=dll.nir_lower_int_to_float).restype, nir_lower_int_to_float.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_load_const_to_scalar:=dll.nir_lower_load_const_to_scalar).restype, nir_lower_load_const_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_read_invocation_to_scalar:=dll.nir_lower_read_invocation_to_scalar).restype, nir_lower_read_invocation_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_phis_to_scalar:=dll.nir_lower_phis_to_scalar).restype, nir_lower_phis_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_all_phis_to_scalar:=dll.nir_lower_all_phis_to_scalar).restype, nir_lower_all_phis_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_array_vars_to_elements:=dll.nir_lower_io_array_vars_to_elements).restype, nir_lower_io_array_vars_to_elements.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_array_vars_to_elements_no_indirects:=dll.nir_lower_io_array_vars_to_elements_no_indirects).restype, nir_lower_io_array_vars_to_elements_no_indirects.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_io_to_scalar:=dll.nir_lower_io_to_scalar).restype, nir_lower_io_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_io_vars_to_scalar:=dll.nir_lower_io_vars_to_scalar).restype, nir_lower_io_vars_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_opt_vectorize_io_vars:=dll.nir_opt_vectorize_io_vars).restype, nir_opt_vectorize_io_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_tess_level_array_vars_to_vec:=dll.nir_lower_tess_level_array_vars_to_vec).restype, nir_lower_tess_level_array_vars_to_vec.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_create_passthrough_tcs_impl:=dll.nir_create_passthrough_tcs_impl).restype, nir_create_passthrough_tcs_impl.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, uint8_t]
except AttributeError: pass

try: (nir_create_passthrough_tcs:=dll.nir_create_passthrough_tcs).restype, nir_create_passthrough_tcs.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(nir_shader), uint8_t]
except AttributeError: pass

try: (nir_create_passthrough_gs:=dll.nir_create_passthrough_gs).restype, nir_create_passthrough_gs.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(nir_shader), enum_mesa_prim, enum_mesa_prim, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_fragcolor:=dll.nir_lower_fragcolor).restype, nir_lower_fragcolor.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_fragcoord_wtrans:=dll.nir_lower_fragcoord_wtrans).restype, nir_lower_fragcoord_wtrans.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_frag_coord_to_pixel_coord:=dll.nir_opt_frag_coord_to_pixel_coord).restype, nir_opt_frag_coord_to_pixel_coord.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_frag_coord_to_pixel_coord:=dll.nir_lower_frag_coord_to_pixel_coord).restype, nir_lower_frag_coord_to_pixel_coord.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_viewport_transform:=dll.nir_lower_viewport_transform).restype, nir_lower_viewport_transform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_uniforms_to_ubo:=dll.nir_lower_uniforms_to_ubo).restype, nir_lower_uniforms_to_ubo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_is_helper_invocation:=dll.nir_lower_is_helper_invocation).restype, nir_lower_is_helper_invocation.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_single_sampled:=dll.nir_lower_single_sampled).restype, nir_lower_single_sampled.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_atomics:=dll.nir_lower_atomics).restype, nir_lower_atomics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb]
except AttributeError: pass

class struct_nir_lower_subgroups_options(Struct): pass
struct_nir_lower_subgroups_options._fields_ = [
  ('filter', nir_instr_filter_cb),
  ('filter_data', ctypes.c_void_p),
  ('subgroup_size', uint8_t),
  ('ballot_bit_size', uint8_t),
  ('ballot_components', uint8_t),
  ('lower_to_scalar', ctypes.c_bool,1),
  ('lower_vote_trivial', ctypes.c_bool,1),
  ('lower_vote_feq', ctypes.c_bool,1),
  ('lower_vote_ieq', ctypes.c_bool,1),
  ('lower_vote_bool_eq', ctypes.c_bool,1),
  ('lower_first_invocation_to_ballot', ctypes.c_bool,1),
  ('lower_read_first_invocation', ctypes.c_bool,1),
  ('lower_subgroup_masks', ctypes.c_bool,1),
  ('lower_relative_shuffle', ctypes.c_bool,1),
  ('lower_shuffle_to_32bit', ctypes.c_bool,1),
  ('lower_shuffle_to_swizzle_amd', ctypes.c_bool,1),
  ('lower_shuffle', ctypes.c_bool,1),
  ('lower_quad', ctypes.c_bool,1),
  ('lower_quad_broadcast_dynamic', ctypes.c_bool,1),
  ('lower_quad_broadcast_dynamic_to_const', ctypes.c_bool,1),
  ('lower_quad_vote', ctypes.c_bool,1),
  ('lower_elect', ctypes.c_bool,1),
  ('lower_read_invocation_to_cond', ctypes.c_bool,1),
  ('lower_rotate_to_shuffle', ctypes.c_bool,1),
  ('lower_rotate_clustered_to_shuffle', ctypes.c_bool,1),
  ('lower_ballot_bit_count_to_mbcnt_amd', ctypes.c_bool,1),
  ('lower_inverse_ballot', ctypes.c_bool,1),
  ('lower_reduce', ctypes.c_bool,1),
  ('lower_boolean_reduce', ctypes.c_bool,1),
  ('lower_boolean_shuffle', ctypes.c_bool,1),
]
nir_lower_subgroups_options = struct_nir_lower_subgroups_options
try: (nir_lower_subgroups:=dll.nir_lower_subgroups).restype, nir_lower_subgroups.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_subgroups_options)]
except AttributeError: pass

try: (nir_lower_system_values:=dll.nir_lower_system_values).restype, nir_lower_system_values.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_build_lowered_load_helper_invocation:=dll.nir_build_lowered_load_helper_invocation).restype, nir_build_lowered_load_helper_invocation.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder)]
except AttributeError: pass

class struct_nir_lower_compute_system_values_options(Struct): pass
struct_nir_lower_compute_system_values_options._fields_ = [
  ('has_base_global_invocation_id', ctypes.c_bool,1),
  ('has_base_workgroup_id', ctypes.c_bool,1),
  ('has_global_size', ctypes.c_bool,1),
  ('shuffle_local_ids_for_quad_derivatives', ctypes.c_bool,1),
  ('lower_local_invocation_index', ctypes.c_bool,1),
  ('lower_cs_local_id_to_index', ctypes.c_bool,1),
  ('lower_workgroup_id_to_index', ctypes.c_bool,1),
  ('global_id_is_32bit', ctypes.c_bool,1),
  ('shortcut_1d_workgroup_id', ctypes.c_bool,1),
  ('num_workgroups', (uint32_t * 3)),
]
nir_lower_compute_system_values_options = struct_nir_lower_compute_system_values_options
try: (nir_lower_compute_system_values:=dll.nir_lower_compute_system_values).restype, nir_lower_compute_system_values.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_compute_system_values_options)]
except AttributeError: pass

class struct_nir_lower_sysvals_to_varyings_options(Struct): pass
struct_nir_lower_sysvals_to_varyings_options._fields_ = [
  ('frag_coord', ctypes.c_bool,1),
  ('front_face', ctypes.c_bool,1),
  ('point_coord', ctypes.c_bool,1),
]
nir_lower_sysvals_to_varyings_options = struct_nir_lower_sysvals_to_varyings_options
try: (nir_lower_sysvals_to_varyings:=dll.nir_lower_sysvals_to_varyings).restype, nir_lower_sysvals_to_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_sysvals_to_varyings_options)]
except AttributeError: pass

enum_nir_lower_tex_packing = CEnum(ctypes.c_ubyte)
nir_lower_tex_packing_none = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_none', 0)
nir_lower_tex_packing_16 = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_16', 1)
nir_lower_tex_packing_8 = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_8', 2)

class struct_nir_lower_tex_options(Struct): pass
struct_nir_lower_tex_options._fields_ = [
  ('lower_txp', ctypes.c_uint32),
  ('lower_txp_array', ctypes.c_bool),
  ('lower_txf_offset', ctypes.c_bool),
  ('lower_rect_offset', ctypes.c_bool),
  ('lower_offset_filter', nir_instr_filter_cb),
  ('lower_rect', ctypes.c_bool),
  ('lower_1d', ctypes.c_bool),
  ('lower_1d_shadow', ctypes.c_bool),
  ('lower_y_uv_external', ctypes.c_uint32),
  ('lower_y_vu_external', ctypes.c_uint32),
  ('lower_y_u_v_external', ctypes.c_uint32),
  ('lower_yx_xuxv_external', ctypes.c_uint32),
  ('lower_yx_xvxu_external', ctypes.c_uint32),
  ('lower_xy_uxvx_external', ctypes.c_uint32),
  ('lower_xy_vxux_external', ctypes.c_uint32),
  ('lower_ayuv_external', ctypes.c_uint32),
  ('lower_xyuv_external', ctypes.c_uint32),
  ('lower_yuv_external', ctypes.c_uint32),
  ('lower_yu_yv_external', ctypes.c_uint32),
  ('lower_yv_yu_external', ctypes.c_uint32),
  ('lower_y41x_external', ctypes.c_uint32),
  ('lower_sx10_external', ctypes.c_uint32),
  ('lower_sx12_external', ctypes.c_uint32),
  ('bt709_external', ctypes.c_uint32),
  ('bt2020_external', ctypes.c_uint32),
  ('yuv_full_range_external', ctypes.c_uint32),
  ('saturate_s', ctypes.c_uint32),
  ('saturate_t', ctypes.c_uint32),
  ('saturate_r', ctypes.c_uint32),
  ('swizzle_result', ctypes.c_uint32),
  ('swizzles', ((uint8_t * 4) * 32)),
  ('scale_factors', (ctypes.c_float * 32)),
  ('lower_srgb', ctypes.c_uint32),
  ('lower_txd_cube_map', ctypes.c_bool),
  ('lower_txd_3d', ctypes.c_bool),
  ('lower_txd_array', ctypes.c_bool),
  ('lower_txd_shadow', ctypes.c_bool),
  ('lower_txd', ctypes.c_bool),
  ('lower_txd_clamp', ctypes.c_bool),
  ('lower_txb_shadow_clamp', ctypes.c_bool),
  ('lower_txd_shadow_clamp', ctypes.c_bool),
  ('lower_txd_offset_clamp', ctypes.c_bool),
  ('lower_txd_clamp_bindless_sampler', ctypes.c_bool),
  ('lower_txd_clamp_if_sampler_index_not_lt_16', ctypes.c_bool),
  ('lower_txs_lod', ctypes.c_bool),
  ('lower_txs_cube_array', ctypes.c_bool),
  ('lower_tg4_broadcom_swizzle', ctypes.c_bool),
  ('lower_tg4_offsets', ctypes.c_bool),
  ('lower_to_fragment_fetch_amd', ctypes.c_bool),
  ('lower_tex_packing_cb', ctypes.CFUNCTYPE(enum_nir_lower_tex_packing, ctypes.POINTER(nir_tex_instr), ctypes.c_void_p)),
  ('lower_tex_packing_data', ctypes.c_void_p),
  ('lower_lod_zero_width', ctypes.c_bool),
  ('lower_sampler_lod_bias', ctypes.c_bool),
  ('lower_invalid_implicit_lod', ctypes.c_bool),
  ('lower_index_to_offset', ctypes.c_bool),
  ('callback_data', ctypes.c_void_p),
]
nir_lower_tex_options = struct_nir_lower_tex_options
try: (nir_lower_tex:=dll.nir_lower_tex).restype, nir_lower_tex.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_tex_options)]
except AttributeError: pass

class struct_nir_lower_tex_shadow_swizzle(Struct): pass
struct_nir_lower_tex_shadow_swizzle._fields_ = [
  ('swizzle_r', ctypes.c_uint32,3),
  ('swizzle_g', ctypes.c_uint32,3),
  ('swizzle_b', ctypes.c_uint32,3),
  ('swizzle_a', ctypes.c_uint32,3),
]
nir_lower_tex_shadow_swizzle = struct_nir_lower_tex_shadow_swizzle
try: (nir_lower_tex_shadow:=dll.nir_lower_tex_shadow).restype, nir_lower_tex_shadow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(enum_compare_func), ctypes.POINTER(nir_lower_tex_shadow_swizzle), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_image_options(Struct): pass
struct_nir_lower_image_options._fields_ = [
  ('lower_cube_size', ctypes.c_bool),
  ('lower_to_fragment_mask_load_amd', ctypes.c_bool),
  ('lower_image_samples_to_one', ctypes.c_bool),
]
nir_lower_image_options = struct_nir_lower_image_options
try: (nir_lower_image:=dll.nir_lower_image).restype, nir_lower_image.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_image_options)]
except AttributeError: pass

try: (nir_lower_image_atomics_to_global:=dll.nir_lower_image_atomics_to_global).restype, nir_lower_image_atomics_to_global.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_intrin_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_readonly_images_to_tex:=dll.nir_lower_readonly_images_to_tex).restype, nir_lower_readonly_images_to_tex.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

enum_nir_lower_non_uniform_access_type = CEnum(ctypes.c_uint32)
nir_lower_non_uniform_ubo_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_ubo_access', 1)
nir_lower_non_uniform_ssbo_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_ssbo_access', 2)
nir_lower_non_uniform_texture_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_texture_access', 4)
nir_lower_non_uniform_image_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_image_access', 8)
nir_lower_non_uniform_get_ssbo_size = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_get_ssbo_size', 16)
nir_lower_non_uniform_texture_offset_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_texture_offset_access', 32)
nir_lower_non_uniform_access_type_count = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_access_type_count', 6)

nir_lower_non_uniform_src_access_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32, ctypes.c_void_p)
nir_lower_non_uniform_access_callback = ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_nir_src), ctypes.c_void_p)
class struct_nir_lower_non_uniform_access_options(Struct): pass
struct_nir_lower_non_uniform_access_options._fields_ = [
  ('types', enum_nir_lower_non_uniform_access_type),
  ('tex_src_callback', nir_lower_non_uniform_src_access_callback),
  ('callback', nir_lower_non_uniform_access_callback),
  ('callback_data', ctypes.c_void_p),
]
nir_lower_non_uniform_access_options = struct_nir_lower_non_uniform_access_options
try: (nir_has_non_uniform_access:=dll.nir_has_non_uniform_access).restype, nir_has_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), enum_nir_lower_non_uniform_access_type]
except AttributeError: pass

try: (nir_opt_non_uniform_access:=dll.nir_opt_non_uniform_access).restype, nir_opt_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_non_uniform_access:=dll.nir_lower_non_uniform_access).restype, nir_lower_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_non_uniform_access_options)]
except AttributeError: pass

class struct_nir_lower_idiv_options(Struct): pass
struct_nir_lower_idiv_options._fields_ = [
  ('allow_fp16', ctypes.c_bool),
]
nir_lower_idiv_options = struct_nir_lower_idiv_options
try: (nir_lower_idiv:=dll.nir_lower_idiv).restype, nir_lower_idiv.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_idiv_options)]
except AttributeError: pass

class struct_nir_input_attachment_options(Struct): pass
struct_nir_input_attachment_options._fields_ = [
  ('use_ia_coord_intrin', ctypes.c_bool),
  ('use_fragcoord_sysval', ctypes.c_bool),
  ('use_layer_id_sysval', ctypes.c_bool),
  ('use_view_id_for_layer', ctypes.c_bool),
  ('unscaled_depth_stencil_ir3', ctypes.c_bool),
  ('unscaled_input_attachment_ir3', uint32_t),
]
nir_input_attachment_options = struct_nir_input_attachment_options
try: (nir_lower_input_attachments:=dll.nir_lower_input_attachments).restype, nir_lower_input_attachments.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_input_attachment_options)]
except AttributeError: pass

try: (nir_lower_clip_vs:=dll.nir_lower_clip_vs).restype, nir_lower_clip_vs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_clip_gs:=dll.nir_lower_clip_gs).restype, nir_lower_clip_gs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_clip_fs:=dll.nir_lower_clip_fs).restype, nir_lower_clip_fs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_clip_cull_distance_to_vec4s:=dll.nir_lower_clip_cull_distance_to_vec4s).restype, nir_lower_clip_cull_distance_to_vec4s.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_cull_distance_array_vars:=dll.nir_lower_clip_cull_distance_array_vars).restype, nir_lower_clip_cull_distance_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_disable:=dll.nir_lower_clip_disable).restype, nir_lower_clip_disable.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_point_size_mov:=dll.nir_lower_point_size_mov).restype, nir_lower_point_size_mov.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

try: (nir_lower_frexp:=dll.nir_lower_frexp).restype, nir_lower_frexp.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_two_sided_color:=dll.nir_lower_two_sided_color).restype, nir_lower_two_sided_color.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_clamp_color_outputs:=dll.nir_lower_clamp_color_outputs).restype, nir_lower_clamp_color_outputs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_flatshade:=dll.nir_lower_flatshade).restype, nir_lower_flatshade.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_passthrough_edgeflags:=dll.nir_lower_passthrough_edgeflags).restype, nir_lower_passthrough_edgeflags.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_patch_vertices:=dll.nir_lower_patch_vertices).restype, nir_lower_patch_vertices.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

class struct_nir_lower_wpos_ytransform_options(Struct): pass
struct_nir_lower_wpos_ytransform_options._fields_ = [
  ('state_tokens', (gl_state_index16 * 4)),
  ('fs_coord_origin_upper_left', ctypes.c_bool,1),
  ('fs_coord_origin_lower_left', ctypes.c_bool,1),
  ('fs_coord_pixel_center_integer', ctypes.c_bool,1),
  ('fs_coord_pixel_center_half_integer', ctypes.c_bool,1),
]
nir_lower_wpos_ytransform_options = struct_nir_lower_wpos_ytransform_options
try: (nir_lower_wpos_ytransform:=dll.nir_lower_wpos_ytransform).restype, nir_lower_wpos_ytransform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_wpos_ytransform_options)]
except AttributeError: pass

try: (nir_lower_wpos_center:=dll.nir_lower_wpos_center).restype, nir_lower_wpos_center.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_pntc_ytransform:=dll.nir_lower_pntc_ytransform).restype, nir_lower_pntc_ytransform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_wrmasks:=dll.nir_lower_wrmasks).restype, nir_lower_wrmasks.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_fb_read:=dll.nir_lower_fb_read).restype, nir_lower_fb_read.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_drawpixels_options(Struct): pass
struct_nir_lower_drawpixels_options._fields_ = [
  ('texcoord_state_tokens', (gl_state_index16 * 4)),
  ('scale_state_tokens', (gl_state_index16 * 4)),
  ('bias_state_tokens', (gl_state_index16 * 4)),
  ('drawpix_sampler', ctypes.c_uint32),
  ('pixelmap_sampler', ctypes.c_uint32),
  ('pixel_maps', ctypes.c_bool,1),
  ('scale_and_bias', ctypes.c_bool,1),
]
nir_lower_drawpixels_options = struct_nir_lower_drawpixels_options
try: (nir_lower_drawpixels:=dll.nir_lower_drawpixels).restype, nir_lower_drawpixels.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_drawpixels_options)]
except AttributeError: pass

class struct_nir_lower_bitmap_options(Struct): pass
struct_nir_lower_bitmap_options._fields_ = [
  ('sampler', ctypes.c_uint32),
  ('swizzle_xxxx', ctypes.c_bool),
]
nir_lower_bitmap_options = struct_nir_lower_bitmap_options
try: (nir_lower_bitmap:=dll.nir_lower_bitmap).restype, nir_lower_bitmap.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_bitmap_options)]
except AttributeError: pass

try: (nir_lower_atomics_to_ssbo:=dll.nir_lower_atomics_to_ssbo).restype, nir_lower_atomics_to_ssbo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

nir_lower_gs_intrinsics_flags = CEnum(ctypes.c_uint32)
nir_lower_gs_intrinsics_per_stream = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_per_stream', 1)
nir_lower_gs_intrinsics_count_primitives = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_count_primitives', 2)
nir_lower_gs_intrinsics_count_vertices_per_primitive = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_count_vertices_per_primitive', 4)
nir_lower_gs_intrinsics_overwrite_incomplete = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_overwrite_incomplete', 8)

try: (nir_lower_gs_intrinsics:=dll.nir_lower_gs_intrinsics).restype, nir_lower_gs_intrinsics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_gs_intrinsics_flags]
except AttributeError: pass

try: (nir_lower_halt_to_return:=dll.nir_lower_halt_to_return).restype, nir_lower_halt_to_return.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_tess_coord_z:=dll.nir_lower_tess_coord_z).restype, nir_lower_tess_coord_z.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_task_shader_options(Struct): pass
struct_nir_lower_task_shader_options._fields_ = [
  ('payload_to_shared_for_atomics', ctypes.c_bool,1),
  ('payload_to_shared_for_small_types', ctypes.c_bool,1),
  ('payload_offset_in_bytes', uint32_t),
]
nir_lower_task_shader_options = struct_nir_lower_task_shader_options
try: (nir_lower_task_shader:=dll.nir_lower_task_shader).restype, nir_lower_task_shader.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_task_shader_options]
except AttributeError: pass

nir_lower_bit_size_callback = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
try: (nir_lower_bit_size:=dll.nir_lower_bit_size).restype, nir_lower_bit_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_bit_size_callback, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_64bit_phis:=dll.nir_lower_64bit_phis).restype, nir_lower_64bit_phis.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_split_conversions_options(Struct): pass
struct_nir_split_conversions_options._fields_ = [
  ('callback', nir_lower_bit_size_callback),
  ('callback_data', ctypes.c_void_p),
  ('has_convert_alu_types', ctypes.c_bool),
]
nir_split_conversions_options = struct_nir_split_conversions_options
try: (nir_split_conversions:=dll.nir_split_conversions).restype, nir_split_conversions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_split_conversions_options)]
except AttributeError: pass

try: (nir_split_64bit_vec3_and_vec4:=dll.nir_split_64bit_vec3_and_vec4).restype, nir_split_64bit_vec3_and_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int64_op_to_options_mask:=dll.nir_lower_int64_op_to_options_mask).restype, nir_lower_int64_op_to_options_mask.argtypes = nir_lower_int64_options, [nir_op]
except AttributeError: pass

try: (nir_lower_int64:=dll.nir_lower_int64).restype, nir_lower_int64.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int64_float_conversions:=dll.nir_lower_int64_float_conversions).restype, nir_lower_int64_float_conversions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_doubles_op_to_options_mask:=dll.nir_lower_doubles_op_to_options_mask).restype, nir_lower_doubles_op_to_options_mask.argtypes = nir_lower_doubles_options, [nir_op]
except AttributeError: pass

try: (nir_lower_doubles:=dll.nir_lower_doubles).restype, nir_lower_doubles.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), nir_lower_doubles_options]
except AttributeError: pass

try: (nir_lower_pack:=dll.nir_lower_pack).restype, nir_lower_pack.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_get_io_intrinsic:=dll.nir_get_io_intrinsic).restype, nir_get_io_intrinsic.argtypes = ctypes.POINTER(nir_intrinsic_instr), [ctypes.POINTER(nir_instr), nir_variable_mode, ctypes.POINTER(nir_variable_mode)]
except AttributeError: pass

try: (nir_recompute_io_bases:=dll.nir_recompute_io_bases).restype, nir_recompute_io_bases.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_mediump_vars:=dll.nir_lower_mediump_vars).restype, nir_lower_mediump_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_mediump_io:=dll.nir_lower_mediump_io).restype, nir_lower_mediump_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, uint64_t, ctypes.c_bool]
except AttributeError: pass

try: (nir_clear_mediump_io_flag:=dll.nir_clear_mediump_io_flag).restype, nir_clear_mediump_io_flag.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_opt_tex_srcs_options(Struct): pass
struct_nir_opt_tex_srcs_options._fields_ = [
  ('sampler_dims', ctypes.c_uint32),
  ('src_types', ctypes.c_uint32),
]
nir_opt_tex_srcs_options = struct_nir_opt_tex_srcs_options
class struct_nir_opt_16bit_tex_image_options(Struct): pass
struct_nir_opt_16bit_tex_image_options._fields_ = [
  ('rounding_mode', nir_rounding_mode),
  ('opt_tex_dest_types', nir_alu_type),
  ('opt_image_dest_types', nir_alu_type),
  ('integer_dest_saturates', ctypes.c_bool),
  ('opt_image_store_data', ctypes.c_bool),
  ('opt_image_srcs', ctypes.c_bool),
  ('opt_srcs_options_count', ctypes.c_uint32),
  ('opt_srcs_options', ctypes.POINTER(nir_opt_tex_srcs_options)),
]
nir_opt_16bit_tex_image_options = struct_nir_opt_16bit_tex_image_options
try: (nir_opt_16bit_tex_image:=dll.nir_opt_16bit_tex_image).restype, nir_opt_16bit_tex_image.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_16bit_tex_image_options)]
except AttributeError: pass

class struct_nir_tex_src_type_constraint(Struct): pass
struct_nir_tex_src_type_constraint._fields_ = [
  ('legalize_type', ctypes.c_bool),
  ('bit_size', uint8_t),
  ('match_src', nir_tex_src_type),
]
nir_tex_src_type_constraint = struct_nir_tex_src_type_constraint
nir_tex_src_type_constraints = (struct_nir_tex_src_type_constraint * 23)
try: (nir_legalize_16bit_sampler_srcs:=dll.nir_legalize_16bit_sampler_srcs).restype, nir_legalize_16bit_sampler_srcs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_tex_src_type_constraints]
except AttributeError: pass

try: (nir_lower_point_size:=dll.nir_lower_point_size).restype, nir_lower_point_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_float, ctypes.c_float]
except AttributeError: pass

try: (nir_lower_default_point_size:=dll.nir_lower_default_point_size).restype, nir_lower_default_point_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_texcoord_replace:=dll.nir_lower_texcoord_replace).restype, nir_lower_texcoord_replace.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_texcoord_replace_late:=dll.nir_lower_texcoord_replace_late).restype, nir_lower_texcoord_replace_late.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

nir_lower_interpolation_options = CEnum(ctypes.c_uint32)
nir_lower_interpolation_at_sample = nir_lower_interpolation_options.define('nir_lower_interpolation_at_sample', 2)
nir_lower_interpolation_at_offset = nir_lower_interpolation_options.define('nir_lower_interpolation_at_offset', 4)
nir_lower_interpolation_centroid = nir_lower_interpolation_options.define('nir_lower_interpolation_centroid', 8)
nir_lower_interpolation_pixel = nir_lower_interpolation_options.define('nir_lower_interpolation_pixel', 16)
nir_lower_interpolation_sample = nir_lower_interpolation_options.define('nir_lower_interpolation_sample', 32)

try: (nir_lower_interpolation:=dll.nir_lower_interpolation).restype, nir_lower_interpolation.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_interpolation_options]
except AttributeError: pass

nir_lower_discard_if_options = CEnum(ctypes.c_uint32)
nir_lower_demote_if_to_cf = nir_lower_discard_if_options.define('nir_lower_demote_if_to_cf', 1)
nir_lower_terminate_if_to_cf = nir_lower_discard_if_options.define('nir_lower_terminate_if_to_cf', 2)
nir_move_terminate_out_of_loops = nir_lower_discard_if_options.define('nir_move_terminate_out_of_loops', 4)

try: (nir_lower_discard_if:=dll.nir_lower_discard_if).restype, nir_lower_discard_if.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_discard_if_options]
except AttributeError: pass

try: (nir_lower_terminate_to_demote:=dll.nir_lower_terminate_to_demote).restype, nir_lower_terminate_to_demote.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_memory_model:=dll.nir_lower_memory_model).restype, nir_lower_memory_model.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_goto_ifs:=dll.nir_lower_goto_ifs).restype, nir_lower_goto_ifs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_continue_constructs:=dll.nir_lower_continue_constructs).restype, nir_lower_continue_constructs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_multiview_options(Struct): pass
struct_nir_lower_multiview_options._fields_ = [
  ('view_mask', uint32_t),
  ('allowed_per_view_outputs', uint64_t),
]
nir_lower_multiview_options = struct_nir_lower_multiview_options
try: (nir_shader_uses_view_index:=dll.nir_shader_uses_view_index).restype, nir_shader_uses_view_index.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_can_lower_multiview:=dll.nir_can_lower_multiview).restype, nir_can_lower_multiview.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_multiview_options]
except AttributeError: pass

try: (nir_lower_multiview:=dll.nir_lower_multiview).restype, nir_lower_multiview.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_multiview_options]
except AttributeError: pass

try: (nir_lower_view_index_to_device_index:=dll.nir_lower_view_index_to_device_index).restype, nir_lower_view_index_to_device_index.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_fp16_cast_options = CEnum(ctypes.c_uint32)
nir_lower_fp16_rtz = nir_lower_fp16_cast_options.define('nir_lower_fp16_rtz', 1)
nir_lower_fp16_rtne = nir_lower_fp16_cast_options.define('nir_lower_fp16_rtne', 2)
nir_lower_fp16_ru = nir_lower_fp16_cast_options.define('nir_lower_fp16_ru', 4)
nir_lower_fp16_rd = nir_lower_fp16_cast_options.define('nir_lower_fp16_rd', 8)
nir_lower_fp16_all = nir_lower_fp16_cast_options.define('nir_lower_fp16_all', 15)
nir_lower_fp16_split_fp64 = nir_lower_fp16_cast_options.define('nir_lower_fp16_split_fp64', 16)

try: (nir_lower_fp16_casts:=dll.nir_lower_fp16_casts).restype, nir_lower_fp16_casts.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_fp16_cast_options]
except AttributeError: pass

try: (nir_normalize_cubemap_coords:=dll.nir_normalize_cubemap_coords).restype, nir_normalize_cubemap_coords.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_supports_implicit_lod:=dll.nir_shader_supports_implicit_lod).restype, nir_shader_supports_implicit_lod.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_live_defs_impl:=dll.nir_live_defs_impl).restype, nir_live_defs_impl.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_get_live_defs:=dll.nir_get_live_defs).restype, nir_get_live_defs.argtypes = ctypes.POINTER(ctypes.c_uint32), [nir_cursor, ctypes.c_void_p]
except AttributeError: pass

try: (nir_loop_analyze_impl:=dll.nir_loop_analyze_impl).restype, nir_loop_analyze_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_variable_mode, ctypes.c_bool]
except AttributeError: pass

try: (nir_defs_interfere:=dll.nir_defs_interfere).restype, nir_defs_interfere.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_repair_ssa_impl:=dll.nir_repair_ssa_impl).restype, nir_repair_ssa_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_repair_ssa:=dll.nir_repair_ssa).restype, nir_repair_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_convert_loop_to_lcssa:=dll.nir_convert_loop_to_lcssa).restype, nir_convert_loop_to_lcssa.argtypes = None, [ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_convert_to_lcssa:=dll.nir_convert_to_lcssa).restype, nir_convert_to_lcssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_divergence_analysis_impl:=dll.nir_divergence_analysis_impl).restype, nir_divergence_analysis_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_divergence_options]
except AttributeError: pass

try: (nir_divergence_analysis:=dll.nir_divergence_analysis).restype, nir_divergence_analysis.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_vertex_divergence_analysis:=dll.nir_vertex_divergence_analysis).restype, nir_vertex_divergence_analysis.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_has_divergent_loop:=dll.nir_has_divergent_loop).restype, nir_has_divergent_loop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_rewrite_uses_to_load_reg:=dll.nir_rewrite_uses_to_load_reg).restype, nir_rewrite_uses_to_load_reg.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_convert_from_ssa:=dll.nir_convert_from_ssa).restype, nir_convert_from_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_phis_to_regs_block:=dll.nir_lower_phis_to_regs_block).restype, nir_lower_phis_to_regs_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_ssa_defs_to_regs_block:=dll.nir_lower_ssa_defs_to_regs_block).restype, nir_lower_ssa_defs_to_regs_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_rematerialize_deref_in_use_blocks:=dll.nir_rematerialize_deref_in_use_blocks).restype, nir_rematerialize_deref_in_use_blocks.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_rematerialize_derefs_in_use_blocks_impl:=dll.nir_rematerialize_derefs_in_use_blocks_impl).restype, nir_rematerialize_derefs_in_use_blocks_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_samplers:=dll.nir_lower_samplers).restype, nir_lower_samplers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_cl_images:=dll.nir_lower_cl_images).restype, nir_lower_cl_images.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_dedup_inline_samplers:=dll.nir_dedup_inline_samplers).restype, nir_dedup_inline_samplers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_ssbo_options(Struct): pass
struct_nir_lower_ssbo_options._fields_ = [
  ('native_loads', ctypes.c_bool),
  ('native_offset', ctypes.c_bool),
]
nir_lower_ssbo_options = struct_nir_lower_ssbo_options
try: (nir_lower_ssbo:=dll.nir_lower_ssbo).restype, nir_lower_ssbo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_ssbo_options)]
except AttributeError: pass

try: (nir_lower_helper_writes:=dll.nir_lower_helper_writes).restype, nir_lower_helper_writes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_printf_options(Struct): pass
struct_nir_lower_printf_options._fields_ = [
  ('max_buffer_size', ctypes.c_uint32),
  ('ptr_bit_size', ctypes.c_uint32),
  ('hash_format_strings', ctypes.c_bool),
]
nir_lower_printf_options = struct_nir_lower_printf_options
try: (nir_lower_printf:=dll.nir_lower_printf).restype, nir_lower_printf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_printf_options)]
except AttributeError: pass

try: (nir_lower_printf_buffer:=dll.nir_lower_printf_buffer).restype, nir_lower_printf_buffer.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint64_t, uint32_t]
except AttributeError: pass

try: (nir_opt_comparison_pre_impl:=dll.nir_opt_comparison_pre_impl).restype, nir_opt_comparison_pre_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_opt_comparison_pre:=dll.nir_opt_comparison_pre).restype, nir_opt_comparison_pre.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_opt_access_options(Struct): pass
struct_nir_opt_access_options._fields_ = [
  ('is_vulkan', ctypes.c_bool),
]
nir_opt_access_options = struct_nir_opt_access_options
try: (nir_opt_access:=dll.nir_opt_access).restype, nir_opt_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_access_options)]
except AttributeError: pass

try: (nir_opt_algebraic:=dll.nir_opt_algebraic).restype, nir_opt_algebraic.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_before_ffma:=dll.nir_opt_algebraic_before_ffma).restype, nir_opt_algebraic_before_ffma.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_before_lower_int64:=dll.nir_opt_algebraic_before_lower_int64).restype, nir_opt_algebraic_before_lower_int64.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_late:=dll.nir_opt_algebraic_late).restype, nir_opt_algebraic_late.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_distribute_src_mods:=dll.nir_opt_algebraic_distribute_src_mods).restype, nir_opt_algebraic_distribute_src_mods.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_integer_promotion:=dll.nir_opt_algebraic_integer_promotion).restype, nir_opt_algebraic_integer_promotion.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_reassociate_matrix_mul:=dll.nir_opt_reassociate_matrix_mul).restype, nir_opt_reassociate_matrix_mul.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_constant_folding:=dll.nir_opt_constant_folding).restype, nir_opt_constant_folding.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_combine_barrier_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
try: (nir_opt_combine_barriers:=dll.nir_opt_combine_barriers).restype, nir_opt_combine_barriers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_combine_barrier_cb, ctypes.c_void_p]
except AttributeError: pass

mesa_scope = CEnum(ctypes.c_uint32)
SCOPE_NONE = mesa_scope.define('SCOPE_NONE', 0)
SCOPE_INVOCATION = mesa_scope.define('SCOPE_INVOCATION', 1)
SCOPE_SUBGROUP = mesa_scope.define('SCOPE_SUBGROUP', 2)
SCOPE_SHADER_CALL = mesa_scope.define('SCOPE_SHADER_CALL', 3)
SCOPE_WORKGROUP = mesa_scope.define('SCOPE_WORKGROUP', 4)
SCOPE_QUEUE_FAMILY = mesa_scope.define('SCOPE_QUEUE_FAMILY', 5)
SCOPE_DEVICE = mesa_scope.define('SCOPE_DEVICE', 6)

try: (nir_opt_acquire_release_barriers:=dll.nir_opt_acquire_release_barriers).restype, nir_opt_acquire_release_barriers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), mesa_scope]
except AttributeError: pass

try: (nir_opt_barrier_modes:=dll.nir_opt_barrier_modes).restype, nir_opt_barrier_modes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_minimize_call_live_states:=dll.nir_minimize_call_live_states).restype, nir_minimize_call_live_states.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_combine_stores:=dll.nir_opt_combine_stores).restype, nir_opt_combine_stores.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_copy_prop_impl:=dll.nir_copy_prop_impl).restype, nir_copy_prop_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_copy_prop:=dll.nir_copy_prop).restype, nir_copy_prop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_copy_prop_vars:=dll.nir_opt_copy_prop_vars).restype, nir_opt_copy_prop_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_cse:=dll.nir_opt_cse).restype, nir_opt_cse.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dce:=dll.nir_opt_dce).restype, nir_opt_dce.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dead_cf:=dll.nir_opt_dead_cf).restype, nir_opt_dead_cf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dead_write_vars:=dll.nir_opt_dead_write_vars).restype, nir_opt_dead_write_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_deref_impl:=dll.nir_opt_deref_impl).restype, nir_opt_deref_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_opt_deref:=dll.nir_opt_deref).restype, nir_opt_deref.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_find_array_copies:=dll.nir_opt_find_array_copies).restype, nir_opt_find_array_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_def_is_frag_coord_z:=dll.nir_def_is_frag_coord_z).restype, nir_def_is_frag_coord_z.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_opt_fragdepth:=dll.nir_opt_fragdepth).restype, nir_opt_fragdepth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_gcm:=dll.nir_opt_gcm).restype, nir_opt_gcm.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_generate_bfi:=dll.nir_opt_generate_bfi).restype, nir_opt_generate_bfi.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_idiv_const:=dll.nir_opt_idiv_const).restype, nir_opt_idiv_const.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_opt_mqsad:=dll.nir_opt_mqsad).restype, nir_opt_mqsad.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_opt_if_options = CEnum(ctypes.c_uint32)
nir_opt_if_optimize_phi_true_false = nir_opt_if_options.define('nir_opt_if_optimize_phi_true_false', 1)
nir_opt_if_avoid_64bit_phis = nir_opt_if_options.define('nir_opt_if_avoid_64bit_phis', 2)

try: (nir_opt_if:=dll.nir_opt_if).restype, nir_opt_if.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_opt_if_options]
except AttributeError: pass

try: (nir_opt_intrinsics:=dll.nir_opt_intrinsics).restype, nir_opt_intrinsics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_large_constants:=dll.nir_opt_large_constants).restype, nir_opt_large_constants.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), glsl_type_size_align_func, ctypes.c_uint32]
except AttributeError: pass

try: (nir_opt_licm:=dll.nir_opt_licm).restype, nir_opt_licm.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_loop:=dll.nir_opt_loop).restype, nir_opt_loop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_loop_unroll:=dll.nir_opt_loop_unroll).restype, nir_opt_loop_unroll.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_move_options = CEnum(ctypes.c_uint32)
nir_move_const_undef = nir_move_options.define('nir_move_const_undef', 1)
nir_move_load_ubo = nir_move_options.define('nir_move_load_ubo', 2)
nir_move_load_input = nir_move_options.define('nir_move_load_input', 4)
nir_move_comparisons = nir_move_options.define('nir_move_comparisons', 8)
nir_move_copies = nir_move_options.define('nir_move_copies', 16)
nir_move_load_ssbo = nir_move_options.define('nir_move_load_ssbo', 32)
nir_move_load_uniform = nir_move_options.define('nir_move_load_uniform', 64)
nir_move_alu = nir_move_options.define('nir_move_alu', 128)
nir_dont_move_byte_word_vecs = nir_move_options.define('nir_dont_move_byte_word_vecs', 256)

try: (nir_can_move_instr:=dll.nir_can_move_instr).restype, nir_can_move_instr.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), nir_move_options]
except AttributeError: pass

try: (nir_opt_sink:=dll.nir_opt_sink).restype, nir_opt_sink.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_move_options]
except AttributeError: pass

try: (nir_opt_move:=dll.nir_opt_move).restype, nir_opt_move.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_move_options]
except AttributeError: pass

class struct_nir_opt_offsets_options(Struct): pass
struct_nir_opt_offsets_options._fields_ = [
  ('uniform_max', uint32_t),
  ('ubo_vec4_max', uint32_t),
  ('shared_max', uint32_t),
  ('shared_atomic_max', uint32_t),
  ('buffer_max', uint32_t),
  ('max_offset_cb', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(nir_intrinsic_instr), ctypes.c_void_p)),
  ('max_offset_data', ctypes.c_void_p),
  ('allow_offset_wrap', ctypes.c_bool),
]
nir_opt_offsets_options = struct_nir_opt_offsets_options
try: (nir_opt_offsets:=dll.nir_opt_offsets).restype, nir_opt_offsets.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_offsets_options)]
except AttributeError: pass

class struct_nir_opt_peephole_select_options(Struct): pass
struct_nir_opt_peephole_select_options._fields_ = [
  ('limit', ctypes.c_uint32),
  ('indirect_load_ok', ctypes.c_bool),
  ('expensive_alu_ok', ctypes.c_bool),
  ('discard_ok', ctypes.c_bool),
]
nir_opt_peephole_select_options = struct_nir_opt_peephole_select_options
try: (nir_opt_peephole_select:=dll.nir_opt_peephole_select).restype, nir_opt_peephole_select.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_peephole_select_options)]
except AttributeError: pass

try: (nir_opt_reassociate_bfi:=dll.nir_opt_reassociate_bfi).restype, nir_opt_reassociate_bfi.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_rematerialize_compares:=dll.nir_opt_rematerialize_compares).restype, nir_opt_rematerialize_compares.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_remove_phis:=dll.nir_opt_remove_phis).restype, nir_opt_remove_phis.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_single_src_phis_block:=dll.nir_remove_single_src_phis_block).restype, nir_remove_single_src_phis_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_opt_phi_precision:=dll.nir_opt_phi_precision).restype, nir_opt_phi_precision.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_phi_to_bool:=dll.nir_opt_phi_to_bool).restype, nir_opt_phi_to_bool.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_shrink_stores:=dll.nir_opt_shrink_stores).restype, nir_opt_shrink_stores.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_shrink_vectors:=dll.nir_opt_shrink_vectors).restype, nir_opt_shrink_vectors.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_undef:=dll.nir_opt_undef).restype, nir_opt_undef.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_undef_to_zero:=dll.nir_lower_undef_to_zero).restype, nir_lower_undef_to_zero.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_uniform_atomics:=dll.nir_opt_uniform_atomics).restype, nir_opt_uniform_atomics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_uniform_subgroup:=dll.nir_opt_uniform_subgroup).restype, nir_opt_uniform_subgroup.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_subgroups_options)]
except AttributeError: pass

try: (nir_opt_vectorize:=dll.nir_opt_vectorize).restype, nir_opt_vectorize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_opt_vectorize_io:=dll.nir_opt_vectorize_io).restype, nir_opt_vectorize_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_move_discards_to_top:=dll.nir_opt_move_discards_to_top).restype, nir_opt_move_discards_to_top.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_ray_queries:=dll.nir_opt_ray_queries).restype, nir_opt_ray_queries.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_ray_query_ranges:=dll.nir_opt_ray_query_ranges).restype, nir_opt_ray_query_ranges.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_tex_skip_helpers:=dll.nir_opt_tex_skip_helpers).restype, nir_opt_tex_skip_helpers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_sweep:=dll.nir_sweep).restype, nir_sweep.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

gl_system_value = CEnum(ctypes.c_uint32)
SYSTEM_VALUE_SUBGROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_SIZE', 0)
SYSTEM_VALUE_SUBGROUP_INVOCATION = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_INVOCATION', 1)
SYSTEM_VALUE_SUBGROUP_EQ_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_EQ_MASK', 2)
SYSTEM_VALUE_SUBGROUP_GE_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_GE_MASK', 3)
SYSTEM_VALUE_SUBGROUP_GT_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_GT_MASK', 4)
SYSTEM_VALUE_SUBGROUP_LE_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_LE_MASK', 5)
SYSTEM_VALUE_SUBGROUP_LT_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_LT_MASK', 6)
SYSTEM_VALUE_NUM_SUBGROUPS = gl_system_value.define('SYSTEM_VALUE_NUM_SUBGROUPS', 7)
SYSTEM_VALUE_SUBGROUP_ID = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_ID', 8)
SYSTEM_VALUE_VERTEX_ID = gl_system_value.define('SYSTEM_VALUE_VERTEX_ID', 9)
SYSTEM_VALUE_INSTANCE_ID = gl_system_value.define('SYSTEM_VALUE_INSTANCE_ID', 10)
SYSTEM_VALUE_INSTANCE_INDEX = gl_system_value.define('SYSTEM_VALUE_INSTANCE_INDEX', 11)
SYSTEM_VALUE_VERTEX_ID_ZERO_BASE = gl_system_value.define('SYSTEM_VALUE_VERTEX_ID_ZERO_BASE', 12)
SYSTEM_VALUE_BASE_VERTEX = gl_system_value.define('SYSTEM_VALUE_BASE_VERTEX', 13)
SYSTEM_VALUE_FIRST_VERTEX = gl_system_value.define('SYSTEM_VALUE_FIRST_VERTEX', 14)
SYSTEM_VALUE_IS_INDEXED_DRAW = gl_system_value.define('SYSTEM_VALUE_IS_INDEXED_DRAW', 15)
SYSTEM_VALUE_BASE_INSTANCE = gl_system_value.define('SYSTEM_VALUE_BASE_INSTANCE', 16)
SYSTEM_VALUE_DRAW_ID = gl_system_value.define('SYSTEM_VALUE_DRAW_ID', 17)
SYSTEM_VALUE_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_INVOCATION_ID', 18)
SYSTEM_VALUE_FRAG_COORD = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD', 19)
SYSTEM_VALUE_PIXEL_COORD = gl_system_value.define('SYSTEM_VALUE_PIXEL_COORD', 20)
SYSTEM_VALUE_FRAG_COORD_Z = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD_Z', 21)
SYSTEM_VALUE_FRAG_COORD_W = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD_W', 22)
SYSTEM_VALUE_POINT_COORD = gl_system_value.define('SYSTEM_VALUE_POINT_COORD', 23)
SYSTEM_VALUE_LINE_COORD = gl_system_value.define('SYSTEM_VALUE_LINE_COORD', 24)
SYSTEM_VALUE_FRONT_FACE = gl_system_value.define('SYSTEM_VALUE_FRONT_FACE', 25)
SYSTEM_VALUE_FRONT_FACE_FSIGN = gl_system_value.define('SYSTEM_VALUE_FRONT_FACE_FSIGN', 26)
SYSTEM_VALUE_SAMPLE_ID = gl_system_value.define('SYSTEM_VALUE_SAMPLE_ID', 27)
SYSTEM_VALUE_SAMPLE_POS = gl_system_value.define('SYSTEM_VALUE_SAMPLE_POS', 28)
SYSTEM_VALUE_SAMPLE_POS_OR_CENTER = gl_system_value.define('SYSTEM_VALUE_SAMPLE_POS_OR_CENTER', 29)
SYSTEM_VALUE_SAMPLE_MASK_IN = gl_system_value.define('SYSTEM_VALUE_SAMPLE_MASK_IN', 30)
SYSTEM_VALUE_LAYER_ID = gl_system_value.define('SYSTEM_VALUE_LAYER_ID', 31)
SYSTEM_VALUE_HELPER_INVOCATION = gl_system_value.define('SYSTEM_VALUE_HELPER_INVOCATION', 32)
SYSTEM_VALUE_COLOR0 = gl_system_value.define('SYSTEM_VALUE_COLOR0', 33)
SYSTEM_VALUE_COLOR1 = gl_system_value.define('SYSTEM_VALUE_COLOR1', 34)
SYSTEM_VALUE_TESS_COORD = gl_system_value.define('SYSTEM_VALUE_TESS_COORD', 35)
SYSTEM_VALUE_VERTICES_IN = gl_system_value.define('SYSTEM_VALUE_VERTICES_IN', 36)
SYSTEM_VALUE_PRIMITIVE_ID = gl_system_value.define('SYSTEM_VALUE_PRIMITIVE_ID', 37)
SYSTEM_VALUE_TESS_LEVEL_OUTER = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_OUTER', 38)
SYSTEM_VALUE_TESS_LEVEL_INNER = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_INNER', 39)
SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT', 40)
SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT', 41)
SYSTEM_VALUE_LOCAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_LOCAL_INVOCATION_ID', 42)
SYSTEM_VALUE_LOCAL_INVOCATION_INDEX = gl_system_value.define('SYSTEM_VALUE_LOCAL_INVOCATION_INDEX', 43)
SYSTEM_VALUE_GLOBAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_GLOBAL_INVOCATION_ID', 44)
SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID', 45)
SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX = gl_system_value.define('SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX', 46)
SYSTEM_VALUE_WORKGROUP_ID = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_ID', 47)
SYSTEM_VALUE_BASE_WORKGROUP_ID = gl_system_value.define('SYSTEM_VALUE_BASE_WORKGROUP_ID', 48)
SYSTEM_VALUE_WORKGROUP_INDEX = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_INDEX', 49)
SYSTEM_VALUE_NUM_WORKGROUPS = gl_system_value.define('SYSTEM_VALUE_NUM_WORKGROUPS', 50)
SYSTEM_VALUE_WORKGROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_SIZE', 51)
SYSTEM_VALUE_GLOBAL_GROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_GLOBAL_GROUP_SIZE', 52)
SYSTEM_VALUE_WORK_DIM = gl_system_value.define('SYSTEM_VALUE_WORK_DIM', 53)
SYSTEM_VALUE_USER_DATA_AMD = gl_system_value.define('SYSTEM_VALUE_USER_DATA_AMD', 54)
SYSTEM_VALUE_DEVICE_INDEX = gl_system_value.define('SYSTEM_VALUE_DEVICE_INDEX', 55)
SYSTEM_VALUE_VIEW_INDEX = gl_system_value.define('SYSTEM_VALUE_VIEW_INDEX', 56)
SYSTEM_VALUE_VERTEX_CNT = gl_system_value.define('SYSTEM_VALUE_VERTEX_CNT', 57)
SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL', 58)
SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE', 59)
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID', 60)
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW', 61)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL', 62)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID', 63)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE', 64)
SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL', 65)
SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD', 66)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD', 67)
SYSTEM_VALUE_RAY_LAUNCH_ID = gl_system_value.define('SYSTEM_VALUE_RAY_LAUNCH_ID', 68)
SYSTEM_VALUE_RAY_LAUNCH_SIZE = gl_system_value.define('SYSTEM_VALUE_RAY_LAUNCH_SIZE', 69)
SYSTEM_VALUE_RAY_WORLD_ORIGIN = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_ORIGIN', 70)
SYSTEM_VALUE_RAY_WORLD_DIRECTION = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_DIRECTION', 71)
SYSTEM_VALUE_RAY_OBJECT_ORIGIN = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_ORIGIN', 72)
SYSTEM_VALUE_RAY_OBJECT_DIRECTION = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_DIRECTION', 73)
SYSTEM_VALUE_RAY_T_MIN = gl_system_value.define('SYSTEM_VALUE_RAY_T_MIN', 74)
SYSTEM_VALUE_RAY_T_MAX = gl_system_value.define('SYSTEM_VALUE_RAY_T_MAX', 75)
SYSTEM_VALUE_RAY_OBJECT_TO_WORLD = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_TO_WORLD', 76)
SYSTEM_VALUE_RAY_WORLD_TO_OBJECT = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_TO_OBJECT', 77)
SYSTEM_VALUE_RAY_HIT_KIND = gl_system_value.define('SYSTEM_VALUE_RAY_HIT_KIND', 78)
SYSTEM_VALUE_RAY_FLAGS = gl_system_value.define('SYSTEM_VALUE_RAY_FLAGS', 79)
SYSTEM_VALUE_RAY_GEOMETRY_INDEX = gl_system_value.define('SYSTEM_VALUE_RAY_GEOMETRY_INDEX', 80)
SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX = gl_system_value.define('SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX', 81)
SYSTEM_VALUE_CULL_MASK = gl_system_value.define('SYSTEM_VALUE_CULL_MASK', 82)
SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS = gl_system_value.define('SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS', 83)
SYSTEM_VALUE_MESH_VIEW_COUNT = gl_system_value.define('SYSTEM_VALUE_MESH_VIEW_COUNT', 84)
SYSTEM_VALUE_MESH_VIEW_INDICES = gl_system_value.define('SYSTEM_VALUE_MESH_VIEW_INDICES', 85)
SYSTEM_VALUE_GS_HEADER_IR3 = gl_system_value.define('SYSTEM_VALUE_GS_HEADER_IR3', 86)
SYSTEM_VALUE_TCS_HEADER_IR3 = gl_system_value.define('SYSTEM_VALUE_TCS_HEADER_IR3', 87)
SYSTEM_VALUE_REL_PATCH_ID_IR3 = gl_system_value.define('SYSTEM_VALUE_REL_PATCH_ID_IR3', 88)
SYSTEM_VALUE_FRAG_SHADING_RATE = gl_system_value.define('SYSTEM_VALUE_FRAG_SHADING_RATE', 89)
SYSTEM_VALUE_FULLY_COVERED = gl_system_value.define('SYSTEM_VALUE_FULLY_COVERED', 90)
SYSTEM_VALUE_FRAG_SIZE = gl_system_value.define('SYSTEM_VALUE_FRAG_SIZE', 91)
SYSTEM_VALUE_FRAG_INVOCATION_COUNT = gl_system_value.define('SYSTEM_VALUE_FRAG_INVOCATION_COUNT', 92)
SYSTEM_VALUE_SHADER_INDEX = gl_system_value.define('SYSTEM_VALUE_SHADER_INDEX', 93)
SYSTEM_VALUE_COALESCED_INPUT_COUNT = gl_system_value.define('SYSTEM_VALUE_COALESCED_INPUT_COUNT', 94)
SYSTEM_VALUE_WARPS_PER_SM_NV = gl_system_value.define('SYSTEM_VALUE_WARPS_PER_SM_NV', 95)
SYSTEM_VALUE_SM_COUNT_NV = gl_system_value.define('SYSTEM_VALUE_SM_COUNT_NV', 96)
SYSTEM_VALUE_WARP_ID_NV = gl_system_value.define('SYSTEM_VALUE_WARP_ID_NV', 97)
SYSTEM_VALUE_SM_ID_NV = gl_system_value.define('SYSTEM_VALUE_SM_ID_NV', 98)
SYSTEM_VALUE_MAX = gl_system_value.define('SYSTEM_VALUE_MAX', 99)

try: (nir_intrinsic_from_system_value:=dll.nir_intrinsic_from_system_value).restype, nir_intrinsic_from_system_value.argtypes = nir_intrinsic_op, [gl_system_value]
except AttributeError: pass

try: (nir_system_value_from_intrinsic:=dll.nir_system_value_from_intrinsic).restype, nir_system_value_from_intrinsic.argtypes = gl_system_value, [nir_intrinsic_op]
except AttributeError: pass

class struct_nir_unsigned_upper_bound_config(Struct): pass
struct_nir_unsigned_upper_bound_config._fields_ = [
  ('min_subgroup_size', ctypes.c_uint32),
  ('max_subgroup_size', ctypes.c_uint32),
  ('max_workgroup_invocations', ctypes.c_uint32),
  ('max_workgroup_count', (ctypes.c_uint32 * 3)),
  ('max_workgroup_size', (ctypes.c_uint32 * 3)),
  ('vertex_attrib_max', (uint32_t * 32)),
]
nir_unsigned_upper_bound_config = struct_nir_unsigned_upper_bound_config
try: (nir_unsigned_upper_bound:=dll.nir_unsigned_upper_bound).restype, nir_unsigned_upper_bound.argtypes = uint32_t, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.POINTER(nir_unsigned_upper_bound_config)]
except AttributeError: pass

try: (nir_addition_might_overflow:=dll.nir_addition_might_overflow).restype, nir_addition_might_overflow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.c_uint32, ctypes.POINTER(nir_unsigned_upper_bound_config)]
except AttributeError: pass

class struct_nir_opt_preamble_options(Struct): pass
struct_nir_opt_preamble_options._fields_ = [
  ('drawid_uniform', ctypes.c_bool),
  ('subgroup_size_uniform', ctypes.c_bool),
  ('load_workgroup_size_allowed', ctypes.c_bool),
  ('def_size', ctypes.CFUNCTYPE(None, ctypes.POINTER(nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(nir_preamble_class))),
  ('preamble_storage_size', (ctypes.c_uint32 * 2)),
  ('instr_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(nir_instr), ctypes.c_void_p)),
  ('rewrite_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(nir_def), ctypes.c_void_p)),
  ('avoid_instr_cb', nir_instr_filter_cb),
  ('cb_data', ctypes.c_void_p),
]
nir_opt_preamble_options = struct_nir_opt_preamble_options
try: (nir_opt_preamble:=dll.nir_opt_preamble).restype, nir_opt_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_preamble_options), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_shader_get_preamble:=dll.nir_shader_get_preamble).restype, nir_shader_get_preamble.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_point_smooth:=dll.nir_lower_point_smooth).restype, nir_lower_point_smooth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_poly_line_smooth:=dll.nir_lower_poly_line_smooth).restype, nir_lower_poly_line_smooth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_mod_analysis:=dll.nir_mod_analysis).restype, nir_mod_analysis.argtypes = ctypes.c_bool, [nir_scalar, nir_alu_type, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_remove_tex_shadow:=dll.nir_remove_tex_shadow).restype, nir_remove_tex_shadow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_trivialize_registers:=dll.nir_trivialize_registers).restype, nir_trivialize_registers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_static_workgroup_size:=dll.nir_static_workgroup_size).restype, nir_static_workgroup_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_use_dominance_state(Struct): pass
nir_use_dominance_state = struct_nir_use_dominance_state
try: (nir_calc_use_dominance_impl:=dll.nir_calc_use_dominance_impl).restype, nir_calc_use_dominance_impl.argtypes = ctypes.POINTER(nir_use_dominance_state), [ctypes.POINTER(nir_function_impl), ctypes.c_bool]
except AttributeError: pass

try: (nir_get_immediate_use_dominator:=dll.nir_get_immediate_use_dominator).restype, nir_get_immediate_use_dominator.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_use_dominance_lca:=dll.nir_use_dominance_lca).restype, nir_use_dominance_lca.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_dominates_use:=dll.nir_instr_dominates_use).restype, nir_instr_dominates_use.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_print_use_dominators:=dll.nir_print_use_dominators).restype, nir_print_use_dominators.argtypes = None, [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(ctypes.POINTER(nir_instr)), ctypes.c_uint32]
except AttributeError: pass

class nir_output_deps(Struct): pass
class nir_output_deps_output(Struct): pass
nir_output_deps_output._fields_ = [
  ('instr_list', ctypes.POINTER(ctypes.POINTER(nir_instr))),
  ('num_instr', ctypes.c_uint32),
]
nir_output_deps._fields_ = [
  ('output', (nir_output_deps_output * 112)),
]
try: (nir_gather_output_dependencies:=dll.nir_gather_output_dependencies).restype, nir_gather_output_dependencies.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_output_deps)]
except AttributeError: pass

try: (nir_free_output_dependencies:=dll.nir_free_output_dependencies).restype, nir_free_output_dependencies.argtypes = None, [ctypes.POINTER(nir_output_deps)]
except AttributeError: pass

class nir_input_to_output_deps(Struct): pass
class nir_input_to_output_deps_output(Struct): pass
nir_input_to_output_deps_output._fields_ = [
  ('inputs', (ctypes.c_uint32 * 28)),
  ('defined', ctypes.c_bool),
  ('uses_ssbo_reads', ctypes.c_bool),
  ('uses_image_reads', ctypes.c_bool),
]
nir_input_to_output_deps._fields_ = [
  ('output', (nir_input_to_output_deps_output * 112)),
]
try: (nir_gather_input_to_output_dependencies:=dll.nir_gather_input_to_output_dependencies).restype, nir_gather_input_to_output_dependencies.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_input_to_output_deps)]
except AttributeError: pass

try: (nir_print_input_to_output_deps:=dll.nir_print_input_to_output_deps).restype, nir_print_input_to_output_deps.argtypes = None, [ctypes.POINTER(nir_input_to_output_deps), ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

class nir_output_clipper_var_groups(Struct): pass
nir_output_clipper_var_groups._fields_ = [
  ('pos_only', (ctypes.c_uint32 * 28)),
  ('var_only', (ctypes.c_uint32 * 28)),
  ('both', (ctypes.c_uint32 * 28)),
]
try: (nir_gather_output_clipper_var_groups:=dll.nir_gather_output_clipper_var_groups).restype, nir_gather_output_clipper_var_groups.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_output_clipper_var_groups)]
except AttributeError: pass

struct_nir_builder._fields_ = [
  ('cursor', nir_cursor),
  ('exact', ctypes.c_bool),
  ('fp_fast_math', uint32_t),
  ('shader', ctypes.POINTER(nir_shader)),
  ('impl', ctypes.POINTER(nir_function_impl)),
]
try: (nir_builder_init_simple_shader:=dll.nir_builder_init_simple_shader).restype, nir_builder_init_simple_shader.argtypes = nir_builder, [gl_shader_stage, ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

nir_instr_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
nir_intrinsic_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
nir_alu_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_void_p)
nir_tex_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_tex_instr), ctypes.c_void_p)
nir_phi_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_phi_instr), ctypes.c_void_p)
try: (nir_builder_instr_insert:=dll.nir_builder_instr_insert).restype, nir_builder_instr_insert.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_builder_instr_insert_at_top:=dll.nir_builder_instr_insert_at_top).restype, nir_builder_instr_insert_at_top.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_build_alu:=dll.nir_build_alu).restype, nir_build_alu.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu1:=dll.nir_build_alu1).restype, nir_build_alu1.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu2:=dll.nir_build_alu2).restype, nir_build_alu2.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu3:=dll.nir_build_alu3).restype, nir_build_alu3.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu4:=dll.nir_build_alu4).restype, nir_build_alu4.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu_src_arr:=dll.nir_build_alu_src_arr).restype, nir_build_alu_src_arr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

try: (nir_build_tex_deref_instr:=dll.nir_build_tex_deref_instr).restype, nir_build_tex_deref_instr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_texop, ctypes.POINTER(nir_deref_instr), ctypes.POINTER(nir_deref_instr), ctypes.c_uint32, ctypes.POINTER(nir_tex_src)]
except AttributeError: pass

try: (nir_builder_cf_insert:=dll.nir_builder_cf_insert).restype, nir_builder_cf_insert.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_builder_is_inside_cf:=dll.nir_builder_is_inside_cf).restype, nir_builder_is_inside_cf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_push_if:=dll.nir_push_if).restype, nir_push_if.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_push_else:=dll.nir_push_else).restype, nir_push_else.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_if)]
except AttributeError: pass

try: (nir_pop_if:=dll.nir_pop_if).restype, nir_pop_if.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_if)]
except AttributeError: pass

try: (nir_if_phi:=dll.nir_if_phi).restype, nir_if_phi.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_push_loop:=dll.nir_push_loop).restype, nir_push_loop.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_builder)]
except AttributeError: pass

try: (nir_push_continue:=dll.nir_push_continue).restype, nir_push_continue.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_pop_loop:=dll.nir_pop_loop).restype, nir_pop_loop.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_builder_alu_instr_finish_and_insert:=dll.nir_builder_alu_instr_finish_and_insert).restype, nir_builder_alu_instr_finish_and_insert.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_build_alu_src_arr:=dll.nir_build_alu_src_arr).restype, nir_build_alu_src_arr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

try: (nir_load_system_value:=dll.nir_load_system_value).restype, nir_load_system_value.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_intrinsic_op, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_type_convert:=dll.nir_type_convert).restype, nir_type_convert.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError: pass

try: (nir_vec_scalars:=dll.nir_vec_scalars).restype, nir_vec_scalars.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_scalar), ctypes.c_uint32]
except AttributeError: pass

try: (nir_ssa_for_alu_src:=dll.nir_ssa_for_alu_src).restype, nir_ssa_for_alu_src.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_build_string:=dll.nir_build_string).restype, nir_build_string.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_compare_func:=dll.nir_compare_func).restype, nir_compare_func.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), enum_compare_func, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_gen_rect_vertices:=dll.nir_gen_rect_vertices).restype, nir_gen_rect_vertices.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_printf_fmt:=dll.nir_printf_fmt).restype, nir_printf_fmt.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_printf_fmt_at_px:=dll.nir_printf_fmt_at_px).restype, nir_printf_fmt_at_px.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_call_serialized:=dll.nir_call_serialized).restype, nir_call_serialized.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(uint32_t), size_t, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

nir_lower_packing_op = CEnum(ctypes.c_uint32)
nir_lower_packing_op_pack_64_2x32 = nir_lower_packing_op.define('nir_lower_packing_op_pack_64_2x32', 0)
nir_lower_packing_op_unpack_64_2x32 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_64_2x32', 1)
nir_lower_packing_op_pack_64_4x16 = nir_lower_packing_op.define('nir_lower_packing_op_pack_64_4x16', 2)
nir_lower_packing_op_unpack_64_4x16 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_64_4x16', 3)
nir_lower_packing_op_pack_32_2x16 = nir_lower_packing_op.define('nir_lower_packing_op_pack_32_2x16', 4)
nir_lower_packing_op_unpack_32_2x16 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_32_2x16', 5)
nir_lower_packing_op_pack_32_4x8 = nir_lower_packing_op.define('nir_lower_packing_op_pack_32_4x8', 6)
nir_lower_packing_op_unpack_32_4x8 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_32_4x8', 7)
nir_lower_packing_num_ops = nir_lower_packing_op.define('nir_lower_packing_num_ops', 8)

class struct_blob(Struct): pass
struct_blob._fields_ = [
  ('data', ctypes.POINTER(uint8_t)),
  ('allocated', size_t),
  ('size', size_t),
  ('fixed_allocation', ctypes.c_bool),
  ('out_of_memory', ctypes.c_bool),
]
try: (nir_serialize:=dll.nir_serialize).restype, nir_serialize.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_blob_reader(Struct): pass
struct_blob_reader._fields_ = [
  ('data', ctypes.POINTER(uint8_t)),
  ('end', ctypes.POINTER(uint8_t)),
  ('current', ctypes.POINTER(uint8_t)),
  ('overrun', ctypes.c_bool),
]
try: (nir_deserialize:=dll.nir_deserialize).restype, nir_deserialize.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (nir_serialize_function:=dll.nir_serialize_function).restype, nir_serialize_function.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_deserialize_function:=dll.nir_deserialize_function).restype, nir_deserialize_function.argtypes = ctypes.POINTER(nir_function), [ctypes.c_void_p, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

nir_intrinsic_index_flag = CEnum(ctypes.c_uint32)
NIR_INTRINSIC_BASE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BASE', 0)
NIR_INTRINSIC_WRITE_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_WRITE_MASK', 1)
NIR_INTRINSIC_STREAM_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_STREAM_ID', 2)
NIR_INTRINSIC_UCP_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_UCP_ID', 3)
NIR_INTRINSIC_RANGE_BASE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RANGE_BASE', 4)
NIR_INTRINSIC_RANGE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RANGE', 5)
NIR_INTRINSIC_DESC_SET = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DESC_SET', 6)
NIR_INTRINSIC_BINDING = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BINDING', 7)
NIR_INTRINSIC_COMPONENT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COMPONENT', 8)
NIR_INTRINSIC_COLUMN = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COLUMN', 9)
NIR_INTRINSIC_INTERP_MODE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_INTERP_MODE', 10)
NIR_INTRINSIC_REDUCTION_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_REDUCTION_OP', 11)
NIR_INTRINSIC_CLUSTER_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CLUSTER_SIZE', 12)
NIR_INTRINSIC_PARAM_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_PARAM_IDX', 13)
NIR_INTRINSIC_IMAGE_DIM = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IMAGE_DIM', 14)
NIR_INTRINSIC_IMAGE_ARRAY = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IMAGE_ARRAY', 15)
NIR_INTRINSIC_FORMAT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FORMAT', 16)
NIR_INTRINSIC_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ACCESS', 17)
NIR_INTRINSIC_CALL_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CALL_IDX', 18)
NIR_INTRINSIC_STACK_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_STACK_SIZE', 19)
NIR_INTRINSIC_ALIGN_MUL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALIGN_MUL', 20)
NIR_INTRINSIC_ALIGN_OFFSET = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALIGN_OFFSET', 21)
NIR_INTRINSIC_DESC_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DESC_TYPE', 22)
NIR_INTRINSIC_SRC_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_TYPE', 23)
NIR_INTRINSIC_DEST_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DEST_TYPE', 24)
NIR_INTRINSIC_SRC_BASE_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_BASE_TYPE', 25)
NIR_INTRINSIC_SRC_BASE_TYPE2 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_BASE_TYPE2', 26)
NIR_INTRINSIC_DEST_BASE_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DEST_BASE_TYPE', 27)
NIR_INTRINSIC_SWIZZLE_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SWIZZLE_MASK', 28)
NIR_INTRINSIC_FETCH_INACTIVE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FETCH_INACTIVE', 29)
NIR_INTRINSIC_OFFSET0 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_OFFSET0', 30)
NIR_INTRINSIC_OFFSET1 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_OFFSET1', 31)
NIR_INTRINSIC_ST64 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ST64', 32)
NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD', 33)
NIR_INTRINSIC_DST_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DST_ACCESS', 34)
NIR_INTRINSIC_SRC_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_ACCESS', 35)
NIR_INTRINSIC_DRIVER_LOCATION = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DRIVER_LOCATION', 36)
NIR_INTRINSIC_MEMORY_SEMANTICS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_SEMANTICS', 37)
NIR_INTRINSIC_MEMORY_MODES = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_MODES', 38)
NIR_INTRINSIC_MEMORY_SCOPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_SCOPE', 39)
NIR_INTRINSIC_EXECUTION_SCOPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_EXECUTION_SCOPE', 40)
NIR_INTRINSIC_IO_SEMANTICS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_SEMANTICS', 41)
NIR_INTRINSIC_IO_XFB = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_XFB', 42)
NIR_INTRINSIC_IO_XFB2 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_XFB2', 43)
NIR_INTRINSIC_RAY_QUERY_VALUE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RAY_QUERY_VALUE', 44)
NIR_INTRINSIC_COMMITTED = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COMMITTED', 45)
NIR_INTRINSIC_ROUNDING_MODE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ROUNDING_MODE', 46)
NIR_INTRINSIC_SATURATE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SATURATE', 47)
NIR_INTRINSIC_SYNCHRONOUS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SYNCHRONOUS', 48)
NIR_INTRINSIC_VALUE_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_VALUE_ID', 49)
NIR_INTRINSIC_SIGN_EXTEND = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SIGN_EXTEND', 50)
NIR_INTRINSIC_FLAGS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FLAGS', 51)
NIR_INTRINSIC_ATOMIC_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ATOMIC_OP', 52)
NIR_INTRINSIC_RESOURCE_BLOCK_INTEL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RESOURCE_BLOCK_INTEL', 53)
NIR_INTRINSIC_RESOURCE_ACCESS_INTEL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RESOURCE_ACCESS_INTEL', 54)
NIR_INTRINSIC_NUM_COMPONENTS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_COMPONENTS', 55)
NIR_INTRINSIC_NUM_ARRAY_ELEMS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_ARRAY_ELEMS', 56)
NIR_INTRINSIC_BIT_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BIT_SIZE', 57)
NIR_INTRINSIC_DIVERGENT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DIVERGENT', 58)
NIR_INTRINSIC_LEGACY_FABS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FABS', 59)
NIR_INTRINSIC_LEGACY_FNEG = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FNEG', 60)
NIR_INTRINSIC_LEGACY_FSAT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FSAT', 61)
NIR_INTRINSIC_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CMAT_DESC', 62)
NIR_INTRINSIC_MATRIX_LAYOUT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MATRIX_LAYOUT', 63)
NIR_INTRINSIC_CMAT_SIGNED_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CMAT_SIGNED_MASK', 64)
NIR_INTRINSIC_ALU_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALU_OP', 65)
NIR_INTRINSIC_NEG_LO_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NEG_LO_AMD', 66)
NIR_INTRINSIC_NEG_HI_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NEG_HI_AMD', 67)
NIR_INTRINSIC_SYSTOLIC_DEPTH = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SYSTOLIC_DEPTH', 68)
NIR_INTRINSIC_REPEAT_COUNT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_REPEAT_COUNT', 69)
NIR_INTRINSIC_DST_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DST_CMAT_DESC', 70)
NIR_INTRINSIC_SRC_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_CMAT_DESC', 71)
NIR_INTRINSIC_EXPLICIT_COORD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_EXPLICIT_COORD', 72)
NIR_INTRINSIC_FMT_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FMT_IDX', 73)
NIR_INTRINSIC_PREAMBLE_CLASS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_PREAMBLE_CLASS', 74)
NIR_INTRINSIC_NUM_INDEX_FLAGS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_INDEX_FLAGS', 75)

try: nir_intrinsic_index_names = (ctypes.POINTER(ctypes.c_char) * 75).in_dll(dll, 'nir_intrinsic_index_names')
except (ValueError,AttributeError): pass
enum_nv_device_type = CEnum(ctypes.c_ubyte)
NV_DEVICE_TYPE_IGP = enum_nv_device_type.define('NV_DEVICE_TYPE_IGP', 0)
NV_DEVICE_TYPE_DIS = enum_nv_device_type.define('NV_DEVICE_TYPE_DIS', 1)
NV_DEVICE_TYPE_SOC = enum_nv_device_type.define('NV_DEVICE_TYPE_SOC', 2)

class struct_nv_device_info(Struct): pass
class struct_nv_device_info_pci(Struct): pass
struct_nv_device_info_pci._fields_ = [
  ('domain', uint16_t),
  ('bus', uint8_t),
  ('dev', uint8_t),
  ('func', uint8_t),
  ('revision_id', uint8_t),
]
struct_nv_device_info._fields_ = [
  ('type', enum_nv_device_type),
  ('device_id', uint16_t),
  ('chipset', uint16_t),
  ('device_name', (ctypes.c_char * 64)),
  ('chipset_name', (ctypes.c_char * 16)),
  ('pci', struct_nv_device_info_pci),
  ('sm', uint8_t),
  ('gpc_count', uint8_t),
  ('tpc_count', uint16_t),
  ('mp_per_tpc', uint8_t),
  ('max_warps_per_mp', uint8_t),
  ('cls_copy', uint16_t),
  ('cls_eng2d', uint16_t),
  ('cls_eng3d', uint16_t),
  ('cls_m2mf', uint16_t),
  ('cls_compute', uint16_t),
  ('vram_size_B', uint64_t),
  ('bar_size_B', uint64_t),
]
class struct_nak_compiler(Struct): pass
try: (nak_compiler_create:=dll.nak_compiler_create).restype, nak_compiler_create.argtypes = ctypes.POINTER(struct_nak_compiler), [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

try: (nak_compiler_destroy:=dll.nak_compiler_destroy).restype, nak_compiler_destroy.argtypes = None, [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_debug_flags:=dll.nak_debug_flags).restype, nak_debug_flags.argtypes = uint64_t, [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_nir_options:=dll.nak_nir_options).restype, nak_nir_options.argtypes = ctypes.POINTER(struct_nir_shader_compiler_options), [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_preprocess_nir:=dll.nak_preprocess_nir).restype, nak_preprocess_nir.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_nir_lower_image_addrs:=dll.nak_nir_lower_image_addrs).restype, nak_nir_lower_image_addrs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

class struct_nak_sample_location(Struct): pass
struct_nak_sample_location._fields_ = [
  ('x_u4', uint8_t,4),
  ('y_u4', uint8_t,4),
]
class struct_nak_sample_mask(Struct): pass
struct_nak_sample_mask._fields_ = [
  ('sample_mask', uint16_t),
]
class struct_nak_fs_key(Struct): pass
struct_nak_fs_key._fields_ = [
  ('zs_self_dep', ctypes.c_bool),
  ('force_sample_shading', ctypes.c_bool),
  ('uses_underestimate', ctypes.c_bool),
  ('sample_info_cb', uint8_t),
  ('sample_locations_offset', uint32_t),
  ('sample_masks_offset', uint32_t),
]
try: (nak_postprocess_nir:=dll.nak_postprocess_nir).restype, nak_postprocess_nir.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError: pass

enum_nak_ts_domain = CEnum(ctypes.c_ubyte)
NAK_TS_DOMAIN_ISOLINE = enum_nak_ts_domain.define('NAK_TS_DOMAIN_ISOLINE', 0)
NAK_TS_DOMAIN_TRIANGLE = enum_nak_ts_domain.define('NAK_TS_DOMAIN_TRIANGLE', 1)
NAK_TS_DOMAIN_QUAD = enum_nak_ts_domain.define('NAK_TS_DOMAIN_QUAD', 2)

enum_nak_ts_spacing = CEnum(ctypes.c_ubyte)
NAK_TS_SPACING_INTEGER = enum_nak_ts_spacing.define('NAK_TS_SPACING_INTEGER', 0)
NAK_TS_SPACING_FRACT_ODD = enum_nak_ts_spacing.define('NAK_TS_SPACING_FRACT_ODD', 1)
NAK_TS_SPACING_FRACT_EVEN = enum_nak_ts_spacing.define('NAK_TS_SPACING_FRACT_EVEN', 2)

enum_nak_ts_prims = CEnum(ctypes.c_ubyte)
NAK_TS_PRIMS_POINTS = enum_nak_ts_prims.define('NAK_TS_PRIMS_POINTS', 0)
NAK_TS_PRIMS_LINES = enum_nak_ts_prims.define('NAK_TS_PRIMS_LINES', 1)
NAK_TS_PRIMS_TRIANGLES_CW = enum_nak_ts_prims.define('NAK_TS_PRIMS_TRIANGLES_CW', 2)
NAK_TS_PRIMS_TRIANGLES_CCW = enum_nak_ts_prims.define('NAK_TS_PRIMS_TRIANGLES_CCW', 3)

class struct_nak_xfb_info(Struct): pass
struct_nak_xfb_info._fields_ = [
  ('stride', (uint32_t * 4)),
  ('stream', (uint8_t * 4)),
  ('attr_count', (uint8_t * 4)),
  ('attr_index', ((uint8_t * 128) * 4)),
]
class struct_nak_shader_info(Struct): pass
class struct_nak_shader_info_0(ctypes.Union): pass
class struct_nak_shader_info_0_cs(Struct): pass
struct_nak_shader_info_0_cs._fields_ = [
  ('local_size', (uint16_t * 3)),
  ('smem_size', uint16_t),
  ('_pad', (uint8_t * 4)),
]
class struct_nak_shader_info_0_fs(Struct): pass
struct_nak_shader_info_0_fs._fields_ = [
  ('writes_depth', ctypes.c_bool),
  ('reads_sample_mask', ctypes.c_bool),
  ('post_depth_coverage', ctypes.c_bool),
  ('uses_sample_shading', ctypes.c_bool),
  ('early_fragment_tests', ctypes.c_bool),
  ('_pad', (uint8_t * 7)),
]
class struct_nak_shader_info_0_ts(Struct): pass
struct_nak_shader_info_0_ts._fields_ = [
  ('domain', enum_nak_ts_domain),
  ('spacing', enum_nak_ts_spacing),
  ('prims', enum_nak_ts_prims),
  ('_pad', (uint8_t * 9)),
]
struct_nak_shader_info_0._fields_ = [
  ('cs', struct_nak_shader_info_0_cs),
  ('fs', struct_nak_shader_info_0_fs),
  ('ts', struct_nak_shader_info_0_ts),
  ('_pad', (uint8_t * 12)),
]
class struct_nak_shader_info_vtg(Struct): pass
struct_nak_shader_info_vtg._fields_ = [
  ('writes_layer', ctypes.c_bool),
  ('writes_point_size', ctypes.c_bool),
  ('writes_vprs_table_index', ctypes.c_bool),
  ('clip_enable', uint8_t),
  ('cull_enable', uint8_t),
  ('_pad', (uint8_t * 3)),
  ('xfb', struct_nak_xfb_info),
]
struct_nak_shader_info._anonymous_ = ['_0']
struct_nak_shader_info._fields_ = [
  ('stage', gl_shader_stage),
  ('sm', uint8_t),
  ('num_gprs', uint8_t),
  ('num_control_barriers', uint8_t),
  ('_pad0', uint8_t),
  ('max_warps_per_sm', uint32_t),
  ('num_instrs', uint32_t),
  ('num_static_cycles', uint32_t),
  ('num_spills_to_mem', uint32_t),
  ('num_fills_from_mem', uint32_t),
  ('num_spills_to_reg', uint32_t),
  ('num_fills_from_reg', uint32_t),
  ('slm_size', uint32_t),
  ('crs_size', uint32_t),
  ('_0', struct_nak_shader_info_0),
  ('vtg', struct_nak_shader_info_vtg),
  ('hdr', (uint32_t * 32)),
]
class struct_nak_shader_bin(Struct): pass
struct_nak_shader_bin._fields_ = [
  ('info', struct_nak_shader_info),
  ('code_size', uint32_t),
  ('code', ctypes.c_void_p),
  ('asm_str', ctypes.POINTER(ctypes.c_char)),
]
try: (nak_shader_bin_destroy:=dll.nak_shader_bin_destroy).restype, nak_shader_bin_destroy.argtypes = None, [ctypes.POINTER(struct_nak_shader_bin)]
except AttributeError: pass

try: (nak_compile_shader:=dll.nak_compile_shader).restype, nak_compile_shader.argtypes = ctypes.POINTER(struct_nak_shader_bin), [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError: pass

class struct_nak_qmd_cbuf(Struct): pass
struct_nak_qmd_cbuf._fields_ = [
  ('index', uint32_t),
  ('size', uint32_t),
  ('addr', uint64_t),
]
class struct_nak_qmd_info(Struct): pass
struct_nak_qmd_info._fields_ = [
  ('addr', uint64_t),
  ('smem_size', uint16_t),
  ('smem_max', uint16_t),
  ('global_size', (uint32_t * 3)),
  ('num_cbufs', uint32_t),
  ('cbufs', (struct_nak_qmd_cbuf * 8)),
]
try: (nak_qmd_size_B:=dll.nak_qmd_size_B).restype, nak_qmd_size_B.argtypes = uint32_t, [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

try: (nak_fill_qmd:=dll.nak_fill_qmd).restype, nak_fill_qmd.argtypes = None, [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(struct_nak_shader_info), ctypes.POINTER(struct_nak_qmd_info), ctypes.c_void_p, size_t]
except AttributeError: pass

class struct_nak_qmd_dispatch_size_layout(Struct): pass
struct_nak_qmd_dispatch_size_layout._fields_ = [
  ('x_start', uint16_t),
  ('x_end', uint16_t),
  ('y_start', uint16_t),
  ('y_end', uint16_t),
  ('z_start', uint16_t),
  ('z_end', uint16_t),
]
try: (nak_get_qmd_dispatch_size_layout:=dll.nak_get_qmd_dispatch_size_layout).restype, nak_get_qmd_dispatch_size_layout.argtypes = struct_nak_qmd_dispatch_size_layout, [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

class struct_nak_qmd_cbuf_desc_layout(Struct): pass
struct_nak_qmd_cbuf_desc_layout._fields_ = [
  ('addr_shift', uint16_t),
  ('addr_lo_start', uint16_t),
  ('addr_lo_end', uint16_t),
  ('addr_hi_start', uint16_t),
  ('addr_hi_end', uint16_t),
]
try: (nak_get_qmd_cbuf_desc_layout:=dll.nak_get_qmd_cbuf_desc_layout).restype, nak_get_qmd_cbuf_desc_layout.argtypes = struct_nak_qmd_cbuf_desc_layout, [ctypes.POINTER(struct_nv_device_info), uint8_t]
except AttributeError: pass

class struct_lp_context_ref(Struct): pass
class struct_LLVMOpaqueContext(Struct): pass
LLVMContextRef = ctypes.POINTER(struct_LLVMOpaqueContext)
struct_lp_context_ref._fields_ = [
  ('ref', LLVMContextRef),
  ('owned', ctypes.c_bool),
]
lp_context_ref = struct_lp_context_ref
class struct_lp_passmgr(Struct): pass
class struct_LLVMOpaqueModule(Struct): pass
LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
try: (lp_passmgr_create:=dll.lp_passmgr_create).restype, lp_passmgr_create.argtypes = ctypes.c_bool, [LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(struct_lp_passmgr))]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachine(Struct): pass
LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
try: (lp_passmgr_run:=dll.lp_passmgr_run).restype, lp_passmgr_run.argtypes = None, [ctypes.POINTER(struct_lp_passmgr), LLVMModuleRef, LLVMTargetMachineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_passmgr_dispose:=dll.lp_passmgr_dispose).restype, lp_passmgr_dispose.argtypes = None, [ctypes.POINTER(struct_lp_passmgr)]
except AttributeError: pass

class struct_lp_cached_code(Struct): pass
struct_lp_cached_code._fields_ = [
  ('data', ctypes.c_void_p),
  ('data_size', size_t),
  ('dont_cache', ctypes.c_bool),
  ('jit_obj_cache', ctypes.c_void_p),
]
class struct_lp_generated_code(Struct): pass
class struct_LLVMOpaqueTargetLibraryInfotData(Struct): pass
LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
try: (gallivm_create_target_library_info:=dll.gallivm_create_target_library_info).restype, gallivm_create_target_library_info.argtypes = LLVMTargetLibraryInfoRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (gallivm_dispose_target_library_info:=dll.gallivm_dispose_target_library_info).restype, gallivm_dispose_target_library_info.argtypes = None, [LLVMTargetLibraryInfoRef]
except AttributeError: pass

try: (lp_set_target_options:=dll.lp_set_target_options).restype, lp_set_target_options.argtypes = None, []
except AttributeError: pass

try: (lp_bld_init_native_targets:=dll.lp_bld_init_native_targets).restype, lp_bld_init_native_targets.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueExecutionEngine(Struct): pass
LLVMExecutionEngineRef = ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)
class struct_LLVMOpaqueMCJITMemoryManager(Struct): pass
LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
try: (lp_build_create_jit_compiler_for_module:=dll.lp_build_create_jit_compiler_for_module).restype, lp_build_create_jit_compiler_for_module.argtypes = ctypes.c_int32, [ctypes.POINTER(LLVMExecutionEngineRef), ctypes.POINTER(ctypes.POINTER(struct_lp_generated_code)), ctypes.POINTER(struct_lp_cached_code), LLVMModuleRef, LLVMMCJITMemoryManagerRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (lp_free_generated_code:=dll.lp_free_generated_code).restype, lp_free_generated_code.argtypes = None, [ctypes.POINTER(struct_lp_generated_code)]
except AttributeError: pass

try: (lp_get_default_memory_manager:=dll.lp_get_default_memory_manager).restype, lp_get_default_memory_manager.argtypes = LLVMMCJITMemoryManagerRef, []
except AttributeError: pass

try: (lp_free_memory_manager:=dll.lp_free_memory_manager).restype, lp_free_memory_manager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

class struct_LLVMOpaqueValue(Struct): pass
LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
try: (lp_get_called_value:=dll.lp_get_called_value).restype, lp_get_called_value.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (lp_is_function:=dll.lp_is_function).restype, lp_is_function.argtypes = ctypes.c_bool, [LLVMValueRef]
except AttributeError: pass

try: (lp_free_objcache:=dll.lp_free_objcache).restype, lp_free_objcache.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (lp_set_module_stack_alignment_override:=dll.lp_set_module_stack_alignment_override).restype, lp_set_module_stack_alignment_override.argtypes = None, [LLVMModuleRef, ctypes.c_uint32]
except AttributeError: pass

try: lp_native_vector_width = ctypes.c_uint32.in_dll(dll, 'lp_native_vector_width')
except (ValueError,AttributeError): pass
class struct_lp_type(Struct): pass
struct_lp_type._fields_ = [
  ('floating', ctypes.c_uint32,1),
  ('fixed', ctypes.c_uint32,1),
  ('sign', ctypes.c_uint32,1),
  ('norm', ctypes.c_uint32,1),
  ('signed_zero_preserve', ctypes.c_uint32,1),
  ('nan_preserve', ctypes.c_uint32,1),
  ('width', ctypes.c_uint32,14),
  ('length', ctypes.c_uint32,14),
]
class struct_lp_build_context(Struct): pass
class struct_gallivm_state(Struct): pass
class struct_LLVMOpaqueType(Struct): pass
LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
struct_lp_build_context._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('type', struct_lp_type),
  ('elem_type', LLVMTypeRef),
  ('vec_type', LLVMTypeRef),
  ('int_elem_type', LLVMTypeRef),
  ('int_vec_type', LLVMTypeRef),
  ('undef', LLVMValueRef),
  ('zero', LLVMValueRef),
  ('one', LLVMValueRef),
]
try: (lp_build_elem_type:=dll.lp_build_elem_type).restype, lp_build_elem_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_vec_type:=dll.lp_build_vec_type).restype, lp_build_vec_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_check_elem_type:=dll.lp_check_elem_type).restype, lp_check_elem_type.argtypes = ctypes.c_bool, [struct_lp_type, LLVMTypeRef]
except AttributeError: pass

try: (lp_check_vec_type:=dll.lp_check_vec_type).restype, lp_check_vec_type.argtypes = ctypes.c_bool, [struct_lp_type, LLVMTypeRef]
except AttributeError: pass

try: (lp_check_value:=dll.lp_check_value).restype, lp_check_value.argtypes = ctypes.c_bool, [struct_lp_type, LLVMValueRef]
except AttributeError: pass

try: (lp_build_int_elem_type:=dll.lp_build_int_elem_type).restype, lp_build_int_elem_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_int_vec_type:=dll.lp_build_int_vec_type).restype, lp_build_int_vec_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_elem_type:=dll.lp_elem_type).restype, lp_elem_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_uint_type:=dll.lp_uint_type).restype, lp_uint_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_int_type:=dll.lp_int_type).restype, lp_int_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_wider_type:=dll.lp_wider_type).restype, lp_wider_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_sizeof_llvm_type:=dll.lp_sizeof_llvm_type).restype, lp_sizeof_llvm_type.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

LLVMTypeKind = CEnum(ctypes.c_uint32)
LLVMVoidTypeKind = LLVMTypeKind.define('LLVMVoidTypeKind', 0)
LLVMHalfTypeKind = LLVMTypeKind.define('LLVMHalfTypeKind', 1)
LLVMFloatTypeKind = LLVMTypeKind.define('LLVMFloatTypeKind', 2)
LLVMDoubleTypeKind = LLVMTypeKind.define('LLVMDoubleTypeKind', 3)
LLVMX86_FP80TypeKind = LLVMTypeKind.define('LLVMX86_FP80TypeKind', 4)
LLVMFP128TypeKind = LLVMTypeKind.define('LLVMFP128TypeKind', 5)
LLVMPPC_FP128TypeKind = LLVMTypeKind.define('LLVMPPC_FP128TypeKind', 6)
LLVMLabelTypeKind = LLVMTypeKind.define('LLVMLabelTypeKind', 7)
LLVMIntegerTypeKind = LLVMTypeKind.define('LLVMIntegerTypeKind', 8)
LLVMFunctionTypeKind = LLVMTypeKind.define('LLVMFunctionTypeKind', 9)
LLVMStructTypeKind = LLVMTypeKind.define('LLVMStructTypeKind', 10)
LLVMArrayTypeKind = LLVMTypeKind.define('LLVMArrayTypeKind', 11)
LLVMPointerTypeKind = LLVMTypeKind.define('LLVMPointerTypeKind', 12)
LLVMVectorTypeKind = LLVMTypeKind.define('LLVMVectorTypeKind', 13)
LLVMMetadataTypeKind = LLVMTypeKind.define('LLVMMetadataTypeKind', 14)
LLVMTokenTypeKind = LLVMTypeKind.define('LLVMTokenTypeKind', 16)
LLVMScalableVectorTypeKind = LLVMTypeKind.define('LLVMScalableVectorTypeKind', 17)
LLVMBFloatTypeKind = LLVMTypeKind.define('LLVMBFloatTypeKind', 18)
LLVMX86_AMXTypeKind = LLVMTypeKind.define('LLVMX86_AMXTypeKind', 19)
LLVMTargetExtTypeKind = LLVMTypeKind.define('LLVMTargetExtTypeKind', 20)

try: (lp_typekind_name:=dll.lp_typekind_name).restype, lp_typekind_name.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeKind]
except AttributeError: pass

try: (lp_dump_llvmtype:=dll.lp_dump_llvmtype).restype, lp_dump_llvmtype.argtypes = None, [LLVMTypeRef]
except AttributeError: pass

try: (lp_build_context_init:=dll.lp_build_context_init).restype, lp_build_context_init.argtypes = None, [ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_count_ir_module:=dll.lp_build_count_ir_module).restype, lp_build_count_ir_module.argtypes = ctypes.c_uint32, [LLVMModuleRef]
except AttributeError: pass

class struct_lp_jit_texture(Struct): pass
class struct_LLVMOpaqueTargetData(Struct): pass
LLVMTargetDataRef = ctypes.POINTER(struct_LLVMOpaqueTargetData)
class struct_LLVMOpaqueBuilder(Struct): pass
LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
class struct_LLVMOpaqueDIBuilder(Struct): pass
LLVMDIBuilderRef = ctypes.POINTER(struct_LLVMOpaqueDIBuilder)
class struct_LLVMOpaqueMetadata(Struct): pass
LLVMMetadataRef = ctypes.POINTER(struct_LLVMOpaqueMetadata)
struct_gallivm_state._fields_ = [
  ('module_name', ctypes.POINTER(ctypes.c_char)),
  ('file_name', ctypes.POINTER(ctypes.c_char)),
  ('module', LLVMModuleRef),
  ('target', LLVMTargetDataRef),
  ('engine', LLVMExecutionEngineRef),
  ('passmgr', ctypes.POINTER(struct_lp_passmgr)),
  ('memorymgr', LLVMMCJITMemoryManagerRef),
  ('code', ctypes.POINTER(struct_lp_generated_code)),
  ('context', LLVMContextRef),
  ('builder', LLVMBuilderRef),
  ('di_builder', LLVMDIBuilderRef),
  ('cache', ctypes.POINTER(struct_lp_cached_code)),
  ('compiled', ctypes.c_uint32),
  ('coro_malloc_hook', LLVMValueRef),
  ('coro_free_hook', LLVMValueRef),
  ('debug_printf_hook', LLVMValueRef),
  ('coro_malloc_hook_type', LLVMTypeRef),
  ('coro_free_hook_type', LLVMTypeRef),
  ('di_function', LLVMMetadataRef),
  ('file', LLVMMetadataRef),
  ('get_time_hook', LLVMValueRef),
  ('texture_descriptor', LLVMValueRef),
  ('texture_dynamic_state', ctypes.POINTER(struct_lp_jit_texture)),
  ('sampler_descriptor', LLVMValueRef),
]
try: (lp_build_init_native_width:=dll.lp_build_init_native_width).restype, lp_build_init_native_width.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (lp_build_init:=dll.lp_build_init).restype, lp_build_init.argtypes = ctypes.c_bool, []
except AttributeError: pass

try: (gallivm_create:=dll.gallivm_create).restype, gallivm_create.argtypes = ctypes.POINTER(struct_gallivm_state), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(lp_context_ref), ctypes.POINTER(struct_lp_cached_code)]
except AttributeError: pass

try: (gallivm_destroy:=dll.gallivm_destroy).restype, gallivm_destroy.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (gallivm_free_ir:=dll.gallivm_free_ir).restype, gallivm_free_ir.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (gallivm_verify_function:=dll.gallivm_verify_function).restype, gallivm_verify_function.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (gallivm_add_global_mapping:=dll.gallivm_add_global_mapping).restype, gallivm_add_global_mapping.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

try: (gallivm_compile_module:=dll.gallivm_compile_module).restype, gallivm_compile_module.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

func_pointer = ctypes.CFUNCTYPE(None, )
try: (gallivm_jit_function:=dll.gallivm_jit_function).restype, gallivm_jit_function.argtypes = func_pointer, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (gallivm_stub_func:=dll.gallivm_stub_func).restype, gallivm_stub_func.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (gallivm_get_perf_flags:=dll.gallivm_get_perf_flags).restype, gallivm_get_perf_flags.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (lp_init_clock_hook:=dll.lp_init_clock_hook).restype, lp_init_clock_hook.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_init_env_options:=dll.lp_init_env_options).restype, lp_init_env_options.argtypes = None, []
except AttributeError: pass

class struct_lp_build_tgsi_params(Struct): pass
class struct_lp_build_mask_context(Struct): pass
class struct_lp_build_skip_context(Struct): pass
class struct_LLVMOpaqueBasicBlock(Struct): pass
LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
struct_lp_build_skip_context._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('block', LLVMBasicBlockRef),
]
struct_lp_build_mask_context._fields_ = [
  ('skip', struct_lp_build_skip_context),
  ('reg_type', LLVMTypeRef),
  ('var_type', LLVMTypeRef),
  ('var', LLVMValueRef),
]
class struct_lp_bld_tgsi_system_values(Struct): pass
struct_lp_bld_tgsi_system_values._fields_ = [
  ('instance_id', LLVMValueRef),
  ('base_instance', LLVMValueRef),
  ('vertex_id', LLVMValueRef),
  ('vertex_id_nobase', LLVMValueRef),
  ('prim_id', LLVMValueRef),
  ('basevertex', LLVMValueRef),
  ('firstvertex', LLVMValueRef),
  ('invocation_id', LLVMValueRef),
  ('draw_id', LLVMValueRef),
  ('thread_id', (LLVMValueRef * 3)),
  ('block_id', (LLVMValueRef * 3)),
  ('grid_size', (LLVMValueRef * 3)),
  ('front_facing', LLVMValueRef),
  ('work_dim', LLVMValueRef),
  ('block_size', (LLVMValueRef * 3)),
  ('tess_coord', LLVMValueRef),
  ('tess_outer', LLVMValueRef),
  ('tess_inner', LLVMValueRef),
  ('vertices_in', LLVMValueRef),
  ('sample_id', LLVMValueRef),
  ('sample_pos_type', LLVMTypeRef),
  ('sample_pos', LLVMValueRef),
  ('sample_mask_in', LLVMValueRef),
  ('view_index', LLVMValueRef),
  ('subgroup_id', LLVMValueRef),
  ('num_subgroups', LLVMValueRef),
]
class struct_lp_build_sampler_soa(Struct): pass
class struct_lp_sampler_params(Struct): pass
class struct_lp_derivatives(Struct): pass
struct_lp_derivatives._fields_ = [
  ('ddx', (LLVMValueRef * 3)),
  ('ddy', (LLVMValueRef * 3)),
]
struct_lp_sampler_params._fields_ = [
  ('type', struct_lp_type),
  ('texture_index', ctypes.c_uint32),
  ('sampler_index', ctypes.c_uint32),
  ('texture_index_offset', LLVMValueRef),
  ('sample_key', ctypes.c_uint32),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('coords', ctypes.POINTER(LLVMValueRef)),
  ('offsets', ctypes.POINTER(LLVMValueRef)),
  ('ms_index', LLVMValueRef),
  ('lod', LLVMValueRef),
  ('min_lod', LLVMValueRef),
  ('derivs', ctypes.POINTER(struct_lp_derivatives)),
  ('texel', ctypes.POINTER(LLVMValueRef)),
  ('texture_resource', LLVMValueRef),
  ('sampler_resource', LLVMValueRef),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
]
class struct_lp_sampler_size_query_params(Struct): pass
enum_lp_sampler_lod_property = CEnum(ctypes.c_uint32)
LP_SAMPLER_LOD_SCALAR = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_SCALAR', 0)
LP_SAMPLER_LOD_PER_ELEMENT = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_PER_ELEMENT', 1)
LP_SAMPLER_LOD_PER_QUAD = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_PER_QUAD', 2)

struct_lp_sampler_size_query_params._fields_ = [
  ('int_type', struct_lp_type),
  ('texture_unit', ctypes.c_uint32),
  ('texture_unit_offset', LLVMValueRef),
  ('target', ctypes.c_uint32),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('is_sviewinfo', ctypes.c_bool),
  ('samples_only', ctypes.c_bool),
  ('ms', ctypes.c_bool),
  ('lod_property', enum_lp_sampler_lod_property),
  ('explicit_lod', LLVMValueRef),
  ('sizes_out', ctypes.POINTER(LLVMValueRef)),
  ('resource', LLVMValueRef),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
  ('format', enum_pipe_format),
]
struct_lp_build_sampler_soa._fields_ = [
  ('emit_tex_sample', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_params))),
  ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]
class struct_tgsi_shader_info(Struct): pass
struct_tgsi_shader_info._fields_ = [
  ('num_inputs', uint8_t),
  ('num_outputs', uint8_t),
  ('input_semantic_name', (uint8_t * 80)),
  ('input_semantic_index', (uint8_t * 80)),
  ('input_interpolate', (uint8_t * 80)),
  ('input_interpolate_loc', (uint8_t * 80)),
  ('input_usage_mask', (uint8_t * 80)),
  ('output_semantic_name', (uint8_t * 80)),
  ('output_semantic_index', (uint8_t * 80)),
  ('output_usagemask', (uint8_t * 80)),
  ('output_streams', (uint8_t * 80)),
  ('num_system_values', uint8_t),
  ('system_value_semantic_name', (uint8_t * 80)),
  ('processor', uint8_t),
  ('file_mask', (uint32_t * 15)),
  ('file_count', (ctypes.c_uint32 * 15)),
  ('file_max', (ctypes.c_int32 * 15)),
  ('const_file_max', (ctypes.c_int32 * 32)),
  ('const_buffers_declared', ctypes.c_uint32),
  ('samplers_declared', ctypes.c_uint32),
  ('sampler_targets', (uint8_t * 128)),
  ('sampler_type', (uint8_t * 128)),
  ('num_stream_output_components', (uint8_t * 4)),
  ('input_array_first', (uint8_t * 80)),
  ('output_array_first', (uint8_t * 80)),
  ('immediate_count', ctypes.c_uint32),
  ('num_instructions', ctypes.c_uint32),
  ('opcode_count', (ctypes.c_uint32 * 252)),
  ('reads_pervertex_outputs', ctypes.c_bool),
  ('reads_perpatch_outputs', ctypes.c_bool),
  ('reads_tessfactor_outputs', ctypes.c_bool),
  ('reads_z', ctypes.c_bool),
  ('writes_z', ctypes.c_bool),
  ('writes_stencil', ctypes.c_bool),
  ('writes_samplemask', ctypes.c_bool),
  ('writes_edgeflag', ctypes.c_bool),
  ('uses_kill', ctypes.c_bool),
  ('uses_instanceid', ctypes.c_bool),
  ('uses_vertexid', ctypes.c_bool),
  ('uses_vertexid_nobase', ctypes.c_bool),
  ('uses_basevertex', ctypes.c_bool),
  ('uses_primid', ctypes.c_bool),
  ('uses_frontface', ctypes.c_bool),
  ('uses_invocationid', ctypes.c_bool),
  ('uses_grid_size', ctypes.c_bool),
  ('writes_position', ctypes.c_bool),
  ('writes_psize', ctypes.c_bool),
  ('writes_clipvertex', ctypes.c_bool),
  ('writes_viewport_index', ctypes.c_bool),
  ('writes_layer', ctypes.c_bool),
  ('writes_memory', ctypes.c_bool),
  ('uses_fbfetch', ctypes.c_bool),
  ('num_written_culldistance', ctypes.c_uint32),
  ('num_written_clipdistance', ctypes.c_uint32),
  ('images_declared', ctypes.c_uint32),
  ('msaa_images_declared', ctypes.c_uint32),
  ('images_buffers', ctypes.c_uint32),
  ('shader_buffers_declared', ctypes.c_uint32),
  ('shader_buffers_load', ctypes.c_uint32),
  ('shader_buffers_store', ctypes.c_uint32),
  ('shader_buffers_atomic', ctypes.c_uint32),
  ('hw_atomic_declared', ctypes.c_uint32),
  ('indirect_files', ctypes.c_uint32),
  ('dim_indirect_files', ctypes.c_uint32),
  ('properties', (ctypes.c_uint32 * 29)),
]
class struct_lp_build_gs_iface(Struct): pass
struct_lp_build_gs_iface._fields_ = [
  ('fetch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef)),
  ('emit_vertex', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.POINTER((LLVMValueRef * 4)), LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('end_primitive', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32)),
  ('gs_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), LLVMValueRef, LLVMValueRef, ctypes.c_uint32)),
]
class struct_lp_build_tcs_iface(Struct): pass
struct_lp_build_tcs_iface._fields_ = [
  ('emit_prologue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_barrier', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('emit_fetch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef)),
  ('emit_fetch_output', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, uint32_t)),
]
class struct_lp_build_tes_iface(Struct): pass
struct_lp_build_tes_iface._fields_ = [
  ('fetch_vertex_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef)),
  ('fetch_patch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, LLVMValueRef)),
]
class struct_lp_build_mesh_iface(Struct): pass
struct_lp_build_mesh_iface._fields_ = [
  ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('emit_vertex_and_primitive_count', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), LLVMValueRef, LLVMValueRef)),
]
class struct_lp_build_image_soa(Struct): pass
class struct_lp_img_params(Struct): pass
LLVMAtomicRMWBinOp = CEnum(ctypes.c_uint32)
LLVMAtomicRMWBinOpXchg = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXchg', 0)
LLVMAtomicRMWBinOpAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAdd', 1)
LLVMAtomicRMWBinOpSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpSub', 2)
LLVMAtomicRMWBinOpAnd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAnd', 3)
LLVMAtomicRMWBinOpNand = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpNand', 4)
LLVMAtomicRMWBinOpOr = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpOr', 5)
LLVMAtomicRMWBinOpXor = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXor', 6)
LLVMAtomicRMWBinOpMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMax', 7)
LLVMAtomicRMWBinOpMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMin', 8)
LLVMAtomicRMWBinOpUMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMax', 9)
LLVMAtomicRMWBinOpUMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMin', 10)
LLVMAtomicRMWBinOpFAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFAdd', 11)
LLVMAtomicRMWBinOpFSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFSub', 12)
LLVMAtomicRMWBinOpFMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMax', 13)
LLVMAtomicRMWBinOpFMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMin', 14)
LLVMAtomicRMWBinOpUIncWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUIncWrap', 15)
LLVMAtomicRMWBinOpUDecWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUDecWrap', 16)
LLVMAtomicRMWBinOpUSubCond = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubCond', 17)
LLVMAtomicRMWBinOpUSubSat = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubSat', 18)

struct_lp_img_params._fields_ = [
  ('type', struct_lp_type),
  ('image_index', ctypes.c_uint32),
  ('image_index_offset', LLVMValueRef),
  ('img_op', ctypes.c_uint32),
  ('target', ctypes.c_uint32),
  ('packed_op', ctypes.c_uint32),
  ('op', LLVMAtomicRMWBinOp),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('coords', ctypes.POINTER(LLVMValueRef)),
  ('ms_index', LLVMValueRef),
  ('indata', (LLVMValueRef * 4)),
  ('indata2', (LLVMValueRef * 4)),
  ('outdata', ctypes.POINTER(LLVMValueRef)),
  ('resource', LLVMValueRef),
  ('format', enum_pipe_format),
]
struct_lp_build_image_soa._fields_ = [
  ('emit_op', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params))),
  ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]
class struct_lp_build_coro_suspend_info(Struct): pass
class struct_lp_build_fs_iface(Struct): pass
struct_lp_build_fs_iface._fields_ = [
  ('interp_fn', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, LLVMValueRef, (LLVMValueRef * 2))),
  ('fb_fetch', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_int32, (LLVMValueRef * 4))),
]
struct_lp_build_tgsi_params._fields_ = [
  ('type', struct_lp_type),
  ('mask', ctypes.POINTER(struct_lp_build_mask_context)),
  ('consts_ptr', LLVMValueRef),
  ('const_sizes_ptr', LLVMValueRef),
  ('system_values', ctypes.POINTER(struct_lp_bld_tgsi_system_values)),
  ('inputs', ctypes.POINTER((LLVMValueRef * 4))),
  ('num_inputs', ctypes.c_int32),
  ('context_type', LLVMTypeRef),
  ('context_ptr', LLVMValueRef),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('sampler', ctypes.POINTER(struct_lp_build_sampler_soa)),
  ('info', ctypes.POINTER(struct_tgsi_shader_info)),
  ('gs_iface', ctypes.POINTER(struct_lp_build_gs_iface)),
  ('tcs_iface', ctypes.POINTER(struct_lp_build_tcs_iface)),
  ('tes_iface', ctypes.POINTER(struct_lp_build_tes_iface)),
  ('mesh_iface', ctypes.POINTER(struct_lp_build_mesh_iface)),
  ('ssbo_ptr', LLVMValueRef),
  ('ssbo_sizes_ptr', LLVMValueRef),
  ('image', ctypes.POINTER(struct_lp_build_image_soa)),
  ('shared_ptr', LLVMValueRef),
  ('payload_ptr', LLVMValueRef),
  ('coro', ctypes.POINTER(struct_lp_build_coro_suspend_info)),
  ('fs_iface', ctypes.POINTER(struct_lp_build_fs_iface)),
  ('gs_vertex_streams', ctypes.c_uint32),
  ('current_func', LLVMValueRef),
  ('fns', ctypes.POINTER(struct_hash_table)),
  ('scratch_ptr', LLVMValueRef),
  ('call_context_ptr', LLVMValueRef),
]
try: (lp_build_nir_soa:=dll.lp_build_nir_soa).restype, lp_build_nir_soa.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER((LLVMValueRef * 4))]
except AttributeError: pass

try: (lp_build_nir_soa_func:=dll.lp_build_nir_soa_func).restype, lp_build_nir_soa_func.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER((LLVMValueRef * 4))]
except AttributeError: pass

class struct_lp_build_sampler_aos(Struct): pass
enum_tgsi_texture_type = CEnum(ctypes.c_uint32)
TGSI_TEXTURE_BUFFER = enum_tgsi_texture_type.define('TGSI_TEXTURE_BUFFER', 0)
TGSI_TEXTURE_1D = enum_tgsi_texture_type.define('TGSI_TEXTURE_1D', 1)
TGSI_TEXTURE_2D = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D', 2)
TGSI_TEXTURE_3D = enum_tgsi_texture_type.define('TGSI_TEXTURE_3D', 3)
TGSI_TEXTURE_CUBE = enum_tgsi_texture_type.define('TGSI_TEXTURE_CUBE', 4)
TGSI_TEXTURE_RECT = enum_tgsi_texture_type.define('TGSI_TEXTURE_RECT', 5)
TGSI_TEXTURE_SHADOW1D = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW1D', 6)
TGSI_TEXTURE_SHADOW2D = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW2D', 7)
TGSI_TEXTURE_SHADOWRECT = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWRECT', 8)
TGSI_TEXTURE_1D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_1D_ARRAY', 9)
TGSI_TEXTURE_2D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_ARRAY', 10)
TGSI_TEXTURE_SHADOW1D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW1D_ARRAY', 11)
TGSI_TEXTURE_SHADOW2D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW2D_ARRAY', 12)
TGSI_TEXTURE_SHADOWCUBE = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWCUBE', 13)
TGSI_TEXTURE_2D_MSAA = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_MSAA', 14)
TGSI_TEXTURE_2D_ARRAY_MSAA = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_ARRAY_MSAA', 15)
TGSI_TEXTURE_CUBE_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_CUBE_ARRAY', 16)
TGSI_TEXTURE_SHADOWCUBE_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWCUBE_ARRAY', 17)
TGSI_TEXTURE_UNKNOWN = enum_tgsi_texture_type.define('TGSI_TEXTURE_UNKNOWN', 18)
TGSI_TEXTURE_COUNT = enum_tgsi_texture_type.define('TGSI_TEXTURE_COUNT', 19)

enum_lp_build_tex_modifier = CEnum(ctypes.c_uint32)
LP_BLD_TEX_MODIFIER_NONE = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_NONE', 0)
LP_BLD_TEX_MODIFIER_PROJECTED = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_PROJECTED', 1)
LP_BLD_TEX_MODIFIER_LOD_BIAS = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_LOD_BIAS', 2)
LP_BLD_TEX_MODIFIER_EXPLICIT_LOD = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_EXPLICIT_LOD', 3)
LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV', 4)
LP_BLD_TEX_MODIFIER_LOD_ZERO = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_LOD_ZERO', 5)

struct_lp_build_sampler_aos._fields_ = [
  ('emit_fetch_texel', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_sampler_aos), ctypes.POINTER(struct_lp_build_context), enum_tgsi_texture_type, ctypes.c_uint32, LLVMValueRef, struct_lp_derivatives, enum_lp_build_tex_modifier)),
]
try: (lp_build_nir_aos:=dll.lp_build_nir_aos).restype, lp_build_nir_aos.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), struct_lp_type, (ctypes.c_ubyte * 4), LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.POINTER(LLVMValueRef), ctypes.POINTER(struct_lp_build_sampler_aos)]
except AttributeError: pass

class struct_lp_build_fn(Struct): pass
struct_lp_build_fn._fields_ = [
  ('fn_type', LLVMTypeRef),
  ('fn', LLVMValueRef),
]
try: (lp_build_nir_soa_prepasses:=dll.lp_build_nir_soa_prepasses).restype, lp_build_nir_soa_prepasses.argtypes = None, [ctypes.POINTER(struct_nir_shader)]
except AttributeError: pass

try: (lp_build_opt_nir:=dll.lp_build_opt_nir).restype, lp_build_opt_nir.argtypes = None, [ctypes.POINTER(struct_nir_shader)]
except AttributeError: pass

try: (lp_translate_atomic_op:=dll.lp_translate_atomic_op).restype, lp_translate_atomic_op.argtypes = LLVMAtomicRMWBinOp, [nir_atomic_op]
except AttributeError: pass

try: (lp_build_nir_sample_key:=dll.lp_build_nir_sample_key).restype, lp_build_nir_sample_key.argtypes = uint32_t, [gl_shader_stage, ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (lp_img_op_from_intrinsic:=dll.lp_img_op_from_intrinsic).restype, lp_img_op_from_intrinsic.argtypes = None, [ctypes.POINTER(struct_lp_img_params), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (lp_packed_img_op_from_intrinsic:=dll.lp_packed_img_op_from_intrinsic).restype, lp_packed_img_op_from_intrinsic.argtypes = uint32_t, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

enum_lp_nir_call_context_args = CEnum(ctypes.c_uint32)
LP_NIR_CALL_CONTEXT_CONTEXT = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_CONTEXT', 0)
LP_NIR_CALL_CONTEXT_RESOURCES = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_RESOURCES', 1)
LP_NIR_CALL_CONTEXT_SHARED = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_SHARED', 2)
LP_NIR_CALL_CONTEXT_SCRATCH = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_SCRATCH', 3)
LP_NIR_CALL_CONTEXT_WORK_DIM = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_WORK_DIM', 4)
LP_NIR_CALL_CONTEXT_THREAD_ID_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_0', 5)
LP_NIR_CALL_CONTEXT_THREAD_ID_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_1', 6)
LP_NIR_CALL_CONTEXT_THREAD_ID_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_2', 7)
LP_NIR_CALL_CONTEXT_BLOCK_ID_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_0', 8)
LP_NIR_CALL_CONTEXT_BLOCK_ID_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_1', 9)
LP_NIR_CALL_CONTEXT_BLOCK_ID_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_2', 10)
LP_NIR_CALL_CONTEXT_GRID_SIZE_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_0', 11)
LP_NIR_CALL_CONTEXT_GRID_SIZE_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_1', 12)
LP_NIR_CALL_CONTEXT_GRID_SIZE_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_2', 13)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0', 14)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1', 15)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2', 16)
LP_NIR_CALL_CONTEXT_MAX_ARGS = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_MAX_ARGS', 17)

try: (lp_build_cs_func_call_context:=dll.lp_build_cs_func_call_context).restype, lp_build_cs_func_call_context.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32, LLVMTypeRef, LLVMTypeRef]
except AttributeError: pass

try: (lp_build_struct_get_ptr2:=dll.lp_build_struct_get_ptr2).restype, lp_build_struct_get_ptr2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_struct_get2:=dll.lp_build_struct_get2).restype, lp_build_struct_get2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_array_get_ptr2:=dll.lp_build_array_get_ptr2).restype, lp_build_array_get_ptr2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_array_get2:=dll.lp_build_array_get2).restype, lp_build_array_get2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_get2:=dll.lp_build_pointer_get2).restype, lp_build_pointer_get2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_get_unaligned2:=dll.lp_build_pointer_get_unaligned2).restype, lp_build_pointer_get_unaligned2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_build_pointer_set:=dll.lp_build_pointer_set).restype, lp_build_pointer_set.argtypes = None, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_set_unaligned:=dll.lp_build_pointer_set_unaligned).restype, lp_build_pointer_set_unaligned.argtypes = None, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

class struct_lp_sampler_dynamic_state(Struct): pass
struct_lp_sampler_dynamic_state._fields_ = [
  ('width', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('height', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('depth', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('first_level', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('last_level', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('row_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('img_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('base_ptr', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('mip_offsets', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('num_samples', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('sample_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('min_lod', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('max_lod', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('lod_bias', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('border_color', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('cache_ptr', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('residency', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('base_offset', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
]
class struct_lp_jit_buffer(Struct): pass
class struct_lp_jit_buffer_0(ctypes.Union): pass
struct_lp_jit_buffer_0._fields_ = [
  ('u', ctypes.POINTER(uint32_t)),
  ('f', ctypes.POINTER(ctypes.c_float)),
]
struct_lp_jit_buffer._anonymous_ = ['_0']
struct_lp_jit_buffer._fields_ = [
  ('_0', struct_lp_jit_buffer_0),
  ('num_elements', uint32_t),
]
_anonenum0 = CEnum(ctypes.c_uint32)
LP_JIT_BUFFER_BASE = _anonenum0.define('LP_JIT_BUFFER_BASE', 0)
LP_JIT_BUFFER_NUM_ELEMENTS = _anonenum0.define('LP_JIT_BUFFER_NUM_ELEMENTS', 1)
LP_JIT_BUFFER_NUM_FIELDS = _anonenum0.define('LP_JIT_BUFFER_NUM_FIELDS', 2)

try: (lp_llvm_descriptor_base:=dll.lp_llvm_descriptor_base).restype, lp_llvm_descriptor_base.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_llvm_buffer_base:=dll.lp_llvm_buffer_base).restype, lp_llvm_buffer_base.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_llvm_buffer_num_elements:=dll.lp_llvm_buffer_num_elements).restype, lp_llvm_buffer_num_elements.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

class struct_lp_jit_texture_0(ctypes.Union): pass
class struct_lp_jit_texture_0_0(Struct): pass
struct_lp_jit_texture_0_0._fields_ = [
  ('row_stride', (uint32_t * 16)),
  ('img_stride', (uint32_t * 16)),
]
struct_lp_jit_texture_0._anonymous_ = ['_0']
struct_lp_jit_texture_0._fields_ = [
  ('_0', struct_lp_jit_texture_0_0),
  ('residency', ctypes.c_void_p),
]
struct_lp_jit_texture._anonymous_ = ['_0']
struct_lp_jit_texture._fields_ = [
  ('base', ctypes.c_void_p),
  ('width', uint32_t),
  ('height', uint16_t),
  ('depth', uint16_t),
  ('_0', struct_lp_jit_texture_0),
  ('first_level', uint8_t),
  ('last_level', uint8_t),
  ('mip_offsets', (uint32_t * 16)),
  ('sampler_index', uint32_t),
]
_anonenum1 = CEnum(ctypes.c_uint32)
LP_JIT_TEXTURE_BASE = _anonenum1.define('LP_JIT_TEXTURE_BASE', 0)
LP_JIT_TEXTURE_WIDTH = _anonenum1.define('LP_JIT_TEXTURE_WIDTH', 1)
LP_JIT_TEXTURE_HEIGHT = _anonenum1.define('LP_JIT_TEXTURE_HEIGHT', 2)
LP_JIT_TEXTURE_DEPTH = _anonenum1.define('LP_JIT_TEXTURE_DEPTH', 3)
LP_JIT_TEXTURE_ROW_STRIDE = _anonenum1.define('LP_JIT_TEXTURE_ROW_STRIDE', 4)
LP_JIT_TEXTURE_IMG_STRIDE = _anonenum1.define('LP_JIT_TEXTURE_IMG_STRIDE', 5)
LP_JIT_TEXTURE_FIRST_LEVEL = _anonenum1.define('LP_JIT_TEXTURE_FIRST_LEVEL', 6)
LP_JIT_TEXTURE_LAST_LEVEL = _anonenum1.define('LP_JIT_TEXTURE_LAST_LEVEL', 7)
LP_JIT_TEXTURE_MIP_OFFSETS = _anonenum1.define('LP_JIT_TEXTURE_MIP_OFFSETS', 8)
LP_JIT_SAMPLER_INDEX_DUMMY = _anonenum1.define('LP_JIT_SAMPLER_INDEX_DUMMY', 9)
LP_JIT_TEXTURE_NUM_FIELDS = _anonenum1.define('LP_JIT_TEXTURE_NUM_FIELDS', 10)

class struct_lp_jit_sampler(Struct): pass
struct_lp_jit_sampler._fields_ = [
  ('min_lod', ctypes.c_float),
  ('max_lod', ctypes.c_float),
  ('lod_bias', ctypes.c_float),
  ('border_color', (ctypes.c_float * 4)),
]
_anonenum2 = CEnum(ctypes.c_uint32)
LP_JIT_SAMPLER_MIN_LOD = _anonenum2.define('LP_JIT_SAMPLER_MIN_LOD', 0)
LP_JIT_SAMPLER_MAX_LOD = _anonenum2.define('LP_JIT_SAMPLER_MAX_LOD', 1)
LP_JIT_SAMPLER_LOD_BIAS = _anonenum2.define('LP_JIT_SAMPLER_LOD_BIAS', 2)
LP_JIT_SAMPLER_BORDER_COLOR = _anonenum2.define('LP_JIT_SAMPLER_BORDER_COLOR', 3)
LP_JIT_SAMPLER_NUM_FIELDS = _anonenum2.define('LP_JIT_SAMPLER_NUM_FIELDS', 4)

class struct_lp_jit_image(Struct): pass
struct_lp_jit_image._fields_ = [
  ('base', ctypes.c_void_p),
  ('width', uint32_t),
  ('height', uint16_t),
  ('depth', uint16_t),
  ('num_samples', uint8_t),
  ('sample_stride', uint32_t),
  ('row_stride', uint32_t),
  ('img_stride', uint32_t),
  ('residency', ctypes.c_void_p),
  ('base_offset', uint32_t),
]
_anonenum3 = CEnum(ctypes.c_uint32)
LP_JIT_IMAGE_BASE = _anonenum3.define('LP_JIT_IMAGE_BASE', 0)
LP_JIT_IMAGE_WIDTH = _anonenum3.define('LP_JIT_IMAGE_WIDTH', 1)
LP_JIT_IMAGE_HEIGHT = _anonenum3.define('LP_JIT_IMAGE_HEIGHT', 2)
LP_JIT_IMAGE_DEPTH = _anonenum3.define('LP_JIT_IMAGE_DEPTH', 3)
LP_JIT_IMAGE_NUM_SAMPLES = _anonenum3.define('LP_JIT_IMAGE_NUM_SAMPLES', 4)
LP_JIT_IMAGE_SAMPLE_STRIDE = _anonenum3.define('LP_JIT_IMAGE_SAMPLE_STRIDE', 5)
LP_JIT_IMAGE_ROW_STRIDE = _anonenum3.define('LP_JIT_IMAGE_ROW_STRIDE', 6)
LP_JIT_IMAGE_IMG_STRIDE = _anonenum3.define('LP_JIT_IMAGE_IMG_STRIDE', 7)
LP_JIT_IMAGE_RESIDENCY = _anonenum3.define('LP_JIT_IMAGE_RESIDENCY', 8)
LP_JIT_IMAGE_BASE_OFFSET = _anonenum3.define('LP_JIT_IMAGE_BASE_OFFSET', 9)
LP_JIT_IMAGE_NUM_FIELDS = _anonenum3.define('LP_JIT_IMAGE_NUM_FIELDS', 10)

class struct_lp_jit_resources(Struct): pass
struct_lp_jit_resources._fields_ = [
  ('constants', (struct_lp_jit_buffer * 16)),
  ('ssbos', (struct_lp_jit_buffer * 32)),
  ('textures', (struct_lp_jit_texture * 128)),
  ('samplers', (struct_lp_jit_sampler * 32)),
  ('images', (struct_lp_jit_image * 64)),
]
_anonenum4 = CEnum(ctypes.c_uint32)
LP_JIT_RES_CONSTANTS = _anonenum4.define('LP_JIT_RES_CONSTANTS', 0)
LP_JIT_RES_SSBOS = _anonenum4.define('LP_JIT_RES_SSBOS', 1)
LP_JIT_RES_TEXTURES = _anonenum4.define('LP_JIT_RES_TEXTURES', 2)
LP_JIT_RES_SAMPLERS = _anonenum4.define('LP_JIT_RES_SAMPLERS', 3)
LP_JIT_RES_IMAGES = _anonenum4.define('LP_JIT_RES_IMAGES', 4)
LP_JIT_RES_COUNT = _anonenum4.define('LP_JIT_RES_COUNT', 5)

try: (lp_build_jit_resources_type:=dll.lp_build_jit_resources_type).restype, lp_build_jit_resources_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

_anonenum5 = CEnum(ctypes.c_uint32)
LP_JIT_VERTEX_HEADER_VERTEX_ID = _anonenum5.define('LP_JIT_VERTEX_HEADER_VERTEX_ID', 0)
LP_JIT_VERTEX_HEADER_CLIP_POS = _anonenum5.define('LP_JIT_VERTEX_HEADER_CLIP_POS', 1)
LP_JIT_VERTEX_HEADER_DATA = _anonenum5.define('LP_JIT_VERTEX_HEADER_DATA', 2)

try: (lp_build_create_jit_vertex_header_type:=dll.lp_build_create_jit_vertex_header_type).restype, lp_build_create_jit_vertex_header_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32]
except AttributeError: pass

try: (lp_build_jit_fill_sampler_dynamic_state:=dll.lp_build_jit_fill_sampler_dynamic_state).restype, lp_build_jit_fill_sampler_dynamic_state.argtypes = None, [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError: pass

try: (lp_build_jit_fill_image_dynamic_state:=dll.lp_build_jit_fill_image_dynamic_state).restype, lp_build_jit_fill_image_dynamic_state.argtypes = None, [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError: pass

try: (lp_build_sample_function_type:=dll.lp_build_sample_function_type).restype, lp_build_sample_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), uint32_t]
except AttributeError: pass

try: (lp_build_size_function_type:=dll.lp_build_size_function_type).restype, lp_build_size_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params)]
except AttributeError: pass

try: (lp_build_image_function_type:=dll.lp_build_image_function_type).restype, lp_build_image_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

class struct_lp_texture_handle_state(Struct): pass
class struct_lp_static_texture_state(Struct): pass
enum_pipe_texture_target = CEnum(ctypes.c_uint32)
PIPE_BUFFER = enum_pipe_texture_target.define('PIPE_BUFFER', 0)
PIPE_TEXTURE_1D = enum_pipe_texture_target.define('PIPE_TEXTURE_1D', 1)
PIPE_TEXTURE_2D = enum_pipe_texture_target.define('PIPE_TEXTURE_2D', 2)
PIPE_TEXTURE_3D = enum_pipe_texture_target.define('PIPE_TEXTURE_3D', 3)
PIPE_TEXTURE_CUBE = enum_pipe_texture_target.define('PIPE_TEXTURE_CUBE', 4)
PIPE_TEXTURE_RECT = enum_pipe_texture_target.define('PIPE_TEXTURE_RECT', 5)
PIPE_TEXTURE_1D_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_1D_ARRAY', 6)
PIPE_TEXTURE_2D_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_2D_ARRAY', 7)
PIPE_TEXTURE_CUBE_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_CUBE_ARRAY', 8)
PIPE_MAX_TEXTURE_TYPES = enum_pipe_texture_target.define('PIPE_MAX_TEXTURE_TYPES', 9)

struct_lp_static_texture_state._fields_ = [
  ('format', enum_pipe_format),
  ('res_format', enum_pipe_format),
  ('swizzle_r', ctypes.c_uint32,3),
  ('swizzle_g', ctypes.c_uint32,3),
  ('swizzle_b', ctypes.c_uint32,3),
  ('swizzle_a', ctypes.c_uint32,3),
  ('target', enum_pipe_texture_target,5),
  ('res_target', enum_pipe_texture_target,5),
  ('pot_width', ctypes.c_uint32,1),
  ('pot_height', ctypes.c_uint32,1),
  ('pot_depth', ctypes.c_uint32,1),
  ('level_zero_only', ctypes.c_uint32,1),
  ('tiled', ctypes.c_uint32,1),
  ('tiled_samples', ctypes.c_uint32,5),
]
struct_lp_texture_handle_state._fields_ = [
  ('static_state', struct_lp_static_texture_state),
  ('dynamic_state', struct_lp_jit_texture),
]
class struct_lp_texture_functions(Struct): pass
struct_lp_texture_functions._fields_ = [
  ('sample_functions', ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
  ('sampler_count', uint32_t),
  ('fetch_functions', ctypes.POINTER(ctypes.c_void_p)),
  ('size_function', ctypes.c_void_p),
  ('samples_function', ctypes.c_void_p),
  ('image_functions', ctypes.POINTER(ctypes.c_void_p)),
  ('state', struct_lp_texture_handle_state),
  ('sampled', ctypes.c_bool),
  ('storage', ctypes.c_bool),
  ('matrix', ctypes.c_void_p),
]
class struct_lp_texture_handle(Struct): pass
struct_lp_texture_handle._fields_ = [
  ('functions', ctypes.c_void_p),
  ('sampler_index', uint32_t),
]
class struct_lp_jit_bindless_texture(Struct): pass
struct_lp_jit_bindless_texture._fields_ = [
  ('base', ctypes.c_void_p),
  ('residency', ctypes.c_void_p),
  ('sampler_index', uint32_t),
]
class struct_lp_descriptor(Struct): pass
class struct_lp_descriptor_0(ctypes.Union): pass
class struct_lp_descriptor_0_0(Struct): pass
struct_lp_descriptor_0_0._fields_ = [
  ('texture', struct_lp_jit_bindless_texture),
  ('sampler', struct_lp_jit_sampler),
]
class struct_lp_descriptor_0_1(Struct): pass
struct_lp_descriptor_0_1._fields_ = [
  ('image', struct_lp_jit_image),
]
struct_lp_descriptor_0._anonymous_ = ['_0', '_1']
struct_lp_descriptor_0._fields_ = [
  ('_0', struct_lp_descriptor_0_0),
  ('_1', struct_lp_descriptor_0_1),
  ('buffer', struct_lp_jit_buffer),
  ('accel_struct', uint64_t),
]
struct_lp_descriptor._anonymous_ = ['_0']
struct_lp_descriptor._fields_ = [
  ('_0', struct_lp_descriptor_0),
  ('functions', ctypes.c_void_p),
]
try: (lp_build_flow_skip_begin:=dll.lp_build_flow_skip_begin).restype, lp_build_flow_skip_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context), ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_build_flow_skip_cond_break:=dll.lp_build_flow_skip_cond_break).restype, lp_build_flow_skip_cond_break.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_flow_skip_end:=dll.lp_build_flow_skip_end).restype, lp_build_flow_skip_end.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context)]
except AttributeError: pass

try: (lp_build_mask_begin:=dll.lp_build_mask_begin).restype, lp_build_mask_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type, LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_value:=dll.lp_build_mask_value).restype, lp_build_mask_value.argtypes = LLVMValueRef, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

try: (lp_build_mask_update:=dll.lp_build_mask_update).restype, lp_build_mask_update.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_force:=dll.lp_build_mask_force).restype, lp_build_mask_force.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_check:=dll.lp_build_mask_check).restype, lp_build_mask_check.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

try: (lp_build_mask_end:=dll.lp_build_mask_end).restype, lp_build_mask_end.argtypes = LLVMValueRef, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

class struct_lp_build_loop_state(Struct): pass
struct_lp_build_loop_state._fields_ = [
  ('block', LLVMBasicBlockRef),
  ('counter_var', LLVMValueRef),
  ('counter', LLVMValueRef),
  ('counter_type', LLVMTypeRef),
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]
try: (lp_build_loop_begin:=dll.lp_build_loop_begin).restype, lp_build_loop_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_end:=dll.lp_build_loop_end).restype, lp_build_loop_end.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_force_set_counter:=dll.lp_build_loop_force_set_counter).restype, lp_build_loop_force_set_counter.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_force_reload_counter:=dll.lp_build_loop_force_reload_counter).restype, lp_build_loop_force_reload_counter.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state)]
except AttributeError: pass

LLVMIntPredicate = CEnum(ctypes.c_uint32)
LLVMIntEQ = LLVMIntPredicate.define('LLVMIntEQ', 32)
LLVMIntNE = LLVMIntPredicate.define('LLVMIntNE', 33)
LLVMIntUGT = LLVMIntPredicate.define('LLVMIntUGT', 34)
LLVMIntUGE = LLVMIntPredicate.define('LLVMIntUGE', 35)
LLVMIntULT = LLVMIntPredicate.define('LLVMIntULT', 36)
LLVMIntULE = LLVMIntPredicate.define('LLVMIntULE', 37)
LLVMIntSGT = LLVMIntPredicate.define('LLVMIntSGT', 38)
LLVMIntSGE = LLVMIntPredicate.define('LLVMIntSGE', 39)
LLVMIntSLT = LLVMIntPredicate.define('LLVMIntSLT', 40)
LLVMIntSLE = LLVMIntPredicate.define('LLVMIntSLE', 41)

try: (lp_build_loop_end_cond:=dll.lp_build_loop_end_cond).restype, lp_build_loop_end_cond.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef, LLVMIntPredicate]
except AttributeError: pass

class struct_lp_build_for_loop_state(Struct): pass
struct_lp_build_for_loop_state._fields_ = [
  ('begin', LLVMBasicBlockRef),
  ('body', LLVMBasicBlockRef),
  ('exit', LLVMBasicBlockRef),
  ('counter_var', LLVMValueRef),
  ('counter', LLVMValueRef),
  ('counter_type', LLVMTypeRef),
  ('step', LLVMValueRef),
  ('cond', LLVMIntPredicate),
  ('end', LLVMValueRef),
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]
try: (lp_build_for_loop_begin:=dll.lp_build_for_loop_begin).restype, lp_build_for_loop_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_for_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_for_loop_end:=dll.lp_build_for_loop_end).restype, lp_build_for_loop_end.argtypes = None, [ctypes.POINTER(struct_lp_build_for_loop_state)]
except AttributeError: pass

class struct_lp_build_if_state(Struct): pass
struct_lp_build_if_state._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('condition', LLVMValueRef),
  ('entry_block', LLVMBasicBlockRef),
  ('true_block', LLVMBasicBlockRef),
  ('false_block', LLVMBasicBlockRef),
  ('merge_block', LLVMBasicBlockRef),
]
try: (lp_build_if:=dll.lp_build_if).restype, lp_build_if.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_else:=dll.lp_build_else).restype, lp_build_else.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError: pass

try: (lp_build_endif:=dll.lp_build_endif).restype, lp_build_endif.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError: pass

try: (lp_build_insert_new_block:=dll.lp_build_insert_new_block).restype, lp_build_insert_new_block.argtypes = LLVMBasicBlockRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_create_builder_at_entry:=dll.lp_create_builder_at_entry).restype, lp_create_builder_at_entry.argtypes = LLVMBuilderRef, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_build_alloca:=dll.lp_build_alloca).restype, lp_build_alloca.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_alloca_undef:=dll.lp_build_alloca_undef).restype, lp_build_alloca_undef.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_array_alloca:=dll.lp_build_array_alloca).restype, lp_build_array_alloca.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_mantissa:=dll.lp_mantissa).restype, lp_mantissa.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_shift:=dll.lp_const_shift).restype, lp_const_shift.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_offset:=dll.lp_const_offset).restype, lp_const_offset.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_scale:=dll.lp_const_scale).restype, lp_const_scale.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_min:=dll.lp_const_min).restype, lp_const_min.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_max:=dll.lp_const_max).restype, lp_const_max.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_eps:=dll.lp_const_eps).restype, lp_const_eps.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_build_undef:=dll.lp_build_undef).restype, lp_build_undef.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_zero:=dll.lp_build_zero).restype, lp_build_zero.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_one:=dll.lp_build_one).restype, lp_build_one.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_const_elem:=dll.lp_build_const_elem).restype, lp_build_const_elem.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError: pass

try: (lp_build_const_vec:=dll.lp_build_const_vec).restype, lp_build_const_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError: pass

try: (lp_build_const_int_vec:=dll.lp_build_const_int_vec).restype, lp_build_const_int_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_int64]
except AttributeError: pass

try: (lp_build_const_channel_vec:=dll.lp_build_const_channel_vec).restype, lp_build_const_channel_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_const_aos:=dll.lp_build_const_aos).restype, lp_build_const_aos.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError: pass

try: (lp_build_const_mask_aos:=dll.lp_build_const_mask_aos).restype, lp_build_const_mask_aos.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (lp_build_const_mask_aos_swizzled:=dll.lp_build_const_mask_aos_swizzled).restype, lp_build_const_mask_aos_swizzled.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError: pass

try: (lp_build_const_string:=dll.lp_build_const_string).restype, lp_build_const_string.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_const_func_pointer:=dll.lp_build_const_func_pointer).restype, lp_build_const_func_pointer.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_void_p, LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_const_func_pointer_from_type:=dll.lp_build_const_func_pointer_from_type).restype, lp_build_const_func_pointer_from_type.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_void_p, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_fd_dev_info(Struct): pass
class struct_fd_dev_info_0(ctypes.Union): pass
struct_fd_dev_info_0._fields_ = [
  ('num_sp_cores', uint32_t),
  ('num_ccu', uint32_t),
]
class struct_fd_dev_info_a6xx(Struct): pass
class struct_fd_dev_info_a6xx_magic(Struct): pass
struct_fd_dev_info_a6xx_magic._fields_ = [
  ('PC_POWER_CNTL', uint32_t),
  ('TPL1_DBG_ECO_CNTL', uint32_t),
  ('GRAS_DBG_ECO_CNTL', uint32_t),
  ('SP_CHICKEN_BITS', uint32_t),
  ('UCHE_CLIENT_PF', uint32_t),
  ('PC_MODE_CNTL', uint32_t),
  ('SP_DBG_ECO_CNTL', uint32_t),
  ('RB_DBG_ECO_CNTL', uint32_t),
  ('RB_DBG_ECO_CNTL_blit', uint32_t),
  ('HLSQ_DBG_ECO_CNTL', uint32_t),
  ('RB_UNKNOWN_8E01', uint32_t),
  ('VPC_DBG_ECO_CNTL', uint32_t),
  ('UCHE_UNKNOWN_0E12', uint32_t),
  ('RB_CCU_DBG_ECO_CNTL', uint32_t),
]
class struct_fd_dev_info_a6xx_magic_raw(Struct): pass
struct_fd_dev_info_a6xx_magic_raw._fields_ = [
  ('reg', uint32_t),
  ('value', uint32_t),
]
struct_fd_dev_info_a6xx._fields_ = [
  ('reg_size_vec4', uint32_t),
  ('instr_cache_size', uint32_t),
  ('has_hw_multiview', ctypes.c_bool),
  ('has_fs_tex_prefetch', ctypes.c_bool),
  ('supports_multiview_mask', ctypes.c_bool),
  ('concurrent_resolve', ctypes.c_bool),
  ('has_z24uint_s8uint', ctypes.c_bool),
  ('tess_use_shared', ctypes.c_bool),
  ('has_legacy_pipeline_shading_rate', ctypes.c_bool),
  ('storage_16bit', ctypes.c_bool),
  ('indirect_draw_wfm_quirk', ctypes.c_bool),
  ('depth_bounds_require_depth_test_quirk', ctypes.c_bool),
  ('has_tex_filter_cubic', ctypes.c_bool),
  ('has_separate_chroma_filter', ctypes.c_bool),
  ('has_sample_locations', ctypes.c_bool),
  ('has_cp_reg_write', ctypes.c_bool),
  ('has_8bpp_ubwc', ctypes.c_bool),
  ('has_lpac', ctypes.c_bool),
  ('has_getfiberid', ctypes.c_bool),
  ('mov_half_shared_quirk', ctypes.c_bool),
  ('has_movs', ctypes.c_bool),
  ('has_dp2acc', ctypes.c_bool),
  ('has_dp4acc', ctypes.c_bool),
  ('enable_lrz_fast_clear', ctypes.c_bool),
  ('has_lrz_dir_tracking', ctypes.c_bool),
  ('lrz_track_quirk', ctypes.c_bool),
  ('has_lrz_feedback', ctypes.c_bool),
  ('has_per_view_viewport', ctypes.c_bool),
  ('has_gmem_fast_clear', ctypes.c_bool),
  ('sysmem_per_ccu_depth_cache_size', uint32_t),
  ('sysmem_per_ccu_color_cache_size', uint32_t),
  ('gmem_ccu_color_cache_fraction', uint32_t),
  ('prim_alloc_threshold', uint32_t),
  ('vs_max_inputs_count', uint32_t),
  ('supports_double_threadsize', ctypes.c_bool),
  ('has_sampler_minmax', ctypes.c_bool),
  ('broken_ds_ubwc_quirk', ctypes.c_bool),
  ('has_scalar_alu', ctypes.c_bool),
  ('has_early_preamble', ctypes.c_bool),
  ('has_isam_v', ctypes.c_bool),
  ('has_ssbo_imm_offsets', ctypes.c_bool),
  ('has_coherent_ubwc_flag_caches', ctypes.c_bool),
  ('has_attachment_shading_rate', ctypes.c_bool),
  ('has_ubwc_linear_mipmap_fallback', ctypes.c_bool),
  ('predtf_nop_quirk', ctypes.c_bool),
  ('prede_nop_quirk', ctypes.c_bool),
  ('has_sad', ctypes.c_bool),
  ('is_a702', ctypes.c_bool),
  ('magic', struct_fd_dev_info_a6xx_magic),
  ('magic_raw', (struct_fd_dev_info_a6xx_magic_raw * 64)),
  ('max_sets', uint32_t),
  ('line_width_min', ctypes.c_float),
  ('line_width_max', ctypes.c_float),
  ('has_bin_mask', ctypes.c_bool),
]
class struct_fd_dev_info_a7xx(Struct): pass
struct_fd_dev_info_a7xx._fields_ = [
  ('stsc_duplication_quirk', ctypes.c_bool),
  ('has_event_write_sample_count', ctypes.c_bool),
  ('has_64b_ssbo_atomics', ctypes.c_bool),
  ('cmdbuf_start_a725_quirk', ctypes.c_bool),
  ('load_inline_uniforms_via_preamble_ldgk', ctypes.c_bool),
  ('load_shader_consts_via_preamble', ctypes.c_bool),
  ('has_gmem_vpc_attr_buf', ctypes.c_bool),
  ('sysmem_vpc_attr_buf_size', uint32_t),
  ('gmem_vpc_attr_buf_size', uint32_t),
  ('supports_uav_ubwc', ctypes.c_bool),
  ('ubwc_unorm_snorm_int_compatible', ctypes.c_bool),
  ('fs_must_have_non_zero_constlen_quirk', ctypes.c_bool),
  ('gs_vpc_adjacency_quirk', ctypes.c_bool),
  ('enable_tp_ubwc_flag_hint', ctypes.c_bool),
  ('storage_8bit', ctypes.c_bool),
  ('ubwc_all_formats_compatible', ctypes.c_bool),
  ('has_compliant_dp4acc', ctypes.c_bool),
  ('has_generic_clear', ctypes.c_bool),
  ('r8g8_faulty_fast_clear_quirk', ctypes.c_bool),
  ('ubwc_coherency_quirk', ctypes.c_bool),
  ('has_persistent_counter', ctypes.c_bool),
  ('has_primitive_shading_rate', ctypes.c_bool),
  ('reading_shading_rate_requires_smask_quirk', ctypes.c_bool),
  ('has_ray_intersection', ctypes.c_bool),
  ('has_sw_fuse', ctypes.c_bool),
  ('has_rt_workaround', ctypes.c_bool),
  ('has_alias_rt', ctypes.c_bool),
  ('has_abs_bin_mask', ctypes.c_bool),
  ('new_control_regs', ctypes.c_bool),
]
struct_fd_dev_info._anonymous_ = ['_0']
struct_fd_dev_info._fields_ = [
  ('chip', uint8_t),
  ('tile_align_w', uint32_t),
  ('tile_align_h', uint32_t),
  ('gmem_align_w', uint32_t),
  ('gmem_align_h', uint32_t),
  ('tile_max_w', uint32_t),
  ('tile_max_h', uint32_t),
  ('num_vsc_pipes', uint32_t),
  ('cs_shared_mem_size', uint32_t),
  ('wave_granularity', ctypes.c_int32),
  ('highest_bank_bit', uint32_t),
  ('ubwc_swizzle', uint32_t),
  ('macrotile_mode', uint32_t),
  ('fibers_per_sp', uint32_t),
  ('threadsize_base', uint32_t),
  ('max_waves', uint32_t),
  ('compute_lb_size', uint32_t),
  ('_0', struct_fd_dev_info_0),
  ('a6xx', struct_fd_dev_info_a6xx),
  ('a7xx', struct_fd_dev_info_a7xx),
]
class struct_fd_dev_id(Struct): pass
struct_fd_dev_id._fields_ = [
  ('gpu_id', uint32_t),
  ('chip_id', uint64_t),
]
try: (fd_dev_info_raw:=dll.fd_dev_info_raw).restype, fd_dev_info_raw.argtypes = ctypes.POINTER(struct_fd_dev_info), [ctypes.POINTER(struct_fd_dev_id)]
except AttributeError: pass

try: (fd_dev_info:=dll.fd_dev_info).restype, fd_dev_info.argtypes = struct_fd_dev_info, [ctypes.POINTER(struct_fd_dev_id)]
except AttributeError: pass

try: (fd_dev_info_raw_by_name:=dll.fd_dev_info_raw_by_name).restype, fd_dev_info_raw_by_name.argtypes = ctypes.POINTER(struct_fd_dev_info), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (fd_dev_name:=dll.fd_dev_name).restype, fd_dev_name.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(struct_fd_dev_id)]
except AttributeError: pass

try: (fd_dev_info_apply_dbg_options:=dll.fd_dev_info_apply_dbg_options).restype, fd_dev_info_apply_dbg_options.argtypes = None, [ctypes.POINTER(struct_fd_dev_info)]
except AttributeError: pass

class struct_ir3_ra_reg_set(Struct): pass
class struct_ir3_shader(Struct): pass
class struct_ir3_compiler_options(Struct): pass
struct_ir3_compiler_options._fields_ = [
  ('push_ubo_with_preamble', ctypes.c_bool),
  ('disable_cache', ctypes.c_bool),
  ('bindless_fb_read_descriptor', ctypes.c_int32),
  ('bindless_fb_read_slot', ctypes.c_int32),
  ('storage_16bit', ctypes.c_bool),
  ('storage_8bit', ctypes.c_bool),
  ('lower_base_vertex', ctypes.c_bool),
  ('shared_push_consts', ctypes.c_bool),
  ('dual_color_blend_by_location', ctypes.c_bool),
  ('uche_trap_base', uint64_t),
]
class struct_ir3_compiler(Struct): pass
class struct_fd_device(Struct): pass
class struct_disk_cache(Struct): pass
type_t = CEnum(ctypes.c_uint32)
TYPE_F16 = type_t.define('TYPE_F16', 0)
TYPE_F32 = type_t.define('TYPE_F32', 1)
TYPE_U16 = type_t.define('TYPE_U16', 2)
TYPE_U32 = type_t.define('TYPE_U32', 3)
TYPE_S16 = type_t.define('TYPE_S16', 4)
TYPE_S32 = type_t.define('TYPE_S32', 5)
TYPE_ATOMIC_U64 = type_t.define('TYPE_ATOMIC_U64', 6)
TYPE_U8 = type_t.define('TYPE_U8', 6)
TYPE_U8_32 = type_t.define('TYPE_U8_32', 7)

class struct_ir3_compiler_delay_slots(Struct): pass
struct_ir3_compiler_delay_slots._fields_ = [
  ('alu_to_alu', ctypes.c_uint32),
  ('non_alu', ctypes.c_uint32),
  ('cat3_src2_read', ctypes.c_uint32),
]
struct_ir3_compiler._fields_ = [
  ('dev', ctypes.POINTER(struct_fd_device)),
  ('dev_id', ctypes.POINTER(struct_fd_dev_id)),
  ('gen', uint8_t),
  ('shader_count', uint32_t),
  ('disk_cache', ctypes.POINTER(struct_disk_cache)),
  ('nir_options', struct_nir_shader_compiler_options),
  ('options', struct_ir3_compiler_options),
  ('is_64bit', ctypes.c_bool),
  ('flat_bypass', ctypes.c_bool),
  ('levels_add_one', ctypes.c_bool),
  ('unminify_coords', ctypes.c_bool),
  ('txf_ms_with_isaml', ctypes.c_bool),
  ('array_index_add_half', ctypes.c_bool),
  ('samgq_workaround', ctypes.c_bool),
  ('tess_use_shared', ctypes.c_bool),
  ('mergedregs', ctypes.c_bool),
  ('max_const_pipeline', uint16_t),
  ('max_const_geom', uint16_t),
  ('max_const_frag', uint16_t),
  ('max_const_safe', uint16_t),
  ('max_const_compute', uint16_t),
  ('compute_lb_size', uint32_t),
  ('instr_align', uint32_t),
  ('const_upload_unit', uint32_t),
  ('threadsize_base', uint32_t),
  ('wave_granularity', uint32_t),
  ('max_waves', uint32_t),
  ('reg_size_vec4', uint32_t),
  ('local_mem_size', uint32_t),
  ('branchstack_size', uint32_t),
  ('pvtmem_per_fiber_align', uint32_t),
  ('has_clip_cull', ctypes.c_bool),
  ('has_pvtmem', ctypes.c_bool),
  ('has_isam_ssbo', ctypes.c_bool),
  ('has_isam_v', ctypes.c_bool),
  ('has_ssbo_imm_offsets', ctypes.c_bool),
  ('has_getfiberid', ctypes.c_bool),
  ('mov_half_shared_quirk', ctypes.c_bool),
  ('has_movs', ctypes.c_bool),
  ('has_shfl', ctypes.c_bool),
  ('has_bitwise_triops', ctypes.c_bool),
  ('num_predicates', uint32_t),
  ('bitops_can_write_predicates', ctypes.c_bool),
  ('has_branch_and_or', ctypes.c_bool),
  ('has_predication', ctypes.c_bool),
  ('predtf_nop_quirk', ctypes.c_bool),
  ('prede_nop_quirk', ctypes.c_bool),
  ('max_variable_workgroup_size', uint32_t),
  ('has_dp2acc', ctypes.c_bool),
  ('has_dp4acc', ctypes.c_bool),
  ('has_compliant_dp4acc', ctypes.c_bool),
  ('bool_type', type_t),
  ('has_shared_regfile', ctypes.c_bool),
  ('has_preamble', ctypes.c_bool),
  ('shared_consts_base_offset', uint16_t),
  ('shared_consts_size', uint64_t),
  ('geom_shared_consts_size_quirk', uint64_t),
  ('has_fs_tex_prefetch', ctypes.c_bool),
  ('stsc_duplication_quirk', ctypes.c_bool),
  ('load_shader_consts_via_preamble', ctypes.c_bool),
  ('load_inline_uniforms_via_preamble_ldgk', ctypes.c_bool),
  ('has_scalar_alu', ctypes.c_bool),
  ('fs_must_have_non_zero_constlen_quirk', ctypes.c_bool),
  ('has_early_preamble', ctypes.c_bool),
  ('has_rpt_bary_f', ctypes.c_bool),
  ('has_alias_tex', ctypes.c_bool),
  ('has_alias_rt', ctypes.c_bool),
  ('reading_shading_rate_requires_smask_quirk', ctypes.c_bool),
  ('delay_slots', struct_ir3_compiler_delay_slots),
]
try: (ir3_compiler_destroy:=dll.ir3_compiler_destroy).restype, ir3_compiler_destroy.argtypes = None, [ctypes.POINTER(struct_ir3_compiler)]
except AttributeError: pass

try: (ir3_compiler_create:=dll.ir3_compiler_create).restype, ir3_compiler_create.argtypes = ctypes.POINTER(struct_ir3_compiler), [ctypes.POINTER(struct_fd_device), ctypes.POINTER(struct_fd_dev_id), ctypes.POINTER(struct_fd_dev_info), ctypes.POINTER(struct_ir3_compiler_options)]
except AttributeError: pass

try: (ir3_disk_cache_init:=dll.ir3_disk_cache_init).restype, ir3_disk_cache_init.argtypes = None, [ctypes.POINTER(struct_ir3_compiler)]
except AttributeError: pass

try: (ir3_disk_cache_init_shader_key:=dll.ir3_disk_cache_init_shader_key).restype, ir3_disk_cache_init_shader_key.argtypes = None, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(struct_ir3_shader)]
except AttributeError: pass

class struct_ir3_shader_variant(Struct): pass
try: (ir3_retrieve_variant:=dll.ir3_retrieve_variant).restype, ir3_retrieve_variant.argtypes = ctypes.POINTER(struct_ir3_shader_variant), [ctypes.POINTER(struct_blob_reader), ctypes.POINTER(struct_ir3_compiler), ctypes.c_void_p]
except AttributeError: pass

try: (ir3_store_variant:=dll.ir3_store_variant).restype, ir3_store_variant.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_disk_cache_retrieve:=dll.ir3_disk_cache_retrieve).restype, ir3_disk_cache_retrieve.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_ir3_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_disk_cache_store:=dll.ir3_disk_cache_store).restype, ir3_disk_cache_store.argtypes = None, [ctypes.POINTER(struct_ir3_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_get_compiler_options:=dll.ir3_get_compiler_options).restype, ir3_get_compiler_options.argtypes = ctypes.POINTER(nir_shader_compiler_options), [ctypes.POINTER(struct_ir3_compiler)]
except AttributeError: pass

try: (ir3_compile_shader_nir:=dll.ir3_compile_shader_nir).restype, ir3_compile_shader_nir.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(struct_ir3_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

enum_ir3_shader_debug = CEnum(ctypes.c_uint32)
IR3_DBG_SHADER_VS = enum_ir3_shader_debug.define('IR3_DBG_SHADER_VS', 1)
IR3_DBG_SHADER_TCS = enum_ir3_shader_debug.define('IR3_DBG_SHADER_TCS', 2)
IR3_DBG_SHADER_TES = enum_ir3_shader_debug.define('IR3_DBG_SHADER_TES', 4)
IR3_DBG_SHADER_GS = enum_ir3_shader_debug.define('IR3_DBG_SHADER_GS', 8)
IR3_DBG_SHADER_FS = enum_ir3_shader_debug.define('IR3_DBG_SHADER_FS', 16)
IR3_DBG_SHADER_CS = enum_ir3_shader_debug.define('IR3_DBG_SHADER_CS', 32)
IR3_DBG_DISASM = enum_ir3_shader_debug.define('IR3_DBG_DISASM', 64)
IR3_DBG_OPTMSGS = enum_ir3_shader_debug.define('IR3_DBG_OPTMSGS', 128)
IR3_DBG_FORCES2EN = enum_ir3_shader_debug.define('IR3_DBG_FORCES2EN', 256)
IR3_DBG_NOUBOOPT = enum_ir3_shader_debug.define('IR3_DBG_NOUBOOPT', 512)
IR3_DBG_NOFP16 = enum_ir3_shader_debug.define('IR3_DBG_NOFP16', 1024)
IR3_DBG_NOCACHE = enum_ir3_shader_debug.define('IR3_DBG_NOCACHE', 2048)
IR3_DBG_SPILLALL = enum_ir3_shader_debug.define('IR3_DBG_SPILLALL', 4096)
IR3_DBG_NOPREAMBLE = enum_ir3_shader_debug.define('IR3_DBG_NOPREAMBLE', 8192)
IR3_DBG_SHADER_INTERNAL = enum_ir3_shader_debug.define('IR3_DBG_SHADER_INTERNAL', 16384)
IR3_DBG_FULLSYNC = enum_ir3_shader_debug.define('IR3_DBG_FULLSYNC', 32768)
IR3_DBG_FULLNOP = enum_ir3_shader_debug.define('IR3_DBG_FULLNOP', 65536)
IR3_DBG_NOEARLYPREAMBLE = enum_ir3_shader_debug.define('IR3_DBG_NOEARLYPREAMBLE', 131072)
IR3_DBG_NODESCPREFETCH = enum_ir3_shader_debug.define('IR3_DBG_NODESCPREFETCH', 262144)
IR3_DBG_EXPANDRPT = enum_ir3_shader_debug.define('IR3_DBG_EXPANDRPT', 524288)
IR3_DBG_ASM_ROUNDTRIP = enum_ir3_shader_debug.define('IR3_DBG_ASM_ROUNDTRIP', 1048576)
IR3_DBG_SCHEDMSGS = enum_ir3_shader_debug.define('IR3_DBG_SCHEDMSGS', 2097152)
IR3_DBG_RAMSGS = enum_ir3_shader_debug.define('IR3_DBG_RAMSGS', 4194304)
IR3_DBG_NOALIASTEX = enum_ir3_shader_debug.define('IR3_DBG_NOALIASTEX', 8388608)
IR3_DBG_NOALIASRT = enum_ir3_shader_debug.define('IR3_DBG_NOALIASRT', 16777216)

try: ir3_shader_debug = enum_ir3_shader_debug.in_dll(dll, 'ir3_shader_debug')
except (ValueError,AttributeError): pass
try: ir3_shader_override_path = ctypes.POINTER(ctypes.c_char).in_dll(dll, 'ir3_shader_override_path')
except (ValueError,AttributeError): pass
try: (ir3_shader_debug_as_string:=dll.ir3_shader_debug_as_string).restype, ir3_shader_debug_as_string.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

class struct_ir3_driver_params_cs(Struct): pass
struct_ir3_driver_params_cs._fields_ = [
  ('num_work_groups_x', uint32_t),
  ('num_work_groups_y', uint32_t),
  ('num_work_groups_z', uint32_t),
  ('work_dim', uint32_t),
  ('base_group_x', uint32_t),
  ('base_group_y', uint32_t),
  ('base_group_z', uint32_t),
  ('subgroup_size', uint32_t),
  ('local_group_size_x', uint32_t),
  ('local_group_size_y', uint32_t),
  ('local_group_size_z', uint32_t),
  ('subgroup_id_shift', uint32_t),
  ('workgroup_id_x', uint32_t),
  ('workgroup_id_y', uint32_t),
  ('workgroup_id_z', uint32_t),
  ('__pad', uint32_t),
]
class struct_ir3_driver_params_vs(Struct): pass
class struct_ir3_driver_params_vs_ucp(Struct): pass
struct_ir3_driver_params_vs_ucp._fields_ = [
  ('x', uint32_t),
  ('y', uint32_t),
  ('z', uint32_t),
  ('w', uint32_t),
]
struct_ir3_driver_params_vs._fields_ = [
  ('draw_id', uint32_t),
  ('vtxid_base', uint32_t),
  ('instid_base', uint32_t),
  ('vtxcnt_max', uint32_t),
  ('is_indexed_draw', uint32_t),
  ('ucp', (struct_ir3_driver_params_vs_ucp * 8)),
  ('__pad_37_39', (uint32_t * 3)),
]
class struct_ir3_driver_params_tcs(Struct): pass
struct_ir3_driver_params_tcs._fields_ = [
  ('default_outer_level_x', uint32_t),
  ('default_outer_level_y', uint32_t),
  ('default_outer_level_z', uint32_t),
  ('default_outer_level_w', uint32_t),
  ('default_inner_level_x', uint32_t),
  ('default_inner_level_y', uint32_t),
  ('__pad_06_07', (uint32_t * 2)),
]
class struct_ir3_driver_params_fs(Struct): pass
struct_ir3_driver_params_fs._fields_ = [
  ('subgroup_size', uint32_t),
  ('__pad_01_03', (uint32_t * 3)),
  ('frag_invocation_count', uint32_t),
  ('__pad_05_07', (uint32_t * 3)),
  ('frag_size', uint32_t),
  ('__pad_09', uint32_t),
  ('frag_offset', uint32_t),
  ('__pad_11_12', (uint32_t * 2)),
]
enum_ir3_bary = CEnum(ctypes.c_uint32)
IJ_PERSP_PIXEL = enum_ir3_bary.define('IJ_PERSP_PIXEL', 0)
IJ_PERSP_SAMPLE = enum_ir3_bary.define('IJ_PERSP_SAMPLE', 1)
IJ_PERSP_CENTROID = enum_ir3_bary.define('IJ_PERSP_CENTROID', 2)
IJ_PERSP_CENTER_RHW = enum_ir3_bary.define('IJ_PERSP_CENTER_RHW', 3)
IJ_LINEAR_PIXEL = enum_ir3_bary.define('IJ_LINEAR_PIXEL', 4)
IJ_LINEAR_CENTROID = enum_ir3_bary.define('IJ_LINEAR_CENTROID', 5)
IJ_LINEAR_SAMPLE = enum_ir3_bary.define('IJ_LINEAR_SAMPLE', 6)
IJ_COUNT = enum_ir3_bary.define('IJ_COUNT', 7)

enum_ir3_wavesize_option = CEnum(ctypes.c_uint32)
IR3_SINGLE_ONLY = enum_ir3_wavesize_option.define('IR3_SINGLE_ONLY', 0)
IR3_SINGLE_OR_DOUBLE = enum_ir3_wavesize_option.define('IR3_SINGLE_OR_DOUBLE', 1)
IR3_DOUBLE_ONLY = enum_ir3_wavesize_option.define('IR3_DOUBLE_ONLY', 2)

class struct_ir3_ubo_info(Struct): pass
struct_ir3_ubo_info._fields_ = [
  ('global_base', ctypes.POINTER(struct_nir_def)),
  ('block', uint32_t),
  ('bindless_base', uint16_t),
  ('bindless', ctypes.c_bool),
  ('global', ctypes.c_bool),
]
class struct_ir3_ubo_range(Struct): pass
struct_ir3_ubo_range._fields_ = [
  ('ubo', struct_ir3_ubo_info),
  ('offset', uint32_t),
  ('start', uint32_t),
  ('end', uint32_t),
]
class struct_ir3_ubo_analysis_state(Struct): pass
struct_ir3_ubo_analysis_state._fields_ = [
  ('range', (struct_ir3_ubo_range * 32)),
  ('num_enabled', uint32_t),
  ('size', uint32_t),
]
enum_ir3_push_consts_type = CEnum(ctypes.c_uint32)
IR3_PUSH_CONSTS_NONE = enum_ir3_push_consts_type.define('IR3_PUSH_CONSTS_NONE', 0)
IR3_PUSH_CONSTS_PER_STAGE = enum_ir3_push_consts_type.define('IR3_PUSH_CONSTS_PER_STAGE', 1)
IR3_PUSH_CONSTS_SHARED = enum_ir3_push_consts_type.define('IR3_PUSH_CONSTS_SHARED', 2)
IR3_PUSH_CONSTS_SHARED_PREAMBLE = enum_ir3_push_consts_type.define('IR3_PUSH_CONSTS_SHARED_PREAMBLE', 3)

class struct_ir3_driver_ubo(Struct): pass
struct_ir3_driver_ubo._fields_ = [
  ('idx', int32_t),
  ('size', uint32_t),
]
enum_ir3_const_alloc_type = CEnum(ctypes.c_uint32)
IR3_CONST_ALLOC_PUSH_CONSTS = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_PUSH_CONSTS', 0)
IR3_CONST_ALLOC_DYN_DESCRIPTOR_OFFSET = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_DYN_DESCRIPTOR_OFFSET', 1)
IR3_CONST_ALLOC_INLINE_UNIFORM_ADDRS = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_INLINE_UNIFORM_ADDRS', 2)
IR3_CONST_ALLOC_DRIVER_PARAMS = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_DRIVER_PARAMS', 3)
IR3_CONST_ALLOC_UBO_RANGES = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_UBO_RANGES', 4)
IR3_CONST_ALLOC_PREAMBLE = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_PREAMBLE', 5)
IR3_CONST_ALLOC_GLOBAL = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_GLOBAL', 6)
IR3_CONST_ALLOC_UBO_PTRS = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_UBO_PTRS', 7)
IR3_CONST_ALLOC_IMAGE_DIMS = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_IMAGE_DIMS', 8)
IR3_CONST_ALLOC_TFBO = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_TFBO', 9)
IR3_CONST_ALLOC_PRIMITIVE_PARAM = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_PRIMITIVE_PARAM', 10)
IR3_CONST_ALLOC_PRIMITIVE_MAP = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_PRIMITIVE_MAP', 11)
IR3_CONST_ALLOC_MAX = enum_ir3_const_alloc_type.define('IR3_CONST_ALLOC_MAX', 12)

class struct_ir3_const_allocation(Struct): pass
struct_ir3_const_allocation._fields_ = [
  ('offset_vec4', uint32_t),
  ('size_vec4', uint32_t),
  ('reserved_size_vec4', uint32_t),
  ('reserved_align_vec4', uint32_t),
]
class struct_ir3_const_allocations(Struct): pass
struct_ir3_const_allocations._fields_ = [
  ('consts', (struct_ir3_const_allocation * 12)),
  ('max_const_offset_vec4', uint32_t),
  ('reserved_vec4', uint32_t),
]
class struct_ir3_const_image_dims(Struct): pass
struct_ir3_const_image_dims._fields_ = [
  ('mask', uint32_t),
  ('count', uint32_t),
  ('off', (uint32_t * 32)),
]
class struct_ir3_imm_const_state(Struct): pass
struct_ir3_imm_const_state._fields_ = [
  ('size', ctypes.c_uint32),
  ('count', ctypes.c_uint32),
  ('values', ctypes.POINTER(uint32_t)),
]
class struct_ir3_const_state(Struct): pass
struct_ir3_const_state._fields_ = [
  ('num_ubos', ctypes.c_uint32),
  ('num_app_ubos', ctypes.c_uint32),
  ('num_driver_params', ctypes.c_uint32),
  ('consts_ubo', struct_ir3_driver_ubo),
  ('driver_params_ubo', struct_ir3_driver_ubo),
  ('primitive_map_ubo', struct_ir3_driver_ubo),
  ('primitive_param_ubo', struct_ir3_driver_ubo),
  ('allocs', struct_ir3_const_allocations),
  ('image_dims', struct_ir3_const_image_dims),
  ('ubo_state', struct_ir3_ubo_analysis_state),
  ('push_consts_type', enum_ir3_push_consts_type),
]
class struct_ir3_stream_output(Struct): pass
struct_ir3_stream_output._fields_ = [
  ('register_index', ctypes.c_uint32,6),
  ('start_component', ctypes.c_uint32,2),
  ('num_components', ctypes.c_uint32,3),
  ('output_buffer', ctypes.c_uint32,3),
  ('dst_offset', ctypes.c_uint32,16),
  ('stream', ctypes.c_uint32,2),
]
class struct_ir3_stream_output_info(Struct): pass
struct_ir3_stream_output_info._fields_ = [
  ('num_outputs', ctypes.c_uint32),
  ('stride', (uint16_t * 4)),
  ('streams_written', uint8_t),
  ('buffer_to_stream', (uint8_t * 4)),
  ('output', (struct_ir3_stream_output * 128)),
]
class struct_ir3_sampler_prefetch(Struct): pass
opc_t = CEnum(ctypes.c_uint32)
OPC_NOP = opc_t.define('OPC_NOP', 0)
OPC_JUMP = opc_t.define('OPC_JUMP', 2)
OPC_CALL = opc_t.define('OPC_CALL', 3)
OPC_RET = opc_t.define('OPC_RET', 4)
OPC_KILL = opc_t.define('OPC_KILL', 5)
OPC_END = opc_t.define('OPC_END', 6)
OPC_EMIT = opc_t.define('OPC_EMIT', 7)
OPC_CUT = opc_t.define('OPC_CUT', 8)
OPC_CHMASK = opc_t.define('OPC_CHMASK', 9)
OPC_CHSH = opc_t.define('OPC_CHSH', 10)
OPC_FLOW_REV = opc_t.define('OPC_FLOW_REV', 11)
OPC_BKT = opc_t.define('OPC_BKT', 16)
OPC_STKS = opc_t.define('OPC_STKS', 17)
OPC_STKR = opc_t.define('OPC_STKR', 18)
OPC_XSET = opc_t.define('OPC_XSET', 19)
OPC_XCLR = opc_t.define('OPC_XCLR', 20)
OPC_GETONE = opc_t.define('OPC_GETONE', 21)
OPC_DBG = opc_t.define('OPC_DBG', 22)
OPC_SHPS = opc_t.define('OPC_SHPS', 23)
OPC_SHPE = opc_t.define('OPC_SHPE', 24)
OPC_GETLAST = opc_t.define('OPC_GETLAST', 25)
OPC_PREDT = opc_t.define('OPC_PREDT', 29)
OPC_PREDF = opc_t.define('OPC_PREDF', 30)
OPC_PREDE = opc_t.define('OPC_PREDE', 31)
OPC_BR = opc_t.define('OPC_BR', 40)
OPC_BRAO = opc_t.define('OPC_BRAO', 41)
OPC_BRAA = opc_t.define('OPC_BRAA', 42)
OPC_BRAC = opc_t.define('OPC_BRAC', 43)
OPC_BANY = opc_t.define('OPC_BANY', 44)
OPC_BALL = opc_t.define('OPC_BALL', 45)
OPC_BRAX = opc_t.define('OPC_BRAX', 46)
OPC_DEMOTE = opc_t.define('OPC_DEMOTE', 47)
OPC_MOV = opc_t.define('OPC_MOV', 128)
OPC_MOVP = opc_t.define('OPC_MOVP', 129)
OPC_MOVS = opc_t.define('OPC_MOVS', 130)
OPC_MOVMSK = opc_t.define('OPC_MOVMSK', 131)
OPC_SWZ = opc_t.define('OPC_SWZ', 132)
OPC_GAT = opc_t.define('OPC_GAT', 133)
OPC_SCT = opc_t.define('OPC_SCT', 134)
OPC_MOV_IMMED = opc_t.define('OPC_MOV_IMMED', 168)
OPC_MOV_CONST = opc_t.define('OPC_MOV_CONST', 169)
OPC_MOV_GPR = opc_t.define('OPC_MOV_GPR', 170)
OPC_MOV_RELGPR = opc_t.define('OPC_MOV_RELGPR', 171)
OPC_MOV_RELCONST = opc_t.define('OPC_MOV_RELCONST', 172)
OPC_MOVS_IMMED = opc_t.define('OPC_MOVS_IMMED', 173)
OPC_MOVS_A0 = opc_t.define('OPC_MOVS_A0', 174)
OPC_BALLOT_MACRO = opc_t.define('OPC_BALLOT_MACRO', 178)
OPC_ANY_MACRO = opc_t.define('OPC_ANY_MACRO', 179)
OPC_ALL_MACRO = opc_t.define('OPC_ALL_MACRO', 180)
OPC_ELECT_MACRO = opc_t.define('OPC_ELECT_MACRO', 181)
OPC_READ_COND_MACRO = opc_t.define('OPC_READ_COND_MACRO', 182)
OPC_READ_FIRST_MACRO = opc_t.define('OPC_READ_FIRST_MACRO', 183)
OPC_SHPS_MACRO = opc_t.define('OPC_SHPS_MACRO', 184)
OPC_READ_GETLAST_MACRO = opc_t.define('OPC_READ_GETLAST_MACRO', 185)
OPC_SCAN_MACRO = opc_t.define('OPC_SCAN_MACRO', 186)
OPC_SCAN_CLUSTERS_MACRO = opc_t.define('OPC_SCAN_CLUSTERS_MACRO', 188)
OPC_ADD_F = opc_t.define('OPC_ADD_F', 256)
OPC_MIN_F = opc_t.define('OPC_MIN_F', 257)
OPC_MAX_F = opc_t.define('OPC_MAX_F', 258)
OPC_MUL_F = opc_t.define('OPC_MUL_F', 259)
OPC_SIGN_F = opc_t.define('OPC_SIGN_F', 260)
OPC_CMPS_F = opc_t.define('OPC_CMPS_F', 261)
OPC_ABSNEG_F = opc_t.define('OPC_ABSNEG_F', 262)
OPC_CMPV_F = opc_t.define('OPC_CMPV_F', 263)
OPC_FLOOR_F = opc_t.define('OPC_FLOOR_F', 265)
OPC_CEIL_F = opc_t.define('OPC_CEIL_F', 266)
OPC_RNDNE_F = opc_t.define('OPC_RNDNE_F', 267)
OPC_RNDAZ_F = opc_t.define('OPC_RNDAZ_F', 268)
OPC_TRUNC_F = opc_t.define('OPC_TRUNC_F', 269)
OPC_ADD_U = opc_t.define('OPC_ADD_U', 272)
OPC_ADD_S = opc_t.define('OPC_ADD_S', 273)
OPC_SUB_U = opc_t.define('OPC_SUB_U', 274)
OPC_SUB_S = opc_t.define('OPC_SUB_S', 275)
OPC_CMPS_U = opc_t.define('OPC_CMPS_U', 276)
OPC_CMPS_S = opc_t.define('OPC_CMPS_S', 277)
OPC_MIN_U = opc_t.define('OPC_MIN_U', 278)
OPC_MIN_S = opc_t.define('OPC_MIN_S', 279)
OPC_MAX_U = opc_t.define('OPC_MAX_U', 280)
OPC_MAX_S = opc_t.define('OPC_MAX_S', 281)
OPC_ABSNEG_S = opc_t.define('OPC_ABSNEG_S', 282)
OPC_AND_B = opc_t.define('OPC_AND_B', 284)
OPC_OR_B = opc_t.define('OPC_OR_B', 285)
OPC_NOT_B = opc_t.define('OPC_NOT_B', 286)
OPC_XOR_B = opc_t.define('OPC_XOR_B', 287)
OPC_CMPV_U = opc_t.define('OPC_CMPV_U', 289)
OPC_CMPV_S = opc_t.define('OPC_CMPV_S', 290)
OPC_MUL_U24 = opc_t.define('OPC_MUL_U24', 304)
OPC_MUL_S24 = opc_t.define('OPC_MUL_S24', 305)
OPC_MULL_U = opc_t.define('OPC_MULL_U', 306)
OPC_BFREV_B = opc_t.define('OPC_BFREV_B', 307)
OPC_CLZ_S = opc_t.define('OPC_CLZ_S', 308)
OPC_CLZ_B = opc_t.define('OPC_CLZ_B', 309)
OPC_SHL_B = opc_t.define('OPC_SHL_B', 310)
OPC_SHR_B = opc_t.define('OPC_SHR_B', 311)
OPC_ASHR_B = opc_t.define('OPC_ASHR_B', 312)
OPC_BARY_F = opc_t.define('OPC_BARY_F', 313)
OPC_MGEN_B = opc_t.define('OPC_MGEN_B', 314)
OPC_GETBIT_B = opc_t.define('OPC_GETBIT_B', 315)
OPC_SETRM = opc_t.define('OPC_SETRM', 316)
OPC_CBITS_B = opc_t.define('OPC_CBITS_B', 317)
OPC_SHB = opc_t.define('OPC_SHB', 318)
OPC_MSAD = opc_t.define('OPC_MSAD', 319)
OPC_FLAT_B = opc_t.define('OPC_FLAT_B', 320)
OPC_MAD_U16 = opc_t.define('OPC_MAD_U16', 384)
OPC_MADSH_U16 = opc_t.define('OPC_MADSH_U16', 385)
OPC_MAD_S16 = opc_t.define('OPC_MAD_S16', 386)
OPC_MADSH_M16 = opc_t.define('OPC_MADSH_M16', 387)
OPC_MAD_U24 = opc_t.define('OPC_MAD_U24', 388)
OPC_MAD_S24 = opc_t.define('OPC_MAD_S24', 389)
OPC_MAD_F16 = opc_t.define('OPC_MAD_F16', 390)
OPC_MAD_F32 = opc_t.define('OPC_MAD_F32', 391)
OPC_SEL_B16 = opc_t.define('OPC_SEL_B16', 392)
OPC_SEL_B32 = opc_t.define('OPC_SEL_B32', 393)
OPC_SEL_S16 = opc_t.define('OPC_SEL_S16', 394)
OPC_SEL_S32 = opc_t.define('OPC_SEL_S32', 395)
OPC_SEL_F16 = opc_t.define('OPC_SEL_F16', 396)
OPC_SEL_F32 = opc_t.define('OPC_SEL_F32', 397)
OPC_SAD_S16 = opc_t.define('OPC_SAD_S16', 398)
OPC_SAD_S32 = opc_t.define('OPC_SAD_S32', 399)
OPC_SHRM = opc_t.define('OPC_SHRM', 400)
OPC_SHLM = opc_t.define('OPC_SHLM', 401)
OPC_SHRG = opc_t.define('OPC_SHRG', 402)
OPC_SHLG = opc_t.define('OPC_SHLG', 403)
OPC_ANDG = opc_t.define('OPC_ANDG', 404)
OPC_DP2ACC = opc_t.define('OPC_DP2ACC', 405)
OPC_DP4ACC = opc_t.define('OPC_DP4ACC', 406)
OPC_WMM = opc_t.define('OPC_WMM', 407)
OPC_WMM_ACCU = opc_t.define('OPC_WMM_ACCU', 408)
OPC_RCP = opc_t.define('OPC_RCP', 512)
OPC_RSQ = opc_t.define('OPC_RSQ', 513)
OPC_LOG2 = opc_t.define('OPC_LOG2', 514)
OPC_EXP2 = opc_t.define('OPC_EXP2', 515)
OPC_SIN = opc_t.define('OPC_SIN', 516)
OPC_COS = opc_t.define('OPC_COS', 517)
OPC_SQRT = opc_t.define('OPC_SQRT', 518)
OPC_HRSQ = opc_t.define('OPC_HRSQ', 521)
OPC_HLOG2 = opc_t.define('OPC_HLOG2', 522)
OPC_HEXP2 = opc_t.define('OPC_HEXP2', 523)
OPC_ISAM = opc_t.define('OPC_ISAM', 640)
OPC_ISAML = opc_t.define('OPC_ISAML', 641)
OPC_ISAMM = opc_t.define('OPC_ISAMM', 642)
OPC_SAM = opc_t.define('OPC_SAM', 643)
OPC_SAMB = opc_t.define('OPC_SAMB', 644)
OPC_SAML = opc_t.define('OPC_SAML', 645)
OPC_SAMGQ = opc_t.define('OPC_SAMGQ', 646)
OPC_GETLOD = opc_t.define('OPC_GETLOD', 647)
OPC_CONV = opc_t.define('OPC_CONV', 648)
OPC_CONVM = opc_t.define('OPC_CONVM', 649)
OPC_GETSIZE = opc_t.define('OPC_GETSIZE', 650)
OPC_GETBUF = opc_t.define('OPC_GETBUF', 651)
OPC_GETPOS = opc_t.define('OPC_GETPOS', 652)
OPC_GETINFO = opc_t.define('OPC_GETINFO', 653)
OPC_DSX = opc_t.define('OPC_DSX', 654)
OPC_DSY = opc_t.define('OPC_DSY', 655)
OPC_GATHER4R = opc_t.define('OPC_GATHER4R', 656)
OPC_GATHER4G = opc_t.define('OPC_GATHER4G', 657)
OPC_GATHER4B = opc_t.define('OPC_GATHER4B', 658)
OPC_GATHER4A = opc_t.define('OPC_GATHER4A', 659)
OPC_SAMGP0 = opc_t.define('OPC_SAMGP0', 660)
OPC_SAMGP1 = opc_t.define('OPC_SAMGP1', 661)
OPC_SAMGP2 = opc_t.define('OPC_SAMGP2', 662)
OPC_SAMGP3 = opc_t.define('OPC_SAMGP3', 663)
OPC_DSXPP_1 = opc_t.define('OPC_DSXPP_1', 664)
OPC_DSYPP_1 = opc_t.define('OPC_DSYPP_1', 665)
OPC_RGETPOS = opc_t.define('OPC_RGETPOS', 666)
OPC_RGETINFO = opc_t.define('OPC_RGETINFO', 667)
OPC_BRCST_ACTIVE = opc_t.define('OPC_BRCST_ACTIVE', 668)
OPC_QUAD_SHUFFLE_BRCST = opc_t.define('OPC_QUAD_SHUFFLE_BRCST', 669)
OPC_QUAD_SHUFFLE_HORIZ = opc_t.define('OPC_QUAD_SHUFFLE_HORIZ', 670)
OPC_QUAD_SHUFFLE_VERT = opc_t.define('OPC_QUAD_SHUFFLE_VERT', 671)
OPC_QUAD_SHUFFLE_DIAG = opc_t.define('OPC_QUAD_SHUFFLE_DIAG', 672)
OPC_TCINV = opc_t.define('OPC_TCINV', 673)
OPC_DSXPP_MACRO = opc_t.define('OPC_DSXPP_MACRO', 675)
OPC_DSYPP_MACRO = opc_t.define('OPC_DSYPP_MACRO', 676)
OPC_LDG = opc_t.define('OPC_LDG', 768)
OPC_LDL = opc_t.define('OPC_LDL', 769)
OPC_LDP = opc_t.define('OPC_LDP', 770)
OPC_STG = opc_t.define('OPC_STG', 771)
OPC_STL = opc_t.define('OPC_STL', 772)
OPC_STP = opc_t.define('OPC_STP', 773)
OPC_LDIB = opc_t.define('OPC_LDIB', 774)
OPC_G2L = opc_t.define('OPC_G2L', 775)
OPC_L2G = opc_t.define('OPC_L2G', 776)
OPC_PREFETCH = opc_t.define('OPC_PREFETCH', 777)
OPC_LDLW = opc_t.define('OPC_LDLW', 778)
OPC_STLW = opc_t.define('OPC_STLW', 779)
OPC_RESFMT = opc_t.define('OPC_RESFMT', 782)
OPC_RESINFO = opc_t.define('OPC_RESINFO', 783)
OPC_ATOMIC_ADD = opc_t.define('OPC_ATOMIC_ADD', 784)
OPC_ATOMIC_SUB = opc_t.define('OPC_ATOMIC_SUB', 785)
OPC_ATOMIC_XCHG = opc_t.define('OPC_ATOMIC_XCHG', 786)
OPC_ATOMIC_INC = opc_t.define('OPC_ATOMIC_INC', 787)
OPC_ATOMIC_DEC = opc_t.define('OPC_ATOMIC_DEC', 788)
OPC_ATOMIC_CMPXCHG = opc_t.define('OPC_ATOMIC_CMPXCHG', 789)
OPC_ATOMIC_MIN = opc_t.define('OPC_ATOMIC_MIN', 790)
OPC_ATOMIC_MAX = opc_t.define('OPC_ATOMIC_MAX', 791)
OPC_ATOMIC_AND = opc_t.define('OPC_ATOMIC_AND', 792)
OPC_ATOMIC_OR = opc_t.define('OPC_ATOMIC_OR', 793)
OPC_ATOMIC_XOR = opc_t.define('OPC_ATOMIC_XOR', 794)
OPC_LDGB = opc_t.define('OPC_LDGB', 795)
OPC_STGB = opc_t.define('OPC_STGB', 796)
OPC_STIB = opc_t.define('OPC_STIB', 797)
OPC_LDC = opc_t.define('OPC_LDC', 798)
OPC_LDLV = opc_t.define('OPC_LDLV', 799)
OPC_PIPR = opc_t.define('OPC_PIPR', 800)
OPC_PIPC = opc_t.define('OPC_PIPC', 801)
OPC_EMIT2 = opc_t.define('OPC_EMIT2', 802)
OPC_ENDLS = opc_t.define('OPC_ENDLS', 803)
OPC_GETSPID = opc_t.define('OPC_GETSPID', 804)
OPC_GETWID = opc_t.define('OPC_GETWID', 805)
OPC_GETFIBERID = opc_t.define('OPC_GETFIBERID', 806)
OPC_SHFL = opc_t.define('OPC_SHFL', 807)
OPC_STC = opc_t.define('OPC_STC', 808)
OPC_RESINFO_B = opc_t.define('OPC_RESINFO_B', 809)
OPC_LDIB_B = opc_t.define('OPC_LDIB_B', 810)
OPC_STIB_B = opc_t.define('OPC_STIB_B', 811)
OPC_ATOMIC_B_ADD = opc_t.define('OPC_ATOMIC_B_ADD', 812)
OPC_ATOMIC_B_SUB = opc_t.define('OPC_ATOMIC_B_SUB', 813)
OPC_ATOMIC_B_XCHG = opc_t.define('OPC_ATOMIC_B_XCHG', 814)
OPC_ATOMIC_B_INC = opc_t.define('OPC_ATOMIC_B_INC', 815)
OPC_ATOMIC_B_DEC = opc_t.define('OPC_ATOMIC_B_DEC', 816)
OPC_ATOMIC_B_CMPXCHG = opc_t.define('OPC_ATOMIC_B_CMPXCHG', 817)
OPC_ATOMIC_B_MIN = opc_t.define('OPC_ATOMIC_B_MIN', 818)
OPC_ATOMIC_B_MAX = opc_t.define('OPC_ATOMIC_B_MAX', 819)
OPC_ATOMIC_B_AND = opc_t.define('OPC_ATOMIC_B_AND', 820)
OPC_ATOMIC_B_OR = opc_t.define('OPC_ATOMIC_B_OR', 821)
OPC_ATOMIC_B_XOR = opc_t.define('OPC_ATOMIC_B_XOR', 822)
OPC_ATOMIC_S_ADD = opc_t.define('OPC_ATOMIC_S_ADD', 823)
OPC_ATOMIC_S_SUB = opc_t.define('OPC_ATOMIC_S_SUB', 824)
OPC_ATOMIC_S_XCHG = opc_t.define('OPC_ATOMIC_S_XCHG', 825)
OPC_ATOMIC_S_INC = opc_t.define('OPC_ATOMIC_S_INC', 826)
OPC_ATOMIC_S_DEC = opc_t.define('OPC_ATOMIC_S_DEC', 827)
OPC_ATOMIC_S_CMPXCHG = opc_t.define('OPC_ATOMIC_S_CMPXCHG', 828)
OPC_ATOMIC_S_MIN = opc_t.define('OPC_ATOMIC_S_MIN', 829)
OPC_ATOMIC_S_MAX = opc_t.define('OPC_ATOMIC_S_MAX', 830)
OPC_ATOMIC_S_AND = opc_t.define('OPC_ATOMIC_S_AND', 831)
OPC_ATOMIC_S_OR = opc_t.define('OPC_ATOMIC_S_OR', 832)
OPC_ATOMIC_S_XOR = opc_t.define('OPC_ATOMIC_S_XOR', 833)
OPC_ATOMIC_G_ADD = opc_t.define('OPC_ATOMIC_G_ADD', 834)
OPC_ATOMIC_G_SUB = opc_t.define('OPC_ATOMIC_G_SUB', 835)
OPC_ATOMIC_G_XCHG = opc_t.define('OPC_ATOMIC_G_XCHG', 836)
OPC_ATOMIC_G_INC = opc_t.define('OPC_ATOMIC_G_INC', 837)
OPC_ATOMIC_G_DEC = opc_t.define('OPC_ATOMIC_G_DEC', 838)
OPC_ATOMIC_G_CMPXCHG = opc_t.define('OPC_ATOMIC_G_CMPXCHG', 839)
OPC_ATOMIC_G_MIN = opc_t.define('OPC_ATOMIC_G_MIN', 840)
OPC_ATOMIC_G_MAX = opc_t.define('OPC_ATOMIC_G_MAX', 841)
OPC_ATOMIC_G_AND = opc_t.define('OPC_ATOMIC_G_AND', 842)
OPC_ATOMIC_G_OR = opc_t.define('OPC_ATOMIC_G_OR', 843)
OPC_ATOMIC_G_XOR = opc_t.define('OPC_ATOMIC_G_XOR', 844)
OPC_LDG_A = opc_t.define('OPC_LDG_A', 845)
OPC_STG_A = opc_t.define('OPC_STG_A', 846)
OPC_SPILL_MACRO = opc_t.define('OPC_SPILL_MACRO', 847)
OPC_RELOAD_MACRO = opc_t.define('OPC_RELOAD_MACRO', 848)
OPC_LDC_K = opc_t.define('OPC_LDC_K', 849)
OPC_STSC = opc_t.define('OPC_STSC', 850)
OPC_LDG_K = opc_t.define('OPC_LDG_K', 851)
OPC_PUSH_CONSTS_LOAD_MACRO = opc_t.define('OPC_PUSH_CONSTS_LOAD_MACRO', 852)
OPC_RAY_INTERSECTION = opc_t.define('OPC_RAY_INTERSECTION', 858)
OPC_RESBASE = opc_t.define('OPC_RESBASE', 859)
OPC_BAR = opc_t.define('OPC_BAR', 896)
OPC_FENCE = opc_t.define('OPC_FENCE', 897)
OPC_SLEEP = opc_t.define('OPC_SLEEP', 898)
OPC_ICINV = opc_t.define('OPC_ICINV', 899)
OPC_DCCLN = opc_t.define('OPC_DCCLN', 900)
OPC_DCINV = opc_t.define('OPC_DCINV', 901)
OPC_DCFLU = opc_t.define('OPC_DCFLU', 902)
OPC_LOCK = opc_t.define('OPC_LOCK', 903)
OPC_UNLOCK = opc_t.define('OPC_UNLOCK', 904)
OPC_ALIAS = opc_t.define('OPC_ALIAS', 905)
OPC_CCINV = opc_t.define('OPC_CCINV', 906)
OPC_META_INPUT = opc_t.define('OPC_META_INPUT', 1024)
OPC_META_SPLIT = opc_t.define('OPC_META_SPLIT', 1026)
OPC_META_COLLECT = opc_t.define('OPC_META_COLLECT', 1027)
OPC_META_TEX_PREFETCH = opc_t.define('OPC_META_TEX_PREFETCH', 1028)
OPC_META_PARALLEL_COPY = opc_t.define('OPC_META_PARALLEL_COPY', 1029)
OPC_META_PHI = opc_t.define('OPC_META_PHI', 1030)
OPC_META_RAW = opc_t.define('OPC_META_RAW', 1031)

struct_ir3_sampler_prefetch._fields_ = [
  ('src', uint8_t),
  ('bindless', ctypes.c_bool),
  ('samp_id', uint8_t),
  ('tex_id', uint8_t),
  ('samp_bindless_id', uint16_t),
  ('tex_bindless_id', uint16_t),
  ('dst', uint8_t),
  ('wrmask', uint8_t),
  ('half_precision', uint8_t),
  ('tex_opc', opc_t),
]
class struct_ir3_shader_key(Struct): pass
class struct_ir3_shader_key_0(ctypes.Union): pass
class struct_ir3_shader_key_0_0(Struct): pass
struct_ir3_shader_key_0_0._fields_ = [
  ('ucp_enables', ctypes.c_uint32,8),
  ('has_per_samp', ctypes.c_uint32,1),
  ('sample_shading', ctypes.c_uint32,1),
  ('msaa', ctypes.c_uint32,1),
  ('rasterflat', ctypes.c_uint32,1),
  ('tessellation', ctypes.c_uint32,2),
  ('has_gs', ctypes.c_uint32,1),
  ('tcs_store_primid', ctypes.c_uint32,1),
  ('safe_constlen', ctypes.c_uint32,1),
  ('force_dual_color_blend', ctypes.c_uint32,1),
]
struct_ir3_shader_key_0._anonymous_ = ['_0']
struct_ir3_shader_key_0._fields_ = [
  ('_0', struct_ir3_shader_key_0_0),
  ('global', uint32_t),
]
struct_ir3_shader_key._anonymous_ = ['_0']
struct_ir3_shader_key._fields_ = [
  ('_0', struct_ir3_shader_key_0),
  ('vsamples', uint32_t),
  ('fsamples', uint32_t),
  ('vastc_srgb', uint16_t),
  ('fastc_srgb', uint16_t),
  ('vsampler_swizzles', (uint16_t * 16)),
  ('fsampler_swizzles', (uint16_t * 16)),
]
class struct_ir3_ibo_mapping(Struct): pass
struct_ir3_ibo_mapping._fields_ = [
  ('ssbo_to_tex', (uint8_t * 32)),
  ('image_to_tex', (uint8_t * 32)),
  ('tex_to_image', (uint8_t * 32)),
  ('num_tex', uint8_t),
  ('tex_base', uint8_t),
]
class struct_ir3_disasm_info(Struct): pass
struct_ir3_disasm_info._fields_ = [
  ('write_disasm', ctypes.c_bool),
  ('nir', ctypes.POINTER(ctypes.c_char)),
  ('disasm', ctypes.POINTER(ctypes.c_char)),
]
class struct_ir3_shader_nir_options(Struct): pass
struct_ir3_shader_nir_options._fields_ = [
  ('robust_modes', nir_variable_mode),
]
class struct_ir3_shader_options(Struct): pass
struct_ir3_shader_options._fields_ = [
  ('api_wavesize', enum_ir3_wavesize_option),
  ('real_wavesize', enum_ir3_wavesize_option),
  ('push_consts_type', enum_ir3_push_consts_type),
  ('push_consts_base', uint32_t),
  ('push_consts_dwords', uint32_t),
  ('const_allocs', struct_ir3_const_allocations),
  ('nir_options', struct_ir3_shader_nir_options),
  ('fragdata_dynamic_remap', ctypes.c_bool),
]
class struct_ir3_shader_output(Struct): pass
struct_ir3_shader_output._fields_ = [
  ('slot', uint8_t),
  ('regid', uint8_t),
  ('view', uint8_t),
  ('aliased_components', uint8_t,4),
  ('half', ctypes.c_bool,1),
]
class struct_fd_bo(Struct): pass
class struct_ir3(Struct): pass
class struct_ir3_instruction(Struct): pass
class struct_ir3_block(Struct): pass
struct_ir3_block._fields_ = [
  ('node', struct_list_head),
  ('shader', ctypes.POINTER(struct_ir3)),
  ('nblock', ctypes.POINTER(struct_nir_block)),
  ('instr_list', struct_list_head),
  ('successors', (ctypes.POINTER(struct_ir3_block) * 2)),
  ('divergent_condition', ctypes.c_bool),
  ('predecessors_count', ctypes.c_uint32),
  ('predecessors_sz', ctypes.c_uint32),
  ('predecessors', ctypes.POINTER(ctypes.POINTER(struct_ir3_block))),
  ('physical_predecessors_count', ctypes.c_uint32),
  ('physical_predecessors_sz', ctypes.c_uint32),
  ('physical_predecessors', ctypes.POINTER(ctypes.POINTER(struct_ir3_block))),
  ('physical_successors_count', ctypes.c_uint32),
  ('physical_successors_sz', ctypes.c_uint32),
  ('physical_successors', ctypes.POINTER(ctypes.POINTER(struct_ir3_block))),
  ('start_ip', uint16_t),
  ('end_ip', uint16_t),
  ('reconvergence_point', ctypes.c_bool),
  ('in_early_preamble', ctypes.c_bool),
  ('keeps_count', ctypes.c_uint32),
  ('keeps_sz', ctypes.c_uint32),
  ('keeps', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('data', ctypes.c_void_p),
  ('index', uint32_t),
  ('imm_dom', ctypes.POINTER(struct_ir3_block)),
  ('dom_children_count', ctypes.c_uint32),
  ('dom_children_sz', ctypes.c_uint32),
  ('dom_children', ctypes.POINTER(ctypes.POINTER(struct_ir3_block))),
  ('dom_pre_index', uint32_t),
  ('dom_post_index', uint32_t),
  ('loop_depth', uint32_t),
]
enum_ir3_instruction_flags = CEnum(ctypes.c_uint32)
IR3_INSTR_SY = enum_ir3_instruction_flags.define('IR3_INSTR_SY', 1)
IR3_INSTR_SS = enum_ir3_instruction_flags.define('IR3_INSTR_SS', 2)
IR3_INSTR_JP = enum_ir3_instruction_flags.define('IR3_INSTR_JP', 4)
IR3_INSTR_EQ = enum_ir3_instruction_flags.define('IR3_INSTR_EQ', 8)
IR3_INSTR_UL = enum_ir3_instruction_flags.define('IR3_INSTR_UL', 16)
IR3_INSTR_3D = enum_ir3_instruction_flags.define('IR3_INSTR_3D', 32)
IR3_INSTR_A = enum_ir3_instruction_flags.define('IR3_INSTR_A', 64)
IR3_INSTR_O = enum_ir3_instruction_flags.define('IR3_INSTR_O', 128)
IR3_INSTR_P = enum_ir3_instruction_flags.define('IR3_INSTR_P', 256)
IR3_INSTR_S = enum_ir3_instruction_flags.define('IR3_INSTR_S', 512)
IR3_INSTR_S2EN = enum_ir3_instruction_flags.define('IR3_INSTR_S2EN', 1024)
IR3_INSTR_SAT = enum_ir3_instruction_flags.define('IR3_INSTR_SAT', 2048)
IR3_INSTR_B = enum_ir3_instruction_flags.define('IR3_INSTR_B', 4096)
IR3_INSTR_NONUNIF = enum_ir3_instruction_flags.define('IR3_INSTR_NONUNIF', 8192)
IR3_INSTR_A1EN = enum_ir3_instruction_flags.define('IR3_INSTR_A1EN', 16384)
IR3_INSTR_U = enum_ir3_instruction_flags.define('IR3_INSTR_U', 32768)
IR3_INSTR_MARK = enum_ir3_instruction_flags.define('IR3_INSTR_MARK', 65536)
IR3_INSTR_SHARED_SPILL = enum_ir3_instruction_flags.define('IR3_INSTR_SHARED_SPILL', 65536)
IR3_INSTR_UNUSED = enum_ir3_instruction_flags.define('IR3_INSTR_UNUSED', 131072)
IR3_INSTR_NEEDS_HELPERS = enum_ir3_instruction_flags.define('IR3_INSTR_NEEDS_HELPERS', 262144)
IR3_INSTR_V = enum_ir3_instruction_flags.define('IR3_INSTR_V', 524288)
IR3_INSTR_INV_1D = enum_ir3_instruction_flags.define('IR3_INSTR_INV_1D', 1048576)
IR3_INSTR_IMM_OFFSET = enum_ir3_instruction_flags.define('IR3_INSTR_IMM_OFFSET', 2097152)

class struct_ir3_register(Struct): pass
enum_ir3_register_flags = CEnum(ctypes.c_uint32)
IR3_REG_CONST = enum_ir3_register_flags.define('IR3_REG_CONST', 1)
IR3_REG_IMMED = enum_ir3_register_flags.define('IR3_REG_IMMED', 2)
IR3_REG_HALF = enum_ir3_register_flags.define('IR3_REG_HALF', 4)
IR3_REG_SHARED = enum_ir3_register_flags.define('IR3_REG_SHARED', 8)
IR3_REG_RELATIV = enum_ir3_register_flags.define('IR3_REG_RELATIV', 16)
IR3_REG_R = enum_ir3_register_flags.define('IR3_REG_R', 32)
IR3_REG_FNEG = enum_ir3_register_flags.define('IR3_REG_FNEG', 64)
IR3_REG_FABS = enum_ir3_register_flags.define('IR3_REG_FABS', 128)
IR3_REG_SNEG = enum_ir3_register_flags.define('IR3_REG_SNEG', 256)
IR3_REG_SABS = enum_ir3_register_flags.define('IR3_REG_SABS', 512)
IR3_REG_BNOT = enum_ir3_register_flags.define('IR3_REG_BNOT', 1024)
IR3_REG_EI = enum_ir3_register_flags.define('IR3_REG_EI', 2048)
IR3_REG_SSA = enum_ir3_register_flags.define('IR3_REG_SSA', 4096)
IR3_REG_ARRAY = enum_ir3_register_flags.define('IR3_REG_ARRAY', 8192)
IR3_REG_KILL = enum_ir3_register_flags.define('IR3_REG_KILL', 16384)
IR3_REG_FIRST_KILL = enum_ir3_register_flags.define('IR3_REG_FIRST_KILL', 32768)
IR3_REG_UNUSED = enum_ir3_register_flags.define('IR3_REG_UNUSED', 65536)
IR3_REG_EARLY_CLOBBER = enum_ir3_register_flags.define('IR3_REG_EARLY_CLOBBER', 131072)
IR3_REG_LAST_USE = enum_ir3_register_flags.define('IR3_REG_LAST_USE', 262144)
IR3_REG_PREDICATE = enum_ir3_register_flags.define('IR3_REG_PREDICATE', 524288)
IR3_REG_RT = enum_ir3_register_flags.define('IR3_REG_RT', 1048576)
IR3_REG_ALIAS = enum_ir3_register_flags.define('IR3_REG_ALIAS', 2097152)
IR3_REG_FIRST_ALIAS = enum_ir3_register_flags.define('IR3_REG_FIRST_ALIAS', 4194304)

class struct_ir3_register_0(ctypes.Union): pass
class struct_ir3_register_0_array(Struct): pass
struct_ir3_register_0_array._fields_ = [
  ('id', uint16_t),
  ('offset', int16_t),
  ('base', uint16_t),
]
struct_ir3_register_0._fields_ = [
  ('iim_val', int32_t),
  ('uim_val', uint32_t),
  ('fim_val', ctypes.c_float),
  ('array', struct_ir3_register_0_array),
]
class struct_ir3_merge_set(Struct): pass
struct_ir3_merge_set._fields_ = [
  ('preferred_reg', uint16_t),
  ('size', uint16_t),
  ('alignment', uint16_t),
  ('interval_start', ctypes.c_uint32),
  ('spill_slot', ctypes.c_uint32),
  ('regs_count', ctypes.c_uint32),
  ('regs', ctypes.POINTER(ctypes.POINTER(struct_ir3_register))),
]
struct_ir3_register._anonymous_ = ['_0']
struct_ir3_register._fields_ = [
  ('flags', enum_ir3_register_flags),
  ('name', ctypes.c_uint32),
  ('wrmask', ctypes.c_uint32,16),
  ('size', ctypes.c_uint32,16),
  ('num', uint16_t),
  ('_0', struct_ir3_register_0),
  ('instr', ctypes.POINTER(struct_ir3_instruction)),
  ('def', ctypes.POINTER(struct_ir3_register)),
  ('tied', ctypes.POINTER(struct_ir3_register)),
  ('spill_slot', ctypes.c_uint32),
  ('next_use', ctypes.c_uint32),
  ('merge_set_offset', ctypes.c_uint32),
  ('merge_set', ctypes.POINTER(struct_ir3_merge_set)),
  ('interval_start', ctypes.c_uint32),
  ('interval_end', ctypes.c_uint32),
]
class struct_ir3_instruction_0(ctypes.Union): pass
class struct_ir3_instruction_0_cat0(Struct): pass
struct_ir3_instruction_0_cat0._fields_ = [
  ('inv1', ctypes.c_char),
  ('inv2', ctypes.c_char),
  ('immed', ctypes.c_int32),
  ('target', ctypes.POINTER(struct_ir3_block)),
  ('target_label', ctypes.POINTER(ctypes.c_char)),
  ('idx', ctypes.c_uint32),
]
class struct_ir3_instruction_0_cat1(Struct): pass
round_t = CEnum(ctypes.c_uint32)
ROUND_ZERO = round_t.define('ROUND_ZERO', 0)
ROUND_EVEN = round_t.define('ROUND_EVEN', 1)
ROUND_POS_INF = round_t.define('ROUND_POS_INF', 2)
ROUND_NEG_INF = round_t.define('ROUND_NEG_INF', 3)

reduce_op_t = CEnum(ctypes.c_uint32)
REDUCE_OP_ADD_U = reduce_op_t.define('REDUCE_OP_ADD_U', 0)
REDUCE_OP_ADD_F = reduce_op_t.define('REDUCE_OP_ADD_F', 1)
REDUCE_OP_MUL_U = reduce_op_t.define('REDUCE_OP_MUL_U', 2)
REDUCE_OP_MUL_F = reduce_op_t.define('REDUCE_OP_MUL_F', 3)
REDUCE_OP_MIN_U = reduce_op_t.define('REDUCE_OP_MIN_U', 4)
REDUCE_OP_MIN_S = reduce_op_t.define('REDUCE_OP_MIN_S', 5)
REDUCE_OP_MIN_F = reduce_op_t.define('REDUCE_OP_MIN_F', 6)
REDUCE_OP_MAX_U = reduce_op_t.define('REDUCE_OP_MAX_U', 7)
REDUCE_OP_MAX_S = reduce_op_t.define('REDUCE_OP_MAX_S', 8)
REDUCE_OP_MAX_F = reduce_op_t.define('REDUCE_OP_MAX_F', 9)
REDUCE_OP_AND_B = reduce_op_t.define('REDUCE_OP_AND_B', 10)
REDUCE_OP_OR_B = reduce_op_t.define('REDUCE_OP_OR_B', 11)
REDUCE_OP_XOR_B = reduce_op_t.define('REDUCE_OP_XOR_B', 12)

struct_ir3_instruction_0_cat1._fields_ = [
  ('src_type', type_t),
  ('dst_type', type_t),
  ('round', round_t),
  ('reduce_op', reduce_op_t),
]
class struct_ir3_instruction_0_cat2(Struct): pass
struct_ir3_instruction_0_cat2_condition = CEnum(ctypes.c_uint32)
IR3_COND_LT = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_LT', 0)
IR3_COND_LE = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_LE', 1)
IR3_COND_GT = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_GT', 2)
IR3_COND_GE = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_GE', 3)
IR3_COND_EQ = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_EQ', 4)
IR3_COND_NE = struct_ir3_instruction_0_cat2_condition.define('IR3_COND_NE', 5)

struct_ir3_instruction_0_cat2._fields_ = [
  ('condition', struct_ir3_instruction_0_cat2_condition),
]
class struct_ir3_instruction_0_cat3(Struct): pass
struct_ir3_instruction_0_cat3_signedness = CEnum(ctypes.c_uint32)
IR3_SRC_UNSIGNED = struct_ir3_instruction_0_cat3_signedness.define('IR3_SRC_UNSIGNED', 0)
IR3_SRC_MIXED = struct_ir3_instruction_0_cat3_signedness.define('IR3_SRC_MIXED', 1)

struct_ir3_instruction_0_cat3_packed = CEnum(ctypes.c_uint32)
IR3_SRC_PACKED_LOW = struct_ir3_instruction_0_cat3_packed.define('IR3_SRC_PACKED_LOW', 0)
IR3_SRC_PACKED_HIGH = struct_ir3_instruction_0_cat3_packed.define('IR3_SRC_PACKED_HIGH', 1)

struct_ir3_instruction_0_cat3._fields_ = [
  ('signedness', struct_ir3_instruction_0_cat3_signedness),
  ('packed', struct_ir3_instruction_0_cat3_packed),
  ('swapped', ctypes.c_bool),
]
class struct_ir3_instruction_0_cat5(Struct): pass
struct_ir3_instruction_0_cat5._fields_ = [
  ('samp', ctypes.c_uint32),
  ('tex', ctypes.c_uint32),
  ('tex_base', ctypes.c_uint32,3),
  ('cluster_size', ctypes.c_uint32,4),
  ('type', type_t),
]
class struct_ir3_instruction_0_cat6(Struct): pass
ir3_shfl_mode = CEnum(ctypes.c_uint32)
SHFL_XOR = ir3_shfl_mode.define('SHFL_XOR', 1)
SHFL_UP = ir3_shfl_mode.define('SHFL_UP', 2)
SHFL_DOWN = ir3_shfl_mode.define('SHFL_DOWN', 3)
SHFL_RUP = ir3_shfl_mode.define('SHFL_RUP', 6)
SHFL_RDOWN = ir3_shfl_mode.define('SHFL_RDOWN', 7)

struct_ir3_instruction_0_cat6._fields_ = [
  ('type', type_t),
  ('dst_offset', ctypes.c_int32),
  ('iim_val', ctypes.c_int32),
  ('d', ctypes.c_uint32,3),
  ('typed', ctypes.c_bool,1),
  ('base', ctypes.c_uint32,3),
  ('shfl_mode', ir3_shfl_mode,3),
]
class struct_ir3_instruction_0_cat7(Struct): pass
ir3_alias_scope = CEnum(ctypes.c_uint32)
ALIAS_TEX = ir3_alias_scope.define('ALIAS_TEX', 0)
ALIAS_RT = ir3_alias_scope.define('ALIAS_RT', 1)
ALIAS_MEM = ir3_alias_scope.define('ALIAS_MEM', 2)

struct_ir3_instruction_0_cat7._fields_ = [
  ('w', ctypes.c_uint32,1),
  ('r', ctypes.c_uint32,1),
  ('l', ctypes.c_uint32,1),
  ('g', ctypes.c_uint32,1),
  ('alias_scope', ir3_alias_scope),
  ('alias_table_size_minus_one', ctypes.c_uint32),
  ('alias_type_float', ctypes.c_bool),
]
class struct_ir3_instruction_0_split(Struct): pass
struct_ir3_instruction_0_split._fields_ = [
  ('off', ctypes.c_int32),
]
class struct_ir3_instruction_0_end(Struct): pass
struct_ir3_instruction_0_end._fields_ = [
  ('outidxs', ctypes.POINTER(ctypes.c_uint32)),
]
class struct_ir3_instruction_0_phi(Struct): pass
struct_ir3_instruction_0_phi._fields_ = [
  ('nphi', ctypes.c_void_p),
  ('comp', ctypes.c_uint32),
]
class struct_ir3_instruction_0_prefetch(Struct): pass
struct_ir3_instruction_0_prefetch._fields_ = [
  ('samp', ctypes.c_uint32),
  ('tex', ctypes.c_uint32),
  ('input_offset', ctypes.c_uint32),
  ('samp_base', ctypes.c_uint32,3),
  ('tex_base', ctypes.c_uint32,3),
]
class struct_ir3_instruction_0_input(Struct): pass
struct_ir3_instruction_0_input._fields_ = [
  ('inidx', ctypes.c_int32),
  ('sysval', gl_system_value),
]
class struct_ir3_instruction_0_push_consts(Struct): pass
struct_ir3_instruction_0_push_consts._fields_ = [
  ('src_base', ctypes.c_uint32),
  ('src_size', ctypes.c_uint32),
  ('dst_base', ctypes.c_uint32),
]
class struct_ir3_instruction_0_raw(Struct): pass
struct_ir3_instruction_0_raw._fields_ = [
  ('value', uint64_t),
]
struct_ir3_instruction_0._fields_ = [
  ('cat0', struct_ir3_instruction_0_cat0),
  ('cat1', struct_ir3_instruction_0_cat1),
  ('cat2', struct_ir3_instruction_0_cat2),
  ('cat3', struct_ir3_instruction_0_cat3),
  ('cat5', struct_ir3_instruction_0_cat5),
  ('cat6', struct_ir3_instruction_0_cat6),
  ('cat7', struct_ir3_instruction_0_cat7),
  ('split', struct_ir3_instruction_0_split),
  ('end', struct_ir3_instruction_0_end),
  ('phi', struct_ir3_instruction_0_phi),
  ('prefetch', struct_ir3_instruction_0_prefetch),
  ('input', struct_ir3_instruction_0_input),
  ('push_consts', struct_ir3_instruction_0_push_consts),
  ('raw', struct_ir3_instruction_0_raw),
]
struct_ir3_instruction_barrier_class = CEnum(ctypes.c_uint32)
IR3_BARRIER_EVERYTHING = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_EVERYTHING', 1)
IR3_BARRIER_SHARED_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_SHARED_R', 2)
IR3_BARRIER_SHARED_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_SHARED_W', 4)
IR3_BARRIER_IMAGE_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_IMAGE_R', 8)
IR3_BARRIER_IMAGE_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_IMAGE_W', 16)
IR3_BARRIER_BUFFER_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_BUFFER_R', 32)
IR3_BARRIER_BUFFER_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_BUFFER_W', 64)
IR3_BARRIER_ARRAY_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_ARRAY_R', 128)
IR3_BARRIER_ARRAY_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_ARRAY_W', 256)
IR3_BARRIER_PRIVATE_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_PRIVATE_R', 512)
IR3_BARRIER_PRIVATE_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_PRIVATE_W', 1024)
IR3_BARRIER_CONST_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_CONST_W', 2048)
IR3_BARRIER_ACTIVE_FIBERS_R = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_ACTIVE_FIBERS_R', 4096)
IR3_BARRIER_ACTIVE_FIBERS_W = struct_ir3_instruction_barrier_class.define('IR3_BARRIER_ACTIVE_FIBERS_W', 8192)

struct_ir3_instruction._anonymous_ = ['_0']
struct_ir3_instruction._fields_ = [
  ('block', ctypes.POINTER(struct_ir3_block)),
  ('opc', opc_t),
  ('flags', enum_ir3_instruction_flags),
  ('repeat', uint8_t),
  ('nop', uint8_t),
  ('srcs_count', ctypes.c_uint32),
  ('dsts_count', ctypes.c_uint32),
  ('dsts', ctypes.POINTER(ctypes.POINTER(struct_ir3_register))),
  ('srcs', ctypes.POINTER(ctypes.POINTER(struct_ir3_register))),
  ('_0', struct_ir3_instruction_0),
  ('ip', uint32_t),
  ('data', ctypes.c_void_p),
  ('uses', ctypes.POINTER(struct_set)),
  ('use_count', ctypes.c_int32),
  ('address', ctypes.POINTER(struct_ir3_register)),
  ('deps_count', ctypes.c_uint32),
  ('deps_sz', ctypes.c_uint32),
  ('deps', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('barrier_class', struct_ir3_instruction_barrier_class),
  ('barrier_conflict', struct_ir3_instruction_barrier_class),
  ('node', struct_list_head),
  ('rpt_node', struct_list_head),
  ('serialno', uint32_t),
  ('line', ctypes.c_int32),
]
struct_ir3._fields_ = [
  ('compiler', ctypes.POINTER(struct_ir3_compiler)),
  ('type', gl_shader_stage),
  ('inputs_count', ctypes.c_uint32),
  ('inputs_sz', ctypes.c_uint32),
  ('inputs', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('baryfs_count', ctypes.c_uint32),
  ('baryfs_sz', ctypes.c_uint32),
  ('baryfs', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('a0_users_count', ctypes.c_uint32),
  ('a0_users_sz', ctypes.c_uint32),
  ('a0_users', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('a1_users_count', ctypes.c_uint32),
  ('a1_users_sz', ctypes.c_uint32),
  ('a1_users', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('astc_srgb_count', ctypes.c_uint32),
  ('astc_srgb_sz', ctypes.c_uint32),
  ('astc_srgb', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('tg4_count', ctypes.c_uint32),
  ('tg4_sz', ctypes.c_uint32),
  ('tg4', ctypes.POINTER(ctypes.POINTER(struct_ir3_instruction))),
  ('block_list', struct_list_head),
  ('array_list', struct_list_head),
  ('instr_count', ctypes.c_uint32),
]
class struct_ir3_info(Struct): pass
struct_ir3_info._fields_ = [
  ('size', uint32_t),
  ('constant_data_offset', uint32_t),
  ('sizedwords', uint16_t),
  ('instrs_count', uint16_t),
  ('preamble_instrs_count', uint16_t),
  ('nops_count', uint16_t),
  ('mov_count', uint16_t),
  ('cov_count', uint16_t),
  ('stp_count', uint16_t),
  ('ldp_count', uint16_t),
  ('max_reg', int8_t),
  ('max_half_reg', int8_t),
  ('max_const', int16_t),
  ('max_waves', int8_t),
  ('subgroup_size', uint8_t),
  ('double_threadsize', ctypes.c_bool),
  ('multi_dword_ldp_stp', ctypes.c_bool),
  ('early_preamble', ctypes.c_bool),
  ('uses_ray_intersection', ctypes.c_bool),
  ('ss', uint16_t),
  ('sy', uint16_t),
  ('sstall', uint16_t),
  ('systall', uint16_t),
  ('last_baryf', uint16_t),
  ('last_helper', uint16_t),
  ('instrs_per_cat', (uint16_t * 8)),
]
class struct_ir3_shader_variant_input(Struct): pass
struct_ir3_shader_variant_input._fields_ = [
  ('slot', uint8_t),
  ('regid', uint8_t),
  ('compmask', uint8_t),
  ('inloc', uint8_t),
  ('sysval', ctypes.c_bool,1),
  ('bary', ctypes.c_bool,1),
  ('rasterflat', ctypes.c_bool,1),
  ('half', ctypes.c_bool,1),
  ('flat', ctypes.c_bool,1),
]
class struct_ir3_shader_variant_astc_srgb(Struct): pass
struct_ir3_shader_variant_astc_srgb._fields_ = [
  ('base', ctypes.c_uint32),
  ('count', ctypes.c_uint32),
  ('orig_idx', (ctypes.c_uint32 * 16)),
]
class struct_ir3_shader_variant_tg4(Struct): pass
struct_ir3_shader_variant_tg4._fields_ = [
  ('base', ctypes.c_uint32),
  ('count', ctypes.c_uint32),
  ('orig_idx', (ctypes.c_uint32 * 16)),
]
class struct_ir3_shader_variant_0(ctypes.Union): pass
class struct_ir3_shader_variant_0_tess(Struct): pass
enum_gl_tess_spacing = CEnum(ctypes.c_uint32)
TESS_SPACING_UNSPECIFIED = enum_gl_tess_spacing.define('TESS_SPACING_UNSPECIFIED', 0)
TESS_SPACING_EQUAL = enum_gl_tess_spacing.define('TESS_SPACING_EQUAL', 1)
TESS_SPACING_FRACTIONAL_ODD = enum_gl_tess_spacing.define('TESS_SPACING_FRACTIONAL_ODD', 2)
TESS_SPACING_FRACTIONAL_EVEN = enum_gl_tess_spacing.define('TESS_SPACING_FRACTIONAL_EVEN', 3)

struct_ir3_shader_variant_0_tess._fields_ = [
  ('primitive_mode', enum_tess_primitive_mode),
  ('tcs_vertices_out', uint8_t),
  ('spacing', enum_gl_tess_spacing,2),
  ('ccw', ctypes.c_bool,1),
  ('point_mode', ctypes.c_bool,1),
]
class struct_ir3_shader_variant_0_gs(Struct): pass
struct_ir3_shader_variant_0_gs._fields_ = [
  ('output_primitive', uint16_t),
  ('vertices_out', uint16_t),
  ('invocations', uint8_t),
  ('vertices_in', uint8_t,3),
]
class struct_ir3_shader_variant_0_fs(Struct): pass
struct_ir3_shader_variant_0_fs._fields_ = [
  ('early_fragment_tests', ctypes.c_bool,1),
  ('color_is_dual_source', ctypes.c_bool,1),
  ('uses_fbfetch_output', ctypes.c_bool,1),
  ('fbfetch_coherent', ctypes.c_bool,1),
  ('depth_layout', enum_gl_frag_depth_layout),
]
class struct_ir3_shader_variant_0_cs(Struct): pass
struct_ir3_shader_variant_0_cs._fields_ = [
  ('req_local_mem', ctypes.c_uint32),
  ('force_linear_dispatch', ctypes.c_bool),
  ('local_invocation_id', uint32_t),
  ('work_group_id', uint32_t),
]
struct_ir3_shader_variant_0._fields_ = [
  ('tess', struct_ir3_shader_variant_0_tess),
  ('gs', struct_ir3_shader_variant_0_gs),
  ('fs', struct_ir3_shader_variant_0_fs),
  ('cs', struct_ir3_shader_variant_0_cs),
]
struct_ir3_shader_variant._anonymous_ = ['_0']
struct_ir3_shader_variant._fields_ = [
  ('bo', ctypes.POINTER(struct_fd_bo)),
  ('id', uint32_t),
  ('shader_id', uint32_t),
  ('key', struct_ir3_shader_key),
  ('binning_pass', ctypes.c_bool),
  ('binning', ctypes.POINTER(struct_ir3_shader_variant)),
  ('nonbinning', ctypes.POINTER(struct_ir3_shader_variant)),
  ('ir', ctypes.POINTER(struct_ir3)),
  ('next', ctypes.POINTER(struct_ir3_shader_variant)),
  ('type', gl_shader_stage),
  ('compiler', ctypes.POINTER(struct_ir3_compiler)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('constant_data', ctypes.c_void_p),
  ('disasm_info', struct_ir3_disasm_info),
  ('bin', ctypes.POINTER(uint32_t)),
  ('const_state', ctypes.POINTER(struct_ir3_const_state)),
  ('imm_state', struct_ir3_imm_const_state),
  ('info', struct_ir3_info),
  ('sha1_str', (ctypes.c_char * 41)),
  ('shader_options', struct_ir3_shader_options),
  ('constant_data_size', uint32_t),
  ('branchstack', ctypes.c_uint32),
  ('loops', ctypes.c_uint32),
  ('instrlen', ctypes.c_uint32),
  ('constlen', ctypes.c_uint32),
  ('pvtmem_size', ctypes.c_uint32),
  ('pvtmem_per_wave', ctypes.c_bool),
  ('multi_pos_output', ctypes.c_bool),
  ('dual_src_blend', ctypes.c_bool),
  ('early_preamble', ctypes.c_bool),
  ('shared_size', ctypes.c_uint32),
  ('frag_face', ctypes.c_bool),
  ('color0_mrt', ctypes.c_bool),
  ('fragcoord_compmask', uint8_t),
  ('outputs_count', ctypes.c_uint32),
  ('outputs', (struct_ir3_shader_output * 34)),
  ('writes_pos', ctypes.c_bool),
  ('writes_smask', ctypes.c_bool),
  ('writes_psize', ctypes.c_bool),
  ('writes_viewport', ctypes.c_bool),
  ('writes_stencilref', ctypes.c_bool),
  ('writes_shading_rate', ctypes.c_bool),
  ('output_size', uint32_t),
  ('input_size', uint32_t),
  ('output_loc', (ctypes.c_uint32 * 45)),
  ('inputs_count', ctypes.c_uint32),
  ('inputs', (struct_ir3_shader_variant_input * 34)),
  ('reads_primid', ctypes.c_bool),
  ('reads_shading_rate', ctypes.c_bool),
  ('reads_smask', ctypes.c_bool),
  ('total_in', ctypes.c_uint32),
  ('sysval_in', ctypes.c_uint32),
  ('varying_in', ctypes.c_uint32),
  ('image_mapping', struct_ir3_ibo_mapping),
  ('num_samp', ctypes.c_int32),
  ('fb_read', ctypes.c_bool),
  ('has_ssbo', ctypes.c_bool),
  ('bindless_tex', ctypes.c_bool),
  ('bindless_samp', ctypes.c_bool),
  ('bindless_ibo', ctypes.c_bool),
  ('bindless_ubo', ctypes.c_bool),
  ('need_pixlod', ctypes.c_bool),
  ('need_full_quad', ctypes.c_bool),
  ('need_driver_params', ctypes.c_bool),
  ('no_earlyz', ctypes.c_bool),
  ('has_kill', ctypes.c_bool),
  ('per_samp', ctypes.c_bool),
  ('post_depth_coverage', ctypes.c_bool),
  ('empty', ctypes.c_bool),
  ('writes_only_color', ctypes.c_bool),
  ('mergedregs', ctypes.c_bool),
  ('clip_mask', uint8_t),
  ('cull_mask', uint8_t),
  ('astc_srgb', struct_ir3_shader_variant_astc_srgb),
  ('tg4', struct_ir3_shader_variant_tg4),
  ('num_sampler_prefetch', uint32_t),
  ('sampler_prefetch', (struct_ir3_sampler_prefetch * 4)),
  ('prefetch_bary_type', enum_ir3_bary),
  ('prefetch_end_of_quad', ctypes.c_bool),
  ('local_size', (uint16_t * 3)),
  ('local_size_variable', ctypes.c_bool),
  ('has_barrier', ctypes.c_bool),
  ('num_ssbos', ctypes.c_uint32),
  ('num_uavs', ctypes.c_uint32),
  ('_0', struct_ir3_shader_variant_0),
  ('vtxid_base', uint32_t),
  ('stream_output', struct_ir3_stream_output_info),
]
class struct_ir3_shader_0(ctypes.Union): pass
class struct_ir3_shader_0_cs(Struct): pass
struct_ir3_shader_0_cs._fields_ = [
  ('req_local_mem', ctypes.c_uint32),
  ('force_linear_dispatch', ctypes.c_bool),
]
class struct_ir3_shader_0_vs(Struct): pass
struct_ir3_shader_0_vs._fields_ = [
  ('passthrough_tcs_compiled', ctypes.c_uint32),
  ('passthrough_tcs', (ctypes.POINTER(struct_ir3_shader) * 32)),
]
struct_ir3_shader_0._fields_ = [
  ('cs', struct_ir3_shader_0_cs),
  ('vs', struct_ir3_shader_0_vs),
]
class pthread_mutex_t(ctypes.Union): pass
mtx_t = pthread_mutex_t
class struct___pthread_mutex_s(Struct): pass
class struct___pthread_internal_list(Struct): pass
__pthread_list_t = struct___pthread_internal_list
struct___pthread_internal_list._fields_ = [
  ('__prev', ctypes.POINTER(struct___pthread_internal_list)),
  ('__next', ctypes.POINTER(struct___pthread_internal_list)),
]
struct___pthread_mutex_s._fields_ = [
  ('__lock', ctypes.c_int32),
  ('__count', ctypes.c_uint32),
  ('__owner', ctypes.c_int32),
  ('__nusers', ctypes.c_uint32),
  ('__kind', ctypes.c_int32),
  ('__spins', ctypes.c_int16),
  ('__elision', ctypes.c_int16),
  ('__list', struct___pthread_internal_list),
]
pthread_mutex_t._fields_ = [
  ('__data', struct___pthread_mutex_s),
  ('__size', (ctypes.c_char * 40)),
  ('__align', ctypes.c_int64),
]
cache_key = (ctypes.c_ubyte * 20)
struct_ir3_shader._anonymous_ = ['_0']
struct_ir3_shader._fields_ = [
  ('type', gl_shader_stage),
  ('id', uint32_t),
  ('variant_count', uint32_t),
  ('initial_variants_done', ctypes.c_bool),
  ('compiler', ctypes.POINTER(struct_ir3_compiler)),
  ('options', struct_ir3_shader_options),
  ('nir_finalized', ctypes.c_bool),
  ('nir', ctypes.POINTER(struct_nir_shader)),
  ('stream_output', struct_ir3_stream_output_info),
  ('_0', struct_ir3_shader_0),
  ('variants', ctypes.POINTER(struct_ir3_shader_variant)),
  ('variants_lock', mtx_t),
  ('cache_key', cache_key),
  ('key_mask', struct_ir3_shader_key),
]
try: (ir3_const_ensure_imm_size:=dll.ir3_const_ensure_imm_size).restype, ir3_const_ensure_imm_size.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_ir3_shader_variant), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_const_imm_index_to_reg:=dll.ir3_const_imm_index_to_reg).restype, ir3_const_imm_index_to_reg.argtypes = uint16_t, [ctypes.POINTER(struct_ir3_const_state), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_const_find_imm:=dll.ir3_const_find_imm).restype, ir3_const_find_imm.argtypes = uint16_t, [ctypes.POINTER(struct_ir3_shader_variant), uint32_t]
except AttributeError: pass

try: (ir3_const_add_imm:=dll.ir3_const_add_imm).restype, ir3_const_add_imm.argtypes = uint16_t, [ctypes.POINTER(struct_ir3_shader_variant), uint32_t]
except AttributeError: pass

try: (ir3_shader_assemble:=dll.ir3_shader_assemble).restype, ir3_shader_assemble.argtypes = ctypes.c_void_p, [ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_shader_create_variant:=dll.ir3_shader_create_variant).restype, ir3_shader_create_variant.argtypes = ctypes.POINTER(struct_ir3_shader_variant), [ctypes.POINTER(struct_ir3_shader), ctypes.POINTER(struct_ir3_shader_key), ctypes.c_bool]
except AttributeError: pass

try: (ir3_shader_get_variant:=dll.ir3_shader_get_variant).restype, ir3_shader_get_variant.argtypes = ctypes.POINTER(struct_ir3_shader_variant), [ctypes.POINTER(struct_ir3_shader), ctypes.POINTER(struct_ir3_shader_key), ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (ir3_shader_from_nir:=dll.ir3_shader_from_nir).restype, ir3_shader_from_nir.argtypes = ctypes.POINTER(struct_ir3_shader), [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_options), ctypes.POINTER(struct_ir3_stream_output_info)]
except AttributeError: pass

try: (ir3_trim_constlen:=dll.ir3_trim_constlen).restype, ir3_trim_constlen.argtypes = uint32_t, [ctypes.POINTER(ctypes.POINTER(struct_ir3_shader_variant)), ctypes.POINTER(struct_ir3_compiler)]
except AttributeError: pass

try: (ir3_shader_passthrough_tcs:=dll.ir3_shader_passthrough_tcs).restype, ir3_shader_passthrough_tcs.argtypes = ctypes.POINTER(struct_ir3_shader), [ctypes.POINTER(struct_ir3_shader), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_shader_destroy:=dll.ir3_shader_destroy).restype, ir3_shader_destroy.argtypes = None, [ctypes.POINTER(struct_ir3_shader)]
except AttributeError: pass

try: (ir3_shader_disasm:=dll.ir3_shader_disasm).restype, ir3_shader_disasm.argtypes = None, [ctypes.POINTER(struct_ir3_shader_variant), ctypes.POINTER(uint32_t), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (ir3_shader_outputs:=dll.ir3_shader_outputs).restype, ir3_shader_outputs.argtypes = uint64_t, [ctypes.POINTER(struct_ir3_shader)]
except AttributeError: pass

try: (ir3_glsl_type_size:=dll.ir3_glsl_type_size).restype, ir3_glsl_type_size.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (ir3_shader_get_subgroup_size:=dll.ir3_shader_get_subgroup_size).restype, ir3_shader_get_subgroup_size.argtypes = None, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(struct_ir3_shader_options), gl_shader_stage, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

class struct_ir3_shader_linkage(Struct): pass
class struct_ir3_shader_linkage_var(Struct): pass
struct_ir3_shader_linkage_var._fields_ = [
  ('slot', uint8_t),
  ('regid', uint8_t),
  ('compmask', uint8_t),
  ('loc', uint8_t),
]
struct_ir3_shader_linkage._fields_ = [
  ('max_loc', uint8_t),
  ('cnt', uint8_t),
  ('varmask', (uint32_t * 4)),
  ('var', (struct_ir3_shader_linkage_var * 32)),
  ('primid_loc', uint8_t),
  ('viewid_loc', uint8_t),
  ('clip0_loc', uint8_t),
  ('clip1_loc', uint8_t),
]
try: (print_raw:=dll.print_raw).restype, print_raw.argtypes = None, [ctypes.POINTER(FILE), ctypes.POINTER(ctypes.c_uint32), size_t]
except AttributeError: pass

try: (ir3_link_stream_out:=dll.ir3_link_stream_out).restype, ir3_link_stream_out.argtypes = None, [ctypes.POINTER(struct_ir3_shader_linkage), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_apply_trig_workarounds:=dll.ir3_nir_apply_trig_workarounds).restype, ir3_nir_apply_trig_workarounds.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_imul:=dll.ir3_nir_lower_imul).restype, ir3_nir_lower_imul.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_io_offsets:=dll.ir3_nir_lower_io_offsets).restype, ir3_nir_lower_io_offsets.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_load_barycentric_at_sample:=dll.ir3_nir_lower_load_barycentric_at_sample).restype, ir3_nir_lower_load_barycentric_at_sample.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_load_barycentric_at_offset:=dll.ir3_nir_lower_load_barycentric_at_offset).restype, ir3_nir_lower_load_barycentric_at_offset.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_push_consts_to_preamble:=dll.ir3_nir_lower_push_consts_to_preamble).restype, ir3_nir_lower_push_consts_to_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_lower_driver_params_to_ubo:=dll.ir3_nir_lower_driver_params_to_ubo).restype, ir3_nir_lower_driver_params_to_ubo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_move_varying_inputs:=dll.ir3_nir_move_varying_inputs).restype, ir3_nir_move_varying_inputs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_coord_offset:=dll.ir3_nir_coord_offset).restype, ir3_nir_coord_offset.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_def), ctypes.POINTER(gl_system_value)]
except AttributeError: pass

try: (ir3_nir_lower_tex_prefetch:=dll.ir3_nir_lower_tex_prefetch).restype, ir3_nir_lower_tex_prefetch.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(enum_ir3_bary)]
except AttributeError: pass

try: (ir3_nir_lower_layer_id:=dll.ir3_nir_lower_layer_id).restype, ir3_nir_lower_layer_id.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_frag_shading_rate:=dll.ir3_nir_lower_frag_shading_rate).restype, ir3_nir_lower_frag_shading_rate.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_primitive_shading_rate:=dll.ir3_nir_lower_primitive_shading_rate).restype, ir3_nir_lower_primitive_shading_rate.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_to_explicit_output:=dll.ir3_nir_lower_to_explicit_output).restype, ir3_nir_lower_to_explicit_output.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_nir_lower_to_explicit_input:=dll.ir3_nir_lower_to_explicit_input).restype, ir3_nir_lower_to_explicit_input.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_lower_tess_ctrl:=dll.ir3_nir_lower_tess_ctrl).restype, ir3_nir_lower_tess_ctrl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_nir_lower_tess_eval:=dll.ir3_nir_lower_tess_eval).restype, ir3_nir_lower_tess_eval.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_nir_lower_gs:=dll.ir3_nir_lower_gs).restype, ir3_nir_lower_gs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_supports_vectorized_nir_op:=dll.ir3_supports_vectorized_nir_op).restype, ir3_supports_vectorized_nir_op.argtypes = ctypes.c_bool, [nir_op]
except AttributeError: pass

try: (ir3_nir_vectorize_filter:=dll.ir3_nir_vectorize_filter).restype, ir3_nir_vectorize_filter.argtypes = uint8_t, [ctypes.POINTER(nir_instr), ctypes.c_void_p]
except AttributeError: pass

try: (ir3_nir_lower_64b_intrinsics:=dll.ir3_nir_lower_64b_intrinsics).restype, ir3_nir_lower_64b_intrinsics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_64b_undef:=dll.ir3_nir_lower_64b_undef).restype, ir3_nir_lower_64b_undef.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_64b_global:=dll.ir3_nir_lower_64b_global).restype, ir3_nir_lower_64b_global.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_64b_regs:=dll.ir3_nir_lower_64b_regs).restype, ir3_nir_lower_64b_regs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_mem_access_size_align:=dll.ir3_mem_access_size_align).restype, ir3_mem_access_size_align.argtypes = nir_mem_access_size_align, [nir_intrinsic_op, uint8_t, uint8_t, uint32_t, uint32_t, ctypes.c_bool, enum_gl_access_qualifier, ctypes.c_void_p]
except AttributeError: pass

try: (ir3_nir_opt_branch_and_or_not:=dll.ir3_nir_opt_branch_and_or_not).restype, ir3_nir_opt_branch_and_or_not.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_opt_triops_bitwise:=dll.ir3_nir_opt_triops_bitwise).restype, ir3_nir_opt_triops_bitwise.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_optimize_loop:=dll.ir3_optimize_loop).restype, ir3_optimize_loop.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(struct_ir3_shader_nir_options), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_lower_io_vars_to_temporaries:=dll.ir3_nir_lower_io_vars_to_temporaries).restype, ir3_nir_lower_io_vars_to_temporaries.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_finalize_nir:=dll.ir3_finalize_nir).restype, ir3_finalize_nir.argtypes = None, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(struct_ir3_shader_nir_options), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_post_finalize:=dll.ir3_nir_post_finalize).restype, ir3_nir_post_finalize.argtypes = None, [ctypes.POINTER(struct_ir3_shader)]
except AttributeError: pass

try: (ir3_nir_lower_variant:=dll.ir3_nir_lower_variant).restype, ir3_nir_lower_variant.argtypes = None, [ctypes.POINTER(struct_ir3_shader_variant), ctypes.POINTER(struct_ir3_shader_nir_options), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_setup_const_state:=dll.ir3_setup_const_state).restype, ir3_setup_const_state.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant), ctypes.POINTER(struct_ir3_const_state)]
except AttributeError: pass

try: (ir3_const_state_get_free_space:=dll.ir3_const_state_get_free_space).restype, ir3_const_state_get_free_space.argtypes = uint32_t, [ctypes.POINTER(struct_ir3_shader_variant), ctypes.POINTER(struct_ir3_const_state), uint32_t]
except AttributeError: pass

try: (ir3_const_alloc:=dll.ir3_const_alloc).restype, ir3_const_alloc.argtypes = None, [ctypes.POINTER(struct_ir3_const_allocations), enum_ir3_const_alloc_type, uint32_t, uint32_t]
except AttributeError: pass

try: (ir3_const_reserve_space:=dll.ir3_const_reserve_space).restype, ir3_const_reserve_space.argtypes = None, [ctypes.POINTER(struct_ir3_const_allocations), enum_ir3_const_alloc_type, uint32_t, uint32_t]
except AttributeError: pass

try: (ir3_const_free_reserved_space:=dll.ir3_const_free_reserved_space).restype, ir3_const_free_reserved_space.argtypes = None, [ctypes.POINTER(struct_ir3_const_allocations), enum_ir3_const_alloc_type]
except AttributeError: pass

try: (ir3_const_alloc_all_reserved_space:=dll.ir3_const_alloc_all_reserved_space).restype, ir3_const_alloc_all_reserved_space.argtypes = None, [ctypes.POINTER(struct_ir3_const_allocations)]
except AttributeError: pass

try: (ir3_nir_scan_driver_consts:=dll.ir3_nir_scan_driver_consts).restype, ir3_nir_scan_driver_consts.argtypes = uint32_t, [ctypes.POINTER(struct_ir3_compiler), ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_const_image_dims)]
except AttributeError: pass

try: (ir3_alloc_driver_params:=dll.ir3_alloc_driver_params).restype, ir3_alloc_driver_params.argtypes = None, [ctypes.POINTER(struct_ir3_const_allocations), ctypes.POINTER(uint32_t), ctypes.POINTER(struct_ir3_compiler), enum_pipe_shader_type]
except AttributeError: pass

try: (ir3_nir_lower_load_constant:=dll.ir3_nir_lower_load_constant).restype, ir3_nir_lower_load_constant.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_analyze_ubo_ranges:=dll.ir3_nir_analyze_ubo_ranges).restype, ir3_nir_analyze_ubo_ranges.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_lower_ubo_loads:=dll.ir3_nir_lower_ubo_loads).restype, ir3_nir_lower_ubo_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_lower_const_global_loads:=dll.ir3_nir_lower_const_global_loads).restype, ir3_nir_lower_const_global_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_fixup_load_const_ir3:=dll.ir3_nir_fixup_load_const_ir3).restype, ir3_nir_fixup_load_const_ir3.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (ir3_nir_opt_preamble:=dll.ir3_nir_opt_preamble).restype, ir3_nir_opt_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_opt_prefetch_descriptors:=dll.ir3_nir_opt_prefetch_descriptors).restype, ir3_nir_opt_prefetch_descriptors.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_lower_preamble:=dll.ir3_nir_lower_preamble).restype, ir3_nir_lower_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_nir_try_propagate_bit_shift:=dll.ir3_nir_try_propagate_bit_shift).restype, ir3_nir_try_propagate_bit_shift.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), int32_t]
except AttributeError: pass

try: (ir3_nir_lower_subgroups_filter:=dll.ir3_nir_lower_subgroups_filter).restype, ir3_nir_lower_subgroups_filter.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), ctypes.c_void_p]
except AttributeError: pass

try: (ir3_nir_lower_shuffle:=dll.ir3_nir_lower_shuffle).restype, ir3_nir_lower_shuffle.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader)]
except AttributeError: pass

try: (ir3_nir_opt_subgroups:=dll.ir3_nir_opt_subgroups).restype, ir3_nir_opt_subgroups.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_get_shared_driver_ubo:=dll.ir3_get_shared_driver_ubo).restype, ir3_get_shared_driver_ubo.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(struct_ir3_driver_ubo)]
except AttributeError: pass

try: (ir3_get_driver_ubo:=dll.ir3_get_driver_ubo).restype, ir3_get_driver_ubo.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(struct_ir3_driver_ubo)]
except AttributeError: pass

try: (ir3_get_driver_consts_ubo:=dll.ir3_get_driver_consts_ubo).restype, ir3_get_driver_consts_ubo.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(struct_ir3_shader_variant)]
except AttributeError: pass

try: (ir3_update_driver_ubo:=dll.ir3_update_driver_ubo).restype, ir3_update_driver_ubo.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_ir3_driver_ubo), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ir3_load_shared_driver_ubo:=dll.ir3_load_shared_driver_ubo).restype, ir3_load_shared_driver_ubo.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.POINTER(struct_ir3_driver_ubo), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_load_driver_ubo:=dll.ir3_load_driver_ubo).restype, ir3_load_driver_ubo.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.POINTER(struct_ir3_driver_ubo), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_load_driver_ubo_indirect:=dll.ir3_load_driver_ubo_indirect).restype, ir3_load_driver_ubo_indirect.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.POINTER(struct_ir3_driver_ubo), ctypes.c_uint32, ctypes.POINTER(nir_def), ctypes.c_uint32]
except AttributeError: pass

try: (ir3_def_is_rematerializable_for_preamble:=dll.ir3_def_is_rematerializable_for_preamble).restype, ir3_def_is_rematerializable_for_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def), ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

try: (ir3_rematerialize_def_for_preamble:=dll.ir3_rematerialize_def_for_preamble).restype, ir3_rematerialize_def_for_preamble.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(struct_set), ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

class struct_driver_param_info(Struct): pass
struct_driver_param_info._fields_ = [
  ('offset', uint32_t),
  ('extra_size', uint32_t),
]
try: (ir3_get_driver_param_info:=dll.ir3_get_driver_param_info).restype, ir3_get_driver_param_info.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(struct_driver_param_info)]
except AttributeError: pass

try: (ir3_nir_max_imm_offset:=dll.ir3_nir_max_imm_offset).restype, ir3_nir_max_imm_offset.argtypes = uint32_t, [ctypes.POINTER(nir_intrinsic_instr), ctypes.c_void_p]
except AttributeError: pass

try: (ir3_nir_intrinsic_barycentric_sysval:=dll.ir3_nir_intrinsic_barycentric_sysval).restype, ir3_nir_intrinsic_barycentric_sysval.argtypes = gl_system_value, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (glsl_type_singleton_init_or_ref:=dll.glsl_type_singleton_init_or_ref).restype, glsl_type_singleton_init_or_ref.argtypes = None, []
except AttributeError: pass

try: (glsl_type_singleton_decref:=dll.glsl_type_singleton_decref).restype, glsl_type_singleton_decref.argtypes = None, []
except AttributeError: pass

try: (encode_type_to_blob:=dll.encode_type_to_blob).restype, encode_type_to_blob.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (decode_type_from_blob:=dll.decode_type_from_blob).restype, decode_type_from_blob.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (glsl_apply_signedness_to_base_type:=dll.glsl_apply_signedness_to_base_type).restype, glsl_apply_signedness_to_base_type.argtypes = enum_glsl_base_type, [enum_glsl_base_type, ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_sampler_dim_coordinate_components:=dll.glsl_get_sampler_dim_coordinate_components).restype, glsl_get_sampler_dim_coordinate_components.argtypes = ctypes.c_int32, [enum_glsl_sampler_dim]
except AttributeError: pass

enum_glsl_matrix_layout = CEnum(ctypes.c_uint32)
GLSL_MATRIX_LAYOUT_INHERITED = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_INHERITED', 0)
GLSL_MATRIX_LAYOUT_COLUMN_MAJOR = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_COLUMN_MAJOR', 1)
GLSL_MATRIX_LAYOUT_ROW_MAJOR = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_ROW_MAJOR', 2)

_anonenum6 = CEnum(ctypes.c_uint32)
GLSL_PRECISION_NONE = _anonenum6.define('GLSL_PRECISION_NONE', 0)
GLSL_PRECISION_HIGH = _anonenum6.define('GLSL_PRECISION_HIGH', 1)
GLSL_PRECISION_MEDIUM = _anonenum6.define('GLSL_PRECISION_MEDIUM', 2)
GLSL_PRECISION_LOW = _anonenum6.define('GLSL_PRECISION_LOW', 3)

enum_glsl_cmat_use = CEnum(ctypes.c_uint32)
GLSL_CMAT_USE_NONE = enum_glsl_cmat_use.define('GLSL_CMAT_USE_NONE', 0)
GLSL_CMAT_USE_A = enum_glsl_cmat_use.define('GLSL_CMAT_USE_A', 1)
GLSL_CMAT_USE_B = enum_glsl_cmat_use.define('GLSL_CMAT_USE_B', 2)
GLSL_CMAT_USE_ACCUMULATOR = enum_glsl_cmat_use.define('GLSL_CMAT_USE_ACCUMULATOR', 3)

try: (glsl_get_type_name:=dll.glsl_get_type_name).restype, glsl_get_type_name.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_vector:=dll.glsl_type_is_vector).restype, glsl_type_is_vector.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_scalar:=dll.glsl_type_is_scalar).restype, glsl_type_is_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_vector_or_scalar:=dll.glsl_type_is_vector_or_scalar).restype, glsl_type_is_vector_or_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_matrix:=dll.glsl_type_is_matrix).restype, glsl_type_is_matrix.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_array_or_matrix:=dll.glsl_type_is_array_or_matrix).restype, glsl_type_is_array_or_matrix.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_dual_slot:=dll.glsl_type_is_dual_slot).restype, glsl_type_is_dual_slot.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_leaf:=dll.glsl_type_is_leaf).restype, glsl_type_is_leaf.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_bare_type:=dll.glsl_get_bare_type).restype, glsl_get_bare_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_scalar_type:=dll.glsl_get_scalar_type).restype, glsl_get_scalar_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_base_glsl_type:=dll.glsl_get_base_glsl_type).restype, glsl_get_base_glsl_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_length:=dll.glsl_get_length).restype, glsl_get_length.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_wrap_in_arrays:=dll.glsl_type_wrap_in_arrays).restype, glsl_type_wrap_in_arrays.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_aoa_size:=dll.glsl_get_aoa_size).restype, glsl_get_aoa_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_array_element:=dll.glsl_get_array_element).restype, glsl_get_array_element.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_without_array:=dll.glsl_without_array).restype, glsl_without_array.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_without_array_or_matrix:=dll.glsl_without_array_or_matrix).restype, glsl_without_array_or_matrix.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_wrap_in_arrays:=dll.glsl_type_wrap_in_arrays).restype, glsl_type_wrap_in_arrays.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cmat_element:=dll.glsl_get_cmat_element).restype, glsl_get_cmat_element.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cmat_description:=dll.glsl_get_cmat_description).restype, glsl_get_cmat_description.argtypes = ctypes.POINTER(struct_glsl_cmat_description), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_atomic_size:=dll.glsl_atomic_size).restype, glsl_atomic_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_32bit:=dll.glsl_type_contains_32bit).restype, glsl_type_contains_32bit.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_64bit:=dll.glsl_type_contains_64bit).restype, glsl_type_contains_64bit.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_image:=dll.glsl_type_contains_image).restype, glsl_type_contains_image.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_atomic:=dll.glsl_contains_atomic).restype, glsl_contains_atomic.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_double:=dll.glsl_contains_double).restype, glsl_contains_double.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_integer:=dll.glsl_contains_integer).restype, glsl_contains_integer.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_opaque:=dll.glsl_contains_opaque).restype, glsl_contains_opaque.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_sampler:=dll.glsl_contains_sampler).restype, glsl_contains_sampler.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_array:=dll.glsl_contains_array).restype, glsl_contains_array.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_subroutine:=dll.glsl_contains_subroutine).restype, glsl_contains_subroutine.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_sampler_coordinate_components:=dll.glsl_get_sampler_coordinate_components).restype, glsl_get_sampler_coordinate_components.argtypes = ctypes.c_int32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_compare_no_precision:=dll.glsl_type_compare_no_precision).restype, glsl_type_compare_no_precision.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_record_compare:=dll.glsl_record_compare).restype, glsl_record_compare.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_struct_field:=dll.glsl_get_struct_field).restype, glsl_get_struct_field.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_struct_field_data:=dll.glsl_get_struct_field_data).restype, glsl_get_struct_field_data.argtypes = ctypes.POINTER(glsl_struct_field), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_struct_location_offset:=dll.glsl_get_struct_location_offset).restype, glsl_get_struct_location_offset.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_field_index:=dll.glsl_get_field_index).restype, glsl_get_field_index.argtypes = ctypes.c_int32, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_get_field_type:=dll.glsl_get_field_type).restype, glsl_get_field_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_vec_type:=dll.glsl_vec_type).restype, glsl_vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_f16vec_type:=dll.glsl_f16vec_type).restype, glsl_f16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_bf16vec_type:=dll.glsl_bf16vec_type).restype, glsl_bf16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_e4m3fnvec_type:=dll.glsl_e4m3fnvec_type).restype, glsl_e4m3fnvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_e5m2vec_type:=dll.glsl_e5m2vec_type).restype, glsl_e5m2vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_dvec_type:=dll.glsl_dvec_type).restype, glsl_dvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_ivec_type:=dll.glsl_ivec_type).restype, glsl_ivec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_uvec_type:=dll.glsl_uvec_type).restype, glsl_uvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_bvec_type:=dll.glsl_bvec_type).restype, glsl_bvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i64vec_type:=dll.glsl_i64vec_type).restype, glsl_i64vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u64vec_type:=dll.glsl_u64vec_type).restype, glsl_u64vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i16vec_type:=dll.glsl_i16vec_type).restype, glsl_i16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u16vec_type:=dll.glsl_u16vec_type).restype, glsl_u16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i8vec_type:=dll.glsl_i8vec_type).restype, glsl_i8vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u8vec_type:=dll.glsl_u8vec_type).restype, glsl_u8vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_simple_explicit_type:=dll.glsl_simple_explicit_type).restype, glsl_simple_explicit_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
except AttributeError: pass

try: (glsl_sampler_type:=dll.glsl_sampler_type).restype, glsl_sampler_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_bare_sampler_type:=dll.glsl_bare_sampler_type).restype, glsl_bare_sampler_type.argtypes = ctypes.POINTER(glsl_type), []
except AttributeError: pass

try: (glsl_bare_shadow_sampler_type:=dll.glsl_bare_shadow_sampler_type).restype, glsl_bare_shadow_sampler_type.argtypes = ctypes.POINTER(glsl_type), []
except AttributeError: pass

try: (glsl_texture_type:=dll.glsl_texture_type).restype, glsl_texture_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_image_type:=dll.glsl_image_type).restype, glsl_image_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_array_type:=dll.glsl_array_type).restype, glsl_array_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (glsl_cmat_type:=dll.glsl_cmat_type).restype, glsl_cmat_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(struct_glsl_cmat_description)]
except AttributeError: pass

try: (glsl_struct_type_with_explicit_alignment:=dll.glsl_struct_type_with_explicit_alignment).restype, glsl_struct_type_with_explicit_alignment.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool, ctypes.c_uint32]
except AttributeError: pass

enum_glsl_interface_packing = CEnum(ctypes.c_uint32)
GLSL_INTERFACE_PACKING_STD140 = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_STD140', 0)
GLSL_INTERFACE_PACKING_SHARED = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_SHARED', 1)
GLSL_INTERFACE_PACKING_PACKED = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_PACKED', 2)
GLSL_INTERFACE_PACKING_STD430 = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_STD430', 3)

try: (glsl_interface_type:=dll.glsl_interface_type).restype, glsl_interface_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_struct_field), ctypes.c_uint32, enum_glsl_interface_packing, ctypes.c_bool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_subroutine_type:=dll.glsl_subroutine_type).restype, glsl_subroutine_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_get_row_type:=dll.glsl_get_row_type).restype, glsl_get_row_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_column_type:=dll.glsl_get_column_type).restype, glsl_get_column_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_explicit_type_for_size_align:=dll.glsl_get_explicit_type_for_size_align).restype, glsl_get_explicit_type_for_size_align.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_type_replace_vec3_with_vec4:=dll.glsl_type_replace_vec3_with_vec4).restype, glsl_type_replace_vec3_with_vec4.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_float16_type:=dll.glsl_float16_type).restype, glsl_float16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_int16_type:=dll.glsl_int16_type).restype, glsl_int16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_uint16_type:=dll.glsl_uint16_type).restype, glsl_uint16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_to_16bit:=dll.glsl_type_to_16bit).restype, glsl_type_to_16bit.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_replace_vector_type:=dll.glsl_replace_vector_type).restype, glsl_replace_vector_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_channel_type:=dll.glsl_channel_type).restype, glsl_channel_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_mul_type:=dll.glsl_get_mul_type).restype, glsl_get_mul_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_sampler_count:=dll.glsl_type_get_sampler_count).restype, glsl_type_get_sampler_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_texture_count:=dll.glsl_type_get_texture_count).restype, glsl_type_get_texture_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_image_count:=dll.glsl_type_get_image_count).restype, glsl_type_get_image_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_count_vec4_slots:=dll.glsl_count_vec4_slots).restype, glsl_count_vec4_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (glsl_count_dword_slots:=dll.glsl_count_dword_slots).restype, glsl_count_dword_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_component_slots:=dll.glsl_get_component_slots).restype, glsl_get_component_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_component_slots_aligned:=dll.glsl_get_component_slots_aligned).restype, glsl_get_component_slots_aligned.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_varying_count:=dll.glsl_varying_count).restype, glsl_varying_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_uniform_locations:=dll.glsl_type_uniform_locations).restype, glsl_type_uniform_locations.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_size:=dll.glsl_get_cl_size).restype, glsl_get_cl_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_alignment:=dll.glsl_get_cl_alignment).restype, glsl_get_cl_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_type_size_align:=dll.glsl_get_cl_type_size_align).restype, glsl_get_cl_type_size_align.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_internal_ifc_packing:=dll.glsl_get_internal_ifc_packing).restype, glsl_get_internal_ifc_packing.argtypes = enum_glsl_interface_packing, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std140_base_alignment:=dll.glsl_get_std140_base_alignment).restype, glsl_get_std140_base_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std140_size:=dll.glsl_get_std140_size).restype, glsl_get_std140_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_array_stride:=dll.glsl_get_std430_array_stride).restype, glsl_get_std430_array_stride.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_base_alignment:=dll.glsl_get_std430_base_alignment).restype, glsl_get_std430_base_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_size:=dll.glsl_get_std430_size).restype, glsl_get_std430_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_size:=dll.glsl_get_explicit_size).restype, glsl_get_explicit_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_std140_type:=dll.glsl_get_explicit_std140_type).restype, glsl_get_explicit_std140_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_std430_type:=dll.glsl_get_explicit_std430_type).restype, glsl_get_explicit_std430_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_size_align_handle_array_and_structs:=dll.glsl_size_align_handle_array_and_structs).restype, glsl_size_align_handle_array_and_structs.argtypes = None, [ctypes.POINTER(glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_natural_size_align_bytes:=dll.glsl_get_natural_size_align_bytes).restype, glsl_get_natural_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_word_size_align_bytes:=dll.glsl_get_word_size_align_bytes).restype, glsl_get_word_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_vec4_size_align_bytes:=dll.glsl_get_vec4_size_align_bytes).restype, glsl_get_vec4_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (blob_init:=dll.blob_init).restype, blob_init.argtypes = None, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_init_fixed:=dll.blob_init_fixed).restype, blob_init_fixed.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_finish_get_buffer:=dll.blob_finish_get_buffer).restype, blob_finish_get_buffer.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t)]
except AttributeError: pass

try: (blob_align:=dll.blob_align).restype, blob_align.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t]
except AttributeError: pass

try: (blob_write_bytes:=dll.blob_write_bytes).restype, blob_write_bytes.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), ctypes.c_void_p, size_t]
except AttributeError: pass

intptr_t = ctypes.c_int64
try: (blob_reserve_bytes:=dll.blob_reserve_bytes).restype, blob_reserve_bytes.argtypes = intptr_t, [ctypes.POINTER(struct_blob), size_t]
except AttributeError: pass

try: (blob_reserve_uint32:=dll.blob_reserve_uint32).restype, blob_reserve_uint32.argtypes = intptr_t, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_reserve_intptr:=dll.blob_reserve_intptr).restype, blob_reserve_intptr.argtypes = intptr_t, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_overwrite_bytes:=dll.blob_overwrite_bytes).restype, blob_overwrite_bytes.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_write_uint8:=dll.blob_write_uint8).restype, blob_write_uint8.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint8_t]
except AttributeError: pass

try: (blob_overwrite_uint8:=dll.blob_overwrite_uint8).restype, blob_overwrite_uint8.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, uint8_t]
except AttributeError: pass

try: (blob_write_uint16:=dll.blob_write_uint16).restype, blob_write_uint16.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint16_t]
except AttributeError: pass

try: (blob_write_uint32:=dll.blob_write_uint32).restype, blob_write_uint32.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint32_t]
except AttributeError: pass

try: (blob_overwrite_uint32:=dll.blob_overwrite_uint32).restype, blob_overwrite_uint32.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, uint32_t]
except AttributeError: pass

try: (blob_write_uint64:=dll.blob_write_uint64).restype, blob_write_uint64.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint64_t]
except AttributeError: pass

try: (blob_write_intptr:=dll.blob_write_intptr).restype, blob_write_intptr.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), intptr_t]
except AttributeError: pass

try: (blob_overwrite_intptr:=dll.blob_overwrite_intptr).restype, blob_overwrite_intptr.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, intptr_t]
except AttributeError: pass

try: (blob_write_string:=dll.blob_write_string).restype, blob_write_string.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (blob_reader_init:=dll.blob_reader_init).restype, blob_reader_init.argtypes = None, [ctypes.POINTER(struct_blob_reader), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_reader_align:=dll.blob_reader_align).restype, blob_reader_align.argtypes = None, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_read_bytes:=dll.blob_read_bytes).restype, blob_read_bytes.argtypes = ctypes.c_void_p, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_copy_bytes:=dll.blob_copy_bytes).restype, blob_copy_bytes.argtypes = None, [ctypes.POINTER(struct_blob_reader), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_skip_bytes:=dll.blob_skip_bytes).restype, blob_skip_bytes.argtypes = None, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_read_uint8:=dll.blob_read_uint8).restype, blob_read_uint8.argtypes = uint8_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint16:=dll.blob_read_uint16).restype, blob_read_uint16.argtypes = uint16_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint32:=dll.blob_read_uint32).restype, blob_read_uint32.argtypes = uint32_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint64:=dll.blob_read_uint64).restype, blob_read_uint64.argtypes = uint64_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_intptr:=dll.blob_read_intptr).restype, blob_read_intptr.argtypes = intptr_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_string:=dll.blob_read_string).restype, blob_read_string.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (ralloc_context:=dll.ralloc_context).restype, ralloc_context.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_size:=dll.ralloc_size).restype, ralloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (rzalloc_size:=dll.rzalloc_size).restype, rzalloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (reralloc_size:=dll.reralloc_size).restype, reralloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (rerzalloc_size:=dll.rerzalloc_size).restype, rerzalloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t]
except AttributeError: pass

try: (ralloc_array_size:=dll.ralloc_array_size).restype, ralloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (rzalloc_array_size:=dll.rzalloc_array_size).restype, rzalloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (reralloc_array_size:=dll.reralloc_array_size).restype, reralloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (rerzalloc_array_size:=dll.rerzalloc_array_size).restype, rerzalloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (ralloc_free:=dll.ralloc_free).restype, ralloc_free.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_steal:=dll.ralloc_steal).restype, ralloc_steal.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_adopt:=dll.ralloc_adopt).restype, ralloc_adopt.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_parent:=dll.ralloc_parent).restype, ralloc_parent.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_set_destructor:=dll.ralloc_set_destructor).restype, ralloc_set_destructor.argtypes = None, [ctypes.c_void_p, ctypes.CFUNCTYPE(None, ctypes.c_void_p)]
except AttributeError: pass

try: (ralloc_memdup:=dll.ralloc_memdup).restype, ralloc_memdup.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (ralloc_strdup:=dll.ralloc_strdup).restype, ralloc_strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_strndup:=dll.ralloc_strndup).restype, ralloc_strndup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (ralloc_strcat:=dll.ralloc_strcat).restype, ralloc_strcat.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_strncat:=dll.ralloc_strncat).restype, ralloc_strncat.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (ralloc_str_append:=dll.ralloc_str_append).restype, ralloc_str_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t, size_t]
except AttributeError: pass

try: (ralloc_asprintf:=dll.ralloc_asprintf).restype, ralloc_asprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct___va_list_tag(Struct): pass
struct___va_list_tag._fields_ = [
  ('gp_offset', ctypes.c_uint32),
  ('fp_offset', ctypes.c_uint32),
  ('overflow_arg_area', ctypes.c_void_p),
  ('reg_save_area', ctypes.c_void_p),
]
va_list = (struct___va_list_tag * 1)
try: (ralloc_vasprintf:=dll.ralloc_vasprintf).restype, ralloc_vasprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_asprintf_rewrite_tail:=dll.ralloc_asprintf_rewrite_tail).restype, ralloc_asprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_vasprintf_rewrite_tail:=dll.ralloc_vasprintf_rewrite_tail).restype, ralloc_vasprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_asprintf_append:=dll.ralloc_asprintf_append).restype, ralloc_asprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_vasprintf_append:=dll.ralloc_vasprintf_append).restype, ralloc_vasprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_total_size:=dll.ralloc_total_size).restype, ralloc_total_size.argtypes = size_t, [ctypes.c_void_p]
except AttributeError: pass

try: (gc_context:=dll.gc_context).restype, gc_context.argtypes = ctypes.POINTER(gc_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (gc_alloc_size:=dll.gc_alloc_size).restype, gc_alloc_size.argtypes = ctypes.c_void_p, [ctypes.POINTER(gc_ctx), size_t, size_t]
except AttributeError: pass

try: (gc_zalloc_size:=dll.gc_zalloc_size).restype, gc_zalloc_size.argtypes = ctypes.c_void_p, [ctypes.POINTER(gc_ctx), size_t, size_t]
except AttributeError: pass

try: (gc_free:=dll.gc_free).restype, gc_free.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (gc_get_context:=dll.gc_get_context).restype, gc_get_context.argtypes = ctypes.POINTER(gc_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (gc_sweep_start:=dll.gc_sweep_start).restype, gc_sweep_start.argtypes = None, [ctypes.POINTER(gc_ctx)]
except AttributeError: pass

try: (gc_mark_live:=dll.gc_mark_live).restype, gc_mark_live.argtypes = None, [ctypes.POINTER(gc_ctx), ctypes.c_void_p]
except AttributeError: pass

try: (gc_sweep_end:=dll.gc_sweep_end).restype, gc_sweep_end.argtypes = None, [ctypes.POINTER(gc_ctx)]
except AttributeError: pass

class struct_linear_ctx(Struct): pass
linear_ctx = struct_linear_ctx
try: (linear_alloc_child:=dll.linear_alloc_child).restype, linear_alloc_child.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), ctypes.c_uint32]
except AttributeError: pass

class linear_opts(Struct): pass
linear_opts._fields_ = [
  ('min_buffer_size', ctypes.c_uint32),
]
try: (linear_context:=dll.linear_context).restype, linear_context.argtypes = ctypes.POINTER(linear_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (linear_context_with_opts:=dll.linear_context_with_opts).restype, linear_context_with_opts.argtypes = ctypes.POINTER(linear_ctx), [ctypes.c_void_p, ctypes.POINTER(linear_opts)]
except AttributeError: pass

try: (linear_zalloc_child:=dll.linear_zalloc_child).restype, linear_zalloc_child.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), ctypes.c_uint32]
except AttributeError: pass

try: (linear_free_context:=dll.linear_free_context).restype, linear_free_context.argtypes = None, [ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (ralloc_steal_linear_context:=dll.ralloc_steal_linear_context).restype, ralloc_steal_linear_context.argtypes = None, [ctypes.c_void_p, ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (ralloc_parent_of_linear_context:=dll.ralloc_parent_of_linear_context).restype, ralloc_parent_of_linear_context.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (linear_alloc_child_array:=dll.linear_alloc_child_array).restype, linear_alloc_child_array.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (linear_zalloc_child_array:=dll.linear_zalloc_child_array).restype, linear_zalloc_child_array.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (linear_strdup:=dll.linear_strdup).restype, linear_strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_asprintf:=dll.linear_asprintf).restype, linear_asprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf:=dll.linear_vasprintf).restype, linear_vasprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_asprintf_append:=dll.linear_asprintf_append).restype, linear_asprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf_append:=dll.linear_vasprintf_append).restype, linear_vasprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_asprintf_rewrite_tail:=dll.linear_asprintf_rewrite_tail).restype, linear_asprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf_rewrite_tail:=dll.linear_vasprintf_rewrite_tail).restype, linear_vasprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_strcat:=dll.linear_strcat).restype, linear_strcat.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

_anonenum7 = CEnum(ctypes.c_uint32)
RALLOC_PRINT_INFO_SUMMARY_ONLY = _anonenum7.define('RALLOC_PRINT_INFO_SUMMARY_ONLY', 1)

try: (ralloc_print_info:=dll.ralloc_print_info).restype, ralloc_print_info.argtypes = None, [ctypes.POINTER(FILE), ctypes.c_void_p, ctypes.c_uint32]
except AttributeError: pass

class struct_isa_decode_options(Struct): pass
class struct_isa_decode_value(Struct): pass
struct_isa_decode_value._fields_ = [
  ('str', ctypes.POINTER(ctypes.c_char)),
  ('num', uint64_t),
]
class struct_isa_print_state(Struct): pass
struct_isa_print_state._fields_ = [
  ('out', ctypes.POINTER(FILE)),
  ('line_column', ctypes.c_uint32),
]
class struct_isa_entrypoint(Struct): pass
struct_isa_entrypoint._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('offset', uint32_t),
]
struct_isa_decode_options._fields_ = [
  ('gpu_id', uint32_t),
  ('show_errors', ctypes.c_bool),
  ('max_errors', ctypes.c_uint32),
  ('branch_labels', ctypes.c_bool),
  ('stop', ctypes.c_bool),
  ('cbdata', ctypes.c_void_p),
  ('field_cb', ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_isa_decode_value))),
  ('field_print_cb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_isa_print_state), ctypes.POINTER(ctypes.c_char), uint64_t)),
  ('pre_instr_cb', ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)),
  ('post_instr_cb', ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)),
  ('no_match_cb', ctypes.CFUNCTYPE(None, ctypes.POINTER(FILE), ctypes.POINTER(ctypes.c_uint32), size_t)),
  ('entrypoint_count', ctypes.c_uint32),
  ('entrypoints', ctypes.POINTER(struct_isa_entrypoint)),
]
try: (ir3_isa_disasm:=dll.ir3_isa_disasm).restype, ir3_isa_disasm.argtypes = None, [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(FILE), ctypes.POINTER(struct_isa_decode_options)]
except AttributeError: pass

try: (ir3_isa_decode:=dll.ir3_isa_decode).restype, ir3_isa_decode.argtypes = ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(struct_isa_decode_options)]
except AttributeError: pass

class struct_decode_scope(Struct): pass
try: (ir3_isa_get_gpu_id:=dll.ir3_isa_get_gpu_id).restype, ir3_isa_get_gpu_id.argtypes = uint32_t, [ctypes.POINTER(struct_decode_scope)]
except AttributeError: pass

try: glsl_type_builtin_error = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_error')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_void = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_void')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bool = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bool')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bvec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bvec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_int = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_int')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_ivec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_ivec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uint = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uint')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uvec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uvec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_float = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_float')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_float16_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_float16_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_double = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_double')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dvec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dvec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_int64_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_int64_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uint64_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uint64_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_int16_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_int16_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i16vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i16vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uint16_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uint16_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u16vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u16vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_int8_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_int8_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i8vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i8vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uint8_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uint8_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u8vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u8vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bfloat16_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bfloat16_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_bf16vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_bf16vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fn_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fn_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e4m3fnvec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e4m3fnvec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2_t = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2_t')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec5 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec5')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec8 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec8')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_e5m2vec16 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_e5m2vec16')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat2x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat2x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat2x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat2x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat3x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat3x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat3x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat3x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat4x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat4x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_mat4x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_mat4x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat2x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat2x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat2x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat2x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat3x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat3x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat3x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat3x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat4x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat4x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_f16mat4x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_f16mat4x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat2x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat2x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat2x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat2x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat3x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat3x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat3x4 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat3x4')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat4x2 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat4x2')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_dmat4x3 = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_dmat4x3')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_atomic_uint = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_atomic_uint')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isamplerCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isamplerCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isamplerCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isamplerCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isamplerBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isamplerBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isampler2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isampler2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usamplerCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usamplerCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usamplerCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usamplerCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usamplerBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usamplerBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usampler2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usampler2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler1DShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler1DShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerCubeShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerCubeShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler1DArrayShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler1DArrayShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DArrayShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DArrayShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerCubeArrayShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerCubeArrayShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_sampler2DRectShadow = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_sampler2DRectShadow')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_samplerExternalOES = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_samplerExternalOES')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_texture2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_texture2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itextureCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itextureCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itextureCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itextureCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itextureBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itextureBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itexture2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itexture2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utextureCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utextureCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utextureCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utextureCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utextureBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utextureBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utexture2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utexture2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureExternalOES = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureExternalOES')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtexture2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtexture2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vtextureBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vtextureBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_imageCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_imageCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_imageBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_imageBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_imageCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_imageCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_image2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_image2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimageCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimageCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimageBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimageBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimageCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimageCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_iimage2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_iimage2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimageCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimageCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimageBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimageBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimageCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimageCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_uimage2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_uimage2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64imageCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64imageCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64imageBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64imageBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64imageCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64imageCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_i64image2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_i64image2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image2DRect = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image2DRect')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64imageCube = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64imageCube')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64imageBuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64imageBuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64imageCubeArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64imageCubeArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_u64image2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_u64image2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vbuffer = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vbuffer')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage1D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage1D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage2D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage2D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage3D = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage3D')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage2DMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage2DMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage2DMSArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage2DMSArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage1DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage1DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_vimage2DArray = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_vimage2DArray')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_subpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_subpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_subpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_subpassInputMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isubpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isubpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_isubpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_isubpassInputMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usubpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usubpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_usubpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_usubpassInputMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureSubpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureSubpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_textureSubpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_textureSubpassInputMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itextureSubpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itextureSubpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_itextureSubpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_itextureSubpassInputMS')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utextureSubpassInput = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utextureSubpassInput')
except (ValueError,AttributeError): pass
try: glsl_type_builtin_utextureSubpassInputMS = struct_glsl_type.in_dll(dll, 'glsl_type_builtin_utextureSubpassInputMS')
except (ValueError,AttributeError): pass
enum_a6xx_shift_amount = CEnum(ctypes.c_uint32)
NO_SHIFT = enum_a6xx_shift_amount.define('NO_SHIFT', 0)
HALF_PIXEL_SHIFT = enum_a6xx_shift_amount.define('HALF_PIXEL_SHIFT', 1)
FULL_PIXEL_SHIFT = enum_a6xx_shift_amount.define('FULL_PIXEL_SHIFT', 2)

enum_a6xx_sequenced_thread_dist = CEnum(ctypes.c_uint32)
DIST_SCREEN_COORD = enum_a6xx_sequenced_thread_dist.define('DIST_SCREEN_COORD', 0)
DIST_ALL_TO_RB0 = enum_a6xx_sequenced_thread_dist.define('DIST_ALL_TO_RB0', 1)

enum_a6xx_single_prim_mode = CEnum(ctypes.c_uint32)
NO_FLUSH = enum_a6xx_single_prim_mode.define('NO_FLUSH', 0)
FLUSH_PER_OVERLAP_AND_OVERWRITE = enum_a6xx_single_prim_mode.define('FLUSH_PER_OVERLAP_AND_OVERWRITE', 1)
FLUSH_PER_OVERLAP = enum_a6xx_single_prim_mode.define('FLUSH_PER_OVERLAP', 3)

enum_a6xx_raster_mode = CEnum(ctypes.c_uint32)
TYPE_TILED = enum_a6xx_raster_mode.define('TYPE_TILED', 0)
TYPE_WRITER = enum_a6xx_raster_mode.define('TYPE_WRITER', 1)

enum_a6xx_raster_direction = CEnum(ctypes.c_uint32)
LR_TB = enum_a6xx_raster_direction.define('LR_TB', 0)
RL_TB = enum_a6xx_raster_direction.define('RL_TB', 1)
LR_BT = enum_a6xx_raster_direction.define('LR_BT', 2)
RB_BT = enum_a6xx_raster_direction.define('RB_BT', 3)

enum_a6xx_render_mode = CEnum(ctypes.c_uint32)
RENDERING_PASS = enum_a6xx_render_mode.define('RENDERING_PASS', 0)
BINNING_PASS = enum_a6xx_render_mode.define('BINNING_PASS', 1)

enum_a6xx_buffers_location = CEnum(ctypes.c_uint32)
BUFFERS_IN_GMEM = enum_a6xx_buffers_location.define('BUFFERS_IN_GMEM', 0)
BUFFERS_IN_SYSMEM = enum_a6xx_buffers_location.define('BUFFERS_IN_SYSMEM', 3)

enum_a6xx_lrz_feedback_mask = CEnum(ctypes.c_uint32)
LRZ_FEEDBACK_NONE = enum_a6xx_lrz_feedback_mask.define('LRZ_FEEDBACK_NONE', 0)
LRZ_FEEDBACK_EARLY_Z = enum_a6xx_lrz_feedback_mask.define('LRZ_FEEDBACK_EARLY_Z', 1)
LRZ_FEEDBACK_EARLY_Z_LATE_Z = enum_a6xx_lrz_feedback_mask.define('LRZ_FEEDBACK_EARLY_Z_LATE_Z', 2)
LRZ_FEEDBACK_EARLY_Z_OR_EARLY_Z_LATE_Z = enum_a6xx_lrz_feedback_mask.define('LRZ_FEEDBACK_EARLY_Z_OR_EARLY_Z_LATE_Z', 3)
LRZ_FEEDBACK_LATE_Z = enum_a6xx_lrz_feedback_mask.define('LRZ_FEEDBACK_LATE_Z', 4)

enum_a6xx_fsr_combiner = CEnum(ctypes.c_uint32)
FSR_COMBINER_OP_KEEP = enum_a6xx_fsr_combiner.define('FSR_COMBINER_OP_KEEP', 0)
FSR_COMBINER_OP_REPLACE = enum_a6xx_fsr_combiner.define('FSR_COMBINER_OP_REPLACE', 1)
FSR_COMBINER_OP_MIN = enum_a6xx_fsr_combiner.define('FSR_COMBINER_OP_MIN', 2)
FSR_COMBINER_OP_MAX = enum_a6xx_fsr_combiner.define('FSR_COMBINER_OP_MAX', 3)
FSR_COMBINER_OP_MUL = enum_a6xx_fsr_combiner.define('FSR_COMBINER_OP_MUL', 4)

enum_a6xx_lrz_dir_status = CEnum(ctypes.c_uint32)
LRZ_DIR_LE = enum_a6xx_lrz_dir_status.define('LRZ_DIR_LE', 1)
LRZ_DIR_GE = enum_a6xx_lrz_dir_status.define('LRZ_DIR_GE', 2)
LRZ_DIR_INVALID = enum_a6xx_lrz_dir_status.define('LRZ_DIR_INVALID', 3)

enum_a6xx_fragcoord_sample_mode = CEnum(ctypes.c_uint32)
FRAGCOORD_CENTER = enum_a6xx_fragcoord_sample_mode.define('FRAGCOORD_CENTER', 0)
FRAGCOORD_SAMPLE = enum_a6xx_fragcoord_sample_mode.define('FRAGCOORD_SAMPLE', 3)

enum_a6xx_rotation = CEnum(ctypes.c_uint32)
ROTATE_0 = enum_a6xx_rotation.define('ROTATE_0', 0)
ROTATE_90 = enum_a6xx_rotation.define('ROTATE_90', 1)
ROTATE_180 = enum_a6xx_rotation.define('ROTATE_180', 2)
ROTATE_270 = enum_a6xx_rotation.define('ROTATE_270', 3)
ROTATE_HFLIP = enum_a6xx_rotation.define('ROTATE_HFLIP', 4)
ROTATE_VFLIP = enum_a6xx_rotation.define('ROTATE_VFLIP', 5)

enum_a6xx_blit_event_type = CEnum(ctypes.c_uint32)
BLIT_EVENT_STORE = enum_a6xx_blit_event_type.define('BLIT_EVENT_STORE', 0)
BLIT_EVENT_STORE_AND_CLEAR = enum_a6xx_blit_event_type.define('BLIT_EVENT_STORE_AND_CLEAR', 1)
BLIT_EVENT_CLEAR = enum_a6xx_blit_event_type.define('BLIT_EVENT_CLEAR', 2)
BLIT_EVENT_LOAD = enum_a6xx_blit_event_type.define('BLIT_EVENT_LOAD', 3)

enum_a7xx_blit_clear_mode = CEnum(ctypes.c_uint32)
CLEAR_MODE_SYSMEM = enum_a7xx_blit_clear_mode.define('CLEAR_MODE_SYSMEM', 0)
CLEAR_MODE_GMEM = enum_a7xx_blit_clear_mode.define('CLEAR_MODE_GMEM', 1)

enum_a6xx_ccu_cache_size = CEnum(ctypes.c_uint32)
CCU_CACHE_SIZE_FULL = enum_a6xx_ccu_cache_size.define('CCU_CACHE_SIZE_FULL', 0)
CCU_CACHE_SIZE_HALF = enum_a6xx_ccu_cache_size.define('CCU_CACHE_SIZE_HALF', 1)
CCU_CACHE_SIZE_QUARTER = enum_a6xx_ccu_cache_size.define('CCU_CACHE_SIZE_QUARTER', 2)
CCU_CACHE_SIZE_EIGHTH = enum_a6xx_ccu_cache_size.define('CCU_CACHE_SIZE_EIGHTH', 3)

enum_a7xx_concurrent_resolve_mode = CEnum(ctypes.c_uint32)
CONCURRENT_RESOLVE_MODE_DISABLED = enum_a7xx_concurrent_resolve_mode.define('CONCURRENT_RESOLVE_MODE_DISABLED', 0)
CONCURRENT_RESOLVE_MODE_1 = enum_a7xx_concurrent_resolve_mode.define('CONCURRENT_RESOLVE_MODE_1', 1)
CONCURRENT_RESOLVE_MODE_2 = enum_a7xx_concurrent_resolve_mode.define('CONCURRENT_RESOLVE_MODE_2', 2)

enum_a7xx_concurrent_unresolve_mode = CEnum(ctypes.c_uint32)
CONCURRENT_UNRESOLVE_MODE_DISABLED = enum_a7xx_concurrent_unresolve_mode.define('CONCURRENT_UNRESOLVE_MODE_DISABLED', 0)
CONCURRENT_UNRESOLVE_MODE_PARTIAL = enum_a7xx_concurrent_unresolve_mode.define('CONCURRENT_UNRESOLVE_MODE_PARTIAL', 1)
CONCURRENT_UNRESOLVE_MODE_FULL = enum_a7xx_concurrent_unresolve_mode.define('CONCURRENT_UNRESOLVE_MODE_FULL', 3)

enum_a6xx_varying_interp_mode = CEnum(ctypes.c_uint32)
INTERP_SMOOTH = enum_a6xx_varying_interp_mode.define('INTERP_SMOOTH', 0)
INTERP_FLAT = enum_a6xx_varying_interp_mode.define('INTERP_FLAT', 1)
INTERP_ZERO = enum_a6xx_varying_interp_mode.define('INTERP_ZERO', 2)
INTERP_ONE = enum_a6xx_varying_interp_mode.define('INTERP_ONE', 3)

enum_a6xx_varying_ps_repl_mode = CEnum(ctypes.c_uint32)
PS_REPL_NONE = enum_a6xx_varying_ps_repl_mode.define('PS_REPL_NONE', 0)
PS_REPL_S = enum_a6xx_varying_ps_repl_mode.define('PS_REPL_S', 1)
PS_REPL_T = enum_a6xx_varying_ps_repl_mode.define('PS_REPL_T', 2)
PS_REPL_ONE_MINUS_T = enum_a6xx_varying_ps_repl_mode.define('PS_REPL_ONE_MINUS_T', 3)

enum_a6xx_threadsize = CEnum(ctypes.c_uint32)
THREAD64 = enum_a6xx_threadsize.define('THREAD64', 0)
THREAD128 = enum_a6xx_threadsize.define('THREAD128', 1)

enum_a6xx_const_ram_mode = CEnum(ctypes.c_uint32)
CONSTLEN_128 = enum_a6xx_const_ram_mode.define('CONSTLEN_128', 0)
CONSTLEN_192 = enum_a6xx_const_ram_mode.define('CONSTLEN_192', 1)
CONSTLEN_256 = enum_a6xx_const_ram_mode.define('CONSTLEN_256', 2)
CONSTLEN_512 = enum_a6xx_const_ram_mode.define('CONSTLEN_512', 3)

enum_a7xx_workitem_rast_order = CEnum(ctypes.c_uint32)
WORKITEMRASTORDER_LINEAR = enum_a7xx_workitem_rast_order.define('WORKITEMRASTORDER_LINEAR', 0)
WORKITEMRASTORDER_TILED = enum_a7xx_workitem_rast_order.define('WORKITEMRASTORDER_TILED', 1)

enum_a6xx_bindless_descriptor_size = CEnum(ctypes.c_uint32)
BINDLESS_DESCRIPTOR_16B = enum_a6xx_bindless_descriptor_size.define('BINDLESS_DESCRIPTOR_16B', 1)
BINDLESS_DESCRIPTOR_64B = enum_a6xx_bindless_descriptor_size.define('BINDLESS_DESCRIPTOR_64B', 3)

enum_a6xx_isam_mode = CEnum(ctypes.c_uint32)
ISAMMODE_CL = enum_a6xx_isam_mode.define('ISAMMODE_CL', 1)
ISAMMODE_GL = enum_a6xx_isam_mode.define('ISAMMODE_GL', 2)

enum_a6xx_sp_a2d_output_ifmt_type = CEnum(ctypes.c_uint32)
OUTPUT_IFMT_2D_FLOAT = enum_a6xx_sp_a2d_output_ifmt_type.define('OUTPUT_IFMT_2D_FLOAT', 0)
OUTPUT_IFMT_2D_SINT = enum_a6xx_sp_a2d_output_ifmt_type.define('OUTPUT_IFMT_2D_SINT', 1)
OUTPUT_IFMT_2D_UINT = enum_a6xx_sp_a2d_output_ifmt_type.define('OUTPUT_IFMT_2D_UINT', 2)

enum_a6xx_coord_round = CEnum(ctypes.c_uint32)
COORD_TRUNCATE = enum_a6xx_coord_round.define('COORD_TRUNCATE', 0)
COORD_ROUND_NEAREST_EVEN = enum_a6xx_coord_round.define('COORD_ROUND_NEAREST_EVEN', 1)

enum_a6xx_nearest_mode = CEnum(ctypes.c_uint32)
ROUND_CLAMP_TRUNCATE = enum_a6xx_nearest_mode.define('ROUND_CLAMP_TRUNCATE', 0)
CLAMP_ROUND_TRUNCATE = enum_a6xx_nearest_mode.define('CLAMP_ROUND_TRUNCATE', 1)

enum_a7xx_cs_yalign = CEnum(ctypes.c_uint32)
CS_YALIGN_1 = enum_a7xx_cs_yalign.define('CS_YALIGN_1', 8)
CS_YALIGN_2 = enum_a7xx_cs_yalign.define('CS_YALIGN_2', 4)
CS_YALIGN_4 = enum_a7xx_cs_yalign.define('CS_YALIGN_4', 2)
CS_YALIGN_8 = enum_a7xx_cs_yalign.define('CS_YALIGN_8', 1)

enum_vgt_event_type = CEnum(ctypes.c_uint32)
VS_DEALLOC = enum_vgt_event_type.define('VS_DEALLOC', 0)
PS_DEALLOC = enum_vgt_event_type.define('PS_DEALLOC', 1)
VS_DONE_TS = enum_vgt_event_type.define('VS_DONE_TS', 2)
PS_DONE_TS = enum_vgt_event_type.define('PS_DONE_TS', 3)
CACHE_FLUSH_TS = enum_vgt_event_type.define('CACHE_FLUSH_TS', 4)
CONTEXT_DONE = enum_vgt_event_type.define('CONTEXT_DONE', 5)
CACHE_FLUSH = enum_vgt_event_type.define('CACHE_FLUSH', 6)
VIZQUERY_START = enum_vgt_event_type.define('VIZQUERY_START', 7)
HLSQ_FLUSH = enum_vgt_event_type.define('HLSQ_FLUSH', 7)
VIZQUERY_END = enum_vgt_event_type.define('VIZQUERY_END', 8)
SC_WAIT_WC = enum_vgt_event_type.define('SC_WAIT_WC', 9)
WRITE_PRIMITIVE_COUNTS = enum_vgt_event_type.define('WRITE_PRIMITIVE_COUNTS', 9)
START_PRIMITIVE_CTRS = enum_vgt_event_type.define('START_PRIMITIVE_CTRS', 11)
STOP_PRIMITIVE_CTRS = enum_vgt_event_type.define('STOP_PRIMITIVE_CTRS', 12)
RST_PIX_CNT = enum_vgt_event_type.define('RST_PIX_CNT', 13)
RST_VTX_CNT = enum_vgt_event_type.define('RST_VTX_CNT', 14)
TILE_FLUSH = enum_vgt_event_type.define('TILE_FLUSH', 15)
STAT_EVENT = enum_vgt_event_type.define('STAT_EVENT', 16)
CACHE_FLUSH_AND_INV_TS_EVENT = enum_vgt_event_type.define('CACHE_FLUSH_AND_INV_TS_EVENT', 20)
ZPASS_DONE = enum_vgt_event_type.define('ZPASS_DONE', 21)
CACHE_FLUSH_AND_INV_EVENT = enum_vgt_event_type.define('CACHE_FLUSH_AND_INV_EVENT', 22)
RB_DONE_TS = enum_vgt_event_type.define('RB_DONE_TS', 22)
PERFCOUNTER_START = enum_vgt_event_type.define('PERFCOUNTER_START', 23)
PERFCOUNTER_STOP = enum_vgt_event_type.define('PERFCOUNTER_STOP', 24)
VS_FETCH_DONE = enum_vgt_event_type.define('VS_FETCH_DONE', 27)
FACENESS_FLUSH = enum_vgt_event_type.define('FACENESS_FLUSH', 28)
WT_DONE_TS = enum_vgt_event_type.define('WT_DONE_TS', 8)
START_FRAGMENT_CTRS = enum_vgt_event_type.define('START_FRAGMENT_CTRS', 13)
STOP_FRAGMENT_CTRS = enum_vgt_event_type.define('STOP_FRAGMENT_CTRS', 14)
START_COMPUTE_CTRS = enum_vgt_event_type.define('START_COMPUTE_CTRS', 15)
STOP_COMPUTE_CTRS = enum_vgt_event_type.define('STOP_COMPUTE_CTRS', 16)
FLUSH_SO_0 = enum_vgt_event_type.define('FLUSH_SO_0', 17)
FLUSH_SO_1 = enum_vgt_event_type.define('FLUSH_SO_1', 18)
FLUSH_SO_2 = enum_vgt_event_type.define('FLUSH_SO_2', 19)
FLUSH_SO_3 = enum_vgt_event_type.define('FLUSH_SO_3', 20)
PC_CCU_INVALIDATE_DEPTH = enum_vgt_event_type.define('PC_CCU_INVALIDATE_DEPTH', 24)
PC_CCU_INVALIDATE_COLOR = enum_vgt_event_type.define('PC_CCU_INVALIDATE_COLOR', 25)
PC_CCU_RESOLVE_TS = enum_vgt_event_type.define('PC_CCU_RESOLVE_TS', 26)
PC_CCU_FLUSH_DEPTH_TS = enum_vgt_event_type.define('PC_CCU_FLUSH_DEPTH_TS', 28)
PC_CCU_FLUSH_COLOR_TS = enum_vgt_event_type.define('PC_CCU_FLUSH_COLOR_TS', 29)
BLIT = enum_vgt_event_type.define('BLIT', 30)
LRZ_FLIP_BUFFER = enum_vgt_event_type.define('LRZ_FLIP_BUFFER', 36)
LRZ_CLEAR = enum_vgt_event_type.define('LRZ_CLEAR', 37)
LRZ_FLUSH = enum_vgt_event_type.define('LRZ_FLUSH', 38)
BLIT_OP_FILL_2D = enum_vgt_event_type.define('BLIT_OP_FILL_2D', 39)
BLIT_OP_COPY_2D = enum_vgt_event_type.define('BLIT_OP_COPY_2D', 40)
UNK_40 = enum_vgt_event_type.define('UNK_40', 40)
LRZ_Q_CACHE_INVALIDATE = enum_vgt_event_type.define('LRZ_Q_CACHE_INVALIDATE', 41)
BLIT_OP_SCALE_2D = enum_vgt_event_type.define('BLIT_OP_SCALE_2D', 42)
CONTEXT_DONE_2D = enum_vgt_event_type.define('CONTEXT_DONE_2D', 43)
UNK_2C = enum_vgt_event_type.define('UNK_2C', 44)
UNK_2D = enum_vgt_event_type.define('UNK_2D', 45)
CACHE_INVALIDATE = enum_vgt_event_type.define('CACHE_INVALIDATE', 49)
LABEL = enum_vgt_event_type.define('LABEL', 63)
DUMMY_EVENT = enum_vgt_event_type.define('DUMMY_EVENT', 1)
CCU_INVALIDATE_DEPTH = enum_vgt_event_type.define('CCU_INVALIDATE_DEPTH', 24)
CCU_INVALIDATE_COLOR = enum_vgt_event_type.define('CCU_INVALIDATE_COLOR', 25)
CCU_RESOLVE_CLEAN = enum_vgt_event_type.define('CCU_RESOLVE_CLEAN', 26)
CCU_FLUSH_DEPTH = enum_vgt_event_type.define('CCU_FLUSH_DEPTH', 28)
CCU_FLUSH_COLOR = enum_vgt_event_type.define('CCU_FLUSH_COLOR', 29)
CCU_RESOLVE = enum_vgt_event_type.define('CCU_RESOLVE', 30)
CCU_END_RESOLVE_GROUP = enum_vgt_event_type.define('CCU_END_RESOLVE_GROUP', 31)
CCU_CLEAN_DEPTH = enum_vgt_event_type.define('CCU_CLEAN_DEPTH', 32)
CCU_CLEAN_COLOR = enum_vgt_event_type.define('CCU_CLEAN_COLOR', 33)
CACHE_RESET = enum_vgt_event_type.define('CACHE_RESET', 48)
CACHE_CLEAN = enum_vgt_event_type.define('CACHE_CLEAN', 49)
CACHE_FLUSH7 = enum_vgt_event_type.define('CACHE_FLUSH7', 50)
CACHE_INVALIDATE7 = enum_vgt_event_type.define('CACHE_INVALIDATE7', 51)

enum_pc_di_primtype = CEnum(ctypes.c_uint32)
DI_PT_NONE = enum_pc_di_primtype.define('DI_PT_NONE', 0)
DI_PT_POINTLIST_PSIZE = enum_pc_di_primtype.define('DI_PT_POINTLIST_PSIZE', 1)
DI_PT_LINELIST = enum_pc_di_primtype.define('DI_PT_LINELIST', 2)
DI_PT_LINESTRIP = enum_pc_di_primtype.define('DI_PT_LINESTRIP', 3)
DI_PT_TRILIST = enum_pc_di_primtype.define('DI_PT_TRILIST', 4)
DI_PT_TRIFAN = enum_pc_di_primtype.define('DI_PT_TRIFAN', 5)
DI_PT_TRISTRIP = enum_pc_di_primtype.define('DI_PT_TRISTRIP', 6)
DI_PT_LINELOOP = enum_pc_di_primtype.define('DI_PT_LINELOOP', 7)
DI_PT_RECTLIST = enum_pc_di_primtype.define('DI_PT_RECTLIST', 8)
DI_PT_POINTLIST = enum_pc_di_primtype.define('DI_PT_POINTLIST', 9)
DI_PT_LINE_ADJ = enum_pc_di_primtype.define('DI_PT_LINE_ADJ', 10)
DI_PT_LINESTRIP_ADJ = enum_pc_di_primtype.define('DI_PT_LINESTRIP_ADJ', 11)
DI_PT_TRI_ADJ = enum_pc_di_primtype.define('DI_PT_TRI_ADJ', 12)
DI_PT_TRISTRIP_ADJ = enum_pc_di_primtype.define('DI_PT_TRISTRIP_ADJ', 13)
DI_PT_PATCHES0 = enum_pc_di_primtype.define('DI_PT_PATCHES0', 31)
DI_PT_PATCHES1 = enum_pc_di_primtype.define('DI_PT_PATCHES1', 32)
DI_PT_PATCHES2 = enum_pc_di_primtype.define('DI_PT_PATCHES2', 33)
DI_PT_PATCHES3 = enum_pc_di_primtype.define('DI_PT_PATCHES3', 34)
DI_PT_PATCHES4 = enum_pc_di_primtype.define('DI_PT_PATCHES4', 35)
DI_PT_PATCHES5 = enum_pc_di_primtype.define('DI_PT_PATCHES5', 36)
DI_PT_PATCHES6 = enum_pc_di_primtype.define('DI_PT_PATCHES6', 37)
DI_PT_PATCHES7 = enum_pc_di_primtype.define('DI_PT_PATCHES7', 38)
DI_PT_PATCHES8 = enum_pc_di_primtype.define('DI_PT_PATCHES8', 39)
DI_PT_PATCHES9 = enum_pc_di_primtype.define('DI_PT_PATCHES9', 40)
DI_PT_PATCHES10 = enum_pc_di_primtype.define('DI_PT_PATCHES10', 41)
DI_PT_PATCHES11 = enum_pc_di_primtype.define('DI_PT_PATCHES11', 42)
DI_PT_PATCHES12 = enum_pc_di_primtype.define('DI_PT_PATCHES12', 43)
DI_PT_PATCHES13 = enum_pc_di_primtype.define('DI_PT_PATCHES13', 44)
DI_PT_PATCHES14 = enum_pc_di_primtype.define('DI_PT_PATCHES14', 45)
DI_PT_PATCHES15 = enum_pc_di_primtype.define('DI_PT_PATCHES15', 46)
DI_PT_PATCHES16 = enum_pc_di_primtype.define('DI_PT_PATCHES16', 47)
DI_PT_PATCHES17 = enum_pc_di_primtype.define('DI_PT_PATCHES17', 48)
DI_PT_PATCHES18 = enum_pc_di_primtype.define('DI_PT_PATCHES18', 49)
DI_PT_PATCHES19 = enum_pc_di_primtype.define('DI_PT_PATCHES19', 50)
DI_PT_PATCHES20 = enum_pc_di_primtype.define('DI_PT_PATCHES20', 51)
DI_PT_PATCHES21 = enum_pc_di_primtype.define('DI_PT_PATCHES21', 52)
DI_PT_PATCHES22 = enum_pc_di_primtype.define('DI_PT_PATCHES22', 53)
DI_PT_PATCHES23 = enum_pc_di_primtype.define('DI_PT_PATCHES23', 54)
DI_PT_PATCHES24 = enum_pc_di_primtype.define('DI_PT_PATCHES24', 55)
DI_PT_PATCHES25 = enum_pc_di_primtype.define('DI_PT_PATCHES25', 56)
DI_PT_PATCHES26 = enum_pc_di_primtype.define('DI_PT_PATCHES26', 57)
DI_PT_PATCHES27 = enum_pc_di_primtype.define('DI_PT_PATCHES27', 58)
DI_PT_PATCHES28 = enum_pc_di_primtype.define('DI_PT_PATCHES28', 59)
DI_PT_PATCHES29 = enum_pc_di_primtype.define('DI_PT_PATCHES29', 60)
DI_PT_PATCHES30 = enum_pc_di_primtype.define('DI_PT_PATCHES30', 61)
DI_PT_PATCHES31 = enum_pc_di_primtype.define('DI_PT_PATCHES31', 62)

enum_pc_di_src_sel = CEnum(ctypes.c_uint32)
DI_SRC_SEL_DMA = enum_pc_di_src_sel.define('DI_SRC_SEL_DMA', 0)
DI_SRC_SEL_IMMEDIATE = enum_pc_di_src_sel.define('DI_SRC_SEL_IMMEDIATE', 1)
DI_SRC_SEL_AUTO_INDEX = enum_pc_di_src_sel.define('DI_SRC_SEL_AUTO_INDEX', 2)
DI_SRC_SEL_AUTO_XFB = enum_pc_di_src_sel.define('DI_SRC_SEL_AUTO_XFB', 3)

enum_pc_di_face_cull_sel = CEnum(ctypes.c_uint32)
DI_FACE_CULL_NONE = enum_pc_di_face_cull_sel.define('DI_FACE_CULL_NONE', 0)
DI_FACE_CULL_FETCH = enum_pc_di_face_cull_sel.define('DI_FACE_CULL_FETCH', 1)
DI_FACE_BACKFACE_CULL = enum_pc_di_face_cull_sel.define('DI_FACE_BACKFACE_CULL', 2)
DI_FACE_FRONTFACE_CULL = enum_pc_di_face_cull_sel.define('DI_FACE_FRONTFACE_CULL', 3)

enum_pc_di_index_size = CEnum(ctypes.c_uint32)
INDEX_SIZE_IGN = enum_pc_di_index_size.define('INDEX_SIZE_IGN', 0)
INDEX_SIZE_16_BIT = enum_pc_di_index_size.define('INDEX_SIZE_16_BIT', 0)
INDEX_SIZE_32_BIT = enum_pc_di_index_size.define('INDEX_SIZE_32_BIT', 1)
INDEX_SIZE_8_BIT = enum_pc_di_index_size.define('INDEX_SIZE_8_BIT', 2)
INDEX_SIZE_INVALID = enum_pc_di_index_size.define('INDEX_SIZE_INVALID', 0)

enum_pc_di_vis_cull_mode = CEnum(ctypes.c_uint32)
IGNORE_VISIBILITY = enum_pc_di_vis_cull_mode.define('IGNORE_VISIBILITY', 0)
USE_VISIBILITY = enum_pc_di_vis_cull_mode.define('USE_VISIBILITY', 1)

enum_adreno_pm4_packet_type = CEnum(ctypes.c_uint32)
CP_TYPE0_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE0_PKT', 0)
CP_TYPE1_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE1_PKT', 1073741824)
CP_TYPE2_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE2_PKT', 2147483648)
CP_TYPE3_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE3_PKT', 3221225472)
CP_TYPE4_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE4_PKT', 1073741824)
CP_TYPE7_PKT = enum_adreno_pm4_packet_type.define('CP_TYPE7_PKT', 1879048192)

enum_adreno_pm4_type3_packets = CEnum(ctypes.c_uint32)
CP_ME_INIT = enum_adreno_pm4_type3_packets.define('CP_ME_INIT', 72)
CP_NOP = enum_adreno_pm4_type3_packets.define('CP_NOP', 16)
CP_PREEMPT_ENABLE = enum_adreno_pm4_type3_packets.define('CP_PREEMPT_ENABLE', 28)
CP_PREEMPT_TOKEN = enum_adreno_pm4_type3_packets.define('CP_PREEMPT_TOKEN', 30)
CP_INDIRECT_BUFFER = enum_adreno_pm4_type3_packets.define('CP_INDIRECT_BUFFER', 63)
CP_INDIRECT_BUFFER_CHAIN = enum_adreno_pm4_type3_packets.define('CP_INDIRECT_BUFFER_CHAIN', 87)
CP_INDIRECT_BUFFER_PFD = enum_adreno_pm4_type3_packets.define('CP_INDIRECT_BUFFER_PFD', 55)
CP_WAIT_FOR_IDLE = enum_adreno_pm4_type3_packets.define('CP_WAIT_FOR_IDLE', 38)
CP_WAIT_REG_MEM = enum_adreno_pm4_type3_packets.define('CP_WAIT_REG_MEM', 60)
CP_WAIT_REG_EQ = enum_adreno_pm4_type3_packets.define('CP_WAIT_REG_EQ', 82)
CP_WAIT_REG_GTE = enum_adreno_pm4_type3_packets.define('CP_WAIT_REG_GTE', 83)
CP_WAIT_UNTIL_READ = enum_adreno_pm4_type3_packets.define('CP_WAIT_UNTIL_READ', 92)
CP_WAIT_IB_PFD_COMPLETE = enum_adreno_pm4_type3_packets.define('CP_WAIT_IB_PFD_COMPLETE', 93)
CP_REG_RMW = enum_adreno_pm4_type3_packets.define('CP_REG_RMW', 33)
CP_SET_BIN_DATA = enum_adreno_pm4_type3_packets.define('CP_SET_BIN_DATA', 47)
CP_SET_BIN_DATA5 = enum_adreno_pm4_type3_packets.define('CP_SET_BIN_DATA5', 47)
CP_REG_TO_MEM = enum_adreno_pm4_type3_packets.define('CP_REG_TO_MEM', 62)
CP_MEM_WRITE = enum_adreno_pm4_type3_packets.define('CP_MEM_WRITE', 61)
CP_MEM_WRITE_CNTR = enum_adreno_pm4_type3_packets.define('CP_MEM_WRITE_CNTR', 79)
CP_COND_EXEC = enum_adreno_pm4_type3_packets.define('CP_COND_EXEC', 68)
CP_COND_WRITE = enum_adreno_pm4_type3_packets.define('CP_COND_WRITE', 69)
CP_COND_WRITE5 = enum_adreno_pm4_type3_packets.define('CP_COND_WRITE5', 69)
CP_EVENT_WRITE = enum_adreno_pm4_type3_packets.define('CP_EVENT_WRITE', 70)
CP_EVENT_WRITE7 = enum_adreno_pm4_type3_packets.define('CP_EVENT_WRITE7', 70)
CP_EVENT_WRITE_SHD = enum_adreno_pm4_type3_packets.define('CP_EVENT_WRITE_SHD', 88)
CP_EVENT_WRITE_CFL = enum_adreno_pm4_type3_packets.define('CP_EVENT_WRITE_CFL', 89)
CP_EVENT_WRITE_ZPD = enum_adreno_pm4_type3_packets.define('CP_EVENT_WRITE_ZPD', 91)
CP_RUN_OPENCL = enum_adreno_pm4_type3_packets.define('CP_RUN_OPENCL', 49)
CP_DRAW_INDX = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX', 34)
CP_DRAW_INDX_2 = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX_2', 54)
CP_DRAW_INDX_BIN = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX_BIN', 52)
CP_DRAW_INDX_2_BIN = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX_2_BIN', 53)
CP_VIZ_QUERY = enum_adreno_pm4_type3_packets.define('CP_VIZ_QUERY', 35)
CP_SET_STATE = enum_adreno_pm4_type3_packets.define('CP_SET_STATE', 37)
CP_SET_CONSTANT = enum_adreno_pm4_type3_packets.define('CP_SET_CONSTANT', 45)
CP_IM_LOAD = enum_adreno_pm4_type3_packets.define('CP_IM_LOAD', 39)
CP_IM_LOAD_IMMEDIATE = enum_adreno_pm4_type3_packets.define('CP_IM_LOAD_IMMEDIATE', 43)
CP_LOAD_CONSTANT_CONTEXT = enum_adreno_pm4_type3_packets.define('CP_LOAD_CONSTANT_CONTEXT', 46)
CP_INVALIDATE_STATE = enum_adreno_pm4_type3_packets.define('CP_INVALIDATE_STATE', 59)
CP_SET_SHADER_BASES = enum_adreno_pm4_type3_packets.define('CP_SET_SHADER_BASES', 74)
CP_SET_BIN_MASK = enum_adreno_pm4_type3_packets.define('CP_SET_BIN_MASK', 80)
CP_SET_BIN_SELECT = enum_adreno_pm4_type3_packets.define('CP_SET_BIN_SELECT', 81)
CP_CONTEXT_UPDATE = enum_adreno_pm4_type3_packets.define('CP_CONTEXT_UPDATE', 94)
CP_INTERRUPT = enum_adreno_pm4_type3_packets.define('CP_INTERRUPT', 64)
CP_IM_STORE = enum_adreno_pm4_type3_packets.define('CP_IM_STORE', 44)
CP_SET_DRAW_INIT_FLAGS = enum_adreno_pm4_type3_packets.define('CP_SET_DRAW_INIT_FLAGS', 75)
CP_SET_PROTECTED_MODE = enum_adreno_pm4_type3_packets.define('CP_SET_PROTECTED_MODE', 95)
CP_BOOTSTRAP_UCODE = enum_adreno_pm4_type3_packets.define('CP_BOOTSTRAP_UCODE', 111)
CP_LOAD_STATE = enum_adreno_pm4_type3_packets.define('CP_LOAD_STATE', 48)
CP_LOAD_STATE4 = enum_adreno_pm4_type3_packets.define('CP_LOAD_STATE4', 48)
CP_COND_INDIRECT_BUFFER_PFE = enum_adreno_pm4_type3_packets.define('CP_COND_INDIRECT_BUFFER_PFE', 58)
CP_COND_INDIRECT_BUFFER_PFD = enum_adreno_pm4_type3_packets.define('CP_COND_INDIRECT_BUFFER_PFD', 50)
CP_INDIRECT_BUFFER_PFE = enum_adreno_pm4_type3_packets.define('CP_INDIRECT_BUFFER_PFE', 63)
CP_SET_BIN = enum_adreno_pm4_type3_packets.define('CP_SET_BIN', 76)
CP_TEST_TWO_MEMS = enum_adreno_pm4_type3_packets.define('CP_TEST_TWO_MEMS', 113)
CP_REG_WR_NO_CTXT = enum_adreno_pm4_type3_packets.define('CP_REG_WR_NO_CTXT', 120)
CP_RECORD_PFP_TIMESTAMP = enum_adreno_pm4_type3_packets.define('CP_RECORD_PFP_TIMESTAMP', 17)
CP_SET_SECURE_MODE = enum_adreno_pm4_type3_packets.define('CP_SET_SECURE_MODE', 102)
CP_WAIT_FOR_ME = enum_adreno_pm4_type3_packets.define('CP_WAIT_FOR_ME', 19)
CP_SET_DRAW_STATE = enum_adreno_pm4_type3_packets.define('CP_SET_DRAW_STATE', 67)
CP_DRAW_INDX_OFFSET = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX_OFFSET', 56)
CP_DRAW_INDIRECT = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDIRECT', 40)
CP_DRAW_INDX_INDIRECT = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDX_INDIRECT', 41)
CP_DRAW_INDIRECT_MULTI = enum_adreno_pm4_type3_packets.define('CP_DRAW_INDIRECT_MULTI', 42)
CP_DRAW_AUTO = enum_adreno_pm4_type3_packets.define('CP_DRAW_AUTO', 36)
CP_DRAW_PRED_ENABLE_GLOBAL = enum_adreno_pm4_type3_packets.define('CP_DRAW_PRED_ENABLE_GLOBAL', 25)
CP_DRAW_PRED_ENABLE_LOCAL = enum_adreno_pm4_type3_packets.define('CP_DRAW_PRED_ENABLE_LOCAL', 26)
CP_DRAW_PRED_SET = enum_adreno_pm4_type3_packets.define('CP_DRAW_PRED_SET', 78)
CP_WIDE_REG_WRITE = enum_adreno_pm4_type3_packets.define('CP_WIDE_REG_WRITE', 116)
CP_SCRATCH_TO_REG = enum_adreno_pm4_type3_packets.define('CP_SCRATCH_TO_REG', 77)
CP_REG_TO_SCRATCH = enum_adreno_pm4_type3_packets.define('CP_REG_TO_SCRATCH', 74)
CP_WAIT_MEM_WRITES = enum_adreno_pm4_type3_packets.define('CP_WAIT_MEM_WRITES', 18)
CP_COND_REG_EXEC = enum_adreno_pm4_type3_packets.define('CP_COND_REG_EXEC', 71)
CP_MEM_TO_REG = enum_adreno_pm4_type3_packets.define('CP_MEM_TO_REG', 66)
CP_EXEC_CS_INDIRECT = enum_adreno_pm4_type3_packets.define('CP_EXEC_CS_INDIRECT', 65)
CP_EXEC_CS = enum_adreno_pm4_type3_packets.define('CP_EXEC_CS', 51)
CP_PERFCOUNTER_ACTION = enum_adreno_pm4_type3_packets.define('CP_PERFCOUNTER_ACTION', 80)
CP_SMMU_TABLE_UPDATE = enum_adreno_pm4_type3_packets.define('CP_SMMU_TABLE_UPDATE', 83)
CP_SET_MARKER = enum_adreno_pm4_type3_packets.define('CP_SET_MARKER', 101)
CP_SET_PSEUDO_REG = enum_adreno_pm4_type3_packets.define('CP_SET_PSEUDO_REG', 86)
CP_CONTEXT_REG_BUNCH = enum_adreno_pm4_type3_packets.define('CP_CONTEXT_REG_BUNCH', 92)
CP_YIELD_ENABLE = enum_adreno_pm4_type3_packets.define('CP_YIELD_ENABLE', 28)
CP_SKIP_IB2_ENABLE_GLOBAL = enum_adreno_pm4_type3_packets.define('CP_SKIP_IB2_ENABLE_GLOBAL', 29)
CP_SKIP_IB2_ENABLE_LOCAL = enum_adreno_pm4_type3_packets.define('CP_SKIP_IB2_ENABLE_LOCAL', 35)
CP_SET_SUBDRAW_SIZE = enum_adreno_pm4_type3_packets.define('CP_SET_SUBDRAW_SIZE', 53)
CP_WHERE_AM_I = enum_adreno_pm4_type3_packets.define('CP_WHERE_AM_I', 98)
CP_SET_VISIBILITY_OVERRIDE = enum_adreno_pm4_type3_packets.define('CP_SET_VISIBILITY_OVERRIDE', 100)
CP_PREEMPT_ENABLE_GLOBAL = enum_adreno_pm4_type3_packets.define('CP_PREEMPT_ENABLE_GLOBAL', 105)
CP_PREEMPT_ENABLE_LOCAL = enum_adreno_pm4_type3_packets.define('CP_PREEMPT_ENABLE_LOCAL', 106)
CP_CONTEXT_SWITCH_YIELD = enum_adreno_pm4_type3_packets.define('CP_CONTEXT_SWITCH_YIELD', 107)
CP_SET_RENDER_MODE = enum_adreno_pm4_type3_packets.define('CP_SET_RENDER_MODE', 108)
CP_COMPUTE_CHECKPOINT = enum_adreno_pm4_type3_packets.define('CP_COMPUTE_CHECKPOINT', 110)
CP_MEM_TO_MEM = enum_adreno_pm4_type3_packets.define('CP_MEM_TO_MEM', 115)
CP_BLIT = enum_adreno_pm4_type3_packets.define('CP_BLIT', 44)
CP_REG_TEST = enum_adreno_pm4_type3_packets.define('CP_REG_TEST', 57)
CP_SET_MODE = enum_adreno_pm4_type3_packets.define('CP_SET_MODE', 99)
CP_LOAD_STATE6_GEOM = enum_adreno_pm4_type3_packets.define('CP_LOAD_STATE6_GEOM', 50)
CP_LOAD_STATE6_FRAG = enum_adreno_pm4_type3_packets.define('CP_LOAD_STATE6_FRAG', 52)
CP_LOAD_STATE6 = enum_adreno_pm4_type3_packets.define('CP_LOAD_STATE6', 54)
IN_IB_PREFETCH_END = enum_adreno_pm4_type3_packets.define('IN_IB_PREFETCH_END', 23)
IN_SUBBLK_PREFETCH = enum_adreno_pm4_type3_packets.define('IN_SUBBLK_PREFETCH', 31)
IN_INSTR_PREFETCH = enum_adreno_pm4_type3_packets.define('IN_INSTR_PREFETCH', 32)
IN_INSTR_MATCH = enum_adreno_pm4_type3_packets.define('IN_INSTR_MATCH', 71)
IN_CONST_PREFETCH = enum_adreno_pm4_type3_packets.define('IN_CONST_PREFETCH', 73)
IN_INCR_UPDT_STATE = enum_adreno_pm4_type3_packets.define('IN_INCR_UPDT_STATE', 85)
IN_INCR_UPDT_CONST = enum_adreno_pm4_type3_packets.define('IN_INCR_UPDT_CONST', 86)
IN_INCR_UPDT_INSTR = enum_adreno_pm4_type3_packets.define('IN_INCR_UPDT_INSTR', 87)
PKT4 = enum_adreno_pm4_type3_packets.define('PKT4', 4)
IN_IB_END = enum_adreno_pm4_type3_packets.define('IN_IB_END', 10)
IN_GMU_INTERRUPT = enum_adreno_pm4_type3_packets.define('IN_GMU_INTERRUPT', 11)
IN_PREEMPT = enum_adreno_pm4_type3_packets.define('IN_PREEMPT', 15)
CP_SCRATCH_WRITE = enum_adreno_pm4_type3_packets.define('CP_SCRATCH_WRITE', 76)
CP_REG_TO_MEM_OFFSET_MEM = enum_adreno_pm4_type3_packets.define('CP_REG_TO_MEM_OFFSET_MEM', 116)
CP_REG_TO_MEM_OFFSET_REG = enum_adreno_pm4_type3_packets.define('CP_REG_TO_MEM_OFFSET_REG', 114)
CP_WAIT_MEM_GTE = enum_adreno_pm4_type3_packets.define('CP_WAIT_MEM_GTE', 20)
CP_WAIT_TWO_REGS = enum_adreno_pm4_type3_packets.define('CP_WAIT_TWO_REGS', 112)
CP_MEMCPY = enum_adreno_pm4_type3_packets.define('CP_MEMCPY', 117)
CP_SET_BIN_DATA5_OFFSET = enum_adreno_pm4_type3_packets.define('CP_SET_BIN_DATA5_OFFSET', 46)
CP_SET_UNK_BIN_DATA = enum_adreno_pm4_type3_packets.define('CP_SET_UNK_BIN_DATA', 45)
CP_CONTEXT_SWITCH = enum_adreno_pm4_type3_packets.define('CP_CONTEXT_SWITCH', 84)
CP_SET_AMBLE = enum_adreno_pm4_type3_packets.define('CP_SET_AMBLE', 85)
CP_REG_WRITE = enum_adreno_pm4_type3_packets.define('CP_REG_WRITE', 109)
CP_START_BIN = enum_adreno_pm4_type3_packets.define('CP_START_BIN', 80)
CP_END_BIN = enum_adreno_pm4_type3_packets.define('CP_END_BIN', 81)
CP_PREEMPT_DISABLE = enum_adreno_pm4_type3_packets.define('CP_PREEMPT_DISABLE', 108)
CP_WAIT_TIMESTAMP = enum_adreno_pm4_type3_packets.define('CP_WAIT_TIMESTAMP', 20)
CP_GLOBAL_TIMESTAMP = enum_adreno_pm4_type3_packets.define('CP_GLOBAL_TIMESTAMP', 21)
CP_LOCAL_TIMESTAMP = enum_adreno_pm4_type3_packets.define('CP_LOCAL_TIMESTAMP', 22)
CP_THREAD_CONTROL = enum_adreno_pm4_type3_packets.define('CP_THREAD_CONTROL', 23)
CP_RESOURCE_LIST = enum_adreno_pm4_type3_packets.define('CP_RESOURCE_LIST', 24)
CP_BV_BR_COUNT_OPS = enum_adreno_pm4_type3_packets.define('CP_BV_BR_COUNT_OPS', 27)
CP_MODIFY_TIMESTAMP = enum_adreno_pm4_type3_packets.define('CP_MODIFY_TIMESTAMP', 28)
CP_CONTEXT_REG_BUNCH2 = enum_adreno_pm4_type3_packets.define('CP_CONTEXT_REG_BUNCH2', 93)
CP_MEM_TO_SCRATCH_MEM = enum_adreno_pm4_type3_packets.define('CP_MEM_TO_SCRATCH_MEM', 73)
CP_FIXED_STRIDE_DRAW_TABLE = enum_adreno_pm4_type3_packets.define('CP_FIXED_STRIDE_DRAW_TABLE', 127)
CP_RESET_CONTEXT_STATE = enum_adreno_pm4_type3_packets.define('CP_RESET_CONTEXT_STATE', 31)
CP_CCHE_INVALIDATE = enum_adreno_pm4_type3_packets.define('CP_CCHE_INVALIDATE', 58)
CP_SCOPE_CNTL = enum_adreno_pm4_type3_packets.define('CP_SCOPE_CNTL', 108)

enum_adreno_state_block = CEnum(ctypes.c_uint32)
SB_VERT_TEX = enum_adreno_state_block.define('SB_VERT_TEX', 0)
SB_VERT_MIPADDR = enum_adreno_state_block.define('SB_VERT_MIPADDR', 1)
SB_FRAG_TEX = enum_adreno_state_block.define('SB_FRAG_TEX', 2)
SB_FRAG_MIPADDR = enum_adreno_state_block.define('SB_FRAG_MIPADDR', 3)
SB_VERT_SHADER = enum_adreno_state_block.define('SB_VERT_SHADER', 4)
SB_GEOM_SHADER = enum_adreno_state_block.define('SB_GEOM_SHADER', 5)
SB_FRAG_SHADER = enum_adreno_state_block.define('SB_FRAG_SHADER', 6)
SB_COMPUTE_SHADER = enum_adreno_state_block.define('SB_COMPUTE_SHADER', 7)

enum_adreno_state_type = CEnum(ctypes.c_uint32)
ST_SHADER = enum_adreno_state_type.define('ST_SHADER', 0)
ST_CONSTANTS = enum_adreno_state_type.define('ST_CONSTANTS', 1)

enum_adreno_state_src = CEnum(ctypes.c_uint32)
SS_DIRECT = enum_adreno_state_src.define('SS_DIRECT', 0)
SS_INVALID_ALL_IC = enum_adreno_state_src.define('SS_INVALID_ALL_IC', 2)
SS_INVALID_PART_IC = enum_adreno_state_src.define('SS_INVALID_PART_IC', 3)
SS_INDIRECT = enum_adreno_state_src.define('SS_INDIRECT', 4)
SS_INDIRECT_TCM = enum_adreno_state_src.define('SS_INDIRECT_TCM', 5)
SS_INDIRECT_STM = enum_adreno_state_src.define('SS_INDIRECT_STM', 6)

enum_a4xx_state_block = CEnum(ctypes.c_uint32)
SB4_VS_TEX = enum_a4xx_state_block.define('SB4_VS_TEX', 0)
SB4_HS_TEX = enum_a4xx_state_block.define('SB4_HS_TEX', 1)
SB4_DS_TEX = enum_a4xx_state_block.define('SB4_DS_TEX', 2)
SB4_GS_TEX = enum_a4xx_state_block.define('SB4_GS_TEX', 3)
SB4_FS_TEX = enum_a4xx_state_block.define('SB4_FS_TEX', 4)
SB4_CS_TEX = enum_a4xx_state_block.define('SB4_CS_TEX', 5)
SB4_VS_SHADER = enum_a4xx_state_block.define('SB4_VS_SHADER', 8)
SB4_HS_SHADER = enum_a4xx_state_block.define('SB4_HS_SHADER', 9)
SB4_DS_SHADER = enum_a4xx_state_block.define('SB4_DS_SHADER', 10)
SB4_GS_SHADER = enum_a4xx_state_block.define('SB4_GS_SHADER', 11)
SB4_FS_SHADER = enum_a4xx_state_block.define('SB4_FS_SHADER', 12)
SB4_CS_SHADER = enum_a4xx_state_block.define('SB4_CS_SHADER', 13)
SB4_SSBO = enum_a4xx_state_block.define('SB4_SSBO', 14)
SB4_CS_SSBO = enum_a4xx_state_block.define('SB4_CS_SSBO', 15)

enum_a4xx_state_type = CEnum(ctypes.c_uint32)
ST4_SHADER = enum_a4xx_state_type.define('ST4_SHADER', 0)
ST4_CONSTANTS = enum_a4xx_state_type.define('ST4_CONSTANTS', 1)
ST4_UBO = enum_a4xx_state_type.define('ST4_UBO', 2)

enum_a4xx_state_src = CEnum(ctypes.c_uint32)
SS4_DIRECT = enum_a4xx_state_src.define('SS4_DIRECT', 0)
SS4_INDIRECT = enum_a4xx_state_src.define('SS4_INDIRECT', 2)

enum_a6xx_state_block = CEnum(ctypes.c_uint32)
SB6_VS_TEX = enum_a6xx_state_block.define('SB6_VS_TEX', 0)
SB6_HS_TEX = enum_a6xx_state_block.define('SB6_HS_TEX', 1)
SB6_DS_TEX = enum_a6xx_state_block.define('SB6_DS_TEX', 2)
SB6_GS_TEX = enum_a6xx_state_block.define('SB6_GS_TEX', 3)
SB6_FS_TEX = enum_a6xx_state_block.define('SB6_FS_TEX', 4)
SB6_CS_TEX = enum_a6xx_state_block.define('SB6_CS_TEX', 5)
SB6_VS_SHADER = enum_a6xx_state_block.define('SB6_VS_SHADER', 8)
SB6_HS_SHADER = enum_a6xx_state_block.define('SB6_HS_SHADER', 9)
SB6_DS_SHADER = enum_a6xx_state_block.define('SB6_DS_SHADER', 10)
SB6_GS_SHADER = enum_a6xx_state_block.define('SB6_GS_SHADER', 11)
SB6_FS_SHADER = enum_a6xx_state_block.define('SB6_FS_SHADER', 12)
SB6_CS_SHADER = enum_a6xx_state_block.define('SB6_CS_SHADER', 13)
SB6_UAV = enum_a6xx_state_block.define('SB6_UAV', 14)
SB6_CS_UAV = enum_a6xx_state_block.define('SB6_CS_UAV', 15)

enum_a6xx_state_type = CEnum(ctypes.c_uint32)
ST6_SHADER = enum_a6xx_state_type.define('ST6_SHADER', 0)
ST6_CONSTANTS = enum_a6xx_state_type.define('ST6_CONSTANTS', 1)
ST6_UBO = enum_a6xx_state_type.define('ST6_UBO', 2)
ST6_UAV = enum_a6xx_state_type.define('ST6_UAV', 3)

enum_a6xx_state_src = CEnum(ctypes.c_uint32)
SS6_DIRECT = enum_a6xx_state_src.define('SS6_DIRECT', 0)
SS6_BINDLESS = enum_a6xx_state_src.define('SS6_BINDLESS', 1)
SS6_INDIRECT = enum_a6xx_state_src.define('SS6_INDIRECT', 2)
SS6_UBO = enum_a6xx_state_src.define('SS6_UBO', 3)

enum_a4xx_index_size = CEnum(ctypes.c_uint32)
INDEX4_SIZE_8_BIT = enum_a4xx_index_size.define('INDEX4_SIZE_8_BIT', 0)
INDEX4_SIZE_16_BIT = enum_a4xx_index_size.define('INDEX4_SIZE_16_BIT', 1)
INDEX4_SIZE_32_BIT = enum_a4xx_index_size.define('INDEX4_SIZE_32_BIT', 2)

enum_a6xx_patch_type = CEnum(ctypes.c_uint32)
TESS_QUADS = enum_a6xx_patch_type.define('TESS_QUADS', 0)
TESS_TRIANGLES = enum_a6xx_patch_type.define('TESS_TRIANGLES', 1)
TESS_ISOLINES = enum_a6xx_patch_type.define('TESS_ISOLINES', 2)

enum_a6xx_draw_indirect_opcode = CEnum(ctypes.c_uint32)
INDIRECT_OP_NORMAL = enum_a6xx_draw_indirect_opcode.define('INDIRECT_OP_NORMAL', 2)
INDIRECT_OP_INDEXED = enum_a6xx_draw_indirect_opcode.define('INDIRECT_OP_INDEXED', 4)
INDIRECT_OP_INDIRECT_COUNT = enum_a6xx_draw_indirect_opcode.define('INDIRECT_OP_INDIRECT_COUNT', 6)
INDIRECT_OP_INDIRECT_COUNT_INDEXED = enum_a6xx_draw_indirect_opcode.define('INDIRECT_OP_INDIRECT_COUNT_INDEXED', 7)

enum_cp_draw_pred_src = CEnum(ctypes.c_uint32)
PRED_SRC_MEM = enum_cp_draw_pred_src.define('PRED_SRC_MEM', 5)

enum_cp_draw_pred_test = CEnum(ctypes.c_uint32)
NE_0_PASS = enum_cp_draw_pred_test.define('NE_0_PASS', 0)
EQ_0_PASS = enum_cp_draw_pred_test.define('EQ_0_PASS', 1)

enum_a7xx_abs_mask_mode = CEnum(ctypes.c_uint32)
ABS_MASK = enum_a7xx_abs_mask_mode.define('ABS_MASK', 1)
NO_ABS_MASK = enum_a7xx_abs_mask_mode.define('NO_ABS_MASK', 0)

enum_cp_cond_function = CEnum(ctypes.c_uint32)
WRITE_ALWAYS = enum_cp_cond_function.define('WRITE_ALWAYS', 0)
WRITE_LT = enum_cp_cond_function.define('WRITE_LT', 1)
WRITE_LE = enum_cp_cond_function.define('WRITE_LE', 2)
WRITE_EQ = enum_cp_cond_function.define('WRITE_EQ', 3)
WRITE_NE = enum_cp_cond_function.define('WRITE_NE', 4)
WRITE_GE = enum_cp_cond_function.define('WRITE_GE', 5)
WRITE_GT = enum_cp_cond_function.define('WRITE_GT', 6)

enum_poll_memory_type = CEnum(ctypes.c_uint32)
POLL_REGISTER = enum_poll_memory_type.define('POLL_REGISTER', 0)
POLL_MEMORY = enum_poll_memory_type.define('POLL_MEMORY', 1)
POLL_SCRATCH = enum_poll_memory_type.define('POLL_SCRATCH', 2)
POLL_ON_CHIP = enum_poll_memory_type.define('POLL_ON_CHIP', 3)

enum_render_mode_cmd = CEnum(ctypes.c_uint32)
BYPASS = enum_render_mode_cmd.define('BYPASS', 1)
BINNING = enum_render_mode_cmd.define('BINNING', 2)
GMEM = enum_render_mode_cmd.define('GMEM', 3)
BLIT2D = enum_render_mode_cmd.define('BLIT2D', 5)
BLIT2DSCALE = enum_render_mode_cmd.define('BLIT2DSCALE', 7)
END2D = enum_render_mode_cmd.define('END2D', 8)

enum_event_write_src = CEnum(ctypes.c_uint32)
EV_WRITE_USER_32B = enum_event_write_src.define('EV_WRITE_USER_32B', 0)
EV_WRITE_USER_64B = enum_event_write_src.define('EV_WRITE_USER_64B', 1)
EV_WRITE_TIMESTAMP_SUM = enum_event_write_src.define('EV_WRITE_TIMESTAMP_SUM', 2)
EV_WRITE_ALWAYSON = enum_event_write_src.define('EV_WRITE_ALWAYSON', 3)
EV_WRITE_REGS_CONTENT = enum_event_write_src.define('EV_WRITE_REGS_CONTENT', 4)

enum_event_write_dst = CEnum(ctypes.c_uint32)
EV_DST_RAM = enum_event_write_dst.define('EV_DST_RAM', 0)
EV_DST_ONCHIP = enum_event_write_dst.define('EV_DST_ONCHIP', 1)

enum_cp_blit_cmd = CEnum(ctypes.c_uint32)
BLIT_OP_FILL = enum_cp_blit_cmd.define('BLIT_OP_FILL', 0)
BLIT_OP_COPY = enum_cp_blit_cmd.define('BLIT_OP_COPY', 1)
BLIT_OP_SCALE = enum_cp_blit_cmd.define('BLIT_OP_SCALE', 3)

enum_set_marker_mode = CEnum(ctypes.c_uint32)
SET_RENDER_MODE = enum_set_marker_mode.define('SET_RENDER_MODE', 0)
SET_IFPC_MODE = enum_set_marker_mode.define('SET_IFPC_MODE', 1)

enum_a6xx_ifpc_mode = CEnum(ctypes.c_uint32)
IFPC_ENABLE = enum_a6xx_ifpc_mode.define('IFPC_ENABLE', 0)
IFPC_DISABLE = enum_a6xx_ifpc_mode.define('IFPC_DISABLE', 1)

enum_a6xx_marker = CEnum(ctypes.c_uint32)
RM6_DIRECT_RENDER = enum_a6xx_marker.define('RM6_DIRECT_RENDER', 1)
RM6_BIN_VISIBILITY = enum_a6xx_marker.define('RM6_BIN_VISIBILITY', 2)
RM6_BIN_DIRECT = enum_a6xx_marker.define('RM6_BIN_DIRECT', 3)
RM6_BIN_RENDER_START = enum_a6xx_marker.define('RM6_BIN_RENDER_START', 4)
RM6_BIN_END_OF_DRAWS = enum_a6xx_marker.define('RM6_BIN_END_OF_DRAWS', 5)
RM6_BIN_RESOLVE = enum_a6xx_marker.define('RM6_BIN_RESOLVE', 6)
RM6_BIN_RENDER_END = enum_a6xx_marker.define('RM6_BIN_RENDER_END', 7)
RM6_COMPUTE = enum_a6xx_marker.define('RM6_COMPUTE', 8)
RM6_BLIT2DSCALE = enum_a6xx_marker.define('RM6_BLIT2DSCALE', 12)
RM6_IB1LIST_START = enum_a6xx_marker.define('RM6_IB1LIST_START', 13)
RM6_IB1LIST_END = enum_a6xx_marker.define('RM6_IB1LIST_END', 14)

enum_pseudo_reg = CEnum(ctypes.c_uint32)
SMMU_INFO = enum_pseudo_reg.define('SMMU_INFO', 0)
NON_SECURE_SAVE_ADDR = enum_pseudo_reg.define('NON_SECURE_SAVE_ADDR', 1)
SECURE_SAVE_ADDR = enum_pseudo_reg.define('SECURE_SAVE_ADDR', 2)
NON_PRIV_SAVE_ADDR = enum_pseudo_reg.define('NON_PRIV_SAVE_ADDR', 3)
COUNTER = enum_pseudo_reg.define('COUNTER', 4)
VSC_PIPE_DATA_DRAW_BASE = enum_pseudo_reg.define('VSC_PIPE_DATA_DRAW_BASE', 8)
VSC_SIZE_BASE = enum_pseudo_reg.define('VSC_SIZE_BASE', 9)
VSC_PIPE_DATA_PRIM_BASE = enum_pseudo_reg.define('VSC_PIPE_DATA_PRIM_BASE', 10)
UNK_STRM_ADDRESS = enum_pseudo_reg.define('UNK_STRM_ADDRESS', 11)
UNK_STRM_SIZE_ADDRESS = enum_pseudo_reg.define('UNK_STRM_SIZE_ADDRESS', 12)
BINDLESS_BASE_0_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_0_ADDR', 16)
BINDLESS_BASE_1_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_1_ADDR', 17)
BINDLESS_BASE_2_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_2_ADDR', 18)
BINDLESS_BASE_3_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_3_ADDR', 19)
BINDLESS_BASE_4_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_4_ADDR', 20)
BINDLESS_BASE_5_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_5_ADDR', 21)
BINDLESS_BASE_6_ADDR = enum_pseudo_reg.define('BINDLESS_BASE_6_ADDR', 22)

enum_source_type = CEnum(ctypes.c_uint32)
SOURCE_REG = enum_source_type.define('SOURCE_REG', 0)
SOURCE_SCRATCH_MEM = enum_source_type.define('SOURCE_SCRATCH_MEM', 1)

enum_compare_mode = CEnum(ctypes.c_uint32)
PRED_TEST = enum_compare_mode.define('PRED_TEST', 1)
REG_COMPARE = enum_compare_mode.define('REG_COMPARE', 2)
RENDER_MODE = enum_compare_mode.define('RENDER_MODE', 3)
REG_COMPARE_IMM = enum_compare_mode.define('REG_COMPARE_IMM', 4)
THREAD_MODE = enum_compare_mode.define('THREAD_MODE', 5)

enum_amble_type = CEnum(ctypes.c_uint32)
PREAMBLE_AMBLE_TYPE = enum_amble_type.define('PREAMBLE_AMBLE_TYPE', 0)
BIN_PREAMBLE_AMBLE_TYPE = enum_amble_type.define('BIN_PREAMBLE_AMBLE_TYPE', 1)
POSTAMBLE_AMBLE_TYPE = enum_amble_type.define('POSTAMBLE_AMBLE_TYPE', 2)
KMD_AMBLE_TYPE = enum_amble_type.define('KMD_AMBLE_TYPE', 3)

enum_reg_tracker = CEnum(ctypes.c_uint32)
TRACK_CNTL_REG = enum_reg_tracker.define('TRACK_CNTL_REG', 1)
TRACK_RENDER_CNTL = enum_reg_tracker.define('TRACK_RENDER_CNTL', 2)
UNK_EVENT_WRITE = enum_reg_tracker.define('UNK_EVENT_WRITE', 4)
TRACK_LRZ = enum_reg_tracker.define('TRACK_LRZ', 8)

enum_ts_wait_value_src = CEnum(ctypes.c_uint32)
TS_WAIT_GE_32B = enum_ts_wait_value_src.define('TS_WAIT_GE_32B', 0)
TS_WAIT_GE_64B = enum_ts_wait_value_src.define('TS_WAIT_GE_64B', 1)
TS_WAIT_GE_TIMESTAMP_SUM = enum_ts_wait_value_src.define('TS_WAIT_GE_TIMESTAMP_SUM', 2)

enum_ts_wait_type = CEnum(ctypes.c_uint32)
TS_WAIT_RAM = enum_ts_wait_type.define('TS_WAIT_RAM', 0)
TS_WAIT_ONCHIP = enum_ts_wait_type.define('TS_WAIT_ONCHIP', 1)

enum_pipe_count_op = CEnum(ctypes.c_uint32)
PIPE_CLEAR_BV_BR = enum_pipe_count_op.define('PIPE_CLEAR_BV_BR', 1)
PIPE_SET_BR_OFFSET = enum_pipe_count_op.define('PIPE_SET_BR_OFFSET', 2)
PIPE_BR_WAIT_FOR_BV = enum_pipe_count_op.define('PIPE_BR_WAIT_FOR_BV', 3)
PIPE_BV_WAIT_FOR_BR = enum_pipe_count_op.define('PIPE_BV_WAIT_FOR_BR', 4)

enum_timestamp_op = CEnum(ctypes.c_uint32)
MODIFY_TIMESTAMP_CLEAR = enum_timestamp_op.define('MODIFY_TIMESTAMP_CLEAR', 0)
MODIFY_TIMESTAMP_ADD_GLOBAL = enum_timestamp_op.define('MODIFY_TIMESTAMP_ADD_GLOBAL', 1)
MODIFY_TIMESTAMP_ADD_LOCAL = enum_timestamp_op.define('MODIFY_TIMESTAMP_ADD_LOCAL', 2)

enum_cp_thread = CEnum(ctypes.c_uint32)
CP_SET_THREAD_BR = enum_cp_thread.define('CP_SET_THREAD_BR', 1)
CP_SET_THREAD_BV = enum_cp_thread.define('CP_SET_THREAD_BV', 2)
CP_SET_THREAD_BOTH = enum_cp_thread.define('CP_SET_THREAD_BOTH', 3)

enum_cp_scope = CEnum(ctypes.c_uint32)
INTERRUPTS = enum_cp_scope.define('INTERRUPTS', 0)

enum_a6xx_tile_mode = CEnum(ctypes.c_uint32)
TILE6_LINEAR = enum_a6xx_tile_mode.define('TILE6_LINEAR', 0)
TILE6_2 = enum_a6xx_tile_mode.define('TILE6_2', 2)
TILE6_3 = enum_a6xx_tile_mode.define('TILE6_3', 3)

enum_a6xx_format = CEnum(ctypes.c_uint32)
FMT6_A8_UNORM = enum_a6xx_format.define('FMT6_A8_UNORM', 2)
FMT6_8_UNORM = enum_a6xx_format.define('FMT6_8_UNORM', 3)
FMT6_8_SNORM = enum_a6xx_format.define('FMT6_8_SNORM', 4)
FMT6_8_UINT = enum_a6xx_format.define('FMT6_8_UINT', 5)
FMT6_8_SINT = enum_a6xx_format.define('FMT6_8_SINT', 6)
FMT6_4_4_4_4_UNORM = enum_a6xx_format.define('FMT6_4_4_4_4_UNORM', 8)
FMT6_5_5_5_1_UNORM = enum_a6xx_format.define('FMT6_5_5_5_1_UNORM', 10)
FMT6_1_5_5_5_UNORM = enum_a6xx_format.define('FMT6_1_5_5_5_UNORM', 12)
FMT6_5_6_5_UNORM = enum_a6xx_format.define('FMT6_5_6_5_UNORM', 14)
FMT6_8_8_UNORM = enum_a6xx_format.define('FMT6_8_8_UNORM', 15)
FMT6_8_8_SNORM = enum_a6xx_format.define('FMT6_8_8_SNORM', 16)
FMT6_8_8_UINT = enum_a6xx_format.define('FMT6_8_8_UINT', 17)
FMT6_8_8_SINT = enum_a6xx_format.define('FMT6_8_8_SINT', 18)
FMT6_L8_A8_UNORM = enum_a6xx_format.define('FMT6_L8_A8_UNORM', 19)
FMT6_16_UNORM = enum_a6xx_format.define('FMT6_16_UNORM', 21)
FMT6_16_SNORM = enum_a6xx_format.define('FMT6_16_SNORM', 22)
FMT6_16_FLOAT = enum_a6xx_format.define('FMT6_16_FLOAT', 23)
FMT6_16_UINT = enum_a6xx_format.define('FMT6_16_UINT', 24)
FMT6_16_SINT = enum_a6xx_format.define('FMT6_16_SINT', 25)
FMT6_8_8_8_UNORM = enum_a6xx_format.define('FMT6_8_8_8_UNORM', 33)
FMT6_8_8_8_SNORM = enum_a6xx_format.define('FMT6_8_8_8_SNORM', 34)
FMT6_8_8_8_UINT = enum_a6xx_format.define('FMT6_8_8_8_UINT', 35)
FMT6_8_8_8_SINT = enum_a6xx_format.define('FMT6_8_8_8_SINT', 36)
FMT6_8_8_8_8_UNORM = enum_a6xx_format.define('FMT6_8_8_8_8_UNORM', 48)
FMT6_8_8_8_X8_UNORM = enum_a6xx_format.define('FMT6_8_8_8_X8_UNORM', 49)
FMT6_8_8_8_8_SNORM = enum_a6xx_format.define('FMT6_8_8_8_8_SNORM', 50)
FMT6_8_8_8_8_UINT = enum_a6xx_format.define('FMT6_8_8_8_8_UINT', 51)
FMT6_8_8_8_8_SINT = enum_a6xx_format.define('FMT6_8_8_8_8_SINT', 52)
FMT6_9_9_9_E5_FLOAT = enum_a6xx_format.define('FMT6_9_9_9_E5_FLOAT', 53)
FMT6_10_10_10_2_UNORM = enum_a6xx_format.define('FMT6_10_10_10_2_UNORM', 54)
FMT6_10_10_10_2_UNORM_DEST = enum_a6xx_format.define('FMT6_10_10_10_2_UNORM_DEST', 55)
FMT6_10_10_10_2_SNORM = enum_a6xx_format.define('FMT6_10_10_10_2_SNORM', 57)
FMT6_10_10_10_2_UINT = enum_a6xx_format.define('FMT6_10_10_10_2_UINT', 58)
FMT6_10_10_10_2_SINT = enum_a6xx_format.define('FMT6_10_10_10_2_SINT', 59)
FMT6_11_11_10_FLOAT = enum_a6xx_format.define('FMT6_11_11_10_FLOAT', 66)
FMT6_16_16_UNORM = enum_a6xx_format.define('FMT6_16_16_UNORM', 67)
FMT6_16_16_SNORM = enum_a6xx_format.define('FMT6_16_16_SNORM', 68)
FMT6_16_16_FLOAT = enum_a6xx_format.define('FMT6_16_16_FLOAT', 69)
FMT6_16_16_UINT = enum_a6xx_format.define('FMT6_16_16_UINT', 70)
FMT6_16_16_SINT = enum_a6xx_format.define('FMT6_16_16_SINT', 71)
FMT6_32_UNORM = enum_a6xx_format.define('FMT6_32_UNORM', 72)
FMT6_32_SNORM = enum_a6xx_format.define('FMT6_32_SNORM', 73)
FMT6_32_FLOAT = enum_a6xx_format.define('FMT6_32_FLOAT', 74)
FMT6_32_UINT = enum_a6xx_format.define('FMT6_32_UINT', 75)
FMT6_32_SINT = enum_a6xx_format.define('FMT6_32_SINT', 76)
FMT6_32_FIXED = enum_a6xx_format.define('FMT6_32_FIXED', 77)
FMT6_16_16_16_UNORM = enum_a6xx_format.define('FMT6_16_16_16_UNORM', 88)
FMT6_16_16_16_SNORM = enum_a6xx_format.define('FMT6_16_16_16_SNORM', 89)
FMT6_16_16_16_FLOAT = enum_a6xx_format.define('FMT6_16_16_16_FLOAT', 90)
FMT6_16_16_16_UINT = enum_a6xx_format.define('FMT6_16_16_16_UINT', 91)
FMT6_16_16_16_SINT = enum_a6xx_format.define('FMT6_16_16_16_SINT', 92)
FMT6_16_16_16_16_UNORM = enum_a6xx_format.define('FMT6_16_16_16_16_UNORM', 96)
FMT6_16_16_16_16_SNORM = enum_a6xx_format.define('FMT6_16_16_16_16_SNORM', 97)
FMT6_16_16_16_16_FLOAT = enum_a6xx_format.define('FMT6_16_16_16_16_FLOAT', 98)
FMT6_16_16_16_16_UINT = enum_a6xx_format.define('FMT6_16_16_16_16_UINT', 99)
FMT6_16_16_16_16_SINT = enum_a6xx_format.define('FMT6_16_16_16_16_SINT', 100)
FMT6_32_32_UNORM = enum_a6xx_format.define('FMT6_32_32_UNORM', 101)
FMT6_32_32_SNORM = enum_a6xx_format.define('FMT6_32_32_SNORM', 102)
FMT6_32_32_FLOAT = enum_a6xx_format.define('FMT6_32_32_FLOAT', 103)
FMT6_32_32_UINT = enum_a6xx_format.define('FMT6_32_32_UINT', 104)
FMT6_32_32_SINT = enum_a6xx_format.define('FMT6_32_32_SINT', 105)
FMT6_32_32_FIXED = enum_a6xx_format.define('FMT6_32_32_FIXED', 106)
FMT6_32_32_32_UNORM = enum_a6xx_format.define('FMT6_32_32_32_UNORM', 112)
FMT6_32_32_32_SNORM = enum_a6xx_format.define('FMT6_32_32_32_SNORM', 113)
FMT6_32_32_32_UINT = enum_a6xx_format.define('FMT6_32_32_32_UINT', 114)
FMT6_32_32_32_SINT = enum_a6xx_format.define('FMT6_32_32_32_SINT', 115)
FMT6_32_32_32_FLOAT = enum_a6xx_format.define('FMT6_32_32_32_FLOAT', 116)
FMT6_32_32_32_FIXED = enum_a6xx_format.define('FMT6_32_32_32_FIXED', 117)
FMT6_32_32_32_32_UNORM = enum_a6xx_format.define('FMT6_32_32_32_32_UNORM', 128)
FMT6_32_32_32_32_SNORM = enum_a6xx_format.define('FMT6_32_32_32_32_SNORM', 129)
FMT6_32_32_32_32_FLOAT = enum_a6xx_format.define('FMT6_32_32_32_32_FLOAT', 130)
FMT6_32_32_32_32_UINT = enum_a6xx_format.define('FMT6_32_32_32_32_UINT', 131)
FMT6_32_32_32_32_SINT = enum_a6xx_format.define('FMT6_32_32_32_32_SINT', 132)
FMT6_32_32_32_32_FIXED = enum_a6xx_format.define('FMT6_32_32_32_32_FIXED', 133)
FMT6_G8R8B8R8_422_UNORM = enum_a6xx_format.define('FMT6_G8R8B8R8_422_UNORM', 140)
FMT6_R8G8R8B8_422_UNORM = enum_a6xx_format.define('FMT6_R8G8R8B8_422_UNORM', 141)
FMT6_R8_G8B8_2PLANE_420_UNORM = enum_a6xx_format.define('FMT6_R8_G8B8_2PLANE_420_UNORM', 142)
FMT6_NV21 = enum_a6xx_format.define('FMT6_NV21', 143)
FMT6_R8_G8_B8_3PLANE_420_UNORM = enum_a6xx_format.define('FMT6_R8_G8_B8_3PLANE_420_UNORM', 144)
FMT6_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = enum_a6xx_format.define('FMT6_Z24_UNORM_S8_UINT_AS_R8G8B8A8', 145)
FMT6_NV12_Y = enum_a6xx_format.define('FMT6_NV12_Y', 148)
FMT6_NV12_UV = enum_a6xx_format.define('FMT6_NV12_UV', 149)
FMT6_NV12_VU = enum_a6xx_format.define('FMT6_NV12_VU', 150)
FMT6_NV12_4R = enum_a6xx_format.define('FMT6_NV12_4R', 151)
FMT6_NV12_4R_Y = enum_a6xx_format.define('FMT6_NV12_4R_Y', 152)
FMT6_NV12_4R_UV = enum_a6xx_format.define('FMT6_NV12_4R_UV', 153)
FMT6_P010 = enum_a6xx_format.define('FMT6_P010', 154)
FMT6_P010_Y = enum_a6xx_format.define('FMT6_P010_Y', 155)
FMT6_P010_UV = enum_a6xx_format.define('FMT6_P010_UV', 156)
FMT6_TP10 = enum_a6xx_format.define('FMT6_TP10', 157)
FMT6_TP10_Y = enum_a6xx_format.define('FMT6_TP10_Y', 158)
FMT6_TP10_UV = enum_a6xx_format.define('FMT6_TP10_UV', 159)
FMT6_Z24_UNORM_S8_UINT = enum_a6xx_format.define('FMT6_Z24_UNORM_S8_UINT', 160)
FMT6_ETC2_RG11_UNORM = enum_a6xx_format.define('FMT6_ETC2_RG11_UNORM', 171)
FMT6_ETC2_RG11_SNORM = enum_a6xx_format.define('FMT6_ETC2_RG11_SNORM', 172)
FMT6_ETC2_R11_UNORM = enum_a6xx_format.define('FMT6_ETC2_R11_UNORM', 173)
FMT6_ETC2_R11_SNORM = enum_a6xx_format.define('FMT6_ETC2_R11_SNORM', 174)
FMT6_ETC1 = enum_a6xx_format.define('FMT6_ETC1', 175)
FMT6_ETC2_RGB8 = enum_a6xx_format.define('FMT6_ETC2_RGB8', 176)
FMT6_ETC2_RGBA8 = enum_a6xx_format.define('FMT6_ETC2_RGBA8', 177)
FMT6_ETC2_RGB8A1 = enum_a6xx_format.define('FMT6_ETC2_RGB8A1', 178)
FMT6_DXT1 = enum_a6xx_format.define('FMT6_DXT1', 179)
FMT6_DXT3 = enum_a6xx_format.define('FMT6_DXT3', 180)
FMT6_DXT5 = enum_a6xx_format.define('FMT6_DXT5', 181)
FMT6_RGTC1_UNORM = enum_a6xx_format.define('FMT6_RGTC1_UNORM', 182)
FMT6_RGTC1_UNORM_FAST = enum_a6xx_format.define('FMT6_RGTC1_UNORM_FAST', 183)
FMT6_RGTC1_SNORM = enum_a6xx_format.define('FMT6_RGTC1_SNORM', 184)
FMT6_RGTC1_SNORM_FAST = enum_a6xx_format.define('FMT6_RGTC1_SNORM_FAST', 185)
FMT6_RGTC2_UNORM = enum_a6xx_format.define('FMT6_RGTC2_UNORM', 186)
FMT6_RGTC2_UNORM_FAST = enum_a6xx_format.define('FMT6_RGTC2_UNORM_FAST', 187)
FMT6_RGTC2_SNORM = enum_a6xx_format.define('FMT6_RGTC2_SNORM', 188)
FMT6_RGTC2_SNORM_FAST = enum_a6xx_format.define('FMT6_RGTC2_SNORM_FAST', 189)
FMT6_BPTC_UFLOAT = enum_a6xx_format.define('FMT6_BPTC_UFLOAT', 190)
FMT6_BPTC_FLOAT = enum_a6xx_format.define('FMT6_BPTC_FLOAT', 191)
FMT6_BPTC = enum_a6xx_format.define('FMT6_BPTC', 192)
FMT6_ASTC_4x4 = enum_a6xx_format.define('FMT6_ASTC_4x4', 193)
FMT6_ASTC_5x4 = enum_a6xx_format.define('FMT6_ASTC_5x4', 194)
FMT6_ASTC_5x5 = enum_a6xx_format.define('FMT6_ASTC_5x5', 195)
FMT6_ASTC_6x5 = enum_a6xx_format.define('FMT6_ASTC_6x5', 196)
FMT6_ASTC_6x6 = enum_a6xx_format.define('FMT6_ASTC_6x6', 197)
FMT6_ASTC_8x5 = enum_a6xx_format.define('FMT6_ASTC_8x5', 198)
FMT6_ASTC_8x6 = enum_a6xx_format.define('FMT6_ASTC_8x6', 199)
FMT6_ASTC_8x8 = enum_a6xx_format.define('FMT6_ASTC_8x8', 200)
FMT6_ASTC_10x5 = enum_a6xx_format.define('FMT6_ASTC_10x5', 201)
FMT6_ASTC_10x6 = enum_a6xx_format.define('FMT6_ASTC_10x6', 202)
FMT6_ASTC_10x8 = enum_a6xx_format.define('FMT6_ASTC_10x8', 203)
FMT6_ASTC_10x10 = enum_a6xx_format.define('FMT6_ASTC_10x10', 204)
FMT6_ASTC_12x10 = enum_a6xx_format.define('FMT6_ASTC_12x10', 205)
FMT6_ASTC_12x12 = enum_a6xx_format.define('FMT6_ASTC_12x12', 206)
FMT6_Z24_UINT_S8_UINT = enum_a6xx_format.define('FMT6_Z24_UINT_S8_UINT', 234)
FMT6_NONE = enum_a6xx_format.define('FMT6_NONE', 255)

enum_a6xx_polygon_mode = CEnum(ctypes.c_uint32)
POLYMODE6_POINTS = enum_a6xx_polygon_mode.define('POLYMODE6_POINTS', 1)
POLYMODE6_LINES = enum_a6xx_polygon_mode.define('POLYMODE6_LINES', 2)
POLYMODE6_TRIANGLES = enum_a6xx_polygon_mode.define('POLYMODE6_TRIANGLES', 3)

enum_a6xx_depth_format = CEnum(ctypes.c_uint32)
DEPTH6_NONE = enum_a6xx_depth_format.define('DEPTH6_NONE', 0)
DEPTH6_16 = enum_a6xx_depth_format.define('DEPTH6_16', 1)
DEPTH6_24_8 = enum_a6xx_depth_format.define('DEPTH6_24_8', 2)
DEPTH6_32 = enum_a6xx_depth_format.define('DEPTH6_32', 4)

enum_a6xx_shader_id = CEnum(ctypes.c_uint32)
A6XX_TP0_TMO_DATA = enum_a6xx_shader_id.define('A6XX_TP0_TMO_DATA', 9)
A6XX_TP0_SMO_DATA = enum_a6xx_shader_id.define('A6XX_TP0_SMO_DATA', 10)
A6XX_TP0_MIPMAP_BASE_DATA = enum_a6xx_shader_id.define('A6XX_TP0_MIPMAP_BASE_DATA', 11)
A6XX_TP1_TMO_DATA = enum_a6xx_shader_id.define('A6XX_TP1_TMO_DATA', 25)
A6XX_TP1_SMO_DATA = enum_a6xx_shader_id.define('A6XX_TP1_SMO_DATA', 26)
A6XX_TP1_MIPMAP_BASE_DATA = enum_a6xx_shader_id.define('A6XX_TP1_MIPMAP_BASE_DATA', 27)
A6XX_SP_INST_DATA = enum_a6xx_shader_id.define('A6XX_SP_INST_DATA', 41)
A6XX_SP_LB_0_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_0_DATA', 42)
A6XX_SP_LB_1_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_1_DATA', 43)
A6XX_SP_LB_2_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_2_DATA', 44)
A6XX_SP_LB_3_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_3_DATA', 45)
A6XX_SP_LB_4_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_4_DATA', 46)
A6XX_SP_LB_5_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_5_DATA', 47)
A6XX_SP_CB_BINDLESS_DATA = enum_a6xx_shader_id.define('A6XX_SP_CB_BINDLESS_DATA', 48)
A6XX_SP_CB_LEGACY_DATA = enum_a6xx_shader_id.define('A6XX_SP_CB_LEGACY_DATA', 49)
A6XX_SP_GFX_UAV_BASE_DATA = enum_a6xx_shader_id.define('A6XX_SP_GFX_UAV_BASE_DATA', 50)
A6XX_SP_INST_TAG = enum_a6xx_shader_id.define('A6XX_SP_INST_TAG', 51)
A6XX_SP_CB_BINDLESS_TAG = enum_a6xx_shader_id.define('A6XX_SP_CB_BINDLESS_TAG', 52)
A6XX_SP_TMO_UMO_TAG = enum_a6xx_shader_id.define('A6XX_SP_TMO_UMO_TAG', 53)
A6XX_SP_SMO_TAG = enum_a6xx_shader_id.define('A6XX_SP_SMO_TAG', 54)
A6XX_SP_STATE_DATA = enum_a6xx_shader_id.define('A6XX_SP_STATE_DATA', 55)
A6XX_HLSQ_CHUNK_CVS_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_CHUNK_CVS_RAM', 73)
A6XX_HLSQ_CHUNK_CPS_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_CHUNK_CPS_RAM', 74)
A6XX_HLSQ_CHUNK_CVS_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_CHUNK_CVS_RAM_TAG', 75)
A6XX_HLSQ_CHUNK_CPS_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_CHUNK_CPS_RAM_TAG', 76)
A6XX_HLSQ_ICB_CVS_CB_BASE_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_ICB_CVS_CB_BASE_TAG', 77)
A6XX_HLSQ_ICB_CPS_CB_BASE_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_ICB_CPS_CB_BASE_TAG', 78)
A6XX_HLSQ_CVS_MISC_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_CVS_MISC_RAM', 80)
A6XX_HLSQ_CPS_MISC_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_CPS_MISC_RAM', 81)
A6XX_HLSQ_INST_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_INST_RAM', 82)
A6XX_HLSQ_GFX_CVS_CONST_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_GFX_CVS_CONST_RAM', 83)
A6XX_HLSQ_GFX_CPS_CONST_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_GFX_CPS_CONST_RAM', 84)
A6XX_HLSQ_CVS_MISC_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_CVS_MISC_RAM_TAG', 85)
A6XX_HLSQ_CPS_MISC_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_CPS_MISC_RAM_TAG', 86)
A6XX_HLSQ_INST_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_INST_RAM_TAG', 87)
A6XX_HLSQ_GFX_CVS_CONST_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_GFX_CVS_CONST_RAM_TAG', 88)
A6XX_HLSQ_GFX_CPS_CONST_RAM_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_GFX_CPS_CONST_RAM_TAG', 89)
A6XX_HLSQ_PWR_REST_RAM = enum_a6xx_shader_id.define('A6XX_HLSQ_PWR_REST_RAM', 90)
A6XX_HLSQ_PWR_REST_TAG = enum_a6xx_shader_id.define('A6XX_HLSQ_PWR_REST_TAG', 91)
A6XX_HLSQ_DATAPATH_META = enum_a6xx_shader_id.define('A6XX_HLSQ_DATAPATH_META', 96)
A6XX_HLSQ_FRONTEND_META = enum_a6xx_shader_id.define('A6XX_HLSQ_FRONTEND_META', 97)
A6XX_HLSQ_INDIRECT_META = enum_a6xx_shader_id.define('A6XX_HLSQ_INDIRECT_META', 98)
A6XX_HLSQ_BACKEND_META = enum_a6xx_shader_id.define('A6XX_HLSQ_BACKEND_META', 99)
A6XX_SP_LB_6_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_6_DATA', 112)
A6XX_SP_LB_7_DATA = enum_a6xx_shader_id.define('A6XX_SP_LB_7_DATA', 113)
A6XX_HLSQ_INST_RAM_1 = enum_a6xx_shader_id.define('A6XX_HLSQ_INST_RAM_1', 115)

enum_a6xx_debugbus_id = CEnum(ctypes.c_uint32)
A6XX_DBGBUS_CP = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CP', 1)
A6XX_DBGBUS_RBBM = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RBBM', 2)
A6XX_DBGBUS_VBIF = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VBIF', 3)
A6XX_DBGBUS_HLSQ = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_HLSQ', 4)
A6XX_DBGBUS_UCHE = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_UCHE', 5)
A6XX_DBGBUS_DPM = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_DPM', 6)
A6XX_DBGBUS_TESS = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TESS', 7)
A6XX_DBGBUS_PC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_PC', 8)
A6XX_DBGBUS_VFDP = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFDP', 9)
A6XX_DBGBUS_VPC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VPC', 10)
A6XX_DBGBUS_TSE = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TSE', 11)
A6XX_DBGBUS_RAS = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RAS', 12)
A6XX_DBGBUS_VSC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VSC', 13)
A6XX_DBGBUS_COM = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_COM', 14)
A6XX_DBGBUS_LRZ = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_LRZ', 16)
A6XX_DBGBUS_A2D = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_A2D', 17)
A6XX_DBGBUS_CCUFCHE = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CCUFCHE', 18)
A6XX_DBGBUS_GMU_CX = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_GMU_CX', 19)
A6XX_DBGBUS_RBP = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RBP', 20)
A6XX_DBGBUS_DCS = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_DCS', 21)
A6XX_DBGBUS_DBGC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_DBGC', 22)
A6XX_DBGBUS_CX = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CX', 23)
A6XX_DBGBUS_GMU_GX = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_GMU_GX', 24)
A6XX_DBGBUS_TPFCHE = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPFCHE', 25)
A6XX_DBGBUS_GBIF_GX = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_GBIF_GX', 26)
A6XX_DBGBUS_GPC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_GPC', 29)
A6XX_DBGBUS_LARC = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_LARC', 30)
A6XX_DBGBUS_HLSQ_SPTP = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_HLSQ_SPTP', 31)
A6XX_DBGBUS_RB_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RB_0', 32)
A6XX_DBGBUS_RB_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RB_1', 33)
A6XX_DBGBUS_RB_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_RB_2', 34)
A6XX_DBGBUS_UCHE_WRAPPER = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_UCHE_WRAPPER', 36)
A6XX_DBGBUS_CCU_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CCU_0', 40)
A6XX_DBGBUS_CCU_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CCU_1', 41)
A6XX_DBGBUS_CCU_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_CCU_2', 42)
A6XX_DBGBUS_VFD_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_0', 56)
A6XX_DBGBUS_VFD_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_1', 57)
A6XX_DBGBUS_VFD_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_2', 58)
A6XX_DBGBUS_VFD_3 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_3', 59)
A6XX_DBGBUS_VFD_4 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_4', 60)
A6XX_DBGBUS_VFD_5 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_VFD_5', 61)
A6XX_DBGBUS_SP_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SP_0', 64)
A6XX_DBGBUS_SP_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SP_1', 65)
A6XX_DBGBUS_SP_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SP_2', 66)
A6XX_DBGBUS_TPL1_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_0', 72)
A6XX_DBGBUS_TPL1_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_1', 73)
A6XX_DBGBUS_TPL1_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_2', 74)
A6XX_DBGBUS_TPL1_3 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_3', 75)
A6XX_DBGBUS_TPL1_4 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_4', 76)
A6XX_DBGBUS_TPL1_5 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_TPL1_5', 77)
A6XX_DBGBUS_SPTP_0 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_0', 88)
A6XX_DBGBUS_SPTP_1 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_1', 89)
A6XX_DBGBUS_SPTP_2 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_2', 90)
A6XX_DBGBUS_SPTP_3 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_3', 91)
A6XX_DBGBUS_SPTP_4 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_4', 92)
A6XX_DBGBUS_SPTP_5 = enum_a6xx_debugbus_id.define('A6XX_DBGBUS_SPTP_5', 93)

enum_a6xx_2d_ifmt = CEnum(ctypes.c_uint32)
R2D_INT32 = enum_a6xx_2d_ifmt.define('R2D_INT32', 7)
R2D_INT16 = enum_a6xx_2d_ifmt.define('R2D_INT16', 6)
R2D_INT8 = enum_a6xx_2d_ifmt.define('R2D_INT8', 5)
R2D_FLOAT32 = enum_a6xx_2d_ifmt.define('R2D_FLOAT32', 4)
R2D_FLOAT16 = enum_a6xx_2d_ifmt.define('R2D_FLOAT16', 3)
R2D_SNORM8 = enum_a6xx_2d_ifmt.define('R2D_SNORM8', 2)
R2D_UNORM8_SRGB = enum_a6xx_2d_ifmt.define('R2D_UNORM8_SRGB', 1)
R2D_UNORM8 = enum_a6xx_2d_ifmt.define('R2D_UNORM8', 0)

enum_a6xx_tex_type = CEnum(ctypes.c_uint32)
A6XX_TEX_1D = enum_a6xx_tex_type.define('A6XX_TEX_1D', 0)
A6XX_TEX_2D = enum_a6xx_tex_type.define('A6XX_TEX_2D', 1)
A6XX_TEX_CUBE = enum_a6xx_tex_type.define('A6XX_TEX_CUBE', 2)
A6XX_TEX_3D = enum_a6xx_tex_type.define('A6XX_TEX_3D', 3)
A6XX_TEX_BUFFER = enum_a6xx_tex_type.define('A6XX_TEX_BUFFER', 4)
A6XX_TEX_IMG_BUFFER = enum_a6xx_tex_type.define('A6XX_TEX_IMG_BUFFER', 5)

enum_a6xx_ztest_mode = CEnum(ctypes.c_uint32)
A6XX_EARLY_Z = enum_a6xx_ztest_mode.define('A6XX_EARLY_Z', 0)
A6XX_LATE_Z = enum_a6xx_ztest_mode.define('A6XX_LATE_Z', 1)
A6XX_EARLY_Z_LATE_Z = enum_a6xx_ztest_mode.define('A6XX_EARLY_Z_LATE_Z', 2)
A6XX_INVALID_ZTEST = enum_a6xx_ztest_mode.define('A6XX_INVALID_ZTEST', 3)

enum_a6xx_tess_spacing = CEnum(ctypes.c_uint32)
TESS_EQUAL = enum_a6xx_tess_spacing.define('TESS_EQUAL', 0)
TESS_FRACTIONAL_ODD = enum_a6xx_tess_spacing.define('TESS_FRACTIONAL_ODD', 2)
TESS_FRACTIONAL_EVEN = enum_a6xx_tess_spacing.define('TESS_FRACTIONAL_EVEN', 3)

enum_a6xx_tess_output = CEnum(ctypes.c_uint32)
TESS_POINTS = enum_a6xx_tess_output.define('TESS_POINTS', 0)
TESS_LINES = enum_a6xx_tess_output.define('TESS_LINES', 1)
TESS_CW_TRIS = enum_a6xx_tess_output.define('TESS_CW_TRIS', 2)
TESS_CCW_TRIS = enum_a6xx_tess_output.define('TESS_CCW_TRIS', 3)

enum_a6xx_tex_filter = CEnum(ctypes.c_uint32)
A6XX_TEX_NEAREST = enum_a6xx_tex_filter.define('A6XX_TEX_NEAREST', 0)
A6XX_TEX_LINEAR = enum_a6xx_tex_filter.define('A6XX_TEX_LINEAR', 1)
A6XX_TEX_ANISO = enum_a6xx_tex_filter.define('A6XX_TEX_ANISO', 2)
A6XX_TEX_CUBIC = enum_a6xx_tex_filter.define('A6XX_TEX_CUBIC', 3)

enum_a6xx_tex_clamp = CEnum(ctypes.c_uint32)
A6XX_TEX_REPEAT = enum_a6xx_tex_clamp.define('A6XX_TEX_REPEAT', 0)
A6XX_TEX_CLAMP_TO_EDGE = enum_a6xx_tex_clamp.define('A6XX_TEX_CLAMP_TO_EDGE', 1)
A6XX_TEX_MIRROR_REPEAT = enum_a6xx_tex_clamp.define('A6XX_TEX_MIRROR_REPEAT', 2)
A6XX_TEX_CLAMP_TO_BORDER = enum_a6xx_tex_clamp.define('A6XX_TEX_CLAMP_TO_BORDER', 3)
A6XX_TEX_MIRROR_CLAMP = enum_a6xx_tex_clamp.define('A6XX_TEX_MIRROR_CLAMP', 4)

enum_a6xx_tex_aniso = CEnum(ctypes.c_uint32)
A6XX_TEX_ANISO_1 = enum_a6xx_tex_aniso.define('A6XX_TEX_ANISO_1', 0)
A6XX_TEX_ANISO_2 = enum_a6xx_tex_aniso.define('A6XX_TEX_ANISO_2', 1)
A6XX_TEX_ANISO_4 = enum_a6xx_tex_aniso.define('A6XX_TEX_ANISO_4', 2)
A6XX_TEX_ANISO_8 = enum_a6xx_tex_aniso.define('A6XX_TEX_ANISO_8', 3)
A6XX_TEX_ANISO_16 = enum_a6xx_tex_aniso.define('A6XX_TEX_ANISO_16', 4)

enum_a6xx_reduction_mode = CEnum(ctypes.c_uint32)
A6XX_REDUCTION_MODE_AVERAGE = enum_a6xx_reduction_mode.define('A6XX_REDUCTION_MODE_AVERAGE', 0)
A6XX_REDUCTION_MODE_MIN = enum_a6xx_reduction_mode.define('A6XX_REDUCTION_MODE_MIN', 1)
A6XX_REDUCTION_MODE_MAX = enum_a6xx_reduction_mode.define('A6XX_REDUCTION_MODE_MAX', 2)

enum_a6xx_fast_border_color = CEnum(ctypes.c_uint32)
A6XX_BORDER_COLOR_0_0_0_0 = enum_a6xx_fast_border_color.define('A6XX_BORDER_COLOR_0_0_0_0', 0)
A6XX_BORDER_COLOR_0_0_0_1 = enum_a6xx_fast_border_color.define('A6XX_BORDER_COLOR_0_0_0_1', 1)
A6XX_BORDER_COLOR_1_1_1_0 = enum_a6xx_fast_border_color.define('A6XX_BORDER_COLOR_1_1_1_0', 2)
A6XX_BORDER_COLOR_1_1_1_1 = enum_a6xx_fast_border_color.define('A6XX_BORDER_COLOR_1_1_1_1', 3)

enum_a6xx_tex_swiz = CEnum(ctypes.c_uint32)
A6XX_TEX_X = enum_a6xx_tex_swiz.define('A6XX_TEX_X', 0)
A6XX_TEX_Y = enum_a6xx_tex_swiz.define('A6XX_TEX_Y', 1)
A6XX_TEX_Z = enum_a6xx_tex_swiz.define('A6XX_TEX_Z', 2)
A6XX_TEX_W = enum_a6xx_tex_swiz.define('A6XX_TEX_W', 3)
A6XX_TEX_ZERO = enum_a6xx_tex_swiz.define('A6XX_TEX_ZERO', 4)
A6XX_TEX_ONE = enum_a6xx_tex_swiz.define('A6XX_TEX_ONE', 5)

NIR_DEBUG_CLONE = (1 << 0)
NIR_DEBUG_SERIALIZE = (1 << 1)
NIR_DEBUG_NOVALIDATE = (1 << 2)
NIR_DEBUG_EXTENDED_VALIDATION = (1 << 3)
NIR_DEBUG_TGSI = (1 << 4)
NIR_DEBUG_PRINT_VS = (1 << 5)
NIR_DEBUG_PRINT_TCS = (1 << 6)
NIR_DEBUG_PRINT_TES = (1 << 7)
NIR_DEBUG_PRINT_GS = (1 << 8)
NIR_DEBUG_PRINT_FS = (1 << 9)
NIR_DEBUG_PRINT_CS = (1 << 10)
NIR_DEBUG_PRINT_TS = (1 << 11)
NIR_DEBUG_PRINT_MS = (1 << 12)
NIR_DEBUG_PRINT_RGS = (1 << 13)
NIR_DEBUG_PRINT_AHS = (1 << 14)
NIR_DEBUG_PRINT_CHS = (1 << 15)
NIR_DEBUG_PRINT_MHS = (1 << 16)
NIR_DEBUG_PRINT_IS = (1 << 17)
NIR_DEBUG_PRINT_CBS = (1 << 18)
NIR_DEBUG_PRINT_KS = (1 << 19)
NIR_DEBUG_PRINT_NO_INLINE_CONSTS = (1 << 20)
NIR_DEBUG_PRINT_INTERNAL = (1 << 21)
NIR_DEBUG_PRINT_PASS_FLAGS = (1 << 22)
NIR_DEBUG_INVALIDATE_METADATA = (1 << 23)
NIR_DEBUG_PRINT_STRUCT_DECLS = (1 << 24)
NIR_DEBUG_PRINT = (NIR_DEBUG_PRINT_VS | NIR_DEBUG_PRINT_TCS | NIR_DEBUG_PRINT_TES | NIR_DEBUG_PRINT_GS | NIR_DEBUG_PRINT_FS | NIR_DEBUG_PRINT_CS | NIR_DEBUG_PRINT_TS | NIR_DEBUG_PRINT_MS | NIR_DEBUG_PRINT_RGS | NIR_DEBUG_PRINT_AHS | NIR_DEBUG_PRINT_CHS | NIR_DEBUG_PRINT_MHS | NIR_DEBUG_PRINT_IS | NIR_DEBUG_PRINT_CBS | NIR_DEBUG_PRINT_KS)
NIR_FALSE = 0
NIR_TRUE = (~0)
NIR_MAX_VEC_COMPONENTS = 16
NIR_MAX_MATRIX_COLUMNS = 4
NIR_STREAM_PACKED = (1 << 8)
NIR_VARIABLE_NO_INDEX = ~0
nir_foreach_variable_in_list = lambda var,var_list: foreach_list_typed(nir_variable, var, node, var_list)
nir_foreach_variable_in_list_safe = lambda var,var_list: foreach_list_typed_safe(nir_variable, var, node, var_list)
nir_foreach_shader_in_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_shader_in)
nir_foreach_shader_in_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_in)
nir_foreach_shader_out_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_shader_out)
nir_foreach_shader_out_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_out)
nir_foreach_uniform_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_uniform)
nir_foreach_uniform_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_uniform)
nir_foreach_image_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_image)
nir_foreach_image_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_image)
NIR_SRC_PARENT_IS_IF = (0x1)
NIR_ALU_MAX_INPUTS = NIR_MAX_VEC_COMPONENTS
NIR_INTRINSIC_MAX_CONST_INDEX = 8
NIR_ALIGN_MUL_MAX = 0x40000000
NIR_INTRINSIC_MAX_INPUTS = 11
nir_log_shadere = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), NULL)
nir_log_shaderw = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_WARN, (MESA_LOG_TAG), (s), NULL)
nir_log_shaderi = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_INFO, (MESA_LOG_TAG), (s), NULL)
nir_log_shader_annotated = lambda s,annotations: nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), annotations)
NIR_STRINGIZE = lambda x: NIR_STRINGIZE_INNER(x)
NVIDIA_VENDOR_ID = 0x10de
NAK_SUBGROUP_SIZE = 32
NAK_QMD_ALIGN_B = 256
NAK_MAX_QMD_SIZE_B = 384
NAK_MAX_QMD_DWORDS = (NAK_MAX_QMD_SIZE_B / 4)
LP_MAX_VECTOR_WIDTH = 512
LP_MIN_VECTOR_ALIGN = 64
LP_MAX_VECTOR_LENGTH = (LP_MAX_VECTOR_WIDTH/8)
LP_RESV_FUNC_ARGS = 2
LP_JIT_TEXTURE_SAMPLE_STRIDE = 15
lp_jit_resources_constants = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_CONSTANTS, "constants")
lp_jit_resources_ssbos = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_SSBOS, "ssbos")
lp_jit_resources_textures = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_TEXTURES, "textures")
lp_jit_resources_samplers = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_SAMPLERS, "samplers")
lp_jit_resources_images = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_IMAGES, "images")
lp_jit_vertex_header_id = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_VERTEX_ID, "id")
lp_jit_vertex_header_clip_pos = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_CLIP_POS, "clip_pos")
lp_jit_vertex_header_data = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_DATA, "data")
LP_MAX_TEX_FUNC_ARGS = 32
A6XX_CCU_DEPTH_SIZE = (64 * 1024)
A6XX_CCU_GMEM_COLOR_SIZE = (16 * 1024)
dword_offsetof = lambda type,name: DIV_ROUND_UP(offsetof(type, name), 4)
dword_sizeof = lambda type: DIV_ROUND_UP(sizeof(type), 4)
IR3_DP_CS = lambda name: dword_offsetof(struct_ir3_driver_params_cs, name)
IR3_DP_VS = lambda name: dword_offsetof(struct_ir3_driver_params_vs, name)
IR3_DP_TCS = lambda name: dword_offsetof(struct_ir3_driver_params_tcs, name)
IR3_DP_FS = lambda name: dword_offsetof(struct_ir3_driver_params_fs, name)
IR3_MAX_SHADER_BUFFERS = 32
IR3_MAX_SHADER_IMAGES = 32
IR3_MAX_SO_BUFFERS = 4
IR3_MAX_SO_STREAMS = 4
IR3_MAX_SO_OUTPUTS = 128
IR3_MAX_UBO_PUSH_RANGES = 32
IR3_MAX_SAMPLER_PREFETCH = 4
IR3_SAMPLER_PREFETCH_CMD = 0x4
IR3_SAMPLER_BINDLESS_PREFETCH_CMD = 0x6
IR3_TESS_NONE = 0
IR3_TESS_QUADS = 1
IR3_TESS_TRIANGLES = 2
IR3_TESS_ISOLINES = 3
UAV_INVALID = 0xff
UAV_SSBO = 0x80
HALF_REG_ID = 0x100
gc_alloc = lambda ctx,type,count: gc_alloc_size(ctx, sizeof(type) * (count), alignof(type))
gc_zalloc = lambda ctx,type,count: gc_zalloc_size(ctx, sizeof(type) * (count), alignof(type))
gc_alloc_zla = lambda ctx,type,type2,count: gc_alloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))
gc_zalloc_zla = lambda ctx,type,type2,count: gc_zalloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))
DECLARE_RALLOC_CXX_OPERATORS = lambda type: DECLARE_RALLOC_CXX_OPERATORS_TEMPLATE(type, ralloc_size)
DECLARE_RZALLOC_CXX_OPERATORS = lambda type: DECLARE_RALLOC_CXX_OPERATORS_TEMPLATE(type, rzalloc_size)
DECLARE_LINEAR_ALLOC_CXX_OPERATORS = lambda type: DECLARE_LINEAR_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_alloc_child)
DECLARE_LINEAR_ZALLOC_CXX_OPERATORS = lambda type: DECLARE_LINEAR_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_zalloc_child)
ISA_GPU_ID = lambda: ir3_isa_get_gpu_id(scope)
__struct__cast = lambda X: (struct_X)
A6XX_RBBM_INT_0_MASK_RBBM_GPU_IDLE = 0x00000001
A6XX_RBBM_INT_0_MASK_CP_AHB_ERROR = 0x00000002
A6XX_RBBM_INT_0_MASK_CP_IPC_INTR_0 = 0x00000010
A6XX_RBBM_INT_0_MASK_CP_IPC_INTR_1 = 0x00000020
A6XX_RBBM_INT_0_MASK_RBBM_ATB_ASYNCFIFO_OVERFLOW = 0x00000040
A6XX_RBBM_INT_0_MASK_RBBM_GPC_ERROR = 0x00000080
A6XX_RBBM_INT_0_MASK_CP_SW = 0x00000100
A6XX_RBBM_INT_0_MASK_CP_HW_ERROR = 0x00000200
A6XX_RBBM_INT_0_MASK_CP_CCU_FLUSH_DEPTH_TS = 0x00000400
A6XX_RBBM_INT_0_MASK_CP_CCU_FLUSH_COLOR_TS = 0x00000800
A6XX_RBBM_INT_0_MASK_CP_CCU_RESOLVE_TS = 0x00001000
A6XX_RBBM_INT_0_MASK_CP_IB2 = 0x00002000
A6XX_RBBM_INT_0_MASK_CP_IB1 = 0x00004000
A6XX_RBBM_INT_0_MASK_CP_RB = 0x00008000
A6XX_RBBM_INT_0_MASK_PM4CPINTERRUPT = 0x00008000
A6XX_RBBM_INT_0_MASK_PM4CPINTERRUPTLPAC = 0x00010000
A6XX_RBBM_INT_0_MASK_CP_RB_DONE_TS = 0x00020000
A6XX_RBBM_INT_0_MASK_CP_WT_DONE_TS = 0x00040000
A6XX_RBBM_INT_0_MASK_CP_CACHE_FLUSH_TS = 0x00100000
A6XX_RBBM_INT_0_MASK_CP_CACHE_FLUSH_TS_LPAC = 0x00200000
A6XX_RBBM_INT_0_MASK_RBBM_ATB_BUS_OVERFLOW = 0x00400000
A6XX_RBBM_INT_0_MASK_RBBM_HANG_DETECT = 0x00800000
A6XX_RBBM_INT_0_MASK_UCHE_OOB_ACCESS = 0x01000000
A6XX_RBBM_INT_0_MASK_UCHE_TRAP_INTR = 0x02000000
A6XX_RBBM_INT_0_MASK_DEBBUS_INTR_0 = 0x04000000
A6XX_RBBM_INT_0_MASK_DEBBUS_INTR_1 = 0x08000000
A6XX_RBBM_INT_0_MASK_TSBWRITEERROR = 0x10000000
A6XX_RBBM_INT_0_MASK_SWFUSEVIOLATION = 0x20000000
A6XX_RBBM_INT_0_MASK_ISDB_CPU_IRQ = 0x40000000
A6XX_RBBM_INT_0_MASK_ISDB_UNDER_DEBUG = 0x80000000
A6XX_CP_INT_CP_OPCODE_ERROR = 0x00000001
A6XX_CP_INT_CP_UCODE_ERROR = 0x00000002
A6XX_CP_INT_CP_HW_FAULT_ERROR = 0x00000004
A6XX_CP_INT_CP_REGISTER_PROTECTION_ERROR = 0x00000010
A6XX_CP_INT_CP_AHB_ERROR = 0x00000020
A6XX_CP_INT_CP_VSD_PARITY_ERROR = 0x00000040
A6XX_CP_INT_CP_ILLEGAL_INSTR_ERROR = 0x00000080
A6XX_CP_INT_CP_OPCODE_ERROR_LPAC = 0x00000100
A6XX_CP_INT_CP_UCODE_ERROR_LPAC = 0x00000200
A6XX_CP_INT_CP_HW_FAULT_ERROR_LPAC = 0x00000400
A6XX_CP_INT_CP_REGISTER_PROTECTION_ERROR_LPAC = 0x00000800
A6XX_CP_INT_CP_ILLEGAL_INSTR_ERROR_LPAC = 0x00001000
A6XX_CP_INT_CP_OPCODE_ERROR_BV = 0x00002000
A6XX_CP_INT_CP_UCODE_ERROR_BV = 0x00004000
A6XX_CP_INT_CP_HW_FAULT_ERROR_BV = 0x00008000
A6XX_CP_INT_CP_REGISTER_PROTECTION_ERROR_BV = 0x00010000
A6XX_CP_INT_CP_ILLEGAL_INSTR_ERROR_BV = 0x00020000
REG_A6XX_CP_RB_BASE = 0x00000800
REG_A6XX_CP_RB_CNTL = 0x00000802
REG_A6XX_CP_RB_RPTR_ADDR = 0x00000804
REG_A6XX_CP_RB_RPTR = 0x00000806
REG_A6XX_CP_RB_WPTR = 0x00000807
REG_A6XX_CP_SQE_CNTL = 0x00000808
REG_A6XX_CP_CP2GMU_STATUS = 0x00000812
A6XX_CP_CP2GMU_STATUS_IFPC = 0x00000001
REG_A6XX_CP_HW_FAULT = 0x00000821
REG_A6XX_CP_INTERRUPT_STATUS = 0x00000823
REG_A6XX_CP_PROTECT_STATUS = 0x00000824
REG_A6XX_CP_STATUS_1 = 0x00000825
REG_A6XX_CP_SQE_INSTR_BASE = 0x00000830
REG_A6XX_CP_MISC_CNTL = 0x00000840
REG_A6XX_CP_APRIV_CNTL = 0x00000844
A6XX_CP_APRIV_CNTL_CDWRITE = 0x00000040
A6XX_CP_APRIV_CNTL_CDREAD = 0x00000020
A6XX_CP_APRIV_CNTL_RBRPWB = 0x00000008
A6XX_CP_APRIV_CNTL_RBPRIVLEVEL = 0x00000004
A6XX_CP_APRIV_CNTL_RBFETCH = 0x00000002
A6XX_CP_APRIV_CNTL_ICACHE = 0x00000001
REG_A6XX_CP_PREEMPT_THRESHOLD = 0x000008c0
REG_A6XX_CP_ROQ_THRESHOLDS_1 = 0x000008c1
A6XX_CP_ROQ_THRESHOLDS_1_MRB_START__MASK = 0x000000ff
A6XX_CP_ROQ_THRESHOLDS_1_MRB_START__SHIFT = 0
A6XX_CP_ROQ_THRESHOLDS_1_VSD_START__MASK = 0x0000ff00
A6XX_CP_ROQ_THRESHOLDS_1_VSD_START__SHIFT = 8
A6XX_CP_ROQ_THRESHOLDS_1_IB1_START__MASK = 0x00ff0000
A6XX_CP_ROQ_THRESHOLDS_1_IB1_START__SHIFT = 16
A6XX_CP_ROQ_THRESHOLDS_1_IB2_START__MASK = 0xff000000
A6XX_CP_ROQ_THRESHOLDS_1_IB2_START__SHIFT = 24
REG_A6XX_CP_ROQ_THRESHOLDS_2 = 0x000008c2
A6XX_CP_ROQ_THRESHOLDS_2_SDS_START__MASK = 0x000001ff
A6XX_CP_ROQ_THRESHOLDS_2_SDS_START__SHIFT = 0
A6XX_CP_ROQ_THRESHOLDS_2_ROQ_SIZE__MASK = 0xffff0000
A6XX_CP_ROQ_THRESHOLDS_2_ROQ_SIZE__SHIFT = 16
REG_A6XX_CP_MEM_POOL_SIZE = 0x000008c3
REG_A6XX_CP_CHICKEN_DBG = 0x00000841
REG_A6XX_CP_ADDR_MODE_CNTL = 0x00000842
REG_A6XX_CP_DBG_ECO_CNTL = 0x00000843
REG_A6XX_CP_PROTECT_CNTL = 0x0000084f
A6XX_CP_PROTECT_CNTL_LAST_SPAN_INF_RANGE = 0x00000008
A6XX_CP_PROTECT_CNTL_ACCESS_FAULT_ON_VIOL_EN = 0x00000002
A6XX_CP_PROTECT_CNTL_ACCESS_PROT_EN = 0x00000001
REG_A6XX_CP_SCRATCH = lambda i0: (0x00000883 + 0x1*i0 )
REG_A6XX_CP_PROTECT = lambda i0: (0x00000850 + 0x1*i0 )
A6XX_CP_PROTECT_REG_BASE_ADDR__MASK = 0x0003ffff
A6XX_CP_PROTECT_REG_BASE_ADDR__SHIFT = 0
A6XX_CP_PROTECT_REG_MASK_LEN__MASK = 0x7ffc0000
A6XX_CP_PROTECT_REG_MASK_LEN__SHIFT = 18
A6XX_CP_PROTECT_REG_READ = 0x80000000
REG_A6XX_CP_CONTEXT_SWITCH_CNTL = 0x000008a0
A6XX_CP_CONTEXT_SWITCH_CNTL_STOP = 0x00000001
A6XX_CP_CONTEXT_SWITCH_CNTL_LEVEL__MASK = 0x000000c0
A6XX_CP_CONTEXT_SWITCH_CNTL_LEVEL__SHIFT = 6
A6XX_CP_CONTEXT_SWITCH_CNTL_USES_GMEM = 0x00000100
A6XX_CP_CONTEXT_SWITCH_CNTL_SKIP_SAVE_RESTORE = 0x00000200
REG_A6XX_CP_CONTEXT_SWITCH_SMMU_INFO = 0x000008a1
REG_A6XX_CP_CONTEXT_SWITCH_PRIV_NON_SECURE_RESTORE_ADDR = 0x000008a3
REG_A6XX_CP_CONTEXT_SWITCH_PRIV_SECURE_RESTORE_ADDR = 0x000008a5
REG_A6XX_CP_CONTEXT_SWITCH_NON_PRIV_RESTORE_ADDR = 0x000008a7
REG_A7XX_CP_CONTEXT_SWITCH_LEVEL_STATUS = 0x000008ab
REG_A6XX_CP_PERFCTR_CP_SEL = lambda i0: (0x000008d0 + 0x1*i0 )
REG_A7XX_CP_BV_PERFCTR_CP_SEL = lambda i0: (0x000008e0 + 0x1*i0 )
REG_A6XX_CP_CRASH_DUMP_SCRIPT_BASE = 0x00000900
REG_A6XX_CP_CRASH_DUMP_CNTL = 0x00000902
REG_A6XX_CP_CRASH_DUMP_STATUS = 0x00000903
REG_A6XX_CP_SQE_STAT_ADDR = 0x00000908
REG_A6XX_CP_SQE_STAT_DATA = 0x00000909
REG_A6XX_CP_DRAW_STATE_ADDR = 0x0000090a
REG_A6XX_CP_DRAW_STATE_DATA = 0x0000090b
REG_A6XX_CP_ROQ_DBG_ADDR = 0x0000090c
REG_A6XX_CP_ROQ_DBG_DATA = 0x0000090d
REG_A6XX_CP_MEM_POOL_DBG_ADDR = 0x0000090e
REG_A6XX_CP_MEM_POOL_DBG_DATA = 0x0000090f
REG_A6XX_CP_SQE_UCODE_DBG_ADDR = 0x00000910
REG_A6XX_CP_SQE_UCODE_DBG_DATA = 0x00000911
REG_A6XX_CP_IB1_BASE = 0x00000928
REG_A6XX_CP_IB1_REM_SIZE = 0x0000092a
REG_A6XX_CP_IB2_BASE = 0x0000092b
REG_A6XX_CP_IB2_REM_SIZE = 0x0000092d
REG_A6XX_CP_SDS_BASE = 0x0000092e
REG_A6XX_CP_SDS_REM_SIZE = 0x00000930
REG_A6XX_CP_MRB_BASE = 0x00000931
REG_A6XX_CP_MRB_REM_SIZE = 0x00000933
REG_A6XX_CP_VSD_BASE = 0x00000934
REG_A6XX_CP_ROQ_RB_STATUS = 0x00000939
A6XX_CP_ROQ_RB_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_RB_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_RB_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_RB_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_ROQ_IB1_STATUS = 0x0000093a
A6XX_CP_ROQ_IB1_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_IB1_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_IB1_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_IB1_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_ROQ_IB2_STATUS = 0x0000093b
A6XX_CP_ROQ_IB2_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_IB2_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_IB2_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_IB2_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_ROQ_SDS_STATUS = 0x0000093c
A6XX_CP_ROQ_SDS_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_SDS_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_SDS_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_SDS_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_ROQ_MRB_STATUS = 0x0000093d
A6XX_CP_ROQ_MRB_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_MRB_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_MRB_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_MRB_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_ROQ_VSD_STATUS = 0x0000093e
A6XX_CP_ROQ_VSD_STATUS_RPTR__MASK = 0x000003ff
A6XX_CP_ROQ_VSD_STATUS_RPTR__SHIFT = 0
A6XX_CP_ROQ_VSD_STATUS_WPTR__MASK = 0x03ff0000
A6XX_CP_ROQ_VSD_STATUS_WPTR__SHIFT = 16
REG_A6XX_CP_IB1_INIT_SIZE = 0x00000943
REG_A6XX_CP_IB2_INIT_SIZE = 0x00000944
REG_A6XX_CP_SDS_INIT_SIZE = 0x00000945
REG_A6XX_CP_MRB_INIT_SIZE = 0x00000946
REG_A6XX_CP_VSD_INIT_SIZE = 0x00000947
REG_A6XX_CP_ROQ_AVAIL_RB = 0x00000948
A6XX_CP_ROQ_AVAIL_RB_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_RB_REM__SHIFT = 16
REG_A6XX_CP_ROQ_AVAIL_IB1 = 0x00000949
A6XX_CP_ROQ_AVAIL_IB1_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_IB1_REM__SHIFT = 16
REG_A6XX_CP_ROQ_AVAIL_IB2 = 0x0000094a
A6XX_CP_ROQ_AVAIL_IB2_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_IB2_REM__SHIFT = 16
REG_A6XX_CP_ROQ_AVAIL_SDS = 0x0000094b
A6XX_CP_ROQ_AVAIL_SDS_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_SDS_REM__SHIFT = 16
REG_A6XX_CP_ROQ_AVAIL_MRB = 0x0000094c
A6XX_CP_ROQ_AVAIL_MRB_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_MRB_REM__SHIFT = 16
REG_A6XX_CP_ROQ_AVAIL_VSD = 0x0000094d
A6XX_CP_ROQ_AVAIL_VSD_REM__MASK = 0xffff0000
A6XX_CP_ROQ_AVAIL_VSD_REM__SHIFT = 16
REG_A6XX_CP_ALWAYS_ON_COUNTER = 0x00000980
REG_A6XX_CP_AHB_CNTL = 0x0000098d
REG_A6XX_CP_APERTURE_CNTL_HOST = 0x00000a00
REG_A7XX_CP_APERTURE_CNTL_HOST = 0x00000a00
A7XX_CP_APERTURE_CNTL_HOST_PIPE__MASK = 0x00003000
A7XX_CP_APERTURE_CNTL_HOST_PIPE__SHIFT = 12
A7XX_CP_APERTURE_CNTL_HOST_CLUSTER__MASK = 0x00000700
A7XX_CP_APERTURE_CNTL_HOST_CLUSTER__SHIFT = 8
A7XX_CP_APERTURE_CNTL_HOST_CONTEXT__MASK = 0x00000030
A7XX_CP_APERTURE_CNTL_HOST_CONTEXT__SHIFT = 4
REG_A6XX_CP_APERTURE_CNTL_SQE = 0x00000a01
REG_A6XX_CP_APERTURE_CNTL_CD = 0x00000a03
REG_A7XX_CP_APERTURE_CNTL_CD = 0x00000a03
A7XX_CP_APERTURE_CNTL_CD_PIPE__MASK = 0x00003000
A7XX_CP_APERTURE_CNTL_CD_PIPE__SHIFT = 12
A7XX_CP_APERTURE_CNTL_CD_CLUSTER__MASK = 0x00000700
A7XX_CP_APERTURE_CNTL_CD_CLUSTER__SHIFT = 8
A7XX_CP_APERTURE_CNTL_CD_CONTEXT__MASK = 0x00000030
A7XX_CP_APERTURE_CNTL_CD_CONTEXT__SHIFT = 4
REG_A7XX_CP_BV_PROTECT_STATUS = 0x00000a61
REG_A7XX_CP_BV_HW_FAULT = 0x00000a64
REG_A7XX_CP_BV_DRAW_STATE_ADDR = 0x00000a81
REG_A7XX_CP_BV_DRAW_STATE_DATA = 0x00000a82
REG_A7XX_CP_BV_ROQ_DBG_ADDR = 0x00000a83
REG_A7XX_CP_BV_ROQ_DBG_DATA = 0x00000a84
REG_A7XX_CP_BV_SQE_UCODE_DBG_ADDR = 0x00000a85
REG_A7XX_CP_BV_SQE_UCODE_DBG_DATA = 0x00000a86
REG_A7XX_CP_BV_SQE_STAT_ADDR = 0x00000a87
REG_A7XX_CP_BV_SQE_STAT_DATA = 0x00000a88
REG_A7XX_CP_BV_MEM_POOL_DBG_ADDR = 0x00000a96
REG_A7XX_CP_BV_MEM_POOL_DBG_DATA = 0x00000a97
REG_A7XX_CP_BV_RB_RPTR_ADDR = 0x00000a98
REG_A7XX_CP_RESOURCE_TABLE_DBG_ADDR = 0x00000a9a
REG_A7XX_CP_RESOURCE_TABLE_DBG_DATA = 0x00000a9b
REG_A7XX_CP_BV_APRIV_CNTL = 0x00000ad0
REG_A7XX_CP_BV_CHICKEN_DBG = 0x00000ada
REG_A7XX_CP_LPAC_DRAW_STATE_ADDR = 0x00000b0a
REG_A7XX_CP_LPAC_DRAW_STATE_DATA = 0x00000b0b
REG_A7XX_CP_LPAC_ROQ_DBG_ADDR = 0x00000b0c
REG_A7XX_CP_SQE_AC_UCODE_DBG_ADDR = 0x00000b27
REG_A7XX_CP_SQE_AC_UCODE_DBG_DATA = 0x00000b28
REG_A7XX_CP_SQE_AC_STAT_ADDR = 0x00000b29
REG_A7XX_CP_SQE_AC_STAT_DATA = 0x00000b2a
REG_A7XX_CP_LPAC_APRIV_CNTL = 0x00000b31
REG_A6XX_CP_LPAC_PROG_FIFO_SIZE = 0x00000b34
REG_A7XX_CP_LPAC_ROQ_DBG_DATA = 0x00000b35
REG_A7XX_CP_LPAC_FIFO_DBG_DATA = 0x00000b36
REG_A7XX_CP_LPAC_FIFO_DBG_ADDR = 0x00000b40
REG_A6XX_CP_LPAC_SQE_CNTL = 0x00000b81
REG_A6XX_CP_LPAC_SQE_INSTR_BASE = 0x00000b82
REG_A7XX_CP_AQE_INSTR_BASE_0 = 0x00000b70
REG_A7XX_CP_AQE_INSTR_BASE_1 = 0x00000b72
REG_A7XX_CP_AQE_APRIV_CNTL = 0x00000b78
REG_A7XX_CP_AQE_ROQ_DBG_ADDR_0 = 0x00000ba8
REG_A7XX_CP_AQE_ROQ_DBG_ADDR_1 = 0x00000ba9
REG_A7XX_CP_AQE_ROQ_DBG_DATA_0 = 0x00000bac
REG_A7XX_CP_AQE_ROQ_DBG_DATA_1 = 0x00000bad
REG_A7XX_CP_AQE_UCODE_DBG_ADDR_0 = 0x00000bb0
REG_A7XX_CP_AQE_UCODE_DBG_ADDR_1 = 0x00000bb1
REG_A7XX_CP_AQE_UCODE_DBG_DATA_0 = 0x00000bb4
REG_A7XX_CP_AQE_UCODE_DBG_DATA_1 = 0x00000bb5
REG_A7XX_CP_AQE_STAT_ADDR_0 = 0x00000bb8
REG_A7XX_CP_AQE_STAT_ADDR_1 = 0x00000bb9
REG_A7XX_CP_AQE_STAT_DATA_0 = 0x00000bbc
REG_A7XX_CP_AQE_STAT_DATA_1 = 0x00000bbd
REG_A6XX_VSC_ADDR_MODE_CNTL = 0x00000c01
REG_A6XX_RBBM_GPR0_CNTL = 0x00000018
REG_A6XX_RBBM_INT_0_STATUS = 0x00000201
REG_A6XX_RBBM_STATUS = 0x00000210
A6XX_RBBM_STATUS_GPU_BUSY_IGN_AHB = 0x00800000
A6XX_RBBM_STATUS_GPU_BUSY_IGN_AHB_CP = 0x00400000
A6XX_RBBM_STATUS_HLSQ_BUSY = 0x00200000
A6XX_RBBM_STATUS_VSC_BUSY = 0x00100000
A6XX_RBBM_STATUS_TPL1_BUSY = 0x00080000
A6XX_RBBM_STATUS_SP_BUSY = 0x00040000
A6XX_RBBM_STATUS_UCHE_BUSY = 0x00020000
A6XX_RBBM_STATUS_VPC_BUSY = 0x00010000
A6XX_RBBM_STATUS_VFD_BUSY = 0x00008000
A6XX_RBBM_STATUS_TESS_BUSY = 0x00004000
A6XX_RBBM_STATUS_PC_VSD_BUSY = 0x00002000
A6XX_RBBM_STATUS_PC_DCALL_BUSY = 0x00001000
A6XX_RBBM_STATUS_COM_DCOM_BUSY = 0x00000800
A6XX_RBBM_STATUS_LRZ_BUSY = 0x00000400
A6XX_RBBM_STATUS_A2D_BUSY = 0x00000200
A6XX_RBBM_STATUS_CCU_BUSY = 0x00000100
A6XX_RBBM_STATUS_RB_BUSY = 0x00000080
A6XX_RBBM_STATUS_RAS_BUSY = 0x00000040
A6XX_RBBM_STATUS_TSE_BUSY = 0x00000020
A6XX_RBBM_STATUS_VBIF_BUSY = 0x00000010
A6XX_RBBM_STATUS_GFX_DBGC_BUSY = 0x00000008
A6XX_RBBM_STATUS_CP_BUSY = 0x00000004
A6XX_RBBM_STATUS_CP_AHB_BUSY_CP_MASTER = 0x00000002
A6XX_RBBM_STATUS_CP_AHB_BUSY_CX_MASTER = 0x00000001
REG_A6XX_RBBM_STATUS1 = 0x00000211
REG_A6XX_RBBM_STATUS2 = 0x00000212
REG_A6XX_RBBM_STATUS3 = 0x00000213
A6XX_RBBM_STATUS3_SMMU_STALLED_ON_FAULT = 0x01000000
REG_A6XX_RBBM_VBIF_GX_RESET_STATUS = 0x00000215
REG_A7XX_RBBM_CLOCK_MODE_CP = 0x00000260
REG_A7XX_RBBM_CLOCK_MODE_BV_LRZ = 0x00000284
REG_A7XX_RBBM_CLOCK_MODE_BV_GRAS = 0x00000285
REG_A7XX_RBBM_CLOCK_MODE2_GRAS = 0x00000286
REG_A7XX_RBBM_CLOCK_MODE_BV_VFD = 0x00000287
REG_A7XX_RBBM_CLOCK_MODE_BV_GPC = 0x00000288
REG_A7XX_RBBM_SW_FUSE_INT_STATUS = 0x000002c0
REG_A7XX_RBBM_SW_FUSE_INT_MASK = 0x000002c1
REG_A6XX_RBBM_PERFCTR_CP = lambda i0: (0x00000400 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_RBBM = lambda i0: (0x0000041c + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_PC = lambda i0: (0x00000424 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_VFD = lambda i0: (0x00000434 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_HLSQ = lambda i0: (0x00000444 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_VPC = lambda i0: (0x00000450 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_CCU = lambda i0: (0x0000045c + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_TSE = lambda i0: (0x00000466 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_RAS = lambda i0: (0x0000046e + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_UCHE = lambda i0: (0x00000476 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_TP = lambda i0: (0x0000048e + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_SP = lambda i0: (0x000004a6 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_RB = lambda i0: (0x000004d6 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_VSC = lambda i0: (0x000004e6 + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_LRZ = lambda i0: (0x000004ea + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_CMP = lambda i0: (0x000004f2 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_CP = lambda i0: (0x00000300 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_RBBM = lambda i0: (0x0000031c + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_PC = lambda i0: (0x00000324 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_VFD = lambda i0: (0x00000334 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_HLSQ = lambda i0: (0x00000344 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_VPC = lambda i0: (0x00000350 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_CCU = lambda i0: (0x0000035c + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_TSE = lambda i0: (0x00000366 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_RAS = lambda i0: (0x0000036e + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_UCHE = lambda i0: (0x00000376 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_TP = lambda i0: (0x0000038e + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_SP = lambda i0: (0x000003a6 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_RB = lambda i0: (0x000003d6 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_VSC = lambda i0: (0x000003e6 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_LRZ = lambda i0: (0x000003ea + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_CMP = lambda i0: (0x000003f2 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_UFC = lambda i0: (0x000003fa + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR2_HLSQ = lambda i0: (0x00000410 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR2_CP = lambda i0: (0x0000041c + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR2_SP = lambda i0: (0x0000042a + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR2_TP = lambda i0: (0x00000442 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR2_UFC = lambda i0: (0x0000044e + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_PC = lambda i0: (0x00000460 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_VFD = lambda i0: (0x00000470 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_VPC = lambda i0: (0x00000480 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_TSE = lambda i0: (0x0000048c + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_RAS = lambda i0: (0x00000494 + 0x2*i0 )
REG_A7XX_RBBM_PERFCTR_BV_LRZ = lambda i0: (0x0000049c + 0x2*i0 )
REG_A6XX_RBBM_PERFCTR_CNTL = 0x00000500
REG_A6XX_RBBM_PERFCTR_LOAD_CMD0 = 0x00000501
REG_A6XX_RBBM_PERFCTR_LOAD_CMD1 = 0x00000502
REG_A6XX_RBBM_PERFCTR_LOAD_CMD2 = 0x00000503
REG_A6XX_RBBM_PERFCTR_LOAD_CMD3 = 0x00000504
REG_A6XX_RBBM_PERFCTR_LOAD_VALUE_LO = 0x00000505
REG_A6XX_RBBM_PERFCTR_LOAD_VALUE_HI = 0x00000506
REG_A6XX_RBBM_PERFCTR_RBBM_SEL = lambda i0: (0x00000507 + 0x1*i0 )
REG_A6XX_RBBM_PERFCTR_GPU_BUSY_MASKED = 0x0000050b
REG_A6XX_RBBM_PERFCTR_SRAM_INIT_CMD = 0x0000050e
REG_A6XX_RBBM_PERFCTR_SRAM_INIT_STATUS = 0x0000050f
REG_A6XX_RBBM_ISDB_CNT = 0x00000533
REG_A6XX_RBBM_NC_MODE_CNTL = 0x00000534
REG_A7XX_RBBM_SNAPSHOT_STATUS = 0x00000535
REG_A6XX_RBBM_PIPESTAT_IAVERTICES = 0x00000540
REG_A6XX_RBBM_PIPESTAT_IAPRIMITIVES = 0x00000542
REG_A6XX_RBBM_PIPESTAT_VSINVOCATIONS = 0x00000544
REG_A6XX_RBBM_PIPESTAT_HSINVOCATIONS = 0x00000546
REG_A6XX_RBBM_PIPESTAT_DSINVOCATIONS = 0x00000548
REG_A6XX_RBBM_PIPESTAT_GSINVOCATIONS = 0x0000054a
REG_A6XX_RBBM_PIPESTAT_GSPRIMITIVES = 0x0000054c
REG_A6XX_RBBM_PIPESTAT_CINVOCATIONS = 0x0000054e
REG_A6XX_RBBM_PIPESTAT_CPRIMITIVES = 0x00000550
REG_A6XX_RBBM_PIPESTAT_PSINVOCATIONS = 0x00000552
REG_A6XX_RBBM_PIPESTAT_CSINVOCATIONS = 0x00000554
REG_A6XX_RBBM_SECVID_TRUST_CNTL = 0x0000f400
REG_A6XX_RBBM_SECVID_TSB_TRUSTED_BASE = 0x0000f800
REG_A6XX_RBBM_SECVID_TSB_TRUSTED_SIZE = 0x0000f802
REG_A6XX_RBBM_SECVID_TSB_CNTL = 0x0000f803
REG_A6XX_RBBM_SECVID_TSB_ADDR_MODE_CNTL = 0x0000f810
REG_A7XX_RBBM_SECVID_TSB_STATUS = 0x0000fc00
REG_A6XX_RBBM_VBIF_CLIENT_QOS_CNTL = 0x00000010
REG_A6XX_RBBM_GBIF_CLIENT_QOS_CNTL = 0x00000011
REG_A6XX_RBBM_GBIF_HALT = 0x00000016
REG_A6XX_RBBM_GBIF_HALT_ACK = 0x00000017
REG_A6XX_RBBM_WAIT_FOR_GPU_IDLE_CMD = 0x0000001c
A6XX_RBBM_WAIT_FOR_GPU_IDLE_CMD_WAIT_GPU_IDLE = 0x00000001
REG_A7XX_RBBM_GBIF_HALT = 0x00000016
REG_A7XX_RBBM_GBIF_HALT_ACK = 0x00000017
REG_A6XX_RBBM_INTERFACE_HANG_INT_CNTL = 0x0000001f
REG_A6XX_RBBM_INT_CLEAR_CMD = 0x00000037
REG_A6XX_RBBM_INT_0_MASK = 0x00000038
REG_A7XX_RBBM_INT_2_MASK = 0x0000003a
REG_A6XX_RBBM_SP_HYST_CNT = 0x00000042
REG_A6XX_RBBM_SW_RESET_CMD = 0x00000043
REG_A6XX_RBBM_RAC_THRESHOLD_CNT = 0x00000044
REG_A6XX_RBBM_BLOCK_SW_RESET_CMD = 0x00000045
REG_A6XX_RBBM_BLOCK_SW_RESET_CMD2 = 0x00000046
REG_A7XX_RBBM_CLOCK_CNTL_GLOBAL = 0x000000ad
REG_A6XX_RBBM_CLOCK_CNTL = 0x000000ae
REG_A6XX_RBBM_CLOCK_CNTL_SP0 = 0x000000b0
REG_A6XX_RBBM_CLOCK_CNTL_SP1 = 0x000000b1
REG_A6XX_RBBM_CLOCK_CNTL_SP2 = 0x000000b2
REG_A6XX_RBBM_CLOCK_CNTL_SP3 = 0x000000b3
REG_A6XX_RBBM_CLOCK_CNTL2_SP0 = 0x000000b4
REG_A6XX_RBBM_CLOCK_CNTL2_SP1 = 0x000000b5
REG_A6XX_RBBM_CLOCK_CNTL2_SP2 = 0x000000b6
REG_A6XX_RBBM_CLOCK_CNTL2_SP3 = 0x000000b7
REG_A6XX_RBBM_CLOCK_DELAY_SP0 = 0x000000b8
REG_A6XX_RBBM_CLOCK_DELAY_SP1 = 0x000000b9
REG_A6XX_RBBM_CLOCK_DELAY_SP2 = 0x000000ba
REG_A6XX_RBBM_CLOCK_DELAY_SP3 = 0x000000bb
REG_A6XX_RBBM_CLOCK_HYST_SP0 = 0x000000bc
REG_A6XX_RBBM_CLOCK_HYST_SP1 = 0x000000bd
REG_A6XX_RBBM_CLOCK_HYST_SP2 = 0x000000be
REG_A6XX_RBBM_CLOCK_HYST_SP3 = 0x000000bf
REG_A6XX_RBBM_CLOCK_CNTL_TP0 = 0x000000c0
REG_A6XX_RBBM_CLOCK_CNTL_TP1 = 0x000000c1
REG_A6XX_RBBM_CLOCK_CNTL_TP2 = 0x000000c2
REG_A6XX_RBBM_CLOCK_CNTL_TP3 = 0x000000c3
REG_A6XX_RBBM_CLOCK_CNTL2_TP0 = 0x000000c4
REG_A6XX_RBBM_CLOCK_CNTL2_TP1 = 0x000000c5
REG_A6XX_RBBM_CLOCK_CNTL2_TP2 = 0x000000c6
REG_A6XX_RBBM_CLOCK_CNTL2_TP3 = 0x000000c7
REG_A6XX_RBBM_CLOCK_CNTL3_TP0 = 0x000000c8
REG_A6XX_RBBM_CLOCK_CNTL3_TP1 = 0x000000c9
REG_A6XX_RBBM_CLOCK_CNTL3_TP2 = 0x000000ca
REG_A6XX_RBBM_CLOCK_CNTL3_TP3 = 0x000000cb
REG_A6XX_RBBM_CLOCK_CNTL4_TP0 = 0x000000cc
REG_A6XX_RBBM_CLOCK_CNTL4_TP1 = 0x000000cd
REG_A6XX_RBBM_CLOCK_CNTL4_TP2 = 0x000000ce
REG_A6XX_RBBM_CLOCK_CNTL4_TP3 = 0x000000cf
REG_A6XX_RBBM_CLOCK_DELAY_TP0 = 0x000000d0
REG_A6XX_RBBM_CLOCK_DELAY_TP1 = 0x000000d1
REG_A6XX_RBBM_CLOCK_DELAY_TP2 = 0x000000d2
REG_A6XX_RBBM_CLOCK_DELAY_TP3 = 0x000000d3
REG_A6XX_RBBM_CLOCK_DELAY2_TP0 = 0x000000d4
REG_A6XX_RBBM_CLOCK_DELAY2_TP1 = 0x000000d5
REG_A6XX_RBBM_CLOCK_DELAY2_TP2 = 0x000000d6
REG_A6XX_RBBM_CLOCK_DELAY2_TP3 = 0x000000d7
REG_A6XX_RBBM_CLOCK_DELAY3_TP0 = 0x000000d8
REG_A6XX_RBBM_CLOCK_DELAY3_TP1 = 0x000000d9
REG_A6XX_RBBM_CLOCK_DELAY3_TP2 = 0x000000da
REG_A6XX_RBBM_CLOCK_DELAY3_TP3 = 0x000000db
REG_A6XX_RBBM_CLOCK_DELAY4_TP0 = 0x000000dc
REG_A6XX_RBBM_CLOCK_DELAY4_TP1 = 0x000000dd
REG_A6XX_RBBM_CLOCK_DELAY4_TP2 = 0x000000de
REG_A6XX_RBBM_CLOCK_DELAY4_TP3 = 0x000000df
REG_A6XX_RBBM_CLOCK_HYST_TP0 = 0x000000e0
REG_A6XX_RBBM_CLOCK_HYST_TP1 = 0x000000e1
REG_A6XX_RBBM_CLOCK_HYST_TP2 = 0x000000e2
REG_A6XX_RBBM_CLOCK_HYST_TP3 = 0x000000e3
REG_A6XX_RBBM_CLOCK_HYST2_TP0 = 0x000000e4
REG_A6XX_RBBM_CLOCK_HYST2_TP1 = 0x000000e5
REG_A6XX_RBBM_CLOCK_HYST2_TP2 = 0x000000e6
REG_A6XX_RBBM_CLOCK_HYST2_TP3 = 0x000000e7
REG_A6XX_RBBM_CLOCK_HYST3_TP0 = 0x000000e8
REG_A6XX_RBBM_CLOCK_HYST3_TP1 = 0x000000e9
REG_A6XX_RBBM_CLOCK_HYST3_TP2 = 0x000000ea
REG_A6XX_RBBM_CLOCK_HYST3_TP3 = 0x000000eb
REG_A6XX_RBBM_CLOCK_HYST4_TP0 = 0x000000ec
REG_A6XX_RBBM_CLOCK_HYST4_TP1 = 0x000000ed
REG_A6XX_RBBM_CLOCK_HYST4_TP2 = 0x000000ee
REG_A6XX_RBBM_CLOCK_HYST4_TP3 = 0x000000ef
REG_A6XX_RBBM_CLOCK_CNTL_RB0 = 0x000000f0
REG_A6XX_RBBM_CLOCK_CNTL_RB1 = 0x000000f1
REG_A6XX_RBBM_CLOCK_CNTL_RB2 = 0x000000f2
REG_A6XX_RBBM_CLOCK_CNTL_RB3 = 0x000000f3
REG_A6XX_RBBM_CLOCK_CNTL2_RB0 = 0x000000f4
REG_A6XX_RBBM_CLOCK_CNTL2_RB1 = 0x000000f5
REG_A6XX_RBBM_CLOCK_CNTL2_RB2 = 0x000000f6
REG_A6XX_RBBM_CLOCK_CNTL2_RB3 = 0x000000f7
REG_A6XX_RBBM_CLOCK_CNTL_CCU0 = 0x000000f8
REG_A6XX_RBBM_CLOCK_CNTL_CCU1 = 0x000000f9
REG_A6XX_RBBM_CLOCK_CNTL_CCU2 = 0x000000fa
REG_A6XX_RBBM_CLOCK_CNTL_CCU3 = 0x000000fb
REG_A6XX_RBBM_CLOCK_HYST_RB_CCU0 = 0x00000100
REG_A6XX_RBBM_CLOCK_HYST_RB_CCU1 = 0x00000101
REG_A6XX_RBBM_CLOCK_HYST_RB_CCU2 = 0x00000102
REG_A6XX_RBBM_CLOCK_HYST_RB_CCU3 = 0x00000103
REG_A6XX_RBBM_CLOCK_CNTL_RAC = 0x00000104
REG_A6XX_RBBM_CLOCK_CNTL2_RAC = 0x00000105
REG_A6XX_RBBM_CLOCK_DELAY_RAC = 0x00000106
REG_A6XX_RBBM_CLOCK_HYST_RAC = 0x00000107
REG_A6XX_RBBM_CLOCK_CNTL_TSE_RAS_RBBM = 0x00000108
REG_A6XX_RBBM_CLOCK_DELAY_TSE_RAS_RBBM = 0x00000109
REG_A6XX_RBBM_CLOCK_HYST_TSE_RAS_RBBM = 0x0000010a
REG_A6XX_RBBM_CLOCK_CNTL_UCHE = 0x0000010b
REG_A6XX_RBBM_CLOCK_CNTL2_UCHE = 0x0000010c
REG_A6XX_RBBM_CLOCK_CNTL3_UCHE = 0x0000010d
REG_A6XX_RBBM_CLOCK_CNTL4_UCHE = 0x0000010e
REG_A6XX_RBBM_CLOCK_DELAY_UCHE = 0x0000010f
REG_A6XX_RBBM_CLOCK_HYST_UCHE = 0x00000110
REG_A6XX_RBBM_CLOCK_MODE_VFD = 0x00000111
REG_A6XX_RBBM_CLOCK_DELAY_VFD = 0x00000112
REG_A6XX_RBBM_CLOCK_HYST_VFD = 0x00000113
REG_A6XX_RBBM_CLOCK_MODE_GPC = 0x00000114
REG_A6XX_RBBM_CLOCK_DELAY_GPC = 0x00000115
REG_A6XX_RBBM_CLOCK_HYST_GPC = 0x00000116
REG_A6XX_RBBM_CLOCK_DELAY_HLSQ_2 = 0x00000117
REG_A6XX_RBBM_CLOCK_CNTL_GMU_GX = 0x00000118
REG_A6XX_RBBM_CLOCK_DELAY_GMU_GX = 0x00000119
REG_A6XX_RBBM_CLOCK_HYST_GMU_GX = 0x0000011a
REG_A6XX_RBBM_CLOCK_MODE_HLSQ = 0x0000011b
REG_A6XX_RBBM_CLOCK_DELAY_HLSQ = 0x0000011c
REG_A6XX_RBBM_CLOCK_HYST_HLSQ = 0x0000011d
REG_A7XX_RBBM_CGC_GLOBAL_LOAD_CMD = 0x0000011e
REG_A7XX_RBBM_CGC_P2S_TRIG_CMD = 0x0000011f
REG_A6XX_RBBM_CLOCK_CNTL_TEX_FCHE = 0x00000120
REG_A6XX_RBBM_CLOCK_DELAY_TEX_FCHE = 0x00000121
REG_A6XX_RBBM_CLOCK_HYST_TEX_FCHE = 0x00000122
REG_A7XX_RBBM_CGC_P2S_STATUS = 0x00000122
A7XX_RBBM_CGC_P2S_STATUS_TXDONE = 0x00000001
REG_A6XX_RBBM_CLOCK_CNTL_FCHE = 0x00000123
REG_A6XX_RBBM_CLOCK_DELAY_FCHE = 0x00000124
REG_A6XX_RBBM_CLOCK_HYST_FCHE = 0x00000125
REG_A6XX_RBBM_CLOCK_CNTL_MHUB = 0x00000126
REG_A6XX_RBBM_CLOCK_DELAY_MHUB = 0x00000127
REG_A6XX_RBBM_CLOCK_HYST_MHUB = 0x00000128
REG_A6XX_RBBM_CLOCK_DELAY_GLC = 0x00000129
REG_A6XX_RBBM_CLOCK_HYST_GLC = 0x0000012a
REG_A6XX_RBBM_CLOCK_CNTL_GLC = 0x0000012b
REG_A7XX_RBBM_CLOCK_HYST2_VFD = 0x0000012f
REG_A6XX_RBBM_LPAC_GBIF_CLIENT_QOS_CNTL = 0x000005ff
REG_A6XX_DBGC_CFG_DBGBUS_SEL_A = 0x00000600
REG_A6XX_DBGC_CFG_DBGBUS_SEL_B = 0x00000601
REG_A6XX_DBGC_CFG_DBGBUS_SEL_C = 0x00000602
REG_A6XX_DBGC_CFG_DBGBUS_SEL_D = 0x00000603
A6XX_DBGC_CFG_DBGBUS_SEL_D_PING_INDEX__MASK = 0x000000ff
A6XX_DBGC_CFG_DBGBUS_SEL_D_PING_INDEX__SHIFT = 0
A6XX_DBGC_CFG_DBGBUS_SEL_D_PING_BLK_SEL__MASK = 0x0000ff00
A6XX_DBGC_CFG_DBGBUS_SEL_D_PING_BLK_SEL__SHIFT = 8
REG_A6XX_DBGC_CFG_DBGBUS_CNTLT = 0x00000604
A6XX_DBGC_CFG_DBGBUS_CNTLT_TRACEEN__MASK = 0x0000003f
A6XX_DBGC_CFG_DBGBUS_CNTLT_TRACEEN__SHIFT = 0
A6XX_DBGC_CFG_DBGBUS_CNTLT_GRANU__MASK = 0x00007000
A6XX_DBGC_CFG_DBGBUS_CNTLT_GRANU__SHIFT = 12
A6XX_DBGC_CFG_DBGBUS_CNTLT_SEGT__MASK = 0xf0000000
A6XX_DBGC_CFG_DBGBUS_CNTLT_SEGT__SHIFT = 28
REG_A6XX_DBGC_CFG_DBGBUS_CNTLM = 0x00000605
A6XX_DBGC_CFG_DBGBUS_CNTLM_ENABLE__MASK = 0x0f000000
A6XX_DBGC_CFG_DBGBUS_CNTLM_ENABLE__SHIFT = 24
REG_A6XX_DBGC_CFG_DBGBUS_IVTL_0 = 0x00000608
REG_A6XX_DBGC_CFG_DBGBUS_IVTL_1 = 0x00000609
REG_A6XX_DBGC_CFG_DBGBUS_IVTL_2 = 0x0000060a
REG_A6XX_DBGC_CFG_DBGBUS_IVTL_3 = 0x0000060b
REG_A6XX_DBGC_CFG_DBGBUS_MASKL_0 = 0x0000060c
REG_A6XX_DBGC_CFG_DBGBUS_MASKL_1 = 0x0000060d
REG_A6XX_DBGC_CFG_DBGBUS_MASKL_2 = 0x0000060e
REG_A6XX_DBGC_CFG_DBGBUS_MASKL_3 = 0x0000060f
REG_A6XX_DBGC_CFG_DBGBUS_BYTEL_0 = 0x00000610
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL0__MASK = 0x0000000f
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL0__SHIFT = 0
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL1__MASK = 0x000000f0
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL1__SHIFT = 4
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL2__MASK = 0x00000f00
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL2__SHIFT = 8
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL3__MASK = 0x0000f000
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL3__SHIFT = 12
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL4__MASK = 0x000f0000
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL4__SHIFT = 16
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL5__MASK = 0x00f00000
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL5__SHIFT = 20
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL6__MASK = 0x0f000000
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL6__SHIFT = 24
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL7__MASK = 0xf0000000
A6XX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL7__SHIFT = 28
REG_A6XX_DBGC_CFG_DBGBUS_BYTEL_1 = 0x00000611
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL8__MASK = 0x0000000f
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL8__SHIFT = 0
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL9__MASK = 0x000000f0
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL9__SHIFT = 4
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL10__MASK = 0x00000f00
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL10__SHIFT = 8
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL11__MASK = 0x0000f000
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL11__SHIFT = 12
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL12__MASK = 0x000f0000
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL12__SHIFT = 16
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL13__MASK = 0x00f00000
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL13__SHIFT = 20
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL14__MASK = 0x0f000000
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL14__SHIFT = 24
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL15__MASK = 0xf0000000
A6XX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL15__SHIFT = 28
REG_A6XX_DBGC_CFG_DBGBUS_TRACE_BUF1 = 0x0000062f
REG_A6XX_DBGC_CFG_DBGBUS_TRACE_BUF2 = 0x00000630
REG_A6XX_VSC_PERFCTR_VSC_SEL = lambda i0: (0x00000cd8 + 0x1*i0 )
REG_A7XX_VSC_UNKNOWN_0CD8 = 0x00000cd8
A7XX_VSC_UNKNOWN_0CD8_BINNING = 0x00000001
REG_A6XX_HLSQ_DBG_AHB_READ_APERTURE = 0x0000c800
REG_A6XX_HLSQ_DBG_READ_SEL = 0x0000d000
REG_A6XX_UCHE_ADDR_MODE_CNTL = 0x00000e00
REG_A6XX_UCHE_MODE_CNTL = 0x00000e01
REG_A6XX_UCHE_WRITE_RANGE_MAX = 0x00000e05
REG_A6XX_UCHE_WRITE_THRU_BASE = 0x00000e07
REG_A6XX_UCHE_TRAP_BASE = 0x00000e09
REG_A6XX_UCHE_GMEM_RANGE_MIN = 0x00000e0b
REG_A6XX_UCHE_GMEM_RANGE_MAX = 0x00000e0d
REG_A6XX_UCHE_CACHE_WAYS = 0x00000e17
REG_A6XX_UCHE_FILTER_CNTL = 0x00000e18
REG_A6XX_UCHE_CLIENT_PF = 0x00000e19
A6XX_UCHE_CLIENT_PF_PERFSEL__MASK = 0x000000ff
A6XX_UCHE_CLIENT_PF_PERFSEL__SHIFT = 0
REG_A6XX_UCHE_PERFCTR_UCHE_SEL = lambda i0: (0x00000e1c + 0x1*i0 )
REG_A6XX_UCHE_GBIF_GX_CONFIG = 0x00000e3a
REG_A6XX_UCHE_CMDQ_CONFIG = 0x00000e3c
REG_A6XX_VBIF_VERSION = 0x00003000
REG_A6XX_VBIF_CLKON = 0x00003001
A6XX_VBIF_CLKON_FORCE_ON_TESTBUS = 0x00000002
REG_A6XX_VBIF_GATE_OFF_WRREQ_EN = 0x0000302a
REG_A6XX_VBIF_XIN_HALT_CTRL0 = 0x00003080
REG_A6XX_VBIF_XIN_HALT_CTRL1 = 0x00003081
REG_A6XX_VBIF_TEST_BUS_OUT_CTRL = 0x00003084
REG_A6XX_VBIF_TEST_BUS1_CTRL0 = 0x00003085
REG_A6XX_VBIF_TEST_BUS1_CTRL1 = 0x00003086
A6XX_VBIF_TEST_BUS1_CTRL1_DATA_SEL__MASK = 0x0000000f
A6XX_VBIF_TEST_BUS1_CTRL1_DATA_SEL__SHIFT = 0
REG_A6XX_VBIF_TEST_BUS2_CTRL0 = 0x00003087
REG_A6XX_VBIF_TEST_BUS2_CTRL1 = 0x00003088
A6XX_VBIF_TEST_BUS2_CTRL1_DATA_SEL__MASK = 0x000001ff
A6XX_VBIF_TEST_BUS2_CTRL1_DATA_SEL__SHIFT = 0
REG_A6XX_VBIF_TEST_BUS_OUT = 0x0000308c
REG_A6XX_VBIF_PERF_CNT_SEL0 = 0x000030d0
REG_A6XX_VBIF_PERF_CNT_SEL1 = 0x000030d1
REG_A6XX_VBIF_PERF_CNT_SEL2 = 0x000030d2
REG_A6XX_VBIF_PERF_CNT_SEL3 = 0x000030d3
REG_A6XX_VBIF_PERF_CNT_LOW0 = 0x000030d8
REG_A6XX_VBIF_PERF_CNT_LOW1 = 0x000030d9
REG_A6XX_VBIF_PERF_CNT_LOW2 = 0x000030da
REG_A6XX_VBIF_PERF_CNT_LOW3 = 0x000030db
REG_A6XX_VBIF_PERF_CNT_HIGH0 = 0x000030e0
REG_A6XX_VBIF_PERF_CNT_HIGH1 = 0x000030e1
REG_A6XX_VBIF_PERF_CNT_HIGH2 = 0x000030e2
REG_A6XX_VBIF_PERF_CNT_HIGH3 = 0x000030e3
REG_A6XX_VBIF_PERF_PWR_CNT_EN0 = 0x00003100
REG_A6XX_VBIF_PERF_PWR_CNT_EN1 = 0x00003101
REG_A6XX_VBIF_PERF_PWR_CNT_EN2 = 0x00003102
REG_A6XX_VBIF_PERF_PWR_CNT_LOW0 = 0x00003110
REG_A6XX_VBIF_PERF_PWR_CNT_LOW1 = 0x00003111
REG_A6XX_VBIF_PERF_PWR_CNT_LOW2 = 0x00003112
REG_A6XX_VBIF_PERF_PWR_CNT_HIGH0 = 0x00003118
REG_A6XX_VBIF_PERF_PWR_CNT_HIGH1 = 0x00003119
REG_A6XX_VBIF_PERF_PWR_CNT_HIGH2 = 0x0000311a
REG_A6XX_GBIF_SCACHE_CNTL0 = 0x00003c01
REG_A6XX_GBIF_SCACHE_CNTL1 = 0x00003c02
REG_A6XX_GBIF_QSB_SIDE0 = 0x00003c03
REG_A6XX_GBIF_QSB_SIDE1 = 0x00003c04
REG_A6XX_GBIF_QSB_SIDE2 = 0x00003c05
REG_A6XX_GBIF_QSB_SIDE3 = 0x00003c06
REG_A6XX_GBIF_HALT = 0x00003c45
REG_A6XX_GBIF_HALT_ACK = 0x00003c46
REG_A6XX_GBIF_PERF_PWR_CNT_EN = 0x00003cc0
REG_A6XX_GBIF_PERF_PWR_CNT_CLR = 0x00003cc1
REG_A6XX_GBIF_PERF_CNT_SEL = 0x00003cc2
REG_A6XX_GBIF_PERF_PWR_CNT_SEL = 0x00003cc3
REG_A6XX_GBIF_PERF_CNT_LOW0 = 0x00003cc4
REG_A6XX_GBIF_PERF_CNT_LOW1 = 0x00003cc5
REG_A6XX_GBIF_PERF_CNT_LOW2 = 0x00003cc6
REG_A6XX_GBIF_PERF_CNT_LOW3 = 0x00003cc7
REG_A6XX_GBIF_PERF_CNT_HIGH0 = 0x00003cc8
REG_A6XX_GBIF_PERF_CNT_HIGH1 = 0x00003cc9
REG_A6XX_GBIF_PERF_CNT_HIGH2 = 0x00003cca
REG_A6XX_GBIF_PERF_CNT_HIGH3 = 0x00003ccb
REG_A6XX_GBIF_PWR_CNT_LOW0 = 0x00003ccc
REG_A6XX_GBIF_PWR_CNT_LOW1 = 0x00003ccd
REG_A6XX_GBIF_PWR_CNT_LOW2 = 0x00003cce
REG_A6XX_GBIF_PWR_CNT_HIGH0 = 0x00003ccf
REG_A6XX_GBIF_PWR_CNT_HIGH1 = 0x00003cd0
REG_A6XX_GBIF_PWR_CNT_HIGH2 = 0x00003cd1
REG_A6XX_VSC_DBG_ECO_CNTL = 0x00000c00
REG_A6XX_VSC_BIN_SIZE = 0x00000c02
A6XX_VSC_BIN_SIZE_WIDTH__MASK = 0x000000ff
A6XX_VSC_BIN_SIZE_WIDTH__SHIFT = 0
A6XX_VSC_BIN_SIZE_HEIGHT__MASK = 0x0001ff00
A6XX_VSC_BIN_SIZE_HEIGHT__SHIFT = 8
REG_A6XX_VSC_SIZE_BASE = 0x00000c03
REG_A6XX_VSC_EXPANDED_BIN_CNTL = 0x00000c06
A6XX_VSC_EXPANDED_BIN_CNTL_NX__MASK = 0x000007fe
A6XX_VSC_EXPANDED_BIN_CNTL_NX__SHIFT = 1
A6XX_VSC_EXPANDED_BIN_CNTL_NY__MASK = 0x001ff800
A6XX_VSC_EXPANDED_BIN_CNTL_NY__SHIFT = 11
REG_A6XX_VSC_PIPE_CONFIG = lambda i0: (0x00000c10 + 0x1*i0 )
A6XX_VSC_PIPE_CONFIG_REG_X__MASK = 0x000003ff
A6XX_VSC_PIPE_CONFIG_REG_X__SHIFT = 0
A6XX_VSC_PIPE_CONFIG_REG_Y__MASK = 0x000ffc00
A6XX_VSC_PIPE_CONFIG_REG_Y__SHIFT = 10
A6XX_VSC_PIPE_CONFIG_REG_W__MASK = 0x03f00000
A6XX_VSC_PIPE_CONFIG_REG_W__SHIFT = 20
A6XX_VSC_PIPE_CONFIG_REG_H__MASK = 0xfc000000
A6XX_VSC_PIPE_CONFIG_REG_H__SHIFT = 26
REG_A6XX_VSC_PIPE_DATA_PRIM_BASE = 0x00000c30
REG_A6XX_VSC_PIPE_DATA_PRIM_STRIDE = 0x00000c32
REG_A6XX_VSC_PIPE_DATA_PRIM_LENGTH = 0x00000c33
REG_A6XX_VSC_PIPE_DATA_DRAW_BASE = 0x00000c34
REG_A6XX_VSC_PIPE_DATA_DRAW_STRIDE = 0x00000c36
REG_A6XX_VSC_PIPE_DATA_DRAW_LENGTH = 0x00000c37
REG_A6XX_VSC_CHANNEL_VISIBILITY = lambda i0: (0x00000c38 + 0x1*i0 )
REG_A6XX_VSC_PIPE_DATA_PRIM_SIZE = lambda i0: (0x00000c58 + 0x1*i0 )
REG_A6XX_VSC_PIPE_DATA_DRAW_SIZE = lambda i0: (0x00000c78 + 0x1*i0 )
REG_A7XX_VSC_UNKNOWN_0D08 = 0x00000d08
REG_A7XX_UCHE_UNKNOWN_0E10 = 0x00000e10
REG_A7XX_UCHE_UNKNOWN_0E11 = 0x00000e11
REG_A6XX_UCHE_UNKNOWN_0E12 = 0x00000e12
REG_A6XX_GRAS_CL_CNTL = 0x00008000
A6XX_GRAS_CL_CNTL_CLIP_DISABLE = 0x00000001
A6XX_GRAS_CL_CNTL_ZNEAR_CLIP_DISABLE = 0x00000002
A6XX_GRAS_CL_CNTL_ZFAR_CLIP_DISABLE = 0x00000004
A6XX_GRAS_CL_CNTL_Z_CLAMP_ENABLE = 0x00000020
A6XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z = 0x00000040
A6XX_GRAS_CL_CNTL_VP_CLIP_CODE_IGNORE = 0x00000080
A6XX_GRAS_CL_CNTL_VP_XFORM_DISABLE = 0x00000100
A6XX_GRAS_CL_CNTL_PERSP_DIVISION_DISABLE = 0x00000200
REG_A6XX_GRAS_CL_VS_CLIP_CULL_DISTANCE = 0x00008001
A6XX_GRAS_CL_VS_CLIP_CULL_DISTANCE_CLIP_MASK__MASK = 0x000000ff
A6XX_GRAS_CL_VS_CLIP_CULL_DISTANCE_CLIP_MASK__SHIFT = 0
A6XX_GRAS_CL_VS_CLIP_CULL_DISTANCE_CULL_MASK__MASK = 0x0000ff00
A6XX_GRAS_CL_VS_CLIP_CULL_DISTANCE_CULL_MASK__SHIFT = 8
REG_A6XX_GRAS_CL_DS_CLIP_CULL_DISTANCE = 0x00008002
A6XX_GRAS_CL_DS_CLIP_CULL_DISTANCE_CLIP_MASK__MASK = 0x000000ff
A6XX_GRAS_CL_DS_CLIP_CULL_DISTANCE_CLIP_MASK__SHIFT = 0
A6XX_GRAS_CL_DS_CLIP_CULL_DISTANCE_CULL_MASK__MASK = 0x0000ff00
A6XX_GRAS_CL_DS_CLIP_CULL_DISTANCE_CULL_MASK__SHIFT = 8
REG_A6XX_GRAS_CL_GS_CLIP_CULL_DISTANCE = 0x00008003
A6XX_GRAS_CL_GS_CLIP_CULL_DISTANCE_CLIP_MASK__MASK = 0x000000ff
A6XX_GRAS_CL_GS_CLIP_CULL_DISTANCE_CLIP_MASK__SHIFT = 0
A6XX_GRAS_CL_GS_CLIP_CULL_DISTANCE_CULL_MASK__MASK = 0x0000ff00
A6XX_GRAS_CL_GS_CLIP_CULL_DISTANCE_CULL_MASK__SHIFT = 8
REG_A6XX_GRAS_CL_ARRAY_SIZE = 0x00008004
REG_A6XX_GRAS_CL_INTERP_CNTL = 0x00008005
A6XX_GRAS_CL_INTERP_CNTL_IJ_PERSP_PIXEL = 0x00000001
A6XX_GRAS_CL_INTERP_CNTL_IJ_PERSP_CENTROID = 0x00000002
A6XX_GRAS_CL_INTERP_CNTL_IJ_PERSP_SAMPLE = 0x00000004
A6XX_GRAS_CL_INTERP_CNTL_IJ_LINEAR_PIXEL = 0x00000008
A6XX_GRAS_CL_INTERP_CNTL_IJ_LINEAR_CENTROID = 0x00000010
A6XX_GRAS_CL_INTERP_CNTL_IJ_LINEAR_SAMPLE = 0x00000020
A6XX_GRAS_CL_INTERP_CNTL_COORD_MASK__MASK = 0x000003c0
A6XX_GRAS_CL_INTERP_CNTL_COORD_MASK__SHIFT = 6
A6XX_GRAS_CL_INTERP_CNTL_UNK10 = 0x00000400
A6XX_GRAS_CL_INTERP_CNTL_UNK11 = 0x00000800
REG_A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ = 0x00008006
A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ_HORZ__MASK = 0x000001ff
A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ_HORZ__SHIFT = 0
A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ_VERT__MASK = 0x0007fc00
A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ_VERT__SHIFT = 10
REG_A7XX_GRAS_UNKNOWN_8007 = 0x00008007
REG_A7XX_GRAS_UNKNOWN_8008 = 0x00008008
REG_A7XX_GRAS_UNKNOWN_8009 = 0x00008009
REG_A7XX_GRAS_UNKNOWN_800A = 0x0000800a
REG_A7XX_GRAS_UNKNOWN_800B = 0x0000800b
REG_A7XX_GRAS_UNKNOWN_800C = 0x0000800c
REG_A6XX_GRAS_CL_VIEWPORT = lambda i0: (0x00008010 + 0x6*i0 )
A6XX_GRAS_CL_VIEWPORT_XOFFSET__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_XOFFSET__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_XSCALE__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_XSCALE__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_YOFFSET__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_YOFFSET__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_YSCALE__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_YSCALE__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_ZOFFSET__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_ZOFFSET__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_ZSCALE__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_ZSCALE__SHIFT = 0
REG_A6XX_GRAS_CL_VIEWPORT_ZCLAMP = lambda i0: (0x00008070 + 0x2*i0 )
A6XX_GRAS_CL_VIEWPORT_ZCLAMP_MIN__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_ZCLAMP_MIN__SHIFT = 0
A6XX_GRAS_CL_VIEWPORT_ZCLAMP_MAX__MASK = 0xffffffff
A6XX_GRAS_CL_VIEWPORT_ZCLAMP_MAX__SHIFT = 0
REG_A6XX_GRAS_SU_CNTL = 0x00008090
A6XX_GRAS_SU_CNTL_CULL_FRONT = 0x00000001
A6XX_GRAS_SU_CNTL_CULL_BACK = 0x00000002
A6XX_GRAS_SU_CNTL_FRONT_CW = 0x00000004
A6XX_GRAS_SU_CNTL_LINEHALFWIDTH__MASK = 0x000007f8
A6XX_GRAS_SU_CNTL_LINEHALFWIDTH__SHIFT = 3
A6XX_GRAS_SU_CNTL_POLY_OFFSET = 0x00000800
A6XX_GRAS_SU_CNTL_UNK12 = 0x00001000
A6XX_GRAS_SU_CNTL_LINE_MODE__MASK = 0x00002000
A6XX_GRAS_SU_CNTL_LINE_MODE__SHIFT = 13
A6XX_GRAS_SU_CNTL_UNK15__MASK = 0x00018000
A6XX_GRAS_SU_CNTL_UNK15__SHIFT = 15
A6XX_GRAS_SU_CNTL_MULTIVIEW_ENABLE = 0x00020000
A6XX_GRAS_SU_CNTL_RENDERTARGETINDEXINCR = 0x00040000
A6XX_GRAS_SU_CNTL_VIEWPORTINDEXINCR = 0x00080000
A6XX_GRAS_SU_CNTL_UNK20__MASK = 0x00700000
A6XX_GRAS_SU_CNTL_UNK20__SHIFT = 20
REG_A6XX_GRAS_SU_POINT_MINMAX = 0x00008091
A6XX_GRAS_SU_POINT_MINMAX_MIN__MASK = 0x0000ffff
A6XX_GRAS_SU_POINT_MINMAX_MIN__SHIFT = 0
A6XX_GRAS_SU_POINT_MINMAX_MAX__MASK = 0xffff0000
A6XX_GRAS_SU_POINT_MINMAX_MAX__SHIFT = 16
REG_A6XX_GRAS_SU_POINT_SIZE = 0x00008092
A6XX_GRAS_SU_POINT_SIZE__MASK = 0x0000ffff
A6XX_GRAS_SU_POINT_SIZE__SHIFT = 0
REG_A6XX_GRAS_SU_DEPTH_PLANE_CNTL = 0x00008094
A6XX_GRAS_SU_DEPTH_PLANE_CNTL_Z_MODE__MASK = 0x00000003
A6XX_GRAS_SU_DEPTH_PLANE_CNTL_Z_MODE__SHIFT = 0
REG_A6XX_GRAS_SU_POLY_OFFSET_SCALE = 0x00008095
A6XX_GRAS_SU_POLY_OFFSET_SCALE__MASK = 0xffffffff
A6XX_GRAS_SU_POLY_OFFSET_SCALE__SHIFT = 0
REG_A6XX_GRAS_SU_POLY_OFFSET_OFFSET = 0x00008096
A6XX_GRAS_SU_POLY_OFFSET_OFFSET__MASK = 0xffffffff
A6XX_GRAS_SU_POLY_OFFSET_OFFSET__SHIFT = 0
REG_A6XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP = 0x00008097
A6XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP__MASK = 0xffffffff
A6XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP__SHIFT = 0
REG_A6XX_GRAS_SU_DEPTH_BUFFER_INFO = 0x00008098
A6XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT__MASK = 0x00000007
A6XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT__SHIFT = 0
A6XX_GRAS_SU_DEPTH_BUFFER_INFO_UNK3 = 0x00000008
REG_A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL = 0x00008099
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_CONSERVATIVERASEN = 0x00000001
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_SHIFTAMOUNT__MASK = 0x00000006
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_SHIFTAMOUNT__SHIFT = 1
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_INNERCONSERVATIVERASEN = 0x00000008
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_UNK4__MASK = 0x00000030
A6XX_GRAS_SU_CONSERVATIVE_RAS_CNTL_UNK4__SHIFT = 4
REG_A6XX_GRAS_SU_PATH_RENDERING_CNTL = 0x0000809a
A6XX_GRAS_SU_PATH_RENDERING_CNTL_UNK0 = 0x00000001
A6XX_GRAS_SU_PATH_RENDERING_CNTL_LINELENGTHEN = 0x00000002
REG_A6XX_GRAS_SU_VS_SIV_CNTL = 0x0000809b
A6XX_GRAS_SU_VS_SIV_CNTL_WRITES_LAYER = 0x00000001
A6XX_GRAS_SU_VS_SIV_CNTL_WRITES_VIEW = 0x00000002
REG_A6XX_GRAS_SU_GS_SIV_CNTL = 0x0000809c
A6XX_GRAS_SU_GS_SIV_CNTL_WRITES_LAYER = 0x00000001
A6XX_GRAS_SU_GS_SIV_CNTL_WRITES_VIEW = 0x00000002
REG_A6XX_GRAS_SU_DS_SIV_CNTL = 0x0000809d
A6XX_GRAS_SU_DS_SIV_CNTL_WRITES_LAYER = 0x00000001
A6XX_GRAS_SU_DS_SIV_CNTL_WRITES_VIEW = 0x00000002
REG_A6XX_GRAS_SC_CNTL = 0x000080a0
A6XX_GRAS_SC_CNTL_CCUSINGLECACHELINESIZE__MASK = 0x00000007
A6XX_GRAS_SC_CNTL_CCUSINGLECACHELINESIZE__SHIFT = 0
A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE__MASK = 0x00000018
A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE__SHIFT = 3
A6XX_GRAS_SC_CNTL_RASTER_MODE__MASK = 0x00000020
A6XX_GRAS_SC_CNTL_RASTER_MODE__SHIFT = 5
A6XX_GRAS_SC_CNTL_RASTER_DIRECTION__MASK = 0x000000c0
A6XX_GRAS_SC_CNTL_RASTER_DIRECTION__SHIFT = 6
A6XX_GRAS_SC_CNTL_SEQUENCED_THREAD_DISTRIBUTION__MASK = 0x00000100
A6XX_GRAS_SC_CNTL_SEQUENCED_THREAD_DISTRIBUTION__SHIFT = 8
A6XX_GRAS_SC_CNTL_UNK9 = 0x00000200
A6XX_GRAS_SC_CNTL_ROTATION__MASK = 0x00000c00
A6XX_GRAS_SC_CNTL_ROTATION__SHIFT = 10
A6XX_GRAS_SC_CNTL_EARLYVIZOUTEN = 0x00001000
REG_A6XX_GRAS_SC_BIN_CNTL = 0x000080a1
A6XX_GRAS_SC_BIN_CNTL_BINW__MASK = 0x0000003f
A6XX_GRAS_SC_BIN_CNTL_BINW__SHIFT = 0
A6XX_GRAS_SC_BIN_CNTL_BINH__MASK = 0x00007f00
A6XX_GRAS_SC_BIN_CNTL_BINH__SHIFT = 8
A6XX_GRAS_SC_BIN_CNTL_RENDER_MODE__MASK = 0x001c0000
A6XX_GRAS_SC_BIN_CNTL_RENDER_MODE__SHIFT = 18
A6XX_GRAS_SC_BIN_CNTL_FORCE_LRZ_WRITE_DIS = 0x00200000
A6XX_GRAS_SC_BIN_CNTL_BUFFERS_LOCATION__MASK = 0x00c00000
A6XX_GRAS_SC_BIN_CNTL_BUFFERS_LOCATION__SHIFT = 22
A6XX_GRAS_SC_BIN_CNTL_LRZ_FEEDBACK_ZMODE_MASK__MASK = 0x07000000
A6XX_GRAS_SC_BIN_CNTL_LRZ_FEEDBACK_ZMODE_MASK__SHIFT = 24
A6XX_GRAS_SC_BIN_CNTL_UNK27 = 0x08000000
REG_A6XX_GRAS_SC_RAS_MSAA_CNTL = 0x000080a2
A6XX_GRAS_SC_RAS_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_GRAS_SC_RAS_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_GRAS_SC_RAS_MSAA_CNTL_UNK2 = 0x00000004
A6XX_GRAS_SC_RAS_MSAA_CNTL_UNK3 = 0x00000008
REG_A6XX_GRAS_SC_DEST_MSAA_CNTL = 0x000080a3
A6XX_GRAS_SC_DEST_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_GRAS_SC_DEST_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_GRAS_SC_DEST_MSAA_CNTL_MSAA_DISABLE = 0x00000004
REG_A6XX_GRAS_SC_MSAA_SAMPLE_POS_CNTL = 0x000080a4
A6XX_GRAS_SC_MSAA_SAMPLE_POS_CNTL_UNK0 = 0x00000001
A6XX_GRAS_SC_MSAA_SAMPLE_POS_CNTL_LOCATION_ENABLE = 0x00000002
REG_A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0 = 0x000080a5
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__MASK = 0x0000000f
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__SHIFT = 0
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__SHIFT = 4
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__MASK = 0x00000f00
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__SHIFT = 8
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__SHIFT = 12
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__MASK = 0x000f0000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__SHIFT = 16
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__SHIFT = 20
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__MASK = 0x0f000000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__SHIFT = 24
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__SHIFT = 28
REG_A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1 = 0x000080a6
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__MASK = 0x0000000f
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__SHIFT = 0
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__SHIFT = 4
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__MASK = 0x00000f00
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__SHIFT = 8
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__SHIFT = 12
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__MASK = 0x000f0000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__SHIFT = 16
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__SHIFT = 20
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__MASK = 0x0f000000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__SHIFT = 24
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_GRAS_SC_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__SHIFT = 28
REG_A7XX_GRAS_UNKNOWN_80A7 = 0x000080a7
REG_A6XX_GRAS_UNKNOWN_80AF = 0x000080af
REG_A6XX_GRAS_SC_SCREEN_SCISSOR = lambda i0: (0x000080b0 + 0x2*i0 )
A6XX_GRAS_SC_SCREEN_SCISSOR_TL_X__MASK = 0x0000ffff
A6XX_GRAS_SC_SCREEN_SCISSOR_TL_X__SHIFT = 0
A6XX_GRAS_SC_SCREEN_SCISSOR_TL_Y__MASK = 0xffff0000
A6XX_GRAS_SC_SCREEN_SCISSOR_TL_Y__SHIFT = 16
A6XX_GRAS_SC_SCREEN_SCISSOR_BR_X__MASK = 0x0000ffff
A6XX_GRAS_SC_SCREEN_SCISSOR_BR_X__SHIFT = 0
A6XX_GRAS_SC_SCREEN_SCISSOR_BR_Y__MASK = 0xffff0000
A6XX_GRAS_SC_SCREEN_SCISSOR_BR_Y__SHIFT = 16
REG_A6XX_GRAS_SC_VIEWPORT_SCISSOR = lambda i0: (0x000080d0 + 0x2*i0 )
A6XX_GRAS_SC_VIEWPORT_SCISSOR_TL_X__MASK = 0x0000ffff
A6XX_GRAS_SC_VIEWPORT_SCISSOR_TL_X__SHIFT = 0
A6XX_GRAS_SC_VIEWPORT_SCISSOR_TL_Y__MASK = 0xffff0000
A6XX_GRAS_SC_VIEWPORT_SCISSOR_TL_Y__SHIFT = 16
A6XX_GRAS_SC_VIEWPORT_SCISSOR_BR_X__MASK = 0x0000ffff
A6XX_GRAS_SC_VIEWPORT_SCISSOR_BR_X__SHIFT = 0
A6XX_GRAS_SC_VIEWPORT_SCISSOR_BR_Y__MASK = 0xffff0000
A6XX_GRAS_SC_VIEWPORT_SCISSOR_BR_Y__SHIFT = 16
REG_A6XX_GRAS_SC_WINDOW_SCISSOR_TL = 0x000080f0
A6XX_GRAS_SC_WINDOW_SCISSOR_TL_X__MASK = 0x00003fff
A6XX_GRAS_SC_WINDOW_SCISSOR_TL_X__SHIFT = 0
A6XX_GRAS_SC_WINDOW_SCISSOR_TL_Y__MASK = 0x3fff0000
A6XX_GRAS_SC_WINDOW_SCISSOR_TL_Y__SHIFT = 16
REG_A6XX_GRAS_SC_WINDOW_SCISSOR_BR = 0x000080f1
A6XX_GRAS_SC_WINDOW_SCISSOR_BR_X__MASK = 0x00003fff
A6XX_GRAS_SC_WINDOW_SCISSOR_BR_X__SHIFT = 0
A6XX_GRAS_SC_WINDOW_SCISSOR_BR_Y__MASK = 0x3fff0000
A6XX_GRAS_SC_WINDOW_SCISSOR_BR_Y__SHIFT = 16
REG_A7XX_GRAS_VRS_CONFIG = 0x000080f4
A7XX_GRAS_VRS_CONFIG_PIPELINE_FSR_ENABLE = 0x00000001
A7XX_GRAS_VRS_CONFIG_FRAG_SIZE_X__MASK = 0x00000006
A7XX_GRAS_VRS_CONFIG_FRAG_SIZE_X__SHIFT = 1
A7XX_GRAS_VRS_CONFIG_FRAG_SIZE_Y__MASK = 0x00000018
A7XX_GRAS_VRS_CONFIG_FRAG_SIZE_Y__SHIFT = 3
A7XX_GRAS_VRS_CONFIG_COMBINER_OP_1__MASK = 0x000000e0
A7XX_GRAS_VRS_CONFIG_COMBINER_OP_1__SHIFT = 5
A7XX_GRAS_VRS_CONFIG_COMBINER_OP_2__MASK = 0x00000700
A7XX_GRAS_VRS_CONFIG_COMBINER_OP_2__SHIFT = 8
A7XX_GRAS_VRS_CONFIG_ATTACHMENT_FSR_ENABLE = 0x00002000
A7XX_GRAS_VRS_CONFIG_PRIMITIVE_FSR_ENABLE = 0x00100000
REG_A7XX_GRAS_QUALITY_BUFFER_INFO = 0x000080f5
A7XX_GRAS_QUALITY_BUFFER_INFO_LAYERED = 0x00000001
A7XX_GRAS_QUALITY_BUFFER_INFO_TILE_MODE__MASK = 0x00000006
A7XX_GRAS_QUALITY_BUFFER_INFO_TILE_MODE__SHIFT = 1
REG_A7XX_GRAS_QUALITY_BUFFER_DIMENSION = 0x000080f6
A7XX_GRAS_QUALITY_BUFFER_DIMENSION_WIDTH__MASK = 0x0000ffff
A7XX_GRAS_QUALITY_BUFFER_DIMENSION_WIDTH__SHIFT = 0
A7XX_GRAS_QUALITY_BUFFER_DIMENSION_HEIGHT__MASK = 0xffff0000
A7XX_GRAS_QUALITY_BUFFER_DIMENSION_HEIGHT__SHIFT = 16
REG_A7XX_GRAS_QUALITY_BUFFER_BASE = 0x000080f8
REG_A7XX_GRAS_QUALITY_BUFFER_PITCH = 0x000080fa
A7XX_GRAS_QUALITY_BUFFER_PITCH_PITCH__MASK = 0x000000ff
A7XX_GRAS_QUALITY_BUFFER_PITCH_PITCH__SHIFT = 0
A7XX_GRAS_QUALITY_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x1ffffc00
A7XX_GRAS_QUALITY_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 10
REG_A6XX_GRAS_LRZ_CNTL = 0x00008100
A6XX_GRAS_LRZ_CNTL_ENABLE = 0x00000001
A6XX_GRAS_LRZ_CNTL_LRZ_WRITE = 0x00000002
A6XX_GRAS_LRZ_CNTL_GREATER = 0x00000004
A6XX_GRAS_LRZ_CNTL_FC_ENABLE = 0x00000008
A6XX_GRAS_LRZ_CNTL_Z_WRITE_ENABLE = 0x00000010
A6XX_GRAS_LRZ_CNTL_Z_BOUNDS_ENABLE = 0x00000020
A6XX_GRAS_LRZ_CNTL_DIR__MASK = 0x000000c0
A6XX_GRAS_LRZ_CNTL_DIR__SHIFT = 6
A6XX_GRAS_LRZ_CNTL_DIR_WRITE = 0x00000100
A6XX_GRAS_LRZ_CNTL_DISABLE_ON_WRONG_DIR = 0x00000200
A6XX_GRAS_LRZ_CNTL_Z_FUNC__MASK = 0x00003800
A6XX_GRAS_LRZ_CNTL_Z_FUNC__SHIFT = 11
REG_A6XX_GRAS_LRZ_PS_INPUT_CNTL = 0x00008101
A6XX_GRAS_LRZ_PS_INPUT_CNTL_SAMPLEID = 0x00000001
A6XX_GRAS_LRZ_PS_INPUT_CNTL_FRAGCOORDSAMPLEMODE__MASK = 0x00000006
A6XX_GRAS_LRZ_PS_INPUT_CNTL_FRAGCOORDSAMPLEMODE__SHIFT = 1
REG_A6XX_GRAS_LRZ_MRT_BUFFER_INFO_0 = 0x00008102
A6XX_GRAS_LRZ_MRT_BUFFER_INFO_0_COLOR_FORMAT__MASK = 0x000000ff
A6XX_GRAS_LRZ_MRT_BUFFER_INFO_0_COLOR_FORMAT__SHIFT = 0
REG_A6XX_GRAS_LRZ_BUFFER_BASE = 0x00008103
REG_A6XX_GRAS_LRZ_BUFFER_PITCH = 0x00008105
A6XX_GRAS_LRZ_BUFFER_PITCH_PITCH__MASK = 0x000000ff
A6XX_GRAS_LRZ_BUFFER_PITCH_PITCH__SHIFT = 0
A6XX_GRAS_LRZ_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x1ffffc00
A6XX_GRAS_LRZ_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 10
REG_A6XX_GRAS_LRZ_FAST_CLEAR_BUFFER_BASE = 0x00008106
REG_A6XX_GRAS_LRZ_PS_SAMPLEFREQ_CNTL = 0x00008109
A6XX_GRAS_LRZ_PS_SAMPLEFREQ_CNTL_PER_SAMP_MODE = 0x00000001
REG_A6XX_GRAS_LRZ_VIEW_INFO = 0x0000810a
A6XX_GRAS_LRZ_VIEW_INFO_BASE_LAYER__MASK = 0x000007ff
A6XX_GRAS_LRZ_VIEW_INFO_BASE_LAYER__SHIFT = 0
A6XX_GRAS_LRZ_VIEW_INFO_LAYER_COUNT__MASK = 0x07ff0000
A6XX_GRAS_LRZ_VIEW_INFO_LAYER_COUNT__SHIFT = 16
A6XX_GRAS_LRZ_VIEW_INFO_BASE_MIP_LEVEL__MASK = 0xf0000000
A6XX_GRAS_LRZ_VIEW_INFO_BASE_MIP_LEVEL__SHIFT = 28
REG_A7XX_GRAS_LRZ_CNTL2 = 0x0000810b
A7XX_GRAS_LRZ_CNTL2_DISABLE_ON_WRONG_DIR = 0x00000001
A7XX_GRAS_LRZ_CNTL2_FC_ENABLE = 0x00000002
REG_A6XX_GRAS_UNKNOWN_8110 = 0x00008110
REG_A7XX_GRAS_LRZ_DEPTH_CLEAR = 0x00008111
A7XX_GRAS_LRZ_DEPTH_CLEAR__MASK = 0xffffffff
A7XX_GRAS_LRZ_DEPTH_CLEAR__SHIFT = 0
REG_A7XX_GRAS_LRZ_DEPTH_BUFFER_INFO = 0x00008113
A7XX_GRAS_LRZ_DEPTH_BUFFER_INFO_DEPTH_FORMAT__MASK = 0x00000007
A7XX_GRAS_LRZ_DEPTH_BUFFER_INFO_DEPTH_FORMAT__SHIFT = 0
A7XX_GRAS_LRZ_DEPTH_BUFFER_INFO_UNK3 = 0x00000008
REG_A7XX_GRAS_UNKNOWN_8120 = 0x00008120
REG_A7XX_GRAS_UNKNOWN_8121 = 0x00008121
REG_A6XX_GRAS_A2D_BLT_CNTL = 0x00008400
A6XX_GRAS_A2D_BLT_CNTL_ROTATE__MASK = 0x00000007
A6XX_GRAS_A2D_BLT_CNTL_ROTATE__SHIFT = 0
A6XX_GRAS_A2D_BLT_CNTL_OVERWRITEEN = 0x00000008
A6XX_GRAS_A2D_BLT_CNTL_UNK4__MASK = 0x00000070
A6XX_GRAS_A2D_BLT_CNTL_UNK4__SHIFT = 4
A6XX_GRAS_A2D_BLT_CNTL_SOLID_COLOR = 0x00000080
A6XX_GRAS_A2D_BLT_CNTL_COLOR_FORMAT__MASK = 0x0000ff00
A6XX_GRAS_A2D_BLT_CNTL_COLOR_FORMAT__SHIFT = 8
A6XX_GRAS_A2D_BLT_CNTL_SCISSOR = 0x00010000
A6XX_GRAS_A2D_BLT_CNTL_UNK17__MASK = 0x00060000
A6XX_GRAS_A2D_BLT_CNTL_UNK17__SHIFT = 17
A6XX_GRAS_A2D_BLT_CNTL_D24S8 = 0x00080000
A6XX_GRAS_A2D_BLT_CNTL_MASK__MASK = 0x00f00000
A6XX_GRAS_A2D_BLT_CNTL_MASK__SHIFT = 20
A6XX_GRAS_A2D_BLT_CNTL_IFMT__MASK = 0x07000000
A6XX_GRAS_A2D_BLT_CNTL_IFMT__SHIFT = 24
A6XX_GRAS_A2D_BLT_CNTL_UNK27 = 0x08000000
A6XX_GRAS_A2D_BLT_CNTL_UNK28 = 0x10000000
A6XX_GRAS_A2D_BLT_CNTL_RASTER_MODE__MASK = 0x20000000
A6XX_GRAS_A2D_BLT_CNTL_RASTER_MODE__SHIFT = 29
A6XX_GRAS_A2D_BLT_CNTL_COPY = 0x40000000
REG_A6XX_GRAS_A2D_SRC_XMIN = 0x00008401
A6XX_GRAS_A2D_SRC_XMIN__MASK = 0x01ffff00
A6XX_GRAS_A2D_SRC_XMIN__SHIFT = 8
REG_A6XX_GRAS_A2D_SRC_XMAX = 0x00008402
A6XX_GRAS_A2D_SRC_XMAX__MASK = 0x01ffff00
A6XX_GRAS_A2D_SRC_XMAX__SHIFT = 8
REG_A6XX_GRAS_A2D_SRC_YMIN = 0x00008403
A6XX_GRAS_A2D_SRC_YMIN__MASK = 0x01ffff00
A6XX_GRAS_A2D_SRC_YMIN__SHIFT = 8
REG_A6XX_GRAS_A2D_SRC_YMAX = 0x00008404
A6XX_GRAS_A2D_SRC_YMAX__MASK = 0x01ffff00
A6XX_GRAS_A2D_SRC_YMAX__SHIFT = 8
REG_A6XX_GRAS_A2D_DEST_TL = 0x00008405
A6XX_GRAS_A2D_DEST_TL_X__MASK = 0x00003fff
A6XX_GRAS_A2D_DEST_TL_X__SHIFT = 0
A6XX_GRAS_A2D_DEST_TL_Y__MASK = 0x3fff0000
A6XX_GRAS_A2D_DEST_TL_Y__SHIFT = 16
REG_A6XX_GRAS_A2D_DEST_BR = 0x00008406
A6XX_GRAS_A2D_DEST_BR_X__MASK = 0x00003fff
A6XX_GRAS_A2D_DEST_BR_X__SHIFT = 0
A6XX_GRAS_A2D_DEST_BR_Y__MASK = 0x3fff0000
A6XX_GRAS_A2D_DEST_BR_Y__SHIFT = 16
REG_A6XX_GRAS_2D_UNKNOWN_8407 = 0x00008407
REG_A6XX_GRAS_2D_UNKNOWN_8408 = 0x00008408
REG_A6XX_GRAS_2D_UNKNOWN_8409 = 0x00008409
REG_A6XX_GRAS_A2D_SCISSOR_TL = 0x0000840a
A6XX_GRAS_A2D_SCISSOR_TL_X__MASK = 0x00003fff
A6XX_GRAS_A2D_SCISSOR_TL_X__SHIFT = 0
A6XX_GRAS_A2D_SCISSOR_TL_Y__MASK = 0x3fff0000
A6XX_GRAS_A2D_SCISSOR_TL_Y__SHIFT = 16
REG_A6XX_GRAS_A2D_SCISSOR_BR = 0x0000840b
A6XX_GRAS_A2D_SCISSOR_BR_X__MASK = 0x00003fff
A6XX_GRAS_A2D_SCISSOR_BR_X__SHIFT = 0
A6XX_GRAS_A2D_SCISSOR_BR_Y__MASK = 0x3fff0000
A6XX_GRAS_A2D_SCISSOR_BR_Y__SHIFT = 16
REG_A6XX_GRAS_DBG_ECO_CNTL = 0x00008600
A6XX_GRAS_DBG_ECO_CNTL_UNK7 = 0x00000080
A6XX_GRAS_DBG_ECO_CNTL_LRZCACHELOCKDIS = 0x00000800
REG_A6XX_GRAS_ADDR_MODE_CNTL = 0x00008601
REG_A7XX_GRAS_NC_MODE_CNTL = 0x00008602
REG_A6XX_GRAS_PERFCTR_TSE_SEL = lambda i0: (0x00008610 + 0x1*i0 )
REG_A6XX_GRAS_PERFCTR_RAS_SEL = lambda i0: (0x00008614 + 0x1*i0 )
REG_A6XX_GRAS_PERFCTR_LRZ_SEL = lambda i0: (0x00008618 + 0x1*i0 )
REG_A6XX_RB_CNTL = 0x00008800
A6XX_RB_CNTL_BINW__MASK = 0x0000003f
A6XX_RB_CNTL_BINW__SHIFT = 0
A6XX_RB_CNTL_BINH__MASK = 0x00007f00
A6XX_RB_CNTL_BINH__SHIFT = 8
A6XX_RB_CNTL_RENDER_MODE__MASK = 0x001c0000
A6XX_RB_CNTL_RENDER_MODE__SHIFT = 18
A6XX_RB_CNTL_FORCE_LRZ_WRITE_DIS = 0x00200000
A6XX_RB_CNTL_BUFFERS_LOCATION__MASK = 0x00c00000
A6XX_RB_CNTL_BUFFERS_LOCATION__SHIFT = 22
A6XX_RB_CNTL_LRZ_FEEDBACK_ZMODE_MASK__MASK = 0x07000000
A6XX_RB_CNTL_LRZ_FEEDBACK_ZMODE_MASK__SHIFT = 24
REG_A7XX_RB_CNTL = 0x00008800
A7XX_RB_CNTL_BINW__MASK = 0x0000003f
A7XX_RB_CNTL_BINW__SHIFT = 0
A7XX_RB_CNTL_BINH__MASK = 0x00007f00
A7XX_RB_CNTL_BINH__SHIFT = 8
A7XX_RB_CNTL_RENDER_MODE__MASK = 0x001c0000
A7XX_RB_CNTL_RENDER_MODE__SHIFT = 18
A7XX_RB_CNTL_FORCE_LRZ_WRITE_DIS = 0x00200000
A7XX_RB_CNTL_LRZ_FEEDBACK_ZMODE_MASK__MASK = 0x07000000
A7XX_RB_CNTL_LRZ_FEEDBACK_ZMODE_MASK__SHIFT = 24
REG_A6XX_RB_RENDER_CNTL = 0x00008801
A6XX_RB_RENDER_CNTL_CCUSINGLECACHELINESIZE__MASK = 0x00000038
A6XX_RB_RENDER_CNTL_CCUSINGLECACHELINESIZE__SHIFT = 3
A6XX_RB_RENDER_CNTL_EARLYVIZOUTEN = 0x00000040
A6XX_RB_RENDER_CNTL_FS_DISABLE = 0x00000080
A6XX_RB_RENDER_CNTL_UNK8__MASK = 0x00000700
A6XX_RB_RENDER_CNTL_UNK8__SHIFT = 8
A6XX_RB_RENDER_CNTL_RASTER_MODE__MASK = 0x00000100
A6XX_RB_RENDER_CNTL_RASTER_MODE__SHIFT = 8
A6XX_RB_RENDER_CNTL_RASTER_DIRECTION__MASK = 0x00000600
A6XX_RB_RENDER_CNTL_RASTER_DIRECTION__SHIFT = 9
A6XX_RB_RENDER_CNTL_CONSERVATIVERASEN = 0x00000800
A6XX_RB_RENDER_CNTL_INNERCONSERVATIVERASEN = 0x00001000
A6XX_RB_RENDER_CNTL_FLAG_DEPTH = 0x00004000
A6XX_RB_RENDER_CNTL_FLAG_MRTS__MASK = 0x00ff0000
A6XX_RB_RENDER_CNTL_FLAG_MRTS__SHIFT = 16
REG_A7XX_RB_RENDER_CNTL = 0x00008801
A7XX_RB_RENDER_CNTL_EARLYVIZOUTEN = 0x00000040
A7XX_RB_RENDER_CNTL_FS_DISABLE = 0x00000080
A7XX_RB_RENDER_CNTL_RASTER_MODE__MASK = 0x00000100
A7XX_RB_RENDER_CNTL_RASTER_MODE__SHIFT = 8
A7XX_RB_RENDER_CNTL_RASTER_DIRECTION__MASK = 0x00000600
A7XX_RB_RENDER_CNTL_RASTER_DIRECTION__SHIFT = 9
A7XX_RB_RENDER_CNTL_CONSERVATIVERASEN = 0x00000800
A7XX_RB_RENDER_CNTL_INNERCONSERVATIVERASEN = 0x00001000
REG_A7XX_GRAS_SU_RENDER_CNTL = 0x00008116
A7XX_GRAS_SU_RENDER_CNTL_FS_DISABLE = 0x00000080
REG_A6XX_RB_RAS_MSAA_CNTL = 0x00008802
A6XX_RB_RAS_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_RB_RAS_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_RB_RAS_MSAA_CNTL_UNK2 = 0x00000004
A6XX_RB_RAS_MSAA_CNTL_UNK3 = 0x00000008
REG_A6XX_RB_DEST_MSAA_CNTL = 0x00008803
A6XX_RB_DEST_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_RB_DEST_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_RB_DEST_MSAA_CNTL_MSAA_DISABLE = 0x00000004
REG_A6XX_RB_MSAA_SAMPLE_POS_CNTL = 0x00008804
A6XX_RB_MSAA_SAMPLE_POS_CNTL_UNK0 = 0x00000001
A6XX_RB_MSAA_SAMPLE_POS_CNTL_LOCATION_ENABLE = 0x00000002
REG_A6XX_RB_PROGRAMMABLE_MSAA_POS_0 = 0x00008805
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__MASK = 0x0000000f
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__SHIFT = 0
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__SHIFT = 4
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__MASK = 0x00000f00
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__SHIFT = 8
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__SHIFT = 12
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__MASK = 0x000f0000
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__SHIFT = 16
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__SHIFT = 20
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__MASK = 0x0f000000
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__SHIFT = 24
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_RB_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__SHIFT = 28
REG_A6XX_RB_PROGRAMMABLE_MSAA_POS_1 = 0x00008806
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__MASK = 0x0000000f
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__SHIFT = 0
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__SHIFT = 4
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__MASK = 0x00000f00
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__SHIFT = 8
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__SHIFT = 12
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__MASK = 0x000f0000
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__SHIFT = 16
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__SHIFT = 20
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__MASK = 0x0f000000
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__SHIFT = 24
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_RB_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__SHIFT = 28
REG_A6XX_RB_INTERP_CNTL = 0x00008809
A6XX_RB_INTERP_CNTL_IJ_PERSP_PIXEL = 0x00000001
A6XX_RB_INTERP_CNTL_IJ_PERSP_CENTROID = 0x00000002
A6XX_RB_INTERP_CNTL_IJ_PERSP_SAMPLE = 0x00000004
A6XX_RB_INTERP_CNTL_IJ_LINEAR_PIXEL = 0x00000008
A6XX_RB_INTERP_CNTL_IJ_LINEAR_CENTROID = 0x00000010
A6XX_RB_INTERP_CNTL_IJ_LINEAR_SAMPLE = 0x00000020
A6XX_RB_INTERP_CNTL_COORD_MASK__MASK = 0x000003c0
A6XX_RB_INTERP_CNTL_COORD_MASK__SHIFT = 6
A6XX_RB_INTERP_CNTL_UNK10 = 0x00000400
REG_A6XX_RB_PS_INPUT_CNTL = 0x0000880a
A6XX_RB_PS_INPUT_CNTL_SAMPLEMASK = 0x00000001
A6XX_RB_PS_INPUT_CNTL_POSTDEPTHCOVERAGE = 0x00000002
A6XX_RB_PS_INPUT_CNTL_FACENESS = 0x00000004
A6XX_RB_PS_INPUT_CNTL_SAMPLEID = 0x00000008
A6XX_RB_PS_INPUT_CNTL_FRAGCOORDSAMPLEMODE__MASK = 0x00000030
A6XX_RB_PS_INPUT_CNTL_FRAGCOORDSAMPLEMODE__SHIFT = 4
A6XX_RB_PS_INPUT_CNTL_CENTERRHW = 0x00000040
A6XX_RB_PS_INPUT_CNTL_LINELENGTHEN = 0x00000080
A6XX_RB_PS_INPUT_CNTL_FOVEATION = 0x00000100
REG_A6XX_RB_PS_OUTPUT_CNTL = 0x0000880b
A6XX_RB_PS_OUTPUT_CNTL_DUAL_COLOR_IN_ENABLE = 0x00000001
A6XX_RB_PS_OUTPUT_CNTL_FRAG_WRITES_Z = 0x00000002
A6XX_RB_PS_OUTPUT_CNTL_FRAG_WRITES_SAMPMASK = 0x00000004
A6XX_RB_PS_OUTPUT_CNTL_FRAG_WRITES_STENCILREF = 0x00000008
REG_A6XX_RB_PS_MRT_CNTL = 0x0000880c
A6XX_RB_PS_MRT_CNTL_MRT__MASK = 0x0000000f
A6XX_RB_PS_MRT_CNTL_MRT__SHIFT = 0
REG_A6XX_RB_PS_OUTPUT_MASK = 0x0000880d
A6XX_RB_PS_OUTPUT_MASK_RT0__MASK = 0x0000000f
A6XX_RB_PS_OUTPUT_MASK_RT0__SHIFT = 0
A6XX_RB_PS_OUTPUT_MASK_RT1__MASK = 0x000000f0
A6XX_RB_PS_OUTPUT_MASK_RT1__SHIFT = 4
A6XX_RB_PS_OUTPUT_MASK_RT2__MASK = 0x00000f00
A6XX_RB_PS_OUTPUT_MASK_RT2__SHIFT = 8
A6XX_RB_PS_OUTPUT_MASK_RT3__MASK = 0x0000f000
A6XX_RB_PS_OUTPUT_MASK_RT3__SHIFT = 12
A6XX_RB_PS_OUTPUT_MASK_RT4__MASK = 0x000f0000
A6XX_RB_PS_OUTPUT_MASK_RT4__SHIFT = 16
A6XX_RB_PS_OUTPUT_MASK_RT5__MASK = 0x00f00000
A6XX_RB_PS_OUTPUT_MASK_RT5__SHIFT = 20
A6XX_RB_PS_OUTPUT_MASK_RT6__MASK = 0x0f000000
A6XX_RB_PS_OUTPUT_MASK_RT6__SHIFT = 24
A6XX_RB_PS_OUTPUT_MASK_RT7__MASK = 0xf0000000
A6XX_RB_PS_OUTPUT_MASK_RT7__SHIFT = 28
REG_A6XX_RB_DITHER_CNTL = 0x0000880e
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT0__MASK = 0x00000003
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT0__SHIFT = 0
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT1__MASK = 0x0000000c
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT1__SHIFT = 2
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT2__MASK = 0x00000030
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT2__SHIFT = 4
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT3__MASK = 0x000000c0
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT3__SHIFT = 6
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT4__MASK = 0x00000300
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT4__SHIFT = 8
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT5__MASK = 0x00000c00
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT5__SHIFT = 10
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT6__MASK = 0x00003000
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT6__SHIFT = 12
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT7__MASK = 0x0000c000
A6XX_RB_DITHER_CNTL_DITHER_MODE_MRT7__SHIFT = 14
REG_A6XX_RB_SRGB_CNTL = 0x0000880f
A6XX_RB_SRGB_CNTL_SRGB_MRT0 = 0x00000001
A6XX_RB_SRGB_CNTL_SRGB_MRT1 = 0x00000002
A6XX_RB_SRGB_CNTL_SRGB_MRT2 = 0x00000004
A6XX_RB_SRGB_CNTL_SRGB_MRT3 = 0x00000008
A6XX_RB_SRGB_CNTL_SRGB_MRT4 = 0x00000010
A6XX_RB_SRGB_CNTL_SRGB_MRT5 = 0x00000020
A6XX_RB_SRGB_CNTL_SRGB_MRT6 = 0x00000040
A6XX_RB_SRGB_CNTL_SRGB_MRT7 = 0x00000080
REG_A6XX_RB_PS_SAMPLEFREQ_CNTL = 0x00008810
A6XX_RB_PS_SAMPLEFREQ_CNTL_PER_SAMP_MODE = 0x00000001
REG_A6XX_RB_UNKNOWN_8811 = 0x00008811
REG_A7XX_RB_UNKNOWN_8812 = 0x00008812
REG_A6XX_RB_UNKNOWN_8818 = 0x00008818
REG_A6XX_RB_UNKNOWN_8819 = 0x00008819
REG_A6XX_RB_UNKNOWN_881A = 0x0000881a
REG_A6XX_RB_UNKNOWN_881B = 0x0000881b
REG_A6XX_RB_UNKNOWN_881C = 0x0000881c
REG_A6XX_RB_UNKNOWN_881D = 0x0000881d
REG_A6XX_RB_UNKNOWN_881E = 0x0000881e
REG_A6XX_RB_MRT = lambda i0: (0x00008820 + 0x8*i0 )
A6XX_RB_MRT_CONTROL_BLEND = 0x00000001
A6XX_RB_MRT_CONTROL_BLEND2 = 0x00000002
A6XX_RB_MRT_CONTROL_ROP_ENABLE = 0x00000004
A6XX_RB_MRT_CONTROL_ROP_CODE__MASK = 0x00000078
A6XX_RB_MRT_CONTROL_ROP_CODE__SHIFT = 3
A6XX_RB_MRT_CONTROL_COMPONENT_ENABLE__MASK = 0x00000780
A6XX_RB_MRT_CONTROL_COMPONENT_ENABLE__SHIFT = 7
A6XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR__MASK = 0x0000001f
A6XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR__SHIFT = 0
A6XX_RB_MRT_BLEND_CONTROL_RGB_BLEND_OPCODE__MASK = 0x000000e0
A6XX_RB_MRT_BLEND_CONTROL_RGB_BLEND_OPCODE__SHIFT = 5
A6XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR__MASK = 0x00001f00
A6XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR__SHIFT = 8
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR__MASK = 0x001f0000
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR__SHIFT = 16
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_BLEND_OPCODE__MASK = 0x00e00000
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_BLEND_OPCODE__SHIFT = 21
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR__MASK = 0x1f000000
A6XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR__SHIFT = 24
A6XX_RB_MRT_BUF_INFO_COLOR_FORMAT__MASK = 0x000000ff
A6XX_RB_MRT_BUF_INFO_COLOR_FORMAT__SHIFT = 0
A6XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE__MASK = 0x00000300
A6XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE__SHIFT = 8
A6XX_RB_MRT_BUF_INFO_UNK10 = 0x00000400
A6XX_RB_MRT_BUF_INFO_COLOR_SWAP__MASK = 0x00006000
A6XX_RB_MRT_BUF_INFO_COLOR_SWAP__SHIFT = 13
A7XX_RB_MRT_BUF_INFO_COLOR_FORMAT__MASK = 0x000000ff
A7XX_RB_MRT_BUF_INFO_COLOR_FORMAT__SHIFT = 0
A7XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE__MASK = 0x00000300
A7XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE__SHIFT = 8
A7XX_RB_MRT_BUF_INFO_UNK10 = 0x00000400
A7XX_RB_MRT_BUF_INFO_LOSSLESSCOMPEN = 0x00000800
A7XX_RB_MRT_BUF_INFO_COLOR_SWAP__MASK = 0x00006000
A7XX_RB_MRT_BUF_INFO_COLOR_SWAP__SHIFT = 13
A7XX_RB_MRT_BUF_INFO_MUTABLEEN = 0x00010000
A6XX_RB_MRT_PITCH__MASK = 0xffffffff
A6XX_RB_MRT_PITCH__SHIFT = 0
A6XX_RB_MRT_ARRAY_PITCH__MASK = 0xffffffff
A6XX_RB_MRT_ARRAY_PITCH__SHIFT = 0
REG_A6XX_RB_BLEND_CONSTANT_RED_FP32 = 0x00008860
A6XX_RB_BLEND_CONSTANT_RED_FP32__MASK = 0xffffffff
A6XX_RB_BLEND_CONSTANT_RED_FP32__SHIFT = 0
REG_A6XX_RB_BLEND_CONSTANT_GREEN_FP32 = 0x00008861
A6XX_RB_BLEND_CONSTANT_GREEN_FP32__MASK = 0xffffffff
A6XX_RB_BLEND_CONSTANT_GREEN_FP32__SHIFT = 0
REG_A6XX_RB_BLEND_CONSTANT_BLUE_FP32 = 0x00008862
A6XX_RB_BLEND_CONSTANT_BLUE_FP32__MASK = 0xffffffff
A6XX_RB_BLEND_CONSTANT_BLUE_FP32__SHIFT = 0
REG_A6XX_RB_BLEND_CONSTANT_ALPHA_FP32 = 0x00008863
A6XX_RB_BLEND_CONSTANT_ALPHA_FP32__MASK = 0xffffffff
A6XX_RB_BLEND_CONSTANT_ALPHA_FP32__SHIFT = 0
REG_A6XX_RB_ALPHA_TEST_CNTL = 0x00008864
A6XX_RB_ALPHA_TEST_CNTL_ALPHA_REF__MASK = 0x000000ff
A6XX_RB_ALPHA_TEST_CNTL_ALPHA_REF__SHIFT = 0
A6XX_RB_ALPHA_TEST_CNTL_ALPHA_TEST = 0x00000100
A6XX_RB_ALPHA_TEST_CNTL_ALPHA_TEST_FUNC__MASK = 0x00000e00
A6XX_RB_ALPHA_TEST_CNTL_ALPHA_TEST_FUNC__SHIFT = 9
REG_A6XX_RB_BLEND_CNTL = 0x00008865
A6XX_RB_BLEND_CNTL_BLEND_READS_DEST__MASK = 0x000000ff
A6XX_RB_BLEND_CNTL_BLEND_READS_DEST__SHIFT = 0
A6XX_RB_BLEND_CNTL_INDEPENDENT_BLEND = 0x00000100
A6XX_RB_BLEND_CNTL_DUAL_COLOR_IN_ENABLE = 0x00000200
A6XX_RB_BLEND_CNTL_ALPHA_TO_COVERAGE = 0x00000400
A6XX_RB_BLEND_CNTL_ALPHA_TO_ONE = 0x00000800
A6XX_RB_BLEND_CNTL_SAMPLE_MASK__MASK = 0xffff0000
A6XX_RB_BLEND_CNTL_SAMPLE_MASK__SHIFT = 16
REG_A6XX_RB_DEPTH_PLANE_CNTL = 0x00008870
A6XX_RB_DEPTH_PLANE_CNTL_Z_MODE__MASK = 0x00000003
A6XX_RB_DEPTH_PLANE_CNTL_Z_MODE__SHIFT = 0
REG_A6XX_RB_DEPTH_CNTL = 0x00008871
A6XX_RB_DEPTH_CNTL_Z_TEST_ENABLE = 0x00000001
A6XX_RB_DEPTH_CNTL_Z_WRITE_ENABLE = 0x00000002
A6XX_RB_DEPTH_CNTL_ZFUNC__MASK = 0x0000001c
A6XX_RB_DEPTH_CNTL_ZFUNC__SHIFT = 2
A6XX_RB_DEPTH_CNTL_Z_CLAMP_ENABLE = 0x00000020
A6XX_RB_DEPTH_CNTL_Z_READ_ENABLE = 0x00000040
A6XX_RB_DEPTH_CNTL_Z_BOUNDS_ENABLE = 0x00000080
REG_A6XX_GRAS_SU_DEPTH_CNTL = 0x00008114
A6XX_GRAS_SU_DEPTH_CNTL_Z_TEST_ENABLE = 0x00000001
REG_A6XX_RB_DEPTH_BUFFER_INFO = 0x00008872
A6XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT__MASK = 0x00000007
A6XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT__SHIFT = 0
A6XX_RB_DEPTH_BUFFER_INFO_UNK3__MASK = 0x00000018
A6XX_RB_DEPTH_BUFFER_INFO_UNK3__SHIFT = 3
REG_A7XX_RB_DEPTH_BUFFER_INFO = 0x00008872
A7XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT__MASK = 0x00000007
A7XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT__SHIFT = 0
A7XX_RB_DEPTH_BUFFER_INFO_UNK3__MASK = 0x00000018
A7XX_RB_DEPTH_BUFFER_INFO_UNK3__SHIFT = 3
A7XX_RB_DEPTH_BUFFER_INFO_TILEMODE__MASK = 0x00000060
A7XX_RB_DEPTH_BUFFER_INFO_TILEMODE__SHIFT = 5
A7XX_RB_DEPTH_BUFFER_INFO_LOSSLESSCOMPEN = 0x00000080
REG_A6XX_RB_DEPTH_BUFFER_PITCH = 0x00008873
A6XX_RB_DEPTH_BUFFER_PITCH__MASK = 0x00003fff
A6XX_RB_DEPTH_BUFFER_PITCH__SHIFT = 0
REG_A6XX_RB_DEPTH_BUFFER_ARRAY_PITCH = 0x00008874
A6XX_RB_DEPTH_BUFFER_ARRAY_PITCH__MASK = 0x0fffffff
A6XX_RB_DEPTH_BUFFER_ARRAY_PITCH__SHIFT = 0
REG_A6XX_RB_DEPTH_BUFFER_BASE = 0x00008875
REG_A6XX_RB_DEPTH_GMEM_BASE = 0x00008877
REG_A6XX_RB_DEPTH_BOUND_MIN = 0x00008878
A6XX_RB_DEPTH_BOUND_MIN__MASK = 0xffffffff
A6XX_RB_DEPTH_BOUND_MIN__SHIFT = 0
REG_A6XX_RB_DEPTH_BOUND_MAX = 0x00008879
A6XX_RB_DEPTH_BOUND_MAX__MASK = 0xffffffff
A6XX_RB_DEPTH_BOUND_MAX__SHIFT = 0
REG_A6XX_RB_STENCIL_CNTL = 0x00008880
A6XX_RB_STENCIL_CNTL_STENCIL_ENABLE = 0x00000001
A6XX_RB_STENCIL_CNTL_STENCIL_ENABLE_BF = 0x00000002
A6XX_RB_STENCIL_CNTL_STENCIL_READ = 0x00000004
A6XX_RB_STENCIL_CNTL_FUNC__MASK = 0x00000700
A6XX_RB_STENCIL_CNTL_FUNC__SHIFT = 8
A6XX_RB_STENCIL_CNTL_FAIL__MASK = 0x00003800
A6XX_RB_STENCIL_CNTL_FAIL__SHIFT = 11
A6XX_RB_STENCIL_CNTL_ZPASS__MASK = 0x0001c000
A6XX_RB_STENCIL_CNTL_ZPASS__SHIFT = 14
A6XX_RB_STENCIL_CNTL_ZFAIL__MASK = 0x000e0000
A6XX_RB_STENCIL_CNTL_ZFAIL__SHIFT = 17
A6XX_RB_STENCIL_CNTL_FUNC_BF__MASK = 0x00700000
A6XX_RB_STENCIL_CNTL_FUNC_BF__SHIFT = 20
A6XX_RB_STENCIL_CNTL_FAIL_BF__MASK = 0x03800000
A6XX_RB_STENCIL_CNTL_FAIL_BF__SHIFT = 23
A6XX_RB_STENCIL_CNTL_ZPASS_BF__MASK = 0x1c000000
A6XX_RB_STENCIL_CNTL_ZPASS_BF__SHIFT = 26
A6XX_RB_STENCIL_CNTL_ZFAIL_BF__MASK = 0xe0000000
A6XX_RB_STENCIL_CNTL_ZFAIL_BF__SHIFT = 29
REG_A6XX_GRAS_SU_STENCIL_CNTL = 0x00008115
A6XX_GRAS_SU_STENCIL_CNTL_STENCIL_ENABLE = 0x00000001
REG_A6XX_RB_STENCIL_BUFFER_INFO = 0x00008881
A6XX_RB_STENCIL_BUFFER_INFO_SEPARATE_STENCIL = 0x00000001
A6XX_RB_STENCIL_BUFFER_INFO_UNK1 = 0x00000002
REG_A7XX_RB_STENCIL_BUFFER_INFO = 0x00008881
A7XX_RB_STENCIL_BUFFER_INFO_SEPARATE_STENCIL = 0x00000001
A7XX_RB_STENCIL_BUFFER_INFO_UNK1 = 0x00000002
A7XX_RB_STENCIL_BUFFER_INFO_TILEMODE__MASK = 0x0000000c
A7XX_RB_STENCIL_BUFFER_INFO_TILEMODE__SHIFT = 2
REG_A6XX_RB_STENCIL_BUFFER_PITCH = 0x00008882
A6XX_RB_STENCIL_BUFFER_PITCH__MASK = 0x00000fff
A6XX_RB_STENCIL_BUFFER_PITCH__SHIFT = 0
REG_A6XX_RB_STENCIL_BUFFER_ARRAY_PITCH = 0x00008883
A6XX_RB_STENCIL_BUFFER_ARRAY_PITCH__MASK = 0x00ffffff
A6XX_RB_STENCIL_BUFFER_ARRAY_PITCH__SHIFT = 0
REG_A6XX_RB_STENCIL_BUFFER_BASE = 0x00008884
REG_A6XX_RB_STENCIL_GMEM_BASE = 0x00008886
REG_A6XX_RB_STENCIL_REF_CNTL = 0x00008887
A6XX_RB_STENCIL_REF_CNTL_REF__MASK = 0x000000ff
A6XX_RB_STENCIL_REF_CNTL_REF__SHIFT = 0
A6XX_RB_STENCIL_REF_CNTL_BFREF__MASK = 0x0000ff00
A6XX_RB_STENCIL_REF_CNTL_BFREF__SHIFT = 8
REG_A6XX_RB_STENCIL_MASK = 0x00008888
A6XX_RB_STENCIL_MASK_MASK__MASK = 0x000000ff
A6XX_RB_STENCIL_MASK_MASK__SHIFT = 0
A6XX_RB_STENCIL_MASK_BFMASK__MASK = 0x0000ff00
A6XX_RB_STENCIL_MASK_BFMASK__SHIFT = 8
REG_A6XX_RB_STENCIL_WRITE_MASK = 0x00008889
A6XX_RB_STENCIL_WRITE_MASK_WRMASK__MASK = 0x000000ff
A6XX_RB_STENCIL_WRITE_MASK_WRMASK__SHIFT = 0
A6XX_RB_STENCIL_WRITE_MASK_BFWRMASK__MASK = 0x0000ff00
A6XX_RB_STENCIL_WRITE_MASK_BFWRMASK__SHIFT = 8
REG_A6XX_RB_WINDOW_OFFSET = 0x00008890
A6XX_RB_WINDOW_OFFSET_X__MASK = 0x00003fff
A6XX_RB_WINDOW_OFFSET_X__SHIFT = 0
A6XX_RB_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A6XX_RB_WINDOW_OFFSET_Y__SHIFT = 16
REG_A6XX_RB_SAMPLE_COUNTER_CNTL = 0x00008891
A6XX_RB_SAMPLE_COUNTER_CNTL_DISABLE = 0x00000001
A6XX_RB_SAMPLE_COUNTER_CNTL_COPY = 0x00000002
REG_A6XX_RB_LRZ_CNTL = 0x00008898
A6XX_RB_LRZ_CNTL_ENABLE = 0x00000001
REG_A7XX_RB_UNKNOWN_8899 = 0x00008899
REG_A6XX_RB_VIEWPORT_ZCLAMP_MIN = 0x000088c0
A6XX_RB_VIEWPORT_ZCLAMP_MIN__MASK = 0xffffffff
A6XX_RB_VIEWPORT_ZCLAMP_MIN__SHIFT = 0
REG_A6XX_RB_VIEWPORT_ZCLAMP_MAX = 0x000088c1
A6XX_RB_VIEWPORT_ZCLAMP_MAX__MASK = 0xffffffff
A6XX_RB_VIEWPORT_ZCLAMP_MAX__SHIFT = 0
REG_A6XX_RB_RESOLVE_CNTL_0 = 0x000088d0
A6XX_RB_RESOLVE_CNTL_0_UNK0__MASK = 0x00001fff
A6XX_RB_RESOLVE_CNTL_0_UNK0__SHIFT = 0
A6XX_RB_RESOLVE_CNTL_0_UNK16__MASK = 0x07ff0000
A6XX_RB_RESOLVE_CNTL_0_UNK16__SHIFT = 16
REG_A6XX_RB_RESOLVE_CNTL_1 = 0x000088d1
A6XX_RB_RESOLVE_CNTL_1_X__MASK = 0x00003fff
A6XX_RB_RESOLVE_CNTL_1_X__SHIFT = 0
A6XX_RB_RESOLVE_CNTL_1_Y__MASK = 0x3fff0000
A6XX_RB_RESOLVE_CNTL_1_Y__SHIFT = 16
REG_A6XX_RB_RESOLVE_CNTL_2 = 0x000088d2
A6XX_RB_RESOLVE_CNTL_2_X__MASK = 0x00003fff
A6XX_RB_RESOLVE_CNTL_2_X__SHIFT = 0
A6XX_RB_RESOLVE_CNTL_2_Y__MASK = 0x3fff0000
A6XX_RB_RESOLVE_CNTL_2_Y__SHIFT = 16
REG_A6XX_RB_RESOLVE_CNTL_3 = 0x000088d3
A6XX_RB_RESOLVE_CNTL_3_BINW__MASK = 0x0000003f
A6XX_RB_RESOLVE_CNTL_3_BINW__SHIFT = 0
A6XX_RB_RESOLVE_CNTL_3_BINH__MASK = 0x00007f00
A6XX_RB_RESOLVE_CNTL_3_BINH__SHIFT = 8
REG_A6XX_RB_RESOLVE_WINDOW_OFFSET = 0x000088d4
A6XX_RB_RESOLVE_WINDOW_OFFSET_X__MASK = 0x00003fff
A6XX_RB_RESOLVE_WINDOW_OFFSET_X__SHIFT = 0
A6XX_RB_RESOLVE_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A6XX_RB_RESOLVE_WINDOW_OFFSET_Y__SHIFT = 16
REG_A6XX_RB_RESOLVE_GMEM_BUFFER_INFO = 0x000088d5
A6XX_RB_RESOLVE_GMEM_BUFFER_INFO_SAMPLES__MASK = 0x00000018
A6XX_RB_RESOLVE_GMEM_BUFFER_INFO_SAMPLES__SHIFT = 3
REG_A6XX_RB_RESOLVE_GMEM_BUFFER_BASE = 0x000088d6
REG_A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO = 0x000088d7
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_TILE_MODE__MASK = 0x00000003
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_TILE_MODE__SHIFT = 0
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_FLAGS = 0x00000004
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_SAMPLES__MASK = 0x00000018
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_SAMPLES__SHIFT = 3
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_COLOR_SWAP__MASK = 0x00000060
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_COLOR_SWAP__SHIFT = 5
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_COLOR_FORMAT__MASK = 0x00007f80
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_COLOR_FORMAT__SHIFT = 7
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_UNK15 = 0x00008000
A6XX_RB_RESOLVE_SYSTEM_BUFFER_INFO_MUTABLEEN = 0x00010000
REG_A6XX_RB_RESOLVE_SYSTEM_BUFFER_BASE = 0x000088d8
REG_A6XX_RB_RESOLVE_SYSTEM_BUFFER_PITCH = 0x000088da
A6XX_RB_RESOLVE_SYSTEM_BUFFER_PITCH__MASK = 0x0000ffff
A6XX_RB_RESOLVE_SYSTEM_BUFFER_PITCH__SHIFT = 0
REG_A6XX_RB_RESOLVE_SYSTEM_BUFFER_ARRAY_PITCH = 0x000088db
A6XX_RB_RESOLVE_SYSTEM_BUFFER_ARRAY_PITCH__MASK = 0x1fffffff
A6XX_RB_RESOLVE_SYSTEM_BUFFER_ARRAY_PITCH__SHIFT = 0
REG_A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_BASE = 0x000088dc
REG_A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_PITCH = 0x000088de
A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_PITCH_PITCH__MASK = 0x000007ff
A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_PITCH_PITCH__SHIFT = 0
A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x0ffff800
A6XX_RB_RESOLVE_SYSTEM_FLAG_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 11
REG_A6XX_RB_RESOLVE_CLEAR_COLOR_DW0 = 0x000088df
REG_A6XX_RB_RESOLVE_CLEAR_COLOR_DW1 = 0x000088e0
REG_A6XX_RB_RESOLVE_CLEAR_COLOR_DW2 = 0x000088e1
REG_A6XX_RB_RESOLVE_CLEAR_COLOR_DW3 = 0x000088e2
REG_A6XX_RB_RESOLVE_OPERATION = 0x000088e3
A6XX_RB_RESOLVE_OPERATION_TYPE__MASK = 0x00000003
A6XX_RB_RESOLVE_OPERATION_TYPE__SHIFT = 0
A6XX_RB_RESOLVE_OPERATION_SAMPLE_0 = 0x00000004
A6XX_RB_RESOLVE_OPERATION_DEPTH = 0x00000008
A6XX_RB_RESOLVE_OPERATION_CLEAR_MASK__MASK = 0x000000f0
A6XX_RB_RESOLVE_OPERATION_CLEAR_MASK__SHIFT = 4
A6XX_RB_RESOLVE_OPERATION_LAST__MASK = 0x00000300
A6XX_RB_RESOLVE_OPERATION_LAST__SHIFT = 8
A6XX_RB_RESOLVE_OPERATION_BUFFER_ID__MASK = 0x0000f000
A6XX_RB_RESOLVE_OPERATION_BUFFER_ID__SHIFT = 12
REG_A7XX_RB_CLEAR_TARGET = 0x000088e4
A7XX_RB_CLEAR_TARGET_CLEAR_MODE__MASK = 0x00000001
A7XX_RB_CLEAR_TARGET_CLEAR_MODE__SHIFT = 0
REG_A7XX_RB_CCU_CACHE_CNTL = 0x000088e5
A7XX_RB_CCU_CACHE_CNTL_DEPTH_OFFSET_HI__MASK = 0x00000001
A7XX_RB_CCU_CACHE_CNTL_DEPTH_OFFSET_HI__SHIFT = 0
A7XX_RB_CCU_CACHE_CNTL_COLOR_OFFSET_HI__MASK = 0x00000004
A7XX_RB_CCU_CACHE_CNTL_COLOR_OFFSET_HI__SHIFT = 2
A7XX_RB_CCU_CACHE_CNTL_DEPTH_CACHE_SIZE__MASK = 0x00000c00
A7XX_RB_CCU_CACHE_CNTL_DEPTH_CACHE_SIZE__SHIFT = 10
A7XX_RB_CCU_CACHE_CNTL_DEPTH_OFFSET__MASK = 0x001ff000
A7XX_RB_CCU_CACHE_CNTL_DEPTH_OFFSET__SHIFT = 12
A7XX_RB_CCU_CACHE_CNTL_COLOR_CACHE_SIZE__MASK = 0x00600000
A7XX_RB_CCU_CACHE_CNTL_COLOR_CACHE_SIZE__SHIFT = 21
A7XX_RB_CCU_CACHE_CNTL_COLOR_OFFSET__MASK = 0xff800000
A7XX_RB_CCU_CACHE_CNTL_COLOR_OFFSET__SHIFT = 23
REG_A6XX_RB_UNKNOWN_88F0 = 0x000088f0
REG_A6XX_RB_UNK_FLAG_BUFFER_BASE = 0x000088f1
REG_A6XX_RB_UNK_FLAG_BUFFER_PITCH = 0x000088f3
A6XX_RB_UNK_FLAG_BUFFER_PITCH_PITCH__MASK = 0x000007ff
A6XX_RB_UNK_FLAG_BUFFER_PITCH_PITCH__SHIFT = 0
A6XX_RB_UNK_FLAG_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x00fff800
A6XX_RB_UNK_FLAG_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 11
REG_A6XX_RB_VRS_CONFIG = 0x000088f4
A6XX_RB_VRS_CONFIG_UNK2 = 0x00000004
A6XX_RB_VRS_CONFIG_PIPELINE_FSR_ENABLE = 0x00000010
A6XX_RB_VRS_CONFIG_ATTACHMENT_FSR_ENABLE = 0x00000020
A6XX_RB_VRS_CONFIG_PRIMITIVE_FSR_ENABLE = 0x00040000
REG_A7XX_RB_UNKNOWN_88F5 = 0x000088f5
REG_A6XX_RB_DEPTH_FLAG_BUFFER_BASE = 0x00008900
REG_A6XX_RB_DEPTH_FLAG_BUFFER_PITCH = 0x00008902
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_PITCH__MASK = 0x0000007f
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_PITCH__SHIFT = 0
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_UNK8__MASK = 0x00000700
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_UNK8__SHIFT = 8
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x0ffff800
A6XX_RB_DEPTH_FLAG_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 11
REG_A6XX_RB_COLOR_FLAG_BUFFER = lambda i0: (0x00008903 + 0x3*i0 )
A6XX_RB_COLOR_FLAG_BUFFER_PITCH_PITCH__MASK = 0x000007ff
A6XX_RB_COLOR_FLAG_BUFFER_PITCH_PITCH__SHIFT = 0
A6XX_RB_COLOR_FLAG_BUFFER_PITCH_ARRAY_PITCH__MASK = 0x1ffff800
A6XX_RB_COLOR_FLAG_BUFFER_PITCH_ARRAY_PITCH__SHIFT = 11
REG_A6XX_RB_SAMPLE_COUNTER_BASE = 0x00008927
REG_A6XX_RB_UNKNOWN_8A00 = 0x00008a00
REG_A6XX_RB_UNKNOWN_8A10 = 0x00008a10
REG_A6XX_RB_UNKNOWN_8A20 = 0x00008a20
REG_A6XX_RB_UNKNOWN_8A30 = 0x00008a30
REG_A6XX_RB_A2D_BLT_CNTL = 0x00008c00
A6XX_RB_A2D_BLT_CNTL_ROTATE__MASK = 0x00000007
A6XX_RB_A2D_BLT_CNTL_ROTATE__SHIFT = 0
A6XX_RB_A2D_BLT_CNTL_OVERWRITEEN = 0x00000008
A6XX_RB_A2D_BLT_CNTL_UNK4__MASK = 0x00000070
A6XX_RB_A2D_BLT_CNTL_UNK4__SHIFT = 4
A6XX_RB_A2D_BLT_CNTL_SOLID_COLOR = 0x00000080
A6XX_RB_A2D_BLT_CNTL_COLOR_FORMAT__MASK = 0x0000ff00
A6XX_RB_A2D_BLT_CNTL_COLOR_FORMAT__SHIFT = 8
A6XX_RB_A2D_BLT_CNTL_SCISSOR = 0x00010000
A6XX_RB_A2D_BLT_CNTL_UNK17__MASK = 0x00060000
A6XX_RB_A2D_BLT_CNTL_UNK17__SHIFT = 17
A6XX_RB_A2D_BLT_CNTL_D24S8 = 0x00080000
A6XX_RB_A2D_BLT_CNTL_MASK__MASK = 0x00f00000
A6XX_RB_A2D_BLT_CNTL_MASK__SHIFT = 20
A6XX_RB_A2D_BLT_CNTL_IFMT__MASK = 0x07000000
A6XX_RB_A2D_BLT_CNTL_IFMT__SHIFT = 24
A6XX_RB_A2D_BLT_CNTL_UNK27 = 0x08000000
A6XX_RB_A2D_BLT_CNTL_UNK28 = 0x10000000
A6XX_RB_A2D_BLT_CNTL_RASTER_MODE__MASK = 0x20000000
A6XX_RB_A2D_BLT_CNTL_RASTER_MODE__SHIFT = 29
A6XX_RB_A2D_BLT_CNTL_COPY = 0x40000000
REG_A6XX_RB_A2D_PIXEL_CNTL = 0x00008c01
REG_A6XX_RB_A2D_DEST_BUFFER_INFO = 0x00008c17
A6XX_RB_A2D_DEST_BUFFER_INFO_COLOR_FORMAT__MASK = 0x000000ff
A6XX_RB_A2D_DEST_BUFFER_INFO_COLOR_FORMAT__SHIFT = 0
A6XX_RB_A2D_DEST_BUFFER_INFO_TILE_MODE__MASK = 0x00000300
A6XX_RB_A2D_DEST_BUFFER_INFO_TILE_MODE__SHIFT = 8
A6XX_RB_A2D_DEST_BUFFER_INFO_COLOR_SWAP__MASK = 0x00000c00
A6XX_RB_A2D_DEST_BUFFER_INFO_COLOR_SWAP__SHIFT = 10
A6XX_RB_A2D_DEST_BUFFER_INFO_FLAGS = 0x00001000
A6XX_RB_A2D_DEST_BUFFER_INFO_SRGB = 0x00002000
A6XX_RB_A2D_DEST_BUFFER_INFO_SAMPLES__MASK = 0x0000c000
A6XX_RB_A2D_DEST_BUFFER_INFO_SAMPLES__SHIFT = 14
A6XX_RB_A2D_DEST_BUFFER_INFO_MUTABLEEN = 0x00020000
REG_A6XX_RB_A2D_DEST_BUFFER_BASE = 0x00008c18
REG_A6XX_RB_A2D_DEST_BUFFER_PITCH = 0x00008c1a
A6XX_RB_A2D_DEST_BUFFER_PITCH__MASK = 0x0000ffff
A6XX_RB_A2D_DEST_BUFFER_PITCH__SHIFT = 0
REG_A6XX_RB_A2D_DEST_BUFFER_BASE_1 = 0x00008c1b
REG_A6XX_RB_A2D_DEST_BUFFER_PITCH_1 = 0x00008c1d
A6XX_RB_A2D_DEST_BUFFER_PITCH_1__MASK = 0x0000ffff
A6XX_RB_A2D_DEST_BUFFER_PITCH_1__SHIFT = 0
REG_A6XX_RB_A2D_DEST_BUFFER_BASE_2 = 0x00008c1e
REG_A6XX_RB_A2D_DEST_FLAG_BUFFER_BASE = 0x00008c20
REG_A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH = 0x00008c22
A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH__MASK = 0x000000ff
A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH__SHIFT = 0
REG_A6XX_RB_A2D_DEST_FLAG_BUFFER_BASE_1 = 0x00008c23
REG_A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH_1 = 0x00008c25
A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH_1__MASK = 0x000000ff
A6XX_RB_A2D_DEST_FLAG_BUFFER_PITCH_1__SHIFT = 0
REG_A6XX_RB_A2D_CLEAR_COLOR_DW0 = 0x00008c2c
REG_A6XX_RB_A2D_CLEAR_COLOR_DW1 = 0x00008c2d
REG_A6XX_RB_A2D_CLEAR_COLOR_DW2 = 0x00008c2e
REG_A6XX_RB_A2D_CLEAR_COLOR_DW3 = 0x00008c2f
REG_A7XX_RB_UNKNOWN_8C34 = 0x00008c34
REG_A6XX_RB_UNKNOWN_8E01 = 0x00008e01
REG_A6XX_RB_DBG_ECO_CNTL = 0x00008e04
REG_A6XX_RB_ADDR_MODE_CNTL = 0x00008e05
REG_A7XX_RB_CCU_DBG_ECO_CNTL = 0x00008e06
REG_A6XX_RB_CCU_CNTL = 0x00008e07
A6XX_RB_CCU_CNTL_GMEM_FAST_CLEAR_DISABLE = 0x00000001
A6XX_RB_CCU_CNTL_CONCURRENT_RESOLVE = 0x00000004
A6XX_RB_CCU_CNTL_DEPTH_OFFSET_HI__MASK = 0x00000080
A6XX_RB_CCU_CNTL_DEPTH_OFFSET_HI__SHIFT = 7
A6XX_RB_CCU_CNTL_COLOR_OFFSET_HI__MASK = 0x00000200
A6XX_RB_CCU_CNTL_COLOR_OFFSET_HI__SHIFT = 9
A6XX_RB_CCU_CNTL_DEPTH_CACHE_SIZE__MASK = 0x00000c00
A6XX_RB_CCU_CNTL_DEPTH_CACHE_SIZE__SHIFT = 10
A6XX_RB_CCU_CNTL_DEPTH_OFFSET__MASK = 0x001ff000
A6XX_RB_CCU_CNTL_DEPTH_OFFSET__SHIFT = 12
A6XX_RB_CCU_CNTL_COLOR_CACHE_SIZE__MASK = 0x00600000
A6XX_RB_CCU_CNTL_COLOR_CACHE_SIZE__SHIFT = 21
A6XX_RB_CCU_CNTL_COLOR_OFFSET__MASK = 0xff800000
A6XX_RB_CCU_CNTL_COLOR_OFFSET__SHIFT = 23
REG_A7XX_RB_CCU_CNTL = 0x00008e07
A7XX_RB_CCU_CNTL_GMEM_FAST_CLEAR_DISABLE = 0x00000001
A7XX_RB_CCU_CNTL_CONCURRENT_RESOLVE_MODE__MASK = 0x0000000c
A7XX_RB_CCU_CNTL_CONCURRENT_RESOLVE_MODE__SHIFT = 2
A7XX_RB_CCU_CNTL_CONCURRENT_UNRESOLVE_MODE__MASK = 0x00000060
A7XX_RB_CCU_CNTL_CONCURRENT_UNRESOLVE_MODE__SHIFT = 5
REG_A6XX_RB_NC_MODE_CNTL = 0x00008e08
A6XX_RB_NC_MODE_CNTL_MODE = 0x00000001
A6XX_RB_NC_MODE_CNTL_LOWER_BIT__MASK = 0x00000006
A6XX_RB_NC_MODE_CNTL_LOWER_BIT__SHIFT = 1
A6XX_RB_NC_MODE_CNTL_MIN_ACCESS_LENGTH = 0x00000008
A6XX_RB_NC_MODE_CNTL_AMSBC = 0x00000010
A6XX_RB_NC_MODE_CNTL_UPPER_BIT__MASK = 0x00000400
A6XX_RB_NC_MODE_CNTL_UPPER_BIT__SHIFT = 10
A6XX_RB_NC_MODE_CNTL_RGB565_PREDICATOR = 0x00000800
A6XX_RB_NC_MODE_CNTL_UNK12__MASK = 0x00003000
A6XX_RB_NC_MODE_CNTL_UNK12__SHIFT = 12
REG_A7XX_RB_UNKNOWN_8E09 = 0x00008e09
REG_A6XX_RB_PERFCTR_RB_SEL = lambda i0: (0x00008e10 + 0x1*i0 )
REG_A6XX_RB_PERFCTR_CCU_SEL = lambda i0: (0x00008e18 + 0x1*i0 )
REG_A6XX_RB_CMP_DBG_ECO_CNTL = 0x00008e28
REG_A6XX_RB_PERFCTR_CMP_SEL = lambda i0: (0x00008e2c + 0x1*i0 )
REG_A7XX_RB_PERFCTR_UFC_SEL = lambda i0: (0x00008e30 + 0x1*i0 )
REG_A6XX_RB_RB_SUB_BLOCK_SEL_CNTL_HOST = 0x00008e3b
REG_A6XX_RB_RB_SUB_BLOCK_SEL_CNTL_CD = 0x00008e3d
REG_A6XX_RB_CONTEXT_SWITCH_GMEM_SAVE_RESTORE_ENABLE = 0x00008e50
REG_A6XX_RB_CONTEXT_SWITCH_GMEM_SAVE_RESTORE_ADDR = 0x00008e51
REG_A7XX_RB_UNKNOWN_8E79 = 0x00008e79
REG_A6XX_VPC_GS_PARAM = 0x00009100
A6XX_VPC_GS_PARAM_LINELENGTHLOC__MASK = 0x000000ff
A6XX_VPC_GS_PARAM_LINELENGTHLOC__SHIFT = 0
REG_A6XX_VPC_VS_CLIP_CULL_CNTL = 0x00009101
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_MASK__SHIFT = 0
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_VS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_GS_CLIP_CULL_CNTL = 0x00009102
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_MASK__SHIFT = 0
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_GS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_DS_CLIP_CULL_CNTL = 0x00009103
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_MASK__SHIFT = 0
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_DS_CLIP_CULL_CNTL_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_VS_CLIP_CULL_CNTL_V2 = 0x00009311
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_MASK__SHIFT = 0
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_VS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_GS_CLIP_CULL_CNTL_V2 = 0x00009312
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_MASK__SHIFT = 0
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_GS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_DS_CLIP_CULL_CNTL_V2 = 0x00009313
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_MASK__MASK = 0x000000ff
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_MASK__SHIFT = 0
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__MASK = 0x0000ff00
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_DIST_03_LOC__SHIFT = 8
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__MASK = 0x00ff0000
A6XX_VPC_DS_CLIP_CULL_CNTL_V2_CLIP_DIST_47_LOC__SHIFT = 16
REG_A6XX_VPC_VS_SIV_CNTL = 0x00009104
A6XX_VPC_VS_SIV_CNTL_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_VS_SIV_CNTL_LAYERLOC__SHIFT = 0
A6XX_VPC_VS_SIV_CNTL_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_VS_SIV_CNTL_VIEWLOC__SHIFT = 8
A6XX_VPC_VS_SIV_CNTL_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_VS_SIV_CNTL_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_GS_SIV_CNTL = 0x00009105
A6XX_VPC_GS_SIV_CNTL_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_GS_SIV_CNTL_LAYERLOC__SHIFT = 0
A6XX_VPC_GS_SIV_CNTL_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_GS_SIV_CNTL_VIEWLOC__SHIFT = 8
A6XX_VPC_GS_SIV_CNTL_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_GS_SIV_CNTL_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_DS_SIV_CNTL = 0x00009106
A6XX_VPC_DS_SIV_CNTL_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_DS_SIV_CNTL_LAYERLOC__SHIFT = 0
A6XX_VPC_DS_SIV_CNTL_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_DS_SIV_CNTL_VIEWLOC__SHIFT = 8
A6XX_VPC_DS_SIV_CNTL_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_DS_SIV_CNTL_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_VS_SIV_CNTL_V2 = 0x00009314
A6XX_VPC_VS_SIV_CNTL_V2_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_VS_SIV_CNTL_V2_LAYERLOC__SHIFT = 0
A6XX_VPC_VS_SIV_CNTL_V2_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_VS_SIV_CNTL_V2_VIEWLOC__SHIFT = 8
A6XX_VPC_VS_SIV_CNTL_V2_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_VS_SIV_CNTL_V2_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_GS_SIV_CNTL_V2 = 0x00009315
A6XX_VPC_GS_SIV_CNTL_V2_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_GS_SIV_CNTL_V2_LAYERLOC__SHIFT = 0
A6XX_VPC_GS_SIV_CNTL_V2_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_GS_SIV_CNTL_V2_VIEWLOC__SHIFT = 8
A6XX_VPC_GS_SIV_CNTL_V2_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_GS_SIV_CNTL_V2_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_DS_SIV_CNTL_V2 = 0x00009316
A6XX_VPC_DS_SIV_CNTL_V2_LAYERLOC__MASK = 0x000000ff
A6XX_VPC_DS_SIV_CNTL_V2_LAYERLOC__SHIFT = 0
A6XX_VPC_DS_SIV_CNTL_V2_VIEWLOC__MASK = 0x0000ff00
A6XX_VPC_DS_SIV_CNTL_V2_VIEWLOC__SHIFT = 8
A6XX_VPC_DS_SIV_CNTL_V2_SHADINGRATELOC__MASK = 0x00ff0000
A6XX_VPC_DS_SIV_CNTL_V2_SHADINGRATELOC__SHIFT = 16
REG_A6XX_VPC_UNKNOWN_9107 = 0x00009107
A6XX_VPC_UNKNOWN_9107_RASTER_DISCARD = 0x00000001
A6XX_VPC_UNKNOWN_9107_UNK2 = 0x00000004
REG_A6XX_VPC_RAST_CNTL = 0x00009108
A6XX_VPC_RAST_CNTL_MODE__MASK = 0x00000003
A6XX_VPC_RAST_CNTL_MODE__SHIFT = 0
REG_A7XX_VPC_PC_CNTL = 0x00009109
A7XX_VPC_PC_CNTL_PRIMITIVE_RESTART = 0x00000001
A7XX_VPC_PC_CNTL_PROVOKING_VTX_LAST = 0x00000002
A7XX_VPC_PC_CNTL_D3D_VERTEX_ORDERING = 0x00000004
A7XX_VPC_PC_CNTL_UNK3 = 0x00000008
REG_A7XX_VPC_GS_PARAM_0 = 0x0000910a
A7XX_VPC_GS_PARAM_0_GS_VERTICES_OUT__MASK = 0x000000ff
A7XX_VPC_GS_PARAM_0_GS_VERTICES_OUT__SHIFT = 0
A7XX_VPC_GS_PARAM_0_GS_INVOCATIONS__MASK = 0x00007c00
A7XX_VPC_GS_PARAM_0_GS_INVOCATIONS__SHIFT = 10
A7XX_VPC_GS_PARAM_0_LINELENGTHEN = 0x00008000
A7XX_VPC_GS_PARAM_0_GS_OUTPUT__MASK = 0x00030000
A7XX_VPC_GS_PARAM_0_GS_OUTPUT__SHIFT = 16
A7XX_VPC_GS_PARAM_0_UNK18 = 0x00040000
REG_A7XX_VPC_STEREO_RENDERING_VIEWMASK = 0x0000910b
REG_A7XX_VPC_STEREO_RENDERING_CNTL = 0x0000910c
A7XX_VPC_STEREO_RENDERING_CNTL_ENABLE = 0x00000001
A7XX_VPC_STEREO_RENDERING_CNTL_DISABLEMULTIPOS = 0x00000002
A7XX_VPC_STEREO_RENDERING_CNTL_VIEWS__MASK = 0x0000007c
A7XX_VPC_STEREO_RENDERING_CNTL_VIEWS__SHIFT = 2
REG_A6XX_VPC_VARYING_INTERP_MODE = lambda i0: (0x00009200 + 0x1*i0 )
REG_A6XX_VPC_VARYING_REPLACE_MODE_0 = lambda i0: (0x00009208 + 0x1*i0 )
REG_A6XX_VPC_UNKNOWN_9210 = 0x00009210
REG_A6XX_VPC_UNKNOWN_9211 = 0x00009211
REG_A6XX_VPC_VARYING_LM_TRANSFER_CNTL_0 = lambda i0: (0x00009212 + 0x1*i0 )
REG_A6XX_VPC_SO_MAPPING_WPTR = 0x00009216
A6XX_VPC_SO_MAPPING_WPTR_ADDR__MASK = 0x000000ff
A6XX_VPC_SO_MAPPING_WPTR_ADDR__SHIFT = 0
A6XX_VPC_SO_MAPPING_WPTR_RESET = 0x00010000
REG_A6XX_VPC_SO_MAPPING_PORT = 0x00009217
A6XX_VPC_SO_MAPPING_PORT_A_BUF__MASK = 0x00000003
A6XX_VPC_SO_MAPPING_PORT_A_BUF__SHIFT = 0
A6XX_VPC_SO_MAPPING_PORT_A_OFF__MASK = 0x000007fc
A6XX_VPC_SO_MAPPING_PORT_A_OFF__SHIFT = 2
A6XX_VPC_SO_MAPPING_PORT_A_EN = 0x00000800
A6XX_VPC_SO_MAPPING_PORT_B_BUF__MASK = 0x00003000
A6XX_VPC_SO_MAPPING_PORT_B_BUF__SHIFT = 12
A6XX_VPC_SO_MAPPING_PORT_B_OFF__MASK = 0x007fc000
A6XX_VPC_SO_MAPPING_PORT_B_OFF__SHIFT = 14
A6XX_VPC_SO_MAPPING_PORT_B_EN = 0x00800000
REG_A6XX_VPC_SO_QUERY_BASE = 0x00009218
REG_A6XX_VPC_SO = lambda i0: (0x0000921a + 0x7*i0 )
REG_A6XX_VPC_REPLACE_MODE_CNTL = 0x00009236
A6XX_VPC_REPLACE_MODE_CNTL_INVERT = 0x00000001
REG_A6XX_VPC_UNKNOWN_9300 = 0x00009300
REG_A6XX_VPC_VS_CNTL = 0x00009301
A6XX_VPC_VS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_VPC_VS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_VPC_VS_CNTL_POSITIONLOC__MASK = 0x0000ff00
A6XX_VPC_VS_CNTL_POSITIONLOC__SHIFT = 8
A6XX_VPC_VS_CNTL_PSIZELOC__MASK = 0x00ff0000
A6XX_VPC_VS_CNTL_PSIZELOC__SHIFT = 16
A6XX_VPC_VS_CNTL_EXTRAPOS__MASK = 0x0f000000
A6XX_VPC_VS_CNTL_EXTRAPOS__SHIFT = 24
REG_A6XX_VPC_GS_CNTL = 0x00009302
A6XX_VPC_GS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_VPC_GS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_VPC_GS_CNTL_POSITIONLOC__MASK = 0x0000ff00
A6XX_VPC_GS_CNTL_POSITIONLOC__SHIFT = 8
A6XX_VPC_GS_CNTL_PSIZELOC__MASK = 0x00ff0000
A6XX_VPC_GS_CNTL_PSIZELOC__SHIFT = 16
A6XX_VPC_GS_CNTL_EXTRAPOS__MASK = 0x0f000000
A6XX_VPC_GS_CNTL_EXTRAPOS__SHIFT = 24
REG_A6XX_VPC_DS_CNTL = 0x00009303
A6XX_VPC_DS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_VPC_DS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_VPC_DS_CNTL_POSITIONLOC__MASK = 0x0000ff00
A6XX_VPC_DS_CNTL_POSITIONLOC__SHIFT = 8
A6XX_VPC_DS_CNTL_PSIZELOC__MASK = 0x00ff0000
A6XX_VPC_DS_CNTL_PSIZELOC__SHIFT = 16
A6XX_VPC_DS_CNTL_EXTRAPOS__MASK = 0x0f000000
A6XX_VPC_DS_CNTL_EXTRAPOS__SHIFT = 24
REG_A6XX_VPC_PS_CNTL = 0x00009304
A6XX_VPC_PS_CNTL_NUMNONPOSVAR__MASK = 0x000000ff
A6XX_VPC_PS_CNTL_NUMNONPOSVAR__SHIFT = 0
A6XX_VPC_PS_CNTL_PRIMIDLOC__MASK = 0x0000ff00
A6XX_VPC_PS_CNTL_PRIMIDLOC__SHIFT = 8
A6XX_VPC_PS_CNTL_VARYING = 0x00010000
A6XX_VPC_PS_CNTL_VIEWIDLOC__MASK = 0xff000000
A6XX_VPC_PS_CNTL_VIEWIDLOC__SHIFT = 24
REG_A6XX_VPC_SO_CNTL = 0x00009305
A6XX_VPC_SO_CNTL_BUF0_STREAM__MASK = 0x00000007
A6XX_VPC_SO_CNTL_BUF0_STREAM__SHIFT = 0
A6XX_VPC_SO_CNTL_BUF1_STREAM__MASK = 0x00000038
A6XX_VPC_SO_CNTL_BUF1_STREAM__SHIFT = 3
A6XX_VPC_SO_CNTL_BUF2_STREAM__MASK = 0x000001c0
A6XX_VPC_SO_CNTL_BUF2_STREAM__SHIFT = 6
A6XX_VPC_SO_CNTL_BUF3_STREAM__MASK = 0x00000e00
A6XX_VPC_SO_CNTL_BUF3_STREAM__SHIFT = 9
A6XX_VPC_SO_CNTL_STREAM_ENABLE__MASK = 0x00078000
A6XX_VPC_SO_CNTL_STREAM_ENABLE__SHIFT = 15
REG_A6XX_VPC_SO_OVERRIDE = 0x00009306
A6XX_VPC_SO_OVERRIDE_DISABLE = 0x00000001
REG_A6XX_VPC_PS_RAST_CNTL = 0x00009307
A6XX_VPC_PS_RAST_CNTL_MODE__MASK = 0x00000003
A6XX_VPC_PS_RAST_CNTL_MODE__SHIFT = 0
REG_A7XX_VPC_ATTR_BUF_GMEM_SIZE = 0x00009308
A7XX_VPC_ATTR_BUF_GMEM_SIZE_SIZE_GMEM__MASK = 0xffffffff
A7XX_VPC_ATTR_BUF_GMEM_SIZE_SIZE_GMEM__SHIFT = 0
REG_A7XX_VPC_ATTR_BUF_GMEM_BASE = 0x00009309
A7XX_VPC_ATTR_BUF_GMEM_BASE_BASE_GMEM__MASK = 0xffffffff
A7XX_VPC_ATTR_BUF_GMEM_BASE_BASE_GMEM__SHIFT = 0
REG_A7XX_PC_ATTR_BUF_GMEM_SIZE = 0x00009b09
A7XX_PC_ATTR_BUF_GMEM_SIZE_SIZE_GMEM__MASK = 0xffffffff
A7XX_PC_ATTR_BUF_GMEM_SIZE_SIZE_GMEM__SHIFT = 0
REG_A6XX_VPC_DBG_ECO_CNTL = 0x00009600
REG_A6XX_VPC_ADDR_MODE_CNTL = 0x00009601
REG_A6XX_VPC_UNKNOWN_9602 = 0x00009602
REG_A6XX_VPC_UNKNOWN_9603 = 0x00009603
REG_A6XX_VPC_PERFCTR_VPC_SEL = lambda i0: (0x00009604 + 0x1*i0 )
REG_A7XX_VPC_PERFCTR_VPC_SEL = lambda i0: (0x0000960b + 0x1*i0 )
REG_A6XX_PC_HS_PARAM_0 = 0x00009800
REG_A6XX_PC_HS_PARAM_1 = 0x00009801
A6XX_PC_HS_PARAM_1_SIZE__MASK = 0x000007ff
A6XX_PC_HS_PARAM_1_SIZE__SHIFT = 0
A6XX_PC_HS_PARAM_1_UNK13 = 0x00002000
REG_A6XX_PC_DS_PARAM = 0x00009802
A6XX_PC_DS_PARAM_SPACING__MASK = 0x00000003
A6XX_PC_DS_PARAM_SPACING__SHIFT = 0
A6XX_PC_DS_PARAM_OUTPUT__MASK = 0x0000000c
A6XX_PC_DS_PARAM_OUTPUT__SHIFT = 2
REG_A6XX_PC_RESTART_INDEX = 0x00009803
REG_A6XX_PC_MODE_CNTL = 0x00009804
REG_A6XX_PC_POWER_CNTL = 0x00009805
REG_A6XX_PC_PS_CNTL = 0x00009806
A6XX_PC_PS_CNTL_PRIMITIVEIDEN = 0x00000001
REG_A6XX_PC_DGEN_SO_CNTL = 0x00009808
A6XX_PC_DGEN_SO_CNTL_STREAM_ENABLE__MASK = 0x00078000
A6XX_PC_DGEN_SO_CNTL_STREAM_ENABLE__SHIFT = 15
REG_A6XX_PC_DGEN_SU_CONSERVATIVE_RAS_CNTL = 0x0000980a
A6XX_PC_DGEN_SU_CONSERVATIVE_RAS_CNTL_CONSERVATIVERASEN = 0x00000001
REG_A6XX_PC_DRAW_INITIATOR = 0x00009840
A6XX_PC_DRAW_INITIATOR_STATE_ID__MASK = 0x000000ff
A6XX_PC_DRAW_INITIATOR_STATE_ID__SHIFT = 0
REG_A6XX_PC_KERNEL_INITIATOR = 0x00009841
A6XX_PC_KERNEL_INITIATOR_STATE_ID__MASK = 0x000000ff
A6XX_PC_KERNEL_INITIATOR_STATE_ID__SHIFT = 0
REG_A6XX_PC_EVENT_INITIATOR = 0x00009842
A6XX_PC_EVENT_INITIATOR_STATE_ID__MASK = 0x00ff0000
A6XX_PC_EVENT_INITIATOR_STATE_ID__SHIFT = 16
A6XX_PC_EVENT_INITIATOR_EVENT__MASK = 0x0000007f
A6XX_PC_EVENT_INITIATOR_EVENT__SHIFT = 0
REG_A6XX_PC_MARKER = 0x00009880
REG_A6XX_PC_DGEN_RAST_CNTL = 0x00009981
A6XX_PC_DGEN_RAST_CNTL_MODE__MASK = 0x00000003
A6XX_PC_DGEN_RAST_CNTL_MODE__SHIFT = 0
REG_A7XX_PC_DGEN_RAST_CNTL = 0x00009809
A7XX_PC_DGEN_RAST_CNTL_MODE__MASK = 0x00000003
A7XX_PC_DGEN_RAST_CNTL_MODE__SHIFT = 0
REG_A6XX_VPC_RAST_STREAM_CNTL = 0x00009980
A6XX_VPC_RAST_STREAM_CNTL_STREAM__MASK = 0x00000003
A6XX_VPC_RAST_STREAM_CNTL_STREAM__SHIFT = 0
A6XX_VPC_RAST_STREAM_CNTL_DISCARD = 0x00000004
REG_A7XX_VPC_RAST_STREAM_CNTL = 0x00009107
A7XX_VPC_RAST_STREAM_CNTL_STREAM__MASK = 0x00000003
A7XX_VPC_RAST_STREAM_CNTL_STREAM__SHIFT = 0
A7XX_VPC_RAST_STREAM_CNTL_DISCARD = 0x00000004
REG_A7XX_VPC_RAST_STREAM_CNTL_V2 = 0x00009317
A7XX_VPC_RAST_STREAM_CNTL_V2_STREAM__MASK = 0x00000003
A7XX_VPC_RAST_STREAM_CNTL_V2_STREAM__SHIFT = 0
A7XX_VPC_RAST_STREAM_CNTL_V2_DISCARD = 0x00000004
REG_A7XX_PC_HS_BUFFER_SIZE = 0x00009885
REG_A7XX_PC_TF_BUFFER_SIZE = 0x00009886
REG_A6XX_PC_CNTL = 0x00009b00
A6XX_PC_CNTL_PRIMITIVE_RESTART = 0x00000001
A6XX_PC_CNTL_PROVOKING_VTX_LAST = 0x00000002
A6XX_PC_CNTL_D3D_VERTEX_ORDERING = 0x00000004
A6XX_PC_CNTL_UNK3 = 0x00000008
REG_A6XX_PC_VS_CNTL = 0x00009b01
A6XX_PC_VS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_PC_VS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_PC_VS_CNTL_PSIZE = 0x00000100
A6XX_PC_VS_CNTL_LAYER = 0x00000200
A6XX_PC_VS_CNTL_VIEW = 0x00000400
A6XX_PC_VS_CNTL_PRIMITIVE_ID = 0x00000800
A6XX_PC_VS_CNTL_CLIP_MASK__MASK = 0x00ff0000
A6XX_PC_VS_CNTL_CLIP_MASK__SHIFT = 16
A6XX_PC_VS_CNTL_SHADINGRATE = 0x01000000
REG_A6XX_PC_GS_CNTL = 0x00009b02
A6XX_PC_GS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_PC_GS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_PC_GS_CNTL_PSIZE = 0x00000100
A6XX_PC_GS_CNTL_LAYER = 0x00000200
A6XX_PC_GS_CNTL_VIEW = 0x00000400
A6XX_PC_GS_CNTL_PRIMITIVE_ID = 0x00000800
A6XX_PC_GS_CNTL_CLIP_MASK__MASK = 0x00ff0000
A6XX_PC_GS_CNTL_CLIP_MASK__SHIFT = 16
A6XX_PC_GS_CNTL_SHADINGRATE = 0x01000000
REG_A6XX_PC_HS_CNTL = 0x00009b03
A6XX_PC_HS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_PC_HS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_PC_HS_CNTL_PSIZE = 0x00000100
A6XX_PC_HS_CNTL_LAYER = 0x00000200
A6XX_PC_HS_CNTL_VIEW = 0x00000400
A6XX_PC_HS_CNTL_PRIMITIVE_ID = 0x00000800
A6XX_PC_HS_CNTL_CLIP_MASK__MASK = 0x00ff0000
A6XX_PC_HS_CNTL_CLIP_MASK__SHIFT = 16
A6XX_PC_HS_CNTL_SHADINGRATE = 0x01000000
REG_A6XX_PC_DS_CNTL = 0x00009b04
A6XX_PC_DS_CNTL_STRIDE_IN_VPC__MASK = 0x000000ff
A6XX_PC_DS_CNTL_STRIDE_IN_VPC__SHIFT = 0
A6XX_PC_DS_CNTL_PSIZE = 0x00000100
A6XX_PC_DS_CNTL_LAYER = 0x00000200
A6XX_PC_DS_CNTL_VIEW = 0x00000400
A6XX_PC_DS_CNTL_PRIMITIVE_ID = 0x00000800
A6XX_PC_DS_CNTL_CLIP_MASK__MASK = 0x00ff0000
A6XX_PC_DS_CNTL_CLIP_MASK__SHIFT = 16
A6XX_PC_DS_CNTL_SHADINGRATE = 0x01000000
REG_A6XX_PC_GS_PARAM_0 = 0x00009b05
A6XX_PC_GS_PARAM_0_GS_VERTICES_OUT__MASK = 0x000000ff
A6XX_PC_GS_PARAM_0_GS_VERTICES_OUT__SHIFT = 0
A6XX_PC_GS_PARAM_0_GS_INVOCATIONS__MASK = 0x00007c00
A6XX_PC_GS_PARAM_0_GS_INVOCATIONS__SHIFT = 10
A6XX_PC_GS_PARAM_0_LINELENGTHEN = 0x00008000
A6XX_PC_GS_PARAM_0_GS_OUTPUT__MASK = 0x00030000
A6XX_PC_GS_PARAM_0_GS_OUTPUT__SHIFT = 16
A6XX_PC_GS_PARAM_0_UNK18 = 0x00040000
REG_A6XX_PC_PRIMITIVE_CNTL_6 = 0x00009b06
A6XX_PC_PRIMITIVE_CNTL_6_STRIDE_IN_VPC__MASK = 0x000007ff
A6XX_PC_PRIMITIVE_CNTL_6_STRIDE_IN_VPC__SHIFT = 0
REG_A6XX_PC_STEREO_RENDERING_CNTL = 0x00009b07
A6XX_PC_STEREO_RENDERING_CNTL_ENABLE = 0x00000001
A6XX_PC_STEREO_RENDERING_CNTL_DISABLEMULTIPOS = 0x00000002
A6XX_PC_STEREO_RENDERING_CNTL_VIEWS__MASK = 0x0000007c
A6XX_PC_STEREO_RENDERING_CNTL_VIEWS__SHIFT = 2
REG_A6XX_PC_STEREO_RENDERING_VIEWMASK = 0x00009b08
REG_A6XX_PC_2D_EVENT_CMD = 0x00009c00
A6XX_PC_2D_EVENT_CMD_EVENT__MASK = 0x0000007f
A6XX_PC_2D_EVENT_CMD_EVENT__SHIFT = 0
A6XX_PC_2D_EVENT_CMD_STATE_ID__MASK = 0x0000ff00
A6XX_PC_2D_EVENT_CMD_STATE_ID__SHIFT = 8
REG_A6XX_PC_DBG_ECO_CNTL = 0x00009e00
REG_A6XX_PC_ADDR_MODE_CNTL = 0x00009e01
REG_A6XX_PC_DMA_BASE = 0x00009e04
REG_A6XX_PC_DMA_OFFSET = 0x00009e06
REG_A6XX_PC_DMA_SIZE = 0x00009e07
REG_A6XX_PC_TESS_BASE = 0x00009e08
REG_A7XX_PC_TESS_BASE = 0x00009810
REG_A6XX_PC_DRAWCALL_CNTL = 0x00009e0b
A6XX_PC_DRAWCALL_CNTL_PRIM_TYPE__MASK = 0x0000003f
A6XX_PC_DRAWCALL_CNTL_PRIM_TYPE__SHIFT = 0
A6XX_PC_DRAWCALL_CNTL_SOURCE_SELECT__MASK = 0x000000c0
A6XX_PC_DRAWCALL_CNTL_SOURCE_SELECT__SHIFT = 6
A6XX_PC_DRAWCALL_CNTL_VIS_CULL__MASK = 0x00000300
A6XX_PC_DRAWCALL_CNTL_VIS_CULL__SHIFT = 8
A6XX_PC_DRAWCALL_CNTL_INDEX_SIZE__MASK = 0x00000c00
A6XX_PC_DRAWCALL_CNTL_INDEX_SIZE__SHIFT = 10
A6XX_PC_DRAWCALL_CNTL_PATCH_TYPE__MASK = 0x00003000
A6XX_PC_DRAWCALL_CNTL_PATCH_TYPE__SHIFT = 12
A6XX_PC_DRAWCALL_CNTL_GS_ENABLE = 0x00010000
A6XX_PC_DRAWCALL_CNTL_TESS_ENABLE = 0x00020000
REG_A6XX_PC_DRAWCALL_INSTANCE_NUM = 0x00009e0c
REG_A6XX_PC_DRAWCALL_SIZE = 0x00009e0d
REG_A6XX_PC_VIS_STREAM_CNTL = 0x00009e11
A6XX_PC_VIS_STREAM_CNTL_UNK0__MASK = 0x0000ffff
A6XX_PC_VIS_STREAM_CNTL_UNK0__SHIFT = 0
A6XX_PC_VIS_STREAM_CNTL_VSC_SIZE__MASK = 0x003f0000
A6XX_PC_VIS_STREAM_CNTL_VSC_SIZE__SHIFT = 16
A6XX_PC_VIS_STREAM_CNTL_VSC_N__MASK = 0x07c00000
A6XX_PC_VIS_STREAM_CNTL_VSC_N__SHIFT = 22
REG_A6XX_PC_PVIS_STREAM_BIN_BASE = 0x00009e12
REG_A6XX_PC_DVIS_STREAM_BIN_BASE = 0x00009e14
REG_A6XX_PC_DRAWCALL_CNTL_OVERRIDE = 0x00009e1c
A6XX_PC_DRAWCALL_CNTL_OVERRIDE_OVERRIDE = 0x00000001
REG_A7XX_PC_UNKNOWN_9E24 = 0x00009e24
REG_A6XX_PC_PERFCTR_PC_SEL = lambda i0: (0x00009e34 + 0x1*i0 )
REG_A7XX_PC_PERFCTR_PC_SEL = lambda i0: (0x00009e42 + 0x1*i0 )
REG_A6XX_PC_UNKNOWN_9E72 = 0x00009e72
REG_A6XX_VFD_CNTL_0 = 0x0000a000
A6XX_VFD_CNTL_0_FETCH_CNT__MASK = 0x0000003f
A6XX_VFD_CNTL_0_FETCH_CNT__SHIFT = 0
A6XX_VFD_CNTL_0_DECODE_CNT__MASK = 0x00003f00
A6XX_VFD_CNTL_0_DECODE_CNT__SHIFT = 8
REG_A6XX_VFD_CNTL_1 = 0x0000a001
A6XX_VFD_CNTL_1_REGID4VTX__MASK = 0x000000ff
A6XX_VFD_CNTL_1_REGID4VTX__SHIFT = 0
A6XX_VFD_CNTL_1_REGID4INST__MASK = 0x0000ff00
A6XX_VFD_CNTL_1_REGID4INST__SHIFT = 8
A6XX_VFD_CNTL_1_REGID4PRIMID__MASK = 0x00ff0000
A6XX_VFD_CNTL_1_REGID4PRIMID__SHIFT = 16
A6XX_VFD_CNTL_1_REGID4VIEWID__MASK = 0xff000000
A6XX_VFD_CNTL_1_REGID4VIEWID__SHIFT = 24
REG_A6XX_VFD_CNTL_2 = 0x0000a002
A6XX_VFD_CNTL_2_REGID_HSRELPATCHID__MASK = 0x000000ff
A6XX_VFD_CNTL_2_REGID_HSRELPATCHID__SHIFT = 0
A6XX_VFD_CNTL_2_REGID_INVOCATIONID__MASK = 0x0000ff00
A6XX_VFD_CNTL_2_REGID_INVOCATIONID__SHIFT = 8
REG_A6XX_VFD_CNTL_3 = 0x0000a003
A6XX_VFD_CNTL_3_REGID_DSPRIMID__MASK = 0x000000ff
A6XX_VFD_CNTL_3_REGID_DSPRIMID__SHIFT = 0
A6XX_VFD_CNTL_3_REGID_DSRELPATCHID__MASK = 0x0000ff00
A6XX_VFD_CNTL_3_REGID_DSRELPATCHID__SHIFT = 8
A6XX_VFD_CNTL_3_REGID_TESSX__MASK = 0x00ff0000
A6XX_VFD_CNTL_3_REGID_TESSX__SHIFT = 16
A6XX_VFD_CNTL_3_REGID_TESSY__MASK = 0xff000000
A6XX_VFD_CNTL_3_REGID_TESSY__SHIFT = 24
REG_A6XX_VFD_CNTL_4 = 0x0000a004
A6XX_VFD_CNTL_4_UNK0__MASK = 0x000000ff
A6XX_VFD_CNTL_4_UNK0__SHIFT = 0
REG_A6XX_VFD_CNTL_5 = 0x0000a005
A6XX_VFD_CNTL_5_REGID_GSHEADER__MASK = 0x000000ff
A6XX_VFD_CNTL_5_REGID_GSHEADER__SHIFT = 0
A6XX_VFD_CNTL_5_UNK8__MASK = 0x0000ff00
A6XX_VFD_CNTL_5_UNK8__SHIFT = 8
REG_A6XX_VFD_CNTL_6 = 0x0000a006
A6XX_VFD_CNTL_6_PRIMID4PSEN = 0x00000001
REG_A6XX_VFD_RENDER_MODE = 0x0000a007
A6XX_VFD_RENDER_MODE_RENDER_MODE__MASK = 0x00000007
A6XX_VFD_RENDER_MODE_RENDER_MODE__SHIFT = 0
REG_A6XX_VFD_STEREO_RENDERING_CNTL = 0x0000a008
A6XX_VFD_STEREO_RENDERING_CNTL_ENABLE = 0x00000001
A6XX_VFD_STEREO_RENDERING_CNTL_DISABLEMULTIPOS = 0x00000002
A6XX_VFD_STEREO_RENDERING_CNTL_VIEWS__MASK = 0x0000007c
A6XX_VFD_STEREO_RENDERING_CNTL_VIEWS__SHIFT = 2
REG_A6XX_VFD_MODE_CNTL = 0x0000a009
A6XX_VFD_MODE_CNTL_VERTEX = 0x00000001
A6XX_VFD_MODE_CNTL_INSTANCE = 0x00000002
REG_A6XX_VFD_INDEX_OFFSET = 0x0000a00e
REG_A6XX_VFD_INSTANCE_START_OFFSET = 0x0000a00f
REG_A6XX_VFD_VERTEX_BUFFER = lambda i0: (0x0000a010 + 0x4*i0 )
REG_A6XX_VFD_FETCH_INSTR = lambda i0: (0x0000a090 + 0x2*i0 )
A6XX_VFD_FETCH_INSTR_INSTR_IDX__MASK = 0x0000001f
A6XX_VFD_FETCH_INSTR_INSTR_IDX__SHIFT = 0
A6XX_VFD_FETCH_INSTR_INSTR_OFFSET__MASK = 0x0001ffe0
A6XX_VFD_FETCH_INSTR_INSTR_OFFSET__SHIFT = 5
A6XX_VFD_FETCH_INSTR_INSTR_INSTANCED = 0x00020000
A6XX_VFD_FETCH_INSTR_INSTR_FORMAT__MASK = 0x0ff00000
A6XX_VFD_FETCH_INSTR_INSTR_FORMAT__SHIFT = 20
A6XX_VFD_FETCH_INSTR_INSTR_SWAP__MASK = 0x30000000
A6XX_VFD_FETCH_INSTR_INSTR_SWAP__SHIFT = 28
A6XX_VFD_FETCH_INSTR_INSTR_UNK30 = 0x40000000
A6XX_VFD_FETCH_INSTR_INSTR_FLOAT = 0x80000000
REG_A6XX_VFD_DEST_CNTL = lambda i0: (0x0000a0d0 + 0x1*i0 )
A6XX_VFD_DEST_CNTL_INSTR_WRITEMASK__MASK = 0x0000000f
A6XX_VFD_DEST_CNTL_INSTR_WRITEMASK__SHIFT = 0
A6XX_VFD_DEST_CNTL_INSTR_REGID__MASK = 0x00000ff0
A6XX_VFD_DEST_CNTL_INSTR_REGID__SHIFT = 4
REG_A6XX_VFD_POWER_CNTL = 0x0000a0f8
REG_A7XX_VFD_DBG_ECO_CNTL = 0x0000a600
REG_A6XX_VFD_ADDR_MODE_CNTL = 0x0000a601
REG_A6XX_VFD_PERFCTR_VFD_SEL = lambda i0: (0x0000a610 + 0x1*i0 )
REG_A7XX_VFD_PERFCTR_VFD_SEL = lambda i0: (0x0000a610 + 0x1*i0 )
REG_A6XX_SP_VS_CNTL_0 = 0x0000a800
A6XX_SP_VS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_VS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_VS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_VS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_VS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_VS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_VS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_VS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_VS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_VS_CNTL_0_MERGEDREGS = 0x00100000
A6XX_SP_VS_CNTL_0_EARLYPREAMBLE = 0x00200000
REG_A6XX_SP_VS_BOOLEAN_CF_MASK = 0x0000a801
REG_A6XX_SP_VS_OUTPUT_CNTL = 0x0000a802
A6XX_SP_VS_OUTPUT_CNTL_OUT__MASK = 0x0000003f
A6XX_SP_VS_OUTPUT_CNTL_OUT__SHIFT = 0
A6XX_SP_VS_OUTPUT_CNTL_FLAGS_REGID__MASK = 0x00003fc0
A6XX_SP_VS_OUTPUT_CNTL_FLAGS_REGID__SHIFT = 6
REG_A6XX_SP_VS_OUTPUT = lambda i0: (0x0000a803 + 0x1*i0 )
A6XX_SP_VS_OUTPUT_REG_A_REGID__MASK = 0x000000ff
A6XX_SP_VS_OUTPUT_REG_A_REGID__SHIFT = 0
A6XX_SP_VS_OUTPUT_REG_A_COMPMASK__MASK = 0x00000f00
A6XX_SP_VS_OUTPUT_REG_A_COMPMASK__SHIFT = 8
A6XX_SP_VS_OUTPUT_REG_B_REGID__MASK = 0x00ff0000
A6XX_SP_VS_OUTPUT_REG_B_REGID__SHIFT = 16
A6XX_SP_VS_OUTPUT_REG_B_COMPMASK__MASK = 0x0f000000
A6XX_SP_VS_OUTPUT_REG_B_COMPMASK__SHIFT = 24
REG_A6XX_SP_VS_VPC_DEST = lambda i0: (0x0000a813 + 0x1*i0 )
A6XX_SP_VS_VPC_DEST_REG_OUTLOC0__MASK = 0x000000ff
A6XX_SP_VS_VPC_DEST_REG_OUTLOC0__SHIFT = 0
A6XX_SP_VS_VPC_DEST_REG_OUTLOC1__MASK = 0x0000ff00
A6XX_SP_VS_VPC_DEST_REG_OUTLOC1__SHIFT = 8
A6XX_SP_VS_VPC_DEST_REG_OUTLOC2__MASK = 0x00ff0000
A6XX_SP_VS_VPC_DEST_REG_OUTLOC2__SHIFT = 16
A6XX_SP_VS_VPC_DEST_REG_OUTLOC3__MASK = 0xff000000
A6XX_SP_VS_VPC_DEST_REG_OUTLOC3__SHIFT = 24
REG_A6XX_SP_VS_PROGRAM_COUNTER_OFFSET = 0x0000a81b
REG_A6XX_SP_VS_BASE = 0x0000a81c
REG_A6XX_SP_VS_PVT_MEM_PARAM = 0x0000a81e
A6XX_SP_VS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_VS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_VS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_VS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_VS_PVT_MEM_BASE = 0x0000a81f
REG_A6XX_SP_VS_PVT_MEM_SIZE = 0x0000a821
A6XX_SP_VS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_VS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_VS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_VS_TSIZE = 0x0000a822
REG_A6XX_SP_VS_CONFIG = 0x0000a823
A6XX_SP_VS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_VS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_VS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_VS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_VS_CONFIG_ENABLED = 0x00000100
A6XX_SP_VS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_VS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_VS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_VS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_VS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_VS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_VS_INSTR_SIZE = 0x0000a824
REG_A6XX_SP_VS_PVT_MEM_STACK_OFFSET = 0x0000a825
A6XX_SP_VS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_VS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_VS_VGS_CNTL = 0x0000a82d
REG_A6XX_SP_HS_CNTL_0 = 0x0000a830
A6XX_SP_HS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_HS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_HS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_HS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_HS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_HS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_HS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_HS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_HS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_HS_CNTL_0_EARLYPREAMBLE = 0x00100000
REG_A6XX_SP_HS_CNTL_1 = 0x0000a831
REG_A6XX_SP_HS_BOOLEAN_CF_MASK = 0x0000a832
REG_A6XX_SP_HS_PROGRAM_COUNTER_OFFSET = 0x0000a833
REG_A6XX_SP_HS_BASE = 0x0000a834
REG_A6XX_SP_HS_PVT_MEM_PARAM = 0x0000a836
A6XX_SP_HS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_HS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_HS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_HS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_HS_PVT_MEM_BASE = 0x0000a837
REG_A6XX_SP_HS_PVT_MEM_SIZE = 0x0000a839
A6XX_SP_HS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_HS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_HS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_HS_TSIZE = 0x0000a83a
REG_A6XX_SP_HS_CONFIG = 0x0000a83b
A6XX_SP_HS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_HS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_HS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_HS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_HS_CONFIG_ENABLED = 0x00000100
A6XX_SP_HS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_HS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_HS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_HS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_HS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_HS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_HS_INSTR_SIZE = 0x0000a83c
REG_A6XX_SP_HS_PVT_MEM_STACK_OFFSET = 0x0000a83d
A6XX_SP_HS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_HS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_HS_VGS_CNTL = 0x0000a82f
REG_A6XX_SP_DS_CNTL_0 = 0x0000a840
A6XX_SP_DS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_DS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_DS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_DS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_DS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_DS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_DS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_DS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_DS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_DS_CNTL_0_EARLYPREAMBLE = 0x00100000
REG_A6XX_SP_DS_BOOLEAN_CF_MASK = 0x0000a841
REG_A6XX_SP_DS_OUTPUT_CNTL = 0x0000a842
A6XX_SP_DS_OUTPUT_CNTL_OUT__MASK = 0x0000003f
A6XX_SP_DS_OUTPUT_CNTL_OUT__SHIFT = 0
A6XX_SP_DS_OUTPUT_CNTL_FLAGS_REGID__MASK = 0x00003fc0
A6XX_SP_DS_OUTPUT_CNTL_FLAGS_REGID__SHIFT = 6
REG_A6XX_SP_DS_OUTPUT = lambda i0: (0x0000a843 + 0x1*i0 )
A6XX_SP_DS_OUTPUT_REG_A_REGID__MASK = 0x000000ff
A6XX_SP_DS_OUTPUT_REG_A_REGID__SHIFT = 0
A6XX_SP_DS_OUTPUT_REG_A_COMPMASK__MASK = 0x00000f00
A6XX_SP_DS_OUTPUT_REG_A_COMPMASK__SHIFT = 8
A6XX_SP_DS_OUTPUT_REG_B_REGID__MASK = 0x00ff0000
A6XX_SP_DS_OUTPUT_REG_B_REGID__SHIFT = 16
A6XX_SP_DS_OUTPUT_REG_B_COMPMASK__MASK = 0x0f000000
A6XX_SP_DS_OUTPUT_REG_B_COMPMASK__SHIFT = 24
REG_A6XX_SP_DS_VPC_DEST = lambda i0: (0x0000a853 + 0x1*i0 )
A6XX_SP_DS_VPC_DEST_REG_OUTLOC0__MASK = 0x000000ff
A6XX_SP_DS_VPC_DEST_REG_OUTLOC0__SHIFT = 0
A6XX_SP_DS_VPC_DEST_REG_OUTLOC1__MASK = 0x0000ff00
A6XX_SP_DS_VPC_DEST_REG_OUTLOC1__SHIFT = 8
A6XX_SP_DS_VPC_DEST_REG_OUTLOC2__MASK = 0x00ff0000
A6XX_SP_DS_VPC_DEST_REG_OUTLOC2__SHIFT = 16
A6XX_SP_DS_VPC_DEST_REG_OUTLOC3__MASK = 0xff000000
A6XX_SP_DS_VPC_DEST_REG_OUTLOC3__SHIFT = 24
REG_A6XX_SP_DS_PROGRAM_COUNTER_OFFSET = 0x0000a85b
REG_A6XX_SP_DS_BASE = 0x0000a85c
REG_A6XX_SP_DS_PVT_MEM_PARAM = 0x0000a85e
A6XX_SP_DS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_DS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_DS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_DS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_DS_PVT_MEM_BASE = 0x0000a85f
REG_A6XX_SP_DS_PVT_MEM_SIZE = 0x0000a861
A6XX_SP_DS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_DS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_DS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_DS_TSIZE = 0x0000a862
REG_A6XX_SP_DS_CONFIG = 0x0000a863
A6XX_SP_DS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_DS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_DS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_DS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_DS_CONFIG_ENABLED = 0x00000100
A6XX_SP_DS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_DS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_DS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_DS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_DS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_DS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_DS_INSTR_SIZE = 0x0000a864
REG_A6XX_SP_DS_PVT_MEM_STACK_OFFSET = 0x0000a865
A6XX_SP_DS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_DS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_DS_VGS_CNTL = 0x0000a868
REG_A6XX_SP_GS_CNTL_0 = 0x0000a870
A6XX_SP_GS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_GS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_GS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_GS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_GS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_GS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_GS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_GS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_GS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_GS_CNTL_0_EARLYPREAMBLE = 0x00100000
REG_A6XX_SP_GS_CNTL_1 = 0x0000a871
REG_A6XX_SP_GS_BOOLEAN_CF_MASK = 0x0000a872
REG_A6XX_SP_GS_OUTPUT_CNTL = 0x0000a873
A6XX_SP_GS_OUTPUT_CNTL_OUT__MASK = 0x0000003f
A6XX_SP_GS_OUTPUT_CNTL_OUT__SHIFT = 0
A6XX_SP_GS_OUTPUT_CNTL_FLAGS_REGID__MASK = 0x00003fc0
A6XX_SP_GS_OUTPUT_CNTL_FLAGS_REGID__SHIFT = 6
REG_A6XX_SP_GS_OUTPUT = lambda i0: (0x0000a874 + 0x1*i0 )
A6XX_SP_GS_OUTPUT_REG_A_REGID__MASK = 0x000000ff
A6XX_SP_GS_OUTPUT_REG_A_REGID__SHIFT = 0
A6XX_SP_GS_OUTPUT_REG_A_COMPMASK__MASK = 0x00000f00
A6XX_SP_GS_OUTPUT_REG_A_COMPMASK__SHIFT = 8
A6XX_SP_GS_OUTPUT_REG_B_REGID__MASK = 0x00ff0000
A6XX_SP_GS_OUTPUT_REG_B_REGID__SHIFT = 16
A6XX_SP_GS_OUTPUT_REG_B_COMPMASK__MASK = 0x0f000000
A6XX_SP_GS_OUTPUT_REG_B_COMPMASK__SHIFT = 24
REG_A6XX_SP_GS_VPC_DEST = lambda i0: (0x0000a884 + 0x1*i0 )
A6XX_SP_GS_VPC_DEST_REG_OUTLOC0__MASK = 0x000000ff
A6XX_SP_GS_VPC_DEST_REG_OUTLOC0__SHIFT = 0
A6XX_SP_GS_VPC_DEST_REG_OUTLOC1__MASK = 0x0000ff00
A6XX_SP_GS_VPC_DEST_REG_OUTLOC1__SHIFT = 8
A6XX_SP_GS_VPC_DEST_REG_OUTLOC2__MASK = 0x00ff0000
A6XX_SP_GS_VPC_DEST_REG_OUTLOC2__SHIFT = 16
A6XX_SP_GS_VPC_DEST_REG_OUTLOC3__MASK = 0xff000000
A6XX_SP_GS_VPC_DEST_REG_OUTLOC3__SHIFT = 24
REG_A6XX_SP_GS_PROGRAM_COUNTER_OFFSET = 0x0000a88c
REG_A6XX_SP_GS_BASE = 0x0000a88d
REG_A6XX_SP_GS_PVT_MEM_PARAM = 0x0000a88f
A6XX_SP_GS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_GS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_GS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_GS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_GS_PVT_MEM_BASE = 0x0000a890
REG_A6XX_SP_GS_PVT_MEM_SIZE = 0x0000a892
A6XX_SP_GS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_GS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_GS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_GS_TSIZE = 0x0000a893
REG_A6XX_SP_GS_CONFIG = 0x0000a894
A6XX_SP_GS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_GS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_GS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_GS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_GS_CONFIG_ENABLED = 0x00000100
A6XX_SP_GS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_GS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_GS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_GS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_GS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_GS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_GS_INSTR_SIZE = 0x0000a895
REG_A6XX_SP_GS_PVT_MEM_STACK_OFFSET = 0x0000a896
A6XX_SP_GS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_GS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_GS_VGS_CNTL = 0x0000a899
REG_A6XX_SP_VS_SAMPLER_BASE = 0x0000a8a0
REG_A6XX_SP_HS_SAMPLER_BASE = 0x0000a8a2
REG_A6XX_SP_DS_SAMPLER_BASE = 0x0000a8a4
REG_A6XX_SP_GS_SAMPLER_BASE = 0x0000a8a6
REG_A6XX_SP_VS_TEXMEMOBJ_BASE = 0x0000a8a8
REG_A6XX_SP_HS_TEXMEMOBJ_BASE = 0x0000a8aa
REG_A6XX_SP_DS_TEXMEMOBJ_BASE = 0x0000a8ac
REG_A6XX_SP_GS_TEXMEMOBJ_BASE = 0x0000a8ae
REG_A6XX_SP_PS_CNTL_0 = 0x0000a980
A6XX_SP_PS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_PS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_PS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_PS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_PS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_PS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_PS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_PS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_PS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_PS_CNTL_0_THREADSIZE__MASK = 0x00100000
A6XX_SP_PS_CNTL_0_THREADSIZE__SHIFT = 20
A6XX_SP_PS_CNTL_0_UNK21 = 0x00200000
A6XX_SP_PS_CNTL_0_VARYING = 0x00400000
A6XX_SP_PS_CNTL_0_LODPIXMASK = 0x00800000
A6XX_SP_PS_CNTL_0_INOUTREGOVERLAP = 0x01000000
A6XX_SP_PS_CNTL_0_UNK25 = 0x02000000
A6XX_SP_PS_CNTL_0_PIXLODENABLE = 0x04000000
A6XX_SP_PS_CNTL_0_UNK27 = 0x08000000
A6XX_SP_PS_CNTL_0_EARLYPREAMBLE = 0x10000000
A6XX_SP_PS_CNTL_0_MERGEDREGS = 0x80000000
REG_A6XX_SP_PS_BOOLEAN_CF_MASK = 0x0000a981
REG_A6XX_SP_PS_PROGRAM_COUNTER_OFFSET = 0x0000a982
REG_A6XX_SP_PS_BASE = 0x0000a983
REG_A6XX_SP_PS_PVT_MEM_PARAM = 0x0000a985
A6XX_SP_PS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_PS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_PS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_PS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_PS_PVT_MEM_BASE = 0x0000a986
REG_A6XX_SP_PS_PVT_MEM_SIZE = 0x0000a988
A6XX_SP_PS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_PS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_PS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_BLEND_CNTL = 0x0000a989
A6XX_SP_BLEND_CNTL_ENABLE_BLEND__MASK = 0x000000ff
A6XX_SP_BLEND_CNTL_ENABLE_BLEND__SHIFT = 0
A6XX_SP_BLEND_CNTL_UNK8 = 0x00000100
A6XX_SP_BLEND_CNTL_DUAL_COLOR_IN_ENABLE = 0x00000200
A6XX_SP_BLEND_CNTL_ALPHA_TO_COVERAGE = 0x00000400
REG_A6XX_SP_SRGB_CNTL = 0x0000a98a
A6XX_SP_SRGB_CNTL_SRGB_MRT0 = 0x00000001
A6XX_SP_SRGB_CNTL_SRGB_MRT1 = 0x00000002
A6XX_SP_SRGB_CNTL_SRGB_MRT2 = 0x00000004
A6XX_SP_SRGB_CNTL_SRGB_MRT3 = 0x00000008
A6XX_SP_SRGB_CNTL_SRGB_MRT4 = 0x00000010
A6XX_SP_SRGB_CNTL_SRGB_MRT5 = 0x00000020
A6XX_SP_SRGB_CNTL_SRGB_MRT6 = 0x00000040
A6XX_SP_SRGB_CNTL_SRGB_MRT7 = 0x00000080
REG_A6XX_SP_PS_OUTPUT_MASK = 0x0000a98b
A6XX_SP_PS_OUTPUT_MASK_RT0__MASK = 0x0000000f
A6XX_SP_PS_OUTPUT_MASK_RT0__SHIFT = 0
A6XX_SP_PS_OUTPUT_MASK_RT1__MASK = 0x000000f0
A6XX_SP_PS_OUTPUT_MASK_RT1__SHIFT = 4
A6XX_SP_PS_OUTPUT_MASK_RT2__MASK = 0x00000f00
A6XX_SP_PS_OUTPUT_MASK_RT2__SHIFT = 8
A6XX_SP_PS_OUTPUT_MASK_RT3__MASK = 0x0000f000
A6XX_SP_PS_OUTPUT_MASK_RT3__SHIFT = 12
A6XX_SP_PS_OUTPUT_MASK_RT4__MASK = 0x000f0000
A6XX_SP_PS_OUTPUT_MASK_RT4__SHIFT = 16
A6XX_SP_PS_OUTPUT_MASK_RT5__MASK = 0x00f00000
A6XX_SP_PS_OUTPUT_MASK_RT5__SHIFT = 20
A6XX_SP_PS_OUTPUT_MASK_RT6__MASK = 0x0f000000
A6XX_SP_PS_OUTPUT_MASK_RT6__SHIFT = 24
A6XX_SP_PS_OUTPUT_MASK_RT7__MASK = 0xf0000000
A6XX_SP_PS_OUTPUT_MASK_RT7__SHIFT = 28
REG_A6XX_SP_PS_OUTPUT_CNTL = 0x0000a98c
A6XX_SP_PS_OUTPUT_CNTL_DUAL_COLOR_IN_ENABLE = 0x00000001
A6XX_SP_PS_OUTPUT_CNTL_DEPTH_REGID__MASK = 0x0000ff00
A6XX_SP_PS_OUTPUT_CNTL_DEPTH_REGID__SHIFT = 8
A6XX_SP_PS_OUTPUT_CNTL_SAMPMASK_REGID__MASK = 0x00ff0000
A6XX_SP_PS_OUTPUT_CNTL_SAMPMASK_REGID__SHIFT = 16
A6XX_SP_PS_OUTPUT_CNTL_STENCILREF_REGID__MASK = 0xff000000
A6XX_SP_PS_OUTPUT_CNTL_STENCILREF_REGID__SHIFT = 24
REG_A6XX_SP_PS_MRT_CNTL = 0x0000a98d
A6XX_SP_PS_MRT_CNTL_MRT__MASK = 0x0000000f
A6XX_SP_PS_MRT_CNTL_MRT__SHIFT = 0
REG_A6XX_SP_PS_OUTPUT = lambda i0: (0x0000a98e + 0x1*i0 )
A6XX_SP_PS_OUTPUT_REG_REGID__MASK = 0x000000ff
A6XX_SP_PS_OUTPUT_REG_REGID__SHIFT = 0
A6XX_SP_PS_OUTPUT_REG_HALF_PRECISION = 0x00000100
REG_A6XX_SP_PS_MRT = lambda i0: (0x0000a996 + 0x1*i0 )
A6XX_SP_PS_MRT_REG_COLOR_FORMAT__MASK = 0x000000ff
A6XX_SP_PS_MRT_REG_COLOR_FORMAT__SHIFT = 0
A6XX_SP_PS_MRT_REG_COLOR_SINT = 0x00000100
A6XX_SP_PS_MRT_REG_COLOR_UINT = 0x00000200
A6XX_SP_PS_MRT_REG_UNK10 = 0x00000400
REG_A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL = 0x0000a99e
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_COUNT__MASK = 0x00000007
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_COUNT__SHIFT = 0
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_IJ_WRITE_DISABLE = 0x00000008
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_ENDOFQUAD = 0x00000010
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_WRITE_COLOR_TO_OUTPUT = 0x00000020
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_CONSTSLOTID__MASK = 0x00007fc0
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_CONSTSLOTID__SHIFT = 6
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_CONSTSLOTID4COORD__MASK = 0x01ff0000
A6XX_SP_PS_INITIAL_TEX_LOAD_CNTL_CONSTSLOTID4COORD__SHIFT = 16
REG_A6XX_SP_PS_INITIAL_TEX_LOAD = lambda i0: (0x0000a99f + 0x1*i0 )
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_SRC__MASK = 0x0000007f
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_SRC__SHIFT = 0
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_SAMP_ID__MASK = 0x00000780
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_SAMP_ID__SHIFT = 7
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_TEX_ID__MASK = 0x0000f800
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_TEX_ID__SHIFT = 11
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_DST__MASK = 0x003f0000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_DST__SHIFT = 16
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_WRMASK__MASK = 0x03c00000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_WRMASK__SHIFT = 22
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_HALF = 0x04000000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_UNK27 = 0x08000000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_BINDLESS = 0x10000000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_CMD__MASK = 0xe0000000
A6XX_SP_PS_INITIAL_TEX_LOAD_CMD_CMD__SHIFT = 29
REG_A7XX_SP_PS_INITIAL_TEX_LOAD = lambda i0: (0x0000a99f + 0x1*i0 )
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_SRC__MASK = 0x0000007f
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_SRC__SHIFT = 0
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_SAMP_ID__MASK = 0x00000380
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_SAMP_ID__SHIFT = 7
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_TEX_ID__MASK = 0x00001c00
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_TEX_ID__SHIFT = 10
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_DST__MASK = 0x0007e000
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_DST__SHIFT = 13
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_WRMASK__MASK = 0x00780000
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_WRMASK__SHIFT = 19
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_HALF = 0x00800000
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_BINDLESS = 0x02000000
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_CMD__MASK = 0x3c000000
A7XX_SP_PS_INITIAL_TEX_LOAD_CMD_CMD__SHIFT = 26
REG_A6XX_SP_PS_INITIAL_TEX_INDEX = lambda i0: (0x0000a9a3 + 0x1*i0 )
A6XX_SP_PS_INITIAL_TEX_INDEX_CMD_SAMP_ID__MASK = 0x0000ffff
A6XX_SP_PS_INITIAL_TEX_INDEX_CMD_SAMP_ID__SHIFT = 0
A6XX_SP_PS_INITIAL_TEX_INDEX_CMD_TEX_ID__MASK = 0xffff0000
A6XX_SP_PS_INITIAL_TEX_INDEX_CMD_TEX_ID__SHIFT = 16
REG_A6XX_SP_PS_TSIZE = 0x0000a9a7
REG_A6XX_SP_UNKNOWN_A9A8 = 0x0000a9a8
REG_A6XX_SP_PS_PVT_MEM_STACK_OFFSET = 0x0000a9a9
A6XX_SP_PS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_PS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_PS_UNKNOWN_A9AB = 0x0000a9ab
REG_A6XX_SP_CS_CNTL_0 = 0x0000a9b0
A6XX_SP_CS_CNTL_0_THREADMODE__MASK = 0x00000001
A6XX_SP_CS_CNTL_0_THREADMODE__SHIFT = 0
A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__MASK = 0x0000007e
A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__SHIFT = 1
A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__MASK = 0x00001f80
A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__SHIFT = 7
A6XX_SP_CS_CNTL_0_UNK13 = 0x00002000
A6XX_SP_CS_CNTL_0_BRANCHSTACK__MASK = 0x000fc000
A6XX_SP_CS_CNTL_0_BRANCHSTACK__SHIFT = 14
A6XX_SP_CS_CNTL_0_THREADSIZE__MASK = 0x00100000
A6XX_SP_CS_CNTL_0_THREADSIZE__SHIFT = 20
A6XX_SP_CS_CNTL_0_UNK21 = 0x00200000
A6XX_SP_CS_CNTL_0_UNK22 = 0x00400000
A6XX_SP_CS_CNTL_0_EARLYPREAMBLE = 0x00800000
A6XX_SP_CS_CNTL_0_MERGEDREGS = 0x80000000
REG_A6XX_SP_CS_CNTL_1 = 0x0000a9b1
A6XX_SP_CS_CNTL_1_SHARED_SIZE__MASK = 0x0000001f
A6XX_SP_CS_CNTL_1_SHARED_SIZE__SHIFT = 0
A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__MASK = 0x00000060
A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__SHIFT = 5
REG_A6XX_SP_CS_BOOLEAN_CF_MASK = 0x0000a9b2
REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET = 0x0000a9b3
REG_A6XX_SP_CS_BASE = 0x0000a9b4
REG_A6XX_SP_CS_PVT_MEM_PARAM = 0x0000a9b6
A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK = 0x000000ff
A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT = 0
A6XX_SP_CS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__MASK = 0xff000000
A6XX_SP_CS_PVT_MEM_PARAM_HWSTACKSIZEPERTHREAD__SHIFT = 24
REG_A6XX_SP_CS_PVT_MEM_BASE = 0x0000a9b7
REG_A6XX_SP_CS_PVT_MEM_SIZE = 0x0000a9b9
A6XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__MASK = 0x0003ffff
A6XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE__SHIFT = 0
A6XX_SP_CS_PVT_MEM_SIZE_PERWAVEMEMLAYOUT = 0x80000000
REG_A6XX_SP_CS_TSIZE = 0x0000a9ba
REG_A6XX_SP_CS_CONFIG = 0x0000a9bb
A6XX_SP_CS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_CS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_CS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_CS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_CS_CONFIG_ENABLED = 0x00000100
A6XX_SP_CS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_CS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_CS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_CS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_CS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_CS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_CS_INSTR_SIZE = 0x0000a9bc
REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET = 0x0000a9bd
A6XX_SP_CS_PVT_MEM_STACK_OFFSET_OFFSET__MASK = 0x0007ffff
A6XX_SP_CS_PVT_MEM_STACK_OFFSET_OFFSET__SHIFT = 0
REG_A7XX_SP_CS_UNKNOWN_A9BE = 0x0000a9be
REG_A7XX_SP_CS_VGS_CNTL = 0x0000a9c5
REG_A6XX_SP_CS_WIE_CNTL_0 = 0x0000a9c2
A6XX_SP_CS_WIE_CNTL_0_WGIDCONSTID__MASK = 0x000000ff
A6XX_SP_CS_WIE_CNTL_0_WGIDCONSTID__SHIFT = 0
A6XX_SP_CS_WIE_CNTL_0_WGSIZECONSTID__MASK = 0x0000ff00
A6XX_SP_CS_WIE_CNTL_0_WGSIZECONSTID__SHIFT = 8
A6XX_SP_CS_WIE_CNTL_0_WGOFFSETCONSTID__MASK = 0x00ff0000
A6XX_SP_CS_WIE_CNTL_0_WGOFFSETCONSTID__SHIFT = 16
A6XX_SP_CS_WIE_CNTL_0_LOCALIDREGID__MASK = 0xff000000
A6XX_SP_CS_WIE_CNTL_0_LOCALIDREGID__SHIFT = 24
REG_A6XX_SP_CS_WIE_CNTL_1 = 0x0000a9c3
A6XX_SP_CS_WIE_CNTL_1_LINEARLOCALIDREGID__MASK = 0x000000ff
A6XX_SP_CS_WIE_CNTL_1_LINEARLOCALIDREGID__SHIFT = 0
A6XX_SP_CS_WIE_CNTL_1_SINGLE_SP_CORE = 0x00000100
A6XX_SP_CS_WIE_CNTL_1_THREADSIZE__MASK = 0x00000200
A6XX_SP_CS_WIE_CNTL_1_THREADSIZE__SHIFT = 9
A6XX_SP_CS_WIE_CNTL_1_THREADSIZE_SCALAR = 0x00000400
REG_A7XX_SP_CS_WIE_CNTL_1 = 0x0000a9c3
A7XX_SP_CS_WIE_CNTL_1_LINEARLOCALIDREGID__MASK = 0x000000ff
A7XX_SP_CS_WIE_CNTL_1_LINEARLOCALIDREGID__SHIFT = 0
A7XX_SP_CS_WIE_CNTL_1_THREADSIZE__MASK = 0x00000100
A7XX_SP_CS_WIE_CNTL_1_THREADSIZE__SHIFT = 8
A7XX_SP_CS_WIE_CNTL_1_THREADSIZE_SCALAR = 0x00000200
A7XX_SP_CS_WIE_CNTL_1_WORKITEMRASTORDER__MASK = 0x00008000
A7XX_SP_CS_WIE_CNTL_1_WORKITEMRASTORDER__SHIFT = 15
REG_A6XX_SP_PS_SAMPLER_BASE = 0x0000a9e0
REG_A6XX_SP_CS_SAMPLER_BASE = 0x0000a9e2
REG_A6XX_SP_PS_TEXMEMOBJ_BASE = 0x0000a9e4
REG_A6XX_SP_CS_TEXMEMOBJ_BASE = 0x0000a9e6
REG_A6XX_SP_CS_BINDLESS_BASE = lambda i0: (0x0000a9e8 + 0x2*i0 )
A6XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A6XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A6XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A6XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A7XX_SP_CS_BINDLESS_BASE = lambda i0: (0x0000a9e8 + 0x2*i0 )
A7XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A7XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A7XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A7XX_SP_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A6XX_SP_CS_UAV_BASE = 0x0000a9f2
REG_A7XX_SP_CS_UAV_BASE = 0x0000a9f8
REG_A6XX_SP_CS_USIZE = 0x0000aa00
REG_A7XX_SP_PS_VGS_CNTL = 0x0000aa01
REG_A7XX_SP_PS_OUTPUT_CONST_CNTL = 0x0000aa02
A7XX_SP_PS_OUTPUT_CONST_CNTL_ENABLED = 0x00000001
REG_A7XX_SP_PS_OUTPUT_CONST_MASK = 0x0000aa03
A7XX_SP_PS_OUTPUT_CONST_MASK_RT0__MASK = 0x0000000f
A7XX_SP_PS_OUTPUT_CONST_MASK_RT0__SHIFT = 0
A7XX_SP_PS_OUTPUT_CONST_MASK_RT1__MASK = 0x000000f0
A7XX_SP_PS_OUTPUT_CONST_MASK_RT1__SHIFT = 4
A7XX_SP_PS_OUTPUT_CONST_MASK_RT2__MASK = 0x00000f00
A7XX_SP_PS_OUTPUT_CONST_MASK_RT2__SHIFT = 8
A7XX_SP_PS_OUTPUT_CONST_MASK_RT3__MASK = 0x0000f000
A7XX_SP_PS_OUTPUT_CONST_MASK_RT3__SHIFT = 12
A7XX_SP_PS_OUTPUT_CONST_MASK_RT4__MASK = 0x000f0000
A7XX_SP_PS_OUTPUT_CONST_MASK_RT4__SHIFT = 16
A7XX_SP_PS_OUTPUT_CONST_MASK_RT5__MASK = 0x00f00000
A7XX_SP_PS_OUTPUT_CONST_MASK_RT5__SHIFT = 20
A7XX_SP_PS_OUTPUT_CONST_MASK_RT6__MASK = 0x0f000000
A7XX_SP_PS_OUTPUT_CONST_MASK_RT6__SHIFT = 24
A7XX_SP_PS_OUTPUT_CONST_MASK_RT7__MASK = 0xf0000000
A7XX_SP_PS_OUTPUT_CONST_MASK_RT7__SHIFT = 28
REG_A6XX_SP_UNKNOWN_AAF2 = 0x0000aaf2
REG_A6XX_SP_MODE_CNTL = 0x0000ab00
A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE = 0x00000001
A6XX_SP_MODE_CNTL_ISAMMODE__MASK = 0x00000006
A6XX_SP_MODE_CNTL_ISAMMODE__SHIFT = 1
A6XX_SP_MODE_CNTL_SHARED_CONSTS_ENABLE = 0x00000008
REG_A7XX_SP_UNKNOWN_AB01 = 0x0000ab01
REG_A7XX_SP_UNKNOWN_AB02 = 0x0000ab02
REG_A6XX_SP_PS_CONFIG = 0x0000ab04
A6XX_SP_PS_CONFIG_BINDLESS_TEX = 0x00000001
A6XX_SP_PS_CONFIG_BINDLESS_SAMP = 0x00000002
A6XX_SP_PS_CONFIG_BINDLESS_UAV = 0x00000004
A6XX_SP_PS_CONFIG_BINDLESS_UBO = 0x00000008
A6XX_SP_PS_CONFIG_ENABLED = 0x00000100
A6XX_SP_PS_CONFIG_NTEX__MASK = 0x0001fe00
A6XX_SP_PS_CONFIG_NTEX__SHIFT = 9
A6XX_SP_PS_CONFIG_NSAMP__MASK = 0x003e0000
A6XX_SP_PS_CONFIG_NSAMP__SHIFT = 17
A6XX_SP_PS_CONFIG_NUAV__MASK = 0x1fc00000
A6XX_SP_PS_CONFIG_NUAV__SHIFT = 22
REG_A6XX_SP_PS_INSTR_SIZE = 0x0000ab05
REG_A6XX_SP_GFX_BINDLESS_BASE = lambda i0: (0x0000ab10 + 0x2*i0 )
A6XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A6XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A6XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A6XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A7XX_SP_GFX_BINDLESS_BASE = lambda i0: (0x0000ab0a + 0x2*i0 )
A7XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A7XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A7XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A7XX_SP_GFX_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A6XX_SP_GFX_UAV_BASE = 0x0000ab1a
REG_A6XX_SP_GFX_USIZE = 0x0000ab20
REG_A7XX_SP_UNKNOWN_AB22 = 0x0000ab22
REG_A6XX_SP_A2D_OUTPUT_INFO = 0x0000acc0
A6XX_SP_A2D_OUTPUT_INFO_HALF_PRECISION = 0x00000001
A6XX_SP_A2D_OUTPUT_INFO_IFMT_TYPE__MASK = 0x00000006
A6XX_SP_A2D_OUTPUT_INFO_IFMT_TYPE__SHIFT = 1
A6XX_SP_A2D_OUTPUT_INFO_COLOR_FORMAT__MASK = 0x000007f8
A6XX_SP_A2D_OUTPUT_INFO_COLOR_FORMAT__SHIFT = 3
A6XX_SP_A2D_OUTPUT_INFO_SRGB = 0x00000800
A6XX_SP_A2D_OUTPUT_INFO_MASK__MASK = 0x0000f000
A6XX_SP_A2D_OUTPUT_INFO_MASK__SHIFT = 12
REG_A7XX_SP_A2D_OUTPUT_INFO = 0x0000a9bf
A7XX_SP_A2D_OUTPUT_INFO_HALF_PRECISION = 0x00000001
A7XX_SP_A2D_OUTPUT_INFO_IFMT_TYPE__MASK = 0x00000006
A7XX_SP_A2D_OUTPUT_INFO_IFMT_TYPE__SHIFT = 1
A7XX_SP_A2D_OUTPUT_INFO_COLOR_FORMAT__MASK = 0x000007f8
A7XX_SP_A2D_OUTPUT_INFO_COLOR_FORMAT__SHIFT = 3
A7XX_SP_A2D_OUTPUT_INFO_SRGB = 0x00000800
A7XX_SP_A2D_OUTPUT_INFO_MASK__MASK = 0x0000f000
A7XX_SP_A2D_OUTPUT_INFO_MASK__SHIFT = 12
REG_A6XX_SP_DBG_ECO_CNTL = 0x0000ae00
REG_A6XX_SP_ADDR_MODE_CNTL = 0x0000ae01
REG_A6XX_SP_NC_MODE_CNTL = 0x0000ae02
REG_A6XX_SP_CHICKEN_BITS = 0x0000ae03
REG_A6XX_SP_NC_MODE_CNTL_2 = 0x0000ae04
A6XX_SP_NC_MODE_CNTL_2_F16_NO_INF = 0x00000008
REG_A7XX_SP_UNKNOWN_AE06 = 0x0000ae06
REG_A7XX_SP_CHICKEN_BITS_1 = 0x0000ae08
REG_A7XX_SP_CHICKEN_BITS_2 = 0x0000ae09
REG_A7XX_SP_CHICKEN_BITS_3 = 0x0000ae0a
REG_A6XX_SP_PERFCTR_SHADER_MASK = 0x0000ae0f
A6XX_SP_PERFCTR_SHADER_MASK_VS = 0x00000001
A6XX_SP_PERFCTR_SHADER_MASK_HS = 0x00000002
A6XX_SP_PERFCTR_SHADER_MASK_DS = 0x00000004
A6XX_SP_PERFCTR_SHADER_MASK_GS = 0x00000008
A6XX_SP_PERFCTR_SHADER_MASK_FS = 0x00000010
A6XX_SP_PERFCTR_SHADER_MASK_CS = 0x00000020
REG_A6XX_SP_PERFCTR_SP_SEL = lambda i0: (0x0000ae10 + 0x1*i0 )
REG_A7XX_SP_PERFCTR_HLSQ_SEL = lambda i0: (0x0000ae60 + 0x1*i0 )
REG_A7XX_SP_UNKNOWN_AE6A = 0x0000ae6a
REG_A7XX_SP_UNKNOWN_AE6B = 0x0000ae6b
REG_A7XX_SP_HLSQ_DBG_ECO_CNTL = 0x0000ae6c
REG_A7XX_SP_READ_SEL = 0x0000ae6d
A7XX_SP_READ_SEL_LOCATION__MASK = 0x000c0000
A7XX_SP_READ_SEL_LOCATION__SHIFT = 18
A7XX_SP_READ_SEL_PIPE__MASK = 0x00030000
A7XX_SP_READ_SEL_PIPE__SHIFT = 16
A7XX_SP_READ_SEL_STATETYPE__MASK = 0x0000ff00
A7XX_SP_READ_SEL_STATETYPE__SHIFT = 8
A7XX_SP_READ_SEL_USPTP__MASK = 0x000000f0
A7XX_SP_READ_SEL_USPTP__SHIFT = 4
A7XX_SP_READ_SEL_SPTP__MASK = 0x0000000f
A7XX_SP_READ_SEL_SPTP__SHIFT = 0
REG_A7XX_SP_DBG_CNTL = 0x0000ae71
REG_A7XX_SP_UNKNOWN_AE73 = 0x0000ae73
REG_A7XX_SP_PERFCTR_SP_SEL = lambda i0: (0x0000ae80 + 0x1*i0 )
REG_A6XX_SP_CONTEXT_SWITCH_GFX_PREEMPTION_SAFE_MODE = 0x0000be22
REG_A6XX_TPL1_CS_BORDER_COLOR_BASE = 0x0000b180
REG_A6XX_SP_UNKNOWN_B182 = 0x0000b182
REG_A6XX_SP_UNKNOWN_B183 = 0x0000b183
REG_A6XX_SP_UNKNOWN_B190 = 0x0000b190
REG_A6XX_SP_UNKNOWN_B191 = 0x0000b191
REG_A6XX_TPL1_RAS_MSAA_CNTL = 0x0000b300
A6XX_TPL1_RAS_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_TPL1_RAS_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_TPL1_RAS_MSAA_CNTL_UNK2__MASK = 0x0000000c
A6XX_TPL1_RAS_MSAA_CNTL_UNK2__SHIFT = 2
REG_A6XX_TPL1_DEST_MSAA_CNTL = 0x0000b301
A6XX_TPL1_DEST_MSAA_CNTL_SAMPLES__MASK = 0x00000003
A6XX_TPL1_DEST_MSAA_CNTL_SAMPLES__SHIFT = 0
A6XX_TPL1_DEST_MSAA_CNTL_MSAA_DISABLE = 0x00000004
REG_A6XX_TPL1_GFX_BORDER_COLOR_BASE = 0x0000b302
REG_A6XX_TPL1_MSAA_SAMPLE_POS_CNTL = 0x0000b304
A6XX_TPL1_MSAA_SAMPLE_POS_CNTL_UNK0 = 0x00000001
A6XX_TPL1_MSAA_SAMPLE_POS_CNTL_LOCATION_ENABLE = 0x00000002
REG_A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0 = 0x0000b305
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__MASK = 0x0000000f
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_X__SHIFT = 0
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_0_Y__SHIFT = 4
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__MASK = 0x00000f00
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_X__SHIFT = 8
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_1_Y__SHIFT = 12
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__MASK = 0x000f0000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_X__SHIFT = 16
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_2_Y__SHIFT = 20
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__MASK = 0x0f000000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_X__SHIFT = 24
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_0_SAMPLE_3_Y__SHIFT = 28
REG_A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1 = 0x0000b306
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__MASK = 0x0000000f
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_X__SHIFT = 0
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__MASK = 0x000000f0
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_0_Y__SHIFT = 4
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__MASK = 0x00000f00
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_X__SHIFT = 8
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__MASK = 0x0000f000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_1_Y__SHIFT = 12
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__MASK = 0x000f0000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_X__SHIFT = 16
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__MASK = 0x00f00000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_2_Y__SHIFT = 20
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__MASK = 0x0f000000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_X__SHIFT = 24
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__MASK = 0xf0000000
A6XX_TPL1_PROGRAMMABLE_MSAA_POS_1_SAMPLE_3_Y__SHIFT = 28
REG_A6XX_TPL1_WINDOW_OFFSET = 0x0000b307
A6XX_TPL1_WINDOW_OFFSET_X__MASK = 0x00003fff
A6XX_TPL1_WINDOW_OFFSET_X__SHIFT = 0
A6XX_TPL1_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A6XX_TPL1_WINDOW_OFFSET_Y__SHIFT = 16
REG_A6XX_TPL1_MODE_CNTL = 0x0000b309
A6XX_TPL1_MODE_CNTL_ISAMMODE__MASK = 0x00000003
A6XX_TPL1_MODE_CNTL_ISAMMODE__SHIFT = 0
A6XX_TPL1_MODE_CNTL_TEXCOORDROUNDMODE__MASK = 0x00000004
A6XX_TPL1_MODE_CNTL_TEXCOORDROUNDMODE__SHIFT = 2
A6XX_TPL1_MODE_CNTL_NEARESTMIPSNAP__MASK = 0x00000020
A6XX_TPL1_MODE_CNTL_NEARESTMIPSNAP__SHIFT = 5
A6XX_TPL1_MODE_CNTL_DESTDATATYPEOVERRIDE = 0x00000080
REG_A7XX_SP_UNKNOWN_B310 = 0x0000b310
REG_A6XX_TPL1_A2D_SRC_TEXTURE_INFO = 0x0000b4c0
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_FORMAT__MASK = 0x000000ff
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_FORMAT__SHIFT = 0
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_TILE_MODE__MASK = 0x00000300
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_TILE_MODE__SHIFT = 8
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_SWAP__MASK = 0x00000c00
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_SWAP__SHIFT = 10
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_FLAGS = 0x00001000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_SRGB = 0x00002000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES__MASK = 0x0000c000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES__SHIFT = 14
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_FILTER = 0x00010000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK17 = 0x00020000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES_AVERAGE = 0x00040000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK19 = 0x00080000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK20 = 0x00100000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK21 = 0x00200000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK22 = 0x00400000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK23__MASK = 0x07800000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK23__SHIFT = 23
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK28 = 0x10000000
A6XX_TPL1_A2D_SRC_TEXTURE_INFO_MUTABLEEN = 0x20000000
REG_A6XX_TPL1_A2D_SRC_TEXTURE_SIZE = 0x0000b4c1
A6XX_TPL1_A2D_SRC_TEXTURE_SIZE_WIDTH__MASK = 0x00007fff
A6XX_TPL1_A2D_SRC_TEXTURE_SIZE_WIDTH__SHIFT = 0
A6XX_TPL1_A2D_SRC_TEXTURE_SIZE_HEIGHT__MASK = 0x3fff8000
A6XX_TPL1_A2D_SRC_TEXTURE_SIZE_HEIGHT__SHIFT = 15
REG_A6XX_TPL1_A2D_SRC_TEXTURE_BASE = 0x0000b4c2
REG_A6XX_TPL1_A2D_SRC_TEXTURE_PITCH = 0x0000b4c4
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_UNK0__MASK = 0x000001ff
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_UNK0__SHIFT = 0
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_PITCH__MASK = 0x00fffe00
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_PITCH__SHIFT = 9
REG_A7XX_TPL1_A2D_SRC_TEXTURE_INFO = 0x0000b2c0
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_FORMAT__MASK = 0x000000ff
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_FORMAT__SHIFT = 0
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_TILE_MODE__MASK = 0x00000300
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_TILE_MODE__SHIFT = 8
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_SWAP__MASK = 0x00000c00
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_COLOR_SWAP__SHIFT = 10
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_FLAGS = 0x00001000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_SRGB = 0x00002000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES__MASK = 0x0000c000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES__SHIFT = 14
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_FILTER = 0x00010000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK17 = 0x00020000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_SAMPLES_AVERAGE = 0x00040000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK19 = 0x00080000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK20 = 0x00100000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK21 = 0x00200000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK22 = 0x00400000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK23__MASK = 0x07800000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK23__SHIFT = 23
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_UNK28 = 0x10000000
A7XX_TPL1_A2D_SRC_TEXTURE_INFO_MUTABLEEN = 0x20000000
REG_A7XX_TPL1_A2D_SRC_TEXTURE_SIZE = 0x0000b2c1
A7XX_TPL1_A2D_SRC_TEXTURE_SIZE_WIDTH__MASK = 0x00007fff
A7XX_TPL1_A2D_SRC_TEXTURE_SIZE_WIDTH__SHIFT = 0
A7XX_TPL1_A2D_SRC_TEXTURE_SIZE_HEIGHT__MASK = 0x3fff8000
A7XX_TPL1_A2D_SRC_TEXTURE_SIZE_HEIGHT__SHIFT = 15
REG_A7XX_TPL1_A2D_SRC_TEXTURE_BASE = 0x0000b2c2
REG_A7XX_TPL1_A2D_SRC_TEXTURE_PITCH = 0x0000b2c4
A7XX_TPL1_A2D_SRC_TEXTURE_PITCH_PITCH__MASK = 0x00fffff8
A7XX_TPL1_A2D_SRC_TEXTURE_PITCH_PITCH__SHIFT = 3
REG_A6XX_TPL1_A2D_SRC_TEXTURE_BASE_1 = 0x0000b4c5
REG_A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_1 = 0x0000b4c7
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_1__MASK = 0x00000fff
A6XX_TPL1_A2D_SRC_TEXTURE_PITCH_1__SHIFT = 0
REG_A6XX_TPL1_A2D_SRC_TEXTURE_BASE_2 = 0x0000b4c8
REG_A7XX_TPL1_A2D_SRC_TEXTURE_BASE_1 = 0x0000b2c5
REG_A7XX_TPL1_A2D_SRC_TEXTURE_PITCH_1 = 0x0000b2c7
A7XX_TPL1_A2D_SRC_TEXTURE_PITCH_1__MASK = 0x00000fff
A7XX_TPL1_A2D_SRC_TEXTURE_PITCH_1__SHIFT = 0
REG_A7XX_TPL1_A2D_SRC_TEXTURE_BASE_2 = 0x0000b2c8
REG_A6XX_TPL1_A2D_SRC_TEXTURE_FLAG_BASE = 0x0000b4ca
REG_A6XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH = 0x0000b4cc
A6XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH__MASK = 0x000000ff
A6XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH__SHIFT = 0
REG_A7XX_TPL1_A2D_SRC_TEXTURE_FLAG_BASE = 0x0000b2ca
REG_A7XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH = 0x0000b2cc
A7XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH__MASK = 0x000000ff
A7XX_TPL1_A2D_SRC_TEXTURE_FLAG_PITCH__SHIFT = 0
REG_A6XX_SP_PS_UNKNOWN_B4CD = 0x0000b4cd
REG_A6XX_SP_PS_UNKNOWN_B4CE = 0x0000b4ce
REG_A6XX_SP_PS_UNKNOWN_B4CF = 0x0000b4cf
REG_A6XX_SP_PS_UNKNOWN_B4D0 = 0x0000b4d0
REG_A6XX_SP_WINDOW_OFFSET = 0x0000b4d1
A6XX_SP_WINDOW_OFFSET_X__MASK = 0x00003fff
A6XX_SP_WINDOW_OFFSET_X__SHIFT = 0
A6XX_SP_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A6XX_SP_WINDOW_OFFSET_Y__SHIFT = 16
REG_A7XX_SP_PS_UNKNOWN_B4CD = 0x0000b2cd
REG_A7XX_SP_PS_UNKNOWN_B4CE = 0x0000b2ce
REG_A7XX_SP_PS_UNKNOWN_B4CF = 0x0000b2cf
REG_A7XX_SP_PS_UNKNOWN_B4D0 = 0x0000b2d0
REG_A7XX_TPL1_A2D_WINDOW_OFFSET = 0x0000b2d1
A7XX_TPL1_A2D_WINDOW_OFFSET_X__MASK = 0x00003fff
A7XX_TPL1_A2D_WINDOW_OFFSET_X__SHIFT = 0
A7XX_TPL1_A2D_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A7XX_TPL1_A2D_WINDOW_OFFSET_Y__SHIFT = 16
REG_A7XX_TPL1_A2D_BLT_CNTL = 0x0000b2d2
A7XX_TPL1_A2D_BLT_CNTL_RAW_COPY = 0x00000001
A7XX_TPL1_A2D_BLT_CNTL_START_OFFSET_TEXELS__MASK = 0x003f0000
A7XX_TPL1_A2D_BLT_CNTL_START_OFFSET_TEXELS__SHIFT = 16
A7XX_TPL1_A2D_BLT_CNTL_TYPE__MASK = 0xe0000000
A7XX_TPL1_A2D_BLT_CNTL_TYPE__SHIFT = 29
REG_A7XX_SP_WINDOW_OFFSET = 0x0000ab21
A7XX_SP_WINDOW_OFFSET_X__MASK = 0x00003fff
A7XX_SP_WINDOW_OFFSET_X__SHIFT = 0
A7XX_SP_WINDOW_OFFSET_Y__MASK = 0x3fff0000
A7XX_SP_WINDOW_OFFSET_Y__SHIFT = 16
REG_A6XX_TPL1_DBG_ECO_CNTL = 0x0000b600
REG_A6XX_TPL1_ADDR_MODE_CNTL = 0x0000b601
REG_A6XX_TPL1_DBG_ECO_CNTL1 = 0x0000b602
A6XX_TPL1_DBG_ECO_CNTL1_TP_UBWC_FLAG_HINT = 0x00040000
REG_A6XX_TPL1_NC_MODE_CNTL = 0x0000b604
A6XX_TPL1_NC_MODE_CNTL_MODE = 0x00000001
A6XX_TPL1_NC_MODE_CNTL_LOWER_BIT__MASK = 0x00000006
A6XX_TPL1_NC_MODE_CNTL_LOWER_BIT__SHIFT = 1
A6XX_TPL1_NC_MODE_CNTL_MIN_ACCESS_LENGTH = 0x00000008
A6XX_TPL1_NC_MODE_CNTL_UPPER_BIT__MASK = 0x00000010
A6XX_TPL1_NC_MODE_CNTL_UPPER_BIT__SHIFT = 4
A6XX_TPL1_NC_MODE_CNTL_UNK6__MASK = 0x000000c0
A6XX_TPL1_NC_MODE_CNTL_UNK6__SHIFT = 6
REG_A6XX_TPL1_UNKNOWN_B605 = 0x0000b605
REG_A6XX_TPL1_BICUBIC_WEIGHTS_TABLE_0 = 0x0000b608
REG_A6XX_TPL1_BICUBIC_WEIGHTS_TABLE_1 = 0x0000b609
REG_A6XX_TPL1_BICUBIC_WEIGHTS_TABLE_2 = 0x0000b60a
REG_A6XX_TPL1_BICUBIC_WEIGHTS_TABLE_3 = 0x0000b60b
REG_A6XX_TPL1_BICUBIC_WEIGHTS_TABLE_4 = 0x0000b60c
REG_A7XX_TPL1_BICUBIC_WEIGHTS_TABLE_0 = 0x0000b608
REG_A7XX_TPL1_BICUBIC_WEIGHTS_TABLE_1 = 0x0000b609
REG_A7XX_TPL1_BICUBIC_WEIGHTS_TABLE_2 = 0x0000b60a
REG_A7XX_TPL1_BICUBIC_WEIGHTS_TABLE_3 = 0x0000b60b
REG_A7XX_TPL1_BICUBIC_WEIGHTS_TABLE_4 = 0x0000b60c
REG_A6XX_TPL1_PERFCTR_TP_SEL = lambda i0: (0x0000b610 + 0x1*i0 )
REG_A7XX_TPL1_PERFCTR_TP_SEL = lambda i0: (0x0000b610 + 0x1*i0 )
REG_A6XX_SP_VS_CONST_CONFIG = 0x0000b800
A6XX_SP_VS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_VS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_VS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_VS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A6XX_SP_HS_CONST_CONFIG = 0x0000b801
A6XX_SP_HS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_HS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_HS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_HS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A6XX_SP_DS_CONST_CONFIG = 0x0000b802
A6XX_SP_DS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_DS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_DS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_DS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A6XX_SP_GS_CONST_CONFIG = 0x0000b803
A6XX_SP_GS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_GS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_GS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_GS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_VS_CONST_CONFIG = 0x0000a827
A7XX_SP_VS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_VS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_VS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_VS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_HS_CONST_CONFIG = 0x0000a83f
A7XX_SP_HS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_HS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_HS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_HS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_DS_CONST_CONFIG = 0x0000a867
A7XX_SP_DS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_DS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_DS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_DS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_GS_CONST_CONFIG = 0x0000a898
A7XX_SP_GS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_GS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_GS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_GS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_RENDER_CNTL = 0x0000a9aa
A7XX_SP_RENDER_CNTL_FS_DISABLE = 0x00000001
REG_A7XX_SP_DITHER_CNTL = 0x0000a9ac
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT0__MASK = 0x00000003
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT0__SHIFT = 0
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT1__MASK = 0x0000000c
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT1__SHIFT = 2
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT2__MASK = 0x00000030
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT2__SHIFT = 4
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT3__MASK = 0x000000c0
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT3__SHIFT = 6
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT4__MASK = 0x00000300
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT4__SHIFT = 8
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT5__MASK = 0x00000c00
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT5__SHIFT = 10
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT6__MASK = 0x00003000
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT6__SHIFT = 12
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT7__MASK = 0x0000c000
A7XX_SP_DITHER_CNTL_DITHER_MODE_MRT7__SHIFT = 14
REG_A7XX_SP_VRS_CONFIG = 0x0000a9ad
A7XX_SP_VRS_CONFIG_PIPELINE_FSR_ENABLE = 0x00000001
A7XX_SP_VRS_CONFIG_ATTACHMENT_FSR_ENABLE = 0x00000002
A7XX_SP_VRS_CONFIG_PRIMITIVE_FSR_ENABLE = 0x00000008
REG_A7XX_SP_PS_CNTL_1 = 0x0000a9ae
A7XX_SP_PS_CNTL_1_SYSVAL_REGS_COUNT__MASK = 0x000000ff
A7XX_SP_PS_CNTL_1_SYSVAL_REGS_COUNT__SHIFT = 0
A7XX_SP_PS_CNTL_1_UNK8 = 0x00000100
A7XX_SP_PS_CNTL_1_UNK9 = 0x00000200
REG_A6XX_HLSQ_LOAD_STATE_GEOM_CMD = 0x0000b820
REG_A6XX_HLSQ_LOAD_STATE_GEOM_EXT_SRC_ADDR = 0x0000b821
REG_A6XX_HLSQ_LOAD_STATE_GEOM_DATA = 0x0000b823
REG_A6XX_SP_PS_WAVE_CNTL = 0x0000b980
A6XX_SP_PS_WAVE_CNTL_THREADSIZE__MASK = 0x00000001
A6XX_SP_PS_WAVE_CNTL_THREADSIZE__SHIFT = 0
A6XX_SP_PS_WAVE_CNTL_VARYINGS = 0x00000002
A6XX_SP_PS_WAVE_CNTL_UNK2__MASK = 0x00000ffc
A6XX_SP_PS_WAVE_CNTL_UNK2__SHIFT = 2
REG_A6XX_HLSQ_UNKNOWN_B981 = 0x0000b981
REG_A6XX_SP_LB_PARAM_LIMIT = 0x0000b982
A6XX_SP_LB_PARAM_LIMIT_PRIMALLOCTHRESHOLD__MASK = 0x00000007
A6XX_SP_LB_PARAM_LIMIT_PRIMALLOCTHRESHOLD__SHIFT = 0
REG_A6XX_SP_REG_PROG_ID_0 = 0x0000b983
A6XX_SP_REG_PROG_ID_0_FACEREGID__MASK = 0x000000ff
A6XX_SP_REG_PROG_ID_0_FACEREGID__SHIFT = 0
A6XX_SP_REG_PROG_ID_0_SAMPLEID__MASK = 0x0000ff00
A6XX_SP_REG_PROG_ID_0_SAMPLEID__SHIFT = 8
A6XX_SP_REG_PROG_ID_0_SAMPLEMASK__MASK = 0x00ff0000
A6XX_SP_REG_PROG_ID_0_SAMPLEMASK__SHIFT = 16
A6XX_SP_REG_PROG_ID_0_CENTERRHW__MASK = 0xff000000
A6XX_SP_REG_PROG_ID_0_CENTERRHW__SHIFT = 24
REG_A6XX_SP_REG_PROG_ID_1 = 0x0000b984
A6XX_SP_REG_PROG_ID_1_IJ_PERSP_PIXEL__MASK = 0x000000ff
A6XX_SP_REG_PROG_ID_1_IJ_PERSP_PIXEL__SHIFT = 0
A6XX_SP_REG_PROG_ID_1_IJ_LINEAR_PIXEL__MASK = 0x0000ff00
A6XX_SP_REG_PROG_ID_1_IJ_LINEAR_PIXEL__SHIFT = 8
A6XX_SP_REG_PROG_ID_1_IJ_PERSP_CENTROID__MASK = 0x00ff0000
A6XX_SP_REG_PROG_ID_1_IJ_PERSP_CENTROID__SHIFT = 16
A6XX_SP_REG_PROG_ID_1_IJ_LINEAR_CENTROID__MASK = 0xff000000
A6XX_SP_REG_PROG_ID_1_IJ_LINEAR_CENTROID__SHIFT = 24
REG_A6XX_SP_REG_PROG_ID_2 = 0x0000b985
A6XX_SP_REG_PROG_ID_2_IJ_PERSP_SAMPLE__MASK = 0x000000ff
A6XX_SP_REG_PROG_ID_2_IJ_PERSP_SAMPLE__SHIFT = 0
A6XX_SP_REG_PROG_ID_2_IJ_LINEAR_SAMPLE__MASK = 0x0000ff00
A6XX_SP_REG_PROG_ID_2_IJ_LINEAR_SAMPLE__SHIFT = 8
A6XX_SP_REG_PROG_ID_2_XYCOORDREGID__MASK = 0x00ff0000
A6XX_SP_REG_PROG_ID_2_XYCOORDREGID__SHIFT = 16
A6XX_SP_REG_PROG_ID_2_ZWCOORDREGID__MASK = 0xff000000
A6XX_SP_REG_PROG_ID_2_ZWCOORDREGID__SHIFT = 24
REG_A6XX_SP_REG_PROG_ID_3 = 0x0000b986
A6XX_SP_REG_PROG_ID_3_LINELENGTHREGID__MASK = 0x000000ff
A6XX_SP_REG_PROG_ID_3_LINELENGTHREGID__SHIFT = 0
A6XX_SP_REG_PROG_ID_3_FOVEATIONQUALITYREGID__MASK = 0x0000ff00
A6XX_SP_REG_PROG_ID_3_FOVEATIONQUALITYREGID__SHIFT = 8
REG_A6XX_SP_CS_CONST_CONFIG = 0x0000b987
A6XX_SP_CS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_CS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_CS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_CS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_PS_WAVE_CNTL = 0x0000a9c6
A7XX_SP_PS_WAVE_CNTL_THREADSIZE__MASK = 0x00000001
A7XX_SP_PS_WAVE_CNTL_THREADSIZE__SHIFT = 0
A7XX_SP_PS_WAVE_CNTL_VARYINGS = 0x00000002
A7XX_SP_PS_WAVE_CNTL_UNK2__MASK = 0x00000ffc
A7XX_SP_PS_WAVE_CNTL_UNK2__SHIFT = 2
REG_A7XX_SP_LB_PARAM_LIMIT = 0x0000a9c7
A7XX_SP_LB_PARAM_LIMIT_PRIMALLOCTHRESHOLD__MASK = 0x00000007
A7XX_SP_LB_PARAM_LIMIT_PRIMALLOCTHRESHOLD__SHIFT = 0
REG_A7XX_SP_REG_PROG_ID_0 = 0x0000a9c8
A7XX_SP_REG_PROG_ID_0_FACEREGID__MASK = 0x000000ff
A7XX_SP_REG_PROG_ID_0_FACEREGID__SHIFT = 0
A7XX_SP_REG_PROG_ID_0_SAMPLEID__MASK = 0x0000ff00
A7XX_SP_REG_PROG_ID_0_SAMPLEID__SHIFT = 8
A7XX_SP_REG_PROG_ID_0_SAMPLEMASK__MASK = 0x00ff0000
A7XX_SP_REG_PROG_ID_0_SAMPLEMASK__SHIFT = 16
A7XX_SP_REG_PROG_ID_0_CENTERRHW__MASK = 0xff000000
A7XX_SP_REG_PROG_ID_0_CENTERRHW__SHIFT = 24
REG_A7XX_SP_REG_PROG_ID_1 = 0x0000a9c9
A7XX_SP_REG_PROG_ID_1_IJ_PERSP_PIXEL__MASK = 0x000000ff
A7XX_SP_REG_PROG_ID_1_IJ_PERSP_PIXEL__SHIFT = 0
A7XX_SP_REG_PROG_ID_1_IJ_LINEAR_PIXEL__MASK = 0x0000ff00
A7XX_SP_REG_PROG_ID_1_IJ_LINEAR_PIXEL__SHIFT = 8
A7XX_SP_REG_PROG_ID_1_IJ_PERSP_CENTROID__MASK = 0x00ff0000
A7XX_SP_REG_PROG_ID_1_IJ_PERSP_CENTROID__SHIFT = 16
A7XX_SP_REG_PROG_ID_1_IJ_LINEAR_CENTROID__MASK = 0xff000000
A7XX_SP_REG_PROG_ID_1_IJ_LINEAR_CENTROID__SHIFT = 24
REG_A7XX_SP_REG_PROG_ID_2 = 0x0000a9ca
A7XX_SP_REG_PROG_ID_2_IJ_PERSP_SAMPLE__MASK = 0x000000ff
A7XX_SP_REG_PROG_ID_2_IJ_PERSP_SAMPLE__SHIFT = 0
A7XX_SP_REG_PROG_ID_2_IJ_LINEAR_SAMPLE__MASK = 0x0000ff00
A7XX_SP_REG_PROG_ID_2_IJ_LINEAR_SAMPLE__SHIFT = 8
A7XX_SP_REG_PROG_ID_2_XYCOORDREGID__MASK = 0x00ff0000
A7XX_SP_REG_PROG_ID_2_XYCOORDREGID__SHIFT = 16
A7XX_SP_REG_PROG_ID_2_ZWCOORDREGID__MASK = 0xff000000
A7XX_SP_REG_PROG_ID_2_ZWCOORDREGID__SHIFT = 24
REG_A7XX_SP_REG_PROG_ID_3 = 0x0000a9cb
A7XX_SP_REG_PROG_ID_3_LINELENGTHREGID__MASK = 0x000000ff
A7XX_SP_REG_PROG_ID_3_LINELENGTHREGID__SHIFT = 0
A7XX_SP_REG_PROG_ID_3_FOVEATIONQUALITYREGID__MASK = 0x0000ff00
A7XX_SP_REG_PROG_ID_3_FOVEATIONQUALITYREGID__SHIFT = 8
REG_A7XX_SP_CS_CONST_CONFIG = 0x0000a9cd
A7XX_SP_CS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_CS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_CS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_CS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A6XX_SP_CS_NDRANGE_0 = 0x0000b990
A6XX_SP_CS_NDRANGE_0_KERNELDIM__MASK = 0x00000003
A6XX_SP_CS_NDRANGE_0_KERNELDIM__SHIFT = 0
A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK = 0x00000ffc
A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT = 2
A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK = 0x003ff000
A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT = 12
A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK = 0xffc00000
A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT = 22
REG_A6XX_SP_CS_NDRANGE_1 = 0x0000b991
A6XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__SHIFT = 0
REG_A6XX_SP_CS_NDRANGE_2 = 0x0000b992
A6XX_SP_CS_NDRANGE_2_GLOBALOFF_X__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_2_GLOBALOFF_X__SHIFT = 0
REG_A6XX_SP_CS_NDRANGE_3 = 0x0000b993
A6XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__SHIFT = 0
REG_A6XX_SP_CS_NDRANGE_4 = 0x0000b994
A6XX_SP_CS_NDRANGE_4_GLOBALOFF_Y__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_4_GLOBALOFF_Y__SHIFT = 0
REG_A6XX_SP_CS_NDRANGE_5 = 0x0000b995
A6XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__SHIFT = 0
REG_A6XX_SP_CS_NDRANGE_6 = 0x0000b996
A6XX_SP_CS_NDRANGE_6_GLOBALOFF_Z__MASK = 0xffffffff
A6XX_SP_CS_NDRANGE_6_GLOBALOFF_Z__SHIFT = 0
REG_A6XX_SP_CS_CONST_CONFIG_0 = 0x0000b997
A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__MASK = 0x000000ff
A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT = 0
A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID__MASK = 0x0000ff00
A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID__SHIFT = 8
A6XX_SP_CS_CONST_CONFIG_0_WGOFFSETCONSTID__MASK = 0x00ff0000
A6XX_SP_CS_CONST_CONFIG_0_WGOFFSETCONSTID__SHIFT = 16
A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__MASK = 0xff000000
A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT = 24
REG_A6XX_SP_CS_WGE_CNTL = 0x0000b998
A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__MASK = 0x000000ff
A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__SHIFT = 0
A6XX_SP_CS_WGE_CNTL_SINGLE_SP_CORE = 0x00000100
A6XX_SP_CS_WGE_CNTL_THREADSIZE__MASK = 0x00000200
A6XX_SP_CS_WGE_CNTL_THREADSIZE__SHIFT = 9
A6XX_SP_CS_WGE_CNTL_THREADSIZE_SCALAR = 0x00000400
REG_A6XX_SP_CS_KERNEL_GROUP_X = 0x0000b999
REG_A6XX_SP_CS_KERNEL_GROUP_Y = 0x0000b99a
REG_A6XX_SP_CS_KERNEL_GROUP_Z = 0x0000b99b
REG_A7XX_SP_CS_NDRANGE_0 = 0x0000a9d4
A7XX_SP_CS_NDRANGE_0_KERNELDIM__MASK = 0x00000003
A7XX_SP_CS_NDRANGE_0_KERNELDIM__SHIFT = 0
A7XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK = 0x00000ffc
A7XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT = 2
A7XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK = 0x003ff000
A7XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT = 12
A7XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK = 0xffc00000
A7XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT = 22
REG_A7XX_SP_CS_NDRANGE_1 = 0x0000a9d5
A7XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__SHIFT = 0
REG_A7XX_SP_CS_NDRANGE_2 = 0x0000a9d6
A7XX_SP_CS_NDRANGE_2_GLOBALOFF_X__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_2_GLOBALOFF_X__SHIFT = 0
REG_A7XX_SP_CS_NDRANGE_3 = 0x0000a9d7
A7XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__SHIFT = 0
REG_A7XX_SP_CS_NDRANGE_4 = 0x0000a9d8
A7XX_SP_CS_NDRANGE_4_GLOBALOFF_Y__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_4_GLOBALOFF_Y__SHIFT = 0
REG_A7XX_SP_CS_NDRANGE_5 = 0x0000a9d9
A7XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__SHIFT = 0
REG_A7XX_SP_CS_NDRANGE_6 = 0x0000a9da
A7XX_SP_CS_NDRANGE_6_GLOBALOFF_Z__MASK = 0xffffffff
A7XX_SP_CS_NDRANGE_6_GLOBALOFF_Z__SHIFT = 0
REG_A7XX_SP_CS_KERNEL_GROUP_X = 0x0000a9dc
REG_A7XX_SP_CS_KERNEL_GROUP_Y = 0x0000a9dd
REG_A7XX_SP_CS_KERNEL_GROUP_Z = 0x0000a9de
REG_A7XX_SP_CS_WGE_CNTL = 0x0000a9db
A7XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__MASK = 0x000000ff
A7XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__SHIFT = 0
A7XX_SP_CS_WGE_CNTL_THREADSIZE__MASK = 0x00000200
A7XX_SP_CS_WGE_CNTL_THREADSIZE__SHIFT = 9
A7XX_SP_CS_WGE_CNTL_WORKGROUPRASTORDERZFIRSTEN = 0x00000800
A7XX_SP_CS_WGE_CNTL_WGTILEWIDTH__MASK = 0x03f00000
A7XX_SP_CS_WGE_CNTL_WGTILEWIDTH__SHIFT = 20
A7XX_SP_CS_WGE_CNTL_WGTILEHEIGHT__MASK = 0xfc000000
A7XX_SP_CS_WGE_CNTL_WGTILEHEIGHT__SHIFT = 26
REG_A7XX_SP_CS_NDRANGE_7 = 0x0000a9df
A7XX_SP_CS_NDRANGE_7_LOCALSIZEX__MASK = 0x00000ffc
A7XX_SP_CS_NDRANGE_7_LOCALSIZEX__SHIFT = 2
A7XX_SP_CS_NDRANGE_7_LOCALSIZEY__MASK = 0x003ff000
A7XX_SP_CS_NDRANGE_7_LOCALSIZEY__SHIFT = 12
A7XX_SP_CS_NDRANGE_7_LOCALSIZEZ__MASK = 0xffc00000
A7XX_SP_CS_NDRANGE_7_LOCALSIZEZ__SHIFT = 22
REG_A6XX_HLSQ_LOAD_STATE_FRAG_CMD = 0x0000b9a0
REG_A6XX_HLSQ_LOAD_STATE_FRAG_EXT_SRC_ADDR = 0x0000b9a1
REG_A6XX_HLSQ_LOAD_STATE_FRAG_DATA = 0x0000b9a3
REG_A6XX_HLSQ_CS_BINDLESS_BASE = lambda i0: (0x0000b9c0 + 0x2*i0 )
A6XX_HLSQ_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A6XX_HLSQ_CS_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A6XX_HLSQ_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A6XX_HLSQ_CS_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A6XX_HLSQ_CS_CTRL_REG1 = 0x0000b9d0
A6XX_HLSQ_CS_CTRL_REG1_SHARED_SIZE__MASK = 0x0000001f
A6XX_HLSQ_CS_CTRL_REG1_SHARED_SIZE__SHIFT = 0
A6XX_HLSQ_CS_CTRL_REG1_CONSTANTRAMMODE__MASK = 0x00000060
A6XX_HLSQ_CS_CTRL_REG1_CONSTANTRAMMODE__SHIFT = 5
REG_A6XX_SP_DRAW_INITIATOR = 0x0000bb00
A6XX_SP_DRAW_INITIATOR_STATE_ID__MASK = 0x000000ff
A6XX_SP_DRAW_INITIATOR_STATE_ID__SHIFT = 0
REG_A6XX_SP_KERNEL_INITIATOR = 0x0000bb01
A6XX_SP_KERNEL_INITIATOR_STATE_ID__MASK = 0x000000ff
A6XX_SP_KERNEL_INITIATOR_STATE_ID__SHIFT = 0
REG_A6XX_SP_EVENT_INITIATOR = 0x0000bb02
A6XX_SP_EVENT_INITIATOR_STATE_ID__MASK = 0x00ff0000
A6XX_SP_EVENT_INITIATOR_STATE_ID__SHIFT = 16
A6XX_SP_EVENT_INITIATOR_EVENT__MASK = 0x0000007f
A6XX_SP_EVENT_INITIATOR_EVENT__SHIFT = 0
REG_A6XX_SP_UPDATE_CNTL = 0x0000bb08
A6XX_SP_UPDATE_CNTL_VS_STATE = 0x00000001
A6XX_SP_UPDATE_CNTL_HS_STATE = 0x00000002
A6XX_SP_UPDATE_CNTL_DS_STATE = 0x00000004
A6XX_SP_UPDATE_CNTL_GS_STATE = 0x00000008
A6XX_SP_UPDATE_CNTL_FS_STATE = 0x00000010
A6XX_SP_UPDATE_CNTL_CS_STATE = 0x00000020
A6XX_SP_UPDATE_CNTL_CS_UAV = 0x00000040
A6XX_SP_UPDATE_CNTL_GFX_UAV = 0x00000080
A6XX_SP_UPDATE_CNTL_CS_SHARED_CONST = 0x00080000
A6XX_SP_UPDATE_CNTL_GFX_SHARED_CONST = 0x00000100
A6XX_SP_UPDATE_CNTL_CS_BINDLESS__MASK = 0x00003e00
A6XX_SP_UPDATE_CNTL_CS_BINDLESS__SHIFT = 9
A6XX_SP_UPDATE_CNTL_GFX_BINDLESS__MASK = 0x0007c000
A6XX_SP_UPDATE_CNTL_GFX_BINDLESS__SHIFT = 14
REG_A7XX_SP_DRAW_INITIATOR = 0x0000ab1c
A7XX_SP_DRAW_INITIATOR_STATE_ID__MASK = 0x000000ff
A7XX_SP_DRAW_INITIATOR_STATE_ID__SHIFT = 0
REG_A7XX_SP_KERNEL_INITIATOR = 0x0000ab1d
A7XX_SP_KERNEL_INITIATOR_STATE_ID__MASK = 0x000000ff
A7XX_SP_KERNEL_INITIATOR_STATE_ID__SHIFT = 0
REG_A7XX_SP_EVENT_INITIATOR = 0x0000ab1e
A7XX_SP_EVENT_INITIATOR_STATE_ID__MASK = 0x00ff0000
A7XX_SP_EVENT_INITIATOR_STATE_ID__SHIFT = 16
A7XX_SP_EVENT_INITIATOR_EVENT__MASK = 0x0000007f
A7XX_SP_EVENT_INITIATOR_EVENT__SHIFT = 0
REG_A7XX_SP_UPDATE_CNTL = 0x0000ab1f
A7XX_SP_UPDATE_CNTL_VS_STATE = 0x00000001
A7XX_SP_UPDATE_CNTL_HS_STATE = 0x00000002
A7XX_SP_UPDATE_CNTL_DS_STATE = 0x00000004
A7XX_SP_UPDATE_CNTL_GS_STATE = 0x00000008
A7XX_SP_UPDATE_CNTL_FS_STATE = 0x00000010
A7XX_SP_UPDATE_CNTL_CS_STATE = 0x00000020
A7XX_SP_UPDATE_CNTL_CS_UAV = 0x00000040
A7XX_SP_UPDATE_CNTL_GFX_UAV = 0x00000080
A7XX_SP_UPDATE_CNTL_CS_BINDLESS__MASK = 0x0001fe00
A7XX_SP_UPDATE_CNTL_CS_BINDLESS__SHIFT = 9
A7XX_SP_UPDATE_CNTL_GFX_BINDLESS__MASK = 0x01fe0000
A7XX_SP_UPDATE_CNTL_GFX_BINDLESS__SHIFT = 17
REG_A6XX_SP_PS_CONST_CONFIG = 0x0000bb10
A6XX_SP_PS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A6XX_SP_PS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A6XX_SP_PS_CONST_CONFIG_ENABLED = 0x00000100
A6XX_SP_PS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_PS_CONST_CONFIG = 0x0000ab03
A7XX_SP_PS_CONST_CONFIG_CONSTLEN__MASK = 0x000000ff
A7XX_SP_PS_CONST_CONFIG_CONSTLEN__SHIFT = 0
A7XX_SP_PS_CONST_CONFIG_ENABLED = 0x00000100
A7XX_SP_PS_CONST_CONFIG_READ_IMM_SHARED_CONSTS = 0x00000200
REG_A7XX_SP_SHARED_CONSTANT_GFX_0 = lambda i0: (0x0000ab40 + 0x1*i0 )
REG_A6XX_HLSQ_SHARED_CONSTS = 0x0000bb11
A6XX_HLSQ_SHARED_CONSTS_ENABLE = 0x00000001
REG_A6XX_HLSQ_BINDLESS_BASE = lambda i0: (0x0000bb20 + 0x2*i0 )
A6XX_HLSQ_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__MASK = 0x00000003
A6XX_HLSQ_BINDLESS_BASE_DESCRIPTOR_DESC_SIZE__SHIFT = 0
A6XX_HLSQ_BINDLESS_BASE_DESCRIPTOR_ADDR__MASK = 0xfffffffffffffffc
A6XX_HLSQ_BINDLESS_BASE_DESCRIPTOR_ADDR__SHIFT = 2
REG_A6XX_HLSQ_2D_EVENT_CMD = 0x0000bd80
A6XX_HLSQ_2D_EVENT_CMD_STATE_ID__MASK = 0x0000ff00
A6XX_HLSQ_2D_EVENT_CMD_STATE_ID__SHIFT = 8
A6XX_HLSQ_2D_EVENT_CMD_EVENT__MASK = 0x0000007f
A6XX_HLSQ_2D_EVENT_CMD_EVENT__SHIFT = 0
REG_A6XX_HLSQ_UNKNOWN_BE00 = 0x0000be00
REG_A6XX_HLSQ_UNKNOWN_BE01 = 0x0000be01
REG_A6XX_HLSQ_DBG_ECO_CNTL = 0x0000be04
REG_A6XX_HLSQ_ADDR_MODE_CNTL = 0x0000be05
REG_A6XX_HLSQ_UNKNOWN_BE08 = 0x0000be08
REG_A6XX_HLSQ_PERFCTR_HLSQ_SEL = lambda i0: (0x0000be10 + 0x1*i0 )
REG_A6XX_HLSQ_CONTEXT_SWITCH_GFX_PREEMPTION_SAFE_MODE = 0x0000be22
REG_A7XX_SP_AHB_READ_APERTURE = 0x0000c000
REG_A7XX_SP_UNKNOWN_0CE2 = 0x00000ce2
REG_A7XX_SP_UNKNOWN_0CE4 = 0x00000ce4
REG_A7XX_SP_UNKNOWN_0CE6 = 0x00000ce6
REG_A6XX_CP_EVENT_START = 0x0000d600
A6XX_CP_EVENT_START_STATE_ID__MASK = 0x000000ff
A6XX_CP_EVENT_START_STATE_ID__SHIFT = 0
REG_A6XX_CP_EVENT_END = 0x0000d601
A6XX_CP_EVENT_END_STATE_ID__MASK = 0x000000ff
A6XX_CP_EVENT_END_STATE_ID__SHIFT = 0
REG_A6XX_CP_2D_EVENT_START = 0x0000d700
A6XX_CP_2D_EVENT_START_STATE_ID__MASK = 0x000000ff
A6XX_CP_2D_EVENT_START_STATE_ID__SHIFT = 0
REG_A6XX_CP_2D_EVENT_END = 0x0000d701
A6XX_CP_2D_EVENT_END_STATE_ID__MASK = 0x000000ff
A6XX_CP_2D_EVENT_END_STATE_ID__SHIFT = 0
REG_A6XX_PDC_GPU_ENABLE_PDC = 0x00001140
REG_A6XX_PDC_GPU_SEQ_START_ADDR = 0x00001148
REG_A6XX_PDC_GPU_TCS0_CONTROL = 0x00001540
REG_A6XX_PDC_GPU_TCS0_CMD_ENABLE_BANK = 0x00001541
REG_A6XX_PDC_GPU_TCS0_CMD_WAIT_FOR_CMPL_BANK = 0x00001542
REG_A6XX_PDC_GPU_TCS0_CMD0_MSGID = 0x00001543
REG_A6XX_PDC_GPU_TCS0_CMD0_ADDR = 0x00001544
REG_A6XX_PDC_GPU_TCS0_CMD0_DATA = 0x00001545
REG_A6XX_PDC_GPU_TCS1_CONTROL = 0x00001572
REG_A6XX_PDC_GPU_TCS1_CMD_ENABLE_BANK = 0x00001573
REG_A6XX_PDC_GPU_TCS1_CMD_WAIT_FOR_CMPL_BANK = 0x00001574
REG_A6XX_PDC_GPU_TCS1_CMD0_MSGID = 0x00001575
REG_A6XX_PDC_GPU_TCS1_CMD0_ADDR = 0x00001576
REG_A6XX_PDC_GPU_TCS1_CMD0_DATA = 0x00001577
REG_A6XX_PDC_GPU_TCS2_CONTROL = 0x000015a4
REG_A6XX_PDC_GPU_TCS2_CMD_ENABLE_BANK = 0x000015a5
REG_A6XX_PDC_GPU_TCS2_CMD_WAIT_FOR_CMPL_BANK = 0x000015a6
REG_A6XX_PDC_GPU_TCS2_CMD0_MSGID = 0x000015a7
REG_A6XX_PDC_GPU_TCS2_CMD0_ADDR = 0x000015a8
REG_A6XX_PDC_GPU_TCS2_CMD0_DATA = 0x000015a9
REG_A6XX_PDC_GPU_TCS3_CONTROL = 0x000015d6
REG_A6XX_PDC_GPU_TCS3_CMD_ENABLE_BANK = 0x000015d7
REG_A6XX_PDC_GPU_TCS3_CMD_WAIT_FOR_CMPL_BANK = 0x000015d8
REG_A6XX_PDC_GPU_TCS3_CMD0_MSGID = 0x000015d9
REG_A6XX_PDC_GPU_TCS3_CMD0_ADDR = 0x000015da
REG_A6XX_PDC_GPU_TCS3_CMD0_DATA = 0x000015db
REG_A6XX_PDC_GPU_SEQ_MEM_0 = 0x00000000
REG_A6XX_CX_DBGC_CFG_DBGBUS_SEL_A = 0x00000000
A6XX_CX_DBGC_CFG_DBGBUS_SEL_A_PING_INDEX__MASK = 0x000000ff
A6XX_CX_DBGC_CFG_DBGBUS_SEL_A_PING_INDEX__SHIFT = 0
A6XX_CX_DBGC_CFG_DBGBUS_SEL_A_PING_BLK_SEL__MASK = 0x0000ff00
A6XX_CX_DBGC_CFG_DBGBUS_SEL_A_PING_BLK_SEL__SHIFT = 8
REG_A6XX_CX_DBGC_CFG_DBGBUS_SEL_B = 0x00000001
REG_A6XX_CX_DBGC_CFG_DBGBUS_SEL_C = 0x00000002
REG_A6XX_CX_DBGC_CFG_DBGBUS_SEL_D = 0x00000003
REG_A6XX_CX_DBGC_CFG_DBGBUS_CNTLT = 0x00000004
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_TRACEEN__MASK = 0x0000003f
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_TRACEEN__SHIFT = 0
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_GRANU__MASK = 0x00007000
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_GRANU__SHIFT = 12
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_SEGT__MASK = 0xf0000000
A6XX_CX_DBGC_CFG_DBGBUS_CNTLT_SEGT__SHIFT = 28
REG_A6XX_CX_DBGC_CFG_DBGBUS_CNTLM = 0x00000005
A6XX_CX_DBGC_CFG_DBGBUS_CNTLM_ENABLE__MASK = 0x0f000000
A6XX_CX_DBGC_CFG_DBGBUS_CNTLM_ENABLE__SHIFT = 24
REG_A6XX_CX_DBGC_CFG_DBGBUS_IVTL_0 = 0x00000008
REG_A6XX_CX_DBGC_CFG_DBGBUS_IVTL_1 = 0x00000009
REG_A6XX_CX_DBGC_CFG_DBGBUS_IVTL_2 = 0x0000000a
REG_A6XX_CX_DBGC_CFG_DBGBUS_IVTL_3 = 0x0000000b
REG_A6XX_CX_DBGC_CFG_DBGBUS_MASKL_0 = 0x0000000c
REG_A6XX_CX_DBGC_CFG_DBGBUS_MASKL_1 = 0x0000000d
REG_A6XX_CX_DBGC_CFG_DBGBUS_MASKL_2 = 0x0000000e
REG_A6XX_CX_DBGC_CFG_DBGBUS_MASKL_3 = 0x0000000f
REG_A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0 = 0x00000010
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL0__MASK = 0x0000000f
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL0__SHIFT = 0
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL1__MASK = 0x000000f0
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL1__SHIFT = 4
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL2__MASK = 0x00000f00
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL2__SHIFT = 8
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL3__MASK = 0x0000f000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL3__SHIFT = 12
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL4__MASK = 0x000f0000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL4__SHIFT = 16
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL5__MASK = 0x00f00000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL5__SHIFT = 20
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL6__MASK = 0x0f000000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL6__SHIFT = 24
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL7__MASK = 0xf0000000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_0_BYTEL7__SHIFT = 28
REG_A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1 = 0x00000011
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL8__MASK = 0x0000000f
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL8__SHIFT = 0
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL9__MASK = 0x000000f0
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL9__SHIFT = 4
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL10__MASK = 0x00000f00
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL10__SHIFT = 8
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL11__MASK = 0x0000f000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL11__SHIFT = 12
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL12__MASK = 0x000f0000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL12__SHIFT = 16
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL13__MASK = 0x00f00000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL13__SHIFT = 20
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL14__MASK = 0x0f000000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL14__SHIFT = 24
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL15__MASK = 0xf0000000
A6XX_CX_DBGC_CFG_DBGBUS_BYTEL_1_BYTEL15__SHIFT = 28
REG_A6XX_CX_DBGC_CFG_DBGBUS_TRACE_BUF1 = 0x0000002f
REG_A6XX_CX_DBGC_CFG_DBGBUS_TRACE_BUF2 = 0x00000030
REG_A6XX_CX_MISC_SYSTEM_CACHE_CNTL_0 = 0x00000001
REG_A6XX_CX_MISC_SYSTEM_CACHE_CNTL_1 = 0x00000002
REG_A7XX_CX_MISC_TCM_RET_CNTL = 0x00000039
REG_A7XX_CX_MISC_SW_FUSE_VALUE = 0x00000400
A7XX_CX_MISC_SW_FUSE_VALUE_FASTBLEND = 0x00000001
A7XX_CX_MISC_SW_FUSE_VALUE_LPAC = 0x00000002
A7XX_CX_MISC_SW_FUSE_VALUE_RAYTRACING = 0x00000004
__struct__cast = lambda X: (struct_X)
REG_CP_LOAD_STATE_0 = 0x00000000
CP_LOAD_STATE_0_DST_OFF__MASK = 0x0000ffff
CP_LOAD_STATE_0_DST_OFF__SHIFT = 0
CP_LOAD_STATE_0_STATE_SRC__MASK = 0x00070000
CP_LOAD_STATE_0_STATE_SRC__SHIFT = 16
CP_LOAD_STATE_0_STATE_BLOCK__MASK = 0x00380000
CP_LOAD_STATE_0_STATE_BLOCK__SHIFT = 19
CP_LOAD_STATE_0_NUM_UNIT__MASK = 0xffc00000
CP_LOAD_STATE_0_NUM_UNIT__SHIFT = 22
REG_CP_LOAD_STATE_1 = 0x00000001
CP_LOAD_STATE_1_STATE_TYPE__MASK = 0x00000003
CP_LOAD_STATE_1_STATE_TYPE__SHIFT = 0
CP_LOAD_STATE_1_EXT_SRC_ADDR__MASK = 0xfffffffc
CP_LOAD_STATE_1_EXT_SRC_ADDR__SHIFT = 2
REG_CP_LOAD_STATE4_0 = 0x00000000
CP_LOAD_STATE4_0_DST_OFF__MASK = 0x00003fff
CP_LOAD_STATE4_0_DST_OFF__SHIFT = 0
CP_LOAD_STATE4_0_STATE_SRC__MASK = 0x00030000
CP_LOAD_STATE4_0_STATE_SRC__SHIFT = 16
CP_LOAD_STATE4_0_STATE_BLOCK__MASK = 0x003c0000
CP_LOAD_STATE4_0_STATE_BLOCK__SHIFT = 18
CP_LOAD_STATE4_0_NUM_UNIT__MASK = 0xffc00000
CP_LOAD_STATE4_0_NUM_UNIT__SHIFT = 22
REG_CP_LOAD_STATE4_1 = 0x00000001
CP_LOAD_STATE4_1_STATE_TYPE__MASK = 0x00000003
CP_LOAD_STATE4_1_STATE_TYPE__SHIFT = 0
CP_LOAD_STATE4_1_EXT_SRC_ADDR__MASK = 0xfffffffc
CP_LOAD_STATE4_1_EXT_SRC_ADDR__SHIFT = 2
REG_CP_LOAD_STATE4_2 = 0x00000002
CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI__MASK = 0xffffffff
CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI__SHIFT = 0
REG_CP_LOAD_STATE6_0 = 0x00000000
CP_LOAD_STATE6_0_DST_OFF__MASK = 0x00003fff
CP_LOAD_STATE6_0_DST_OFF__SHIFT = 0
CP_LOAD_STATE6_0_STATE_TYPE__MASK = 0x0000c000
CP_LOAD_STATE6_0_STATE_TYPE__SHIFT = 14
CP_LOAD_STATE6_0_STATE_SRC__MASK = 0x00030000
CP_LOAD_STATE6_0_STATE_SRC__SHIFT = 16
CP_LOAD_STATE6_0_STATE_BLOCK__MASK = 0x003c0000
CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT = 18
CP_LOAD_STATE6_0_NUM_UNIT__MASK = 0xffc00000
CP_LOAD_STATE6_0_NUM_UNIT__SHIFT = 22
REG_CP_LOAD_STATE6_1 = 0x00000001
CP_LOAD_STATE6_1_EXT_SRC_ADDR__MASK = 0xfffffffc
CP_LOAD_STATE6_1_EXT_SRC_ADDR__SHIFT = 2
REG_CP_LOAD_STATE6_2 = 0x00000002
CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI__MASK = 0xffffffff
CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI__SHIFT = 0
REG_CP_LOAD_STATE6_EXT_SRC_ADDR = 0x00000001
REG_CP_DRAW_INDX_0 = 0x00000000
CP_DRAW_INDX_0_VIZ_QUERY__MASK = 0xffffffff
CP_DRAW_INDX_0_VIZ_QUERY__SHIFT = 0
REG_CP_DRAW_INDX_1 = 0x00000001
CP_DRAW_INDX_1_PRIM_TYPE__MASK = 0x0000003f
CP_DRAW_INDX_1_PRIM_TYPE__SHIFT = 0
CP_DRAW_INDX_1_SOURCE_SELECT__MASK = 0x000000c0
CP_DRAW_INDX_1_SOURCE_SELECT__SHIFT = 6
CP_DRAW_INDX_1_VIS_CULL__MASK = 0x00000600
CP_DRAW_INDX_1_VIS_CULL__SHIFT = 9
CP_DRAW_INDX_1_INDEX_SIZE__MASK = 0x00000800
CP_DRAW_INDX_1_INDEX_SIZE__SHIFT = 11
CP_DRAW_INDX_1_NOT_EOP = 0x00001000
CP_DRAW_INDX_1_SMALL_INDEX = 0x00002000
CP_DRAW_INDX_1_PRE_DRAW_INITIATOR_ENABLE = 0x00004000
CP_DRAW_INDX_1_NUM_INSTANCES__MASK = 0xff000000
CP_DRAW_INDX_1_NUM_INSTANCES__SHIFT = 24
REG_CP_DRAW_INDX_2 = 0x00000002
CP_DRAW_INDX_2_NUM_INDICES__MASK = 0xffffffff
CP_DRAW_INDX_2_NUM_INDICES__SHIFT = 0
REG_CP_DRAW_INDX_3 = 0x00000003
CP_DRAW_INDX_3_INDX_BASE__MASK = 0xffffffff
CP_DRAW_INDX_3_INDX_BASE__SHIFT = 0
REG_CP_DRAW_INDX_4 = 0x00000004
CP_DRAW_INDX_4_INDX_SIZE__MASK = 0xffffffff
CP_DRAW_INDX_4_INDX_SIZE__SHIFT = 0
REG_CP_DRAW_INDX_2_0 = 0x00000000
CP_DRAW_INDX_2_0_VIZ_QUERY__MASK = 0xffffffff
CP_DRAW_INDX_2_0_VIZ_QUERY__SHIFT = 0
REG_CP_DRAW_INDX_2_1 = 0x00000001
CP_DRAW_INDX_2_1_PRIM_TYPE__MASK = 0x0000003f
CP_DRAW_INDX_2_1_PRIM_TYPE__SHIFT = 0
CP_DRAW_INDX_2_1_SOURCE_SELECT__MASK = 0x000000c0
CP_DRAW_INDX_2_1_SOURCE_SELECT__SHIFT = 6
CP_DRAW_INDX_2_1_VIS_CULL__MASK = 0x00000600
CP_DRAW_INDX_2_1_VIS_CULL__SHIFT = 9
CP_DRAW_INDX_2_1_INDEX_SIZE__MASK = 0x00000800
CP_DRAW_INDX_2_1_INDEX_SIZE__SHIFT = 11
CP_DRAW_INDX_2_1_NOT_EOP = 0x00001000
CP_DRAW_INDX_2_1_SMALL_INDEX = 0x00002000
CP_DRAW_INDX_2_1_PRE_DRAW_INITIATOR_ENABLE = 0x00004000
CP_DRAW_INDX_2_1_NUM_INSTANCES__MASK = 0xff000000
CP_DRAW_INDX_2_1_NUM_INSTANCES__SHIFT = 24
REG_CP_DRAW_INDX_2_2 = 0x00000002
CP_DRAW_INDX_2_2_NUM_INDICES__MASK = 0xffffffff
CP_DRAW_INDX_2_2_NUM_INDICES__SHIFT = 0
REG_CP_DRAW_INDX_OFFSET_0 = 0x00000000
CP_DRAW_INDX_OFFSET_0_PRIM_TYPE__MASK = 0x0000003f
CP_DRAW_INDX_OFFSET_0_PRIM_TYPE__SHIFT = 0
CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT__MASK = 0x000000c0
CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT__SHIFT = 6
CP_DRAW_INDX_OFFSET_0_VIS_CULL__MASK = 0x00000300
CP_DRAW_INDX_OFFSET_0_VIS_CULL__SHIFT = 8
CP_DRAW_INDX_OFFSET_0_INDEX_SIZE__MASK = 0x00000c00
CP_DRAW_INDX_OFFSET_0_INDEX_SIZE__SHIFT = 10
CP_DRAW_INDX_OFFSET_0_PATCH_TYPE__MASK = 0x00003000
CP_DRAW_INDX_OFFSET_0_PATCH_TYPE__SHIFT = 12
CP_DRAW_INDX_OFFSET_0_GS_ENABLE = 0x00010000
CP_DRAW_INDX_OFFSET_0_TESS_ENABLE = 0x00020000
REG_CP_DRAW_INDX_OFFSET_1 = 0x00000001
CP_DRAW_INDX_OFFSET_1_NUM_INSTANCES__MASK = 0xffffffff
CP_DRAW_INDX_OFFSET_1_NUM_INSTANCES__SHIFT = 0
REG_CP_DRAW_INDX_OFFSET_2 = 0x00000002
CP_DRAW_INDX_OFFSET_2_NUM_INDICES__MASK = 0xffffffff
CP_DRAW_INDX_OFFSET_2_NUM_INDICES__SHIFT = 0
REG_CP_DRAW_INDX_OFFSET_3 = 0x00000003
CP_DRAW_INDX_OFFSET_3_FIRST_INDX__MASK = 0xffffffff
CP_DRAW_INDX_OFFSET_3_FIRST_INDX__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_OFFSET_4 = 0x00000004
A5XX_CP_DRAW_INDX_OFFSET_4_INDX_BASE_LO__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_OFFSET_4_INDX_BASE_LO__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_OFFSET_5 = 0x00000005
A5XX_CP_DRAW_INDX_OFFSET_5_INDX_BASE_HI__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_OFFSET_5_INDX_BASE_HI__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_OFFSET_INDX_BASE = 0x00000004
REG_A5XX_CP_DRAW_INDX_OFFSET_6 = 0x00000006
A5XX_CP_DRAW_INDX_OFFSET_6_MAX_INDICES__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_OFFSET_6_MAX_INDICES__SHIFT = 0
REG_CP_DRAW_INDX_OFFSET_4 = 0x00000004
CP_DRAW_INDX_OFFSET_4_INDX_BASE__MASK = 0xffffffff
CP_DRAW_INDX_OFFSET_4_INDX_BASE__SHIFT = 0
REG_CP_DRAW_INDX_OFFSET_5 = 0x00000005
CP_DRAW_INDX_OFFSET_5_INDX_SIZE__MASK = 0xffffffff
CP_DRAW_INDX_OFFSET_5_INDX_SIZE__SHIFT = 0
REG_A4XX_CP_DRAW_INDIRECT_0 = 0x00000000
A4XX_CP_DRAW_INDIRECT_0_PRIM_TYPE__MASK = 0x0000003f
A4XX_CP_DRAW_INDIRECT_0_PRIM_TYPE__SHIFT = 0
A4XX_CP_DRAW_INDIRECT_0_SOURCE_SELECT__MASK = 0x000000c0
A4XX_CP_DRAW_INDIRECT_0_SOURCE_SELECT__SHIFT = 6
A4XX_CP_DRAW_INDIRECT_0_VIS_CULL__MASK = 0x00000300
A4XX_CP_DRAW_INDIRECT_0_VIS_CULL__SHIFT = 8
A4XX_CP_DRAW_INDIRECT_0_INDEX_SIZE__MASK = 0x00000c00
A4XX_CP_DRAW_INDIRECT_0_INDEX_SIZE__SHIFT = 10
A4XX_CP_DRAW_INDIRECT_0_PATCH_TYPE__MASK = 0x00003000
A4XX_CP_DRAW_INDIRECT_0_PATCH_TYPE__SHIFT = 12
A4XX_CP_DRAW_INDIRECT_0_GS_ENABLE = 0x00010000
A4XX_CP_DRAW_INDIRECT_0_TESS_ENABLE = 0x00020000
REG_A4XX_CP_DRAW_INDIRECT_1 = 0x00000001
A4XX_CP_DRAW_INDIRECT_1_INDIRECT__MASK = 0xffffffff
A4XX_CP_DRAW_INDIRECT_1_INDIRECT__SHIFT = 0
REG_A5XX_CP_DRAW_INDIRECT_1 = 0x00000001
A5XX_CP_DRAW_INDIRECT_1_INDIRECT_LO__MASK = 0xffffffff
A5XX_CP_DRAW_INDIRECT_1_INDIRECT_LO__SHIFT = 0
REG_A5XX_CP_DRAW_INDIRECT_2 = 0x00000002
A5XX_CP_DRAW_INDIRECT_2_INDIRECT_HI__MASK = 0xffffffff
A5XX_CP_DRAW_INDIRECT_2_INDIRECT_HI__SHIFT = 0
REG_A5XX_CP_DRAW_INDIRECT_INDIRECT = 0x00000001
REG_A4XX_CP_DRAW_INDX_INDIRECT_0 = 0x00000000
A4XX_CP_DRAW_INDX_INDIRECT_0_PRIM_TYPE__MASK = 0x0000003f
A4XX_CP_DRAW_INDX_INDIRECT_0_PRIM_TYPE__SHIFT = 0
A4XX_CP_DRAW_INDX_INDIRECT_0_SOURCE_SELECT__MASK = 0x000000c0
A4XX_CP_DRAW_INDX_INDIRECT_0_SOURCE_SELECT__SHIFT = 6
A4XX_CP_DRAW_INDX_INDIRECT_0_VIS_CULL__MASK = 0x00000300
A4XX_CP_DRAW_INDX_INDIRECT_0_VIS_CULL__SHIFT = 8
A4XX_CP_DRAW_INDX_INDIRECT_0_INDEX_SIZE__MASK = 0x00000c00
A4XX_CP_DRAW_INDX_INDIRECT_0_INDEX_SIZE__SHIFT = 10
A4XX_CP_DRAW_INDX_INDIRECT_0_PATCH_TYPE__MASK = 0x00003000
A4XX_CP_DRAW_INDX_INDIRECT_0_PATCH_TYPE__SHIFT = 12
A4XX_CP_DRAW_INDX_INDIRECT_0_GS_ENABLE = 0x00010000
A4XX_CP_DRAW_INDX_INDIRECT_0_TESS_ENABLE = 0x00020000
REG_A4XX_CP_DRAW_INDX_INDIRECT_1 = 0x00000001
A4XX_CP_DRAW_INDX_INDIRECT_1_INDX_BASE__MASK = 0xffffffff
A4XX_CP_DRAW_INDX_INDIRECT_1_INDX_BASE__SHIFT = 0
REG_A4XX_CP_DRAW_INDX_INDIRECT_2 = 0x00000002
A4XX_CP_DRAW_INDX_INDIRECT_2_INDX_SIZE__MASK = 0xffffffff
A4XX_CP_DRAW_INDX_INDIRECT_2_INDX_SIZE__SHIFT = 0
REG_A4XX_CP_DRAW_INDX_INDIRECT_3 = 0x00000003
A4XX_CP_DRAW_INDX_INDIRECT_3_INDIRECT__MASK = 0xffffffff
A4XX_CP_DRAW_INDX_INDIRECT_3_INDIRECT__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_1 = 0x00000001
A5XX_CP_DRAW_INDX_INDIRECT_1_INDX_BASE_LO__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_INDIRECT_1_INDX_BASE_LO__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_2 = 0x00000002
A5XX_CP_DRAW_INDX_INDIRECT_2_INDX_BASE_HI__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_INDIRECT_2_INDX_BASE_HI__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_INDX_BASE = 0x00000001
REG_A5XX_CP_DRAW_INDX_INDIRECT_3 = 0x00000003
A5XX_CP_DRAW_INDX_INDIRECT_3_MAX_INDICES__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_INDIRECT_3_MAX_INDICES__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_4 = 0x00000004
A5XX_CP_DRAW_INDX_INDIRECT_4_INDIRECT_LO__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_INDIRECT_4_INDIRECT_LO__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_5 = 0x00000005
A5XX_CP_DRAW_INDX_INDIRECT_5_INDIRECT_HI__MASK = 0xffffffff
A5XX_CP_DRAW_INDX_INDIRECT_5_INDIRECT_HI__SHIFT = 0
REG_A5XX_CP_DRAW_INDX_INDIRECT_INDIRECT = 0x00000004
REG_A6XX_CP_DRAW_INDIRECT_MULTI_0 = 0x00000000
A6XX_CP_DRAW_INDIRECT_MULTI_0_PRIM_TYPE__MASK = 0x0000003f
A6XX_CP_DRAW_INDIRECT_MULTI_0_PRIM_TYPE__SHIFT = 0
A6XX_CP_DRAW_INDIRECT_MULTI_0_SOURCE_SELECT__MASK = 0x000000c0
A6XX_CP_DRAW_INDIRECT_MULTI_0_SOURCE_SELECT__SHIFT = 6
A6XX_CP_DRAW_INDIRECT_MULTI_0_VIS_CULL__MASK = 0x00000300
A6XX_CP_DRAW_INDIRECT_MULTI_0_VIS_CULL__SHIFT = 8
A6XX_CP_DRAW_INDIRECT_MULTI_0_INDEX_SIZE__MASK = 0x00000c00
A6XX_CP_DRAW_INDIRECT_MULTI_0_INDEX_SIZE__SHIFT = 10
A6XX_CP_DRAW_INDIRECT_MULTI_0_PATCH_TYPE__MASK = 0x00003000
A6XX_CP_DRAW_INDIRECT_MULTI_0_PATCH_TYPE__SHIFT = 12
A6XX_CP_DRAW_INDIRECT_MULTI_0_GS_ENABLE = 0x00010000
A6XX_CP_DRAW_INDIRECT_MULTI_0_TESS_ENABLE = 0x00020000
REG_A6XX_CP_DRAW_INDIRECT_MULTI_1 = 0x00000001
A6XX_CP_DRAW_INDIRECT_MULTI_1_OPCODE__MASK = 0x0000000f
A6XX_CP_DRAW_INDIRECT_MULTI_1_OPCODE__SHIFT = 0
A6XX_CP_DRAW_INDIRECT_MULTI_1_DST_OFF__MASK = 0x003fff00
A6XX_CP_DRAW_INDIRECT_MULTI_1_DST_OFF__SHIFT = 8
REG_A6XX_CP_DRAW_INDIRECT_MULTI_DRAW_COUNT = 0x00000002
REG_INDIRECT_OP_NORMAL_CP_DRAW_INDIRECT_MULTI_INDIRECT = 0x00000003
REG_INDIRECT_OP_NORMAL_CP_DRAW_INDIRECT_MULTI_STRIDE = 0x00000005
REG_INDIRECT_OP_INDEXED_CP_DRAW_INDIRECT_MULTI_INDEX = 0x00000003
REG_INDIRECT_OP_INDEXED_CP_DRAW_INDIRECT_MULTI_MAX_INDICES = 0x00000005
REG_INDIRECT_OP_INDEXED_CP_DRAW_INDIRECT_MULTI_INDIRECT = 0x00000006
REG_INDIRECT_OP_INDEXED_CP_DRAW_INDIRECT_MULTI_STRIDE = 0x00000008
REG_INDIRECT_OP_INDIRECT_COUNT_CP_DRAW_INDIRECT_MULTI_INDIRECT = 0x00000003
REG_INDIRECT_OP_INDIRECT_COUNT_CP_DRAW_INDIRECT_MULTI_INDIRECT_COUNT = 0x00000005
REG_INDIRECT_OP_INDIRECT_COUNT_CP_DRAW_INDIRECT_MULTI_STRIDE = 0x00000007
REG_INDIRECT_OP_INDIRECT_COUNT_INDEXED_CP_DRAW_INDIRECT_MULTI_INDEX = 0x00000003
REG_INDIRECT_OP_INDIRECT_COUNT_INDEXED_CP_DRAW_INDIRECT_MULTI_MAX_INDICES = 0x00000005
REG_INDIRECT_OP_INDIRECT_COUNT_INDEXED_CP_DRAW_INDIRECT_MULTI_INDIRECT = 0x00000006
REG_INDIRECT_OP_INDIRECT_COUNT_INDEXED_CP_DRAW_INDIRECT_MULTI_INDIRECT_COUNT = 0x00000008
REG_INDIRECT_OP_INDIRECT_COUNT_INDEXED_CP_DRAW_INDIRECT_MULTI_STRIDE = 0x0000000a
REG_CP_DRAW_AUTO_0 = 0x00000000
CP_DRAW_AUTO_0_PRIM_TYPE__MASK = 0x0000003f
CP_DRAW_AUTO_0_PRIM_TYPE__SHIFT = 0
CP_DRAW_AUTO_0_SOURCE_SELECT__MASK = 0x000000c0
CP_DRAW_AUTO_0_SOURCE_SELECT__SHIFT = 6
CP_DRAW_AUTO_0_VIS_CULL__MASK = 0x00000300
CP_DRAW_AUTO_0_VIS_CULL__SHIFT = 8
CP_DRAW_AUTO_0_INDEX_SIZE__MASK = 0x00000c00
CP_DRAW_AUTO_0_INDEX_SIZE__SHIFT = 10
CP_DRAW_AUTO_0_PATCH_TYPE__MASK = 0x00003000
CP_DRAW_AUTO_0_PATCH_TYPE__SHIFT = 12
CP_DRAW_AUTO_0_GS_ENABLE = 0x00010000
CP_DRAW_AUTO_0_TESS_ENABLE = 0x00020000
REG_CP_DRAW_AUTO_1 = 0x00000001
CP_DRAW_AUTO_1_NUM_INSTANCES__MASK = 0xffffffff
CP_DRAW_AUTO_1_NUM_INSTANCES__SHIFT = 0
REG_CP_DRAW_AUTO_NUM_VERTICES_BASE = 0x00000002
REG_CP_DRAW_AUTO_4 = 0x00000004
CP_DRAW_AUTO_4_NUM_VERTICES_OFFSET__MASK = 0xffffffff
CP_DRAW_AUTO_4_NUM_VERTICES_OFFSET__SHIFT = 0
REG_CP_DRAW_AUTO_5 = 0x00000005
CP_DRAW_AUTO_5_STRIDE__MASK = 0xffffffff
CP_DRAW_AUTO_5_STRIDE__SHIFT = 0
REG_CP_DRAW_PRED_ENABLE_GLOBAL_0 = 0x00000000
CP_DRAW_PRED_ENABLE_GLOBAL_0_ENABLE = 0x00000001
REG_CP_DRAW_PRED_ENABLE_LOCAL_0 = 0x00000000
CP_DRAW_PRED_ENABLE_LOCAL_0_ENABLE = 0x00000001
REG_CP_DRAW_PRED_SET_0 = 0x00000000
CP_DRAW_PRED_SET_0_SRC__MASK = 0x000000f0
CP_DRAW_PRED_SET_0_SRC__SHIFT = 4
CP_DRAW_PRED_SET_0_TEST__MASK = 0x00000100
CP_DRAW_PRED_SET_0_TEST__SHIFT = 8
REG_CP_DRAW_PRED_SET_MEM_ADDR = 0x00000001
REG_CP_SET_DRAW_STATE_ = lambda i0: (0x00000000 + 0x3*i0 )
CP_SET_DRAW_STATE__0_COUNT__MASK = 0x0000ffff
CP_SET_DRAW_STATE__0_COUNT__SHIFT = 0
CP_SET_DRAW_STATE__0_DIRTY = 0x00010000
CP_SET_DRAW_STATE__0_DISABLE = 0x00020000
CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS = 0x00040000
CP_SET_DRAW_STATE__0_LOAD_IMMED = 0x00080000
CP_SET_DRAW_STATE__0_BINNING = 0x00100000
CP_SET_DRAW_STATE__0_GMEM = 0x00200000
CP_SET_DRAW_STATE__0_SYSMEM = 0x00400000
CP_SET_DRAW_STATE__0_GROUP_ID__MASK = 0x1f000000
CP_SET_DRAW_STATE__0_GROUP_ID__SHIFT = 24
CP_SET_DRAW_STATE__1_ADDR_LO__MASK = 0xffffffff
CP_SET_DRAW_STATE__1_ADDR_LO__SHIFT = 0
CP_SET_DRAW_STATE__2_ADDR_HI__MASK = 0xffffffff
CP_SET_DRAW_STATE__2_ADDR_HI__SHIFT = 0
REG_CP_SET_BIN_0 = 0x00000000
REG_CP_SET_BIN_1 = 0x00000001
CP_SET_BIN_1_X1__MASK = 0x0000ffff
CP_SET_BIN_1_X1__SHIFT = 0
CP_SET_BIN_1_Y1__MASK = 0xffff0000
CP_SET_BIN_1_Y1__SHIFT = 16
REG_CP_SET_BIN_2 = 0x00000002
CP_SET_BIN_2_X2__MASK = 0x0000ffff
CP_SET_BIN_2_X2__SHIFT = 0
CP_SET_BIN_2_Y2__MASK = 0xffff0000
CP_SET_BIN_2_Y2__SHIFT = 16
REG_CP_SET_BIN_DATA_0 = 0x00000000
CP_SET_BIN_DATA_0_BIN_DATA_ADDR__MASK = 0xffffffff
CP_SET_BIN_DATA_0_BIN_DATA_ADDR__SHIFT = 0
REG_CP_SET_BIN_DATA_1 = 0x00000001
CP_SET_BIN_DATA_1_BIN_SIZE_ADDRESS__MASK = 0xffffffff
CP_SET_BIN_DATA_1_BIN_SIZE_ADDRESS__SHIFT = 0
REG_CP_SET_BIN_DATA5_0 = 0x00000000
CP_SET_BIN_DATA5_0_VSC_MASK__MASK = 0x0000ffff
CP_SET_BIN_DATA5_0_VSC_MASK__SHIFT = 0
CP_SET_BIN_DATA5_0_VSC_SIZE__MASK = 0x003f0000
CP_SET_BIN_DATA5_0_VSC_SIZE__SHIFT = 16
CP_SET_BIN_DATA5_0_VSC_N__MASK = 0x07c00000
CP_SET_BIN_DATA5_0_VSC_N__SHIFT = 22
CP_SET_BIN_DATA5_0_ABS_MASK__MASK = 0x10000000
CP_SET_BIN_DATA5_0_ABS_MASK__SHIFT = 28
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_1 = 0x00000001
NO_ABS_MASK_CP_SET_BIN_DATA5_1_BIN_DATA_ADDR_LO__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_1_BIN_DATA_ADDR_LO__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_2 = 0x00000002
NO_ABS_MASK_CP_SET_BIN_DATA5_2_BIN_DATA_ADDR_HI__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_2_BIN_DATA_ADDR_HI__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_3 = 0x00000003
NO_ABS_MASK_CP_SET_BIN_DATA5_3_BIN_SIZE_ADDRESS_LO__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_3_BIN_SIZE_ADDRESS_LO__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_4 = 0x00000004
NO_ABS_MASK_CP_SET_BIN_DATA5_4_BIN_SIZE_ADDRESS_HI__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_4_BIN_SIZE_ADDRESS_HI__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_5 = 0x00000005
NO_ABS_MASK_CP_SET_BIN_DATA5_5_BIN_PRIM_STRM_LO__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_5_BIN_PRIM_STRM_LO__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_6 = 0x00000006
NO_ABS_MASK_CP_SET_BIN_DATA5_6_BIN_PRIM_STRM_HI__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_6_BIN_PRIM_STRM_HI__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_7 = 0x00000007
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_9 = 0x00000009
REG_ABS_MASK_CP_SET_BIN_DATA5_ABS_MASK = 0x00000001
REG_ABS_MASK_CP_SET_BIN_DATA5_2 = 0x00000002
ABS_MASK_CP_SET_BIN_DATA5_2_BIN_DATA_ADDR_LO__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_2_BIN_DATA_ADDR_LO__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_3 = 0x00000003
ABS_MASK_CP_SET_BIN_DATA5_3_BIN_DATA_ADDR_HI__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_3_BIN_DATA_ADDR_HI__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_4 = 0x00000004
ABS_MASK_CP_SET_BIN_DATA5_4_BIN_SIZE_ADDRESS_LO__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_4_BIN_SIZE_ADDRESS_LO__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_5 = 0x00000005
ABS_MASK_CP_SET_BIN_DATA5_5_BIN_SIZE_ADDRESS_HI__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_5_BIN_SIZE_ADDRESS_HI__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_6 = 0x00000006
ABS_MASK_CP_SET_BIN_DATA5_6_BIN_PRIM_STRM_LO__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_6_BIN_PRIM_STRM_LO__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_7 = 0x00000007
ABS_MASK_CP_SET_BIN_DATA5_7_BIN_PRIM_STRM_HI__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_7_BIN_PRIM_STRM_HI__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_8 = 0x00000008
REG_ABS_MASK_CP_SET_BIN_DATA5_10 = 0x0000000a
REG_CP_SET_BIN_DATA5_OFFSET_0 = 0x00000000
CP_SET_BIN_DATA5_OFFSET_0_VSC_MASK__MASK = 0x0000ffff
CP_SET_BIN_DATA5_OFFSET_0_VSC_MASK__SHIFT = 0
CP_SET_BIN_DATA5_OFFSET_0_VSC_SIZE__MASK = 0x003f0000
CP_SET_BIN_DATA5_OFFSET_0_VSC_SIZE__SHIFT = 16
CP_SET_BIN_DATA5_OFFSET_0_VSC_N__MASK = 0x07c00000
CP_SET_BIN_DATA5_OFFSET_0_VSC_N__SHIFT = 22
CP_SET_BIN_DATA5_OFFSET_0_ABS_MASK__MASK = 0x10000000
CP_SET_BIN_DATA5_OFFSET_0_ABS_MASK__SHIFT = 28
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_1 = 0x00000001
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_1_BIN_DATA_OFFSET__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_1_BIN_DATA_OFFSET__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2 = 0x00000002
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2_BIN_SIZE_OFFSET__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2_BIN_SIZE_OFFSET__SHIFT = 0
REG_NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3 = 0x00000003
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3_BIN_DATA2_OFFSET__MASK = 0xffffffff
NO_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3_BIN_DATA2_OFFSET__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_ABS_MASK = 0x00000001
REG_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2 = 0x00000002
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2_BIN_DATA_OFFSET__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_2_BIN_DATA_OFFSET__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3 = 0x00000003
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3_BIN_SIZE_OFFSET__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_3_BIN_SIZE_OFFSET__SHIFT = 0
REG_ABS_MASK_CP_SET_BIN_DATA5_OFFSET_4 = 0x00000004
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_4_BIN_DATA2_OFFSET__MASK = 0xffffffff
ABS_MASK_CP_SET_BIN_DATA5_OFFSET_4_BIN_DATA2_OFFSET__SHIFT = 0
REG_CP_REG_RMW_0 = 0x00000000
CP_REG_RMW_0_DST_REG__MASK = 0x0003ffff
CP_REG_RMW_0_DST_REG__SHIFT = 0
CP_REG_RMW_0_DST_SCRATCH = 0x00080000
CP_REG_RMW_0_SKIP_WAIT_FOR_ME = 0x00800000
CP_REG_RMW_0_ROTATE__MASK = 0x1f000000
CP_REG_RMW_0_ROTATE__SHIFT = 24
CP_REG_RMW_0_SRC1_ADD = 0x20000000
CP_REG_RMW_0_SRC1_IS_REG = 0x40000000
CP_REG_RMW_0_SRC0_IS_REG = 0x80000000
REG_CP_REG_RMW_1 = 0x00000001
CP_REG_RMW_1_SRC0__MASK = 0xffffffff
CP_REG_RMW_1_SRC0__SHIFT = 0
REG_CP_REG_RMW_2 = 0x00000002
CP_REG_RMW_2_SRC1__MASK = 0xffffffff
CP_REG_RMW_2_SRC1__SHIFT = 0
REG_CP_REG_TO_MEM_0 = 0x00000000
CP_REG_TO_MEM_0_REG__MASK = 0x0003ffff
CP_REG_TO_MEM_0_REG__SHIFT = 0
CP_REG_TO_MEM_0_CNT__MASK = 0x3ffc0000
CP_REG_TO_MEM_0_CNT__SHIFT = 18
CP_REG_TO_MEM_0_64B = 0x40000000
CP_REG_TO_MEM_0_ACCUMULATE = 0x80000000
REG_CP_REG_TO_MEM_1 = 0x00000001
CP_REG_TO_MEM_1_DEST__MASK = 0xffffffff
CP_REG_TO_MEM_1_DEST__SHIFT = 0
REG_CP_REG_TO_MEM_2 = 0x00000002
CP_REG_TO_MEM_2_DEST_HI__MASK = 0xffffffff
CP_REG_TO_MEM_2_DEST_HI__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_REG_0 = 0x00000000
CP_REG_TO_MEM_OFFSET_REG_0_REG__MASK = 0x0003ffff
CP_REG_TO_MEM_OFFSET_REG_0_REG__SHIFT = 0
CP_REG_TO_MEM_OFFSET_REG_0_CNT__MASK = 0x3ffc0000
CP_REG_TO_MEM_OFFSET_REG_0_CNT__SHIFT = 18
CP_REG_TO_MEM_OFFSET_REG_0_64B = 0x40000000
CP_REG_TO_MEM_OFFSET_REG_0_ACCUMULATE = 0x80000000
REG_CP_REG_TO_MEM_OFFSET_REG_1 = 0x00000001
CP_REG_TO_MEM_OFFSET_REG_1_DEST__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_REG_1_DEST__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_REG_2 = 0x00000002
CP_REG_TO_MEM_OFFSET_REG_2_DEST_HI__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_REG_2_DEST_HI__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_REG_3 = 0x00000003
CP_REG_TO_MEM_OFFSET_REG_3_OFFSET0__MASK = 0x0003ffff
CP_REG_TO_MEM_OFFSET_REG_3_OFFSET0__SHIFT = 0
CP_REG_TO_MEM_OFFSET_REG_3_OFFSET0_SCRATCH = 0x00080000
REG_CP_REG_TO_MEM_OFFSET_MEM_0 = 0x00000000
CP_REG_TO_MEM_OFFSET_MEM_0_REG__MASK = 0x0003ffff
CP_REG_TO_MEM_OFFSET_MEM_0_REG__SHIFT = 0
CP_REG_TO_MEM_OFFSET_MEM_0_CNT__MASK = 0x3ffc0000
CP_REG_TO_MEM_OFFSET_MEM_0_CNT__SHIFT = 18
CP_REG_TO_MEM_OFFSET_MEM_0_64B = 0x40000000
CP_REG_TO_MEM_OFFSET_MEM_0_ACCUMULATE = 0x80000000
REG_CP_REG_TO_MEM_OFFSET_MEM_1 = 0x00000001
CP_REG_TO_MEM_OFFSET_MEM_1_DEST__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_MEM_1_DEST__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_MEM_2 = 0x00000002
CP_REG_TO_MEM_OFFSET_MEM_2_DEST_HI__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_MEM_2_DEST_HI__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_MEM_3 = 0x00000003
CP_REG_TO_MEM_OFFSET_MEM_3_OFFSET_LO__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_MEM_3_OFFSET_LO__SHIFT = 0
REG_CP_REG_TO_MEM_OFFSET_MEM_4 = 0x00000004
CP_REG_TO_MEM_OFFSET_MEM_4_OFFSET_HI__MASK = 0xffffffff
CP_REG_TO_MEM_OFFSET_MEM_4_OFFSET_HI__SHIFT = 0
REG_CP_MEM_TO_REG_0 = 0x00000000
CP_MEM_TO_REG_0_REG__MASK = 0x0003ffff
CP_MEM_TO_REG_0_REG__SHIFT = 0
CP_MEM_TO_REG_0_CNT__MASK = 0x3ff80000
CP_MEM_TO_REG_0_CNT__SHIFT = 19
CP_MEM_TO_REG_0_SHIFT_BY_2 = 0x40000000
CP_MEM_TO_REG_0_UNK31 = 0x80000000
REG_CP_MEM_TO_REG_1 = 0x00000001
CP_MEM_TO_REG_1_SRC__MASK = 0xffffffff
CP_MEM_TO_REG_1_SRC__SHIFT = 0
REG_CP_MEM_TO_REG_2 = 0x00000002
CP_MEM_TO_REG_2_SRC_HI__MASK = 0xffffffff
CP_MEM_TO_REG_2_SRC_HI__SHIFT = 0
REG_CP_MEM_TO_MEM_0 = 0x00000000
CP_MEM_TO_MEM_0_NEG_A = 0x00000001
CP_MEM_TO_MEM_0_NEG_B = 0x00000002
CP_MEM_TO_MEM_0_NEG_C = 0x00000004
CP_MEM_TO_MEM_0_DOUBLE = 0x20000000
CP_MEM_TO_MEM_0_WAIT_FOR_MEM_WRITES = 0x40000000
CP_MEM_TO_MEM_0_UNK31 = 0x80000000
REG_CP_MEMCPY_0 = 0x00000000
CP_MEMCPY_0_DWORDS__MASK = 0xffffffff
CP_MEMCPY_0_DWORDS__SHIFT = 0
REG_CP_MEMCPY_1 = 0x00000001
CP_MEMCPY_1_SRC_LO__MASK = 0xffffffff
CP_MEMCPY_1_SRC_LO__SHIFT = 0
REG_CP_MEMCPY_2 = 0x00000002
CP_MEMCPY_2_SRC_HI__MASK = 0xffffffff
CP_MEMCPY_2_SRC_HI__SHIFT = 0
REG_CP_MEMCPY_3 = 0x00000003
CP_MEMCPY_3_DST_LO__MASK = 0xffffffff
CP_MEMCPY_3_DST_LO__SHIFT = 0
REG_CP_MEMCPY_4 = 0x00000004
CP_MEMCPY_4_DST_HI__MASK = 0xffffffff
CP_MEMCPY_4_DST_HI__SHIFT = 0
REG_CP_REG_TO_SCRATCH_0 = 0x00000000
CP_REG_TO_SCRATCH_0_REG__MASK = 0x0003ffff
CP_REG_TO_SCRATCH_0_REG__SHIFT = 0
CP_REG_TO_SCRATCH_0_SCRATCH__MASK = 0x00700000
CP_REG_TO_SCRATCH_0_SCRATCH__SHIFT = 20
CP_REG_TO_SCRATCH_0_CNT__MASK = 0x07000000
CP_REG_TO_SCRATCH_0_CNT__SHIFT = 24
CP_REG_TO_SCRATCH_0_SKIP_WAIT_FOR_ME = 0x08000000
REG_CP_SCRATCH_TO_REG_0 = 0x00000000
CP_SCRATCH_TO_REG_0_REG__MASK = 0x0003ffff
CP_SCRATCH_TO_REG_0_REG__SHIFT = 0
CP_SCRATCH_TO_REG_0_UNK18 = 0x00040000
CP_SCRATCH_TO_REG_0_SCRATCH__MASK = 0x00700000
CP_SCRATCH_TO_REG_0_SCRATCH__SHIFT = 20
CP_SCRATCH_TO_REG_0_CNT__MASK = 0x07000000
CP_SCRATCH_TO_REG_0_CNT__SHIFT = 24
REG_CP_SCRATCH_WRITE_0 = 0x00000000
CP_SCRATCH_WRITE_0_SCRATCH__MASK = 0x00700000
CP_SCRATCH_WRITE_0_SCRATCH__SHIFT = 20
REG_CP_MEM_WRITE_0 = 0x00000000
CP_MEM_WRITE_0_ADDR_LO__MASK = 0xffffffff
CP_MEM_WRITE_0_ADDR_LO__SHIFT = 0
REG_CP_MEM_WRITE_1 = 0x00000001
CP_MEM_WRITE_1_ADDR_HI__MASK = 0xffffffff
CP_MEM_WRITE_1_ADDR_HI__SHIFT = 0
REG_CP_COND_WRITE_0 = 0x00000000
CP_COND_WRITE_0_FUNCTION__MASK = 0x00000007
CP_COND_WRITE_0_FUNCTION__SHIFT = 0
CP_COND_WRITE_0_POLL_MEMORY = 0x00000010
CP_COND_WRITE_0_WRITE_MEMORY = 0x00000100
REG_CP_COND_WRITE_1 = 0x00000001
CP_COND_WRITE_1_POLL_ADDR__MASK = 0xffffffff
CP_COND_WRITE_1_POLL_ADDR__SHIFT = 0
REG_CP_COND_WRITE_2 = 0x00000002
CP_COND_WRITE_2_REF__MASK = 0xffffffff
CP_COND_WRITE_2_REF__SHIFT = 0
REG_CP_COND_WRITE_3 = 0x00000003
CP_COND_WRITE_3_MASK__MASK = 0xffffffff
CP_COND_WRITE_3_MASK__SHIFT = 0
REG_CP_COND_WRITE_4 = 0x00000004
CP_COND_WRITE_4_WRITE_ADDR__MASK = 0xffffffff
CP_COND_WRITE_4_WRITE_ADDR__SHIFT = 0
REG_CP_COND_WRITE_5 = 0x00000005
CP_COND_WRITE_5_WRITE_DATA__MASK = 0xffffffff
CP_COND_WRITE_5_WRITE_DATA__SHIFT = 0
REG_CP_COND_WRITE5_0 = 0x00000000
CP_COND_WRITE5_0_FUNCTION__MASK = 0x00000007
CP_COND_WRITE5_0_FUNCTION__SHIFT = 0
CP_COND_WRITE5_0_SIGNED_COMPARE = 0x00000008
CP_COND_WRITE5_0_POLL__MASK = 0x00000030
CP_COND_WRITE5_0_POLL__SHIFT = 4
CP_COND_WRITE5_0_WRITE_MEMORY = 0x00000100
REG_CP_COND_WRITE5_1 = 0x00000001
CP_COND_WRITE5_1_POLL_ADDR_LO__MASK = 0xffffffff
CP_COND_WRITE5_1_POLL_ADDR_LO__SHIFT = 0
REG_CP_COND_WRITE5_2 = 0x00000002
CP_COND_WRITE5_2_POLL_ADDR_HI__MASK = 0xffffffff
CP_COND_WRITE5_2_POLL_ADDR_HI__SHIFT = 0
REG_CP_COND_WRITE5_3 = 0x00000003
CP_COND_WRITE5_3_REF__MASK = 0xffffffff
CP_COND_WRITE5_3_REF__SHIFT = 0
REG_CP_COND_WRITE5_4 = 0x00000004
CP_COND_WRITE5_4_MASK__MASK = 0xffffffff
CP_COND_WRITE5_4_MASK__SHIFT = 0
REG_CP_COND_WRITE5_5 = 0x00000005
CP_COND_WRITE5_5_WRITE_ADDR_LO__MASK = 0xffffffff
CP_COND_WRITE5_5_WRITE_ADDR_LO__SHIFT = 0
REG_CP_COND_WRITE5_6 = 0x00000006
CP_COND_WRITE5_6_WRITE_ADDR_HI__MASK = 0xffffffff
CP_COND_WRITE5_6_WRITE_ADDR_HI__SHIFT = 0
REG_CP_COND_WRITE5_7 = 0x00000007
CP_COND_WRITE5_7_WRITE_DATA__MASK = 0xffffffff
CP_COND_WRITE5_7_WRITE_DATA__SHIFT = 0
REG_CP_WAIT_MEM_GTE_0 = 0x00000000
CP_WAIT_MEM_GTE_0_RESERVED__MASK = 0xffffffff
CP_WAIT_MEM_GTE_0_RESERVED__SHIFT = 0
REG_CP_WAIT_MEM_GTE_1 = 0x00000001
CP_WAIT_MEM_GTE_1_POLL_ADDR_LO__MASK = 0xffffffff
CP_WAIT_MEM_GTE_1_POLL_ADDR_LO__SHIFT = 0
REG_CP_WAIT_MEM_GTE_2 = 0x00000002
CP_WAIT_MEM_GTE_2_POLL_ADDR_HI__MASK = 0xffffffff
CP_WAIT_MEM_GTE_2_POLL_ADDR_HI__SHIFT = 0
REG_CP_WAIT_MEM_GTE_3 = 0x00000003
CP_WAIT_MEM_GTE_3_REF__MASK = 0xffffffff
CP_WAIT_MEM_GTE_3_REF__SHIFT = 0
REG_CP_WAIT_REG_MEM_0 = 0x00000000
CP_WAIT_REG_MEM_0_FUNCTION__MASK = 0x00000007
CP_WAIT_REG_MEM_0_FUNCTION__SHIFT = 0
CP_WAIT_REG_MEM_0_SIGNED_COMPARE = 0x00000008
CP_WAIT_REG_MEM_0_POLL__MASK = 0x00000030
CP_WAIT_REG_MEM_0_POLL__SHIFT = 4
CP_WAIT_REG_MEM_0_WRITE_MEMORY = 0x00000100
REG_CP_WAIT_REG_MEM_1 = 0x00000001
CP_WAIT_REG_MEM_1_POLL_ADDR_LO__MASK = 0xffffffff
CP_WAIT_REG_MEM_1_POLL_ADDR_LO__SHIFT = 0
REG_CP_WAIT_REG_MEM_2 = 0x00000002
CP_WAIT_REG_MEM_2_POLL_ADDR_HI__MASK = 0xffffffff
CP_WAIT_REG_MEM_2_POLL_ADDR_HI__SHIFT = 0
REG_CP_WAIT_REG_MEM_3 = 0x00000003
CP_WAIT_REG_MEM_3_REF__MASK = 0xffffffff
CP_WAIT_REG_MEM_3_REF__SHIFT = 0
REG_CP_WAIT_REG_MEM_4 = 0x00000004
CP_WAIT_REG_MEM_4_MASK__MASK = 0xffffffff
CP_WAIT_REG_MEM_4_MASK__SHIFT = 0
REG_CP_WAIT_REG_MEM_5 = 0x00000005
CP_WAIT_REG_MEM_5_DELAY_LOOP_CYCLES__MASK = 0xffffffff
CP_WAIT_REG_MEM_5_DELAY_LOOP_CYCLES__SHIFT = 0
REG_CP_WAIT_TWO_REGS_0 = 0x00000000
CP_WAIT_TWO_REGS_0_REG0__MASK = 0x0003ffff
CP_WAIT_TWO_REGS_0_REG0__SHIFT = 0
REG_CP_WAIT_TWO_REGS_1 = 0x00000001
CP_WAIT_TWO_REGS_1_REG1__MASK = 0x0003ffff
CP_WAIT_TWO_REGS_1_REG1__SHIFT = 0
REG_CP_WAIT_TWO_REGS_2 = 0x00000002
CP_WAIT_TWO_REGS_2_REF__MASK = 0xffffffff
CP_WAIT_TWO_REGS_2_REF__SHIFT = 0
REG_CP_DISPATCH_COMPUTE_0 = 0x00000000
REG_CP_DISPATCH_COMPUTE_1 = 0x00000001
CP_DISPATCH_COMPUTE_1_X__MASK = 0xffffffff
CP_DISPATCH_COMPUTE_1_X__SHIFT = 0
REG_CP_DISPATCH_COMPUTE_2 = 0x00000002
CP_DISPATCH_COMPUTE_2_Y__MASK = 0xffffffff
CP_DISPATCH_COMPUTE_2_Y__SHIFT = 0
REG_CP_DISPATCH_COMPUTE_3 = 0x00000003
CP_DISPATCH_COMPUTE_3_Z__MASK = 0xffffffff
CP_DISPATCH_COMPUTE_3_Z__SHIFT = 0
REG_CP_SET_RENDER_MODE_0 = 0x00000000
CP_SET_RENDER_MODE_0_MODE__MASK = 0x000001ff
CP_SET_RENDER_MODE_0_MODE__SHIFT = 0
REG_CP_SET_RENDER_MODE_1 = 0x00000001
CP_SET_RENDER_MODE_1_ADDR_0_LO__MASK = 0xffffffff
CP_SET_RENDER_MODE_1_ADDR_0_LO__SHIFT = 0
REG_CP_SET_RENDER_MODE_2 = 0x00000002
CP_SET_RENDER_MODE_2_ADDR_0_HI__MASK = 0xffffffff
CP_SET_RENDER_MODE_2_ADDR_0_HI__SHIFT = 0
REG_CP_SET_RENDER_MODE_3 = 0x00000003
CP_SET_RENDER_MODE_3_VSC_ENABLE = 0x00000008
CP_SET_RENDER_MODE_3_GMEM_ENABLE = 0x00000010
REG_CP_SET_RENDER_MODE_4 = 0x00000004
REG_CP_SET_RENDER_MODE_5 = 0x00000005
CP_SET_RENDER_MODE_5_ADDR_1_LEN__MASK = 0xffffffff
CP_SET_RENDER_MODE_5_ADDR_1_LEN__SHIFT = 0
REG_CP_SET_RENDER_MODE_6 = 0x00000006
CP_SET_RENDER_MODE_6_ADDR_1_LO__MASK = 0xffffffff
CP_SET_RENDER_MODE_6_ADDR_1_LO__SHIFT = 0
REG_CP_SET_RENDER_MODE_7 = 0x00000007
CP_SET_RENDER_MODE_7_ADDR_1_HI__MASK = 0xffffffff
CP_SET_RENDER_MODE_7_ADDR_1_HI__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_0 = 0x00000000
CP_COMPUTE_CHECKPOINT_0_ADDR_0_LO__MASK = 0xffffffff
CP_COMPUTE_CHECKPOINT_0_ADDR_0_LO__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_1 = 0x00000001
CP_COMPUTE_CHECKPOINT_1_ADDR_0_HI__MASK = 0xffffffff
CP_COMPUTE_CHECKPOINT_1_ADDR_0_HI__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_2 = 0x00000002
REG_CP_COMPUTE_CHECKPOINT_3 = 0x00000003
REG_CP_COMPUTE_CHECKPOINT_4 = 0x00000004
CP_COMPUTE_CHECKPOINT_4_ADDR_1_LEN__MASK = 0xffffffff
CP_COMPUTE_CHECKPOINT_4_ADDR_1_LEN__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_5 = 0x00000005
CP_COMPUTE_CHECKPOINT_5_ADDR_1_LO__MASK = 0xffffffff
CP_COMPUTE_CHECKPOINT_5_ADDR_1_LO__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_6 = 0x00000006
CP_COMPUTE_CHECKPOINT_6_ADDR_1_HI__MASK = 0xffffffff
CP_COMPUTE_CHECKPOINT_6_ADDR_1_HI__SHIFT = 0
REG_CP_COMPUTE_CHECKPOINT_7 = 0x00000007
REG_CP_PERFCOUNTER_ACTION_0 = 0x00000000
REG_CP_PERFCOUNTER_ACTION_1 = 0x00000001
CP_PERFCOUNTER_ACTION_1_ADDR_0_LO__MASK = 0xffffffff
CP_PERFCOUNTER_ACTION_1_ADDR_0_LO__SHIFT = 0
REG_CP_PERFCOUNTER_ACTION_2 = 0x00000002
CP_PERFCOUNTER_ACTION_2_ADDR_0_HI__MASK = 0xffffffff
CP_PERFCOUNTER_ACTION_2_ADDR_0_HI__SHIFT = 0
REG_CP_EVENT_WRITE_0 = 0x00000000
CP_EVENT_WRITE_0_EVENT__MASK = 0x000000ff
CP_EVENT_WRITE_0_EVENT__SHIFT = 0
CP_EVENT_WRITE_0_TIMESTAMP = 0x40000000
CP_EVENT_WRITE_0_IRQ = 0x80000000
REG_CP_EVENT_WRITE_1 = 0x00000001
CP_EVENT_WRITE_1_ADDR_0_LO__MASK = 0xffffffff
CP_EVENT_WRITE_1_ADDR_0_LO__SHIFT = 0
REG_CP_EVENT_WRITE_2 = 0x00000002
CP_EVENT_WRITE_2_ADDR_0_HI__MASK = 0xffffffff
CP_EVENT_WRITE_2_ADDR_0_HI__SHIFT = 0
REG_CP_EVENT_WRITE_3 = 0x00000003
REG_CP_EVENT_WRITE7_0 = 0x00000000
CP_EVENT_WRITE7_0_EVENT__MASK = 0x000000ff
CP_EVENT_WRITE7_0_EVENT__SHIFT = 0
CP_EVENT_WRITE7_0_WRITE_SAMPLE_COUNT = 0x00001000
CP_EVENT_WRITE7_0_SAMPLE_COUNT_END_OFFSET = 0x00002000
CP_EVENT_WRITE7_0_WRITE_ACCUM_SAMPLE_COUNT_DIFF = 0x00004000
CP_EVENT_WRITE7_0_INC_BV_COUNT = 0x00010000
CP_EVENT_WRITE7_0_INC_BR_COUNT = 0x00020000
CP_EVENT_WRITE7_0_CLEAR_RENDER_RESOURCE = 0x00040000
CP_EVENT_WRITE7_0_CLEAR_LRZ_RESOURCE = 0x00080000
CP_EVENT_WRITE7_0_WRITE_SRC__MASK = 0x00700000
CP_EVENT_WRITE7_0_WRITE_SRC__SHIFT = 20
CP_EVENT_WRITE7_0_WRITE_DST__MASK = 0x01000000
CP_EVENT_WRITE7_0_WRITE_DST__SHIFT = 24
CP_EVENT_WRITE7_0_WRITE_ENABLED = 0x08000000
CP_EVENT_WRITE7_0_IRQ = 0x80000000
REG_EV_DST_RAM_CP_EVENT_WRITE7_1 = 0x00000001
REG_EV_DST_RAM_CP_EVENT_WRITE7_3 = 0x00000003
EV_DST_RAM_CP_EVENT_WRITE7_3_PAYLOAD_0__MASK = 0xffffffff
EV_DST_RAM_CP_EVENT_WRITE7_3_PAYLOAD_0__SHIFT = 0
REG_EV_DST_RAM_CP_EVENT_WRITE7_4 = 0x00000004
EV_DST_RAM_CP_EVENT_WRITE7_4_PAYLOAD_1__MASK = 0xffffffff
EV_DST_RAM_CP_EVENT_WRITE7_4_PAYLOAD_1__SHIFT = 0
REG_EV_DST_ONCHIP_CP_EVENT_WRITE7_1 = 0x00000001
EV_DST_ONCHIP_CP_EVENT_WRITE7_1_ONCHIP_ADDR_0__MASK = 0xffffffff
EV_DST_ONCHIP_CP_EVENT_WRITE7_1_ONCHIP_ADDR_0__SHIFT = 0
REG_EV_DST_ONCHIP_CP_EVENT_WRITE7_3 = 0x00000003
EV_DST_ONCHIP_CP_EVENT_WRITE7_3_PAYLOAD_0__MASK = 0xffffffff
EV_DST_ONCHIP_CP_EVENT_WRITE7_3_PAYLOAD_0__SHIFT = 0
REG_EV_DST_ONCHIP_CP_EVENT_WRITE7_4 = 0x00000004
EV_DST_ONCHIP_CP_EVENT_WRITE7_4_PAYLOAD_1__MASK = 0xffffffff
EV_DST_ONCHIP_CP_EVENT_WRITE7_4_PAYLOAD_1__SHIFT = 0
REG_CP_BLIT_0 = 0x00000000
CP_BLIT_0_OP__MASK = 0x0000000f
CP_BLIT_0_OP__SHIFT = 0
REG_CP_BLIT_1 = 0x00000001
CP_BLIT_1_SRC_X1__MASK = 0x00003fff
CP_BLIT_1_SRC_X1__SHIFT = 0
CP_BLIT_1_SRC_Y1__MASK = 0x3fff0000
CP_BLIT_1_SRC_Y1__SHIFT = 16
REG_CP_BLIT_2 = 0x00000002
CP_BLIT_2_SRC_X2__MASK = 0x00003fff
CP_BLIT_2_SRC_X2__SHIFT = 0
CP_BLIT_2_SRC_Y2__MASK = 0x3fff0000
CP_BLIT_2_SRC_Y2__SHIFT = 16
REG_CP_BLIT_3 = 0x00000003
CP_BLIT_3_DST_X1__MASK = 0x00003fff
CP_BLIT_3_DST_X1__SHIFT = 0
CP_BLIT_3_DST_Y1__MASK = 0x3fff0000
CP_BLIT_3_DST_Y1__SHIFT = 16
REG_CP_BLIT_4 = 0x00000004
CP_BLIT_4_DST_X2__MASK = 0x00003fff
CP_BLIT_4_DST_X2__SHIFT = 0
CP_BLIT_4_DST_Y2__MASK = 0x3fff0000
CP_BLIT_4_DST_Y2__SHIFT = 16
REG_CP_EXEC_CS_0 = 0x00000000
REG_CP_EXEC_CS_1 = 0x00000001
CP_EXEC_CS_1_NGROUPS_X__MASK = 0xffffffff
CP_EXEC_CS_1_NGROUPS_X__SHIFT = 0
REG_CP_EXEC_CS_2 = 0x00000002
CP_EXEC_CS_2_NGROUPS_Y__MASK = 0xffffffff
CP_EXEC_CS_2_NGROUPS_Y__SHIFT = 0
REG_CP_EXEC_CS_3 = 0x00000003
CP_EXEC_CS_3_NGROUPS_Z__MASK = 0xffffffff
CP_EXEC_CS_3_NGROUPS_Z__SHIFT = 0
REG_A4XX_CP_EXEC_CS_INDIRECT_0 = 0x00000000
REG_A4XX_CP_EXEC_CS_INDIRECT_1 = 0x00000001
A4XX_CP_EXEC_CS_INDIRECT_1_ADDR__MASK = 0xffffffff
A4XX_CP_EXEC_CS_INDIRECT_1_ADDR__SHIFT = 0
REG_A4XX_CP_EXEC_CS_INDIRECT_2 = 0x00000002
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEX__MASK = 0x00000ffc
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEX__SHIFT = 2
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEY__MASK = 0x003ff000
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEY__SHIFT = 12
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEZ__MASK = 0xffc00000
A4XX_CP_EXEC_CS_INDIRECT_2_LOCALSIZEZ__SHIFT = 22
REG_A5XX_CP_EXEC_CS_INDIRECT_1 = 0x00000001
A5XX_CP_EXEC_CS_INDIRECT_1_ADDR_LO__MASK = 0xffffffff
A5XX_CP_EXEC_CS_INDIRECT_1_ADDR_LO__SHIFT = 0
REG_A5XX_CP_EXEC_CS_INDIRECT_2 = 0x00000002
A5XX_CP_EXEC_CS_INDIRECT_2_ADDR_HI__MASK = 0xffffffff
A5XX_CP_EXEC_CS_INDIRECT_2_ADDR_HI__SHIFT = 0
REG_A5XX_CP_EXEC_CS_INDIRECT_3 = 0x00000003
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEX__MASK = 0x00000ffc
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEX__SHIFT = 2
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEY__MASK = 0x003ff000
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEY__SHIFT = 12
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEZ__MASK = 0xffc00000
A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEZ__SHIFT = 22
REG_A6XX_CP_SET_MARKER_0 = 0x00000000
A6XX_CP_SET_MARKER_0_MARKER_MODE__MASK = 0x00000100
A6XX_CP_SET_MARKER_0_MARKER_MODE__SHIFT = 8
A6XX_CP_SET_MARKER_0_MODE__MASK = 0x0000000f
A6XX_CP_SET_MARKER_0_MODE__SHIFT = 0
A6XX_CP_SET_MARKER_0_USES_GMEM = 0x00000010
A6XX_CP_SET_MARKER_0_IFPC_MODE__MASK = 0x00000001
A6XX_CP_SET_MARKER_0_IFPC_MODE__SHIFT = 0
A6XX_CP_SET_MARKER_0_SHADER_USES_RT = 0x00000200
A6XX_CP_SET_MARKER_0_RT_WA_START = 0x00000400
A6XX_CP_SET_MARKER_0_RT_WA_END = 0x00000800
REG_A6XX_CP_SET_PSEUDO_REG_ = lambda i0: (0x00000000 + 0x3*i0 )
A6XX_CP_SET_PSEUDO_REG__0_PSEUDO_REG__MASK = 0x000007ff
A6XX_CP_SET_PSEUDO_REG__0_PSEUDO_REG__SHIFT = 0
A6XX_CP_SET_PSEUDO_REG__1_LO__MASK = 0xffffffff
A6XX_CP_SET_PSEUDO_REG__1_LO__SHIFT = 0
A6XX_CP_SET_PSEUDO_REG__2_HI__MASK = 0xffffffff
A6XX_CP_SET_PSEUDO_REG__2_HI__SHIFT = 0
REG_A6XX_CP_REG_TEST_0 = 0x00000000
A6XX_CP_REG_TEST_0_REG__MASK = 0x0003ffff
A6XX_CP_REG_TEST_0_REG__SHIFT = 0
A6XX_CP_REG_TEST_0_SCRATCH_MEM_OFFSET__MASK = 0x0003ffff
A6XX_CP_REG_TEST_0_SCRATCH_MEM_OFFSET__SHIFT = 0
A6XX_CP_REG_TEST_0_SOURCE__MASK = 0x00040000
A6XX_CP_REG_TEST_0_SOURCE__SHIFT = 18
A6XX_CP_REG_TEST_0_BIT__MASK = 0x01f00000
A6XX_CP_REG_TEST_0_BIT__SHIFT = 20
A6XX_CP_REG_TEST_0_SKIP_WAIT_FOR_ME = 0x02000000
A6XX_CP_REG_TEST_0_PRED_BIT__MASK = 0x7c000000
A6XX_CP_REG_TEST_0_PRED_BIT__SHIFT = 26
A6XX_CP_REG_TEST_0_PRED_UPDATE = 0x80000000
REG_A6XX_CP_REG_TEST_PRED_MASK = 0x00000001
REG_A6XX_CP_REG_TEST_PRED_VAL = 0x00000002
REG_CP_COND_REG_EXEC_0 = 0x00000000
CP_COND_REG_EXEC_0_REG0__MASK = 0x0003ffff
CP_COND_REG_EXEC_0_REG0__SHIFT = 0
CP_COND_REG_EXEC_0_PRED_BIT__MASK = 0x007c0000
CP_COND_REG_EXEC_0_PRED_BIT__SHIFT = 18
CP_COND_REG_EXEC_0_SKIP_WAIT_FOR_ME = 0x00800000
CP_COND_REG_EXEC_0_ONCHIP_MEM = 0x01000000
CP_COND_REG_EXEC_0_BINNING = 0x02000000
CP_COND_REG_EXEC_0_GMEM = 0x04000000
CP_COND_REG_EXEC_0_SYSMEM = 0x08000000
CP_COND_REG_EXEC_0_BV = 0x02000000
CP_COND_REG_EXEC_0_BR = 0x04000000
CP_COND_REG_EXEC_0_LPAC = 0x08000000
CP_COND_REG_EXEC_0_MODE__MASK = 0xf0000000
CP_COND_REG_EXEC_0_MODE__SHIFT = 28
REG_PRED_TEST_CP_COND_REG_EXEC_1 = 0x00000001
PRED_TEST_CP_COND_REG_EXEC_1_DWORDS__MASK = 0x00ffffff
PRED_TEST_CP_COND_REG_EXEC_1_DWORDS__SHIFT = 0
REG_REG_COMPARE_CP_COND_REG_EXEC_1 = 0x00000001
REG_COMPARE_CP_COND_REG_EXEC_1_REG1__MASK = 0x0003ffff
REG_COMPARE_CP_COND_REG_EXEC_1_REG1__SHIFT = 0
REG_COMPARE_CP_COND_REG_EXEC_1_ONCHIP_MEM = 0x01000000
REG_RENDER_MODE_CP_COND_REG_EXEC_1 = 0x00000001
RENDER_MODE_CP_COND_REG_EXEC_1_DWORDS__MASK = 0x00ffffff
RENDER_MODE_CP_COND_REG_EXEC_1_DWORDS__SHIFT = 0
REG_REG_COMPARE_IMM_CP_COND_REG_EXEC_1 = 0x00000001
REG_COMPARE_IMM_CP_COND_REG_EXEC_1_IMM__MASK = 0xffffffff
REG_COMPARE_IMM_CP_COND_REG_EXEC_1_IMM__SHIFT = 0
REG_THREAD_MODE_CP_COND_REG_EXEC_1 = 0x00000001
THREAD_MODE_CP_COND_REG_EXEC_1_DWORDS__MASK = 0x00ffffff
THREAD_MODE_CP_COND_REG_EXEC_1_DWORDS__SHIFT = 0
REG_CP_COND_REG_EXEC_2 = 0x00000002
CP_COND_REG_EXEC_2_DWORDS__MASK = 0x00ffffff
CP_COND_REG_EXEC_2_DWORDS__SHIFT = 0
REG_CP_COND_EXEC_0 = 0x00000000
CP_COND_EXEC_0_ADDR0_LO__MASK = 0xffffffff
CP_COND_EXEC_0_ADDR0_LO__SHIFT = 0
REG_CP_COND_EXEC_1 = 0x00000001
CP_COND_EXEC_1_ADDR0_HI__MASK = 0xffffffff
CP_COND_EXEC_1_ADDR0_HI__SHIFT = 0
REG_CP_COND_EXEC_2 = 0x00000002
CP_COND_EXEC_2_ADDR1_LO__MASK = 0xffffffff
CP_COND_EXEC_2_ADDR1_LO__SHIFT = 0
REG_CP_COND_EXEC_3 = 0x00000003
CP_COND_EXEC_3_ADDR1_HI__MASK = 0xffffffff
CP_COND_EXEC_3_ADDR1_HI__SHIFT = 0
REG_CP_COND_EXEC_4 = 0x00000004
CP_COND_EXEC_4_REF__MASK = 0xffffffff
CP_COND_EXEC_4_REF__SHIFT = 0
REG_CP_COND_EXEC_5 = 0x00000005
CP_COND_EXEC_5_DWORDS__MASK = 0xffffffff
CP_COND_EXEC_5_DWORDS__SHIFT = 0
REG_CP_SET_AMBLE_0 = 0x00000000
CP_SET_AMBLE_0_ADDR_LO__MASK = 0xffffffff
CP_SET_AMBLE_0_ADDR_LO__SHIFT = 0
REG_CP_SET_AMBLE_1 = 0x00000001
CP_SET_AMBLE_1_ADDR_HI__MASK = 0xffffffff
CP_SET_AMBLE_1_ADDR_HI__SHIFT = 0
REG_CP_SET_AMBLE_2 = 0x00000002
CP_SET_AMBLE_2_DWORDS__MASK = 0x000fffff
CP_SET_AMBLE_2_DWORDS__SHIFT = 0
CP_SET_AMBLE_2_TYPE__MASK = 0x00300000
CP_SET_AMBLE_2_TYPE__SHIFT = 20
REG_CP_REG_WRITE_0 = 0x00000000
CP_REG_WRITE_0_TRACKER__MASK = 0x0000000f
CP_REG_WRITE_0_TRACKER__SHIFT = 0
REG_CP_REG_WRITE_1 = 0x00000001
REG_CP_REG_WRITE_2 = 0x00000002
REG_CP_SMMU_TABLE_UPDATE_0 = 0x00000000
CP_SMMU_TABLE_UPDATE_0_TTBR0_LO__MASK = 0xffffffff
CP_SMMU_TABLE_UPDATE_0_TTBR0_LO__SHIFT = 0
REG_CP_SMMU_TABLE_UPDATE_1 = 0x00000001
CP_SMMU_TABLE_UPDATE_1_TTBR0_HI__MASK = 0x0000ffff
CP_SMMU_TABLE_UPDATE_1_TTBR0_HI__SHIFT = 0
CP_SMMU_TABLE_UPDATE_1_ASID__MASK = 0xffff0000
CP_SMMU_TABLE_UPDATE_1_ASID__SHIFT = 16
REG_CP_SMMU_TABLE_UPDATE_2 = 0x00000002
CP_SMMU_TABLE_UPDATE_2_CONTEXTIDR__MASK = 0xffffffff
CP_SMMU_TABLE_UPDATE_2_CONTEXTIDR__SHIFT = 0
REG_CP_SMMU_TABLE_UPDATE_3 = 0x00000003
CP_SMMU_TABLE_UPDATE_3_CONTEXTBANK__MASK = 0xffffffff
CP_SMMU_TABLE_UPDATE_3_CONTEXTBANK__SHIFT = 0
REG_CP_START_BIN_BIN_COUNT = 0x00000000
REG_CP_START_BIN_PREFIX_ADDR = 0x00000001
REG_CP_START_BIN_PREFIX_DWORDS = 0x00000003
REG_CP_START_BIN_BODY_DWORDS = 0x00000004
REG_CP_WAIT_TIMESTAMP_0 = 0x00000000
CP_WAIT_TIMESTAMP_0_WAIT_VALUE_SRC__MASK = 0x00000003
CP_WAIT_TIMESTAMP_0_WAIT_VALUE_SRC__SHIFT = 0
CP_WAIT_TIMESTAMP_0_WAIT_DST__MASK = 0x00000010
CP_WAIT_TIMESTAMP_0_WAIT_DST__SHIFT = 4
REG_TS_WAIT_RAM_CP_WAIT_TIMESTAMP_ADDR = 0x00000001
REG_TS_WAIT_ONCHIP_CP_WAIT_TIMESTAMP_ONCHIP_ADDR_0 = 0x00000001
REG_CP_WAIT_TIMESTAMP_SRC_0 = 0x00000003
REG_CP_WAIT_TIMESTAMP_SRC_1 = 0x00000004
REG_CP_BV_BR_COUNT_OPS_0 = 0x00000000
CP_BV_BR_COUNT_OPS_0_OP__MASK = 0x0000000f
CP_BV_BR_COUNT_OPS_0_OP__SHIFT = 0
REG_CP_BV_BR_COUNT_OPS_1 = 0x00000001
CP_BV_BR_COUNT_OPS_1_BR_OFFSET__MASK = 0x0000ffff
CP_BV_BR_COUNT_OPS_1_BR_OFFSET__SHIFT = 0
REG_CP_MODIFY_TIMESTAMP_0 = 0x00000000
CP_MODIFY_TIMESTAMP_0_ADD__MASK = 0x000000ff
CP_MODIFY_TIMESTAMP_0_ADD__SHIFT = 0
CP_MODIFY_TIMESTAMP_0_OP__MASK = 0xf0000000
CP_MODIFY_TIMESTAMP_0_OP__SHIFT = 28
REG_CP_MEM_TO_SCRATCH_MEM_0 = 0x00000000
CP_MEM_TO_SCRATCH_MEM_0_CNT__MASK = 0x0000003f
CP_MEM_TO_SCRATCH_MEM_0_CNT__SHIFT = 0
REG_CP_MEM_TO_SCRATCH_MEM_1 = 0x00000001
CP_MEM_TO_SCRATCH_MEM_1_OFFSET__MASK = 0x0000003f
CP_MEM_TO_SCRATCH_MEM_1_OFFSET__SHIFT = 0
REG_CP_MEM_TO_SCRATCH_MEM_2 = 0x00000002
CP_MEM_TO_SCRATCH_MEM_2_SRC__MASK = 0xffffffff
CP_MEM_TO_SCRATCH_MEM_2_SRC__SHIFT = 0
REG_CP_MEM_TO_SCRATCH_MEM_3 = 0x00000003
CP_MEM_TO_SCRATCH_MEM_3_SRC_HI__MASK = 0xffffffff
CP_MEM_TO_SCRATCH_MEM_3_SRC_HI__SHIFT = 0
REG_CP_THREAD_CONTROL_0 = 0x00000000
CP_THREAD_CONTROL_0_THREAD__MASK = 0x00000003
CP_THREAD_CONTROL_0_THREAD__SHIFT = 0
CP_THREAD_CONTROL_0_CONCURRENT_BIN_DISABLE = 0x08000000
CP_THREAD_CONTROL_0_SYNC_THREADS = 0x80000000
REG_CP_FIXED_STRIDE_DRAW_TABLE_IB_BASE = 0x00000000
REG_CP_FIXED_STRIDE_DRAW_TABLE_2 = 0x00000002
CP_FIXED_STRIDE_DRAW_TABLE_2_IB_SIZE__MASK = 0x00000fff
CP_FIXED_STRIDE_DRAW_TABLE_2_IB_SIZE__SHIFT = 0
CP_FIXED_STRIDE_DRAW_TABLE_2_STRIDE__MASK = 0xfff00000
CP_FIXED_STRIDE_DRAW_TABLE_2_STRIDE__SHIFT = 20
REG_CP_FIXED_STRIDE_DRAW_TABLE_3 = 0x00000003
CP_FIXED_STRIDE_DRAW_TABLE_3_COUNT__MASK = 0xffffffff
CP_FIXED_STRIDE_DRAW_TABLE_3_COUNT__SHIFT = 0
REG_CP_RESET_CONTEXT_STATE_0 = 0x00000000
CP_RESET_CONTEXT_STATE_0_CLEAR_ON_CHIP_TS = 0x00000001
CP_RESET_CONTEXT_STATE_0_CLEAR_RESOURCE_TABLE = 0x00000002
CP_RESET_CONTEXT_STATE_0_CLEAR_BV_BR_COUNTER = 0x00000004
CP_RESET_CONTEXT_STATE_0_RESET_GLOBAL_LOCAL_TS = 0x00000008
REG_CP_SCOPE_CNTL_0 = 0x00000000
CP_SCOPE_CNTL_0_DISABLE_PREEMPTION = 0x00000001
CP_SCOPE_CNTL_0_SCOPE__MASK = 0xf0000000
CP_SCOPE_CNTL_0_SCOPE__SHIFT = 28
REG_A5XX_CP_INDIRECT_BUFFER_IB_BASE = 0x00000000
REG_A5XX_CP_INDIRECT_BUFFER_2 = 0x00000002
A5XX_CP_INDIRECT_BUFFER_2_IB_SIZE__MASK = 0x000fffff
A5XX_CP_INDIRECT_BUFFER_2_IB_SIZE__SHIFT = 0
__struct__cast = lambda X: (struct_X)
__struct__cast = lambda X: (struct_X)
REG_A6XX_TEX_SAMP_0 = 0x00000000
A6XX_TEX_SAMP_0_MIPFILTER_LINEAR_NEAR = 0x00000001
A6XX_TEX_SAMP_0_XY_MAG__MASK = 0x00000006
A6XX_TEX_SAMP_0_XY_MAG__SHIFT = 1
A6XX_TEX_SAMP_0_XY_MIN__MASK = 0x00000018
A6XX_TEX_SAMP_0_XY_MIN__SHIFT = 3
A6XX_TEX_SAMP_0_WRAP_S__MASK = 0x000000e0
A6XX_TEX_SAMP_0_WRAP_S__SHIFT = 5
A6XX_TEX_SAMP_0_WRAP_T__MASK = 0x00000700
A6XX_TEX_SAMP_0_WRAP_T__SHIFT = 8
A6XX_TEX_SAMP_0_WRAP_R__MASK = 0x00003800
A6XX_TEX_SAMP_0_WRAP_R__SHIFT = 11
A6XX_TEX_SAMP_0_ANISO__MASK = 0x0001c000
A6XX_TEX_SAMP_0_ANISO__SHIFT = 14
A6XX_TEX_SAMP_0_LOD_BIAS__MASK = 0xfff80000
A6XX_TEX_SAMP_0_LOD_BIAS__SHIFT = 19
REG_A6XX_TEX_SAMP_1 = 0x00000001
A6XX_TEX_SAMP_1_CLAMPENABLE = 0x00000001
A6XX_TEX_SAMP_1_COMPARE_FUNC__MASK = 0x0000000e
A6XX_TEX_SAMP_1_COMPARE_FUNC__SHIFT = 1
A6XX_TEX_SAMP_1_CUBEMAPSEAMLESSFILTOFF = 0x00000010
A6XX_TEX_SAMP_1_UNNORM_COORDS = 0x00000020
A6XX_TEX_SAMP_1_MIPFILTER_LINEAR_FAR = 0x00000040
A6XX_TEX_SAMP_1_MAX_LOD__MASK = 0x000fff00
A6XX_TEX_SAMP_1_MAX_LOD__SHIFT = 8
A6XX_TEX_SAMP_1_MIN_LOD__MASK = 0xfff00000
A6XX_TEX_SAMP_1_MIN_LOD__SHIFT = 20
REG_A6XX_TEX_SAMP_2 = 0x00000002
A6XX_TEX_SAMP_2_REDUCTION_MODE__MASK = 0x00000003
A6XX_TEX_SAMP_2_REDUCTION_MODE__SHIFT = 0
A6XX_TEX_SAMP_2_FASTBORDERCOLOR__MASK = 0x0000000c
A6XX_TEX_SAMP_2_FASTBORDERCOLOR__SHIFT = 2
A6XX_TEX_SAMP_2_FASTBORDERCOLOREN = 0x00000010
A6XX_TEX_SAMP_2_CHROMA_LINEAR = 0x00000020
A6XX_TEX_SAMP_2_BCOLOR__MASK = 0xffffff80
A6XX_TEX_SAMP_2_BCOLOR__SHIFT = 7
REG_A6XX_TEX_SAMP_3 = 0x00000003
REG_A6XX_TEX_CONST_0 = 0x00000000
A6XX_TEX_CONST_0_TILE_MODE__MASK = 0x00000003
A6XX_TEX_CONST_0_TILE_MODE__SHIFT = 0
A6XX_TEX_CONST_0_SRGB = 0x00000004
A6XX_TEX_CONST_0_SWIZ_X__MASK = 0x00000070
A6XX_TEX_CONST_0_SWIZ_X__SHIFT = 4
A6XX_TEX_CONST_0_SWIZ_Y__MASK = 0x00000380
A6XX_TEX_CONST_0_SWIZ_Y__SHIFT = 7
A6XX_TEX_CONST_0_SWIZ_Z__MASK = 0x00001c00
A6XX_TEX_CONST_0_SWIZ_Z__SHIFT = 10
A6XX_TEX_CONST_0_SWIZ_W__MASK = 0x0000e000
A6XX_TEX_CONST_0_SWIZ_W__SHIFT = 13
A6XX_TEX_CONST_0_MIPLVLS__MASK = 0x000f0000
A6XX_TEX_CONST_0_MIPLVLS__SHIFT = 16
A6XX_TEX_CONST_0_CHROMA_MIDPOINT_X = 0x00010000
A6XX_TEX_CONST_0_CHROMA_MIDPOINT_Y = 0x00040000
A6XX_TEX_CONST_0_SAMPLES__MASK = 0x00300000
A6XX_TEX_CONST_0_SAMPLES__SHIFT = 20
A6XX_TEX_CONST_0_FMT__MASK = 0x3fc00000
A6XX_TEX_CONST_0_FMT__SHIFT = 22
A6XX_TEX_CONST_0_SWAP__MASK = 0xc0000000
A6XX_TEX_CONST_0_SWAP__SHIFT = 30
REG_A6XX_TEX_CONST_1 = 0x00000001
A6XX_TEX_CONST_1_WIDTH__MASK = 0x00007fff
A6XX_TEX_CONST_1_WIDTH__SHIFT = 0
A6XX_TEX_CONST_1_HEIGHT__MASK = 0x3fff8000
A6XX_TEX_CONST_1_HEIGHT__SHIFT = 15
A6XX_TEX_CONST_1_MUTABLEEN = 0x80000000
REG_A6XX_TEX_CONST_2 = 0x00000002
A6XX_TEX_CONST_2_STRUCTSIZETEXELS__MASK = 0x0000fff0
A6XX_TEX_CONST_2_STRUCTSIZETEXELS__SHIFT = 4
A6XX_TEX_CONST_2_STARTOFFSETTEXELS__MASK = 0x003f0000
A6XX_TEX_CONST_2_STARTOFFSETTEXELS__SHIFT = 16
A6XX_TEX_CONST_2_PITCHALIGN__MASK = 0x0000000f
A6XX_TEX_CONST_2_PITCHALIGN__SHIFT = 0
A6XX_TEX_CONST_2_PITCH__MASK = 0x1fffff80
A6XX_TEX_CONST_2_PITCH__SHIFT = 7
A6XX_TEX_CONST_2_TYPE__MASK = 0xe0000000
A6XX_TEX_CONST_2_TYPE__SHIFT = 29
REG_A6XX_TEX_CONST_3 = 0x00000003
A6XX_TEX_CONST_3_ARRAY_PITCH__MASK = 0x007fffff
A6XX_TEX_CONST_3_ARRAY_PITCH__SHIFT = 0
A6XX_TEX_CONST_3_MIN_LAYERSZ__MASK = 0x07800000
A6XX_TEX_CONST_3_MIN_LAYERSZ__SHIFT = 23
A6XX_TEX_CONST_3_TILE_ALL = 0x08000000
A6XX_TEX_CONST_3_FLAG = 0x10000000
REG_A6XX_TEX_CONST_4 = 0x00000004
A6XX_TEX_CONST_4_BASE_LO__MASK = 0xffffffe0
A6XX_TEX_CONST_4_BASE_LO__SHIFT = 5
REG_A6XX_TEX_CONST_5 = 0x00000005
A6XX_TEX_CONST_5_BASE_HI__MASK = 0x0001ffff
A6XX_TEX_CONST_5_BASE_HI__SHIFT = 0
A6XX_TEX_CONST_5_DEPTH__MASK = 0x3ffe0000
A6XX_TEX_CONST_5_DEPTH__SHIFT = 17
REG_A6XX_TEX_CONST_6 = 0x00000006
A6XX_TEX_CONST_6_MIN_LOD_CLAMP__MASK = 0x00000fff
A6XX_TEX_CONST_6_MIN_LOD_CLAMP__SHIFT = 0
A6XX_TEX_CONST_6_PLANE_PITCH__MASK = 0xffffff00
A6XX_TEX_CONST_6_PLANE_PITCH__SHIFT = 8
REG_A6XX_TEX_CONST_7 = 0x00000007
A6XX_TEX_CONST_7_FLAG_LO__MASK = 0xffffffe0
A6XX_TEX_CONST_7_FLAG_LO__SHIFT = 5
REG_A6XX_TEX_CONST_8 = 0x00000008
A6XX_TEX_CONST_8_FLAG_HI__MASK = 0x0001ffff
A6XX_TEX_CONST_8_FLAG_HI__SHIFT = 0
REG_A6XX_TEX_CONST_9 = 0x00000009
A6XX_TEX_CONST_9_FLAG_BUFFER_ARRAY_PITCH__MASK = 0x0001ffff
A6XX_TEX_CONST_9_FLAG_BUFFER_ARRAY_PITCH__SHIFT = 0
REG_A6XX_TEX_CONST_10 = 0x0000000a
A6XX_TEX_CONST_10_FLAG_BUFFER_PITCH__MASK = 0x0000007f
A6XX_TEX_CONST_10_FLAG_BUFFER_PITCH__SHIFT = 0
A6XX_TEX_CONST_10_FLAG_BUFFER_LOGW__MASK = 0x00000f00
A6XX_TEX_CONST_10_FLAG_BUFFER_LOGW__SHIFT = 8
A6XX_TEX_CONST_10_FLAG_BUFFER_LOGH__MASK = 0x0000f000
A6XX_TEX_CONST_10_FLAG_BUFFER_LOGH__SHIFT = 12
REG_A6XX_TEX_CONST_11 = 0x0000000b
REG_A6XX_TEX_CONST_12 = 0x0000000c
REG_A6XX_TEX_CONST_13 = 0x0000000d
REG_A6XX_TEX_CONST_14 = 0x0000000e
REG_A6XX_TEX_CONST_15 = 0x0000000f
REG_A6XX_UBO_0 = 0x00000000
A6XX_UBO_0_BASE_LO__MASK = 0xffffffff
A6XX_UBO_0_BASE_LO__SHIFT = 0
REG_A6XX_UBO_1 = 0x00000001
A6XX_UBO_1_BASE_HI__MASK = 0x0001ffff
A6XX_UBO_1_BASE_HI__SHIFT = 0
A6XX_UBO_1_SIZE__MASK = 0xfffe0000
A6XX_UBO_1_SIZE__SHIFT = 17
lvp_nir_options = gzip.decompress(base64.b64decode("H4sIAAAAAAAAA2NgZGRkYGAAkYxgCsQFsxigwgwQBoxmhCqFq2WEKwIrAEGIkQxoAEMALwCqVsCiGUwLMHA0QPn29nBJkswHANb8YpH4AAAA"))