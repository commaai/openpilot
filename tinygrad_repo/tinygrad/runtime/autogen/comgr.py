# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('comgr', [os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamd_comgr.so', 'amd_comgr'])
class amd_comgr_status_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_STATUS_SUCCESS = amd_comgr_status_s.define('AMD_COMGR_STATUS_SUCCESS', 0)
AMD_COMGR_STATUS_ERROR = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR', 1)
AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT', 2)
AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES', 3)

amd_comgr_status_t: TypeAlias = amd_comgr_status_s
class amd_comgr_language_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_LANGUAGE_NONE = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_NONE', 0)
AMD_COMGR_LANGUAGE_OPENCL_1_2 = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_OPENCL_1_2', 1)
AMD_COMGR_LANGUAGE_OPENCL_2_0 = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_OPENCL_2_0', 2)
AMD_COMGR_LANGUAGE_HC = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_HC', 3)
AMD_COMGR_LANGUAGE_HIP = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_HIP', 4)
AMD_COMGR_LANGUAGE_LLVM_IR = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_LLVM_IR', 5)
AMD_COMGR_LANGUAGE_LAST = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_LAST', 5)

amd_comgr_language_t: TypeAlias = amd_comgr_language_s
@dll.bind
def amd_comgr_status_string(status:amd_comgr_status_t, status_string:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> amd_comgr_status_t: ...
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def amd_comgr_get_version(major:c.POINTER[size_t], minor:c.POINTER[size_t]) -> None: ...
class amd_comgr_data_kind_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_DATA_KIND_UNDEF = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_UNDEF', 0)
AMD_COMGR_DATA_KIND_SOURCE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_SOURCE', 1)
AMD_COMGR_DATA_KIND_INCLUDE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_INCLUDE', 2)
AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER', 3)
AMD_COMGR_DATA_KIND_DIAGNOSTIC = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_DIAGNOSTIC', 4)
AMD_COMGR_DATA_KIND_LOG = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_LOG', 5)
AMD_COMGR_DATA_KIND_BC = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BC', 6)
AMD_COMGR_DATA_KIND_RELOCATABLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_RELOCATABLE', 7)
AMD_COMGR_DATA_KIND_EXECUTABLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_EXECUTABLE', 8)
AMD_COMGR_DATA_KIND_BYTES = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BYTES', 9)
AMD_COMGR_DATA_KIND_FATBIN = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_FATBIN', 16)
AMD_COMGR_DATA_KIND_AR = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_AR', 17)
AMD_COMGR_DATA_KIND_BC_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BC_BUNDLE', 18)
AMD_COMGR_DATA_KIND_AR_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_AR_BUNDLE', 19)
AMD_COMGR_DATA_KIND_OBJ_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_OBJ_BUNDLE', 20)
AMD_COMGR_DATA_KIND_LAST = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_LAST', 20)

amd_comgr_data_kind_t: TypeAlias = amd_comgr_data_kind_s
@c.record
class amd_comgr_data_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
amd_comgr_data_t: TypeAlias = amd_comgr_data_s
@c.record
class amd_comgr_data_set_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_data_set_t: TypeAlias = amd_comgr_data_set_s
@c.record
class amd_comgr_action_info_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_action_info_t: TypeAlias = amd_comgr_action_info_s
@c.record
class amd_comgr_metadata_node_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_metadata_node_t: TypeAlias = amd_comgr_metadata_node_s
@c.record
class amd_comgr_symbol_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_symbol_t: TypeAlias = amd_comgr_symbol_s
@c.record
class amd_comgr_disassembly_info_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_disassembly_info_t: TypeAlias = amd_comgr_disassembly_info_s
@c.record
class amd_comgr_symbolizer_info_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
amd_comgr_symbolizer_info_t: TypeAlias = amd_comgr_symbolizer_info_s
@dll.bind
def amd_comgr_get_isa_count(count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_isa_name(index:size_t, isa_name:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_isa_metadata(isa_name:c.POINTER[Annotated[bytes, ctypes.c_char]], metadata:c.POINTER[amd_comgr_metadata_node_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_create_data(kind:amd_comgr_data_kind_t, data:c.POINTER[amd_comgr_data_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_release_data(data:amd_comgr_data_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_data_kind(data:amd_comgr_data_t, kind:c.POINTER[amd_comgr_data_kind_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_set_data(data:amd_comgr_data_t, size:size_t, bytes:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_set_data_from_file_slice(data:amd_comgr_data_t, file_descriptor:Annotated[int, ctypes.c_int32], offset:uint64_t, size:uint64_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_set_data_name(data:amd_comgr_data_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_data(data:amd_comgr_data_t, size:c.POINTER[size_t], bytes:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_data_name(data:amd_comgr_data_t, size:c.POINTER[size_t], name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_data_isa_name(data:amd_comgr_data_t, size:c.POINTER[size_t], isa_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_create_symbolizer_info(code_object:amd_comgr_data_t, print_symbol_callback:c.CFUNCTYPE[None, [c.POINTER[Annotated[bytes, ctypes.c_char]], ctypes.c_void_p]], symbolizer_info:c.POINTER[amd_comgr_symbolizer_info_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_destroy_symbolizer_info(symbolizer_info:amd_comgr_symbolizer_info_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_symbolize(symbolizer_info:amd_comgr_symbolizer_info_t, address:uint64_t, is_code:Annotated[bool, ctypes.c_bool], user_data:ctypes.c_void_p) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_data_metadata(data:amd_comgr_data_t, metadata:c.POINTER[amd_comgr_metadata_node_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_destroy_metadata(metadata:amd_comgr_metadata_node_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_create_data_set(data_set:c.POINTER[amd_comgr_data_set_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_destroy_data_set(data_set:amd_comgr_data_set_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_data_set_add(data_set:amd_comgr_data_set_t, data:amd_comgr_data_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_data_set_remove(data_set:amd_comgr_data_set_t, data_kind:amd_comgr_data_kind_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_data_count(data_set:amd_comgr_data_set_t, data_kind:amd_comgr_data_kind_t, count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_data_get_data(data_set:amd_comgr_data_set_t, data_kind:amd_comgr_data_kind_t, index:size_t, data:c.POINTER[amd_comgr_data_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_create_action_info(action_info:c.POINTER[amd_comgr_action_info_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_destroy_action_info(action_info:amd_comgr_action_info_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_isa_name(action_info:amd_comgr_action_info_t, isa_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_isa_name(action_info:amd_comgr_action_info_t, size:c.POINTER[size_t], isa_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_language(action_info:amd_comgr_action_info_t, language:amd_comgr_language_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_language(action_info:amd_comgr_action_info_t, language:c.POINTER[amd_comgr_language_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_options(action_info:amd_comgr_action_info_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_options(action_info:amd_comgr_action_info_t, size:c.POINTER[size_t], options:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_option_list(action_info:amd_comgr_action_info_t, options:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]], count:size_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_option_list_count(action_info:amd_comgr_action_info_t, count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_option_list_item(action_info:amd_comgr_action_info_t, index:size_t, size:c.POINTER[size_t], option:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_bundle_entry_ids(action_info:amd_comgr_action_info_t, bundle_entry_ids:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]], count:size_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_bundle_entry_id_count(action_info:amd_comgr_action_info_t, count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_bundle_entry_id(action_info:amd_comgr_action_info_t, index:size_t, size:c.POINTER[size_t], bundle_entry_id:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_working_directory_path(action_info:amd_comgr_action_info_t, path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_working_directory_path(action_info:amd_comgr_action_info_t, size:c.POINTER[size_t], path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_set_logging(action_info:amd_comgr_action_info_t, logging:Annotated[bool, ctypes.c_bool]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_action_info_get_logging(action_info:amd_comgr_action_info_t, logging:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> amd_comgr_status_t: ...
class amd_comgr_action_kind_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR', 0)
AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS', 1)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC', 2)
AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES', 3)
AMD_COMGR_ACTION_LINK_BC_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_BC_TO_BC', 4)
AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC', 5)
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE', 6)
AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY', 7)
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE', 8)
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE', 9)
AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE', 10)
AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE', 11)
AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE', 12)
AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE', 13)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN', 14)
AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC', 15)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE', 16)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE', 17)
AMD_COMGR_ACTION_UNBUNDLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_UNBUNDLE', 18)
AMD_COMGR_ACTION_LAST = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LAST', 18)

amd_comgr_action_kind_t: TypeAlias = amd_comgr_action_kind_s
@dll.bind
def amd_comgr_do_action(kind:amd_comgr_action_kind_t, info:amd_comgr_action_info_t, input:amd_comgr_data_set_t, result:amd_comgr_data_set_t) -> amd_comgr_status_t: ...
class amd_comgr_metadata_kind_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_METADATA_KIND_NULL = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_NULL', 0)
AMD_COMGR_METADATA_KIND_STRING = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_STRING', 1)
AMD_COMGR_METADATA_KIND_MAP = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_MAP', 2)
AMD_COMGR_METADATA_KIND_LIST = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_LIST', 3)
AMD_COMGR_METADATA_KIND_LAST = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_LAST', 3)

amd_comgr_metadata_kind_t: TypeAlias = amd_comgr_metadata_kind_s
@dll.bind
def amd_comgr_get_metadata_kind(metadata:amd_comgr_metadata_node_t, kind:c.POINTER[amd_comgr_metadata_kind_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_metadata_string(metadata:amd_comgr_metadata_node_t, size:c.POINTER[size_t], string:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_metadata_map_size(metadata:amd_comgr_metadata_node_t, size:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_iterate_map_metadata(metadata:amd_comgr_metadata_node_t, callback:c.CFUNCTYPE[amd_comgr_status_t, [amd_comgr_metadata_node_t, amd_comgr_metadata_node_t, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_metadata_lookup(metadata:amd_comgr_metadata_node_t, key:c.POINTER[Annotated[bytes, ctypes.c_char]], value:c.POINTER[amd_comgr_metadata_node_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_metadata_list_size(metadata:amd_comgr_metadata_node_t, size:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_index_list_metadata(metadata:amd_comgr_metadata_node_t, index:size_t, value:c.POINTER[amd_comgr_metadata_node_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_iterate_symbols(data:amd_comgr_data_t, callback:c.CFUNCTYPE[amd_comgr_status_t, [amd_comgr_symbol_t, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_symbol_lookup(data:amd_comgr_data_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol:c.POINTER[amd_comgr_symbol_t]) -> amd_comgr_status_t: ...
class amd_comgr_symbol_type_s(Annotated[int, ctypes.c_int32], c.Enum): pass
AMD_COMGR_SYMBOL_TYPE_UNKNOWN = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_UNKNOWN', -1)
AMD_COMGR_SYMBOL_TYPE_NOTYPE = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_NOTYPE', 0)
AMD_COMGR_SYMBOL_TYPE_OBJECT = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_OBJECT', 1)
AMD_COMGR_SYMBOL_TYPE_FUNC = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_FUNC', 2)
AMD_COMGR_SYMBOL_TYPE_SECTION = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_SECTION', 3)
AMD_COMGR_SYMBOL_TYPE_FILE = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_FILE', 4)
AMD_COMGR_SYMBOL_TYPE_COMMON = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_COMMON', 5)
AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL', 10)

amd_comgr_symbol_type_t: TypeAlias = amd_comgr_symbol_type_s
class amd_comgr_symbol_info_s(Annotated[int, ctypes.c_uint32], c.Enum): pass
AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_NAME_LENGTH', 0)
AMD_COMGR_SYMBOL_INFO_NAME = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_NAME', 1)
AMD_COMGR_SYMBOL_INFO_TYPE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_TYPE', 2)
AMD_COMGR_SYMBOL_INFO_SIZE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_SIZE', 3)
AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED', 4)
AMD_COMGR_SYMBOL_INFO_VALUE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_VALUE', 5)
AMD_COMGR_SYMBOL_INFO_LAST = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_LAST', 5)

amd_comgr_symbol_info_t: TypeAlias = amd_comgr_symbol_info_s
@dll.bind
def amd_comgr_symbol_get_info(symbol:amd_comgr_symbol_t, attribute:amd_comgr_symbol_info_t, value:ctypes.c_void_p) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_create_disassembly_info(isa_name:c.POINTER[Annotated[bytes, ctypes.c_char]], read_memory_callback:c.CFUNCTYPE[uint64_t, [uint64_t, c.POINTER[Annotated[bytes, ctypes.c_char]], uint64_t, ctypes.c_void_p]], print_instruction_callback:c.CFUNCTYPE[None, [c.POINTER[Annotated[bytes, ctypes.c_char]], ctypes.c_void_p]], print_address_annotation_callback:c.CFUNCTYPE[None, [uint64_t, ctypes.c_void_p]], disassembly_info:c.POINTER[amd_comgr_disassembly_info_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_destroy_disassembly_info(disassembly_info:amd_comgr_disassembly_info_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_disassemble_instruction(disassembly_info:amd_comgr_disassembly_info_t, address:uint64_t, user_data:ctypes.c_void_p, size:c.POINTER[uint64_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_demangle_symbol_name(mangled_symbol_name:amd_comgr_data_t, demangled_symbol_name:c.POINTER[amd_comgr_data_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_populate_mangled_names(data:amd_comgr_data_t, count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_get_mangled_name(data:amd_comgr_data_t, index:size_t, size:c.POINTER[size_t], mangled_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_populate_name_expression_map(data:amd_comgr_data_t, count:c.POINTER[size_t]) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_map_name_expression_to_symbol_name(data:amd_comgr_data_t, size:c.POINTER[size_t], name_expression:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> amd_comgr_status_t: ...
@c.record
class code_object_info_s(c.Struct):
  SIZE = 24
  isa: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  size: Annotated[size_t, 8]
  offset: Annotated[uint64_t, 16]
amd_comgr_code_object_info_t: TypeAlias = code_object_info_s
@dll.bind
def amd_comgr_lookup_code_object(data:amd_comgr_data_t, info_list:c.POINTER[amd_comgr_code_object_info_t], info_list_size:size_t) -> amd_comgr_status_t: ...
@dll.bind
def amd_comgr_map_elf_virtual_address_to_code_object_offset(data:amd_comgr_data_t, elf_virtual_address:uint64_t, code_object_offset:c.POINTER[uint64_t], slice_size:c.POINTER[uint64_t], nobits:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> amd_comgr_status_t: ...
c.init_records()
AMD_COMGR_DEPRECATED = lambda msg: __attribute__((deprecated(msg))) # type: ignore
AMD_COMGR_INTERFACE_VERSION_MAJOR = 2 # type: ignore
AMD_COMGR_INTERFACE_VERSION_MINOR = 8 # type: ignore