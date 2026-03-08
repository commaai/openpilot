# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
from tinygrad.helpers import OSX
dll = c.DLL('libclang', '/opt/homebrew/opt/llvm@20/lib/libclang.dylib' if OSX else ['clang-20', 'clang'])
CXIndex: TypeAlias = ctypes.c_void_p
class struct_CXTargetInfoImpl(ctypes.Structure): pass
CXTargetInfo: TypeAlias = c.POINTER[struct_CXTargetInfoImpl]
class struct_CXTranslationUnitImpl(ctypes.Structure): pass
CXTranslationUnit: TypeAlias = c.POINTER[struct_CXTranslationUnitImpl]
CXClientData: TypeAlias = ctypes.c_void_p
@c.record
class struct_CXUnsavedFile(c.Struct):
  SIZE = 24
  Filename: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  Contents: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 8]
  Length: Annotated[Annotated[int, ctypes.c_uint64], 16]
class enum_CXAvailabilityKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXAvailability_Available = enum_CXAvailabilityKind.define('CXAvailability_Available', 0)
CXAvailability_Deprecated = enum_CXAvailabilityKind.define('CXAvailability_Deprecated', 1)
CXAvailability_NotAvailable = enum_CXAvailabilityKind.define('CXAvailability_NotAvailable', 2)
CXAvailability_NotAccessible = enum_CXAvailabilityKind.define('CXAvailability_NotAccessible', 3)

@c.record
class struct_CXVersion(c.Struct):
  SIZE = 12
  Major: Annotated[Annotated[int, ctypes.c_int32], 0]
  Minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  Subminor: Annotated[Annotated[int, ctypes.c_int32], 8]
CXVersion: TypeAlias = struct_CXVersion
class enum_CXCursor_ExceptionSpecificationKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCursor_ExceptionSpecificationKind_None = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_None', 0)
CXCursor_ExceptionSpecificationKind_DynamicNone = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_DynamicNone', 1)
CXCursor_ExceptionSpecificationKind_Dynamic = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Dynamic', 2)
CXCursor_ExceptionSpecificationKind_MSAny = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_MSAny', 3)
CXCursor_ExceptionSpecificationKind_BasicNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_BasicNoexcept', 4)
CXCursor_ExceptionSpecificationKind_ComputedNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_ComputedNoexcept', 5)
CXCursor_ExceptionSpecificationKind_Unevaluated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unevaluated', 6)
CXCursor_ExceptionSpecificationKind_Uninstantiated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Uninstantiated', 7)
CXCursor_ExceptionSpecificationKind_Unparsed = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unparsed', 8)
CXCursor_ExceptionSpecificationKind_NoThrow = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_NoThrow', 9)

@dll.bind
def clang_createIndex(excludeDeclarationsFromPCH:Annotated[int, ctypes.c_int32], displayDiagnostics:Annotated[int, ctypes.c_int32]) -> CXIndex: ...
@dll.bind
def clang_disposeIndex(index:CXIndex) -> None: ...
class CXChoice(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXChoice_Default = CXChoice.define('CXChoice_Default', 0)
CXChoice_Enabled = CXChoice.define('CXChoice_Enabled', 1)
CXChoice_Disabled = CXChoice.define('CXChoice_Disabled', 2)

class CXGlobalOptFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXGlobalOpt_None = CXGlobalOptFlags.define('CXGlobalOpt_None', 0)
CXGlobalOpt_ThreadBackgroundPriorityForIndexing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForIndexing', 1)
CXGlobalOpt_ThreadBackgroundPriorityForEditing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForEditing', 2)
CXGlobalOpt_ThreadBackgroundPriorityForAll = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForAll', 3)

@c.record
class struct_CXIndexOptions(c.Struct):
  SIZE = 24
  Size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ThreadBackgroundPriorityForIndexing: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  ThreadBackgroundPriorityForEditing: Annotated[Annotated[int, ctypes.c_ubyte], 5]
  ExcludeDeclarationsFromPCH: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 0]
  DisplayDiagnostics: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 1]
  StorePreamblesInMemory: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 2]
  PreambleStoragePath: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 8]
  InvocationEmissionPath: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 16]
CXIndexOptions: TypeAlias = struct_CXIndexOptions
@dll.bind
def clang_createIndexWithOptions(options:c.POINTER[CXIndexOptions]) -> CXIndex: ...
@dll.bind
def clang_CXIndex_setGlobalOptions(_0:CXIndex, options:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_CXIndex_getGlobalOptions(_0:CXIndex) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXIndex_setInvocationEmissionPathOption(_0:CXIndex, Path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> None: ...
CXFile: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_isFileMultipleIncludeGuarded(tu:CXTranslationUnit, file:CXFile) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getFile(tu:CXTranslationUnit, file_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXFile: ...
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def clang_getFileContents(tu:CXTranslationUnit, file:CXFile, size:c.POINTER[size_t]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@c.record
class CXSourceLocation(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[ctypes.c_void_p, Literal[2]], 0]
  int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
@dll.bind
def clang_getLocation(tu:CXTranslationUnit, file:CXFile, line:Annotated[int, ctypes.c_uint32], column:Annotated[int, ctypes.c_uint32]) -> CXSourceLocation: ...
@dll.bind
def clang_getLocationForOffset(tu:CXTranslationUnit, file:CXFile, offset:Annotated[int, ctypes.c_uint32]) -> CXSourceLocation: ...
@c.record
class CXSourceRangeList(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ranges: Annotated[c.POINTER[CXSourceRange], 8]
@c.record
class CXSourceRange(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[ctypes.c_void_p, Literal[2]], 0]
  begin_int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
  end_int_data: Annotated[Annotated[int, ctypes.c_uint32], 20]
@dll.bind
def clang_getSkippedRanges(tu:CXTranslationUnit, file:CXFile) -> c.POINTER[CXSourceRangeList]: ...
@dll.bind
def clang_getAllSkippedRanges(tu:CXTranslationUnit) -> c.POINTER[CXSourceRangeList]: ...
@dll.bind
def clang_getNumDiagnostics(Unit:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
CXDiagnostic: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_getDiagnostic(Unit:CXTranslationUnit, Index:Annotated[int, ctypes.c_uint32]) -> CXDiagnostic: ...
CXDiagnosticSet: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_getDiagnosticSetFromTU(Unit:CXTranslationUnit) -> CXDiagnosticSet: ...
@c.record
class CXString(c.Struct):
  SIZE = 16
  data: Annotated[ctypes.c_void_p, 0]
  private_flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_getTranslationUnitSpelling(CTUnit:CXTranslationUnit) -> CXString: ...
@dll.bind
def clang_createTranslationUnitFromSourceFile(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], num_clang_command_line_args:Annotated[int, ctypes.c_int32], clang_command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_unsaved_files:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile]) -> CXTranslationUnit: ...
@dll.bind
def clang_createTranslationUnit(CIdx:CXIndex, ast_filename:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXTranslationUnit: ...
class enum_CXErrorCode(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXError_Success = enum_CXErrorCode.define('CXError_Success', 0)
CXError_Failure = enum_CXErrorCode.define('CXError_Failure', 1)
CXError_Crashed = enum_CXErrorCode.define('CXError_Crashed', 2)
CXError_InvalidArguments = enum_CXErrorCode.define('CXError_InvalidArguments', 3)
CXError_ASTReadError = enum_CXErrorCode.define('CXError_ASTReadError', 4)

@dll.bind
def clang_createTranslationUnit2(CIdx:CXIndex, ast_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
class enum_CXTranslationUnit_Flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXTranslationUnit_None = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_None', 0)
CXTranslationUnit_DetailedPreprocessingRecord = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_DetailedPreprocessingRecord', 1)
CXTranslationUnit_Incomplete = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_Incomplete', 2)
CXTranslationUnit_PrecompiledPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_PrecompiledPreamble', 4)
CXTranslationUnit_CacheCompletionResults = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CacheCompletionResults', 8)
CXTranslationUnit_ForSerialization = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_ForSerialization', 16)
CXTranslationUnit_CXXChainedPCH = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CXXChainedPCH', 32)
CXTranslationUnit_SkipFunctionBodies = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SkipFunctionBodies', 64)
CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeBriefCommentsInCodeCompletion', 128)
CXTranslationUnit_CreatePreambleOnFirstParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CreatePreambleOnFirstParse', 256)
CXTranslationUnit_KeepGoing = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_KeepGoing', 512)
CXTranslationUnit_SingleFileParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SingleFileParse', 1024)
CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_LimitSkipFunctionBodiesToPreamble', 2048)
CXTranslationUnit_IncludeAttributedTypes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeAttributedTypes', 4096)
CXTranslationUnit_VisitImplicitAttributes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_VisitImplicitAttributes', 8192)
CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles', 16384)
CXTranslationUnit_RetainExcludedConditionalBlocks = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_RetainExcludedConditionalBlocks', 32768)

@dll.bind
def clang_defaultEditingTranslationUnitOptions() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_parseTranslationUnit(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> CXTranslationUnit: ...
@dll.bind
def clang_parseTranslationUnit2(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
@dll.bind
def clang_parseTranslationUnit2FullArgv(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
class enum_CXSaveTranslationUnit_Flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXSaveTranslationUnit_None = enum_CXSaveTranslationUnit_Flags.define('CXSaveTranslationUnit_None', 0)

@dll.bind
def clang_defaultSaveOptions(TU:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
class enum_CXSaveError(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXSaveError_None = enum_CXSaveError.define('CXSaveError_None', 0)
CXSaveError_Unknown = enum_CXSaveError.define('CXSaveError_Unknown', 1)
CXSaveError_TranslationErrors = enum_CXSaveError.define('CXSaveError_TranslationErrors', 2)
CXSaveError_InvalidTU = enum_CXSaveError.define('CXSaveError_InvalidTU', 3)

@dll.bind
def clang_saveTranslationUnit(TU:CXTranslationUnit, FileName:c.POINTER[Annotated[bytes, ctypes.c_char]], options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_suspendTranslationUnit(_0:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_disposeTranslationUnit(_0:CXTranslationUnit) -> None: ...
class enum_CXReparse_Flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXReparse_None = enum_CXReparse_Flags.define('CXReparse_None', 0)

@dll.bind
def clang_defaultReparseOptions(TU:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_reparseTranslationUnit(TU:CXTranslationUnit, num_unsaved_files:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile], options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
class enum_CXTUResourceUsageKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXTUResourceUsage_AST = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST', 1)
CXTUResourceUsage_Identifiers = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Identifiers', 2)
CXTUResourceUsage_Selectors = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Selectors', 3)
CXTUResourceUsage_GlobalCompletionResults = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_GlobalCompletionResults', 4)
CXTUResourceUsage_SourceManagerContentCache = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManagerContentCache', 5)
CXTUResourceUsage_AST_SideTables = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST_SideTables', 6)
CXTUResourceUsage_SourceManager_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_Malloc', 7)
CXTUResourceUsage_SourceManager_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_MMap', 8)
CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc', 9)
CXTUResourceUsage_ExternalASTSource_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_MMap', 10)
CXTUResourceUsage_Preprocessor = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor', 11)
CXTUResourceUsage_PreprocessingRecord = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_PreprocessingRecord', 12)
CXTUResourceUsage_SourceManager_DataStructures = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_DataStructures', 13)
CXTUResourceUsage_Preprocessor_HeaderSearch = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor_HeaderSearch', 14)
CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN', 1)
CXTUResourceUsage_MEMORY_IN_BYTES_END = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_END', 14)
CXTUResourceUsage_First = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_First', 1)
CXTUResourceUsage_Last = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Last', 14)

@dll.bind
def clang_getTUResourceUsageName(kind:enum_CXTUResourceUsageKind) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@c.record
class struct_CXTUResourceUsageEntry(c.Struct):
  SIZE = 16
  kind: Annotated[enum_CXTUResourceUsageKind, 0]
  amount: Annotated[Annotated[int, ctypes.c_uint64], 8]
CXTUResourceUsageEntry: TypeAlias = struct_CXTUResourceUsageEntry
@c.record
class struct_CXTUResourceUsage(c.Struct):
  SIZE = 24
  data: Annotated[ctypes.c_void_p, 0]
  numEntries: Annotated[Annotated[int, ctypes.c_uint32], 8]
  entries: Annotated[c.POINTER[CXTUResourceUsageEntry], 16]
CXTUResourceUsage: TypeAlias = struct_CXTUResourceUsage
@dll.bind
def clang_getCXTUResourceUsage(TU:CXTranslationUnit) -> CXTUResourceUsage: ...
@dll.bind
def clang_disposeCXTUResourceUsage(usage:CXTUResourceUsage) -> None: ...
@dll.bind
def clang_getTranslationUnitTargetInfo(CTUnit:CXTranslationUnit) -> CXTargetInfo: ...
@dll.bind
def clang_TargetInfo_dispose(Info:CXTargetInfo) -> None: ...
@dll.bind
def clang_TargetInfo_getTriple(Info:CXTargetInfo) -> CXString: ...
@dll.bind
def clang_TargetInfo_getPointerWidth(Info:CXTargetInfo) -> Annotated[int, ctypes.c_int32]: ...
class enum_CXCursorKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCursor_UnexposedDecl = enum_CXCursorKind.define('CXCursor_UnexposedDecl', 1)
CXCursor_StructDecl = enum_CXCursorKind.define('CXCursor_StructDecl', 2)
CXCursor_UnionDecl = enum_CXCursorKind.define('CXCursor_UnionDecl', 3)
CXCursor_ClassDecl = enum_CXCursorKind.define('CXCursor_ClassDecl', 4)
CXCursor_EnumDecl = enum_CXCursorKind.define('CXCursor_EnumDecl', 5)
CXCursor_FieldDecl = enum_CXCursorKind.define('CXCursor_FieldDecl', 6)
CXCursor_EnumConstantDecl = enum_CXCursorKind.define('CXCursor_EnumConstantDecl', 7)
CXCursor_FunctionDecl = enum_CXCursorKind.define('CXCursor_FunctionDecl', 8)
CXCursor_VarDecl = enum_CXCursorKind.define('CXCursor_VarDecl', 9)
CXCursor_ParmDecl = enum_CXCursorKind.define('CXCursor_ParmDecl', 10)
CXCursor_ObjCInterfaceDecl = enum_CXCursorKind.define('CXCursor_ObjCInterfaceDecl', 11)
CXCursor_ObjCCategoryDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryDecl', 12)
CXCursor_ObjCProtocolDecl = enum_CXCursorKind.define('CXCursor_ObjCProtocolDecl', 13)
CXCursor_ObjCPropertyDecl = enum_CXCursorKind.define('CXCursor_ObjCPropertyDecl', 14)
CXCursor_ObjCIvarDecl = enum_CXCursorKind.define('CXCursor_ObjCIvarDecl', 15)
CXCursor_ObjCInstanceMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCInstanceMethodDecl', 16)
CXCursor_ObjCClassMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCClassMethodDecl', 17)
CXCursor_ObjCImplementationDecl = enum_CXCursorKind.define('CXCursor_ObjCImplementationDecl', 18)
CXCursor_ObjCCategoryImplDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryImplDecl', 19)
CXCursor_TypedefDecl = enum_CXCursorKind.define('CXCursor_TypedefDecl', 20)
CXCursor_CXXMethod = enum_CXCursorKind.define('CXCursor_CXXMethod', 21)
CXCursor_Namespace = enum_CXCursorKind.define('CXCursor_Namespace', 22)
CXCursor_LinkageSpec = enum_CXCursorKind.define('CXCursor_LinkageSpec', 23)
CXCursor_Constructor = enum_CXCursorKind.define('CXCursor_Constructor', 24)
CXCursor_Destructor = enum_CXCursorKind.define('CXCursor_Destructor', 25)
CXCursor_ConversionFunction = enum_CXCursorKind.define('CXCursor_ConversionFunction', 26)
CXCursor_TemplateTypeParameter = enum_CXCursorKind.define('CXCursor_TemplateTypeParameter', 27)
CXCursor_NonTypeTemplateParameter = enum_CXCursorKind.define('CXCursor_NonTypeTemplateParameter', 28)
CXCursor_TemplateTemplateParameter = enum_CXCursorKind.define('CXCursor_TemplateTemplateParameter', 29)
CXCursor_FunctionTemplate = enum_CXCursorKind.define('CXCursor_FunctionTemplate', 30)
CXCursor_ClassTemplate = enum_CXCursorKind.define('CXCursor_ClassTemplate', 31)
CXCursor_ClassTemplatePartialSpecialization = enum_CXCursorKind.define('CXCursor_ClassTemplatePartialSpecialization', 32)
CXCursor_NamespaceAlias = enum_CXCursorKind.define('CXCursor_NamespaceAlias', 33)
CXCursor_UsingDirective = enum_CXCursorKind.define('CXCursor_UsingDirective', 34)
CXCursor_UsingDeclaration = enum_CXCursorKind.define('CXCursor_UsingDeclaration', 35)
CXCursor_TypeAliasDecl = enum_CXCursorKind.define('CXCursor_TypeAliasDecl', 36)
CXCursor_ObjCSynthesizeDecl = enum_CXCursorKind.define('CXCursor_ObjCSynthesizeDecl', 37)
CXCursor_ObjCDynamicDecl = enum_CXCursorKind.define('CXCursor_ObjCDynamicDecl', 38)
CXCursor_CXXAccessSpecifier = enum_CXCursorKind.define('CXCursor_CXXAccessSpecifier', 39)
CXCursor_FirstDecl = enum_CXCursorKind.define('CXCursor_FirstDecl', 1)
CXCursor_LastDecl = enum_CXCursorKind.define('CXCursor_LastDecl', 39)
CXCursor_FirstRef = enum_CXCursorKind.define('CXCursor_FirstRef', 40)
CXCursor_ObjCSuperClassRef = enum_CXCursorKind.define('CXCursor_ObjCSuperClassRef', 40)
CXCursor_ObjCProtocolRef = enum_CXCursorKind.define('CXCursor_ObjCProtocolRef', 41)
CXCursor_ObjCClassRef = enum_CXCursorKind.define('CXCursor_ObjCClassRef', 42)
CXCursor_TypeRef = enum_CXCursorKind.define('CXCursor_TypeRef', 43)
CXCursor_CXXBaseSpecifier = enum_CXCursorKind.define('CXCursor_CXXBaseSpecifier', 44)
CXCursor_TemplateRef = enum_CXCursorKind.define('CXCursor_TemplateRef', 45)
CXCursor_NamespaceRef = enum_CXCursorKind.define('CXCursor_NamespaceRef', 46)
CXCursor_MemberRef = enum_CXCursorKind.define('CXCursor_MemberRef', 47)
CXCursor_LabelRef = enum_CXCursorKind.define('CXCursor_LabelRef', 48)
CXCursor_OverloadedDeclRef = enum_CXCursorKind.define('CXCursor_OverloadedDeclRef', 49)
CXCursor_VariableRef = enum_CXCursorKind.define('CXCursor_VariableRef', 50)
CXCursor_LastRef = enum_CXCursorKind.define('CXCursor_LastRef', 50)
CXCursor_FirstInvalid = enum_CXCursorKind.define('CXCursor_FirstInvalid', 70)
CXCursor_InvalidFile = enum_CXCursorKind.define('CXCursor_InvalidFile', 70)
CXCursor_NoDeclFound = enum_CXCursorKind.define('CXCursor_NoDeclFound', 71)
CXCursor_NotImplemented = enum_CXCursorKind.define('CXCursor_NotImplemented', 72)
CXCursor_InvalidCode = enum_CXCursorKind.define('CXCursor_InvalidCode', 73)
CXCursor_LastInvalid = enum_CXCursorKind.define('CXCursor_LastInvalid', 73)
CXCursor_FirstExpr = enum_CXCursorKind.define('CXCursor_FirstExpr', 100)
CXCursor_UnexposedExpr = enum_CXCursorKind.define('CXCursor_UnexposedExpr', 100)
CXCursor_DeclRefExpr = enum_CXCursorKind.define('CXCursor_DeclRefExpr', 101)
CXCursor_MemberRefExpr = enum_CXCursorKind.define('CXCursor_MemberRefExpr', 102)
CXCursor_CallExpr = enum_CXCursorKind.define('CXCursor_CallExpr', 103)
CXCursor_ObjCMessageExpr = enum_CXCursorKind.define('CXCursor_ObjCMessageExpr', 104)
CXCursor_BlockExpr = enum_CXCursorKind.define('CXCursor_BlockExpr', 105)
CXCursor_IntegerLiteral = enum_CXCursorKind.define('CXCursor_IntegerLiteral', 106)
CXCursor_FloatingLiteral = enum_CXCursorKind.define('CXCursor_FloatingLiteral', 107)
CXCursor_ImaginaryLiteral = enum_CXCursorKind.define('CXCursor_ImaginaryLiteral', 108)
CXCursor_StringLiteral = enum_CXCursorKind.define('CXCursor_StringLiteral', 109)
CXCursor_CharacterLiteral = enum_CXCursorKind.define('CXCursor_CharacterLiteral', 110)
CXCursor_ParenExpr = enum_CXCursorKind.define('CXCursor_ParenExpr', 111)
CXCursor_UnaryOperator = enum_CXCursorKind.define('CXCursor_UnaryOperator', 112)
CXCursor_ArraySubscriptExpr = enum_CXCursorKind.define('CXCursor_ArraySubscriptExpr', 113)
CXCursor_BinaryOperator = enum_CXCursorKind.define('CXCursor_BinaryOperator', 114)
CXCursor_CompoundAssignOperator = enum_CXCursorKind.define('CXCursor_CompoundAssignOperator', 115)
CXCursor_ConditionalOperator = enum_CXCursorKind.define('CXCursor_ConditionalOperator', 116)
CXCursor_CStyleCastExpr = enum_CXCursorKind.define('CXCursor_CStyleCastExpr', 117)
CXCursor_CompoundLiteralExpr = enum_CXCursorKind.define('CXCursor_CompoundLiteralExpr', 118)
CXCursor_InitListExpr = enum_CXCursorKind.define('CXCursor_InitListExpr', 119)
CXCursor_AddrLabelExpr = enum_CXCursorKind.define('CXCursor_AddrLabelExpr', 120)
CXCursor_StmtExpr = enum_CXCursorKind.define('CXCursor_StmtExpr', 121)
CXCursor_GenericSelectionExpr = enum_CXCursorKind.define('CXCursor_GenericSelectionExpr', 122)
CXCursor_GNUNullExpr = enum_CXCursorKind.define('CXCursor_GNUNullExpr', 123)
CXCursor_CXXStaticCastExpr = enum_CXCursorKind.define('CXCursor_CXXStaticCastExpr', 124)
CXCursor_CXXDynamicCastExpr = enum_CXCursorKind.define('CXCursor_CXXDynamicCastExpr', 125)
CXCursor_CXXReinterpretCastExpr = enum_CXCursorKind.define('CXCursor_CXXReinterpretCastExpr', 126)
CXCursor_CXXConstCastExpr = enum_CXCursorKind.define('CXCursor_CXXConstCastExpr', 127)
CXCursor_CXXFunctionalCastExpr = enum_CXCursorKind.define('CXCursor_CXXFunctionalCastExpr', 128)
CXCursor_CXXTypeidExpr = enum_CXCursorKind.define('CXCursor_CXXTypeidExpr', 129)
CXCursor_CXXBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXBoolLiteralExpr', 130)
CXCursor_CXXNullPtrLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXNullPtrLiteralExpr', 131)
CXCursor_CXXThisExpr = enum_CXCursorKind.define('CXCursor_CXXThisExpr', 132)
CXCursor_CXXThrowExpr = enum_CXCursorKind.define('CXCursor_CXXThrowExpr', 133)
CXCursor_CXXNewExpr = enum_CXCursorKind.define('CXCursor_CXXNewExpr', 134)
CXCursor_CXXDeleteExpr = enum_CXCursorKind.define('CXCursor_CXXDeleteExpr', 135)
CXCursor_UnaryExpr = enum_CXCursorKind.define('CXCursor_UnaryExpr', 136)
CXCursor_ObjCStringLiteral = enum_CXCursorKind.define('CXCursor_ObjCStringLiteral', 137)
CXCursor_ObjCEncodeExpr = enum_CXCursorKind.define('CXCursor_ObjCEncodeExpr', 138)
CXCursor_ObjCSelectorExpr = enum_CXCursorKind.define('CXCursor_ObjCSelectorExpr', 139)
CXCursor_ObjCProtocolExpr = enum_CXCursorKind.define('CXCursor_ObjCProtocolExpr', 140)
CXCursor_ObjCBridgedCastExpr = enum_CXCursorKind.define('CXCursor_ObjCBridgedCastExpr', 141)
CXCursor_PackExpansionExpr = enum_CXCursorKind.define('CXCursor_PackExpansionExpr', 142)
CXCursor_SizeOfPackExpr = enum_CXCursorKind.define('CXCursor_SizeOfPackExpr', 143)
CXCursor_LambdaExpr = enum_CXCursorKind.define('CXCursor_LambdaExpr', 144)
CXCursor_ObjCBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_ObjCBoolLiteralExpr', 145)
CXCursor_ObjCSelfExpr = enum_CXCursorKind.define('CXCursor_ObjCSelfExpr', 146)
CXCursor_ArraySectionExpr = enum_CXCursorKind.define('CXCursor_ArraySectionExpr', 147)
CXCursor_ObjCAvailabilityCheckExpr = enum_CXCursorKind.define('CXCursor_ObjCAvailabilityCheckExpr', 148)
CXCursor_FixedPointLiteral = enum_CXCursorKind.define('CXCursor_FixedPointLiteral', 149)
CXCursor_OMPArrayShapingExpr = enum_CXCursorKind.define('CXCursor_OMPArrayShapingExpr', 150)
CXCursor_OMPIteratorExpr = enum_CXCursorKind.define('CXCursor_OMPIteratorExpr', 151)
CXCursor_CXXAddrspaceCastExpr = enum_CXCursorKind.define('CXCursor_CXXAddrspaceCastExpr', 152)
CXCursor_ConceptSpecializationExpr = enum_CXCursorKind.define('CXCursor_ConceptSpecializationExpr', 153)
CXCursor_RequiresExpr = enum_CXCursorKind.define('CXCursor_RequiresExpr', 154)
CXCursor_CXXParenListInitExpr = enum_CXCursorKind.define('CXCursor_CXXParenListInitExpr', 155)
CXCursor_PackIndexingExpr = enum_CXCursorKind.define('CXCursor_PackIndexingExpr', 156)
CXCursor_LastExpr = enum_CXCursorKind.define('CXCursor_LastExpr', 156)
CXCursor_FirstStmt = enum_CXCursorKind.define('CXCursor_FirstStmt', 200)
CXCursor_UnexposedStmt = enum_CXCursorKind.define('CXCursor_UnexposedStmt', 200)
CXCursor_LabelStmt = enum_CXCursorKind.define('CXCursor_LabelStmt', 201)
CXCursor_CompoundStmt = enum_CXCursorKind.define('CXCursor_CompoundStmt', 202)
CXCursor_CaseStmt = enum_CXCursorKind.define('CXCursor_CaseStmt', 203)
CXCursor_DefaultStmt = enum_CXCursorKind.define('CXCursor_DefaultStmt', 204)
CXCursor_IfStmt = enum_CXCursorKind.define('CXCursor_IfStmt', 205)
CXCursor_SwitchStmt = enum_CXCursorKind.define('CXCursor_SwitchStmt', 206)
CXCursor_WhileStmt = enum_CXCursorKind.define('CXCursor_WhileStmt', 207)
CXCursor_DoStmt = enum_CXCursorKind.define('CXCursor_DoStmt', 208)
CXCursor_ForStmt = enum_CXCursorKind.define('CXCursor_ForStmt', 209)
CXCursor_GotoStmt = enum_CXCursorKind.define('CXCursor_GotoStmt', 210)
CXCursor_IndirectGotoStmt = enum_CXCursorKind.define('CXCursor_IndirectGotoStmt', 211)
CXCursor_ContinueStmt = enum_CXCursorKind.define('CXCursor_ContinueStmt', 212)
CXCursor_BreakStmt = enum_CXCursorKind.define('CXCursor_BreakStmt', 213)
CXCursor_ReturnStmt = enum_CXCursorKind.define('CXCursor_ReturnStmt', 214)
CXCursor_GCCAsmStmt = enum_CXCursorKind.define('CXCursor_GCCAsmStmt', 215)
CXCursor_AsmStmt = enum_CXCursorKind.define('CXCursor_AsmStmt', 215)
CXCursor_ObjCAtTryStmt = enum_CXCursorKind.define('CXCursor_ObjCAtTryStmt', 216)
CXCursor_ObjCAtCatchStmt = enum_CXCursorKind.define('CXCursor_ObjCAtCatchStmt', 217)
CXCursor_ObjCAtFinallyStmt = enum_CXCursorKind.define('CXCursor_ObjCAtFinallyStmt', 218)
CXCursor_ObjCAtThrowStmt = enum_CXCursorKind.define('CXCursor_ObjCAtThrowStmt', 219)
CXCursor_ObjCAtSynchronizedStmt = enum_CXCursorKind.define('CXCursor_ObjCAtSynchronizedStmt', 220)
CXCursor_ObjCAutoreleasePoolStmt = enum_CXCursorKind.define('CXCursor_ObjCAutoreleasePoolStmt', 221)
CXCursor_ObjCForCollectionStmt = enum_CXCursorKind.define('CXCursor_ObjCForCollectionStmt', 222)
CXCursor_CXXCatchStmt = enum_CXCursorKind.define('CXCursor_CXXCatchStmt', 223)
CXCursor_CXXTryStmt = enum_CXCursorKind.define('CXCursor_CXXTryStmt', 224)
CXCursor_CXXForRangeStmt = enum_CXCursorKind.define('CXCursor_CXXForRangeStmt', 225)
CXCursor_SEHTryStmt = enum_CXCursorKind.define('CXCursor_SEHTryStmt', 226)
CXCursor_SEHExceptStmt = enum_CXCursorKind.define('CXCursor_SEHExceptStmt', 227)
CXCursor_SEHFinallyStmt = enum_CXCursorKind.define('CXCursor_SEHFinallyStmt', 228)
CXCursor_MSAsmStmt = enum_CXCursorKind.define('CXCursor_MSAsmStmt', 229)
CXCursor_NullStmt = enum_CXCursorKind.define('CXCursor_NullStmt', 230)
CXCursor_DeclStmt = enum_CXCursorKind.define('CXCursor_DeclStmt', 231)
CXCursor_OMPParallelDirective = enum_CXCursorKind.define('CXCursor_OMPParallelDirective', 232)
CXCursor_OMPSimdDirective = enum_CXCursorKind.define('CXCursor_OMPSimdDirective', 233)
CXCursor_OMPForDirective = enum_CXCursorKind.define('CXCursor_OMPForDirective', 234)
CXCursor_OMPSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPSectionsDirective', 235)
CXCursor_OMPSectionDirective = enum_CXCursorKind.define('CXCursor_OMPSectionDirective', 236)
CXCursor_OMPSingleDirective = enum_CXCursorKind.define('CXCursor_OMPSingleDirective', 237)
CXCursor_OMPParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForDirective', 238)
CXCursor_OMPParallelSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPParallelSectionsDirective', 239)
CXCursor_OMPTaskDirective = enum_CXCursorKind.define('CXCursor_OMPTaskDirective', 240)
CXCursor_OMPMasterDirective = enum_CXCursorKind.define('CXCursor_OMPMasterDirective', 241)
CXCursor_OMPCriticalDirective = enum_CXCursorKind.define('CXCursor_OMPCriticalDirective', 242)
CXCursor_OMPTaskyieldDirective = enum_CXCursorKind.define('CXCursor_OMPTaskyieldDirective', 243)
CXCursor_OMPBarrierDirective = enum_CXCursorKind.define('CXCursor_OMPBarrierDirective', 244)
CXCursor_OMPTaskwaitDirective = enum_CXCursorKind.define('CXCursor_OMPTaskwaitDirective', 245)
CXCursor_OMPFlushDirective = enum_CXCursorKind.define('CXCursor_OMPFlushDirective', 246)
CXCursor_SEHLeaveStmt = enum_CXCursorKind.define('CXCursor_SEHLeaveStmt', 247)
CXCursor_OMPOrderedDirective = enum_CXCursorKind.define('CXCursor_OMPOrderedDirective', 248)
CXCursor_OMPAtomicDirective = enum_CXCursorKind.define('CXCursor_OMPAtomicDirective', 249)
CXCursor_OMPForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPForSimdDirective', 250)
CXCursor_OMPParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForSimdDirective', 251)
CXCursor_OMPTargetDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDirective', 252)
CXCursor_OMPTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDirective', 253)
CXCursor_OMPTaskgroupDirective = enum_CXCursorKind.define('CXCursor_OMPTaskgroupDirective', 254)
CXCursor_OMPCancellationPointDirective = enum_CXCursorKind.define('CXCursor_OMPCancellationPointDirective', 255)
CXCursor_OMPCancelDirective = enum_CXCursorKind.define('CXCursor_OMPCancelDirective', 256)
CXCursor_OMPTargetDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDataDirective', 257)
CXCursor_OMPTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopDirective', 258)
CXCursor_OMPTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopSimdDirective', 259)
CXCursor_OMPDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeDirective', 260)
CXCursor_OMPTargetEnterDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetEnterDataDirective', 261)
CXCursor_OMPTargetExitDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetExitDataDirective', 262)
CXCursor_OMPTargetParallelDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelDirective', 263)
CXCursor_OMPTargetParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForDirective', 264)
CXCursor_OMPTargetUpdateDirective = enum_CXCursorKind.define('CXCursor_OMPTargetUpdateDirective', 265)
CXCursor_OMPDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForDirective', 266)
CXCursor_OMPDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForSimdDirective', 267)
CXCursor_OMPDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeSimdDirective', 268)
CXCursor_OMPTargetParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForSimdDirective', 269)
CXCursor_OMPTargetSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetSimdDirective', 270)
CXCursor_OMPTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeDirective', 271)
CXCursor_OMPTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeSimdDirective', 272)
CXCursor_OMPTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForSimdDirective', 273)
CXCursor_OMPTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForDirective', 274)
CXCursor_OMPTargetTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDirective', 275)
CXCursor_OMPTargetTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeDirective', 276)
CXCursor_OMPTargetTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForDirective', 277)
CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective', 278)
CXCursor_OMPTargetTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeSimdDirective', 279)
CXCursor_BuiltinBitCastExpr = enum_CXCursorKind.define('CXCursor_BuiltinBitCastExpr', 280)
CXCursor_OMPMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopDirective', 281)
CXCursor_OMPParallelMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopDirective', 282)
CXCursor_OMPMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopSimdDirective', 283)
CXCursor_OMPParallelMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopSimdDirective', 284)
CXCursor_OMPParallelMasterDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterDirective', 285)
CXCursor_OMPDepobjDirective = enum_CXCursorKind.define('CXCursor_OMPDepobjDirective', 286)
CXCursor_OMPScanDirective = enum_CXCursorKind.define('CXCursor_OMPScanDirective', 287)
CXCursor_OMPTileDirective = enum_CXCursorKind.define('CXCursor_OMPTileDirective', 288)
CXCursor_OMPCanonicalLoop = enum_CXCursorKind.define('CXCursor_OMPCanonicalLoop', 289)
CXCursor_OMPInteropDirective = enum_CXCursorKind.define('CXCursor_OMPInteropDirective', 290)
CXCursor_OMPDispatchDirective = enum_CXCursorKind.define('CXCursor_OMPDispatchDirective', 291)
CXCursor_OMPMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedDirective', 292)
CXCursor_OMPUnrollDirective = enum_CXCursorKind.define('CXCursor_OMPUnrollDirective', 293)
CXCursor_OMPMetaDirective = enum_CXCursorKind.define('CXCursor_OMPMetaDirective', 294)
CXCursor_OMPGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPGenericLoopDirective', 295)
CXCursor_OMPTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsGenericLoopDirective', 296)
CXCursor_OMPTargetTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsGenericLoopDirective', 297)
CXCursor_OMPParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelGenericLoopDirective', 298)
CXCursor_OMPTargetParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelGenericLoopDirective', 299)
CXCursor_OMPParallelMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedDirective', 300)
CXCursor_OMPMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopDirective', 301)
CXCursor_OMPMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopSimdDirective', 302)
CXCursor_OMPParallelMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopDirective', 303)
CXCursor_OMPParallelMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopSimdDirective', 304)
CXCursor_OMPErrorDirective = enum_CXCursorKind.define('CXCursor_OMPErrorDirective', 305)
CXCursor_OMPScopeDirective = enum_CXCursorKind.define('CXCursor_OMPScopeDirective', 306)
CXCursor_OMPReverseDirective = enum_CXCursorKind.define('CXCursor_OMPReverseDirective', 307)
CXCursor_OMPInterchangeDirective = enum_CXCursorKind.define('CXCursor_OMPInterchangeDirective', 308)
CXCursor_OMPAssumeDirective = enum_CXCursorKind.define('CXCursor_OMPAssumeDirective', 309)
CXCursor_OpenACCComputeConstruct = enum_CXCursorKind.define('CXCursor_OpenACCComputeConstruct', 320)
CXCursor_OpenACCLoopConstruct = enum_CXCursorKind.define('CXCursor_OpenACCLoopConstruct', 321)
CXCursor_OpenACCCombinedConstruct = enum_CXCursorKind.define('CXCursor_OpenACCCombinedConstruct', 322)
CXCursor_OpenACCDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCDataConstruct', 323)
CXCursor_OpenACCEnterDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCEnterDataConstruct', 324)
CXCursor_OpenACCExitDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCExitDataConstruct', 325)
CXCursor_OpenACCHostDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCHostDataConstruct', 326)
CXCursor_OpenACCWaitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCWaitConstruct', 327)
CXCursor_OpenACCInitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCInitConstruct', 328)
CXCursor_OpenACCShutdownConstruct = enum_CXCursorKind.define('CXCursor_OpenACCShutdownConstruct', 329)
CXCursor_OpenACCSetConstruct = enum_CXCursorKind.define('CXCursor_OpenACCSetConstruct', 330)
CXCursor_OpenACCUpdateConstruct = enum_CXCursorKind.define('CXCursor_OpenACCUpdateConstruct', 331)
CXCursor_LastStmt = enum_CXCursorKind.define('CXCursor_LastStmt', 331)
CXCursor_TranslationUnit = enum_CXCursorKind.define('CXCursor_TranslationUnit', 350)
CXCursor_FirstAttr = enum_CXCursorKind.define('CXCursor_FirstAttr', 400)
CXCursor_UnexposedAttr = enum_CXCursorKind.define('CXCursor_UnexposedAttr', 400)
CXCursor_IBActionAttr = enum_CXCursorKind.define('CXCursor_IBActionAttr', 401)
CXCursor_IBOutletAttr = enum_CXCursorKind.define('CXCursor_IBOutletAttr', 402)
CXCursor_IBOutletCollectionAttr = enum_CXCursorKind.define('CXCursor_IBOutletCollectionAttr', 403)
CXCursor_CXXFinalAttr = enum_CXCursorKind.define('CXCursor_CXXFinalAttr', 404)
CXCursor_CXXOverrideAttr = enum_CXCursorKind.define('CXCursor_CXXOverrideAttr', 405)
CXCursor_AnnotateAttr = enum_CXCursorKind.define('CXCursor_AnnotateAttr', 406)
CXCursor_AsmLabelAttr = enum_CXCursorKind.define('CXCursor_AsmLabelAttr', 407)
CXCursor_PackedAttr = enum_CXCursorKind.define('CXCursor_PackedAttr', 408)
CXCursor_PureAttr = enum_CXCursorKind.define('CXCursor_PureAttr', 409)
CXCursor_ConstAttr = enum_CXCursorKind.define('CXCursor_ConstAttr', 410)
CXCursor_NoDuplicateAttr = enum_CXCursorKind.define('CXCursor_NoDuplicateAttr', 411)
CXCursor_CUDAConstantAttr = enum_CXCursorKind.define('CXCursor_CUDAConstantAttr', 412)
CXCursor_CUDADeviceAttr = enum_CXCursorKind.define('CXCursor_CUDADeviceAttr', 413)
CXCursor_CUDAGlobalAttr = enum_CXCursorKind.define('CXCursor_CUDAGlobalAttr', 414)
CXCursor_CUDAHostAttr = enum_CXCursorKind.define('CXCursor_CUDAHostAttr', 415)
CXCursor_CUDASharedAttr = enum_CXCursorKind.define('CXCursor_CUDASharedAttr', 416)
CXCursor_VisibilityAttr = enum_CXCursorKind.define('CXCursor_VisibilityAttr', 417)
CXCursor_DLLExport = enum_CXCursorKind.define('CXCursor_DLLExport', 418)
CXCursor_DLLImport = enum_CXCursorKind.define('CXCursor_DLLImport', 419)
CXCursor_NSReturnsRetained = enum_CXCursorKind.define('CXCursor_NSReturnsRetained', 420)
CXCursor_NSReturnsNotRetained = enum_CXCursorKind.define('CXCursor_NSReturnsNotRetained', 421)
CXCursor_NSReturnsAutoreleased = enum_CXCursorKind.define('CXCursor_NSReturnsAutoreleased', 422)
CXCursor_NSConsumesSelf = enum_CXCursorKind.define('CXCursor_NSConsumesSelf', 423)
CXCursor_NSConsumed = enum_CXCursorKind.define('CXCursor_NSConsumed', 424)
CXCursor_ObjCException = enum_CXCursorKind.define('CXCursor_ObjCException', 425)
CXCursor_ObjCNSObject = enum_CXCursorKind.define('CXCursor_ObjCNSObject', 426)
CXCursor_ObjCIndependentClass = enum_CXCursorKind.define('CXCursor_ObjCIndependentClass', 427)
CXCursor_ObjCPreciseLifetime = enum_CXCursorKind.define('CXCursor_ObjCPreciseLifetime', 428)
CXCursor_ObjCReturnsInnerPointer = enum_CXCursorKind.define('CXCursor_ObjCReturnsInnerPointer', 429)
CXCursor_ObjCRequiresSuper = enum_CXCursorKind.define('CXCursor_ObjCRequiresSuper', 430)
CXCursor_ObjCRootClass = enum_CXCursorKind.define('CXCursor_ObjCRootClass', 431)
CXCursor_ObjCSubclassingRestricted = enum_CXCursorKind.define('CXCursor_ObjCSubclassingRestricted', 432)
CXCursor_ObjCExplicitProtocolImpl = enum_CXCursorKind.define('CXCursor_ObjCExplicitProtocolImpl', 433)
CXCursor_ObjCDesignatedInitializer = enum_CXCursorKind.define('CXCursor_ObjCDesignatedInitializer', 434)
CXCursor_ObjCRuntimeVisible = enum_CXCursorKind.define('CXCursor_ObjCRuntimeVisible', 435)
CXCursor_ObjCBoxable = enum_CXCursorKind.define('CXCursor_ObjCBoxable', 436)
CXCursor_FlagEnum = enum_CXCursorKind.define('CXCursor_FlagEnum', 437)
CXCursor_ConvergentAttr = enum_CXCursorKind.define('CXCursor_ConvergentAttr', 438)
CXCursor_WarnUnusedAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedAttr', 439)
CXCursor_WarnUnusedResultAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedResultAttr', 440)
CXCursor_AlignedAttr = enum_CXCursorKind.define('CXCursor_AlignedAttr', 441)
CXCursor_LastAttr = enum_CXCursorKind.define('CXCursor_LastAttr', 441)
CXCursor_PreprocessingDirective = enum_CXCursorKind.define('CXCursor_PreprocessingDirective', 500)
CXCursor_MacroDefinition = enum_CXCursorKind.define('CXCursor_MacroDefinition', 501)
CXCursor_MacroExpansion = enum_CXCursorKind.define('CXCursor_MacroExpansion', 502)
CXCursor_MacroInstantiation = enum_CXCursorKind.define('CXCursor_MacroInstantiation', 502)
CXCursor_InclusionDirective = enum_CXCursorKind.define('CXCursor_InclusionDirective', 503)
CXCursor_FirstPreprocessing = enum_CXCursorKind.define('CXCursor_FirstPreprocessing', 500)
CXCursor_LastPreprocessing = enum_CXCursorKind.define('CXCursor_LastPreprocessing', 503)
CXCursor_ModuleImportDecl = enum_CXCursorKind.define('CXCursor_ModuleImportDecl', 600)
CXCursor_TypeAliasTemplateDecl = enum_CXCursorKind.define('CXCursor_TypeAliasTemplateDecl', 601)
CXCursor_StaticAssert = enum_CXCursorKind.define('CXCursor_StaticAssert', 602)
CXCursor_FriendDecl = enum_CXCursorKind.define('CXCursor_FriendDecl', 603)
CXCursor_ConceptDecl = enum_CXCursorKind.define('CXCursor_ConceptDecl', 604)
CXCursor_FirstExtraDecl = enum_CXCursorKind.define('CXCursor_FirstExtraDecl', 600)
CXCursor_LastExtraDecl = enum_CXCursorKind.define('CXCursor_LastExtraDecl', 604)
CXCursor_OverloadCandidate = enum_CXCursorKind.define('CXCursor_OverloadCandidate', 700)

@c.record
class CXCursor(c.Struct):
  SIZE = 32
  kind: Annotated[enum_CXCursorKind, 0]
  xdata: Annotated[Annotated[int, ctypes.c_int32], 4]
  data: Annotated[c.Array[ctypes.c_void_p, Literal[3]], 8]
@dll.bind
def clang_getNullCursor() -> CXCursor: ...
@dll.bind
def clang_getTranslationUnitCursor(_0:CXTranslationUnit) -> CXCursor: ...
@dll.bind
def clang_equalCursors(_0:CXCursor, _1:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isNull(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_hashCursor(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorKind(_0:CXCursor) -> enum_CXCursorKind: ...
@dll.bind
def clang_isDeclaration(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isInvalidDeclaration(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isReference(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isExpression(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isStatement(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isAttribute(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_hasAttrs(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isInvalid(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isTranslationUnit(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isPreprocessing(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isUnexposed(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
class enum_CXLinkageKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXLinkage_Invalid = enum_CXLinkageKind.define('CXLinkage_Invalid', 0)
CXLinkage_NoLinkage = enum_CXLinkageKind.define('CXLinkage_NoLinkage', 1)
CXLinkage_Internal = enum_CXLinkageKind.define('CXLinkage_Internal', 2)
CXLinkage_UniqueExternal = enum_CXLinkageKind.define('CXLinkage_UniqueExternal', 3)
CXLinkage_External = enum_CXLinkageKind.define('CXLinkage_External', 4)

@dll.bind
def clang_getCursorLinkage(cursor:CXCursor) -> enum_CXLinkageKind: ...
class enum_CXVisibilityKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXVisibility_Invalid = enum_CXVisibilityKind.define('CXVisibility_Invalid', 0)
CXVisibility_Hidden = enum_CXVisibilityKind.define('CXVisibility_Hidden', 1)
CXVisibility_Protected = enum_CXVisibilityKind.define('CXVisibility_Protected', 2)
CXVisibility_Default = enum_CXVisibilityKind.define('CXVisibility_Default', 3)

@dll.bind
def clang_getCursorVisibility(cursor:CXCursor) -> enum_CXVisibilityKind: ...
@dll.bind
def clang_getCursorAvailability(cursor:CXCursor) -> enum_CXAvailabilityKind: ...
@c.record
class struct_CXPlatformAvailability(c.Struct):
  SIZE = 72
  Platform: Annotated[CXString, 0]
  Introduced: Annotated[CXVersion, 16]
  Deprecated: Annotated[CXVersion, 28]
  Obsoleted: Annotated[CXVersion, 40]
  Unavailable: Annotated[Annotated[int, ctypes.c_int32], 52]
  Message: Annotated[CXString, 56]
CXPlatformAvailability: TypeAlias = struct_CXPlatformAvailability
@dll.bind
def clang_getCursorPlatformAvailability(cursor:CXCursor, always_deprecated:c.POINTER[Annotated[int, ctypes.c_int32]], deprecated_message:c.POINTER[CXString], always_unavailable:c.POINTER[Annotated[int, ctypes.c_int32]], unavailable_message:c.POINTER[CXString], availability:c.POINTER[CXPlatformAvailability], availability_size:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_disposeCXPlatformAvailability(availability:c.POINTER[CXPlatformAvailability]) -> None: ...
@dll.bind
def clang_Cursor_getVarDeclInitializer(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_Cursor_hasVarDeclGlobalStorage(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_hasVarDeclExternalStorage(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
class enum_CXLanguageKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXLanguage_Invalid = enum_CXLanguageKind.define('CXLanguage_Invalid', 0)
CXLanguage_C = enum_CXLanguageKind.define('CXLanguage_C', 1)
CXLanguage_ObjC = enum_CXLanguageKind.define('CXLanguage_ObjC', 2)
CXLanguage_CPlusPlus = enum_CXLanguageKind.define('CXLanguage_CPlusPlus', 3)

@dll.bind
def clang_getCursorLanguage(cursor:CXCursor) -> enum_CXLanguageKind: ...
class enum_CXTLSKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXTLS_None = enum_CXTLSKind.define('CXTLS_None', 0)
CXTLS_Dynamic = enum_CXTLSKind.define('CXTLS_Dynamic', 1)
CXTLS_Static = enum_CXTLSKind.define('CXTLS_Static', 2)

@dll.bind
def clang_getCursorTLSKind(cursor:CXCursor) -> enum_CXTLSKind: ...
@dll.bind
def clang_Cursor_getTranslationUnit(_0:CXCursor) -> CXTranslationUnit: ...
class struct_CXCursorSetImpl(ctypes.Structure): pass
CXCursorSet: TypeAlias = c.POINTER[struct_CXCursorSetImpl]
@dll.bind
def clang_createCXCursorSet() -> CXCursorSet: ...
@dll.bind
def clang_disposeCXCursorSet(cset:CXCursorSet) -> None: ...
@dll.bind
def clang_CXCursorSet_contains(cset:CXCursorSet, cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXCursorSet_insert(cset:CXCursorSet, cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorSemanticParent(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorLexicalParent(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getOverriddenCursors(cursor:CXCursor, overridden:c.POINTER[c.POINTER[CXCursor]], num_overridden:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_disposeOverriddenCursors(overridden:c.POINTER[CXCursor]) -> None: ...
@dll.bind
def clang_getIncludedFile(cursor:CXCursor) -> CXFile: ...
@dll.bind
def clang_getCursor(_0:CXTranslationUnit, _1:CXSourceLocation) -> CXCursor: ...
@dll.bind
def clang_getCursorLocation(_0:CXCursor) -> CXSourceLocation: ...
@dll.bind
def clang_getCursorExtent(_0:CXCursor) -> CXSourceRange: ...
class enum_CXTypeKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXType_Invalid = enum_CXTypeKind.define('CXType_Invalid', 0)
CXType_Unexposed = enum_CXTypeKind.define('CXType_Unexposed', 1)
CXType_Void = enum_CXTypeKind.define('CXType_Void', 2)
CXType_Bool = enum_CXTypeKind.define('CXType_Bool', 3)
CXType_Char_U = enum_CXTypeKind.define('CXType_Char_U', 4)
CXType_UChar = enum_CXTypeKind.define('CXType_UChar', 5)
CXType_Char16 = enum_CXTypeKind.define('CXType_Char16', 6)
CXType_Char32 = enum_CXTypeKind.define('CXType_Char32', 7)
CXType_UShort = enum_CXTypeKind.define('CXType_UShort', 8)
CXType_UInt = enum_CXTypeKind.define('CXType_UInt', 9)
CXType_ULong = enum_CXTypeKind.define('CXType_ULong', 10)
CXType_ULongLong = enum_CXTypeKind.define('CXType_ULongLong', 11)
CXType_UInt128 = enum_CXTypeKind.define('CXType_UInt128', 12)
CXType_Char_S = enum_CXTypeKind.define('CXType_Char_S', 13)
CXType_SChar = enum_CXTypeKind.define('CXType_SChar', 14)
CXType_WChar = enum_CXTypeKind.define('CXType_WChar', 15)
CXType_Short = enum_CXTypeKind.define('CXType_Short', 16)
CXType_Int = enum_CXTypeKind.define('CXType_Int', 17)
CXType_Long = enum_CXTypeKind.define('CXType_Long', 18)
CXType_LongLong = enum_CXTypeKind.define('CXType_LongLong', 19)
CXType_Int128 = enum_CXTypeKind.define('CXType_Int128', 20)
CXType_Float = enum_CXTypeKind.define('CXType_Float', 21)
CXType_Double = enum_CXTypeKind.define('CXType_Double', 22)
CXType_LongDouble = enum_CXTypeKind.define('CXType_LongDouble', 23)
CXType_NullPtr = enum_CXTypeKind.define('CXType_NullPtr', 24)
CXType_Overload = enum_CXTypeKind.define('CXType_Overload', 25)
CXType_Dependent = enum_CXTypeKind.define('CXType_Dependent', 26)
CXType_ObjCId = enum_CXTypeKind.define('CXType_ObjCId', 27)
CXType_ObjCClass = enum_CXTypeKind.define('CXType_ObjCClass', 28)
CXType_ObjCSel = enum_CXTypeKind.define('CXType_ObjCSel', 29)
CXType_Float128 = enum_CXTypeKind.define('CXType_Float128', 30)
CXType_Half = enum_CXTypeKind.define('CXType_Half', 31)
CXType_Float16 = enum_CXTypeKind.define('CXType_Float16', 32)
CXType_ShortAccum = enum_CXTypeKind.define('CXType_ShortAccum', 33)
CXType_Accum = enum_CXTypeKind.define('CXType_Accum', 34)
CXType_LongAccum = enum_CXTypeKind.define('CXType_LongAccum', 35)
CXType_UShortAccum = enum_CXTypeKind.define('CXType_UShortAccum', 36)
CXType_UAccum = enum_CXTypeKind.define('CXType_UAccum', 37)
CXType_ULongAccum = enum_CXTypeKind.define('CXType_ULongAccum', 38)
CXType_BFloat16 = enum_CXTypeKind.define('CXType_BFloat16', 39)
CXType_Ibm128 = enum_CXTypeKind.define('CXType_Ibm128', 40)
CXType_FirstBuiltin = enum_CXTypeKind.define('CXType_FirstBuiltin', 2)
CXType_LastBuiltin = enum_CXTypeKind.define('CXType_LastBuiltin', 40)
CXType_Complex = enum_CXTypeKind.define('CXType_Complex', 100)
CXType_Pointer = enum_CXTypeKind.define('CXType_Pointer', 101)
CXType_BlockPointer = enum_CXTypeKind.define('CXType_BlockPointer', 102)
CXType_LValueReference = enum_CXTypeKind.define('CXType_LValueReference', 103)
CXType_RValueReference = enum_CXTypeKind.define('CXType_RValueReference', 104)
CXType_Record = enum_CXTypeKind.define('CXType_Record', 105)
CXType_Enum = enum_CXTypeKind.define('CXType_Enum', 106)
CXType_Typedef = enum_CXTypeKind.define('CXType_Typedef', 107)
CXType_ObjCInterface = enum_CXTypeKind.define('CXType_ObjCInterface', 108)
CXType_ObjCObjectPointer = enum_CXTypeKind.define('CXType_ObjCObjectPointer', 109)
CXType_FunctionNoProto = enum_CXTypeKind.define('CXType_FunctionNoProto', 110)
CXType_FunctionProto = enum_CXTypeKind.define('CXType_FunctionProto', 111)
CXType_ConstantArray = enum_CXTypeKind.define('CXType_ConstantArray', 112)
CXType_Vector = enum_CXTypeKind.define('CXType_Vector', 113)
CXType_IncompleteArray = enum_CXTypeKind.define('CXType_IncompleteArray', 114)
CXType_VariableArray = enum_CXTypeKind.define('CXType_VariableArray', 115)
CXType_DependentSizedArray = enum_CXTypeKind.define('CXType_DependentSizedArray', 116)
CXType_MemberPointer = enum_CXTypeKind.define('CXType_MemberPointer', 117)
CXType_Auto = enum_CXTypeKind.define('CXType_Auto', 118)
CXType_Elaborated = enum_CXTypeKind.define('CXType_Elaborated', 119)
CXType_Pipe = enum_CXTypeKind.define('CXType_Pipe', 120)
CXType_OCLImage1dRO = enum_CXTypeKind.define('CXType_OCLImage1dRO', 121)
CXType_OCLImage1dArrayRO = enum_CXTypeKind.define('CXType_OCLImage1dArrayRO', 122)
CXType_OCLImage1dBufferRO = enum_CXTypeKind.define('CXType_OCLImage1dBufferRO', 123)
CXType_OCLImage2dRO = enum_CXTypeKind.define('CXType_OCLImage2dRO', 124)
CXType_OCLImage2dArrayRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayRO', 125)
CXType_OCLImage2dDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dDepthRO', 126)
CXType_OCLImage2dArrayDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRO', 127)
CXType_OCLImage2dMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dMSAARO', 128)
CXType_OCLImage2dArrayMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARO', 129)
CXType_OCLImage2dMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRO', 130)
CXType_OCLImage2dArrayMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRO', 131)
CXType_OCLImage3dRO = enum_CXTypeKind.define('CXType_OCLImage3dRO', 132)
CXType_OCLImage1dWO = enum_CXTypeKind.define('CXType_OCLImage1dWO', 133)
CXType_OCLImage1dArrayWO = enum_CXTypeKind.define('CXType_OCLImage1dArrayWO', 134)
CXType_OCLImage1dBufferWO = enum_CXTypeKind.define('CXType_OCLImage1dBufferWO', 135)
CXType_OCLImage2dWO = enum_CXTypeKind.define('CXType_OCLImage2dWO', 136)
CXType_OCLImage2dArrayWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayWO', 137)
CXType_OCLImage2dDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dDepthWO', 138)
CXType_OCLImage2dArrayDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthWO', 139)
CXType_OCLImage2dMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAAWO', 140)
CXType_OCLImage2dArrayMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAAWO', 141)
CXType_OCLImage2dMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthWO', 142)
CXType_OCLImage2dArrayMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthWO', 143)
CXType_OCLImage3dWO = enum_CXTypeKind.define('CXType_OCLImage3dWO', 144)
CXType_OCLImage1dRW = enum_CXTypeKind.define('CXType_OCLImage1dRW', 145)
CXType_OCLImage1dArrayRW = enum_CXTypeKind.define('CXType_OCLImage1dArrayRW', 146)
CXType_OCLImage1dBufferRW = enum_CXTypeKind.define('CXType_OCLImage1dBufferRW', 147)
CXType_OCLImage2dRW = enum_CXTypeKind.define('CXType_OCLImage2dRW', 148)
CXType_OCLImage2dArrayRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayRW', 149)
CXType_OCLImage2dDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dDepthRW', 150)
CXType_OCLImage2dArrayDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRW', 151)
CXType_OCLImage2dMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dMSAARW', 152)
CXType_OCLImage2dArrayMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARW', 153)
CXType_OCLImage2dMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRW', 154)
CXType_OCLImage2dArrayMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRW', 155)
CXType_OCLImage3dRW = enum_CXTypeKind.define('CXType_OCLImage3dRW', 156)
CXType_OCLSampler = enum_CXTypeKind.define('CXType_OCLSampler', 157)
CXType_OCLEvent = enum_CXTypeKind.define('CXType_OCLEvent', 158)
CXType_OCLQueue = enum_CXTypeKind.define('CXType_OCLQueue', 159)
CXType_OCLReserveID = enum_CXTypeKind.define('CXType_OCLReserveID', 160)
CXType_ObjCObject = enum_CXTypeKind.define('CXType_ObjCObject', 161)
CXType_ObjCTypeParam = enum_CXTypeKind.define('CXType_ObjCTypeParam', 162)
CXType_Attributed = enum_CXTypeKind.define('CXType_Attributed', 163)
CXType_OCLIntelSubgroupAVCMcePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMcePayload', 164)
CXType_OCLIntelSubgroupAVCImePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImePayload', 165)
CXType_OCLIntelSubgroupAVCRefPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefPayload', 166)
CXType_OCLIntelSubgroupAVCSicPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicPayload', 167)
CXType_OCLIntelSubgroupAVCMceResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMceResult', 168)
CXType_OCLIntelSubgroupAVCImeResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResult', 169)
CXType_OCLIntelSubgroupAVCRefResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefResult', 170)
CXType_OCLIntelSubgroupAVCSicResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicResult', 171)
CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout', 172)
CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout', 173)
CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin', 174)
CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin', 175)
CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout', 172)
CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout', 173)
CXType_OCLIntelSubgroupAVCImeSingleRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleRefStreamin', 174)
CXType_OCLIntelSubgroupAVCImeDualRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualRefStreamin', 175)
CXType_ExtVector = enum_CXTypeKind.define('CXType_ExtVector', 176)
CXType_Atomic = enum_CXTypeKind.define('CXType_Atomic', 177)
CXType_BTFTagAttributed = enum_CXTypeKind.define('CXType_BTFTagAttributed', 178)
CXType_HLSLResource = enum_CXTypeKind.define('CXType_HLSLResource', 179)
CXType_HLSLAttributedResource = enum_CXTypeKind.define('CXType_HLSLAttributedResource', 180)

class enum_CXCallingConv(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCallingConv_Default = enum_CXCallingConv.define('CXCallingConv_Default', 0)
CXCallingConv_C = enum_CXCallingConv.define('CXCallingConv_C', 1)
CXCallingConv_X86StdCall = enum_CXCallingConv.define('CXCallingConv_X86StdCall', 2)
CXCallingConv_X86FastCall = enum_CXCallingConv.define('CXCallingConv_X86FastCall', 3)
CXCallingConv_X86ThisCall = enum_CXCallingConv.define('CXCallingConv_X86ThisCall', 4)
CXCallingConv_X86Pascal = enum_CXCallingConv.define('CXCallingConv_X86Pascal', 5)
CXCallingConv_AAPCS = enum_CXCallingConv.define('CXCallingConv_AAPCS', 6)
CXCallingConv_AAPCS_VFP = enum_CXCallingConv.define('CXCallingConv_AAPCS_VFP', 7)
CXCallingConv_X86RegCall = enum_CXCallingConv.define('CXCallingConv_X86RegCall', 8)
CXCallingConv_IntelOclBicc = enum_CXCallingConv.define('CXCallingConv_IntelOclBicc', 9)
CXCallingConv_Win64 = enum_CXCallingConv.define('CXCallingConv_Win64', 10)
CXCallingConv_X86_64Win64 = enum_CXCallingConv.define('CXCallingConv_X86_64Win64', 10)
CXCallingConv_X86_64SysV = enum_CXCallingConv.define('CXCallingConv_X86_64SysV', 11)
CXCallingConv_X86VectorCall = enum_CXCallingConv.define('CXCallingConv_X86VectorCall', 12)
CXCallingConv_Swift = enum_CXCallingConv.define('CXCallingConv_Swift', 13)
CXCallingConv_PreserveMost = enum_CXCallingConv.define('CXCallingConv_PreserveMost', 14)
CXCallingConv_PreserveAll = enum_CXCallingConv.define('CXCallingConv_PreserveAll', 15)
CXCallingConv_AArch64VectorCall = enum_CXCallingConv.define('CXCallingConv_AArch64VectorCall', 16)
CXCallingConv_SwiftAsync = enum_CXCallingConv.define('CXCallingConv_SwiftAsync', 17)
CXCallingConv_AArch64SVEPCS = enum_CXCallingConv.define('CXCallingConv_AArch64SVEPCS', 18)
CXCallingConv_M68kRTD = enum_CXCallingConv.define('CXCallingConv_M68kRTD', 19)
CXCallingConv_PreserveNone = enum_CXCallingConv.define('CXCallingConv_PreserveNone', 20)
CXCallingConv_RISCVVectorCall = enum_CXCallingConv.define('CXCallingConv_RISCVVectorCall', 21)
CXCallingConv_Invalid = enum_CXCallingConv.define('CXCallingConv_Invalid', 100)
CXCallingConv_Unexposed = enum_CXCallingConv.define('CXCallingConv_Unexposed', 200)

@c.record
class CXType(c.Struct):
  SIZE = 24
  kind: Annotated[enum_CXTypeKind, 0]
  data: Annotated[c.Array[ctypes.c_void_p, Literal[2]], 8]
@dll.bind
def clang_getCursorType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getTypeSpelling(CT:CXType) -> CXString: ...
@dll.bind
def clang_getTypedefDeclUnderlyingType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getEnumDeclIntegerType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getEnumConstantDeclValue(C:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_getEnumConstantDeclUnsignedValue(C:CXCursor) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_Cursor_isBitField(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getFieldDeclBitWidth(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getNumArguments(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getArgument(C:CXCursor, i:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
class enum_CXTemplateArgumentKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXTemplateArgumentKind_Null = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Null', 0)
CXTemplateArgumentKind_Type = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Type', 1)
CXTemplateArgumentKind_Declaration = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Declaration', 2)
CXTemplateArgumentKind_NullPtr = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_NullPtr', 3)
CXTemplateArgumentKind_Integral = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Integral', 4)
CXTemplateArgumentKind_Template = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Template', 5)
CXTemplateArgumentKind_TemplateExpansion = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_TemplateExpansion', 6)
CXTemplateArgumentKind_Expression = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Expression', 7)
CXTemplateArgumentKind_Pack = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Pack', 8)
CXTemplateArgumentKind_Invalid = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Invalid', 9)

@dll.bind
def clang_Cursor_getNumTemplateArguments(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getTemplateArgumentKind(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> enum_CXTemplateArgumentKind: ...
@dll.bind
def clang_Cursor_getTemplateArgumentType(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Cursor_getTemplateArgumentValue(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Cursor_getTemplateArgumentUnsignedValue(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_equalTypes(A:CXType, B:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCanonicalType(T:CXType) -> CXType: ...
@dll.bind
def clang_isConstQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isMacroFunctionLike(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isMacroBuiltin(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isFunctionInlined(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isVolatileQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isRestrictQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getAddressSpace(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getTypedefName(CT:CXType) -> CXString: ...
@dll.bind
def clang_getPointeeType(T:CXType) -> CXType: ...
@dll.bind
def clang_getUnqualifiedType(CT:CXType) -> CXType: ...
@dll.bind
def clang_getNonReferenceType(CT:CXType) -> CXType: ...
@dll.bind
def clang_getTypeDeclaration(T:CXType) -> CXCursor: ...
@dll.bind
def clang_getDeclObjCTypeEncoding(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Type_getObjCEncoding(type:CXType) -> CXString: ...
@dll.bind
def clang_getTypeKindSpelling(K:enum_CXTypeKind) -> CXString: ...
@dll.bind
def clang_getFunctionTypeCallingConv(T:CXType) -> enum_CXCallingConv: ...
@dll.bind
def clang_getResultType(T:CXType) -> CXType: ...
@dll.bind
def clang_getExceptionSpecificationType(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getNumArgTypes(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getArgType(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Type_getObjCObjectBaseType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getNumObjCProtocolRefs(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Type_getObjCProtocolDecl(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
@dll.bind
def clang_Type_getNumObjCTypeArgs(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Type_getObjCTypeArg(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_isFunctionTypeVariadic(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorResultType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getCursorExceptionSpecificationType(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_isPODType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getElementType(T:CXType) -> CXType: ...
@dll.bind
def clang_getNumElements(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_getArrayElementType(T:CXType) -> CXType: ...
@dll.bind
def clang_getArraySize(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getNamedType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_isTransparentTagTypedef(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
class enum_CXTypeNullabilityKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXTypeNullability_NonNull = enum_CXTypeNullabilityKind.define('CXTypeNullability_NonNull', 0)
CXTypeNullability_Nullable = enum_CXTypeNullabilityKind.define('CXTypeNullability_Nullable', 1)
CXTypeNullability_Unspecified = enum_CXTypeNullabilityKind.define('CXTypeNullability_Unspecified', 2)
CXTypeNullability_Invalid = enum_CXTypeNullabilityKind.define('CXTypeNullability_Invalid', 3)
CXTypeNullability_NullableResult = enum_CXTypeNullabilityKind.define('CXTypeNullability_NullableResult', 4)

@dll.bind
def clang_Type_getNullability(T:CXType) -> enum_CXTypeNullabilityKind: ...
class enum_CXTypeLayoutError(Annotated[int, ctypes.c_int32], c.Enum): pass
CXTypeLayoutError_Invalid = enum_CXTypeLayoutError.define('CXTypeLayoutError_Invalid', -1)
CXTypeLayoutError_Incomplete = enum_CXTypeLayoutError.define('CXTypeLayoutError_Incomplete', -2)
CXTypeLayoutError_Dependent = enum_CXTypeLayoutError.define('CXTypeLayoutError_Dependent', -3)
CXTypeLayoutError_NotConstantSize = enum_CXTypeLayoutError.define('CXTypeLayoutError_NotConstantSize', -4)
CXTypeLayoutError_InvalidFieldName = enum_CXTypeLayoutError.define('CXTypeLayoutError_InvalidFieldName', -5)
CXTypeLayoutError_Undeduced = enum_CXTypeLayoutError.define('CXTypeLayoutError_Undeduced', -6)

@dll.bind
def clang_Type_getAlignOf(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getClassType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getSizeOf(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getOffsetOf(T:CXType, S:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getModifiedType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getValueType(CT:CXType) -> CXType: ...
@dll.bind
def clang_Cursor_getOffsetOfField(C:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Cursor_isAnonymous(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isAnonymousRecordDecl(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isInlineNamespace(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
class enum_CXRefQualifierKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXRefQualifier_None = enum_CXRefQualifierKind.define('CXRefQualifier_None', 0)
CXRefQualifier_LValue = enum_CXRefQualifierKind.define('CXRefQualifier_LValue', 1)
CXRefQualifier_RValue = enum_CXRefQualifierKind.define('CXRefQualifier_RValue', 2)

@dll.bind
def clang_Type_getNumTemplateArguments(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Type_getTemplateArgumentAsType(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Type_getCXXRefQualifier(T:CXType) -> enum_CXRefQualifierKind: ...
@dll.bind
def clang_isVirtualBase(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getOffsetOfBase(Parent:CXCursor, Base:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
class enum_CX_CXXAccessSpecifier(Annotated[int, ctypes.c_uint32], c.Enum): pass
CX_CXXInvalidAccessSpecifier = enum_CX_CXXAccessSpecifier.define('CX_CXXInvalidAccessSpecifier', 0)
CX_CXXPublic = enum_CX_CXXAccessSpecifier.define('CX_CXXPublic', 1)
CX_CXXProtected = enum_CX_CXXAccessSpecifier.define('CX_CXXProtected', 2)
CX_CXXPrivate = enum_CX_CXXAccessSpecifier.define('CX_CXXPrivate', 3)

@dll.bind
def clang_getCXXAccessSpecifier(_0:CXCursor) -> enum_CX_CXXAccessSpecifier: ...
class enum_CX_StorageClass(Annotated[int, ctypes.c_uint32], c.Enum): pass
CX_SC_Invalid = enum_CX_StorageClass.define('CX_SC_Invalid', 0)
CX_SC_None = enum_CX_StorageClass.define('CX_SC_None', 1)
CX_SC_Extern = enum_CX_StorageClass.define('CX_SC_Extern', 2)
CX_SC_Static = enum_CX_StorageClass.define('CX_SC_Static', 3)
CX_SC_PrivateExtern = enum_CX_StorageClass.define('CX_SC_PrivateExtern', 4)
CX_SC_OpenCLWorkGroupLocal = enum_CX_StorageClass.define('CX_SC_OpenCLWorkGroupLocal', 5)
CX_SC_Auto = enum_CX_StorageClass.define('CX_SC_Auto', 6)
CX_SC_Register = enum_CX_StorageClass.define('CX_SC_Register', 7)

class enum_CX_BinaryOperatorKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CX_BO_Invalid = enum_CX_BinaryOperatorKind.define('CX_BO_Invalid', 0)
CX_BO_PtrMemD = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemD', 1)
CX_BO_PtrMemI = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemI', 2)
CX_BO_Mul = enum_CX_BinaryOperatorKind.define('CX_BO_Mul', 3)
CX_BO_Div = enum_CX_BinaryOperatorKind.define('CX_BO_Div', 4)
CX_BO_Rem = enum_CX_BinaryOperatorKind.define('CX_BO_Rem', 5)
CX_BO_Add = enum_CX_BinaryOperatorKind.define('CX_BO_Add', 6)
CX_BO_Sub = enum_CX_BinaryOperatorKind.define('CX_BO_Sub', 7)
CX_BO_Shl = enum_CX_BinaryOperatorKind.define('CX_BO_Shl', 8)
CX_BO_Shr = enum_CX_BinaryOperatorKind.define('CX_BO_Shr', 9)
CX_BO_Cmp = enum_CX_BinaryOperatorKind.define('CX_BO_Cmp', 10)
CX_BO_LT = enum_CX_BinaryOperatorKind.define('CX_BO_LT', 11)
CX_BO_GT = enum_CX_BinaryOperatorKind.define('CX_BO_GT', 12)
CX_BO_LE = enum_CX_BinaryOperatorKind.define('CX_BO_LE', 13)
CX_BO_GE = enum_CX_BinaryOperatorKind.define('CX_BO_GE', 14)
CX_BO_EQ = enum_CX_BinaryOperatorKind.define('CX_BO_EQ', 15)
CX_BO_NE = enum_CX_BinaryOperatorKind.define('CX_BO_NE', 16)
CX_BO_And = enum_CX_BinaryOperatorKind.define('CX_BO_And', 17)
CX_BO_Xor = enum_CX_BinaryOperatorKind.define('CX_BO_Xor', 18)
CX_BO_Or = enum_CX_BinaryOperatorKind.define('CX_BO_Or', 19)
CX_BO_LAnd = enum_CX_BinaryOperatorKind.define('CX_BO_LAnd', 20)
CX_BO_LOr = enum_CX_BinaryOperatorKind.define('CX_BO_LOr', 21)
CX_BO_Assign = enum_CX_BinaryOperatorKind.define('CX_BO_Assign', 22)
CX_BO_MulAssign = enum_CX_BinaryOperatorKind.define('CX_BO_MulAssign', 23)
CX_BO_DivAssign = enum_CX_BinaryOperatorKind.define('CX_BO_DivAssign', 24)
CX_BO_RemAssign = enum_CX_BinaryOperatorKind.define('CX_BO_RemAssign', 25)
CX_BO_AddAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AddAssign', 26)
CX_BO_SubAssign = enum_CX_BinaryOperatorKind.define('CX_BO_SubAssign', 27)
CX_BO_ShlAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShlAssign', 28)
CX_BO_ShrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShrAssign', 29)
CX_BO_AndAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AndAssign', 30)
CX_BO_XorAssign = enum_CX_BinaryOperatorKind.define('CX_BO_XorAssign', 31)
CX_BO_OrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_OrAssign', 32)
CX_BO_Comma = enum_CX_BinaryOperatorKind.define('CX_BO_Comma', 33)
CX_BO_LAST = enum_CX_BinaryOperatorKind.define('CX_BO_LAST', 33)

@dll.bind
def clang_Cursor_getBinaryOpcode(C:CXCursor) -> enum_CX_BinaryOperatorKind: ...
@dll.bind
def clang_Cursor_getBinaryOpcodeStr(Op:enum_CX_BinaryOperatorKind) -> CXString: ...
@dll.bind
def clang_Cursor_getStorageClass(_0:CXCursor) -> enum_CX_StorageClass: ...
@dll.bind
def clang_getNumOverloadedDecls(cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getOverloadedDecl(cursor:CXCursor, index:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
@dll.bind
def clang_getIBOutletCollectionType(_0:CXCursor) -> CXType: ...
class enum_CXChildVisitResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXChildVisit_Break = enum_CXChildVisitResult.define('CXChildVisit_Break', 0)
CXChildVisit_Continue = enum_CXChildVisitResult.define('CXChildVisit_Continue', 1)
CXChildVisit_Recurse = enum_CXChildVisitResult.define('CXChildVisit_Recurse', 2)

CXCursorVisitor: TypeAlias = c.CFUNCTYPE[enum_CXChildVisitResult, [CXCursor, CXCursor, ctypes.c_void_p]]
@dll.bind
def clang_visitChildren(parent:CXCursor, visitor:CXCursorVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
class struct__CXChildVisitResult(ctypes.Structure): pass
CXCursorVisitorBlock: TypeAlias = c.POINTER[struct__CXChildVisitResult]
@dll.bind
def clang_visitChildrenWithBlock(parent:CXCursor, block:CXCursorVisitorBlock) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorUSR(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCClass(class_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCCategory(class_name:c.POINTER[Annotated[bytes, ctypes.c_char]], category_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCProtocol(protocol_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCIvar(name:c.POINTER[Annotated[bytes, ctypes.c_char]], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCMethod(name:c.POINTER[Annotated[bytes, ctypes.c_char]], isInstanceMethod:Annotated[int, ctypes.c_uint32], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCProperty(property:c.POINTER[Annotated[bytes, ctypes.c_char]], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_getCursorSpelling(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getSpellingNameRange(_0:CXCursor, pieceIndex:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> CXSourceRange: ...
CXPrintingPolicy: TypeAlias = ctypes.c_void_p
class enum_CXPrintingPolicyProperty(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXPrintingPolicy_Indentation = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Indentation', 0)
CXPrintingPolicy_SuppressSpecifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressSpecifiers', 1)
CXPrintingPolicy_SuppressTagKeyword = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTagKeyword', 2)
CXPrintingPolicy_IncludeTagDefinition = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeTagDefinition', 3)
CXPrintingPolicy_SuppressScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressScope', 4)
CXPrintingPolicy_SuppressUnwrittenScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressUnwrittenScope', 5)
CXPrintingPolicy_SuppressInitializers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressInitializers', 6)
CXPrintingPolicy_ConstantArraySizeAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantArraySizeAsWritten', 7)
CXPrintingPolicy_AnonymousTagLocations = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_AnonymousTagLocations', 8)
CXPrintingPolicy_SuppressStrongLifetime = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressStrongLifetime', 9)
CXPrintingPolicy_SuppressLifetimeQualifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressLifetimeQualifiers', 10)
CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors', 11)
CXPrintingPolicy_Bool = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Bool', 12)
CXPrintingPolicy_Restrict = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Restrict', 13)
CXPrintingPolicy_Alignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Alignof', 14)
CXPrintingPolicy_UnderscoreAlignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UnderscoreAlignof', 15)
CXPrintingPolicy_UseVoidForZeroParams = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UseVoidForZeroParams', 16)
CXPrintingPolicy_TerseOutput = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_TerseOutput', 17)
CXPrintingPolicy_PolishForDeclaration = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_PolishForDeclaration', 18)
CXPrintingPolicy_Half = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Half', 19)
CXPrintingPolicy_MSWChar = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSWChar', 20)
CXPrintingPolicy_IncludeNewlines = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeNewlines', 21)
CXPrintingPolicy_MSVCFormatting = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSVCFormatting', 22)
CXPrintingPolicy_ConstantsAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantsAsWritten', 23)
CXPrintingPolicy_SuppressImplicitBase = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressImplicitBase', 24)
CXPrintingPolicy_FullyQualifiedName = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_FullyQualifiedName', 25)
CXPrintingPolicy_LastProperty = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_LastProperty', 25)

@dll.bind
def clang_PrintingPolicy_getProperty(Policy:CXPrintingPolicy, Property:enum_CXPrintingPolicyProperty) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_PrintingPolicy_setProperty(Policy:CXPrintingPolicy, Property:enum_CXPrintingPolicyProperty, Value:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_getCursorPrintingPolicy(_0:CXCursor) -> CXPrintingPolicy: ...
@dll.bind
def clang_PrintingPolicy_dispose(Policy:CXPrintingPolicy) -> None: ...
@dll.bind
def clang_getCursorPrettyPrinted(Cursor:CXCursor, Policy:CXPrintingPolicy) -> CXString: ...
@dll.bind
def clang_getTypePrettyPrinted(CT:CXType, cxPolicy:CXPrintingPolicy) -> CXString: ...
@dll.bind
def clang_getCursorDisplayName(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_getCursorReferenced(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorDefinition(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_isCursorDefinition(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCanonicalCursor(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_Cursor_getObjCSelectorIndex(_0:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_isDynamicCall(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getReceiverType(C:CXCursor) -> CXType: ...
class CXObjCPropertyAttrKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXObjCPropertyAttr_noattr = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_noattr', 0)
CXObjCPropertyAttr_readonly = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readonly', 1)
CXObjCPropertyAttr_getter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_getter', 2)
CXObjCPropertyAttr_assign = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_assign', 4)
CXObjCPropertyAttr_readwrite = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readwrite', 8)
CXObjCPropertyAttr_retain = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_retain', 16)
CXObjCPropertyAttr_copy = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_copy', 32)
CXObjCPropertyAttr_nonatomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_nonatomic', 64)
CXObjCPropertyAttr_setter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_setter', 128)
CXObjCPropertyAttr_atomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_atomic', 256)
CXObjCPropertyAttr_weak = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_weak', 512)
CXObjCPropertyAttr_strong = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_strong', 1024)
CXObjCPropertyAttr_unsafe_unretained = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_unsafe_unretained', 2048)
CXObjCPropertyAttr_class = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_class', 4096)

@dll.bind
def clang_Cursor_getObjCPropertyAttributes(C:CXCursor, reserved:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_getObjCPropertyGetterName(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getObjCPropertySetterName(C:CXCursor) -> CXString: ...
class CXObjCDeclQualifierKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXObjCDeclQualifier_None = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_None', 0)
CXObjCDeclQualifier_In = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_In', 1)
CXObjCDeclQualifier_Inout = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Inout', 2)
CXObjCDeclQualifier_Out = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Out', 4)
CXObjCDeclQualifier_Bycopy = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Bycopy', 8)
CXObjCDeclQualifier_Byref = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Byref', 16)
CXObjCDeclQualifier_Oneway = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Oneway', 32)

@dll.bind
def clang_Cursor_getObjCDeclQualifiers(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isObjCOptional(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isVariadic(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isExternalSymbol(C:CXCursor, language:c.POINTER[CXString], definedIn:c.POINTER[CXString], isGenerated:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_getCommentRange(C:CXCursor) -> CXSourceRange: ...
@dll.bind
def clang_Cursor_getRawCommentText(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getBriefCommentText(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getMangling(_0:CXCursor) -> CXString: ...
@c.record
class CXStringSet(c.Struct):
  SIZE = 16
  Strings: Annotated[c.POINTER[CXString], 0]
  Count: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_Cursor_getCXXManglings(_0:CXCursor) -> c.POINTER[CXStringSet]: ...
@dll.bind
def clang_Cursor_getObjCManglings(_0:CXCursor) -> c.POINTER[CXStringSet]: ...
CXModule: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_Cursor_getModule(C:CXCursor) -> CXModule: ...
@dll.bind
def clang_getModuleForFile(_0:CXTranslationUnit, _1:CXFile) -> CXModule: ...
@dll.bind
def clang_Module_getASTFile(Module:CXModule) -> CXFile: ...
@dll.bind
def clang_Module_getParent(Module:CXModule) -> CXModule: ...
@dll.bind
def clang_Module_getName(Module:CXModule) -> CXString: ...
@dll.bind
def clang_Module_getFullName(Module:CXModule) -> CXString: ...
@dll.bind
def clang_Module_isSystem(Module:CXModule) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Module_getNumTopLevelHeaders(_0:CXTranslationUnit, Module:CXModule) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Module_getTopLevelHeader(_0:CXTranslationUnit, Module:CXModule, Index:Annotated[int, ctypes.c_uint32]) -> CXFile: ...
@dll.bind
def clang_CXXConstructor_isConvertingConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isCopyConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isDefaultConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isMoveConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXField_isMutable(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isDefaulted(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isDeleted(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isPureVirtual(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isStatic(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isVirtual(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isCopyAssignmentOperator(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isMoveAssignmentOperator(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isExplicit(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXRecord_isAbstract(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_EnumDecl_isScoped(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isConst(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getTemplateCursorKind(C:CXCursor) -> enum_CXCursorKind: ...
@dll.bind
def clang_getSpecializedCursorTemplate(C:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorReferenceNameRange(C:CXCursor, NameFlags:Annotated[int, ctypes.c_uint32], PieceIndex:Annotated[int, ctypes.c_uint32]) -> CXSourceRange: ...
class enum_CXNameRefFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXNameRange_WantQualifier = enum_CXNameRefFlags.define('CXNameRange_WantQualifier', 1)
CXNameRange_WantTemplateArgs = enum_CXNameRefFlags.define('CXNameRange_WantTemplateArgs', 2)
CXNameRange_WantSinglePiece = enum_CXNameRefFlags.define('CXNameRange_WantSinglePiece', 4)

class enum_CXTokenKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXToken_Punctuation = enum_CXTokenKind.define('CXToken_Punctuation', 0)
CXToken_Keyword = enum_CXTokenKind.define('CXToken_Keyword', 1)
CXToken_Identifier = enum_CXTokenKind.define('CXToken_Identifier', 2)
CXToken_Literal = enum_CXTokenKind.define('CXToken_Literal', 3)
CXToken_Comment = enum_CXTokenKind.define('CXToken_Comment', 4)

CXTokenKind: TypeAlias = enum_CXTokenKind
@c.record
class CXToken(c.Struct):
  SIZE = 24
  int_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 0]
  ptr_data: Annotated[ctypes.c_void_p, 16]
@dll.bind
def clang_getToken(TU:CXTranslationUnit, Location:CXSourceLocation) -> c.POINTER[CXToken]: ...
@dll.bind
def clang_getTokenKind(_0:CXToken) -> CXTokenKind: ...
@dll.bind
def clang_getTokenSpelling(_0:CXTranslationUnit, _1:CXToken) -> CXString: ...
@dll.bind
def clang_getTokenLocation(_0:CXTranslationUnit, _1:CXToken) -> CXSourceLocation: ...
@dll.bind
def clang_getTokenExtent(_0:CXTranslationUnit, _1:CXToken) -> CXSourceRange: ...
@dll.bind
def clang_tokenize(TU:CXTranslationUnit, Range:CXSourceRange, Tokens:c.POINTER[c.POINTER[CXToken]], NumTokens:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_annotateTokens(TU:CXTranslationUnit, Tokens:c.POINTER[CXToken], NumTokens:Annotated[int, ctypes.c_uint32], Cursors:c.POINTER[CXCursor]) -> None: ...
@dll.bind
def clang_disposeTokens(TU:CXTranslationUnit, Tokens:c.POINTER[CXToken], NumTokens:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_getCursorKindSpelling(Kind:enum_CXCursorKind) -> CXString: ...
@dll.bind
def clang_getDefinitionSpellingAndExtent(_0:CXCursor, startBuf:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], endBuf:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], startLine:c.POINTER[Annotated[int, ctypes.c_uint32]], startColumn:c.POINTER[Annotated[int, ctypes.c_uint32]], endLine:c.POINTER[Annotated[int, ctypes.c_uint32]], endColumn:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_enableStackTraces() -> None: ...
@dll.bind
def clang_executeOnThread(fn:c.CFUNCTYPE[None, [ctypes.c_void_p]], user_data:ctypes.c_void_p, stack_size:Annotated[int, ctypes.c_uint32]) -> None: ...
CXCompletionString: TypeAlias = ctypes.c_void_p
@c.record
class CXCompletionResult(c.Struct):
  SIZE = 16
  CursorKind: Annotated[enum_CXCursorKind, 0]
  CompletionString: Annotated[CXCompletionString, 8]
class enum_CXCompletionChunkKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCompletionChunk_Optional = enum_CXCompletionChunkKind.define('CXCompletionChunk_Optional', 0)
CXCompletionChunk_TypedText = enum_CXCompletionChunkKind.define('CXCompletionChunk_TypedText', 1)
CXCompletionChunk_Text = enum_CXCompletionChunkKind.define('CXCompletionChunk_Text', 2)
CXCompletionChunk_Placeholder = enum_CXCompletionChunkKind.define('CXCompletionChunk_Placeholder', 3)
CXCompletionChunk_Informative = enum_CXCompletionChunkKind.define('CXCompletionChunk_Informative', 4)
CXCompletionChunk_CurrentParameter = enum_CXCompletionChunkKind.define('CXCompletionChunk_CurrentParameter', 5)
CXCompletionChunk_LeftParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftParen', 6)
CXCompletionChunk_RightParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightParen', 7)
CXCompletionChunk_LeftBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBracket', 8)
CXCompletionChunk_RightBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBracket', 9)
CXCompletionChunk_LeftBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBrace', 10)
CXCompletionChunk_RightBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBrace', 11)
CXCompletionChunk_LeftAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftAngle', 12)
CXCompletionChunk_RightAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightAngle', 13)
CXCompletionChunk_Comma = enum_CXCompletionChunkKind.define('CXCompletionChunk_Comma', 14)
CXCompletionChunk_ResultType = enum_CXCompletionChunkKind.define('CXCompletionChunk_ResultType', 15)
CXCompletionChunk_Colon = enum_CXCompletionChunkKind.define('CXCompletionChunk_Colon', 16)
CXCompletionChunk_SemiColon = enum_CXCompletionChunkKind.define('CXCompletionChunk_SemiColon', 17)
CXCompletionChunk_Equal = enum_CXCompletionChunkKind.define('CXCompletionChunk_Equal', 18)
CXCompletionChunk_HorizontalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_HorizontalSpace', 19)
CXCompletionChunk_VerticalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_VerticalSpace', 20)

@dll.bind
def clang_getCompletionChunkKind(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> enum_CXCompletionChunkKind: ...
@dll.bind
def clang_getCompletionChunkText(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> CXString: ...
@dll.bind
def clang_getCompletionChunkCompletionString(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> CXCompletionString: ...
@dll.bind
def clang_getNumCompletionChunks(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionPriority(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionAvailability(completion_string:CXCompletionString) -> enum_CXAvailabilityKind: ...
@dll.bind
def clang_getCompletionNumAnnotations(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionAnnotation(completion_string:CXCompletionString, annotation_number:Annotated[int, ctypes.c_uint32]) -> CXString: ...
@dll.bind
def clang_getCompletionParent(completion_string:CXCompletionString, kind:c.POINTER[enum_CXCursorKind]) -> CXString: ...
@dll.bind
def clang_getCompletionBriefComment(completion_string:CXCompletionString) -> CXString: ...
@dll.bind
def clang_getCursorCompletionString(cursor:CXCursor) -> CXCompletionString: ...
@c.record
class CXCodeCompleteResults(c.Struct):
  SIZE = 16
  Results: Annotated[c.POINTER[CXCompletionResult], 0]
  NumResults: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_getCompletionNumFixIts(results:c.POINTER[CXCodeCompleteResults], completion_index:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionFixIt(results:c.POINTER[CXCodeCompleteResults], completion_index:Annotated[int, ctypes.c_uint32], fixit_index:Annotated[int, ctypes.c_uint32], replacement_range:c.POINTER[CXSourceRange]) -> CXString: ...
class enum_CXCodeComplete_Flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCodeComplete_IncludeMacros = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeMacros', 1)
CXCodeComplete_IncludeCodePatterns = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCodePatterns', 2)
CXCodeComplete_IncludeBriefComments = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeBriefComments', 4)
CXCodeComplete_SkipPreamble = enum_CXCodeComplete_Flags.define('CXCodeComplete_SkipPreamble', 8)
CXCodeComplete_IncludeCompletionsWithFixIts = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCompletionsWithFixIts', 16)

class enum_CXCompletionContext(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXCompletionContext_Unexposed = enum_CXCompletionContext.define('CXCompletionContext_Unexposed', 0)
CXCompletionContext_AnyType = enum_CXCompletionContext.define('CXCompletionContext_AnyType', 1)
CXCompletionContext_AnyValue = enum_CXCompletionContext.define('CXCompletionContext_AnyValue', 2)
CXCompletionContext_ObjCObjectValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCObjectValue', 4)
CXCompletionContext_ObjCSelectorValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorValue', 8)
CXCompletionContext_CXXClassTypeValue = enum_CXCompletionContext.define('CXCompletionContext_CXXClassTypeValue', 16)
CXCompletionContext_DotMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_DotMemberAccess', 32)
CXCompletionContext_ArrowMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_ArrowMemberAccess', 64)
CXCompletionContext_ObjCPropertyAccess = enum_CXCompletionContext.define('CXCompletionContext_ObjCPropertyAccess', 128)
CXCompletionContext_EnumTag = enum_CXCompletionContext.define('CXCompletionContext_EnumTag', 256)
CXCompletionContext_UnionTag = enum_CXCompletionContext.define('CXCompletionContext_UnionTag', 512)
CXCompletionContext_StructTag = enum_CXCompletionContext.define('CXCompletionContext_StructTag', 1024)
CXCompletionContext_ClassTag = enum_CXCompletionContext.define('CXCompletionContext_ClassTag', 2048)
CXCompletionContext_Namespace = enum_CXCompletionContext.define('CXCompletionContext_Namespace', 4096)
CXCompletionContext_NestedNameSpecifier = enum_CXCompletionContext.define('CXCompletionContext_NestedNameSpecifier', 8192)
CXCompletionContext_ObjCInterface = enum_CXCompletionContext.define('CXCompletionContext_ObjCInterface', 16384)
CXCompletionContext_ObjCProtocol = enum_CXCompletionContext.define('CXCompletionContext_ObjCProtocol', 32768)
CXCompletionContext_ObjCCategory = enum_CXCompletionContext.define('CXCompletionContext_ObjCCategory', 65536)
CXCompletionContext_ObjCInstanceMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCInstanceMessage', 131072)
CXCompletionContext_ObjCClassMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCClassMessage', 262144)
CXCompletionContext_ObjCSelectorName = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorName', 524288)
CXCompletionContext_MacroName = enum_CXCompletionContext.define('CXCompletionContext_MacroName', 1048576)
CXCompletionContext_NaturalLanguage = enum_CXCompletionContext.define('CXCompletionContext_NaturalLanguage', 2097152)
CXCompletionContext_IncludedFile = enum_CXCompletionContext.define('CXCompletionContext_IncludedFile', 4194304)
CXCompletionContext_Unknown = enum_CXCompletionContext.define('CXCompletionContext_Unknown', 8388607)

@dll.bind
def clang_defaultCodeCompleteOptions() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_codeCompleteAt(TU:CXTranslationUnit, complete_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], complete_line:Annotated[int, ctypes.c_uint32], complete_column:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> c.POINTER[CXCodeCompleteResults]: ...
@dll.bind
def clang_sortCodeCompletionResults(Results:c.POINTER[CXCompletionResult], NumResults:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_disposeCodeCompleteResults(Results:c.POINTER[CXCodeCompleteResults]) -> None: ...
@dll.bind
def clang_codeCompleteGetNumDiagnostics(Results:c.POINTER[CXCodeCompleteResults]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_codeCompleteGetDiagnostic(Results:c.POINTER[CXCodeCompleteResults], Index:Annotated[int, ctypes.c_uint32]) -> CXDiagnostic: ...
@dll.bind
def clang_codeCompleteGetContexts(Results:c.POINTER[CXCodeCompleteResults]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_codeCompleteGetContainerKind(Results:c.POINTER[CXCodeCompleteResults], IsIncomplete:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> enum_CXCursorKind: ...
@dll.bind
def clang_codeCompleteGetContainerUSR(Results:c.POINTER[CXCodeCompleteResults]) -> CXString: ...
@dll.bind
def clang_codeCompleteGetObjCSelector(Results:c.POINTER[CXCodeCompleteResults]) -> CXString: ...
@dll.bind
def clang_getClangVersion() -> CXString: ...
@dll.bind
def clang_toggleCrashRecovery(isEnabled:Annotated[int, ctypes.c_uint32]) -> None: ...
CXInclusionVisitor: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p, c.POINTER[CXSourceLocation], Annotated[int, ctypes.c_uint32], ctypes.c_void_p]]
@dll.bind
def clang_getInclusions(tu:CXTranslationUnit, visitor:CXInclusionVisitor, client_data:CXClientData) -> None: ...
class CXEvalResultKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXEval_Int = CXEvalResultKind.define('CXEval_Int', 1)
CXEval_Float = CXEvalResultKind.define('CXEval_Float', 2)
CXEval_ObjCStrLiteral = CXEvalResultKind.define('CXEval_ObjCStrLiteral', 3)
CXEval_StrLiteral = CXEvalResultKind.define('CXEval_StrLiteral', 4)
CXEval_CFStr = CXEvalResultKind.define('CXEval_CFStr', 5)
CXEval_Other = CXEvalResultKind.define('CXEval_Other', 6)
CXEval_UnExposed = CXEvalResultKind.define('CXEval_UnExposed', 0)

CXEvalResult: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_Cursor_Evaluate(C:CXCursor) -> CXEvalResult: ...
@dll.bind
def clang_EvalResult_getKind(E:CXEvalResult) -> CXEvalResultKind: ...
@dll.bind
def clang_EvalResult_getAsInt(E:CXEvalResult) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_EvalResult_getAsLongLong(E:CXEvalResult) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_EvalResult_isUnsignedInt(E:CXEvalResult) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_EvalResult_getAsUnsigned(E:CXEvalResult) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_EvalResult_getAsDouble(E:CXEvalResult) -> Annotated[float, ctypes.c_double]: ...
@dll.bind
def clang_EvalResult_getAsStr(E:CXEvalResult) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def clang_EvalResult_dispose(E:CXEvalResult) -> None: ...
CXRemapping: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_getRemappings(path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXRemapping: ...
@dll.bind
def clang_getRemappingsFromFileList(filePaths:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], numFiles:Annotated[int, ctypes.c_uint32]) -> CXRemapping: ...
@dll.bind
def clang_remap_getNumFiles(_0:CXRemapping) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_remap_getFilenames(_0:CXRemapping, index:Annotated[int, ctypes.c_uint32], original:c.POINTER[CXString], transformed:c.POINTER[CXString]) -> None: ...
@dll.bind
def clang_remap_dispose(_0:CXRemapping) -> None: ...
class enum_CXVisitorResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXVisit_Break = enum_CXVisitorResult.define('CXVisit_Break', 0)
CXVisit_Continue = enum_CXVisitorResult.define('CXVisit_Continue', 1)

@c.record
class struct_CXCursorAndRangeVisitor(c.Struct):
  SIZE = 16
  context: Annotated[ctypes.c_void_p, 0]
  visit: Annotated[c.CFUNCTYPE[enum_CXVisitorResult, [ctypes.c_void_p, CXCursor, CXSourceRange]], 8]
CXCursorAndRangeVisitor: TypeAlias = struct_CXCursorAndRangeVisitor
class CXResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXResult_Success = CXResult.define('CXResult_Success', 0)
CXResult_Invalid = CXResult.define('CXResult_Invalid', 1)
CXResult_VisitBreak = CXResult.define('CXResult_VisitBreak', 2)

@dll.bind
def clang_findReferencesInFile(cursor:CXCursor, file:CXFile, visitor:CXCursorAndRangeVisitor) -> CXResult: ...
@dll.bind
def clang_findIncludesInFile(TU:CXTranslationUnit, file:CXFile, visitor:CXCursorAndRangeVisitor) -> CXResult: ...
class struct__CXCursorAndRangeVisitorBlock(ctypes.Structure): pass
CXCursorAndRangeVisitorBlock: TypeAlias = c.POINTER[struct__CXCursorAndRangeVisitorBlock]
@dll.bind
def clang_findReferencesInFileWithBlock(_0:CXCursor, _1:CXFile, _2:CXCursorAndRangeVisitorBlock) -> CXResult: ...
@dll.bind
def clang_findIncludesInFileWithBlock(_0:CXTranslationUnit, _1:CXFile, _2:CXCursorAndRangeVisitorBlock) -> CXResult: ...
CXIdxClientFile: TypeAlias = ctypes.c_void_p
CXIdxClientEntity: TypeAlias = ctypes.c_void_p
CXIdxClientContainer: TypeAlias = ctypes.c_void_p
CXIdxClientASTFile: TypeAlias = ctypes.c_void_p
@c.record
class CXIdxLoc(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[ctypes.c_void_p, Literal[2]], 0]
  int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
@c.record
class CXIdxIncludedFileInfo(c.Struct):
  SIZE = 56
  hashLoc: Annotated[CXIdxLoc, 0]
  filename: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  file: Annotated[CXFile, 32]
  isImport: Annotated[Annotated[int, ctypes.c_int32], 40]
  isAngled: Annotated[Annotated[int, ctypes.c_int32], 44]
  isModuleImport: Annotated[Annotated[int, ctypes.c_int32], 48]
@c.record
class CXIdxImportedASTFileInfo(c.Struct):
  SIZE = 48
  file: Annotated[CXFile, 0]
  module: Annotated[CXModule, 8]
  loc: Annotated[CXIdxLoc, 16]
  isImplicit: Annotated[Annotated[int, ctypes.c_int32], 40]
class CXIdxEntityKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxEntity_Unexposed = CXIdxEntityKind.define('CXIdxEntity_Unexposed', 0)
CXIdxEntity_Typedef = CXIdxEntityKind.define('CXIdxEntity_Typedef', 1)
CXIdxEntity_Function = CXIdxEntityKind.define('CXIdxEntity_Function', 2)
CXIdxEntity_Variable = CXIdxEntityKind.define('CXIdxEntity_Variable', 3)
CXIdxEntity_Field = CXIdxEntityKind.define('CXIdxEntity_Field', 4)
CXIdxEntity_EnumConstant = CXIdxEntityKind.define('CXIdxEntity_EnumConstant', 5)
CXIdxEntity_ObjCClass = CXIdxEntityKind.define('CXIdxEntity_ObjCClass', 6)
CXIdxEntity_ObjCProtocol = CXIdxEntityKind.define('CXIdxEntity_ObjCProtocol', 7)
CXIdxEntity_ObjCCategory = CXIdxEntityKind.define('CXIdxEntity_ObjCCategory', 8)
CXIdxEntity_ObjCInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCInstanceMethod', 9)
CXIdxEntity_ObjCClassMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCClassMethod', 10)
CXIdxEntity_ObjCProperty = CXIdxEntityKind.define('CXIdxEntity_ObjCProperty', 11)
CXIdxEntity_ObjCIvar = CXIdxEntityKind.define('CXIdxEntity_ObjCIvar', 12)
CXIdxEntity_Enum = CXIdxEntityKind.define('CXIdxEntity_Enum', 13)
CXIdxEntity_Struct = CXIdxEntityKind.define('CXIdxEntity_Struct', 14)
CXIdxEntity_Union = CXIdxEntityKind.define('CXIdxEntity_Union', 15)
CXIdxEntity_CXXClass = CXIdxEntityKind.define('CXIdxEntity_CXXClass', 16)
CXIdxEntity_CXXNamespace = CXIdxEntityKind.define('CXIdxEntity_CXXNamespace', 17)
CXIdxEntity_CXXNamespaceAlias = CXIdxEntityKind.define('CXIdxEntity_CXXNamespaceAlias', 18)
CXIdxEntity_CXXStaticVariable = CXIdxEntityKind.define('CXIdxEntity_CXXStaticVariable', 19)
CXIdxEntity_CXXStaticMethod = CXIdxEntityKind.define('CXIdxEntity_CXXStaticMethod', 20)
CXIdxEntity_CXXInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_CXXInstanceMethod', 21)
CXIdxEntity_CXXConstructor = CXIdxEntityKind.define('CXIdxEntity_CXXConstructor', 22)
CXIdxEntity_CXXDestructor = CXIdxEntityKind.define('CXIdxEntity_CXXDestructor', 23)
CXIdxEntity_CXXConversionFunction = CXIdxEntityKind.define('CXIdxEntity_CXXConversionFunction', 24)
CXIdxEntity_CXXTypeAlias = CXIdxEntityKind.define('CXIdxEntity_CXXTypeAlias', 25)
CXIdxEntity_CXXInterface = CXIdxEntityKind.define('CXIdxEntity_CXXInterface', 26)
CXIdxEntity_CXXConcept = CXIdxEntityKind.define('CXIdxEntity_CXXConcept', 27)

class CXIdxEntityLanguage(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxEntityLang_None = CXIdxEntityLanguage.define('CXIdxEntityLang_None', 0)
CXIdxEntityLang_C = CXIdxEntityLanguage.define('CXIdxEntityLang_C', 1)
CXIdxEntityLang_ObjC = CXIdxEntityLanguage.define('CXIdxEntityLang_ObjC', 2)
CXIdxEntityLang_CXX = CXIdxEntityLanguage.define('CXIdxEntityLang_CXX', 3)
CXIdxEntityLang_Swift = CXIdxEntityLanguage.define('CXIdxEntityLang_Swift', 4)

class CXIdxEntityCXXTemplateKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxEntity_NonTemplate = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_NonTemplate', 0)
CXIdxEntity_Template = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_Template', 1)
CXIdxEntity_TemplatePartialSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplatePartialSpecialization', 2)
CXIdxEntity_TemplateSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplateSpecialization', 3)

class CXIdxAttrKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxAttr_Unexposed = CXIdxAttrKind.define('CXIdxAttr_Unexposed', 0)
CXIdxAttr_IBAction = CXIdxAttrKind.define('CXIdxAttr_IBAction', 1)
CXIdxAttr_IBOutlet = CXIdxAttrKind.define('CXIdxAttr_IBOutlet', 2)
CXIdxAttr_IBOutletCollection = CXIdxAttrKind.define('CXIdxAttr_IBOutletCollection', 3)

@c.record
class CXIdxAttrInfo(c.Struct):
  SIZE = 64
  kind: Annotated[CXIdxAttrKind, 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxEntityInfo(c.Struct):
  SIZE = 80
  kind: Annotated[CXIdxEntityKind, 0]
  templateKind: Annotated[CXIdxEntityCXXTemplateKind, 4]
  lang: Annotated[CXIdxEntityLanguage, 8]
  name: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 16]
  USR: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  cursor: Annotated[CXCursor, 32]
  attributes: Annotated[c.POINTER[c.POINTER[CXIdxAttrInfo]], 64]
  numAttributes: Annotated[Annotated[int, ctypes.c_uint32], 72]
@c.record
class CXIdxContainerInfo(c.Struct):
  SIZE = 32
  cursor: Annotated[CXCursor, 0]
@c.record
class CXIdxIBOutletCollectionAttrInfo(c.Struct):
  SIZE = 72
  attrInfo: Annotated[c.POINTER[CXIdxAttrInfo], 0]
  objcClass: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  classCursor: Annotated[CXCursor, 16]
  classLoc: Annotated[CXIdxLoc, 48]
class CXIdxDeclInfoFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxDeclFlag_Skipped = CXIdxDeclInfoFlags.define('CXIdxDeclFlag_Skipped', 1)

@c.record
class CXIdxDeclInfo(c.Struct):
  SIZE = 128
  entityInfo: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
  semanticContainer: Annotated[c.POINTER[CXIdxContainerInfo], 64]
  lexicalContainer: Annotated[c.POINTER[CXIdxContainerInfo], 72]
  isRedeclaration: Annotated[Annotated[int, ctypes.c_int32], 80]
  isDefinition: Annotated[Annotated[int, ctypes.c_int32], 84]
  isContainer: Annotated[Annotated[int, ctypes.c_int32], 88]
  declAsContainer: Annotated[c.POINTER[CXIdxContainerInfo], 96]
  isImplicit: Annotated[Annotated[int, ctypes.c_int32], 104]
  attributes: Annotated[c.POINTER[c.POINTER[CXIdxAttrInfo]], 112]
  numAttributes: Annotated[Annotated[int, ctypes.c_uint32], 120]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 124]
class CXIdxObjCContainerKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxObjCContainer_ForwardRef = CXIdxObjCContainerKind.define('CXIdxObjCContainer_ForwardRef', 0)
CXIdxObjCContainer_Interface = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Interface', 1)
CXIdxObjCContainer_Implementation = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Implementation', 2)

@c.record
class CXIdxObjCContainerDeclInfo(c.Struct):
  SIZE = 16
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  kind: Annotated[CXIdxObjCContainerKind, 8]
@c.record
class CXIdxBaseClassInfo(c.Struct):
  SIZE = 64
  base: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxObjCProtocolRefInfo(c.Struct):
  SIZE = 64
  protocol: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxObjCProtocolRefListInfo(c.Struct):
  SIZE = 16
  protocols: Annotated[c.POINTER[c.POINTER[CXIdxObjCProtocolRefInfo]], 0]
  numProtocols: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class CXIdxObjCInterfaceDeclInfo(c.Struct):
  SIZE = 24
  containerInfo: Annotated[c.POINTER[CXIdxObjCContainerDeclInfo], 0]
  superInfo: Annotated[c.POINTER[CXIdxBaseClassInfo], 8]
  protocols: Annotated[c.POINTER[CXIdxObjCProtocolRefListInfo], 16]
@c.record
class CXIdxObjCCategoryDeclInfo(c.Struct):
  SIZE = 80
  containerInfo: Annotated[c.POINTER[CXIdxObjCContainerDeclInfo], 0]
  objcClass: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  classCursor: Annotated[CXCursor, 16]
  classLoc: Annotated[CXIdxLoc, 48]
  protocols: Annotated[c.POINTER[CXIdxObjCProtocolRefListInfo], 72]
@c.record
class CXIdxObjCPropertyDeclInfo(c.Struct):
  SIZE = 24
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  getter: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  setter: Annotated[c.POINTER[CXIdxEntityInfo], 16]
@c.record
class CXIdxCXXClassDeclInfo(c.Struct):
  SIZE = 24
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  bases: Annotated[c.POINTER[c.POINTER[CXIdxBaseClassInfo]], 8]
  numBases: Annotated[Annotated[int, ctypes.c_uint32], 16]
class CXIdxEntityRefKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIdxEntityRef_Direct = CXIdxEntityRefKind.define('CXIdxEntityRef_Direct', 1)
CXIdxEntityRef_Implicit = CXIdxEntityRefKind.define('CXIdxEntityRef_Implicit', 2)

class CXSymbolRole(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXSymbolRole_None = CXSymbolRole.define('CXSymbolRole_None', 0)
CXSymbolRole_Declaration = CXSymbolRole.define('CXSymbolRole_Declaration', 1)
CXSymbolRole_Definition = CXSymbolRole.define('CXSymbolRole_Definition', 2)
CXSymbolRole_Reference = CXSymbolRole.define('CXSymbolRole_Reference', 4)
CXSymbolRole_Read = CXSymbolRole.define('CXSymbolRole_Read', 8)
CXSymbolRole_Write = CXSymbolRole.define('CXSymbolRole_Write', 16)
CXSymbolRole_Call = CXSymbolRole.define('CXSymbolRole_Call', 32)
CXSymbolRole_Dynamic = CXSymbolRole.define('CXSymbolRole_Dynamic', 64)
CXSymbolRole_AddressOf = CXSymbolRole.define('CXSymbolRole_AddressOf', 128)
CXSymbolRole_Implicit = CXSymbolRole.define('CXSymbolRole_Implicit', 256)

@c.record
class CXIdxEntityRefInfo(c.Struct):
  SIZE = 96
  kind: Annotated[CXIdxEntityRefKind, 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
  referencedEntity: Annotated[c.POINTER[CXIdxEntityInfo], 64]
  parentEntity: Annotated[c.POINTER[CXIdxEntityInfo], 72]
  container: Annotated[c.POINTER[CXIdxContainerInfo], 80]
  role: Annotated[CXSymbolRole, 88]
@c.record
class IndexerCallbacks(c.Struct):
  SIZE = 64
  abortQuery: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [CXClientData, ctypes.c_void_p]], 0]
  diagnostic: Annotated[c.CFUNCTYPE[None, [CXClientData, CXDiagnosticSet, ctypes.c_void_p]], 8]
  enteredMainFile: Annotated[c.CFUNCTYPE[CXIdxClientFile, [CXClientData, CXFile, ctypes.c_void_p]], 16]
  ppIncludedFile: Annotated[c.CFUNCTYPE[CXIdxClientFile, [CXClientData, c.POINTER[CXIdxIncludedFileInfo]]], 24]
  importedASTFile: Annotated[c.CFUNCTYPE[CXIdxClientASTFile, [CXClientData, c.POINTER[CXIdxImportedASTFileInfo]]], 32]
  startedTranslationUnit: Annotated[c.CFUNCTYPE[CXIdxClientContainer, [CXClientData, ctypes.c_void_p]], 40]
  indexDeclaration: Annotated[c.CFUNCTYPE[None, [CXClientData, c.POINTER[CXIdxDeclInfo]]], 48]
  indexEntityReference: Annotated[c.CFUNCTYPE[None, [CXClientData, c.POINTER[CXIdxEntityRefInfo]]], 56]
@dll.bind
def clang_index_isEntityObjCContainerKind(_0:CXIdxEntityKind) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_index_getObjCContainerDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCContainerDeclInfo]: ...
@dll.bind
def clang_index_getObjCInterfaceDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCInterfaceDeclInfo]: ...
@dll.bind
def clang_index_getObjCCategoryDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCCategoryDeclInfo]: ...
@dll.bind
def clang_index_getObjCProtocolRefListInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCProtocolRefListInfo]: ...
@dll.bind
def clang_index_getObjCPropertyDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCPropertyDeclInfo]: ...
@dll.bind
def clang_index_getIBOutletCollectionAttrInfo(_0:c.POINTER[CXIdxAttrInfo]) -> c.POINTER[CXIdxIBOutletCollectionAttrInfo]: ...
@dll.bind
def clang_index_getCXXClassDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxCXXClassDeclInfo]: ...
@dll.bind
def clang_index_getClientContainer(_0:c.POINTER[CXIdxContainerInfo]) -> CXIdxClientContainer: ...
@dll.bind
def clang_index_setClientContainer(_0:c.POINTER[CXIdxContainerInfo], _1:CXIdxClientContainer) -> None: ...
@dll.bind
def clang_index_getClientEntity(_0:c.POINTER[CXIdxEntityInfo]) -> CXIdxClientEntity: ...
@dll.bind
def clang_index_setClientEntity(_0:c.POINTER[CXIdxEntityInfo], _1:CXIdxClientEntity) -> None: ...
CXIndexAction: TypeAlias = ctypes.c_void_p
@dll.bind
def clang_IndexAction_create(CIdx:CXIndex) -> CXIndexAction: ...
@dll.bind
def clang_IndexAction_dispose(_0:CXIndexAction) -> None: ...
class CXIndexOptFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXIndexOpt_None = CXIndexOptFlags.define('CXIndexOpt_None', 0)
CXIndexOpt_SuppressRedundantRefs = CXIndexOptFlags.define('CXIndexOpt_SuppressRedundantRefs', 1)
CXIndexOpt_IndexFunctionLocalSymbols = CXIndexOptFlags.define('CXIndexOpt_IndexFunctionLocalSymbols', 2)
CXIndexOpt_IndexImplicitTemplateInstantiations = CXIndexOptFlags.define('CXIndexOpt_IndexImplicitTemplateInstantiations', 4)
CXIndexOpt_SuppressWarnings = CXIndexOptFlags.define('CXIndexOpt_SuppressWarnings', 8)
CXIndexOpt_SkipParsedBodiesInSession = CXIndexOptFlags.define('CXIndexOpt_SkipParsedBodiesInSession', 16)

@dll.bind
def clang_indexSourceFile(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit], TU_options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexSourceFileFullArgv(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit], TU_options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexTranslationUnit(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], _5:CXTranslationUnit) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexLoc_getFileLocation(loc:CXIdxLoc, indexFile:c.POINTER[CXIdxClientFile], file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_indexLoc_getCXSourceLocation(loc:CXIdxLoc) -> CXSourceLocation: ...
CXFieldVisitor: TypeAlias = c.CFUNCTYPE[enum_CXVisitorResult, [CXCursor, ctypes.c_void_p]]
@dll.bind
def clang_Type_visitFields(T:CXType, visitor:CXFieldVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_visitCXXBaseClasses(T:CXType, visitor:CXFieldVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
class enum_CXBinaryOperatorKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXBinaryOperator_Invalid = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Invalid', 0)
CXBinaryOperator_PtrMemD = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemD', 1)
CXBinaryOperator_PtrMemI = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemI', 2)
CXBinaryOperator_Mul = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Mul', 3)
CXBinaryOperator_Div = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Div', 4)
CXBinaryOperator_Rem = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Rem', 5)
CXBinaryOperator_Add = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Add', 6)
CXBinaryOperator_Sub = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Sub', 7)
CXBinaryOperator_Shl = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shl', 8)
CXBinaryOperator_Shr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shr', 9)
CXBinaryOperator_Cmp = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Cmp', 10)
CXBinaryOperator_LT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LT', 11)
CXBinaryOperator_GT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GT', 12)
CXBinaryOperator_LE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LE', 13)
CXBinaryOperator_GE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GE', 14)
CXBinaryOperator_EQ = enum_CXBinaryOperatorKind.define('CXBinaryOperator_EQ', 15)
CXBinaryOperator_NE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_NE', 16)
CXBinaryOperator_And = enum_CXBinaryOperatorKind.define('CXBinaryOperator_And', 17)
CXBinaryOperator_Xor = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Xor', 18)
CXBinaryOperator_Or = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Or', 19)
CXBinaryOperator_LAnd = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LAnd', 20)
CXBinaryOperator_LOr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LOr', 21)
CXBinaryOperator_Assign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Assign', 22)
CXBinaryOperator_MulAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_MulAssign', 23)
CXBinaryOperator_DivAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_DivAssign', 24)
CXBinaryOperator_RemAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_RemAssign', 25)
CXBinaryOperator_AddAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AddAssign', 26)
CXBinaryOperator_SubAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_SubAssign', 27)
CXBinaryOperator_ShlAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShlAssign', 28)
CXBinaryOperator_ShrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShrAssign', 29)
CXBinaryOperator_AndAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AndAssign', 30)
CXBinaryOperator_XorAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_XorAssign', 31)
CXBinaryOperator_OrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_OrAssign', 32)
CXBinaryOperator_Comma = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Comma', 33)

@dll.bind
def clang_getBinaryOperatorKindSpelling(kind:enum_CXBinaryOperatorKind) -> CXString: ...
@dll.bind
def clang_getCursorBinaryOperatorKind(cursor:CXCursor) -> enum_CXBinaryOperatorKind: ...
class enum_CXUnaryOperatorKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
CXUnaryOperator_Invalid = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Invalid', 0)
CXUnaryOperator_PostInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostInc', 1)
CXUnaryOperator_PostDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostDec', 2)
CXUnaryOperator_PreInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreInc', 3)
CXUnaryOperator_PreDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreDec', 4)
CXUnaryOperator_AddrOf = enum_CXUnaryOperatorKind.define('CXUnaryOperator_AddrOf', 5)
CXUnaryOperator_Deref = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Deref', 6)
CXUnaryOperator_Plus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Plus', 7)
CXUnaryOperator_Minus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Minus', 8)
CXUnaryOperator_Not = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Not', 9)
CXUnaryOperator_LNot = enum_CXUnaryOperatorKind.define('CXUnaryOperator_LNot', 10)
CXUnaryOperator_Real = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Real', 11)
CXUnaryOperator_Imag = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Imag', 12)
CXUnaryOperator_Extension = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Extension', 13)
CXUnaryOperator_Coawait = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Coawait', 14)

@dll.bind
def clang_getUnaryOperatorKindSpelling(kind:enum_CXUnaryOperatorKind) -> CXString: ...
@dll.bind
def clang_getCursorUnaryOperatorKind(cursor:CXCursor) -> enum_CXUnaryOperatorKind: ...
@dll.bind
def clang_getCString(string:CXString) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def clang_disposeString(string:CXString) -> None: ...
@dll.bind
def clang_disposeStringSet(set:c.POINTER[CXStringSet]) -> None: ...
@dll.bind
def clang_getNullLocation() -> CXSourceLocation: ...
@dll.bind
def clang_equalLocations(loc1:CXSourceLocation, loc2:CXSourceLocation) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isBeforeInTranslationUnit(loc1:CXSourceLocation, loc2:CXSourceLocation) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Location_isInSystemHeader(location:CXSourceLocation) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Location_isFromMainFile(location:CXSourceLocation) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getNullRange() -> CXSourceRange: ...
@dll.bind
def clang_getRange(begin:CXSourceLocation, end:CXSourceLocation) -> CXSourceRange: ...
@dll.bind
def clang_equalRanges(range1:CXSourceRange, range2:CXSourceRange) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Range_isNull(range:CXSourceRange) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getExpansionLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getPresumedLocation(location:CXSourceLocation, filename:c.POINTER[CXString], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getInstantiationLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getSpellingLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getFileLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getRangeStart(range:CXSourceRange) -> CXSourceLocation: ...
@dll.bind
def clang_getRangeEnd(range:CXSourceRange) -> CXSourceLocation: ...
@dll.bind
def clang_disposeSourceRangeList(ranges:c.POINTER[CXSourceRangeList]) -> None: ...
@dll.bind
def clang_getFileName(SFile:CXFile) -> CXString: ...
time_t: TypeAlias = Annotated[int, ctypes.c_int64]
@dll.bind
def clang_getFileTime(SFile:CXFile) -> time_t: ...
@c.record
class CXFileUniqueID(c.Struct):
  SIZE = 24
  data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[3]], 0]
@dll.bind
def clang_getFileUniqueID(file:CXFile, outID:c.POINTER[CXFileUniqueID]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_File_isEqual(file1:CXFile, file2:CXFile) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_File_tryGetRealPathName(file:CXFile) -> CXString: ...
c.init_records()
CINDEX_VERSION_MAJOR = 0 # type: ignore
CINDEX_VERSION_MINOR = 64 # type: ignore
CINDEX_VERSION_ENCODE = lambda major,minor: (((major)*10000) + ((minor)*1)) # type: ignore
CINDEX_VERSION = CINDEX_VERSION_ENCODE(CINDEX_VERSION_MAJOR, CINDEX_VERSION_MINOR) # type: ignore
CINDEX_VERSION_STRINGIZE = lambda major,minor: CINDEX_VERSION_STRINGIZE_(major, minor) # type: ignore