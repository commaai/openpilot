# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('libclang', 'clang-20')
CXIndex = ctypes.c_void_p
class struct_CXTargetInfoImpl(Struct): pass
CXTargetInfo = ctypes.POINTER(struct_CXTargetInfoImpl)
class struct_CXTranslationUnitImpl(Struct): pass
CXTranslationUnit = ctypes.POINTER(struct_CXTranslationUnitImpl)
CXClientData = ctypes.c_void_p
class struct_CXUnsavedFile(Struct): pass
struct_CXUnsavedFile._fields_ = [
  ('Filename', ctypes.POINTER(ctypes.c_char)),
  ('Contents', ctypes.POINTER(ctypes.c_char)),
  ('Length', ctypes.c_uint64),
]
enum_CXAvailabilityKind = CEnum(ctypes.c_uint32)
CXAvailability_Available = enum_CXAvailabilityKind.define('CXAvailability_Available', 0)
CXAvailability_Deprecated = enum_CXAvailabilityKind.define('CXAvailability_Deprecated', 1)
CXAvailability_NotAvailable = enum_CXAvailabilityKind.define('CXAvailability_NotAvailable', 2)
CXAvailability_NotAccessible = enum_CXAvailabilityKind.define('CXAvailability_NotAccessible', 3)

class struct_CXVersion(Struct): pass
struct_CXVersion._fields_ = [
  ('Major', ctypes.c_int32),
  ('Minor', ctypes.c_int32),
  ('Subminor', ctypes.c_int32),
]
CXVersion = struct_CXVersion
enum_CXCursor_ExceptionSpecificationKind = CEnum(ctypes.c_uint32)
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

try: (clang_createIndex:=dll.clang_createIndex).restype, clang_createIndex.argtypes = CXIndex, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (clang_disposeIndex:=dll.clang_disposeIndex).restype, clang_disposeIndex.argtypes = None, [CXIndex]
except AttributeError: pass

CXChoice = CEnum(ctypes.c_uint32)
CXChoice_Default = CXChoice.define('CXChoice_Default', 0)
CXChoice_Enabled = CXChoice.define('CXChoice_Enabled', 1)
CXChoice_Disabled = CXChoice.define('CXChoice_Disabled', 2)

CXGlobalOptFlags = CEnum(ctypes.c_uint32)
CXGlobalOpt_None = CXGlobalOptFlags.define('CXGlobalOpt_None', 0)
CXGlobalOpt_ThreadBackgroundPriorityForIndexing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForIndexing', 1)
CXGlobalOpt_ThreadBackgroundPriorityForEditing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForEditing', 2)
CXGlobalOpt_ThreadBackgroundPriorityForAll = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForAll', 3)

class struct_CXIndexOptions(Struct): pass
struct_CXIndexOptions._fields_ = [
  ('Size', ctypes.c_uint32),
  ('ThreadBackgroundPriorityForIndexing', ctypes.c_ubyte),
  ('ThreadBackgroundPriorityForEditing', ctypes.c_ubyte),
  ('ExcludeDeclarationsFromPCH', ctypes.c_uint32,1),
  ('DisplayDiagnostics', ctypes.c_uint32,1),
  ('StorePreamblesInMemory', ctypes.c_uint32,1),
  ('', ctypes.c_uint32,13),
  ('PreambleStoragePath', ctypes.POINTER(ctypes.c_char)),
  ('InvocationEmissionPath', ctypes.POINTER(ctypes.c_char)),
]
CXIndexOptions = struct_CXIndexOptions
try: (clang_createIndexWithOptions:=dll.clang_createIndexWithOptions).restype, clang_createIndexWithOptions.argtypes = CXIndex, [ctypes.POINTER(CXIndexOptions)]
except AttributeError: pass

try: (clang_CXIndex_setGlobalOptions:=dll.clang_CXIndex_setGlobalOptions).restype, clang_CXIndex_setGlobalOptions.argtypes = None, [CXIndex, ctypes.c_uint32]
except AttributeError: pass

try: (clang_CXIndex_getGlobalOptions:=dll.clang_CXIndex_getGlobalOptions).restype, clang_CXIndex_getGlobalOptions.argtypes = ctypes.c_uint32, [CXIndex]
except AttributeError: pass

try: (clang_CXIndex_setInvocationEmissionPathOption:=dll.clang_CXIndex_setInvocationEmissionPathOption).restype, clang_CXIndex_setInvocationEmissionPathOption.argtypes = None, [CXIndex, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

CXFile = ctypes.c_void_p
try: (clang_isFileMultipleIncludeGuarded:=dll.clang_isFileMultipleIncludeGuarded).restype, clang_isFileMultipleIncludeGuarded.argtypes = ctypes.c_uint32, [CXTranslationUnit, CXFile]
except AttributeError: pass

try: (clang_getFile:=dll.clang_getFile).restype, clang_getFile.argtypes = CXFile, [CXTranslationUnit, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (clang_getFileContents:=dll.clang_getFileContents).restype, clang_getFileContents.argtypes = ctypes.POINTER(ctypes.c_char), [CXTranslationUnit, CXFile, ctypes.POINTER(size_t)]
except AttributeError: pass

class CXSourceLocation(Struct): pass
CXSourceLocation._fields_ = [
  ('ptr_data', (ctypes.c_void_p * 2)),
  ('int_data', ctypes.c_uint32),
]
try: (clang_getLocation:=dll.clang_getLocation).restype, clang_getLocation.argtypes = CXSourceLocation, [CXTranslationUnit, CXFile, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getLocationForOffset:=dll.clang_getLocationForOffset).restype, clang_getLocationForOffset.argtypes = CXSourceLocation, [CXTranslationUnit, CXFile, ctypes.c_uint32]
except AttributeError: pass

class CXSourceRangeList(Struct): pass
class CXSourceRange(Struct): pass
CXSourceRange._fields_ = [
  ('ptr_data', (ctypes.c_void_p * 2)),
  ('begin_int_data', ctypes.c_uint32),
  ('end_int_data', ctypes.c_uint32),
]
CXSourceRangeList._fields_ = [
  ('count', ctypes.c_uint32),
  ('ranges', ctypes.POINTER(CXSourceRange)),
]
try: (clang_getSkippedRanges:=dll.clang_getSkippedRanges).restype, clang_getSkippedRanges.argtypes = ctypes.POINTER(CXSourceRangeList), [CXTranslationUnit, CXFile]
except AttributeError: pass

try: (clang_getAllSkippedRanges:=dll.clang_getAllSkippedRanges).restype, clang_getAllSkippedRanges.argtypes = ctypes.POINTER(CXSourceRangeList), [CXTranslationUnit]
except AttributeError: pass

try: (clang_getNumDiagnostics:=dll.clang_getNumDiagnostics).restype, clang_getNumDiagnostics.argtypes = ctypes.c_uint32, [CXTranslationUnit]
except AttributeError: pass

CXDiagnostic = ctypes.c_void_p
try: (clang_getDiagnostic:=dll.clang_getDiagnostic).restype, clang_getDiagnostic.argtypes = CXDiagnostic, [CXTranslationUnit, ctypes.c_uint32]
except AttributeError: pass

CXDiagnosticSet = ctypes.c_void_p
try: (clang_getDiagnosticSetFromTU:=dll.clang_getDiagnosticSetFromTU).restype, clang_getDiagnosticSetFromTU.argtypes = CXDiagnosticSet, [CXTranslationUnit]
except AttributeError: pass

class CXString(Struct): pass
CXString._fields_ = [
  ('data', ctypes.c_void_p),
  ('private_flags', ctypes.c_uint32),
]
try: (clang_getTranslationUnitSpelling:=dll.clang_getTranslationUnitSpelling).restype, clang_getTranslationUnitSpelling.argtypes = CXString, [CXTranslationUnit]
except AttributeError: pass

try: (clang_createTranslationUnitFromSourceFile:=dll.clang_createTranslationUnitFromSourceFile).restype, clang_createTranslationUnitFromSourceFile.argtypes = CXTranslationUnit, [CXIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_uint32, ctypes.POINTER(struct_CXUnsavedFile)]
except AttributeError: pass

try: (clang_createTranslationUnit:=dll.clang_createTranslationUnit).restype, clang_createTranslationUnit.argtypes = CXTranslationUnit, [CXIndex, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

enum_CXErrorCode = CEnum(ctypes.c_uint32)
CXError_Success = enum_CXErrorCode.define('CXError_Success', 0)
CXError_Failure = enum_CXErrorCode.define('CXError_Failure', 1)
CXError_Crashed = enum_CXErrorCode.define('CXError_Crashed', 2)
CXError_InvalidArguments = enum_CXErrorCode.define('CXError_InvalidArguments', 3)
CXError_ASTReadError = enum_CXErrorCode.define('CXError_ASTReadError', 4)

try: (clang_createTranslationUnit2:=dll.clang_createTranslationUnit2).restype, clang_createTranslationUnit2.argtypes = enum_CXErrorCode, [CXIndex, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(CXTranslationUnit)]
except AttributeError: pass

enum_CXTranslationUnit_Flags = CEnum(ctypes.c_uint32)
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

try: (clang_defaultEditingTranslationUnitOptions:=dll.clang_defaultEditingTranslationUnitOptions).restype, clang_defaultEditingTranslationUnitOptions.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (clang_parseTranslationUnit:=dll.clang_parseTranslationUnit).restype, clang_parseTranslationUnit.argtypes = CXTranslationUnit, [CXIndex, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (clang_parseTranslationUnit2:=dll.clang_parseTranslationUnit2).restype, clang_parseTranslationUnit2.argtypes = enum_CXErrorCode, [CXIndex, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CXTranslationUnit)]
except AttributeError: pass

try: (clang_parseTranslationUnit2FullArgv:=dll.clang_parseTranslationUnit2FullArgv).restype, clang_parseTranslationUnit2FullArgv.argtypes = enum_CXErrorCode, [CXIndex, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CXTranslationUnit)]
except AttributeError: pass

enum_CXSaveTranslationUnit_Flags = CEnum(ctypes.c_uint32)
CXSaveTranslationUnit_None = enum_CXSaveTranslationUnit_Flags.define('CXSaveTranslationUnit_None', 0)

try: (clang_defaultSaveOptions:=dll.clang_defaultSaveOptions).restype, clang_defaultSaveOptions.argtypes = ctypes.c_uint32, [CXTranslationUnit]
except AttributeError: pass

enum_CXSaveError = CEnum(ctypes.c_uint32)
CXSaveError_None = enum_CXSaveError.define('CXSaveError_None', 0)
CXSaveError_Unknown = enum_CXSaveError.define('CXSaveError_Unknown', 1)
CXSaveError_TranslationErrors = enum_CXSaveError.define('CXSaveError_TranslationErrors', 2)
CXSaveError_InvalidTU = enum_CXSaveError.define('CXSaveError_InvalidTU', 3)

try: (clang_saveTranslationUnit:=dll.clang_saveTranslationUnit).restype, clang_saveTranslationUnit.argtypes = ctypes.c_int32, [CXTranslationUnit, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (clang_suspendTranslationUnit:=dll.clang_suspendTranslationUnit).restype, clang_suspendTranslationUnit.argtypes = ctypes.c_uint32, [CXTranslationUnit]
except AttributeError: pass

try: (clang_disposeTranslationUnit:=dll.clang_disposeTranslationUnit).restype, clang_disposeTranslationUnit.argtypes = None, [CXTranslationUnit]
except AttributeError: pass

enum_CXReparse_Flags = CEnum(ctypes.c_uint32)
CXReparse_None = enum_CXReparse_Flags.define('CXReparse_None', 0)

try: (clang_defaultReparseOptions:=dll.clang_defaultReparseOptions).restype, clang_defaultReparseOptions.argtypes = ctypes.c_uint32, [CXTranslationUnit]
except AttributeError: pass

try: (clang_reparseTranslationUnit:=dll.clang_reparseTranslationUnit).restype, clang_reparseTranslationUnit.argtypes = ctypes.c_int32, [CXTranslationUnit, ctypes.c_uint32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32]
except AttributeError: pass

enum_CXTUResourceUsageKind = CEnum(ctypes.c_uint32)
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

try: (clang_getTUResourceUsageName:=dll.clang_getTUResourceUsageName).restype, clang_getTUResourceUsageName.argtypes = ctypes.POINTER(ctypes.c_char), [enum_CXTUResourceUsageKind]
except AttributeError: pass

class struct_CXTUResourceUsageEntry(Struct): pass
struct_CXTUResourceUsageEntry._fields_ = [
  ('kind', enum_CXTUResourceUsageKind),
  ('amount', ctypes.c_uint64),
]
CXTUResourceUsageEntry = struct_CXTUResourceUsageEntry
class struct_CXTUResourceUsage(Struct): pass
struct_CXTUResourceUsage._fields_ = [
  ('data', ctypes.c_void_p),
  ('numEntries', ctypes.c_uint32),
  ('entries', ctypes.POINTER(CXTUResourceUsageEntry)),
]
CXTUResourceUsage = struct_CXTUResourceUsage
try: (clang_getCXTUResourceUsage:=dll.clang_getCXTUResourceUsage).restype, clang_getCXTUResourceUsage.argtypes = CXTUResourceUsage, [CXTranslationUnit]
except AttributeError: pass

try: (clang_disposeCXTUResourceUsage:=dll.clang_disposeCXTUResourceUsage).restype, clang_disposeCXTUResourceUsage.argtypes = None, [CXTUResourceUsage]
except AttributeError: pass

try: (clang_getTranslationUnitTargetInfo:=dll.clang_getTranslationUnitTargetInfo).restype, clang_getTranslationUnitTargetInfo.argtypes = CXTargetInfo, [CXTranslationUnit]
except AttributeError: pass

try: (clang_TargetInfo_dispose:=dll.clang_TargetInfo_dispose).restype, clang_TargetInfo_dispose.argtypes = None, [CXTargetInfo]
except AttributeError: pass

try: (clang_TargetInfo_getTriple:=dll.clang_TargetInfo_getTriple).restype, clang_TargetInfo_getTriple.argtypes = CXString, [CXTargetInfo]
except AttributeError: pass

try: (clang_TargetInfo_getPointerWidth:=dll.clang_TargetInfo_getPointerWidth).restype, clang_TargetInfo_getPointerWidth.argtypes = ctypes.c_int32, [CXTargetInfo]
except AttributeError: pass

enum_CXCursorKind = CEnum(ctypes.c_uint32)
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

class CXCursor(Struct): pass
CXCursor._fields_ = [
  ('kind', enum_CXCursorKind),
  ('xdata', ctypes.c_int32),
  ('data', (ctypes.c_void_p * 3)),
]
try: (clang_getNullCursor:=dll.clang_getNullCursor).restype, clang_getNullCursor.argtypes = CXCursor, []
except AttributeError: pass

try: (clang_getTranslationUnitCursor:=dll.clang_getTranslationUnitCursor).restype, clang_getTranslationUnitCursor.argtypes = CXCursor, [CXTranslationUnit]
except AttributeError: pass

try: (clang_equalCursors:=dll.clang_equalCursors).restype, clang_equalCursors.argtypes = ctypes.c_uint32, [CXCursor, CXCursor]
except AttributeError: pass

try: (clang_Cursor_isNull:=dll.clang_Cursor_isNull).restype, clang_Cursor_isNull.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_hashCursor:=dll.clang_hashCursor).restype, clang_hashCursor.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getCursorKind:=dll.clang_getCursorKind).restype, clang_getCursorKind.argtypes = enum_CXCursorKind, [CXCursor]
except AttributeError: pass

try: (clang_isDeclaration:=dll.clang_isDeclaration).restype, clang_isDeclaration.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isInvalidDeclaration:=dll.clang_isInvalidDeclaration).restype, clang_isInvalidDeclaration.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_isReference:=dll.clang_isReference).restype, clang_isReference.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isExpression:=dll.clang_isExpression).restype, clang_isExpression.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isStatement:=dll.clang_isStatement).restype, clang_isStatement.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isAttribute:=dll.clang_isAttribute).restype, clang_isAttribute.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_Cursor_hasAttrs:=dll.clang_Cursor_hasAttrs).restype, clang_Cursor_hasAttrs.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_isInvalid:=dll.clang_isInvalid).restype, clang_isInvalid.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isTranslationUnit:=dll.clang_isTranslationUnit).restype, clang_isTranslationUnit.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isPreprocessing:=dll.clang_isPreprocessing).restype, clang_isPreprocessing.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_isUnexposed:=dll.clang_isUnexposed).restype, clang_isUnexposed.argtypes = ctypes.c_uint32, [enum_CXCursorKind]
except AttributeError: pass

enum_CXLinkageKind = CEnum(ctypes.c_uint32)
CXLinkage_Invalid = enum_CXLinkageKind.define('CXLinkage_Invalid', 0)
CXLinkage_NoLinkage = enum_CXLinkageKind.define('CXLinkage_NoLinkage', 1)
CXLinkage_Internal = enum_CXLinkageKind.define('CXLinkage_Internal', 2)
CXLinkage_UniqueExternal = enum_CXLinkageKind.define('CXLinkage_UniqueExternal', 3)
CXLinkage_External = enum_CXLinkageKind.define('CXLinkage_External', 4)

try: (clang_getCursorLinkage:=dll.clang_getCursorLinkage).restype, clang_getCursorLinkage.argtypes = enum_CXLinkageKind, [CXCursor]
except AttributeError: pass

enum_CXVisibilityKind = CEnum(ctypes.c_uint32)
CXVisibility_Invalid = enum_CXVisibilityKind.define('CXVisibility_Invalid', 0)
CXVisibility_Hidden = enum_CXVisibilityKind.define('CXVisibility_Hidden', 1)
CXVisibility_Protected = enum_CXVisibilityKind.define('CXVisibility_Protected', 2)
CXVisibility_Default = enum_CXVisibilityKind.define('CXVisibility_Default', 3)

try: (clang_getCursorVisibility:=dll.clang_getCursorVisibility).restype, clang_getCursorVisibility.argtypes = enum_CXVisibilityKind, [CXCursor]
except AttributeError: pass

try: (clang_getCursorAvailability:=dll.clang_getCursorAvailability).restype, clang_getCursorAvailability.argtypes = enum_CXAvailabilityKind, [CXCursor]
except AttributeError: pass

class struct_CXPlatformAvailability(Struct): pass
struct_CXPlatformAvailability._fields_ = [
  ('Platform', CXString),
  ('Introduced', CXVersion),
  ('Deprecated', CXVersion),
  ('Obsoleted', CXVersion),
  ('Unavailable', ctypes.c_int32),
  ('Message', CXString),
]
CXPlatformAvailability = struct_CXPlatformAvailability
try: (clang_getCursorPlatformAvailability:=dll.clang_getCursorPlatformAvailability).restype, clang_getCursorPlatformAvailability.argtypes = ctypes.c_int32, [CXCursor, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(CXString), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(CXString), ctypes.POINTER(CXPlatformAvailability), ctypes.c_int32]
except AttributeError: pass

try: (clang_disposeCXPlatformAvailability:=dll.clang_disposeCXPlatformAvailability).restype, clang_disposeCXPlatformAvailability.argtypes = None, [ctypes.POINTER(CXPlatformAvailability)]
except AttributeError: pass

try: (clang_Cursor_getVarDeclInitializer:=dll.clang_Cursor_getVarDeclInitializer).restype, clang_Cursor_getVarDeclInitializer.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_hasVarDeclGlobalStorage:=dll.clang_Cursor_hasVarDeclGlobalStorage).restype, clang_Cursor_hasVarDeclGlobalStorage.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_hasVarDeclExternalStorage:=dll.clang_Cursor_hasVarDeclExternalStorage).restype, clang_Cursor_hasVarDeclExternalStorage.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

enum_CXLanguageKind = CEnum(ctypes.c_uint32)
CXLanguage_Invalid = enum_CXLanguageKind.define('CXLanguage_Invalid', 0)
CXLanguage_C = enum_CXLanguageKind.define('CXLanguage_C', 1)
CXLanguage_ObjC = enum_CXLanguageKind.define('CXLanguage_ObjC', 2)
CXLanguage_CPlusPlus = enum_CXLanguageKind.define('CXLanguage_CPlusPlus', 3)

try: (clang_getCursorLanguage:=dll.clang_getCursorLanguage).restype, clang_getCursorLanguage.argtypes = enum_CXLanguageKind, [CXCursor]
except AttributeError: pass

enum_CXTLSKind = CEnum(ctypes.c_uint32)
CXTLS_None = enum_CXTLSKind.define('CXTLS_None', 0)
CXTLS_Dynamic = enum_CXTLSKind.define('CXTLS_Dynamic', 1)
CXTLS_Static = enum_CXTLSKind.define('CXTLS_Static', 2)

try: (clang_getCursorTLSKind:=dll.clang_getCursorTLSKind).restype, clang_getCursorTLSKind.argtypes = enum_CXTLSKind, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getTranslationUnit:=dll.clang_Cursor_getTranslationUnit).restype, clang_Cursor_getTranslationUnit.argtypes = CXTranslationUnit, [CXCursor]
except AttributeError: pass

class struct_CXCursorSetImpl(Struct): pass
CXCursorSet = ctypes.POINTER(struct_CXCursorSetImpl)
try: (clang_createCXCursorSet:=dll.clang_createCXCursorSet).restype, clang_createCXCursorSet.argtypes = CXCursorSet, []
except AttributeError: pass

try: (clang_disposeCXCursorSet:=dll.clang_disposeCXCursorSet).restype, clang_disposeCXCursorSet.argtypes = None, [CXCursorSet]
except AttributeError: pass

try: (clang_CXCursorSet_contains:=dll.clang_CXCursorSet_contains).restype, clang_CXCursorSet_contains.argtypes = ctypes.c_uint32, [CXCursorSet, CXCursor]
except AttributeError: pass

try: (clang_CXCursorSet_insert:=dll.clang_CXCursorSet_insert).restype, clang_CXCursorSet_insert.argtypes = ctypes.c_uint32, [CXCursorSet, CXCursor]
except AttributeError: pass

try: (clang_getCursorSemanticParent:=dll.clang_getCursorSemanticParent).restype, clang_getCursorSemanticParent.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_getCursorLexicalParent:=dll.clang_getCursorLexicalParent).restype, clang_getCursorLexicalParent.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_getOverriddenCursors:=dll.clang_getOverriddenCursors).restype, clang_getOverriddenCursors.argtypes = None, [CXCursor, ctypes.POINTER(ctypes.POINTER(CXCursor)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_disposeOverriddenCursors:=dll.clang_disposeOverriddenCursors).restype, clang_disposeOverriddenCursors.argtypes = None, [ctypes.POINTER(CXCursor)]
except AttributeError: pass

try: (clang_getIncludedFile:=dll.clang_getIncludedFile).restype, clang_getIncludedFile.argtypes = CXFile, [CXCursor]
except AttributeError: pass

try: (clang_getCursor:=dll.clang_getCursor).restype, clang_getCursor.argtypes = CXCursor, [CXTranslationUnit, CXSourceLocation]
except AttributeError: pass

try: (clang_getCursorLocation:=dll.clang_getCursorLocation).restype, clang_getCursorLocation.argtypes = CXSourceLocation, [CXCursor]
except AttributeError: pass

try: (clang_getCursorExtent:=dll.clang_getCursorExtent).restype, clang_getCursorExtent.argtypes = CXSourceRange, [CXCursor]
except AttributeError: pass

enum_CXTypeKind = CEnum(ctypes.c_uint32)
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

enum_CXCallingConv = CEnum(ctypes.c_uint32)
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

class CXType(Struct): pass
CXType._fields_ = [
  ('kind', enum_CXTypeKind),
  ('data', (ctypes.c_void_p * 2)),
]
try: (clang_getCursorType:=dll.clang_getCursorType).restype, clang_getCursorType.argtypes = CXType, [CXCursor]
except AttributeError: pass

try: (clang_getTypeSpelling:=dll.clang_getTypeSpelling).restype, clang_getTypeSpelling.argtypes = CXString, [CXType]
except AttributeError: pass

try: (clang_getTypedefDeclUnderlyingType:=dll.clang_getTypedefDeclUnderlyingType).restype, clang_getTypedefDeclUnderlyingType.argtypes = CXType, [CXCursor]
except AttributeError: pass

try: (clang_getEnumDeclIntegerType:=dll.clang_getEnumDeclIntegerType).restype, clang_getEnumDeclIntegerType.argtypes = CXType, [CXCursor]
except AttributeError: pass

try: (clang_getEnumConstantDeclValue:=dll.clang_getEnumConstantDeclValue).restype, clang_getEnumConstantDeclValue.argtypes = ctypes.c_int64, [CXCursor]
except AttributeError: pass

try: (clang_getEnumConstantDeclUnsignedValue:=dll.clang_getEnumConstantDeclUnsignedValue).restype, clang_getEnumConstantDeclUnsignedValue.argtypes = ctypes.c_uint64, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isBitField:=dll.clang_Cursor_isBitField).restype, clang_Cursor_isBitField.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getFieldDeclBitWidth:=dll.clang_getFieldDeclBitWidth).restype, clang_getFieldDeclBitWidth.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getNumArguments:=dll.clang_Cursor_getNumArguments).restype, clang_Cursor_getNumArguments.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getArgument:=dll.clang_Cursor_getArgument).restype, clang_Cursor_getArgument.argtypes = CXCursor, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

enum_CXTemplateArgumentKind = CEnum(ctypes.c_uint32)
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

try: (clang_Cursor_getNumTemplateArguments:=dll.clang_Cursor_getNumTemplateArguments).restype, clang_Cursor_getNumTemplateArguments.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getTemplateArgumentKind:=dll.clang_Cursor_getTemplateArgumentKind).restype, clang_Cursor_getTemplateArgumentKind.argtypes = enum_CXTemplateArgumentKind, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Cursor_getTemplateArgumentType:=dll.clang_Cursor_getTemplateArgumentType).restype, clang_Cursor_getTemplateArgumentType.argtypes = CXType, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Cursor_getTemplateArgumentValue:=dll.clang_Cursor_getTemplateArgumentValue).restype, clang_Cursor_getTemplateArgumentValue.argtypes = ctypes.c_int64, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Cursor_getTemplateArgumentUnsignedValue:=dll.clang_Cursor_getTemplateArgumentUnsignedValue).restype, clang_Cursor_getTemplateArgumentUnsignedValue.argtypes = ctypes.c_uint64, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_equalTypes:=dll.clang_equalTypes).restype, clang_equalTypes.argtypes = ctypes.c_uint32, [CXType, CXType]
except AttributeError: pass

try: (clang_getCanonicalType:=dll.clang_getCanonicalType).restype, clang_getCanonicalType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_isConstQualifiedType:=dll.clang_isConstQualifiedType).restype, clang_isConstQualifiedType.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_Cursor_isMacroFunctionLike:=dll.clang_Cursor_isMacroFunctionLike).restype, clang_Cursor_isMacroFunctionLike.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isMacroBuiltin:=dll.clang_Cursor_isMacroBuiltin).restype, clang_Cursor_isMacroBuiltin.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isFunctionInlined:=dll.clang_Cursor_isFunctionInlined).restype, clang_Cursor_isFunctionInlined.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_isVolatileQualifiedType:=dll.clang_isVolatileQualifiedType).restype, clang_isVolatileQualifiedType.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_isRestrictQualifiedType:=dll.clang_isRestrictQualifiedType).restype, clang_isRestrictQualifiedType.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_getAddressSpace:=dll.clang_getAddressSpace).restype, clang_getAddressSpace.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_getTypedefName:=dll.clang_getTypedefName).restype, clang_getTypedefName.argtypes = CXString, [CXType]
except AttributeError: pass

try: (clang_getPointeeType:=dll.clang_getPointeeType).restype, clang_getPointeeType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getUnqualifiedType:=dll.clang_getUnqualifiedType).restype, clang_getUnqualifiedType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getNonReferenceType:=dll.clang_getNonReferenceType).restype, clang_getNonReferenceType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getTypeDeclaration:=dll.clang_getTypeDeclaration).restype, clang_getTypeDeclaration.argtypes = CXCursor, [CXType]
except AttributeError: pass

try: (clang_getDeclObjCTypeEncoding:=dll.clang_getDeclObjCTypeEncoding).restype, clang_getDeclObjCTypeEncoding.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_Type_getObjCEncoding:=dll.clang_Type_getObjCEncoding).restype, clang_Type_getObjCEncoding.argtypes = CXString, [CXType]
except AttributeError: pass

try: (clang_getTypeKindSpelling:=dll.clang_getTypeKindSpelling).restype, clang_getTypeKindSpelling.argtypes = CXString, [enum_CXTypeKind]
except AttributeError: pass

try: (clang_getFunctionTypeCallingConv:=dll.clang_getFunctionTypeCallingConv).restype, clang_getFunctionTypeCallingConv.argtypes = enum_CXCallingConv, [CXType]
except AttributeError: pass

try: (clang_getResultType:=dll.clang_getResultType).restype, clang_getResultType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getExceptionSpecificationType:=dll.clang_getExceptionSpecificationType).restype, clang_getExceptionSpecificationType.argtypes = ctypes.c_int32, [CXType]
except AttributeError: pass

try: (clang_getNumArgTypes:=dll.clang_getNumArgTypes).restype, clang_getNumArgTypes.argtypes = ctypes.c_int32, [CXType]
except AttributeError: pass

try: (clang_getArgType:=dll.clang_getArgType).restype, clang_getArgType.argtypes = CXType, [CXType, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Type_getObjCObjectBaseType:=dll.clang_Type_getObjCObjectBaseType).restype, clang_Type_getObjCObjectBaseType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_Type_getNumObjCProtocolRefs:=dll.clang_Type_getNumObjCProtocolRefs).restype, clang_Type_getNumObjCProtocolRefs.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_Type_getObjCProtocolDecl:=dll.clang_Type_getObjCProtocolDecl).restype, clang_Type_getObjCProtocolDecl.argtypes = CXCursor, [CXType, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Type_getNumObjCTypeArgs:=dll.clang_Type_getNumObjCTypeArgs).restype, clang_Type_getNumObjCTypeArgs.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_Type_getObjCTypeArg:=dll.clang_Type_getObjCTypeArg).restype, clang_Type_getObjCTypeArg.argtypes = CXType, [CXType, ctypes.c_uint32]
except AttributeError: pass

try: (clang_isFunctionTypeVariadic:=dll.clang_isFunctionTypeVariadic).restype, clang_isFunctionTypeVariadic.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_getCursorResultType:=dll.clang_getCursorResultType).restype, clang_getCursorResultType.argtypes = CXType, [CXCursor]
except AttributeError: pass

try: (clang_getCursorExceptionSpecificationType:=dll.clang_getCursorExceptionSpecificationType).restype, clang_getCursorExceptionSpecificationType.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_isPODType:=dll.clang_isPODType).restype, clang_isPODType.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

try: (clang_getElementType:=dll.clang_getElementType).restype, clang_getElementType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getNumElements:=dll.clang_getNumElements).restype, clang_getNumElements.argtypes = ctypes.c_int64, [CXType]
except AttributeError: pass

try: (clang_getArrayElementType:=dll.clang_getArrayElementType).restype, clang_getArrayElementType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_getArraySize:=dll.clang_getArraySize).restype, clang_getArraySize.argtypes = ctypes.c_int64, [CXType]
except AttributeError: pass

try: (clang_Type_getNamedType:=dll.clang_Type_getNamedType).restype, clang_Type_getNamedType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_Type_isTransparentTagTypedef:=dll.clang_Type_isTransparentTagTypedef).restype, clang_Type_isTransparentTagTypedef.argtypes = ctypes.c_uint32, [CXType]
except AttributeError: pass

enum_CXTypeNullabilityKind = CEnum(ctypes.c_uint32)
CXTypeNullability_NonNull = enum_CXTypeNullabilityKind.define('CXTypeNullability_NonNull', 0)
CXTypeNullability_Nullable = enum_CXTypeNullabilityKind.define('CXTypeNullability_Nullable', 1)
CXTypeNullability_Unspecified = enum_CXTypeNullabilityKind.define('CXTypeNullability_Unspecified', 2)
CXTypeNullability_Invalid = enum_CXTypeNullabilityKind.define('CXTypeNullability_Invalid', 3)
CXTypeNullability_NullableResult = enum_CXTypeNullabilityKind.define('CXTypeNullability_NullableResult', 4)

try: (clang_Type_getNullability:=dll.clang_Type_getNullability).restype, clang_Type_getNullability.argtypes = enum_CXTypeNullabilityKind, [CXType]
except AttributeError: pass

enum_CXTypeLayoutError = CEnum(ctypes.c_int32)
CXTypeLayoutError_Invalid = enum_CXTypeLayoutError.define('CXTypeLayoutError_Invalid', -1)
CXTypeLayoutError_Incomplete = enum_CXTypeLayoutError.define('CXTypeLayoutError_Incomplete', -2)
CXTypeLayoutError_Dependent = enum_CXTypeLayoutError.define('CXTypeLayoutError_Dependent', -3)
CXTypeLayoutError_NotConstantSize = enum_CXTypeLayoutError.define('CXTypeLayoutError_NotConstantSize', -4)
CXTypeLayoutError_InvalidFieldName = enum_CXTypeLayoutError.define('CXTypeLayoutError_InvalidFieldName', -5)
CXTypeLayoutError_Undeduced = enum_CXTypeLayoutError.define('CXTypeLayoutError_Undeduced', -6)

try: (clang_Type_getAlignOf:=dll.clang_Type_getAlignOf).restype, clang_Type_getAlignOf.argtypes = ctypes.c_int64, [CXType]
except AttributeError: pass

try: (clang_Type_getClassType:=dll.clang_Type_getClassType).restype, clang_Type_getClassType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_Type_getSizeOf:=dll.clang_Type_getSizeOf).restype, clang_Type_getSizeOf.argtypes = ctypes.c_int64, [CXType]
except AttributeError: pass

try: (clang_Type_getOffsetOf:=dll.clang_Type_getOffsetOf).restype, clang_Type_getOffsetOf.argtypes = ctypes.c_int64, [CXType, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (clang_Type_getModifiedType:=dll.clang_Type_getModifiedType).restype, clang_Type_getModifiedType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_Type_getValueType:=dll.clang_Type_getValueType).restype, clang_Type_getValueType.argtypes = CXType, [CXType]
except AttributeError: pass

try: (clang_Cursor_getOffsetOfField:=dll.clang_Cursor_getOffsetOfField).restype, clang_Cursor_getOffsetOfField.argtypes = ctypes.c_int64, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isAnonymous:=dll.clang_Cursor_isAnonymous).restype, clang_Cursor_isAnonymous.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isAnonymousRecordDecl:=dll.clang_Cursor_isAnonymousRecordDecl).restype, clang_Cursor_isAnonymousRecordDecl.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isInlineNamespace:=dll.clang_Cursor_isInlineNamespace).restype, clang_Cursor_isInlineNamespace.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

enum_CXRefQualifierKind = CEnum(ctypes.c_uint32)
CXRefQualifier_None = enum_CXRefQualifierKind.define('CXRefQualifier_None', 0)
CXRefQualifier_LValue = enum_CXRefQualifierKind.define('CXRefQualifier_LValue', 1)
CXRefQualifier_RValue = enum_CXRefQualifierKind.define('CXRefQualifier_RValue', 2)

try: (clang_Type_getNumTemplateArguments:=dll.clang_Type_getNumTemplateArguments).restype, clang_Type_getNumTemplateArguments.argtypes = ctypes.c_int32, [CXType]
except AttributeError: pass

try: (clang_Type_getTemplateArgumentAsType:=dll.clang_Type_getTemplateArgumentAsType).restype, clang_Type_getTemplateArgumentAsType.argtypes = CXType, [CXType, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Type_getCXXRefQualifier:=dll.clang_Type_getCXXRefQualifier).restype, clang_Type_getCXXRefQualifier.argtypes = enum_CXRefQualifierKind, [CXType]
except AttributeError: pass

try: (clang_isVirtualBase:=dll.clang_isVirtualBase).restype, clang_isVirtualBase.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getOffsetOfBase:=dll.clang_getOffsetOfBase).restype, clang_getOffsetOfBase.argtypes = ctypes.c_int64, [CXCursor, CXCursor]
except AttributeError: pass

enum_CX_CXXAccessSpecifier = CEnum(ctypes.c_uint32)
CX_CXXInvalidAccessSpecifier = enum_CX_CXXAccessSpecifier.define('CX_CXXInvalidAccessSpecifier', 0)
CX_CXXPublic = enum_CX_CXXAccessSpecifier.define('CX_CXXPublic', 1)
CX_CXXProtected = enum_CX_CXXAccessSpecifier.define('CX_CXXProtected', 2)
CX_CXXPrivate = enum_CX_CXXAccessSpecifier.define('CX_CXXPrivate', 3)

try: (clang_getCXXAccessSpecifier:=dll.clang_getCXXAccessSpecifier).restype, clang_getCXXAccessSpecifier.argtypes = enum_CX_CXXAccessSpecifier, [CXCursor]
except AttributeError: pass

enum_CX_StorageClass = CEnum(ctypes.c_uint32)
CX_SC_Invalid = enum_CX_StorageClass.define('CX_SC_Invalid', 0)
CX_SC_None = enum_CX_StorageClass.define('CX_SC_None', 1)
CX_SC_Extern = enum_CX_StorageClass.define('CX_SC_Extern', 2)
CX_SC_Static = enum_CX_StorageClass.define('CX_SC_Static', 3)
CX_SC_PrivateExtern = enum_CX_StorageClass.define('CX_SC_PrivateExtern', 4)
CX_SC_OpenCLWorkGroupLocal = enum_CX_StorageClass.define('CX_SC_OpenCLWorkGroupLocal', 5)
CX_SC_Auto = enum_CX_StorageClass.define('CX_SC_Auto', 6)
CX_SC_Register = enum_CX_StorageClass.define('CX_SC_Register', 7)

enum_CX_BinaryOperatorKind = CEnum(ctypes.c_uint32)
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

try: (clang_Cursor_getBinaryOpcode:=dll.clang_Cursor_getBinaryOpcode).restype, clang_Cursor_getBinaryOpcode.argtypes = enum_CX_BinaryOperatorKind, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getBinaryOpcodeStr:=dll.clang_Cursor_getBinaryOpcodeStr).restype, clang_Cursor_getBinaryOpcodeStr.argtypes = CXString, [enum_CX_BinaryOperatorKind]
except AttributeError: pass

try: (clang_Cursor_getStorageClass:=dll.clang_Cursor_getStorageClass).restype, clang_Cursor_getStorageClass.argtypes = enum_CX_StorageClass, [CXCursor]
except AttributeError: pass

try: (clang_getNumOverloadedDecls:=dll.clang_getNumOverloadedDecls).restype, clang_getNumOverloadedDecls.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getOverloadedDecl:=dll.clang_getOverloadedDecl).restype, clang_getOverloadedDecl.argtypes = CXCursor, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getIBOutletCollectionType:=dll.clang_getIBOutletCollectionType).restype, clang_getIBOutletCollectionType.argtypes = CXType, [CXCursor]
except AttributeError: pass

enum_CXChildVisitResult = CEnum(ctypes.c_uint32)
CXChildVisit_Break = enum_CXChildVisitResult.define('CXChildVisit_Break', 0)
CXChildVisit_Continue = enum_CXChildVisitResult.define('CXChildVisit_Continue', 1)
CXChildVisit_Recurse = enum_CXChildVisitResult.define('CXChildVisit_Recurse', 2)

CXCursorVisitor = ctypes.CFUNCTYPE(enum_CXChildVisitResult, CXCursor, CXCursor, ctypes.c_void_p)
try: (clang_visitChildren:=dll.clang_visitChildren).restype, clang_visitChildren.argtypes = ctypes.c_uint32, [CXCursor, CXCursorVisitor, CXClientData]
except AttributeError: pass

class struct__CXChildVisitResult(Struct): pass
CXCursorVisitorBlock = ctypes.POINTER(struct__CXChildVisitResult)
try: (clang_visitChildrenWithBlock:=dll.clang_visitChildrenWithBlock).restype, clang_visitChildrenWithBlock.argtypes = ctypes.c_uint32, [CXCursor, CXCursorVisitorBlock]
except AttributeError: pass

try: (clang_getCursorUSR:=dll.clang_getCursorUSR).restype, clang_getCursorUSR.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_constructUSR_ObjCClass:=dll.clang_constructUSR_ObjCClass).restype, clang_constructUSR_ObjCClass.argtypes = CXString, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (clang_constructUSR_ObjCCategory:=dll.clang_constructUSR_ObjCCategory).restype, clang_constructUSR_ObjCCategory.argtypes = CXString, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (clang_constructUSR_ObjCProtocol:=dll.clang_constructUSR_ObjCProtocol).restype, clang_constructUSR_ObjCProtocol.argtypes = CXString, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (clang_constructUSR_ObjCIvar:=dll.clang_constructUSR_ObjCIvar).restype, clang_constructUSR_ObjCIvar.argtypes = CXString, [ctypes.POINTER(ctypes.c_char), CXString]
except AttributeError: pass

try: (clang_constructUSR_ObjCMethod:=dll.clang_constructUSR_ObjCMethod).restype, clang_constructUSR_ObjCMethod.argtypes = CXString, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, CXString]
except AttributeError: pass

try: (clang_constructUSR_ObjCProperty:=dll.clang_constructUSR_ObjCProperty).restype, clang_constructUSR_ObjCProperty.argtypes = CXString, [ctypes.POINTER(ctypes.c_char), CXString]
except AttributeError: pass

try: (clang_getCursorSpelling:=dll.clang_getCursorSpelling).restype, clang_getCursorSpelling.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getSpellingNameRange:=dll.clang_Cursor_getSpellingNameRange).restype, clang_Cursor_getSpellingNameRange.argtypes = CXSourceRange, [CXCursor, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

CXPrintingPolicy = ctypes.c_void_p
enum_CXPrintingPolicyProperty = CEnum(ctypes.c_uint32)
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

try: (clang_PrintingPolicy_getProperty:=dll.clang_PrintingPolicy_getProperty).restype, clang_PrintingPolicy_getProperty.argtypes = ctypes.c_uint32, [CXPrintingPolicy, enum_CXPrintingPolicyProperty]
except AttributeError: pass

try: (clang_PrintingPolicy_setProperty:=dll.clang_PrintingPolicy_setProperty).restype, clang_PrintingPolicy_setProperty.argtypes = None, [CXPrintingPolicy, enum_CXPrintingPolicyProperty, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCursorPrintingPolicy:=dll.clang_getCursorPrintingPolicy).restype, clang_getCursorPrintingPolicy.argtypes = CXPrintingPolicy, [CXCursor]
except AttributeError: pass

try: (clang_PrintingPolicy_dispose:=dll.clang_PrintingPolicy_dispose).restype, clang_PrintingPolicy_dispose.argtypes = None, [CXPrintingPolicy]
except AttributeError: pass

try: (clang_getCursorPrettyPrinted:=dll.clang_getCursorPrettyPrinted).restype, clang_getCursorPrettyPrinted.argtypes = CXString, [CXCursor, CXPrintingPolicy]
except AttributeError: pass

try: (clang_getTypePrettyPrinted:=dll.clang_getTypePrettyPrinted).restype, clang_getTypePrettyPrinted.argtypes = CXString, [CXType, CXPrintingPolicy]
except AttributeError: pass

try: (clang_getCursorDisplayName:=dll.clang_getCursorDisplayName).restype, clang_getCursorDisplayName.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_getCursorReferenced:=dll.clang_getCursorReferenced).restype, clang_getCursorReferenced.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_getCursorDefinition:=dll.clang_getCursorDefinition).restype, clang_getCursorDefinition.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_isCursorDefinition:=dll.clang_isCursorDefinition).restype, clang_isCursorDefinition.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getCanonicalCursor:=dll.clang_getCanonicalCursor).restype, clang_getCanonicalCursor.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getObjCSelectorIndex:=dll.clang_Cursor_getObjCSelectorIndex).restype, clang_Cursor_getObjCSelectorIndex.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isDynamicCall:=dll.clang_Cursor_isDynamicCall).restype, clang_Cursor_isDynamicCall.argtypes = ctypes.c_int32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getReceiverType:=dll.clang_Cursor_getReceiverType).restype, clang_Cursor_getReceiverType.argtypes = CXType, [CXCursor]
except AttributeError: pass

CXObjCPropertyAttrKind = CEnum(ctypes.c_uint32)
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

try: (clang_Cursor_getObjCPropertyAttributes:=dll.clang_Cursor_getObjCPropertyAttributes).restype, clang_Cursor_getObjCPropertyAttributes.argtypes = ctypes.c_uint32, [CXCursor, ctypes.c_uint32]
except AttributeError: pass

try: (clang_Cursor_getObjCPropertyGetterName:=dll.clang_Cursor_getObjCPropertyGetterName).restype, clang_Cursor_getObjCPropertyGetterName.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getObjCPropertySetterName:=dll.clang_Cursor_getObjCPropertySetterName).restype, clang_Cursor_getObjCPropertySetterName.argtypes = CXString, [CXCursor]
except AttributeError: pass

CXObjCDeclQualifierKind = CEnum(ctypes.c_uint32)
CXObjCDeclQualifier_None = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_None', 0)
CXObjCDeclQualifier_In = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_In', 1)
CXObjCDeclQualifier_Inout = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Inout', 2)
CXObjCDeclQualifier_Out = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Out', 4)
CXObjCDeclQualifier_Bycopy = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Bycopy', 8)
CXObjCDeclQualifier_Byref = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Byref', 16)
CXObjCDeclQualifier_Oneway = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Oneway', 32)

try: (clang_Cursor_getObjCDeclQualifiers:=dll.clang_Cursor_getObjCDeclQualifiers).restype, clang_Cursor_getObjCDeclQualifiers.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isObjCOptional:=dll.clang_Cursor_isObjCOptional).restype, clang_Cursor_isObjCOptional.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isVariadic:=dll.clang_Cursor_isVariadic).restype, clang_Cursor_isVariadic.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_isExternalSymbol:=dll.clang_Cursor_isExternalSymbol).restype, clang_Cursor_isExternalSymbol.argtypes = ctypes.c_uint32, [CXCursor, ctypes.POINTER(CXString), ctypes.POINTER(CXString), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_Cursor_getCommentRange:=dll.clang_Cursor_getCommentRange).restype, clang_Cursor_getCommentRange.argtypes = CXSourceRange, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getRawCommentText:=dll.clang_Cursor_getRawCommentText).restype, clang_Cursor_getRawCommentText.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getBriefCommentText:=dll.clang_Cursor_getBriefCommentText).restype, clang_Cursor_getBriefCommentText.argtypes = CXString, [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getMangling:=dll.clang_Cursor_getMangling).restype, clang_Cursor_getMangling.argtypes = CXString, [CXCursor]
except AttributeError: pass

class CXStringSet(Struct): pass
CXStringSet._fields_ = [
  ('Strings', ctypes.POINTER(CXString)),
  ('Count', ctypes.c_uint32),
]
try: (clang_Cursor_getCXXManglings:=dll.clang_Cursor_getCXXManglings).restype, clang_Cursor_getCXXManglings.argtypes = ctypes.POINTER(CXStringSet), [CXCursor]
except AttributeError: pass

try: (clang_Cursor_getObjCManglings:=dll.clang_Cursor_getObjCManglings).restype, clang_Cursor_getObjCManglings.argtypes = ctypes.POINTER(CXStringSet), [CXCursor]
except AttributeError: pass

CXModule = ctypes.c_void_p
try: (clang_Cursor_getModule:=dll.clang_Cursor_getModule).restype, clang_Cursor_getModule.argtypes = CXModule, [CXCursor]
except AttributeError: pass

try: (clang_getModuleForFile:=dll.clang_getModuleForFile).restype, clang_getModuleForFile.argtypes = CXModule, [CXTranslationUnit, CXFile]
except AttributeError: pass

try: (clang_Module_getASTFile:=dll.clang_Module_getASTFile).restype, clang_Module_getASTFile.argtypes = CXFile, [CXModule]
except AttributeError: pass

try: (clang_Module_getParent:=dll.clang_Module_getParent).restype, clang_Module_getParent.argtypes = CXModule, [CXModule]
except AttributeError: pass

try: (clang_Module_getName:=dll.clang_Module_getName).restype, clang_Module_getName.argtypes = CXString, [CXModule]
except AttributeError: pass

try: (clang_Module_getFullName:=dll.clang_Module_getFullName).restype, clang_Module_getFullName.argtypes = CXString, [CXModule]
except AttributeError: pass

try: (clang_Module_isSystem:=dll.clang_Module_isSystem).restype, clang_Module_isSystem.argtypes = ctypes.c_int32, [CXModule]
except AttributeError: pass

try: (clang_Module_getNumTopLevelHeaders:=dll.clang_Module_getNumTopLevelHeaders).restype, clang_Module_getNumTopLevelHeaders.argtypes = ctypes.c_uint32, [CXTranslationUnit, CXModule]
except AttributeError: pass

try: (clang_Module_getTopLevelHeader:=dll.clang_Module_getTopLevelHeader).restype, clang_Module_getTopLevelHeader.argtypes = CXFile, [CXTranslationUnit, CXModule, ctypes.c_uint32]
except AttributeError: pass

try: (clang_CXXConstructor_isConvertingConstructor:=dll.clang_CXXConstructor_isConvertingConstructor).restype, clang_CXXConstructor_isConvertingConstructor.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXConstructor_isCopyConstructor:=dll.clang_CXXConstructor_isCopyConstructor).restype, clang_CXXConstructor_isCopyConstructor.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXConstructor_isDefaultConstructor:=dll.clang_CXXConstructor_isDefaultConstructor).restype, clang_CXXConstructor_isDefaultConstructor.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXConstructor_isMoveConstructor:=dll.clang_CXXConstructor_isMoveConstructor).restype, clang_CXXConstructor_isMoveConstructor.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXField_isMutable:=dll.clang_CXXField_isMutable).restype, clang_CXXField_isMutable.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isDefaulted:=dll.clang_CXXMethod_isDefaulted).restype, clang_CXXMethod_isDefaulted.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isDeleted:=dll.clang_CXXMethod_isDeleted).restype, clang_CXXMethod_isDeleted.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isPureVirtual:=dll.clang_CXXMethod_isPureVirtual).restype, clang_CXXMethod_isPureVirtual.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isStatic:=dll.clang_CXXMethod_isStatic).restype, clang_CXXMethod_isStatic.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isVirtual:=dll.clang_CXXMethod_isVirtual).restype, clang_CXXMethod_isVirtual.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isCopyAssignmentOperator:=dll.clang_CXXMethod_isCopyAssignmentOperator).restype, clang_CXXMethod_isCopyAssignmentOperator.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isMoveAssignmentOperator:=dll.clang_CXXMethod_isMoveAssignmentOperator).restype, clang_CXXMethod_isMoveAssignmentOperator.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isExplicit:=dll.clang_CXXMethod_isExplicit).restype, clang_CXXMethod_isExplicit.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXRecord_isAbstract:=dll.clang_CXXRecord_isAbstract).restype, clang_CXXRecord_isAbstract.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_EnumDecl_isScoped:=dll.clang_EnumDecl_isScoped).restype, clang_EnumDecl_isScoped.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_CXXMethod_isConst:=dll.clang_CXXMethod_isConst).restype, clang_CXXMethod_isConst.argtypes = ctypes.c_uint32, [CXCursor]
except AttributeError: pass

try: (clang_getTemplateCursorKind:=dll.clang_getTemplateCursorKind).restype, clang_getTemplateCursorKind.argtypes = enum_CXCursorKind, [CXCursor]
except AttributeError: pass

try: (clang_getSpecializedCursorTemplate:=dll.clang_getSpecializedCursorTemplate).restype, clang_getSpecializedCursorTemplate.argtypes = CXCursor, [CXCursor]
except AttributeError: pass

try: (clang_getCursorReferenceNameRange:=dll.clang_getCursorReferenceNameRange).restype, clang_getCursorReferenceNameRange.argtypes = CXSourceRange, [CXCursor, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

enum_CXNameRefFlags = CEnum(ctypes.c_uint32)
CXNameRange_WantQualifier = enum_CXNameRefFlags.define('CXNameRange_WantQualifier', 1)
CXNameRange_WantTemplateArgs = enum_CXNameRefFlags.define('CXNameRange_WantTemplateArgs', 2)
CXNameRange_WantSinglePiece = enum_CXNameRefFlags.define('CXNameRange_WantSinglePiece', 4)

enum_CXTokenKind = CEnum(ctypes.c_uint32)
CXToken_Punctuation = enum_CXTokenKind.define('CXToken_Punctuation', 0)
CXToken_Keyword = enum_CXTokenKind.define('CXToken_Keyword', 1)
CXToken_Identifier = enum_CXTokenKind.define('CXToken_Identifier', 2)
CXToken_Literal = enum_CXTokenKind.define('CXToken_Literal', 3)
CXToken_Comment = enum_CXTokenKind.define('CXToken_Comment', 4)

CXTokenKind = enum_CXTokenKind
class CXToken(Struct): pass
CXToken._fields_ = [
  ('int_data', (ctypes.c_uint32 * 4)),
  ('ptr_data', ctypes.c_void_p),
]
try: (clang_getToken:=dll.clang_getToken).restype, clang_getToken.argtypes = ctypes.POINTER(CXToken), [CXTranslationUnit, CXSourceLocation]
except AttributeError: pass

try: (clang_getTokenKind:=dll.clang_getTokenKind).restype, clang_getTokenKind.argtypes = CXTokenKind, [CXToken]
except AttributeError: pass

try: (clang_getTokenSpelling:=dll.clang_getTokenSpelling).restype, clang_getTokenSpelling.argtypes = CXString, [CXTranslationUnit, CXToken]
except AttributeError: pass

try: (clang_getTokenLocation:=dll.clang_getTokenLocation).restype, clang_getTokenLocation.argtypes = CXSourceLocation, [CXTranslationUnit, CXToken]
except AttributeError: pass

try: (clang_getTokenExtent:=dll.clang_getTokenExtent).restype, clang_getTokenExtent.argtypes = CXSourceRange, [CXTranslationUnit, CXToken]
except AttributeError: pass

try: (clang_tokenize:=dll.clang_tokenize).restype, clang_tokenize.argtypes = None, [CXTranslationUnit, CXSourceRange, ctypes.POINTER(ctypes.POINTER(CXToken)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_annotateTokens:=dll.clang_annotateTokens).restype, clang_annotateTokens.argtypes = None, [CXTranslationUnit, ctypes.POINTER(CXToken), ctypes.c_uint32, ctypes.POINTER(CXCursor)]
except AttributeError: pass

try: (clang_disposeTokens:=dll.clang_disposeTokens).restype, clang_disposeTokens.argtypes = None, [CXTranslationUnit, ctypes.POINTER(CXToken), ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCursorKindSpelling:=dll.clang_getCursorKindSpelling).restype, clang_getCursorKindSpelling.argtypes = CXString, [enum_CXCursorKind]
except AttributeError: pass

try: (clang_getDefinitionSpellingAndExtent:=dll.clang_getDefinitionSpellingAndExtent).restype, clang_getDefinitionSpellingAndExtent.argtypes = None, [CXCursor, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_enableStackTraces:=dll.clang_enableStackTraces).restype, clang_enableStackTraces.argtypes = None, []
except AttributeError: pass

try: (clang_executeOnThread:=dll.clang_executeOnThread).restype, clang_executeOnThread.argtypes = None, [ctypes.CFUNCTYPE(None, ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32]
except AttributeError: pass

CXCompletionString = ctypes.c_void_p
class CXCompletionResult(Struct): pass
CXCompletionResult._fields_ = [
  ('CursorKind', enum_CXCursorKind),
  ('CompletionString', CXCompletionString),
]
enum_CXCompletionChunkKind = CEnum(ctypes.c_uint32)
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

try: (clang_getCompletionChunkKind:=dll.clang_getCompletionChunkKind).restype, clang_getCompletionChunkKind.argtypes = enum_CXCompletionChunkKind, [CXCompletionString, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCompletionChunkText:=dll.clang_getCompletionChunkText).restype, clang_getCompletionChunkText.argtypes = CXString, [CXCompletionString, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCompletionChunkCompletionString:=dll.clang_getCompletionChunkCompletionString).restype, clang_getCompletionChunkCompletionString.argtypes = CXCompletionString, [CXCompletionString, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getNumCompletionChunks:=dll.clang_getNumCompletionChunks).restype, clang_getNumCompletionChunks.argtypes = ctypes.c_uint32, [CXCompletionString]
except AttributeError: pass

try: (clang_getCompletionPriority:=dll.clang_getCompletionPriority).restype, clang_getCompletionPriority.argtypes = ctypes.c_uint32, [CXCompletionString]
except AttributeError: pass

try: (clang_getCompletionAvailability:=dll.clang_getCompletionAvailability).restype, clang_getCompletionAvailability.argtypes = enum_CXAvailabilityKind, [CXCompletionString]
except AttributeError: pass

try: (clang_getCompletionNumAnnotations:=dll.clang_getCompletionNumAnnotations).restype, clang_getCompletionNumAnnotations.argtypes = ctypes.c_uint32, [CXCompletionString]
except AttributeError: pass

try: (clang_getCompletionAnnotation:=dll.clang_getCompletionAnnotation).restype, clang_getCompletionAnnotation.argtypes = CXString, [CXCompletionString, ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCompletionParent:=dll.clang_getCompletionParent).restype, clang_getCompletionParent.argtypes = CXString, [CXCompletionString, ctypes.POINTER(enum_CXCursorKind)]
except AttributeError: pass

try: (clang_getCompletionBriefComment:=dll.clang_getCompletionBriefComment).restype, clang_getCompletionBriefComment.argtypes = CXString, [CXCompletionString]
except AttributeError: pass

try: (clang_getCursorCompletionString:=dll.clang_getCursorCompletionString).restype, clang_getCursorCompletionString.argtypes = CXCompletionString, [CXCursor]
except AttributeError: pass

class CXCodeCompleteResults(Struct): pass
CXCodeCompleteResults._fields_ = [
  ('Results', ctypes.POINTER(CXCompletionResult)),
  ('NumResults', ctypes.c_uint32),
]
try: (clang_getCompletionNumFixIts:=dll.clang_getCompletionNumFixIts).restype, clang_getCompletionNumFixIts.argtypes = ctypes.c_uint32, [ctypes.POINTER(CXCodeCompleteResults), ctypes.c_uint32]
except AttributeError: pass

try: (clang_getCompletionFixIt:=dll.clang_getCompletionFixIt).restype, clang_getCompletionFixIt.argtypes = CXString, [ctypes.POINTER(CXCodeCompleteResults), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CXSourceRange)]
except AttributeError: pass

enum_CXCodeComplete_Flags = CEnum(ctypes.c_uint32)
CXCodeComplete_IncludeMacros = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeMacros', 1)
CXCodeComplete_IncludeCodePatterns = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCodePatterns', 2)
CXCodeComplete_IncludeBriefComments = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeBriefComments', 4)
CXCodeComplete_SkipPreamble = enum_CXCodeComplete_Flags.define('CXCodeComplete_SkipPreamble', 8)
CXCodeComplete_IncludeCompletionsWithFixIts = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCompletionsWithFixIts', 16)

enum_CXCompletionContext = CEnum(ctypes.c_uint32)
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

try: (clang_defaultCodeCompleteOptions:=dll.clang_defaultCodeCompleteOptions).restype, clang_defaultCodeCompleteOptions.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (clang_codeCompleteAt:=dll.clang_codeCompleteAt).restype, clang_codeCompleteAt.argtypes = ctypes.POINTER(CXCodeCompleteResults), [CXTranslationUnit, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (clang_sortCodeCompletionResults:=dll.clang_sortCodeCompletionResults).restype, clang_sortCodeCompletionResults.argtypes = None, [ctypes.POINTER(CXCompletionResult), ctypes.c_uint32]
except AttributeError: pass

try: (clang_disposeCodeCompleteResults:=dll.clang_disposeCodeCompleteResults).restype, clang_disposeCodeCompleteResults.argtypes = None, [ctypes.POINTER(CXCodeCompleteResults)]
except AttributeError: pass

try: (clang_codeCompleteGetNumDiagnostics:=dll.clang_codeCompleteGetNumDiagnostics).restype, clang_codeCompleteGetNumDiagnostics.argtypes = ctypes.c_uint32, [ctypes.POINTER(CXCodeCompleteResults)]
except AttributeError: pass

try: (clang_codeCompleteGetDiagnostic:=dll.clang_codeCompleteGetDiagnostic).restype, clang_codeCompleteGetDiagnostic.argtypes = CXDiagnostic, [ctypes.POINTER(CXCodeCompleteResults), ctypes.c_uint32]
except AttributeError: pass

try: (clang_codeCompleteGetContexts:=dll.clang_codeCompleteGetContexts).restype, clang_codeCompleteGetContexts.argtypes = ctypes.c_uint64, [ctypes.POINTER(CXCodeCompleteResults)]
except AttributeError: pass

try: (clang_codeCompleteGetContainerKind:=dll.clang_codeCompleteGetContainerKind).restype, clang_codeCompleteGetContainerKind.argtypes = enum_CXCursorKind, [ctypes.POINTER(CXCodeCompleteResults), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_codeCompleteGetContainerUSR:=dll.clang_codeCompleteGetContainerUSR).restype, clang_codeCompleteGetContainerUSR.argtypes = CXString, [ctypes.POINTER(CXCodeCompleteResults)]
except AttributeError: pass

try: (clang_codeCompleteGetObjCSelector:=dll.clang_codeCompleteGetObjCSelector).restype, clang_codeCompleteGetObjCSelector.argtypes = CXString, [ctypes.POINTER(CXCodeCompleteResults)]
except AttributeError: pass

try: (clang_getClangVersion:=dll.clang_getClangVersion).restype, clang_getClangVersion.argtypes = CXString, []
except AttributeError: pass

try: (clang_toggleCrashRecovery:=dll.clang_toggleCrashRecovery).restype, clang_toggleCrashRecovery.argtypes = None, [ctypes.c_uint32]
except AttributeError: pass

CXInclusionVisitor = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(CXSourceLocation), ctypes.c_uint32, ctypes.c_void_p)
try: (clang_getInclusions:=dll.clang_getInclusions).restype, clang_getInclusions.argtypes = None, [CXTranslationUnit, CXInclusionVisitor, CXClientData]
except AttributeError: pass

CXEvalResultKind = CEnum(ctypes.c_uint32)
CXEval_Int = CXEvalResultKind.define('CXEval_Int', 1)
CXEval_Float = CXEvalResultKind.define('CXEval_Float', 2)
CXEval_ObjCStrLiteral = CXEvalResultKind.define('CXEval_ObjCStrLiteral', 3)
CXEval_StrLiteral = CXEvalResultKind.define('CXEval_StrLiteral', 4)
CXEval_CFStr = CXEvalResultKind.define('CXEval_CFStr', 5)
CXEval_Other = CXEvalResultKind.define('CXEval_Other', 6)
CXEval_UnExposed = CXEvalResultKind.define('CXEval_UnExposed', 0)

CXEvalResult = ctypes.c_void_p
try: (clang_Cursor_Evaluate:=dll.clang_Cursor_Evaluate).restype, clang_Cursor_Evaluate.argtypes = CXEvalResult, [CXCursor]
except AttributeError: pass

try: (clang_EvalResult_getKind:=dll.clang_EvalResult_getKind).restype, clang_EvalResult_getKind.argtypes = CXEvalResultKind, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_getAsInt:=dll.clang_EvalResult_getAsInt).restype, clang_EvalResult_getAsInt.argtypes = ctypes.c_int32, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_getAsLongLong:=dll.clang_EvalResult_getAsLongLong).restype, clang_EvalResult_getAsLongLong.argtypes = ctypes.c_int64, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_isUnsignedInt:=dll.clang_EvalResult_isUnsignedInt).restype, clang_EvalResult_isUnsignedInt.argtypes = ctypes.c_uint32, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_getAsUnsigned:=dll.clang_EvalResult_getAsUnsigned).restype, clang_EvalResult_getAsUnsigned.argtypes = ctypes.c_uint64, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_getAsDouble:=dll.clang_EvalResult_getAsDouble).restype, clang_EvalResult_getAsDouble.argtypes = ctypes.c_double, [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_getAsStr:=dll.clang_EvalResult_getAsStr).restype, clang_EvalResult_getAsStr.argtypes = ctypes.POINTER(ctypes.c_char), [CXEvalResult]
except AttributeError: pass

try: (clang_EvalResult_dispose:=dll.clang_EvalResult_dispose).restype, clang_EvalResult_dispose.argtypes = None, [CXEvalResult]
except AttributeError: pass

CXRemapping = ctypes.c_void_p
try: (clang_getRemappings:=dll.clang_getRemappings).restype, clang_getRemappings.argtypes = CXRemapping, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (clang_getRemappingsFromFileList:=dll.clang_getRemappingsFromFileList).restype, clang_getRemappingsFromFileList.argtypes = CXRemapping, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_uint32]
except AttributeError: pass

try: (clang_remap_getNumFiles:=dll.clang_remap_getNumFiles).restype, clang_remap_getNumFiles.argtypes = ctypes.c_uint32, [CXRemapping]
except AttributeError: pass

try: (clang_remap_getFilenames:=dll.clang_remap_getFilenames).restype, clang_remap_getFilenames.argtypes = None, [CXRemapping, ctypes.c_uint32, ctypes.POINTER(CXString), ctypes.POINTER(CXString)]
except AttributeError: pass

try: (clang_remap_dispose:=dll.clang_remap_dispose).restype, clang_remap_dispose.argtypes = None, [CXRemapping]
except AttributeError: pass

enum_CXVisitorResult = CEnum(ctypes.c_uint32)
CXVisit_Break = enum_CXVisitorResult.define('CXVisit_Break', 0)
CXVisit_Continue = enum_CXVisitorResult.define('CXVisit_Continue', 1)

class struct_CXCursorAndRangeVisitor(Struct): pass
struct_CXCursorAndRangeVisitor._fields_ = [
  ('context', ctypes.c_void_p),
  ('visit', ctypes.CFUNCTYPE(enum_CXVisitorResult, ctypes.c_void_p, CXCursor, CXSourceRange)),
]
CXCursorAndRangeVisitor = struct_CXCursorAndRangeVisitor
CXResult = CEnum(ctypes.c_uint32)
CXResult_Success = CXResult.define('CXResult_Success', 0)
CXResult_Invalid = CXResult.define('CXResult_Invalid', 1)
CXResult_VisitBreak = CXResult.define('CXResult_VisitBreak', 2)

try: (clang_findReferencesInFile:=dll.clang_findReferencesInFile).restype, clang_findReferencesInFile.argtypes = CXResult, [CXCursor, CXFile, CXCursorAndRangeVisitor]
except AttributeError: pass

try: (clang_findIncludesInFile:=dll.clang_findIncludesInFile).restype, clang_findIncludesInFile.argtypes = CXResult, [CXTranslationUnit, CXFile, CXCursorAndRangeVisitor]
except AttributeError: pass

class struct__CXCursorAndRangeVisitorBlock(Struct): pass
CXCursorAndRangeVisitorBlock = ctypes.POINTER(struct__CXCursorAndRangeVisitorBlock)
try: (clang_findReferencesInFileWithBlock:=dll.clang_findReferencesInFileWithBlock).restype, clang_findReferencesInFileWithBlock.argtypes = CXResult, [CXCursor, CXFile, CXCursorAndRangeVisitorBlock]
except AttributeError: pass

try: (clang_findIncludesInFileWithBlock:=dll.clang_findIncludesInFileWithBlock).restype, clang_findIncludesInFileWithBlock.argtypes = CXResult, [CXTranslationUnit, CXFile, CXCursorAndRangeVisitorBlock]
except AttributeError: pass

CXIdxClientFile = ctypes.c_void_p
CXIdxClientEntity = ctypes.c_void_p
CXIdxClientContainer = ctypes.c_void_p
CXIdxClientASTFile = ctypes.c_void_p
class CXIdxLoc(Struct): pass
CXIdxLoc._fields_ = [
  ('ptr_data', (ctypes.c_void_p * 2)),
  ('int_data', ctypes.c_uint32),
]
class CXIdxIncludedFileInfo(Struct): pass
CXIdxIncludedFileInfo._fields_ = [
  ('hashLoc', CXIdxLoc),
  ('filename', ctypes.POINTER(ctypes.c_char)),
  ('file', CXFile),
  ('isImport', ctypes.c_int32),
  ('isAngled', ctypes.c_int32),
  ('isModuleImport', ctypes.c_int32),
]
class CXIdxImportedASTFileInfo(Struct): pass
CXIdxImportedASTFileInfo._fields_ = [
  ('file', CXFile),
  ('module', CXModule),
  ('loc', CXIdxLoc),
  ('isImplicit', ctypes.c_int32),
]
CXIdxEntityKind = CEnum(ctypes.c_uint32)
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

CXIdxEntityLanguage = CEnum(ctypes.c_uint32)
CXIdxEntityLang_None = CXIdxEntityLanguage.define('CXIdxEntityLang_None', 0)
CXIdxEntityLang_C = CXIdxEntityLanguage.define('CXIdxEntityLang_C', 1)
CXIdxEntityLang_ObjC = CXIdxEntityLanguage.define('CXIdxEntityLang_ObjC', 2)
CXIdxEntityLang_CXX = CXIdxEntityLanguage.define('CXIdxEntityLang_CXX', 3)
CXIdxEntityLang_Swift = CXIdxEntityLanguage.define('CXIdxEntityLang_Swift', 4)

CXIdxEntityCXXTemplateKind = CEnum(ctypes.c_uint32)
CXIdxEntity_NonTemplate = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_NonTemplate', 0)
CXIdxEntity_Template = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_Template', 1)
CXIdxEntity_TemplatePartialSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplatePartialSpecialization', 2)
CXIdxEntity_TemplateSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplateSpecialization', 3)

CXIdxAttrKind = CEnum(ctypes.c_uint32)
CXIdxAttr_Unexposed = CXIdxAttrKind.define('CXIdxAttr_Unexposed', 0)
CXIdxAttr_IBAction = CXIdxAttrKind.define('CXIdxAttr_IBAction', 1)
CXIdxAttr_IBOutlet = CXIdxAttrKind.define('CXIdxAttr_IBOutlet', 2)
CXIdxAttr_IBOutletCollection = CXIdxAttrKind.define('CXIdxAttr_IBOutletCollection', 3)

class CXIdxAttrInfo(Struct): pass
CXIdxAttrInfo._fields_ = [
  ('kind', CXIdxAttrKind),
  ('cursor', CXCursor),
  ('loc', CXIdxLoc),
]
class CXIdxEntityInfo(Struct): pass
CXIdxEntityInfo._fields_ = [
  ('kind', CXIdxEntityKind),
  ('templateKind', CXIdxEntityCXXTemplateKind),
  ('lang', CXIdxEntityLanguage),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('USR', ctypes.POINTER(ctypes.c_char)),
  ('cursor', CXCursor),
  ('attributes', ctypes.POINTER(ctypes.POINTER(CXIdxAttrInfo))),
  ('numAttributes', ctypes.c_uint32),
]
class CXIdxContainerInfo(Struct): pass
CXIdxContainerInfo._fields_ = [
  ('cursor', CXCursor),
]
class CXIdxIBOutletCollectionAttrInfo(Struct): pass
CXIdxIBOutletCollectionAttrInfo._fields_ = [
  ('attrInfo', ctypes.POINTER(CXIdxAttrInfo)),
  ('objcClass', ctypes.POINTER(CXIdxEntityInfo)),
  ('classCursor', CXCursor),
  ('classLoc', CXIdxLoc),
]
CXIdxDeclInfoFlags = CEnum(ctypes.c_uint32)
CXIdxDeclFlag_Skipped = CXIdxDeclInfoFlags.define('CXIdxDeclFlag_Skipped', 1)

class CXIdxDeclInfo(Struct): pass
CXIdxDeclInfo._fields_ = [
  ('entityInfo', ctypes.POINTER(CXIdxEntityInfo)),
  ('cursor', CXCursor),
  ('loc', CXIdxLoc),
  ('semanticContainer', ctypes.POINTER(CXIdxContainerInfo)),
  ('lexicalContainer', ctypes.POINTER(CXIdxContainerInfo)),
  ('isRedeclaration', ctypes.c_int32),
  ('isDefinition', ctypes.c_int32),
  ('isContainer', ctypes.c_int32),
  ('declAsContainer', ctypes.POINTER(CXIdxContainerInfo)),
  ('isImplicit', ctypes.c_int32),
  ('attributes', ctypes.POINTER(ctypes.POINTER(CXIdxAttrInfo))),
  ('numAttributes', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
CXIdxObjCContainerKind = CEnum(ctypes.c_uint32)
CXIdxObjCContainer_ForwardRef = CXIdxObjCContainerKind.define('CXIdxObjCContainer_ForwardRef', 0)
CXIdxObjCContainer_Interface = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Interface', 1)
CXIdxObjCContainer_Implementation = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Implementation', 2)

class CXIdxObjCContainerDeclInfo(Struct): pass
CXIdxObjCContainerDeclInfo._fields_ = [
  ('declInfo', ctypes.POINTER(CXIdxDeclInfo)),
  ('kind', CXIdxObjCContainerKind),
]
class CXIdxBaseClassInfo(Struct): pass
CXIdxBaseClassInfo._fields_ = [
  ('base', ctypes.POINTER(CXIdxEntityInfo)),
  ('cursor', CXCursor),
  ('loc', CXIdxLoc),
]
class CXIdxObjCProtocolRefInfo(Struct): pass
CXIdxObjCProtocolRefInfo._fields_ = [
  ('protocol', ctypes.POINTER(CXIdxEntityInfo)),
  ('cursor', CXCursor),
  ('loc', CXIdxLoc),
]
class CXIdxObjCProtocolRefListInfo(Struct): pass
CXIdxObjCProtocolRefListInfo._fields_ = [
  ('protocols', ctypes.POINTER(ctypes.POINTER(CXIdxObjCProtocolRefInfo))),
  ('numProtocols', ctypes.c_uint32),
]
class CXIdxObjCInterfaceDeclInfo(Struct): pass
CXIdxObjCInterfaceDeclInfo._fields_ = [
  ('containerInfo', ctypes.POINTER(CXIdxObjCContainerDeclInfo)),
  ('superInfo', ctypes.POINTER(CXIdxBaseClassInfo)),
  ('protocols', ctypes.POINTER(CXIdxObjCProtocolRefListInfo)),
]
class CXIdxObjCCategoryDeclInfo(Struct): pass
CXIdxObjCCategoryDeclInfo._fields_ = [
  ('containerInfo', ctypes.POINTER(CXIdxObjCContainerDeclInfo)),
  ('objcClass', ctypes.POINTER(CXIdxEntityInfo)),
  ('classCursor', CXCursor),
  ('classLoc', CXIdxLoc),
  ('protocols', ctypes.POINTER(CXIdxObjCProtocolRefListInfo)),
]
class CXIdxObjCPropertyDeclInfo(Struct): pass
CXIdxObjCPropertyDeclInfo._fields_ = [
  ('declInfo', ctypes.POINTER(CXIdxDeclInfo)),
  ('getter', ctypes.POINTER(CXIdxEntityInfo)),
  ('setter', ctypes.POINTER(CXIdxEntityInfo)),
]
class CXIdxCXXClassDeclInfo(Struct): pass
CXIdxCXXClassDeclInfo._fields_ = [
  ('declInfo', ctypes.POINTER(CXIdxDeclInfo)),
  ('bases', ctypes.POINTER(ctypes.POINTER(CXIdxBaseClassInfo))),
  ('numBases', ctypes.c_uint32),
]
CXIdxEntityRefKind = CEnum(ctypes.c_uint32)
CXIdxEntityRef_Direct = CXIdxEntityRefKind.define('CXIdxEntityRef_Direct', 1)
CXIdxEntityRef_Implicit = CXIdxEntityRefKind.define('CXIdxEntityRef_Implicit', 2)

CXSymbolRole = CEnum(ctypes.c_uint32)
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

class CXIdxEntityRefInfo(Struct): pass
CXIdxEntityRefInfo._fields_ = [
  ('kind', CXIdxEntityRefKind),
  ('cursor', CXCursor),
  ('loc', CXIdxLoc),
  ('referencedEntity', ctypes.POINTER(CXIdxEntityInfo)),
  ('parentEntity', ctypes.POINTER(CXIdxEntityInfo)),
  ('container', ctypes.POINTER(CXIdxContainerInfo)),
  ('role', CXSymbolRole),
]
class IndexerCallbacks(Struct): pass
IndexerCallbacks._fields_ = [
  ('abortQuery', ctypes.CFUNCTYPE(ctypes.c_int32, CXClientData, ctypes.c_void_p)),
  ('diagnostic', ctypes.CFUNCTYPE(None, CXClientData, CXDiagnosticSet, ctypes.c_void_p)),
  ('enteredMainFile', ctypes.CFUNCTYPE(CXIdxClientFile, CXClientData, CXFile, ctypes.c_void_p)),
  ('ppIncludedFile', ctypes.CFUNCTYPE(CXIdxClientFile, CXClientData, ctypes.POINTER(CXIdxIncludedFileInfo))),
  ('importedASTFile', ctypes.CFUNCTYPE(CXIdxClientASTFile, CXClientData, ctypes.POINTER(CXIdxImportedASTFileInfo))),
  ('startedTranslationUnit', ctypes.CFUNCTYPE(CXIdxClientContainer, CXClientData, ctypes.c_void_p)),
  ('indexDeclaration', ctypes.CFUNCTYPE(None, CXClientData, ctypes.POINTER(CXIdxDeclInfo))),
  ('indexEntityReference', ctypes.CFUNCTYPE(None, CXClientData, ctypes.POINTER(CXIdxEntityRefInfo))),
]
try: (clang_index_isEntityObjCContainerKind:=dll.clang_index_isEntityObjCContainerKind).restype, clang_index_isEntityObjCContainerKind.argtypes = ctypes.c_int32, [CXIdxEntityKind]
except AttributeError: pass

try: (clang_index_getObjCContainerDeclInfo:=dll.clang_index_getObjCContainerDeclInfo).restype, clang_index_getObjCContainerDeclInfo.argtypes = ctypes.POINTER(CXIdxObjCContainerDeclInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getObjCInterfaceDeclInfo:=dll.clang_index_getObjCInterfaceDeclInfo).restype, clang_index_getObjCInterfaceDeclInfo.argtypes = ctypes.POINTER(CXIdxObjCInterfaceDeclInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getObjCCategoryDeclInfo:=dll.clang_index_getObjCCategoryDeclInfo).restype, clang_index_getObjCCategoryDeclInfo.argtypes = ctypes.POINTER(CXIdxObjCCategoryDeclInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getObjCProtocolRefListInfo:=dll.clang_index_getObjCProtocolRefListInfo).restype, clang_index_getObjCProtocolRefListInfo.argtypes = ctypes.POINTER(CXIdxObjCProtocolRefListInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getObjCPropertyDeclInfo:=dll.clang_index_getObjCPropertyDeclInfo).restype, clang_index_getObjCPropertyDeclInfo.argtypes = ctypes.POINTER(CXIdxObjCPropertyDeclInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getIBOutletCollectionAttrInfo:=dll.clang_index_getIBOutletCollectionAttrInfo).restype, clang_index_getIBOutletCollectionAttrInfo.argtypes = ctypes.POINTER(CXIdxIBOutletCollectionAttrInfo), [ctypes.POINTER(CXIdxAttrInfo)]
except AttributeError: pass

try: (clang_index_getCXXClassDeclInfo:=dll.clang_index_getCXXClassDeclInfo).restype, clang_index_getCXXClassDeclInfo.argtypes = ctypes.POINTER(CXIdxCXXClassDeclInfo), [ctypes.POINTER(CXIdxDeclInfo)]
except AttributeError: pass

try: (clang_index_getClientContainer:=dll.clang_index_getClientContainer).restype, clang_index_getClientContainer.argtypes = CXIdxClientContainer, [ctypes.POINTER(CXIdxContainerInfo)]
except AttributeError: pass

try: (clang_index_setClientContainer:=dll.clang_index_setClientContainer).restype, clang_index_setClientContainer.argtypes = None, [ctypes.POINTER(CXIdxContainerInfo), CXIdxClientContainer]
except AttributeError: pass

try: (clang_index_getClientEntity:=dll.clang_index_getClientEntity).restype, clang_index_getClientEntity.argtypes = CXIdxClientEntity, [ctypes.POINTER(CXIdxEntityInfo)]
except AttributeError: pass

try: (clang_index_setClientEntity:=dll.clang_index_setClientEntity).restype, clang_index_setClientEntity.argtypes = None, [ctypes.POINTER(CXIdxEntityInfo), CXIdxClientEntity]
except AttributeError: pass

CXIndexAction = ctypes.c_void_p
try: (clang_IndexAction_create:=dll.clang_IndexAction_create).restype, clang_IndexAction_create.argtypes = CXIndexAction, [CXIndex]
except AttributeError: pass

try: (clang_IndexAction_dispose:=dll.clang_IndexAction_dispose).restype, clang_IndexAction_dispose.argtypes = None, [CXIndexAction]
except AttributeError: pass

CXIndexOptFlags = CEnum(ctypes.c_uint32)
CXIndexOpt_None = CXIndexOptFlags.define('CXIndexOpt_None', 0)
CXIndexOpt_SuppressRedundantRefs = CXIndexOptFlags.define('CXIndexOpt_SuppressRedundantRefs', 1)
CXIndexOpt_IndexFunctionLocalSymbols = CXIndexOptFlags.define('CXIndexOpt_IndexFunctionLocalSymbols', 2)
CXIndexOpt_IndexImplicitTemplateInstantiations = CXIndexOptFlags.define('CXIndexOpt_IndexImplicitTemplateInstantiations', 4)
CXIndexOpt_SuppressWarnings = CXIndexOptFlags.define('CXIndexOpt_SuppressWarnings', 8)
CXIndexOpt_SkipParsedBodiesInSession = CXIndexOptFlags.define('CXIndexOpt_SkipParsedBodiesInSession', 16)

try: (clang_indexSourceFile:=dll.clang_indexSourceFile).restype, clang_indexSourceFile.argtypes = ctypes.c_int32, [CXIndexAction, CXClientData, ctypes.POINTER(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.POINTER(CXTranslationUnit), ctypes.c_uint32]
except AttributeError: pass

try: (clang_indexSourceFileFullArgv:=dll.clang_indexSourceFileFullArgv).restype, clang_indexSourceFileFullArgv.argtypes = ctypes.c_int32, [CXIndexAction, CXClientData, ctypes.POINTER(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.POINTER(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.POINTER(CXTranslationUnit), ctypes.c_uint32]
except AttributeError: pass

try: (clang_indexTranslationUnit:=dll.clang_indexTranslationUnit).restype, clang_indexTranslationUnit.argtypes = ctypes.c_int32, [CXIndexAction, CXClientData, ctypes.POINTER(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, CXTranslationUnit]
except AttributeError: pass

try: (clang_indexLoc_getFileLocation:=dll.clang_indexLoc_getFileLocation).restype, clang_indexLoc_getFileLocation.argtypes = None, [CXIdxLoc, ctypes.POINTER(CXIdxClientFile), ctypes.POINTER(CXFile), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_indexLoc_getCXSourceLocation:=dll.clang_indexLoc_getCXSourceLocation).restype, clang_indexLoc_getCXSourceLocation.argtypes = CXSourceLocation, [CXIdxLoc]
except AttributeError: pass

CXFieldVisitor = ctypes.CFUNCTYPE(enum_CXVisitorResult, CXCursor, ctypes.c_void_p)
try: (clang_Type_visitFields:=dll.clang_Type_visitFields).restype, clang_Type_visitFields.argtypes = ctypes.c_uint32, [CXType, CXFieldVisitor, CXClientData]
except AttributeError: pass

try: (clang_visitCXXBaseClasses:=dll.clang_visitCXXBaseClasses).restype, clang_visitCXXBaseClasses.argtypes = ctypes.c_uint32, [CXType, CXFieldVisitor, CXClientData]
except AttributeError: pass

enum_CXBinaryOperatorKind = CEnum(ctypes.c_uint32)
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

try: (clang_getBinaryOperatorKindSpelling:=dll.clang_getBinaryOperatorKindSpelling).restype, clang_getBinaryOperatorKindSpelling.argtypes = CXString, [enum_CXBinaryOperatorKind]
except AttributeError: pass

try: (clang_getCursorBinaryOperatorKind:=dll.clang_getCursorBinaryOperatorKind).restype, clang_getCursorBinaryOperatorKind.argtypes = enum_CXBinaryOperatorKind, [CXCursor]
except AttributeError: pass

enum_CXUnaryOperatorKind = CEnum(ctypes.c_uint32)
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

try: (clang_getUnaryOperatorKindSpelling:=dll.clang_getUnaryOperatorKindSpelling).restype, clang_getUnaryOperatorKindSpelling.argtypes = CXString, [enum_CXUnaryOperatorKind]
except AttributeError: pass

try: (clang_getCursorUnaryOperatorKind:=dll.clang_getCursorUnaryOperatorKind).restype, clang_getCursorUnaryOperatorKind.argtypes = enum_CXUnaryOperatorKind, [CXCursor]
except AttributeError: pass

try: (clang_getCString:=dll.clang_getCString).restype, clang_getCString.argtypes = ctypes.POINTER(ctypes.c_char), [CXString]
except AttributeError: pass

try: (clang_disposeString:=dll.clang_disposeString).restype, clang_disposeString.argtypes = None, [CXString]
except AttributeError: pass

try: (clang_disposeStringSet:=dll.clang_disposeStringSet).restype, clang_disposeStringSet.argtypes = None, [ctypes.POINTER(CXStringSet)]
except AttributeError: pass

try: (clang_getNullLocation:=dll.clang_getNullLocation).restype, clang_getNullLocation.argtypes = CXSourceLocation, []
except AttributeError: pass

try: (clang_equalLocations:=dll.clang_equalLocations).restype, clang_equalLocations.argtypes = ctypes.c_uint32, [CXSourceLocation, CXSourceLocation]
except AttributeError: pass

try: (clang_isBeforeInTranslationUnit:=dll.clang_isBeforeInTranslationUnit).restype, clang_isBeforeInTranslationUnit.argtypes = ctypes.c_uint32, [CXSourceLocation, CXSourceLocation]
except AttributeError: pass

try: (clang_Location_isInSystemHeader:=dll.clang_Location_isInSystemHeader).restype, clang_Location_isInSystemHeader.argtypes = ctypes.c_int32, [CXSourceLocation]
except AttributeError: pass

try: (clang_Location_isFromMainFile:=dll.clang_Location_isFromMainFile).restype, clang_Location_isFromMainFile.argtypes = ctypes.c_int32, [CXSourceLocation]
except AttributeError: pass

try: (clang_getNullRange:=dll.clang_getNullRange).restype, clang_getNullRange.argtypes = CXSourceRange, []
except AttributeError: pass

try: (clang_getRange:=dll.clang_getRange).restype, clang_getRange.argtypes = CXSourceRange, [CXSourceLocation, CXSourceLocation]
except AttributeError: pass

try: (clang_equalRanges:=dll.clang_equalRanges).restype, clang_equalRanges.argtypes = ctypes.c_uint32, [CXSourceRange, CXSourceRange]
except AttributeError: pass

try: (clang_Range_isNull:=dll.clang_Range_isNull).restype, clang_Range_isNull.argtypes = ctypes.c_int32, [CXSourceRange]
except AttributeError: pass

try: (clang_getExpansionLocation:=dll.clang_getExpansionLocation).restype, clang_getExpansionLocation.argtypes = None, [CXSourceLocation, ctypes.POINTER(CXFile), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_getPresumedLocation:=dll.clang_getPresumedLocation).restype, clang_getPresumedLocation.argtypes = None, [CXSourceLocation, ctypes.POINTER(CXString), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_getInstantiationLocation:=dll.clang_getInstantiationLocation).restype, clang_getInstantiationLocation.argtypes = None, [CXSourceLocation, ctypes.POINTER(CXFile), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_getSpellingLocation:=dll.clang_getSpellingLocation).restype, clang_getSpellingLocation.argtypes = None, [CXSourceLocation, ctypes.POINTER(CXFile), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_getFileLocation:=dll.clang_getFileLocation).restype, clang_getFileLocation.argtypes = None, [CXSourceLocation, ctypes.POINTER(CXFile), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (clang_getRangeStart:=dll.clang_getRangeStart).restype, clang_getRangeStart.argtypes = CXSourceLocation, [CXSourceRange]
except AttributeError: pass

try: (clang_getRangeEnd:=dll.clang_getRangeEnd).restype, clang_getRangeEnd.argtypes = CXSourceLocation, [CXSourceRange]
except AttributeError: pass

try: (clang_disposeSourceRangeList:=dll.clang_disposeSourceRangeList).restype, clang_disposeSourceRangeList.argtypes = None, [ctypes.POINTER(CXSourceRangeList)]
except AttributeError: pass

try: (clang_getFileName:=dll.clang_getFileName).restype, clang_getFileName.argtypes = CXString, [CXFile]
except AttributeError: pass

time_t = ctypes.c_int64
try: (clang_getFileTime:=dll.clang_getFileTime).restype, clang_getFileTime.argtypes = time_t, [CXFile]
except AttributeError: pass

class CXFileUniqueID(Struct): pass
CXFileUniqueID._fields_ = [
  ('data', (ctypes.c_uint64 * 3)),
]
try: (clang_getFileUniqueID:=dll.clang_getFileUniqueID).restype, clang_getFileUniqueID.argtypes = ctypes.c_int32, [CXFile, ctypes.POINTER(CXFileUniqueID)]
except AttributeError: pass

try: (clang_File_isEqual:=dll.clang_File_isEqual).restype, clang_File_isEqual.argtypes = ctypes.c_int32, [CXFile, CXFile]
except AttributeError: pass

try: (clang_File_tryGetRealPathName:=dll.clang_File_tryGetRealPathName).restype, clang_File_tryGetRealPathName.argtypes = CXString, [CXFile]
except AttributeError: pass

CINDEX_VERSION_MAJOR = 0
CINDEX_VERSION_MINOR = 64
CINDEX_VERSION_ENCODE = lambda major,minor: (((major)*10000) + ((minor)*1))
CINDEX_VERSION = CINDEX_VERSION_ENCODE(CINDEX_VERSION_MAJOR, CINDEX_VERSION_MINOR)
CINDEX_VERSION_STRINGIZE = lambda major,minor: CINDEX_VERSION_STRINGIZE_(major, minor)