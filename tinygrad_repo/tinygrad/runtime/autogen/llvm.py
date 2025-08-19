# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/usr/lib/llvm-14/include', '-D_GNU_SOURCE', '-D__STDC_CONSTANT_MACROS', '-D__STDC_FORMAT_MACROS', '-D__STDC_LIMIT_MACROS']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, tinygrad.runtime.support.llvm as llvm_support


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['llvm'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['llvm'] = ctypes.CDLL(llvm_support.LLVM_PATH) #  ctypes.CDLL('llvm')
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



LLVM_C_ANALYSIS_H = True # macro
LLVM_C_EXTERNC_H = True # macro
# LLVM_C_STRICT_PROTOTYPES_BEGIN = _Pragma ( "clang diagnostic push" ) _Pragma ( "clang diagnostic error \"-Wstrict-prototypes\"" ) # macro
# LLVM_C_STRICT_PROTOTYPES_END = _Pragma ( "clang diagnostic pop" ) # macro
# LLVM_C_EXTERN_C_BEGIN = _Pragma ( "clang diagnostic push" ) _Pragma ( "clang diagnostic error \"-Wstrict-prototypes\"" ) # macro
# LLVM_C_EXTERN_C_END = _Pragma ( "clang diagnostic pop" ) # macro
LLVM_C_TYPES_H = True # macro
LLVM_C_DATATYPES_H = True # macro
LLVMBool = ctypes.c_int32
class struct_LLVMOpaqueMemoryBuffer(Structure):
    pass

LLVMMemoryBufferRef = ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)
class struct_LLVMOpaqueContext(Structure):
    pass

LLVMContextRef = ctypes.POINTER(struct_LLVMOpaqueContext)
class struct_LLVMOpaqueModule(Structure):
    pass

LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
class struct_LLVMOpaqueType(Structure):
    pass

LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
class struct_LLVMOpaqueValue(Structure):
    pass

LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
class struct_LLVMOpaqueBasicBlock(Structure):
    pass

LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
class struct_LLVMOpaqueMetadata(Structure):
    pass

LLVMMetadataRef = ctypes.POINTER(struct_LLVMOpaqueMetadata)
class struct_LLVMOpaqueNamedMDNode(Structure):
    pass

LLVMNamedMDNodeRef = ctypes.POINTER(struct_LLVMOpaqueNamedMDNode)
class struct_LLVMOpaqueValueMetadataEntry(Structure):
    pass

LLVMValueMetadataEntry = struct_LLVMOpaqueValueMetadataEntry
class struct_LLVMOpaqueBuilder(Structure):
    pass

LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
class struct_LLVMOpaqueDIBuilder(Structure):
    pass

LLVMDIBuilderRef = ctypes.POINTER(struct_LLVMOpaqueDIBuilder)
class struct_LLVMOpaqueModuleProvider(Structure):
    pass

LLVMModuleProviderRef = ctypes.POINTER(struct_LLVMOpaqueModuleProvider)
class struct_LLVMOpaquePassManager(Structure):
    pass

LLVMPassManagerRef = ctypes.POINTER(struct_LLVMOpaquePassManager)
class struct_LLVMOpaquePassRegistry(Structure):
    pass

LLVMPassRegistryRef = ctypes.POINTER(struct_LLVMOpaquePassRegistry)
class struct_LLVMOpaqueUse(Structure):
    pass

LLVMUseRef = ctypes.POINTER(struct_LLVMOpaqueUse)
class struct_LLVMOpaqueAttributeRef(Structure):
    pass

LLVMAttributeRef = ctypes.POINTER(struct_LLVMOpaqueAttributeRef)
class struct_LLVMOpaqueDiagnosticInfo(Structure):
    pass

LLVMDiagnosticInfoRef = ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo)
class struct_LLVMComdat(Structure):
    pass

LLVMComdatRef = ctypes.POINTER(struct_LLVMComdat)
class struct_LLVMOpaqueModuleFlagEntry(Structure):
    pass

LLVMModuleFlagEntry = struct_LLVMOpaqueModuleFlagEntry
class struct_LLVMOpaqueJITEventListener(Structure):
    pass

LLVMJITEventListenerRef = ctypes.POINTER(struct_LLVMOpaqueJITEventListener)
class struct_LLVMOpaqueBinary(Structure):
    pass

LLVMBinaryRef = ctypes.POINTER(struct_LLVMOpaqueBinary)

# values for enumeration 'c__EA_LLVMVerifierFailureAction'
c__EA_LLVMVerifierFailureAction__enumvalues = {
    0: 'LLVMAbortProcessAction',
    1: 'LLVMPrintMessageAction',
    2: 'LLVMReturnStatusAction',
}
LLVMAbortProcessAction = 0
LLVMPrintMessageAction = 1
LLVMReturnStatusAction = 2
c__EA_LLVMVerifierFailureAction = ctypes.c_uint32 # enum
LLVMVerifierFailureAction = c__EA_LLVMVerifierFailureAction
LLVMVerifierFailureAction__enumvalues = c__EA_LLVMVerifierFailureAction__enumvalues
try:
    LLVMVerifyModule = _libraries['llvm'].LLVMVerifyModule
    LLVMVerifyModule.restype = LLVMBool
    LLVMVerifyModule.argtypes = [LLVMModuleRef, LLVMVerifierFailureAction, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMVerifyFunction = _libraries['llvm'].LLVMVerifyFunction
    LLVMVerifyFunction.restype = LLVMBool
    LLVMVerifyFunction.argtypes = [LLVMValueRef, LLVMVerifierFailureAction]
except AttributeError:
    pass
try:
    LLVMViewFunctionCFG = _libraries['llvm'].LLVMViewFunctionCFG
    LLVMViewFunctionCFG.restype = None
    LLVMViewFunctionCFG.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMViewFunctionCFGOnly = _libraries['llvm'].LLVMViewFunctionCFGOnly
    LLVMViewFunctionCFGOnly.restype = None
    LLVMViewFunctionCFGOnly.argtypes = [LLVMValueRef]
except AttributeError:
    pass
LLVM_C_BITREADER_H = True # macro
try:
    LLVMParseBitcode = _libraries['llvm'].LLVMParseBitcode
    LLVMParseBitcode.restype = LLVMBool
    LLVMParseBitcode.argtypes = [LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMParseBitcode2 = _libraries['llvm'].LLVMParseBitcode2
    LLVMParseBitcode2.restype = LLVMBool
    LLVMParseBitcode2.argtypes = [LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule))]
except AttributeError:
    pass
try:
    LLVMParseBitcodeInContext = _libraries['llvm'].LLVMParseBitcodeInContext
    LLVMParseBitcodeInContext.restype = LLVMBool
    LLVMParseBitcodeInContext.argtypes = [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMParseBitcodeInContext2 = _libraries['llvm'].LLVMParseBitcodeInContext2
    LLVMParseBitcodeInContext2.restype = LLVMBool
    LLVMParseBitcodeInContext2.argtypes = [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule))]
except AttributeError:
    pass
try:
    LLVMGetBitcodeModuleInContext = _libraries['llvm'].LLVMGetBitcodeModuleInContext
    LLVMGetBitcodeModuleInContext.restype = LLVMBool
    LLVMGetBitcodeModuleInContext.argtypes = [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMGetBitcodeModuleInContext2 = _libraries['llvm'].LLVMGetBitcodeModuleInContext2
    LLVMGetBitcodeModuleInContext2.restype = LLVMBool
    LLVMGetBitcodeModuleInContext2.argtypes = [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule))]
except AttributeError:
    pass
try:
    LLVMGetBitcodeModule = _libraries['llvm'].LLVMGetBitcodeModule
    LLVMGetBitcodeModule.restype = LLVMBool
    LLVMGetBitcodeModule.argtypes = [LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMGetBitcodeModule2 = _libraries['llvm'].LLVMGetBitcodeModule2
    LLVMGetBitcodeModule2.restype = LLVMBool
    LLVMGetBitcodeModule2.argtypes = [LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule))]
except AttributeError:
    pass
LLVM_C_BITWRITER_H = True # macro
try:
    LLVMWriteBitcodeToFile = _libraries['llvm'].LLVMWriteBitcodeToFile
    LLVMWriteBitcodeToFile.restype = ctypes.c_int32
    LLVMWriteBitcodeToFile.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMWriteBitcodeToFD = _libraries['llvm'].LLVMWriteBitcodeToFD
    LLVMWriteBitcodeToFD.restype = ctypes.c_int32
    LLVMWriteBitcodeToFD.argtypes = [LLVMModuleRef, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    LLVMWriteBitcodeToFileHandle = _libraries['llvm'].LLVMWriteBitcodeToFileHandle
    LLVMWriteBitcodeToFileHandle.restype = ctypes.c_int32
    LLVMWriteBitcodeToFileHandle.argtypes = [LLVMModuleRef, ctypes.c_int32]
except AttributeError:
    pass
try:
    LLVMWriteBitcodeToMemoryBuffer = _libraries['llvm'].LLVMWriteBitcodeToMemoryBuffer
    LLVMWriteBitcodeToMemoryBuffer.restype = LLVMMemoryBufferRef
    LLVMWriteBitcodeToMemoryBuffer.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
LLVM_C_COMDAT_H = True # macro

# values for enumeration 'c__EA_LLVMComdatSelectionKind'
c__EA_LLVMComdatSelectionKind__enumvalues = {
    0: 'LLVMAnyComdatSelectionKind',
    1: 'LLVMExactMatchComdatSelectionKind',
    2: 'LLVMLargestComdatSelectionKind',
    3: 'LLVMNoDeduplicateComdatSelectionKind',
    4: 'LLVMSameSizeComdatSelectionKind',
}
LLVMAnyComdatSelectionKind = 0
LLVMExactMatchComdatSelectionKind = 1
LLVMLargestComdatSelectionKind = 2
LLVMNoDeduplicateComdatSelectionKind = 3
LLVMSameSizeComdatSelectionKind = 4
c__EA_LLVMComdatSelectionKind = ctypes.c_uint32 # enum
LLVMComdatSelectionKind = c__EA_LLVMComdatSelectionKind
LLVMComdatSelectionKind__enumvalues = c__EA_LLVMComdatSelectionKind__enumvalues
try:
    LLVMGetOrInsertComdat = _libraries['llvm'].LLVMGetOrInsertComdat
    LLVMGetOrInsertComdat.restype = LLVMComdatRef
    LLVMGetOrInsertComdat.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetComdat = _libraries['llvm'].LLVMGetComdat
    LLVMGetComdat.restype = LLVMComdatRef
    LLVMGetComdat.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetComdat = _libraries['llvm'].LLVMSetComdat
    LLVMSetComdat.restype = None
    LLVMSetComdat.argtypes = [LLVMValueRef, LLVMComdatRef]
except AttributeError:
    pass
try:
    LLVMGetComdatSelectionKind = _libraries['llvm'].LLVMGetComdatSelectionKind
    LLVMGetComdatSelectionKind.restype = LLVMComdatSelectionKind
    LLVMGetComdatSelectionKind.argtypes = [LLVMComdatRef]
except AttributeError:
    pass
try:
    LLVMSetComdatSelectionKind = _libraries['llvm'].LLVMSetComdatSelectionKind
    LLVMSetComdatSelectionKind.restype = None
    LLVMSetComdatSelectionKind.argtypes = [LLVMComdatRef, LLVMComdatSelectionKind]
except AttributeError:
    pass
LLVM_C_CORE_H = True # macro
LLVM_C_DEPRECATED_H = True # macro
# def LLVM_ATTRIBUTE_C_DEPRECATED(decl, message):  # macro
#    return decl((deprecated(message)))
LLVM_C_ERRORHANDLING_H = True # macro
# def LLVM_FOR_EACH_VALUE_SUBCLASS(macro):  # macro
#    return macro(Argument)macro(BasicBlock)macro(InlineAsm)macro(User)macro(Constant)macro(BlockAddress)macro(ConstantAggregateZero)macro(ConstantArray)macro(ConstantDataSequential)macro(ConstantDataArray)macro(ConstantDataVector)macro(ConstantExpr)macro(ConstantFP)macro(ConstantInt)macro(ConstantPointerNull)macro(ConstantStruct)macro(ConstantTokenNone)macro(ConstantVector)macro(GlobalValue)macro(GlobalAlias)macro(GlobalObject)macro(Function)macro(GlobalVariable)macro(GlobalIFunc)macro(UndefValue)macro(PoisonValue)macro(Instruction)macro(UnaryOperator)macro(BinaryOperator)macro(CallInst)macro(IntrinsicInst)macro(DbgInfoIntrinsic)macro(DbgVariableIntrinsic)macro(DbgDeclareInst)macro(DbgLabelInst)macro(MemIntrinsic)macro(MemCpyInst)macro(MemMoveInst)macro(MemSetInst)macro(CmpInst)macro(FCmpInst)macro(ICmpInst)macro(ExtractElementInst)macro(GetElementPtrInst)macro(InsertElementInst)macro(InsertValueInst)macro(LandingPadInst)macro(PHINode)macro(SelectInst)macro(ShuffleVectorInst)macro(StoreInst)macro(BranchInst)macro(IndirectBrInst)macro(InvokeInst)macro(ReturnInst)macro(SwitchInst)macro(UnreachableInst)macro(ResumeInst)macro(CleanupReturnInst)macro(CatchReturnInst)macro(CatchSwitchInst)macro(CallBrInst)macro(FuncletPadInst)macro(CatchPadInst)macro(CleanupPadInst)macro(UnaryInstruction)macro(AllocaInst)macro(CastInst)macro(AddrSpaceCastInst)macro(BitCastInst)macro(FPExtInst)macro(FPToSIInst)macro(FPToUIInst)macro(FPTruncInst)macro(IntToPtrInst)macro(PtrToIntInst)macro(SExtInst)macro(SIToFPInst)macro(TruncInst)macro(UIToFPInst)macro(ZExtInst)macro(ExtractValueInst)macro(LoadInst)macro(VAArgInst)macro(FreezeInst)macro(AtomicCmpXchgInst)macro(AtomicRMWInst)macro(FenceInst)
# def LLVM_DECLARE_VALUE_CAST(name):  # macro
#    return LLVMIsA##name(Val);
LLVMFatalErrorHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char))
try:
    LLVMInstallFatalErrorHandler = _libraries['llvm'].LLVMInstallFatalErrorHandler
    LLVMInstallFatalErrorHandler.restype = None
    LLVMInstallFatalErrorHandler.argtypes = [LLVMFatalErrorHandler]
except AttributeError:
    pass
try:
    LLVMResetFatalErrorHandler = _libraries['llvm'].LLVMResetFatalErrorHandler
    LLVMResetFatalErrorHandler.restype = None
    LLVMResetFatalErrorHandler.argtypes = []
except AttributeError:
    pass
try:
    LLVMEnablePrettyStackTrace = _libraries['llvm'].LLVMEnablePrettyStackTrace
    LLVMEnablePrettyStackTrace.restype = None
    LLVMEnablePrettyStackTrace.argtypes = []
except AttributeError:
    pass

# values for enumeration 'c__EA_LLVMOpcode'
c__EA_LLVMOpcode__enumvalues = {
    1: 'LLVMRet',
    2: 'LLVMBr',
    3: 'LLVMSwitch',
    4: 'LLVMIndirectBr',
    5: 'LLVMInvoke',
    7: 'LLVMUnreachable',
    67: 'LLVMCallBr',
    66: 'LLVMFNeg',
    8: 'LLVMAdd',
    9: 'LLVMFAdd',
    10: 'LLVMSub',
    11: 'LLVMFSub',
    12: 'LLVMMul',
    13: 'LLVMFMul',
    14: 'LLVMUDiv',
    15: 'LLVMSDiv',
    16: 'LLVMFDiv',
    17: 'LLVMURem',
    18: 'LLVMSRem',
    19: 'LLVMFRem',
    20: 'LLVMShl',
    21: 'LLVMLShr',
    22: 'LLVMAShr',
    23: 'LLVMAnd',
    24: 'LLVMOr',
    25: 'LLVMXor',
    26: 'LLVMAlloca',
    27: 'LLVMLoad',
    28: 'LLVMStore',
    29: 'LLVMGetElementPtr',
    30: 'LLVMTrunc',
    31: 'LLVMZExt',
    32: 'LLVMSExt',
    33: 'LLVMFPToUI',
    34: 'LLVMFPToSI',
    35: 'LLVMUIToFP',
    36: 'LLVMSIToFP',
    37: 'LLVMFPTrunc',
    38: 'LLVMFPExt',
    39: 'LLVMPtrToInt',
    40: 'LLVMIntToPtr',
    41: 'LLVMBitCast',
    60: 'LLVMAddrSpaceCast',
    42: 'LLVMICmp',
    43: 'LLVMFCmp',
    44: 'LLVMPHI',
    45: 'LLVMCall',
    46: 'LLVMSelect',
    47: 'LLVMUserOp1',
    48: 'LLVMUserOp2',
    49: 'LLVMVAArg',
    50: 'LLVMExtractElement',
    51: 'LLVMInsertElement',
    52: 'LLVMShuffleVector',
    53: 'LLVMExtractValue',
    54: 'LLVMInsertValue',
    68: 'LLVMFreeze',
    55: 'LLVMFence',
    56: 'LLVMAtomicCmpXchg',
    57: 'LLVMAtomicRMW',
    58: 'LLVMResume',
    59: 'LLVMLandingPad',
    61: 'LLVMCleanupRet',
    62: 'LLVMCatchRet',
    63: 'LLVMCatchPad',
    64: 'LLVMCleanupPad',
    65: 'LLVMCatchSwitch',
}
LLVMRet = 1
LLVMBr = 2
LLVMSwitch = 3
LLVMIndirectBr = 4
LLVMInvoke = 5
LLVMUnreachable = 7
LLVMCallBr = 67
LLVMFNeg = 66
LLVMAdd = 8
LLVMFAdd = 9
LLVMSub = 10
LLVMFSub = 11
LLVMMul = 12
LLVMFMul = 13
LLVMUDiv = 14
LLVMSDiv = 15
LLVMFDiv = 16
LLVMURem = 17
LLVMSRem = 18
LLVMFRem = 19
LLVMShl = 20
LLVMLShr = 21
LLVMAShr = 22
LLVMAnd = 23
LLVMOr = 24
LLVMXor = 25
LLVMAlloca = 26
LLVMLoad = 27
LLVMStore = 28
LLVMGetElementPtr = 29
LLVMTrunc = 30
LLVMZExt = 31
LLVMSExt = 32
LLVMFPToUI = 33
LLVMFPToSI = 34
LLVMUIToFP = 35
LLVMSIToFP = 36
LLVMFPTrunc = 37
LLVMFPExt = 38
LLVMPtrToInt = 39
LLVMIntToPtr = 40
LLVMBitCast = 41
LLVMAddrSpaceCast = 60
LLVMICmp = 42
LLVMFCmp = 43
LLVMPHI = 44
LLVMCall = 45
LLVMSelect = 46
LLVMUserOp1 = 47
LLVMUserOp2 = 48
LLVMVAArg = 49
LLVMExtractElement = 50
LLVMInsertElement = 51
LLVMShuffleVector = 52
LLVMExtractValue = 53
LLVMInsertValue = 54
LLVMFreeze = 68
LLVMFence = 55
LLVMAtomicCmpXchg = 56
LLVMAtomicRMW = 57
LLVMResume = 58
LLVMLandingPad = 59
LLVMCleanupRet = 61
LLVMCatchRet = 62
LLVMCatchPad = 63
LLVMCleanupPad = 64
LLVMCatchSwitch = 65
c__EA_LLVMOpcode = ctypes.c_uint32 # enum
LLVMOpcode = c__EA_LLVMOpcode
LLVMOpcode__enumvalues = c__EA_LLVMOpcode__enumvalues

# values for enumeration 'c__EA_LLVMTypeKind'
c__EA_LLVMTypeKind__enumvalues = {
    0: 'LLVMVoidTypeKind',
    1: 'LLVMHalfTypeKind',
    2: 'LLVMFloatTypeKind',
    3: 'LLVMDoubleTypeKind',
    4: 'LLVMX86_FP80TypeKind',
    5: 'LLVMFP128TypeKind',
    6: 'LLVMPPC_FP128TypeKind',
    7: 'LLVMLabelTypeKind',
    8: 'LLVMIntegerTypeKind',
    9: 'LLVMFunctionTypeKind',
    10: 'LLVMStructTypeKind',
    11: 'LLVMArrayTypeKind',
    12: 'LLVMPointerTypeKind',
    13: 'LLVMVectorTypeKind',
    14: 'LLVMMetadataTypeKind',
    15: 'LLVMX86_MMXTypeKind',
    16: 'LLVMTokenTypeKind',
    17: 'LLVMScalableVectorTypeKind',
    18: 'LLVMBFloatTypeKind',
    19: 'LLVMX86_AMXTypeKind',
}
LLVMVoidTypeKind = 0
LLVMHalfTypeKind = 1
LLVMFloatTypeKind = 2
LLVMDoubleTypeKind = 3
LLVMX86_FP80TypeKind = 4
LLVMFP128TypeKind = 5
LLVMPPC_FP128TypeKind = 6
LLVMLabelTypeKind = 7
LLVMIntegerTypeKind = 8
LLVMFunctionTypeKind = 9
LLVMStructTypeKind = 10
LLVMArrayTypeKind = 11
LLVMPointerTypeKind = 12
LLVMVectorTypeKind = 13
LLVMMetadataTypeKind = 14
LLVMX86_MMXTypeKind = 15
LLVMTokenTypeKind = 16
LLVMScalableVectorTypeKind = 17
LLVMBFloatTypeKind = 18
LLVMX86_AMXTypeKind = 19
c__EA_LLVMTypeKind = ctypes.c_uint32 # enum
LLVMTypeKind = c__EA_LLVMTypeKind
LLVMTypeKind__enumvalues = c__EA_LLVMTypeKind__enumvalues

# values for enumeration 'c__EA_LLVMLinkage'
c__EA_LLVMLinkage__enumvalues = {
    0: 'LLVMExternalLinkage',
    1: 'LLVMAvailableExternallyLinkage',
    2: 'LLVMLinkOnceAnyLinkage',
    3: 'LLVMLinkOnceODRLinkage',
    4: 'LLVMLinkOnceODRAutoHideLinkage',
    5: 'LLVMWeakAnyLinkage',
    6: 'LLVMWeakODRLinkage',
    7: 'LLVMAppendingLinkage',
    8: 'LLVMInternalLinkage',
    9: 'LLVMPrivateLinkage',
    10: 'LLVMDLLImportLinkage',
    11: 'LLVMDLLExportLinkage',
    12: 'LLVMExternalWeakLinkage',
    13: 'LLVMGhostLinkage',
    14: 'LLVMCommonLinkage',
    15: 'LLVMLinkerPrivateLinkage',
    16: 'LLVMLinkerPrivateWeakLinkage',
}
LLVMExternalLinkage = 0
LLVMAvailableExternallyLinkage = 1
LLVMLinkOnceAnyLinkage = 2
LLVMLinkOnceODRLinkage = 3
LLVMLinkOnceODRAutoHideLinkage = 4
LLVMWeakAnyLinkage = 5
LLVMWeakODRLinkage = 6
LLVMAppendingLinkage = 7
LLVMInternalLinkage = 8
LLVMPrivateLinkage = 9
LLVMDLLImportLinkage = 10
LLVMDLLExportLinkage = 11
LLVMExternalWeakLinkage = 12
LLVMGhostLinkage = 13
LLVMCommonLinkage = 14
LLVMLinkerPrivateLinkage = 15
LLVMLinkerPrivateWeakLinkage = 16
c__EA_LLVMLinkage = ctypes.c_uint32 # enum
LLVMLinkage = c__EA_LLVMLinkage
LLVMLinkage__enumvalues = c__EA_LLVMLinkage__enumvalues

# values for enumeration 'c__EA_LLVMVisibility'
c__EA_LLVMVisibility__enumvalues = {
    0: 'LLVMDefaultVisibility',
    1: 'LLVMHiddenVisibility',
    2: 'LLVMProtectedVisibility',
}
LLVMDefaultVisibility = 0
LLVMHiddenVisibility = 1
LLVMProtectedVisibility = 2
c__EA_LLVMVisibility = ctypes.c_uint32 # enum
LLVMVisibility = c__EA_LLVMVisibility
LLVMVisibility__enumvalues = c__EA_LLVMVisibility__enumvalues

# values for enumeration 'c__EA_LLVMUnnamedAddr'
c__EA_LLVMUnnamedAddr__enumvalues = {
    0: 'LLVMNoUnnamedAddr',
    1: 'LLVMLocalUnnamedAddr',
    2: 'LLVMGlobalUnnamedAddr',
}
LLVMNoUnnamedAddr = 0
LLVMLocalUnnamedAddr = 1
LLVMGlobalUnnamedAddr = 2
c__EA_LLVMUnnamedAddr = ctypes.c_uint32 # enum
LLVMUnnamedAddr = c__EA_LLVMUnnamedAddr
LLVMUnnamedAddr__enumvalues = c__EA_LLVMUnnamedAddr__enumvalues

# values for enumeration 'c__EA_LLVMDLLStorageClass'
c__EA_LLVMDLLStorageClass__enumvalues = {
    0: 'LLVMDefaultStorageClass',
    1: 'LLVMDLLImportStorageClass',
    2: 'LLVMDLLExportStorageClass',
}
LLVMDefaultStorageClass = 0
LLVMDLLImportStorageClass = 1
LLVMDLLExportStorageClass = 2
c__EA_LLVMDLLStorageClass = ctypes.c_uint32 # enum
LLVMDLLStorageClass = c__EA_LLVMDLLStorageClass
LLVMDLLStorageClass__enumvalues = c__EA_LLVMDLLStorageClass__enumvalues

# values for enumeration 'c__EA_LLVMCallConv'
c__EA_LLVMCallConv__enumvalues = {
    0: 'LLVMCCallConv',
    8: 'LLVMFastCallConv',
    9: 'LLVMColdCallConv',
    10: 'LLVMGHCCallConv',
    11: 'LLVMHiPECallConv',
    12: 'LLVMWebKitJSCallConv',
    13: 'LLVMAnyRegCallConv',
    14: 'LLVMPreserveMostCallConv',
    15: 'LLVMPreserveAllCallConv',
    16: 'LLVMSwiftCallConv',
    17: 'LLVMCXXFASTTLSCallConv',
    64: 'LLVMX86StdcallCallConv',
    65: 'LLVMX86FastcallCallConv',
    66: 'LLVMARMAPCSCallConv',
    67: 'LLVMARMAAPCSCallConv',
    68: 'LLVMARMAAPCSVFPCallConv',
    69: 'LLVMMSP430INTRCallConv',
    70: 'LLVMX86ThisCallCallConv',
    71: 'LLVMPTXKernelCallConv',
    72: 'LLVMPTXDeviceCallConv',
    75: 'LLVMSPIRFUNCCallConv',
    76: 'LLVMSPIRKERNELCallConv',
    77: 'LLVMIntelOCLBICallConv',
    78: 'LLVMX8664SysVCallConv',
    79: 'LLVMWin64CallConv',
    80: 'LLVMX86VectorCallCallConv',
    81: 'LLVMHHVMCallConv',
    82: 'LLVMHHVMCCallConv',
    83: 'LLVMX86INTRCallConv',
    84: 'LLVMAVRINTRCallConv',
    85: 'LLVMAVRSIGNALCallConv',
    86: 'LLVMAVRBUILTINCallConv',
    87: 'LLVMAMDGPUVSCallConv',
    88: 'LLVMAMDGPUGSCallConv',
    89: 'LLVMAMDGPUPSCallConv',
    90: 'LLVMAMDGPUCSCallConv',
    91: 'LLVMAMDGPUKERNELCallConv',
    92: 'LLVMX86RegCallCallConv',
    93: 'LLVMAMDGPUHSCallConv',
    94: 'LLVMMSP430BUILTINCallConv',
    95: 'LLVMAMDGPULSCallConv',
    96: 'LLVMAMDGPUESCallConv',
}
LLVMCCallConv = 0
LLVMFastCallConv = 8
LLVMColdCallConv = 9
LLVMGHCCallConv = 10
LLVMHiPECallConv = 11
LLVMWebKitJSCallConv = 12
LLVMAnyRegCallConv = 13
LLVMPreserveMostCallConv = 14
LLVMPreserveAllCallConv = 15
LLVMSwiftCallConv = 16
LLVMCXXFASTTLSCallConv = 17
LLVMX86StdcallCallConv = 64
LLVMX86FastcallCallConv = 65
LLVMARMAPCSCallConv = 66
LLVMARMAAPCSCallConv = 67
LLVMARMAAPCSVFPCallConv = 68
LLVMMSP430INTRCallConv = 69
LLVMX86ThisCallCallConv = 70
LLVMPTXKernelCallConv = 71
LLVMPTXDeviceCallConv = 72
LLVMSPIRFUNCCallConv = 75
LLVMSPIRKERNELCallConv = 76
LLVMIntelOCLBICallConv = 77
LLVMX8664SysVCallConv = 78
LLVMWin64CallConv = 79
LLVMX86VectorCallCallConv = 80
LLVMHHVMCallConv = 81
LLVMHHVMCCallConv = 82
LLVMX86INTRCallConv = 83
LLVMAVRINTRCallConv = 84
LLVMAVRSIGNALCallConv = 85
LLVMAVRBUILTINCallConv = 86
LLVMAMDGPUVSCallConv = 87
LLVMAMDGPUGSCallConv = 88
LLVMAMDGPUPSCallConv = 89
LLVMAMDGPUCSCallConv = 90
LLVMAMDGPUKERNELCallConv = 91
LLVMX86RegCallCallConv = 92
LLVMAMDGPUHSCallConv = 93
LLVMMSP430BUILTINCallConv = 94
LLVMAMDGPULSCallConv = 95
LLVMAMDGPUESCallConv = 96
c__EA_LLVMCallConv = ctypes.c_uint32 # enum
LLVMCallConv = c__EA_LLVMCallConv
LLVMCallConv__enumvalues = c__EA_LLVMCallConv__enumvalues

# values for enumeration 'c__EA_LLVMValueKind'
c__EA_LLVMValueKind__enumvalues = {
    0: 'LLVMArgumentValueKind',
    1: 'LLVMBasicBlockValueKind',
    2: 'LLVMMemoryUseValueKind',
    3: 'LLVMMemoryDefValueKind',
    4: 'LLVMMemoryPhiValueKind',
    5: 'LLVMFunctionValueKind',
    6: 'LLVMGlobalAliasValueKind',
    7: 'LLVMGlobalIFuncValueKind',
    8: 'LLVMGlobalVariableValueKind',
    9: 'LLVMBlockAddressValueKind',
    10: 'LLVMConstantExprValueKind',
    11: 'LLVMConstantArrayValueKind',
    12: 'LLVMConstantStructValueKind',
    13: 'LLVMConstantVectorValueKind',
    14: 'LLVMUndefValueValueKind',
    15: 'LLVMConstantAggregateZeroValueKind',
    16: 'LLVMConstantDataArrayValueKind',
    17: 'LLVMConstantDataVectorValueKind',
    18: 'LLVMConstantIntValueKind',
    19: 'LLVMConstantFPValueKind',
    20: 'LLVMConstantPointerNullValueKind',
    21: 'LLVMConstantTokenNoneValueKind',
    22: 'LLVMMetadataAsValueValueKind',
    23: 'LLVMInlineAsmValueKind',
    24: 'LLVMInstructionValueKind',
    25: 'LLVMPoisonValueValueKind',
}
LLVMArgumentValueKind = 0
LLVMBasicBlockValueKind = 1
LLVMMemoryUseValueKind = 2
LLVMMemoryDefValueKind = 3
LLVMMemoryPhiValueKind = 4
LLVMFunctionValueKind = 5
LLVMGlobalAliasValueKind = 6
LLVMGlobalIFuncValueKind = 7
LLVMGlobalVariableValueKind = 8
LLVMBlockAddressValueKind = 9
LLVMConstantExprValueKind = 10
LLVMConstantArrayValueKind = 11
LLVMConstantStructValueKind = 12
LLVMConstantVectorValueKind = 13
LLVMUndefValueValueKind = 14
LLVMConstantAggregateZeroValueKind = 15
LLVMConstantDataArrayValueKind = 16
LLVMConstantDataVectorValueKind = 17
LLVMConstantIntValueKind = 18
LLVMConstantFPValueKind = 19
LLVMConstantPointerNullValueKind = 20
LLVMConstantTokenNoneValueKind = 21
LLVMMetadataAsValueValueKind = 22
LLVMInlineAsmValueKind = 23
LLVMInstructionValueKind = 24
LLVMPoisonValueValueKind = 25
c__EA_LLVMValueKind = ctypes.c_uint32 # enum
LLVMValueKind = c__EA_LLVMValueKind
LLVMValueKind__enumvalues = c__EA_LLVMValueKind__enumvalues

# values for enumeration 'c__EA_LLVMIntPredicate'
c__EA_LLVMIntPredicate__enumvalues = {
    32: 'LLVMIntEQ',
    33: 'LLVMIntNE',
    34: 'LLVMIntUGT',
    35: 'LLVMIntUGE',
    36: 'LLVMIntULT',
    37: 'LLVMIntULE',
    38: 'LLVMIntSGT',
    39: 'LLVMIntSGE',
    40: 'LLVMIntSLT',
    41: 'LLVMIntSLE',
}
LLVMIntEQ = 32
LLVMIntNE = 33
LLVMIntUGT = 34
LLVMIntUGE = 35
LLVMIntULT = 36
LLVMIntULE = 37
LLVMIntSGT = 38
LLVMIntSGE = 39
LLVMIntSLT = 40
LLVMIntSLE = 41
c__EA_LLVMIntPredicate = ctypes.c_uint32 # enum
LLVMIntPredicate = c__EA_LLVMIntPredicate
LLVMIntPredicate__enumvalues = c__EA_LLVMIntPredicate__enumvalues

# values for enumeration 'c__EA_LLVMRealPredicate'
c__EA_LLVMRealPredicate__enumvalues = {
    0: 'LLVMRealPredicateFalse',
    1: 'LLVMRealOEQ',
    2: 'LLVMRealOGT',
    3: 'LLVMRealOGE',
    4: 'LLVMRealOLT',
    5: 'LLVMRealOLE',
    6: 'LLVMRealONE',
    7: 'LLVMRealORD',
    8: 'LLVMRealUNO',
    9: 'LLVMRealUEQ',
    10: 'LLVMRealUGT',
    11: 'LLVMRealUGE',
    12: 'LLVMRealULT',
    13: 'LLVMRealULE',
    14: 'LLVMRealUNE',
    15: 'LLVMRealPredicateTrue',
}
LLVMRealPredicateFalse = 0
LLVMRealOEQ = 1
LLVMRealOGT = 2
LLVMRealOGE = 3
LLVMRealOLT = 4
LLVMRealOLE = 5
LLVMRealONE = 6
LLVMRealORD = 7
LLVMRealUNO = 8
LLVMRealUEQ = 9
LLVMRealUGT = 10
LLVMRealUGE = 11
LLVMRealULT = 12
LLVMRealULE = 13
LLVMRealUNE = 14
LLVMRealPredicateTrue = 15
c__EA_LLVMRealPredicate = ctypes.c_uint32 # enum
LLVMRealPredicate = c__EA_LLVMRealPredicate
LLVMRealPredicate__enumvalues = c__EA_LLVMRealPredicate__enumvalues

# values for enumeration 'c__EA_LLVMLandingPadClauseTy'
c__EA_LLVMLandingPadClauseTy__enumvalues = {
    0: 'LLVMLandingPadCatch',
    1: 'LLVMLandingPadFilter',
}
LLVMLandingPadCatch = 0
LLVMLandingPadFilter = 1
c__EA_LLVMLandingPadClauseTy = ctypes.c_uint32 # enum
LLVMLandingPadClauseTy = c__EA_LLVMLandingPadClauseTy
LLVMLandingPadClauseTy__enumvalues = c__EA_LLVMLandingPadClauseTy__enumvalues

# values for enumeration 'c__EA_LLVMThreadLocalMode'
c__EA_LLVMThreadLocalMode__enumvalues = {
    0: 'LLVMNotThreadLocal',
    1: 'LLVMGeneralDynamicTLSModel',
    2: 'LLVMLocalDynamicTLSModel',
    3: 'LLVMInitialExecTLSModel',
    4: 'LLVMLocalExecTLSModel',
}
LLVMNotThreadLocal = 0
LLVMGeneralDynamicTLSModel = 1
LLVMLocalDynamicTLSModel = 2
LLVMInitialExecTLSModel = 3
LLVMLocalExecTLSModel = 4
c__EA_LLVMThreadLocalMode = ctypes.c_uint32 # enum
LLVMThreadLocalMode = c__EA_LLVMThreadLocalMode
LLVMThreadLocalMode__enumvalues = c__EA_LLVMThreadLocalMode__enumvalues

# values for enumeration 'c__EA_LLVMAtomicOrdering'
c__EA_LLVMAtomicOrdering__enumvalues = {
    0: 'LLVMAtomicOrderingNotAtomic',
    1: 'LLVMAtomicOrderingUnordered',
    2: 'LLVMAtomicOrderingMonotonic',
    4: 'LLVMAtomicOrderingAcquire',
    5: 'LLVMAtomicOrderingRelease',
    6: 'LLVMAtomicOrderingAcquireRelease',
    7: 'LLVMAtomicOrderingSequentiallyConsistent',
}
LLVMAtomicOrderingNotAtomic = 0
LLVMAtomicOrderingUnordered = 1
LLVMAtomicOrderingMonotonic = 2
LLVMAtomicOrderingAcquire = 4
LLVMAtomicOrderingRelease = 5
LLVMAtomicOrderingAcquireRelease = 6
LLVMAtomicOrderingSequentiallyConsistent = 7
c__EA_LLVMAtomicOrdering = ctypes.c_uint32 # enum
LLVMAtomicOrdering = c__EA_LLVMAtomicOrdering
LLVMAtomicOrdering__enumvalues = c__EA_LLVMAtomicOrdering__enumvalues

# values for enumeration 'c__EA_LLVMAtomicRMWBinOp'
c__EA_LLVMAtomicRMWBinOp__enumvalues = {
    0: 'LLVMAtomicRMWBinOpXchg',
    1: 'LLVMAtomicRMWBinOpAdd',
    2: 'LLVMAtomicRMWBinOpSub',
    3: 'LLVMAtomicRMWBinOpAnd',
    4: 'LLVMAtomicRMWBinOpNand',
    5: 'LLVMAtomicRMWBinOpOr',
    6: 'LLVMAtomicRMWBinOpXor',
    7: 'LLVMAtomicRMWBinOpMax',
    8: 'LLVMAtomicRMWBinOpMin',
    9: 'LLVMAtomicRMWBinOpUMax',
    10: 'LLVMAtomicRMWBinOpUMin',
    11: 'LLVMAtomicRMWBinOpFAdd',
    12: 'LLVMAtomicRMWBinOpFSub',
}
LLVMAtomicRMWBinOpXchg = 0
LLVMAtomicRMWBinOpAdd = 1
LLVMAtomicRMWBinOpSub = 2
LLVMAtomicRMWBinOpAnd = 3
LLVMAtomicRMWBinOpNand = 4
LLVMAtomicRMWBinOpOr = 5
LLVMAtomicRMWBinOpXor = 6
LLVMAtomicRMWBinOpMax = 7
LLVMAtomicRMWBinOpMin = 8
LLVMAtomicRMWBinOpUMax = 9
LLVMAtomicRMWBinOpUMin = 10
LLVMAtomicRMWBinOpFAdd = 11
LLVMAtomicRMWBinOpFSub = 12
c__EA_LLVMAtomicRMWBinOp = ctypes.c_uint32 # enum
LLVMAtomicRMWBinOp = c__EA_LLVMAtomicRMWBinOp
LLVMAtomicRMWBinOp__enumvalues = c__EA_LLVMAtomicRMWBinOp__enumvalues

# values for enumeration 'c__EA_LLVMDiagnosticSeverity'
c__EA_LLVMDiagnosticSeverity__enumvalues = {
    0: 'LLVMDSError',
    1: 'LLVMDSWarning',
    2: 'LLVMDSRemark',
    3: 'LLVMDSNote',
}
LLVMDSError = 0
LLVMDSWarning = 1
LLVMDSRemark = 2
LLVMDSNote = 3
c__EA_LLVMDiagnosticSeverity = ctypes.c_uint32 # enum
LLVMDiagnosticSeverity = c__EA_LLVMDiagnosticSeverity
LLVMDiagnosticSeverity__enumvalues = c__EA_LLVMDiagnosticSeverity__enumvalues

# values for enumeration 'c__EA_LLVMInlineAsmDialect'
c__EA_LLVMInlineAsmDialect__enumvalues = {
    0: 'LLVMInlineAsmDialectATT',
    1: 'LLVMInlineAsmDialectIntel',
}
LLVMInlineAsmDialectATT = 0
LLVMInlineAsmDialectIntel = 1
c__EA_LLVMInlineAsmDialect = ctypes.c_uint32 # enum
LLVMInlineAsmDialect = c__EA_LLVMInlineAsmDialect
LLVMInlineAsmDialect__enumvalues = c__EA_LLVMInlineAsmDialect__enumvalues

# values for enumeration 'c__EA_LLVMModuleFlagBehavior'
c__EA_LLVMModuleFlagBehavior__enumvalues = {
    0: 'LLVMModuleFlagBehaviorError',
    1: 'LLVMModuleFlagBehaviorWarning',
    2: 'LLVMModuleFlagBehaviorRequire',
    3: 'LLVMModuleFlagBehaviorOverride',
    4: 'LLVMModuleFlagBehaviorAppend',
    5: 'LLVMModuleFlagBehaviorAppendUnique',
}
LLVMModuleFlagBehaviorError = 0
LLVMModuleFlagBehaviorWarning = 1
LLVMModuleFlagBehaviorRequire = 2
LLVMModuleFlagBehaviorOverride = 3
LLVMModuleFlagBehaviorAppend = 4
LLVMModuleFlagBehaviorAppendUnique = 5
c__EA_LLVMModuleFlagBehavior = ctypes.c_uint32 # enum
LLVMModuleFlagBehavior = c__EA_LLVMModuleFlagBehavior
LLVMModuleFlagBehavior__enumvalues = c__EA_LLVMModuleFlagBehavior__enumvalues

# values for enumeration 'c__Ea_LLVMAttributeReturnIndex'
c__Ea_LLVMAttributeReturnIndex__enumvalues = {
    0: 'LLVMAttributeReturnIndex',
    -1: 'LLVMAttributeFunctionIndex',
}
LLVMAttributeReturnIndex = 0
LLVMAttributeFunctionIndex = -1
c__Ea_LLVMAttributeReturnIndex = ctypes.c_int32 # enum
LLVMAttributeIndex = ctypes.c_uint32
try:
    LLVMInitializeCore = _libraries['llvm'].LLVMInitializeCore
    LLVMInitializeCore.restype = None
    LLVMInitializeCore.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMShutdown = _libraries['llvm'].LLVMShutdown
    LLVMShutdown.restype = None
    LLVMShutdown.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreateMessage = _libraries['llvm'].LLVMCreateMessage
    LLVMCreateMessage.restype = ctypes.POINTER(ctypes.c_char)
    LLVMCreateMessage.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMDisposeMessage = _libraries['llvm'].LLVMDisposeMessage
    LLVMDisposeMessage.restype = None
    LLVMDisposeMessage.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
LLVMDiagnosticHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo), ctypes.POINTER(None))
LLVMYieldCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueContext), ctypes.POINTER(None))
try:
    LLVMContextCreate = _libraries['llvm'].LLVMContextCreate
    LLVMContextCreate.restype = LLVMContextRef
    LLVMContextCreate.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetGlobalContext = _libraries['llvm'].LLVMGetGlobalContext
    LLVMGetGlobalContext.restype = LLVMContextRef
    LLVMGetGlobalContext.argtypes = []
except AttributeError:
    pass
try:
    LLVMContextSetDiagnosticHandler = _libraries['llvm'].LLVMContextSetDiagnosticHandler
    LLVMContextSetDiagnosticHandler.restype = None
    LLVMContextSetDiagnosticHandler.argtypes = [LLVMContextRef, LLVMDiagnosticHandler, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMContextGetDiagnosticHandler = _libraries['llvm'].LLVMContextGetDiagnosticHandler
    LLVMContextGetDiagnosticHandler.restype = LLVMDiagnosticHandler
    LLVMContextGetDiagnosticHandler.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMContextGetDiagnosticContext = _libraries['llvm'].LLVMContextGetDiagnosticContext
    LLVMContextGetDiagnosticContext.restype = ctypes.POINTER(None)
    LLVMContextGetDiagnosticContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMContextSetYieldCallback = _libraries['llvm'].LLVMContextSetYieldCallback
    LLVMContextSetYieldCallback.restype = None
    LLVMContextSetYieldCallback.argtypes = [LLVMContextRef, LLVMYieldCallback, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMContextShouldDiscardValueNames = _libraries['llvm'].LLVMContextShouldDiscardValueNames
    LLVMContextShouldDiscardValueNames.restype = LLVMBool
    LLVMContextShouldDiscardValueNames.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMContextSetDiscardValueNames = _libraries['llvm'].LLVMContextSetDiscardValueNames
    LLVMContextSetDiscardValueNames.restype = None
    LLVMContextSetDiscardValueNames.argtypes = [LLVMContextRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMContextDispose = _libraries['llvm'].LLVMContextDispose
    LLVMContextDispose.restype = None
    LLVMContextDispose.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMGetDiagInfoDescription = _libraries['llvm'].LLVMGetDiagInfoDescription
    LLVMGetDiagInfoDescription.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDiagInfoDescription.argtypes = [LLVMDiagnosticInfoRef]
except AttributeError:
    pass
try:
    LLVMGetDiagInfoSeverity = _libraries['llvm'].LLVMGetDiagInfoSeverity
    LLVMGetDiagInfoSeverity.restype = LLVMDiagnosticSeverity
    LLVMGetDiagInfoSeverity.argtypes = [LLVMDiagnosticInfoRef]
except AttributeError:
    pass
try:
    LLVMGetMDKindIDInContext = _libraries['llvm'].LLVMGetMDKindIDInContext
    LLVMGetMDKindIDInContext.restype = ctypes.c_uint32
    LLVMGetMDKindIDInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetMDKindID = _libraries['llvm'].LLVMGetMDKindID
    LLVMGetMDKindID.restype = ctypes.c_uint32
    LLVMGetMDKindID.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    LLVMGetEnumAttributeKindForName = _libraries['llvm'].LLVMGetEnumAttributeKindForName
    LLVMGetEnumAttributeKindForName.restype = ctypes.c_uint32
    LLVMGetEnumAttributeKindForName.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetLastEnumAttributeKind = _libraries['llvm'].LLVMGetLastEnumAttributeKind
    LLVMGetLastEnumAttributeKind.restype = ctypes.c_uint32
    LLVMGetLastEnumAttributeKind.argtypes = []
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    LLVMCreateEnumAttribute = _libraries['llvm'].LLVMCreateEnumAttribute
    LLVMCreateEnumAttribute.restype = LLVMAttributeRef
    LLVMCreateEnumAttribute.argtypes = [LLVMContextRef, ctypes.c_uint32, uint64_t]
except AttributeError:
    pass
try:
    LLVMGetEnumAttributeKind = _libraries['llvm'].LLVMGetEnumAttributeKind
    LLVMGetEnumAttributeKind.restype = ctypes.c_uint32
    LLVMGetEnumAttributeKind.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMGetEnumAttributeValue = _libraries['llvm'].LLVMGetEnumAttributeValue
    LLVMGetEnumAttributeValue.restype = uint64_t
    LLVMGetEnumAttributeValue.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMCreateTypeAttribute = _libraries['llvm'].LLVMCreateTypeAttribute
    LLVMCreateTypeAttribute.restype = LLVMAttributeRef
    LLVMCreateTypeAttribute.argtypes = [LLVMContextRef, ctypes.c_uint32, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetTypeAttributeValue = _libraries['llvm'].LLVMGetTypeAttributeValue
    LLVMGetTypeAttributeValue.restype = LLVMTypeRef
    LLVMGetTypeAttributeValue.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMCreateStringAttribute = _libraries['llvm'].LLVMCreateStringAttribute
    LLVMCreateStringAttribute.restype = LLVMAttributeRef
    LLVMCreateStringAttribute.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetStringAttributeKind = _libraries['llvm'].LLVMGetStringAttributeKind
    LLVMGetStringAttributeKind.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetStringAttributeKind.argtypes = [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMGetStringAttributeValue = _libraries['llvm'].LLVMGetStringAttributeValue
    LLVMGetStringAttributeValue.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetStringAttributeValue.argtypes = [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMIsEnumAttribute = _libraries['llvm'].LLVMIsEnumAttribute
    LLVMIsEnumAttribute.restype = LLVMBool
    LLVMIsEnumAttribute.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMIsStringAttribute = _libraries['llvm'].LLVMIsStringAttribute
    LLVMIsStringAttribute.restype = LLVMBool
    LLVMIsStringAttribute.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMIsTypeAttribute = _libraries['llvm'].LLVMIsTypeAttribute
    LLVMIsTypeAttribute.restype = LLVMBool
    LLVMIsTypeAttribute.argtypes = [LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMGetTypeByName2 = _libraries['llvm'].LLVMGetTypeByName2
    LLVMGetTypeByName2.restype = LLVMTypeRef
    LLVMGetTypeByName2.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMModuleCreateWithName = _libraries['llvm'].LLVMModuleCreateWithName
    LLVMModuleCreateWithName.restype = LLVMModuleRef
    LLVMModuleCreateWithName.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMModuleCreateWithNameInContext = _libraries['llvm'].LLVMModuleCreateWithNameInContext
    LLVMModuleCreateWithNameInContext.restype = LLVMModuleRef
    LLVMModuleCreateWithNameInContext.argtypes = [ctypes.POINTER(ctypes.c_char), LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMCloneModule = _libraries['llvm'].LLVMCloneModule
    LLVMCloneModule.restype = LLVMModuleRef
    LLVMCloneModule.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMDisposeModule = _libraries['llvm'].LLVMDisposeModule
    LLVMDisposeModule.restype = None
    LLVMDisposeModule.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetModuleIdentifier = _libraries['llvm'].LLVMGetModuleIdentifier
    LLVMGetModuleIdentifier.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetModuleIdentifier.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMSetModuleIdentifier = _libraries['llvm'].LLVMSetModuleIdentifier
    LLVMSetModuleIdentifier.restype = None
    LLVMSetModuleIdentifier.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetSourceFileName = _libraries['llvm'].LLVMGetSourceFileName
    LLVMGetSourceFileName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetSourceFileName.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMSetSourceFileName = _libraries['llvm'].LLVMSetSourceFileName
    LLVMSetSourceFileName.restype = None
    LLVMSetSourceFileName.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetDataLayoutStr = _libraries['llvm'].LLVMGetDataLayoutStr
    LLVMGetDataLayoutStr.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDataLayoutStr.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetDataLayout = _libraries['llvm'].LLVMGetDataLayout
    LLVMGetDataLayout.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDataLayout.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMSetDataLayout = _libraries['llvm'].LLVMSetDataLayout
    LLVMSetDataLayout.restype = None
    LLVMSetDataLayout.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetTarget = _libraries['llvm'].LLVMGetTarget
    LLVMGetTarget.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTarget.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMSetTarget = _libraries['llvm'].LLVMSetTarget
    LLVMSetTarget.restype = None
    LLVMSetTarget.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMCopyModuleFlagsMetadata = _libraries['llvm'].LLVMCopyModuleFlagsMetadata
    LLVMCopyModuleFlagsMetadata.restype = ctypes.POINTER(struct_LLVMOpaqueModuleFlagEntry)
    LLVMCopyModuleFlagsMetadata.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMDisposeModuleFlagsMetadata = _libraries['llvm'].LLVMDisposeModuleFlagsMetadata
    LLVMDisposeModuleFlagsMetadata.restype = None
    LLVMDisposeModuleFlagsMetadata.argtypes = [ctypes.POINTER(struct_LLVMOpaqueModuleFlagEntry)]
except AttributeError:
    pass
try:
    LLVMModuleFlagEntriesGetFlagBehavior = _libraries['llvm'].LLVMModuleFlagEntriesGetFlagBehavior
    LLVMModuleFlagEntriesGetFlagBehavior.restype = LLVMModuleFlagBehavior
    LLVMModuleFlagEntriesGetFlagBehavior.argtypes = [ctypes.POINTER(struct_LLVMOpaqueModuleFlagEntry), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMModuleFlagEntriesGetKey = _libraries['llvm'].LLVMModuleFlagEntriesGetKey
    LLVMModuleFlagEntriesGetKey.restype = ctypes.POINTER(ctypes.c_char)
    LLVMModuleFlagEntriesGetKey.argtypes = [ctypes.POINTER(struct_LLVMOpaqueModuleFlagEntry), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMModuleFlagEntriesGetMetadata = _libraries['llvm'].LLVMModuleFlagEntriesGetMetadata
    LLVMModuleFlagEntriesGetMetadata.restype = LLVMMetadataRef
    LLVMModuleFlagEntriesGetMetadata.argtypes = [ctypes.POINTER(struct_LLVMOpaqueModuleFlagEntry), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetModuleFlag = _libraries['llvm'].LLVMGetModuleFlag
    LLVMGetModuleFlag.restype = LLVMMetadataRef
    LLVMGetModuleFlag.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMAddModuleFlag = _libraries['llvm'].LLVMAddModuleFlag
    LLVMAddModuleFlag.restype = None
    LLVMAddModuleFlag.argtypes = [LLVMModuleRef, LLVMModuleFlagBehavior, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDumpModule = _libraries['llvm'].LLVMDumpModule
    LLVMDumpModule.restype = None
    LLVMDumpModule.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMPrintModuleToFile = _libraries['llvm'].LLVMPrintModuleToFile
    LLVMPrintModuleToFile.restype = LLVMBool
    LLVMPrintModuleToFile.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMPrintModuleToString = _libraries['llvm'].LLVMPrintModuleToString
    LLVMPrintModuleToString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMPrintModuleToString.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetModuleInlineAsm = _libraries['llvm'].LLVMGetModuleInlineAsm
    LLVMGetModuleInlineAsm.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetModuleInlineAsm.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMSetModuleInlineAsm2 = _libraries['llvm'].LLVMSetModuleInlineAsm2
    LLVMSetModuleInlineAsm2.restype = None
    LLVMSetModuleInlineAsm2.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMAppendModuleInlineAsm = _libraries['llvm'].LLVMAppendModuleInlineAsm
    LLVMAppendModuleInlineAsm.restype = None
    LLVMAppendModuleInlineAsm.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetInlineAsm = _libraries['llvm'].LLVMGetInlineAsm
    LLVMGetInlineAsm.restype = LLVMValueRef
    LLVMGetInlineAsm.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, LLVMBool, LLVMInlineAsmDialect, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetModuleContext = _libraries['llvm'].LLVMGetModuleContext
    LLVMGetModuleContext.restype = LLVMContextRef
    LLVMGetModuleContext.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetTypeByName = _libraries['llvm'].LLVMGetTypeByName
    LLVMGetTypeByName.restype = LLVMTypeRef
    LLVMGetTypeByName.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetFirstNamedMetadata = _libraries['llvm'].LLVMGetFirstNamedMetadata
    LLVMGetFirstNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetFirstNamedMetadata.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetLastNamedMetadata = _libraries['llvm'].LLVMGetLastNamedMetadata
    LLVMGetLastNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetLastNamedMetadata.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetNextNamedMetadata = _libraries['llvm'].LLVMGetNextNamedMetadata
    LLVMGetNextNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetNextNamedMetadata.argtypes = [LLVMNamedMDNodeRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousNamedMetadata = _libraries['llvm'].LLVMGetPreviousNamedMetadata
    LLVMGetPreviousNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetPreviousNamedMetadata.argtypes = [LLVMNamedMDNodeRef]
except AttributeError:
    pass
try:
    LLVMGetNamedMetadata = _libraries['llvm'].LLVMGetNamedMetadata
    LLVMGetNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetNamedMetadata.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetOrInsertNamedMetadata = _libraries['llvm'].LLVMGetOrInsertNamedMetadata
    LLVMGetOrInsertNamedMetadata.restype = LLVMNamedMDNodeRef
    LLVMGetOrInsertNamedMetadata.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetNamedMetadataName = _libraries['llvm'].LLVMGetNamedMetadataName
    LLVMGetNamedMetadataName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetNamedMetadataName.argtypes = [LLVMNamedMDNodeRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMGetNamedMetadataNumOperands = _libraries['llvm'].LLVMGetNamedMetadataNumOperands
    LLVMGetNamedMetadataNumOperands.restype = ctypes.c_uint32
    LLVMGetNamedMetadataNumOperands.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetNamedMetadataOperands = _libraries['llvm'].LLVMGetNamedMetadataOperands
    LLVMGetNamedMetadataOperands.restype = None
    LLVMGetNamedMetadataOperands.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))]
except AttributeError:
    pass
try:
    LLVMAddNamedMetadataOperand = _libraries['llvm'].LLVMAddNamedMetadataOperand
    LLVMAddNamedMetadataOperand.restype = None
    LLVMAddNamedMetadataOperand.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetDebugLocDirectory = _libraries['llvm'].LLVMGetDebugLocDirectory
    LLVMGetDebugLocDirectory.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDebugLocDirectory.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMGetDebugLocFilename = _libraries['llvm'].LLVMGetDebugLocFilename
    LLVMGetDebugLocFilename.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDebugLocFilename.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMGetDebugLocLine = _libraries['llvm'].LLVMGetDebugLocLine
    LLVMGetDebugLocLine.restype = ctypes.c_uint32
    LLVMGetDebugLocLine.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetDebugLocColumn = _libraries['llvm'].LLVMGetDebugLocColumn
    LLVMGetDebugLocColumn.restype = ctypes.c_uint32
    LLVMGetDebugLocColumn.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAddFunction = _libraries['llvm'].LLVMAddFunction
    LLVMAddFunction.restype = LLVMValueRef
    LLVMAddFunction.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetNamedFunction = _libraries['llvm'].LLVMGetNamedFunction
    LLVMGetNamedFunction.restype = LLVMValueRef
    LLVMGetNamedFunction.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetFirstFunction = _libraries['llvm'].LLVMGetFirstFunction
    LLVMGetFirstFunction.restype = LLVMValueRef
    LLVMGetFirstFunction.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetLastFunction = _libraries['llvm'].LLVMGetLastFunction
    LLVMGetLastFunction.restype = LLVMValueRef
    LLVMGetLastFunction.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetNextFunction = _libraries['llvm'].LLVMGetNextFunction
    LLVMGetNextFunction.restype = LLVMValueRef
    LLVMGetNextFunction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousFunction = _libraries['llvm'].LLVMGetPreviousFunction
    LLVMGetPreviousFunction.restype = LLVMValueRef
    LLVMGetPreviousFunction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetModuleInlineAsm = _libraries['llvm'].LLVMSetModuleInlineAsm
    LLVMSetModuleInlineAsm.restype = None
    LLVMSetModuleInlineAsm.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetTypeKind = _libraries['llvm'].LLVMGetTypeKind
    LLVMGetTypeKind.restype = LLVMTypeKind
    LLVMGetTypeKind.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMTypeIsSized = _libraries['llvm'].LLVMTypeIsSized
    LLVMTypeIsSized.restype = LLVMBool
    LLVMTypeIsSized.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetTypeContext = _libraries['llvm'].LLVMGetTypeContext
    LLVMGetTypeContext.restype = LLVMContextRef
    LLVMGetTypeContext.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMDumpType = _libraries['llvm'].LLVMDumpType
    LLVMDumpType.restype = None
    LLVMDumpType.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMPrintTypeToString = _libraries['llvm'].LLVMPrintTypeToString
    LLVMPrintTypeToString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMPrintTypeToString.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMInt1TypeInContext = _libraries['llvm'].LLVMInt1TypeInContext
    LLVMInt1TypeInContext.restype = LLVMTypeRef
    LLVMInt1TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMInt8TypeInContext = _libraries['llvm'].LLVMInt8TypeInContext
    LLVMInt8TypeInContext.restype = LLVMTypeRef
    LLVMInt8TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMInt16TypeInContext = _libraries['llvm'].LLVMInt16TypeInContext
    LLVMInt16TypeInContext.restype = LLVMTypeRef
    LLVMInt16TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMInt32TypeInContext = _libraries['llvm'].LLVMInt32TypeInContext
    LLVMInt32TypeInContext.restype = LLVMTypeRef
    LLVMInt32TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMInt64TypeInContext = _libraries['llvm'].LLVMInt64TypeInContext
    LLVMInt64TypeInContext.restype = LLVMTypeRef
    LLVMInt64TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMInt128TypeInContext = _libraries['llvm'].LLVMInt128TypeInContext
    LLVMInt128TypeInContext.restype = LLVMTypeRef
    LLVMInt128TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMIntTypeInContext = _libraries['llvm'].LLVMIntTypeInContext
    LLVMIntTypeInContext.restype = LLVMTypeRef
    LLVMIntTypeInContext.argtypes = [LLVMContextRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMInt1Type = _libraries['llvm'].LLVMInt1Type
    LLVMInt1Type.restype = LLVMTypeRef
    LLVMInt1Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMInt8Type = _libraries['llvm'].LLVMInt8Type
    LLVMInt8Type.restype = LLVMTypeRef
    LLVMInt8Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMInt16Type = _libraries['llvm'].LLVMInt16Type
    LLVMInt16Type.restype = LLVMTypeRef
    LLVMInt16Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMInt32Type = _libraries['llvm'].LLVMInt32Type
    LLVMInt32Type.restype = LLVMTypeRef
    LLVMInt32Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMInt64Type = _libraries['llvm'].LLVMInt64Type
    LLVMInt64Type.restype = LLVMTypeRef
    LLVMInt64Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMInt128Type = _libraries['llvm'].LLVMInt128Type
    LLVMInt128Type.restype = LLVMTypeRef
    LLVMInt128Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMIntType = _libraries['llvm'].LLVMIntType
    LLVMIntType.restype = LLVMTypeRef
    LLVMIntType.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetIntTypeWidth = _libraries['llvm'].LLVMGetIntTypeWidth
    LLVMGetIntTypeWidth.restype = ctypes.c_uint32
    LLVMGetIntTypeWidth.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMHalfTypeInContext = _libraries['llvm'].LLVMHalfTypeInContext
    LLVMHalfTypeInContext.restype = LLVMTypeRef
    LLVMHalfTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMBFloatTypeInContext = _libraries['llvm'].LLVMBFloatTypeInContext
    LLVMBFloatTypeInContext.restype = LLVMTypeRef
    LLVMBFloatTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMFloatTypeInContext = _libraries['llvm'].LLVMFloatTypeInContext
    LLVMFloatTypeInContext.restype = LLVMTypeRef
    LLVMFloatTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMDoubleTypeInContext = _libraries['llvm'].LLVMDoubleTypeInContext
    LLVMDoubleTypeInContext.restype = LLVMTypeRef
    LLVMDoubleTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMX86FP80TypeInContext = _libraries['llvm'].LLVMX86FP80TypeInContext
    LLVMX86FP80TypeInContext.restype = LLVMTypeRef
    LLVMX86FP80TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMFP128TypeInContext = _libraries['llvm'].LLVMFP128TypeInContext
    LLVMFP128TypeInContext.restype = LLVMTypeRef
    LLVMFP128TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMPPCFP128TypeInContext = _libraries['llvm'].LLVMPPCFP128TypeInContext
    LLVMPPCFP128TypeInContext.restype = LLVMTypeRef
    LLVMPPCFP128TypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMHalfType = _libraries['llvm'].LLVMHalfType
    LLVMHalfType.restype = LLVMTypeRef
    LLVMHalfType.argtypes = []
except AttributeError:
    pass
try:
    LLVMBFloatType = _libraries['llvm'].LLVMBFloatType
    LLVMBFloatType.restype = LLVMTypeRef
    LLVMBFloatType.argtypes = []
except AttributeError:
    pass
try:
    LLVMFloatType = _libraries['llvm'].LLVMFloatType
    LLVMFloatType.restype = LLVMTypeRef
    LLVMFloatType.argtypes = []
except AttributeError:
    pass
try:
    LLVMDoubleType = _libraries['llvm'].LLVMDoubleType
    LLVMDoubleType.restype = LLVMTypeRef
    LLVMDoubleType.argtypes = []
except AttributeError:
    pass
try:
    LLVMX86FP80Type = _libraries['llvm'].LLVMX86FP80Type
    LLVMX86FP80Type.restype = LLVMTypeRef
    LLVMX86FP80Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMFP128Type = _libraries['llvm'].LLVMFP128Type
    LLVMFP128Type.restype = LLVMTypeRef
    LLVMFP128Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMPPCFP128Type = _libraries['llvm'].LLVMPPCFP128Type
    LLVMPPCFP128Type.restype = LLVMTypeRef
    LLVMPPCFP128Type.argtypes = []
except AttributeError:
    pass
try:
    LLVMFunctionType = _libraries['llvm'].LLVMFunctionType
    LLVMFunctionType.restype = LLVMTypeRef
    LLVMFunctionType.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMIsFunctionVarArg = _libraries['llvm'].LLVMIsFunctionVarArg
    LLVMIsFunctionVarArg.restype = LLVMBool
    LLVMIsFunctionVarArg.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetReturnType = _libraries['llvm'].LLVMGetReturnType
    LLVMGetReturnType.restype = LLVMTypeRef
    LLVMGetReturnType.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMCountParamTypes = _libraries['llvm'].LLVMCountParamTypes
    LLVMCountParamTypes.restype = ctypes.c_uint32
    LLVMCountParamTypes.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetParamTypes = _libraries['llvm'].LLVMGetParamTypes
    LLVMGetParamTypes.restype = None
    LLVMGetParamTypes.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType))]
except AttributeError:
    pass
try:
    LLVMStructTypeInContext = _libraries['llvm'].LLVMStructTypeInContext
    LLVMStructTypeInContext.restype = LLVMTypeRef
    LLVMStructTypeInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMStructType = _libraries['llvm'].LLVMStructType
    LLVMStructType.restype = LLVMTypeRef
    LLVMStructType.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMStructCreateNamed = _libraries['llvm'].LLVMStructCreateNamed
    LLVMStructCreateNamed.restype = LLVMTypeRef
    LLVMStructCreateNamed.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetStructName = _libraries['llvm'].LLVMGetStructName
    LLVMGetStructName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetStructName.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMStructSetBody = _libraries['llvm'].LLVMStructSetBody
    LLVMStructSetBody.restype = None
    LLVMStructSetBody.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMCountStructElementTypes = _libraries['llvm'].LLVMCountStructElementTypes
    LLVMCountStructElementTypes.restype = ctypes.c_uint32
    LLVMCountStructElementTypes.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetStructElementTypes = _libraries['llvm'].LLVMGetStructElementTypes
    LLVMGetStructElementTypes.restype = None
    LLVMGetStructElementTypes.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType))]
except AttributeError:
    pass
try:
    LLVMStructGetTypeAtIndex = _libraries['llvm'].LLVMStructGetTypeAtIndex
    LLVMStructGetTypeAtIndex.restype = LLVMTypeRef
    LLVMStructGetTypeAtIndex.argtypes = [LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMIsPackedStruct = _libraries['llvm'].LLVMIsPackedStruct
    LLVMIsPackedStruct.restype = LLVMBool
    LLVMIsPackedStruct.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMIsOpaqueStruct = _libraries['llvm'].LLVMIsOpaqueStruct
    LLVMIsOpaqueStruct.restype = LLVMBool
    LLVMIsOpaqueStruct.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMIsLiteralStruct = _libraries['llvm'].LLVMIsLiteralStruct
    LLVMIsLiteralStruct.restype = LLVMBool
    LLVMIsLiteralStruct.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetElementType = _libraries['llvm'].LLVMGetElementType
    LLVMGetElementType.restype = LLVMTypeRef
    LLVMGetElementType.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetSubtypes = _libraries['llvm'].LLVMGetSubtypes
    LLVMGetSubtypes.restype = None
    LLVMGetSubtypes.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType))]
except AttributeError:
    pass
try:
    LLVMGetNumContainedTypes = _libraries['llvm'].LLVMGetNumContainedTypes
    LLVMGetNumContainedTypes.restype = ctypes.c_uint32
    LLVMGetNumContainedTypes.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMArrayType = _libraries['llvm'].LLVMArrayType
    LLVMArrayType.restype = LLVMTypeRef
    LLVMArrayType.argtypes = [LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetArrayLength = _libraries['llvm'].LLVMGetArrayLength
    LLVMGetArrayLength.restype = ctypes.c_uint32
    LLVMGetArrayLength.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMPointerType = _libraries['llvm'].LLVMPointerType
    LLVMPointerType.restype = LLVMTypeRef
    LLVMPointerType.argtypes = [LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetPointerAddressSpace = _libraries['llvm'].LLVMGetPointerAddressSpace
    LLVMGetPointerAddressSpace.restype = ctypes.c_uint32
    LLVMGetPointerAddressSpace.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMVectorType = _libraries['llvm'].LLVMVectorType
    LLVMVectorType.restype = LLVMTypeRef
    LLVMVectorType.argtypes = [LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMScalableVectorType = _libraries['llvm'].LLVMScalableVectorType
    LLVMScalableVectorType.restype = LLVMTypeRef
    LLVMScalableVectorType.argtypes = [LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetVectorSize = _libraries['llvm'].LLVMGetVectorSize
    LLVMGetVectorSize.restype = ctypes.c_uint32
    LLVMGetVectorSize.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMVoidTypeInContext = _libraries['llvm'].LLVMVoidTypeInContext
    LLVMVoidTypeInContext.restype = LLVMTypeRef
    LLVMVoidTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMLabelTypeInContext = _libraries['llvm'].LLVMLabelTypeInContext
    LLVMLabelTypeInContext.restype = LLVMTypeRef
    LLVMLabelTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMX86MMXTypeInContext = _libraries['llvm'].LLVMX86MMXTypeInContext
    LLVMX86MMXTypeInContext.restype = LLVMTypeRef
    LLVMX86MMXTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMX86AMXTypeInContext = _libraries['llvm'].LLVMX86AMXTypeInContext
    LLVMX86AMXTypeInContext.restype = LLVMTypeRef
    LLVMX86AMXTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMTokenTypeInContext = _libraries['llvm'].LLVMTokenTypeInContext
    LLVMTokenTypeInContext.restype = LLVMTypeRef
    LLVMTokenTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMMetadataTypeInContext = _libraries['llvm'].LLVMMetadataTypeInContext
    LLVMMetadataTypeInContext.restype = LLVMTypeRef
    LLVMMetadataTypeInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMVoidType = _libraries['llvm'].LLVMVoidType
    LLVMVoidType.restype = LLVMTypeRef
    LLVMVoidType.argtypes = []
except AttributeError:
    pass
try:
    LLVMLabelType = _libraries['llvm'].LLVMLabelType
    LLVMLabelType.restype = LLVMTypeRef
    LLVMLabelType.argtypes = []
except AttributeError:
    pass
try:
    LLVMX86MMXType = _libraries['llvm'].LLVMX86MMXType
    LLVMX86MMXType.restype = LLVMTypeRef
    LLVMX86MMXType.argtypes = []
except AttributeError:
    pass
try:
    LLVMX86AMXType = _libraries['llvm'].LLVMX86AMXType
    LLVMX86AMXType.restype = LLVMTypeRef
    LLVMX86AMXType.argtypes = []
except AttributeError:
    pass
try:
    LLVMTypeOf = _libraries['llvm'].LLVMTypeOf
    LLVMTypeOf.restype = LLVMTypeRef
    LLVMTypeOf.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetValueKind = _libraries['llvm'].LLVMGetValueKind
    LLVMGetValueKind.restype = LLVMValueKind
    LLVMGetValueKind.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetValueName2 = _libraries['llvm'].LLVMGetValueName2
    LLVMGetValueName2.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetValueName2.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMSetValueName2 = _libraries['llvm'].LLVMSetValueName2
    LLVMSetValueName2.restype = None
    LLVMSetValueName2.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDumpValue = _libraries['llvm'].LLVMDumpValue
    LLVMDumpValue.restype = None
    LLVMDumpValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMPrintValueToString = _libraries['llvm'].LLVMPrintValueToString
    LLVMPrintValueToString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMPrintValueToString.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMReplaceAllUsesWith = _libraries['llvm'].LLVMReplaceAllUsesWith
    LLVMReplaceAllUsesWith.restype = None
    LLVMReplaceAllUsesWith.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsConstant = _libraries['llvm'].LLVMIsConstant
    LLVMIsConstant.restype = LLVMBool
    LLVMIsConstant.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsUndef = _libraries['llvm'].LLVMIsUndef
    LLVMIsUndef.restype = LLVMBool
    LLVMIsUndef.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsPoison = _libraries['llvm'].LLVMIsPoison
    LLVMIsPoison.restype = LLVMBool
    LLVMIsPoison.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAArgument = _libraries['llvm'].LLVMIsAArgument
    LLVMIsAArgument.restype = LLVMValueRef
    LLVMIsAArgument.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsABasicBlock = _libraries['llvm'].LLVMIsABasicBlock
    LLVMIsABasicBlock.restype = LLVMValueRef
    LLVMIsABasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAInlineAsm = _libraries['llvm'].LLVMIsAInlineAsm
    LLVMIsAInlineAsm.restype = LLVMValueRef
    LLVMIsAInlineAsm.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUser = _libraries['llvm'].LLVMIsAUser
    LLVMIsAUser.restype = LLVMValueRef
    LLVMIsAUser.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstant = _libraries['llvm'].LLVMIsAConstant
    LLVMIsAConstant.restype = LLVMValueRef
    LLVMIsAConstant.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsABlockAddress = _libraries['llvm'].LLVMIsABlockAddress
    LLVMIsABlockAddress.restype = LLVMValueRef
    LLVMIsABlockAddress.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantAggregateZero = _libraries['llvm'].LLVMIsAConstantAggregateZero
    LLVMIsAConstantAggregateZero.restype = LLVMValueRef
    LLVMIsAConstantAggregateZero.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantArray = _libraries['llvm'].LLVMIsAConstantArray
    LLVMIsAConstantArray.restype = LLVMValueRef
    LLVMIsAConstantArray.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantDataSequential = _libraries['llvm'].LLVMIsAConstantDataSequential
    LLVMIsAConstantDataSequential.restype = LLVMValueRef
    LLVMIsAConstantDataSequential.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantDataArray = _libraries['llvm'].LLVMIsAConstantDataArray
    LLVMIsAConstantDataArray.restype = LLVMValueRef
    LLVMIsAConstantDataArray.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantDataVector = _libraries['llvm'].LLVMIsAConstantDataVector
    LLVMIsAConstantDataVector.restype = LLVMValueRef
    LLVMIsAConstantDataVector.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantExpr = _libraries['llvm'].LLVMIsAConstantExpr
    LLVMIsAConstantExpr.restype = LLVMValueRef
    LLVMIsAConstantExpr.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantFP = _libraries['llvm'].LLVMIsAConstantFP
    LLVMIsAConstantFP.restype = LLVMValueRef
    LLVMIsAConstantFP.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantInt = _libraries['llvm'].LLVMIsAConstantInt
    LLVMIsAConstantInt.restype = LLVMValueRef
    LLVMIsAConstantInt.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantPointerNull = _libraries['llvm'].LLVMIsAConstantPointerNull
    LLVMIsAConstantPointerNull.restype = LLVMValueRef
    LLVMIsAConstantPointerNull.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantStruct = _libraries['llvm'].LLVMIsAConstantStruct
    LLVMIsAConstantStruct.restype = LLVMValueRef
    LLVMIsAConstantStruct.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantTokenNone = _libraries['llvm'].LLVMIsAConstantTokenNone
    LLVMIsAConstantTokenNone.restype = LLVMValueRef
    LLVMIsAConstantTokenNone.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAConstantVector = _libraries['llvm'].LLVMIsAConstantVector
    LLVMIsAConstantVector.restype = LLVMValueRef
    LLVMIsAConstantVector.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGlobalValue = _libraries['llvm'].LLVMIsAGlobalValue
    LLVMIsAGlobalValue.restype = LLVMValueRef
    LLVMIsAGlobalValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGlobalAlias = _libraries['llvm'].LLVMIsAGlobalAlias
    LLVMIsAGlobalAlias.restype = LLVMValueRef
    LLVMIsAGlobalAlias.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGlobalObject = _libraries['llvm'].LLVMIsAGlobalObject
    LLVMIsAGlobalObject.restype = LLVMValueRef
    LLVMIsAGlobalObject.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFunction = _libraries['llvm'].LLVMIsAFunction
    LLVMIsAFunction.restype = LLVMValueRef
    LLVMIsAFunction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGlobalVariable = _libraries['llvm'].LLVMIsAGlobalVariable
    LLVMIsAGlobalVariable.restype = LLVMValueRef
    LLVMIsAGlobalVariable.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGlobalIFunc = _libraries['llvm'].LLVMIsAGlobalIFunc
    LLVMIsAGlobalIFunc.restype = LLVMValueRef
    LLVMIsAGlobalIFunc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUndefValue = _libraries['llvm'].LLVMIsAUndefValue
    LLVMIsAUndefValue.restype = LLVMValueRef
    LLVMIsAUndefValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAPoisonValue = _libraries['llvm'].LLVMIsAPoisonValue
    LLVMIsAPoisonValue.restype = LLVMValueRef
    LLVMIsAPoisonValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAInstruction = _libraries['llvm'].LLVMIsAInstruction
    LLVMIsAInstruction.restype = LLVMValueRef
    LLVMIsAInstruction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUnaryOperator = _libraries['llvm'].LLVMIsAUnaryOperator
    LLVMIsAUnaryOperator.restype = LLVMValueRef
    LLVMIsAUnaryOperator.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsABinaryOperator = _libraries['llvm'].LLVMIsABinaryOperator
    LLVMIsABinaryOperator.restype = LLVMValueRef
    LLVMIsABinaryOperator.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACallInst = _libraries['llvm'].LLVMIsACallInst
    LLVMIsACallInst.restype = LLVMValueRef
    LLVMIsACallInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAIntrinsicInst = _libraries['llvm'].LLVMIsAIntrinsicInst
    LLVMIsAIntrinsicInst.restype = LLVMValueRef
    LLVMIsAIntrinsicInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsADbgInfoIntrinsic = _libraries['llvm'].LLVMIsADbgInfoIntrinsic
    LLVMIsADbgInfoIntrinsic.restype = LLVMValueRef
    LLVMIsADbgInfoIntrinsic.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsADbgVariableIntrinsic = _libraries['llvm'].LLVMIsADbgVariableIntrinsic
    LLVMIsADbgVariableIntrinsic.restype = LLVMValueRef
    LLVMIsADbgVariableIntrinsic.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsADbgDeclareInst = _libraries['llvm'].LLVMIsADbgDeclareInst
    LLVMIsADbgDeclareInst.restype = LLVMValueRef
    LLVMIsADbgDeclareInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsADbgLabelInst = _libraries['llvm'].LLVMIsADbgLabelInst
    LLVMIsADbgLabelInst.restype = LLVMValueRef
    LLVMIsADbgLabelInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMemIntrinsic = _libraries['llvm'].LLVMIsAMemIntrinsic
    LLVMIsAMemIntrinsic.restype = LLVMValueRef
    LLVMIsAMemIntrinsic.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMemCpyInst = _libraries['llvm'].LLVMIsAMemCpyInst
    LLVMIsAMemCpyInst.restype = LLVMValueRef
    LLVMIsAMemCpyInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMemMoveInst = _libraries['llvm'].LLVMIsAMemMoveInst
    LLVMIsAMemMoveInst.restype = LLVMValueRef
    LLVMIsAMemMoveInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMemSetInst = _libraries['llvm'].LLVMIsAMemSetInst
    LLVMIsAMemSetInst.restype = LLVMValueRef
    LLVMIsAMemSetInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACmpInst = _libraries['llvm'].LLVMIsACmpInst
    LLVMIsACmpInst.restype = LLVMValueRef
    LLVMIsACmpInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFCmpInst = _libraries['llvm'].LLVMIsAFCmpInst
    LLVMIsAFCmpInst.restype = LLVMValueRef
    LLVMIsAFCmpInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAICmpInst = _libraries['llvm'].LLVMIsAICmpInst
    LLVMIsAICmpInst.restype = LLVMValueRef
    LLVMIsAICmpInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAExtractElementInst = _libraries['llvm'].LLVMIsAExtractElementInst
    LLVMIsAExtractElementInst.restype = LLVMValueRef
    LLVMIsAExtractElementInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAGetElementPtrInst = _libraries['llvm'].LLVMIsAGetElementPtrInst
    LLVMIsAGetElementPtrInst.restype = LLVMValueRef
    LLVMIsAGetElementPtrInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAInsertElementInst = _libraries['llvm'].LLVMIsAInsertElementInst
    LLVMIsAInsertElementInst.restype = LLVMValueRef
    LLVMIsAInsertElementInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAInsertValueInst = _libraries['llvm'].LLVMIsAInsertValueInst
    LLVMIsAInsertValueInst.restype = LLVMValueRef
    LLVMIsAInsertValueInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsALandingPadInst = _libraries['llvm'].LLVMIsALandingPadInst
    LLVMIsALandingPadInst.restype = LLVMValueRef
    LLVMIsALandingPadInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAPHINode = _libraries['llvm'].LLVMIsAPHINode
    LLVMIsAPHINode.restype = LLVMValueRef
    LLVMIsAPHINode.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsASelectInst = _libraries['llvm'].LLVMIsASelectInst
    LLVMIsASelectInst.restype = LLVMValueRef
    LLVMIsASelectInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAShuffleVectorInst = _libraries['llvm'].LLVMIsAShuffleVectorInst
    LLVMIsAShuffleVectorInst.restype = LLVMValueRef
    LLVMIsAShuffleVectorInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAStoreInst = _libraries['llvm'].LLVMIsAStoreInst
    LLVMIsAStoreInst.restype = LLVMValueRef
    LLVMIsAStoreInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsABranchInst = _libraries['llvm'].LLVMIsABranchInst
    LLVMIsABranchInst.restype = LLVMValueRef
    LLVMIsABranchInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAIndirectBrInst = _libraries['llvm'].LLVMIsAIndirectBrInst
    LLVMIsAIndirectBrInst.restype = LLVMValueRef
    LLVMIsAIndirectBrInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAInvokeInst = _libraries['llvm'].LLVMIsAInvokeInst
    LLVMIsAInvokeInst.restype = LLVMValueRef
    LLVMIsAInvokeInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAReturnInst = _libraries['llvm'].LLVMIsAReturnInst
    LLVMIsAReturnInst.restype = LLVMValueRef
    LLVMIsAReturnInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsASwitchInst = _libraries['llvm'].LLVMIsASwitchInst
    LLVMIsASwitchInst.restype = LLVMValueRef
    LLVMIsASwitchInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUnreachableInst = _libraries['llvm'].LLVMIsAUnreachableInst
    LLVMIsAUnreachableInst.restype = LLVMValueRef
    LLVMIsAUnreachableInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAResumeInst = _libraries['llvm'].LLVMIsAResumeInst
    LLVMIsAResumeInst.restype = LLVMValueRef
    LLVMIsAResumeInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACleanupReturnInst = _libraries['llvm'].LLVMIsACleanupReturnInst
    LLVMIsACleanupReturnInst.restype = LLVMValueRef
    LLVMIsACleanupReturnInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACatchReturnInst = _libraries['llvm'].LLVMIsACatchReturnInst
    LLVMIsACatchReturnInst.restype = LLVMValueRef
    LLVMIsACatchReturnInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACatchSwitchInst = _libraries['llvm'].LLVMIsACatchSwitchInst
    LLVMIsACatchSwitchInst.restype = LLVMValueRef
    LLVMIsACatchSwitchInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACallBrInst = _libraries['llvm'].LLVMIsACallBrInst
    LLVMIsACallBrInst.restype = LLVMValueRef
    LLVMIsACallBrInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFuncletPadInst = _libraries['llvm'].LLVMIsAFuncletPadInst
    LLVMIsAFuncletPadInst.restype = LLVMValueRef
    LLVMIsAFuncletPadInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACatchPadInst = _libraries['llvm'].LLVMIsACatchPadInst
    LLVMIsACatchPadInst.restype = LLVMValueRef
    LLVMIsACatchPadInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACleanupPadInst = _libraries['llvm'].LLVMIsACleanupPadInst
    LLVMIsACleanupPadInst.restype = LLVMValueRef
    LLVMIsACleanupPadInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUnaryInstruction = _libraries['llvm'].LLVMIsAUnaryInstruction
    LLVMIsAUnaryInstruction.restype = LLVMValueRef
    LLVMIsAUnaryInstruction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAAllocaInst = _libraries['llvm'].LLVMIsAAllocaInst
    LLVMIsAAllocaInst.restype = LLVMValueRef
    LLVMIsAAllocaInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsACastInst = _libraries['llvm'].LLVMIsACastInst
    LLVMIsACastInst.restype = LLVMValueRef
    LLVMIsACastInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAAddrSpaceCastInst = _libraries['llvm'].LLVMIsAAddrSpaceCastInst
    LLVMIsAAddrSpaceCastInst.restype = LLVMValueRef
    LLVMIsAAddrSpaceCastInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsABitCastInst = _libraries['llvm'].LLVMIsABitCastInst
    LLVMIsABitCastInst.restype = LLVMValueRef
    LLVMIsABitCastInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFPExtInst = _libraries['llvm'].LLVMIsAFPExtInst
    LLVMIsAFPExtInst.restype = LLVMValueRef
    LLVMIsAFPExtInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFPToSIInst = _libraries['llvm'].LLVMIsAFPToSIInst
    LLVMIsAFPToSIInst.restype = LLVMValueRef
    LLVMIsAFPToSIInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFPToUIInst = _libraries['llvm'].LLVMIsAFPToUIInst
    LLVMIsAFPToUIInst.restype = LLVMValueRef
    LLVMIsAFPToUIInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFPTruncInst = _libraries['llvm'].LLVMIsAFPTruncInst
    LLVMIsAFPTruncInst.restype = LLVMValueRef
    LLVMIsAFPTruncInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAIntToPtrInst = _libraries['llvm'].LLVMIsAIntToPtrInst
    LLVMIsAIntToPtrInst.restype = LLVMValueRef
    LLVMIsAIntToPtrInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAPtrToIntInst = _libraries['llvm'].LLVMIsAPtrToIntInst
    LLVMIsAPtrToIntInst.restype = LLVMValueRef
    LLVMIsAPtrToIntInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsASExtInst = _libraries['llvm'].LLVMIsASExtInst
    LLVMIsASExtInst.restype = LLVMValueRef
    LLVMIsASExtInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsASIToFPInst = _libraries['llvm'].LLVMIsASIToFPInst
    LLVMIsASIToFPInst.restype = LLVMValueRef
    LLVMIsASIToFPInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsATruncInst = _libraries['llvm'].LLVMIsATruncInst
    LLVMIsATruncInst.restype = LLVMValueRef
    LLVMIsATruncInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAUIToFPInst = _libraries['llvm'].LLVMIsAUIToFPInst
    LLVMIsAUIToFPInst.restype = LLVMValueRef
    LLVMIsAUIToFPInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAZExtInst = _libraries['llvm'].LLVMIsAZExtInst
    LLVMIsAZExtInst.restype = LLVMValueRef
    LLVMIsAZExtInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAExtractValueInst = _libraries['llvm'].LLVMIsAExtractValueInst
    LLVMIsAExtractValueInst.restype = LLVMValueRef
    LLVMIsAExtractValueInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsALoadInst = _libraries['llvm'].LLVMIsALoadInst
    LLVMIsALoadInst.restype = LLVMValueRef
    LLVMIsALoadInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAVAArgInst = _libraries['llvm'].LLVMIsAVAArgInst
    LLVMIsAVAArgInst.restype = LLVMValueRef
    LLVMIsAVAArgInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFreezeInst = _libraries['llvm'].LLVMIsAFreezeInst
    LLVMIsAFreezeInst.restype = LLVMValueRef
    LLVMIsAFreezeInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAAtomicCmpXchgInst = _libraries['llvm'].LLVMIsAAtomicCmpXchgInst
    LLVMIsAAtomicCmpXchgInst.restype = LLVMValueRef
    LLVMIsAAtomicCmpXchgInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAAtomicRMWInst = _libraries['llvm'].LLVMIsAAtomicRMWInst
    LLVMIsAAtomicRMWInst.restype = LLVMValueRef
    LLVMIsAAtomicRMWInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAFenceInst = _libraries['llvm'].LLVMIsAFenceInst
    LLVMIsAFenceInst.restype = LLVMValueRef
    LLVMIsAFenceInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMDNode = _libraries['llvm'].LLVMIsAMDNode
    LLVMIsAMDNode.restype = LLVMValueRef
    LLVMIsAMDNode.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsAMDString = _libraries['llvm'].LLVMIsAMDString
    LLVMIsAMDString.restype = LLVMValueRef
    LLVMIsAMDString.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetValueName = _libraries['llvm'].LLVMGetValueName
    LLVMGetValueName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetValueName.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetValueName = _libraries['llvm'].LLVMSetValueName
    LLVMSetValueName.restype = None
    LLVMSetValueName.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetFirstUse = _libraries['llvm'].LLVMGetFirstUse
    LLVMGetFirstUse.restype = LLVMUseRef
    LLVMGetFirstUse.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNextUse = _libraries['llvm'].LLVMGetNextUse
    LLVMGetNextUse.restype = LLVMUseRef
    LLVMGetNextUse.argtypes = [LLVMUseRef]
except AttributeError:
    pass
try:
    LLVMGetUser = _libraries['llvm'].LLVMGetUser
    LLVMGetUser.restype = LLVMValueRef
    LLVMGetUser.argtypes = [LLVMUseRef]
except AttributeError:
    pass
try:
    LLVMGetUsedValue = _libraries['llvm'].LLVMGetUsedValue
    LLVMGetUsedValue.restype = LLVMValueRef
    LLVMGetUsedValue.argtypes = [LLVMUseRef]
except AttributeError:
    pass
try:
    LLVMGetOperand = _libraries['llvm'].LLVMGetOperand
    LLVMGetOperand.restype = LLVMValueRef
    LLVMGetOperand.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetOperandUse = _libraries['llvm'].LLVMGetOperandUse
    LLVMGetOperandUse.restype = LLVMUseRef
    LLVMGetOperandUse.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMSetOperand = _libraries['llvm'].LLVMSetOperand
    LLVMSetOperand.restype = None
    LLVMSetOperand.argtypes = [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNumOperands = _libraries['llvm'].LLVMGetNumOperands
    LLVMGetNumOperands.restype = ctypes.c_int32
    LLVMGetNumOperands.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNull = _libraries['llvm'].LLVMConstNull
    LLVMConstNull.restype = LLVMValueRef
    LLVMConstNull.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstAllOnes = _libraries['llvm'].LLVMConstAllOnes
    LLVMConstAllOnes.restype = LLVMValueRef
    LLVMConstAllOnes.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetUndef = _libraries['llvm'].LLVMGetUndef
    LLVMGetUndef.restype = LLVMValueRef
    LLVMGetUndef.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMGetPoison = _libraries['llvm'].LLVMGetPoison
    LLVMGetPoison.restype = LLVMValueRef
    LLVMGetPoison.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMIsNull = _libraries['llvm'].LLVMIsNull
    LLVMIsNull.restype = LLVMBool
    LLVMIsNull.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstPointerNull = _libraries['llvm'].LLVMConstPointerNull
    LLVMConstPointerNull.restype = LLVMValueRef
    LLVMConstPointerNull.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstInt = _libraries['llvm'].LLVMConstInt
    LLVMConstInt.restype = LLVMValueRef
    LLVMConstInt.argtypes = [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError:
    pass
try:
    LLVMConstIntOfArbitraryPrecision = _libraries['llvm'].LLVMConstIntOfArbitraryPrecision
    LLVMConstIntOfArbitraryPrecision.restype = LLVMValueRef
    LLVMConstIntOfArbitraryPrecision.argtypes = [LLVMTypeRef, ctypes.c_uint32, ctypes.c_uint64 * 0]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    LLVMConstIntOfString = _libraries['llvm'].LLVMConstIntOfString
    LLVMConstIntOfString.restype = LLVMValueRef
    LLVMConstIntOfString.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), uint8_t]
except AttributeError:
    pass
try:
    LLVMConstIntOfStringAndSize = _libraries['llvm'].LLVMConstIntOfStringAndSize
    LLVMConstIntOfStringAndSize.restype = LLVMValueRef
    LLVMConstIntOfStringAndSize.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, uint8_t]
except AttributeError:
    pass
try:
    LLVMConstReal = _libraries['llvm'].LLVMConstReal
    LLVMConstReal.restype = LLVMValueRef
    LLVMConstReal.argtypes = [LLVMTypeRef, ctypes.c_double]
except AttributeError:
    pass
try:
    LLVMConstRealOfString = _libraries['llvm'].LLVMConstRealOfString
    LLVMConstRealOfString.restype = LLVMValueRef
    LLVMConstRealOfString.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMConstRealOfStringAndSize = _libraries['llvm'].LLVMConstRealOfStringAndSize
    LLVMConstRealOfStringAndSize.restype = LLVMValueRef
    LLVMConstRealOfStringAndSize.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstIntGetZExtValue = _libraries['llvm'].LLVMConstIntGetZExtValue
    LLVMConstIntGetZExtValue.restype = ctypes.c_uint64
    LLVMConstIntGetZExtValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstIntGetSExtValue = _libraries['llvm'].LLVMConstIntGetSExtValue
    LLVMConstIntGetSExtValue.restype = ctypes.c_int64
    LLVMConstIntGetSExtValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstRealGetDouble = _libraries['llvm'].LLVMConstRealGetDouble
    LLVMConstRealGetDouble.restype = ctypes.c_double
    LLVMConstRealGetDouble.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    LLVMConstStringInContext = _libraries['llvm'].LLVMConstStringInContext
    LLVMConstStringInContext.restype = LLVMValueRef
    LLVMConstStringInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMConstString = _libraries['llvm'].LLVMConstString
    LLVMConstString.restype = LLVMValueRef
    LLVMConstString.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMIsConstantString = _libraries['llvm'].LLVMIsConstantString
    LLVMIsConstantString.restype = LLVMBool
    LLVMIsConstantString.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetAsString = _libraries['llvm'].LLVMGetAsString
    LLVMGetAsString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetAsString.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMConstStructInContext = _libraries['llvm'].LLVMConstStructInContext
    LLVMConstStructInContext.restype = LLVMValueRef
    LLVMConstStructInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMConstStruct = _libraries['llvm'].LLVMConstStruct
    LLVMConstStruct.restype = LLVMValueRef
    LLVMConstStruct.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, LLVMBool]
except AttributeError:
    pass
try:
    LLVMConstArray = _libraries['llvm'].LLVMConstArray
    LLVMConstArray.restype = LLVMValueRef
    LLVMConstArray.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstNamedStruct = _libraries['llvm'].LLVMConstNamedStruct
    LLVMConstNamedStruct.restype = LLVMValueRef
    LLVMConstNamedStruct.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetElementAsConstant = _libraries['llvm'].LLVMGetElementAsConstant
    LLVMGetElementAsConstant.restype = LLVMValueRef
    LLVMGetElementAsConstant.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstVector = _libraries['llvm'].LLVMConstVector
    LLVMConstVector.restype = LLVMValueRef
    LLVMConstVector.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetConstOpcode = _libraries['llvm'].LLVMGetConstOpcode
    LLVMGetConstOpcode.restype = LLVMOpcode
    LLVMGetConstOpcode.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAlignOf = _libraries['llvm'].LLVMAlignOf
    LLVMAlignOf.restype = LLVMValueRef
    LLVMAlignOf.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMSizeOf = _libraries['llvm'].LLVMSizeOf
    LLVMSizeOf.restype = LLVMValueRef
    LLVMSizeOf.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstNeg = _libraries['llvm'].LLVMConstNeg
    LLVMConstNeg.restype = LLVMValueRef
    LLVMConstNeg.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNSWNeg = _libraries['llvm'].LLVMConstNSWNeg
    LLVMConstNSWNeg.restype = LLVMValueRef
    LLVMConstNSWNeg.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNUWNeg = _libraries['llvm'].LLVMConstNUWNeg
    LLVMConstNUWNeg.restype = LLVMValueRef
    LLVMConstNUWNeg.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFNeg = _libraries['llvm'].LLVMConstFNeg
    LLVMConstFNeg.restype = LLVMValueRef
    LLVMConstFNeg.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNot = _libraries['llvm'].LLVMConstNot
    LLVMConstNot.restype = LLVMValueRef
    LLVMConstNot.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstAdd = _libraries['llvm'].LLVMConstAdd
    LLVMConstAdd.restype = LLVMValueRef
    LLVMConstAdd.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNSWAdd = _libraries['llvm'].LLVMConstNSWAdd
    LLVMConstNSWAdd.restype = LLVMValueRef
    LLVMConstNSWAdd.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNUWAdd = _libraries['llvm'].LLVMConstNUWAdd
    LLVMConstNUWAdd.restype = LLVMValueRef
    LLVMConstNUWAdd.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFAdd = _libraries['llvm'].LLVMConstFAdd
    LLVMConstFAdd.restype = LLVMValueRef
    LLVMConstFAdd.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstSub = _libraries['llvm'].LLVMConstSub
    LLVMConstSub.restype = LLVMValueRef
    LLVMConstSub.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNSWSub = _libraries['llvm'].LLVMConstNSWSub
    LLVMConstNSWSub.restype = LLVMValueRef
    LLVMConstNSWSub.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNUWSub = _libraries['llvm'].LLVMConstNUWSub
    LLVMConstNUWSub.restype = LLVMValueRef
    LLVMConstNUWSub.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFSub = _libraries['llvm'].LLVMConstFSub
    LLVMConstFSub.restype = LLVMValueRef
    LLVMConstFSub.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstMul = _libraries['llvm'].LLVMConstMul
    LLVMConstMul.restype = LLVMValueRef
    LLVMConstMul.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNSWMul = _libraries['llvm'].LLVMConstNSWMul
    LLVMConstNSWMul.restype = LLVMValueRef
    LLVMConstNSWMul.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstNUWMul = _libraries['llvm'].LLVMConstNUWMul
    LLVMConstNUWMul.restype = LLVMValueRef
    LLVMConstNUWMul.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFMul = _libraries['llvm'].LLVMConstFMul
    LLVMConstFMul.restype = LLVMValueRef
    LLVMConstFMul.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstUDiv = _libraries['llvm'].LLVMConstUDiv
    LLVMConstUDiv.restype = LLVMValueRef
    LLVMConstUDiv.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstExactUDiv = _libraries['llvm'].LLVMConstExactUDiv
    LLVMConstExactUDiv.restype = LLVMValueRef
    LLVMConstExactUDiv.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstSDiv = _libraries['llvm'].LLVMConstSDiv
    LLVMConstSDiv.restype = LLVMValueRef
    LLVMConstSDiv.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstExactSDiv = _libraries['llvm'].LLVMConstExactSDiv
    LLVMConstExactSDiv.restype = LLVMValueRef
    LLVMConstExactSDiv.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFDiv = _libraries['llvm'].LLVMConstFDiv
    LLVMConstFDiv.restype = LLVMValueRef
    LLVMConstFDiv.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstURem = _libraries['llvm'].LLVMConstURem
    LLVMConstURem.restype = LLVMValueRef
    LLVMConstURem.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstSRem = _libraries['llvm'].LLVMConstSRem
    LLVMConstSRem.restype = LLVMValueRef
    LLVMConstSRem.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFRem = _libraries['llvm'].LLVMConstFRem
    LLVMConstFRem.restype = LLVMValueRef
    LLVMConstFRem.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstAnd = _libraries['llvm'].LLVMConstAnd
    LLVMConstAnd.restype = LLVMValueRef
    LLVMConstAnd.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstOr = _libraries['llvm'].LLVMConstOr
    LLVMConstOr.restype = LLVMValueRef
    LLVMConstOr.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstXor = _libraries['llvm'].LLVMConstXor
    LLVMConstXor.restype = LLVMValueRef
    LLVMConstXor.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstICmp = _libraries['llvm'].LLVMConstICmp
    LLVMConstICmp.restype = LLVMValueRef
    LLVMConstICmp.argtypes = [LLVMIntPredicate, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstFCmp = _libraries['llvm'].LLVMConstFCmp
    LLVMConstFCmp.restype = LLVMValueRef
    LLVMConstFCmp.argtypes = [LLVMRealPredicate, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstShl = _libraries['llvm'].LLVMConstShl
    LLVMConstShl.restype = LLVMValueRef
    LLVMConstShl.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstLShr = _libraries['llvm'].LLVMConstLShr
    LLVMConstLShr.restype = LLVMValueRef
    LLVMConstLShr.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstAShr = _libraries['llvm'].LLVMConstAShr
    LLVMConstAShr.restype = LLVMValueRef
    LLVMConstAShr.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstGEP = _libraries['llvm'].LLVMConstGEP
    LLVMConstGEP.restype = LLVMValueRef
    LLVMConstGEP.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstGEP2 = _libraries['llvm'].LLVMConstGEP2
    LLVMConstGEP2.restype = LLVMValueRef
    LLVMConstGEP2.argtypes = [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstInBoundsGEP = _libraries['llvm'].LLVMConstInBoundsGEP
    LLVMConstInBoundsGEP.restype = LLVMValueRef
    LLVMConstInBoundsGEP.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstInBoundsGEP2 = _libraries['llvm'].LLVMConstInBoundsGEP2
    LLVMConstInBoundsGEP2.restype = LLVMValueRef
    LLVMConstInBoundsGEP2.argtypes = [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstTrunc = _libraries['llvm'].LLVMConstTrunc
    LLVMConstTrunc.restype = LLVMValueRef
    LLVMConstTrunc.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstSExt = _libraries['llvm'].LLVMConstSExt
    LLVMConstSExt.restype = LLVMValueRef
    LLVMConstSExt.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstZExt = _libraries['llvm'].LLVMConstZExt
    LLVMConstZExt.restype = LLVMValueRef
    LLVMConstZExt.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstFPTrunc = _libraries['llvm'].LLVMConstFPTrunc
    LLVMConstFPTrunc.restype = LLVMValueRef
    LLVMConstFPTrunc.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstFPExt = _libraries['llvm'].LLVMConstFPExt
    LLVMConstFPExt.restype = LLVMValueRef
    LLVMConstFPExt.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstUIToFP = _libraries['llvm'].LLVMConstUIToFP
    LLVMConstUIToFP.restype = LLVMValueRef
    LLVMConstUIToFP.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstSIToFP = _libraries['llvm'].LLVMConstSIToFP
    LLVMConstSIToFP.restype = LLVMValueRef
    LLVMConstSIToFP.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstFPToUI = _libraries['llvm'].LLVMConstFPToUI
    LLVMConstFPToUI.restype = LLVMValueRef
    LLVMConstFPToUI.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstFPToSI = _libraries['llvm'].LLVMConstFPToSI
    LLVMConstFPToSI.restype = LLVMValueRef
    LLVMConstFPToSI.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstPtrToInt = _libraries['llvm'].LLVMConstPtrToInt
    LLVMConstPtrToInt.restype = LLVMValueRef
    LLVMConstPtrToInt.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstIntToPtr = _libraries['llvm'].LLVMConstIntToPtr
    LLVMConstIntToPtr.restype = LLVMValueRef
    LLVMConstIntToPtr.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstBitCast = _libraries['llvm'].LLVMConstBitCast
    LLVMConstBitCast.restype = LLVMValueRef
    LLVMConstBitCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstAddrSpaceCast = _libraries['llvm'].LLVMConstAddrSpaceCast
    LLVMConstAddrSpaceCast.restype = LLVMValueRef
    LLVMConstAddrSpaceCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstZExtOrBitCast = _libraries['llvm'].LLVMConstZExtOrBitCast
    LLVMConstZExtOrBitCast.restype = LLVMValueRef
    LLVMConstZExtOrBitCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstSExtOrBitCast = _libraries['llvm'].LLVMConstSExtOrBitCast
    LLVMConstSExtOrBitCast.restype = LLVMValueRef
    LLVMConstSExtOrBitCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstTruncOrBitCast = _libraries['llvm'].LLVMConstTruncOrBitCast
    LLVMConstTruncOrBitCast.restype = LLVMValueRef
    LLVMConstTruncOrBitCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstPointerCast = _libraries['llvm'].LLVMConstPointerCast
    LLVMConstPointerCast.restype = LLVMValueRef
    LLVMConstPointerCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstIntCast = _libraries['llvm'].LLVMConstIntCast
    LLVMConstIntCast.restype = LLVMValueRef
    LLVMConstIntCast.argtypes = [LLVMValueRef, LLVMTypeRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMConstFPCast = _libraries['llvm'].LLVMConstFPCast
    LLVMConstFPCast.restype = LLVMValueRef
    LLVMConstFPCast.argtypes = [LLVMValueRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMConstSelect = _libraries['llvm'].LLVMConstSelect
    LLVMConstSelect.restype = LLVMValueRef
    LLVMConstSelect.argtypes = [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstExtractElement = _libraries['llvm'].LLVMConstExtractElement
    LLVMConstExtractElement.restype = LLVMValueRef
    LLVMConstExtractElement.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstInsertElement = _libraries['llvm'].LLVMConstInsertElement
    LLVMConstInsertElement.restype = LLVMValueRef
    LLVMConstInsertElement.argtypes = [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstShuffleVector = _libraries['llvm'].LLVMConstShuffleVector
    LLVMConstShuffleVector.restype = LLVMValueRef
    LLVMConstShuffleVector.argtypes = [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMConstExtractValue = _libraries['llvm'].LLVMConstExtractValue
    LLVMConstExtractValue.restype = LLVMValueRef
    LLVMConstExtractValue.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMConstInsertValue = _libraries['llvm'].LLVMConstInsertValue
    LLVMConstInsertValue.restype = LLVMValueRef
    LLVMConstInsertValue.argtypes = [LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBlockAddress = _libraries['llvm'].LLVMBlockAddress
    LLVMBlockAddress.restype = LLVMValueRef
    LLVMBlockAddress.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMConstInlineAsm = _libraries['llvm'].LLVMConstInlineAsm
    LLVMConstInlineAsm.restype = LLVMValueRef
    LLVMConstInlineAsm.argtypes = [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMBool, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetGlobalParent = _libraries['llvm'].LLVMGetGlobalParent
    LLVMGetGlobalParent.restype = LLVMModuleRef
    LLVMGetGlobalParent.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsDeclaration = _libraries['llvm'].LLVMIsDeclaration
    LLVMIsDeclaration.restype = LLVMBool
    LLVMIsDeclaration.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetLinkage = _libraries['llvm'].LLVMGetLinkage
    LLVMGetLinkage.restype = LLVMLinkage
    LLVMGetLinkage.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetLinkage = _libraries['llvm'].LLVMSetLinkage
    LLVMSetLinkage.restype = None
    LLVMSetLinkage.argtypes = [LLVMValueRef, LLVMLinkage]
except AttributeError:
    pass
try:
    LLVMGetSection = _libraries['llvm'].LLVMGetSection
    LLVMGetSection.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetSection.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetSection = _libraries['llvm'].LLVMSetSection
    LLVMSetSection.restype = None
    LLVMSetSection.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetVisibility = _libraries['llvm'].LLVMGetVisibility
    LLVMGetVisibility.restype = LLVMVisibility
    LLVMGetVisibility.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetVisibility = _libraries['llvm'].LLVMSetVisibility
    LLVMSetVisibility.restype = None
    LLVMSetVisibility.argtypes = [LLVMValueRef, LLVMVisibility]
except AttributeError:
    pass
try:
    LLVMGetDLLStorageClass = _libraries['llvm'].LLVMGetDLLStorageClass
    LLVMGetDLLStorageClass.restype = LLVMDLLStorageClass
    LLVMGetDLLStorageClass.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetDLLStorageClass = _libraries['llvm'].LLVMSetDLLStorageClass
    LLVMSetDLLStorageClass.restype = None
    LLVMSetDLLStorageClass.argtypes = [LLVMValueRef, LLVMDLLStorageClass]
except AttributeError:
    pass
try:
    LLVMGetUnnamedAddress = _libraries['llvm'].LLVMGetUnnamedAddress
    LLVMGetUnnamedAddress.restype = LLVMUnnamedAddr
    LLVMGetUnnamedAddress.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetUnnamedAddress = _libraries['llvm'].LLVMSetUnnamedAddress
    LLVMSetUnnamedAddress.restype = None
    LLVMSetUnnamedAddress.argtypes = [LLVMValueRef, LLVMUnnamedAddr]
except AttributeError:
    pass
try:
    LLVMGlobalGetValueType = _libraries['llvm'].LLVMGlobalGetValueType
    LLVMGlobalGetValueType.restype = LLVMTypeRef
    LLVMGlobalGetValueType.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMHasUnnamedAddr = _libraries['llvm'].LLVMHasUnnamedAddr
    LLVMHasUnnamedAddr.restype = LLVMBool
    LLVMHasUnnamedAddr.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetUnnamedAddr = _libraries['llvm'].LLVMSetUnnamedAddr
    LLVMSetUnnamedAddr.restype = None
    LLVMSetUnnamedAddr.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetAlignment = _libraries['llvm'].LLVMGetAlignment
    LLVMGetAlignment.restype = ctypes.c_uint32
    LLVMGetAlignment.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetAlignment = _libraries['llvm'].LLVMSetAlignment
    LLVMSetAlignment.restype = None
    LLVMSetAlignment.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGlobalSetMetadata = _libraries['llvm'].LLVMGlobalSetMetadata
    LLVMGlobalSetMetadata.restype = None
    LLVMGlobalSetMetadata.argtypes = [LLVMValueRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMGlobalEraseMetadata = _libraries['llvm'].LLVMGlobalEraseMetadata
    LLVMGlobalEraseMetadata.restype = None
    LLVMGlobalEraseMetadata.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGlobalClearMetadata = _libraries['llvm'].LLVMGlobalClearMetadata
    LLVMGlobalClearMetadata.restype = None
    LLVMGlobalClearMetadata.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGlobalCopyAllMetadata = _libraries['llvm'].LLVMGlobalCopyAllMetadata
    LLVMGlobalCopyAllMetadata.restype = ctypes.POINTER(struct_LLVMOpaqueValueMetadataEntry)
    LLVMGlobalCopyAllMetadata.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMDisposeValueMetadataEntries = _libraries['llvm'].LLVMDisposeValueMetadataEntries
    LLVMDisposeValueMetadataEntries.restype = None
    LLVMDisposeValueMetadataEntries.argtypes = [ctypes.POINTER(struct_LLVMOpaqueValueMetadataEntry)]
except AttributeError:
    pass
try:
    LLVMValueMetadataEntriesGetKind = _libraries['llvm'].LLVMValueMetadataEntriesGetKind
    LLVMValueMetadataEntriesGetKind.restype = ctypes.c_uint32
    LLVMValueMetadataEntriesGetKind.argtypes = [ctypes.POINTER(struct_LLVMOpaqueValueMetadataEntry), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMValueMetadataEntriesGetMetadata = _libraries['llvm'].LLVMValueMetadataEntriesGetMetadata
    LLVMValueMetadataEntriesGetMetadata.restype = LLVMMetadataRef
    LLVMValueMetadataEntriesGetMetadata.argtypes = [ctypes.POINTER(struct_LLVMOpaqueValueMetadataEntry), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddGlobal = _libraries['llvm'].LLVMAddGlobal
    LLVMAddGlobal.restype = LLVMValueRef
    LLVMAddGlobal.argtypes = [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAddGlobalInAddressSpace = _libraries['llvm'].LLVMAddGlobalInAddressSpace
    LLVMAddGlobalInAddressSpace.restype = LLVMValueRef
    LLVMAddGlobalInAddressSpace.argtypes = [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetNamedGlobal = _libraries['llvm'].LLVMGetNamedGlobal
    LLVMGetNamedGlobal.restype = LLVMValueRef
    LLVMGetNamedGlobal.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetFirstGlobal = _libraries['llvm'].LLVMGetFirstGlobal
    LLVMGetFirstGlobal.restype = LLVMValueRef
    LLVMGetFirstGlobal.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetLastGlobal = _libraries['llvm'].LLVMGetLastGlobal
    LLVMGetLastGlobal.restype = LLVMValueRef
    LLVMGetLastGlobal.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetNextGlobal = _libraries['llvm'].LLVMGetNextGlobal
    LLVMGetNextGlobal.restype = LLVMValueRef
    LLVMGetNextGlobal.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousGlobal = _libraries['llvm'].LLVMGetPreviousGlobal
    LLVMGetPreviousGlobal.restype = LLVMValueRef
    LLVMGetPreviousGlobal.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMDeleteGlobal = _libraries['llvm'].LLVMDeleteGlobal
    LLVMDeleteGlobal.restype = None
    LLVMDeleteGlobal.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetInitializer = _libraries['llvm'].LLVMGetInitializer
    LLVMGetInitializer.restype = LLVMValueRef
    LLVMGetInitializer.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetInitializer = _libraries['llvm'].LLVMSetInitializer
    LLVMSetInitializer.restype = None
    LLVMSetInitializer.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsThreadLocal = _libraries['llvm'].LLVMIsThreadLocal
    LLVMIsThreadLocal.restype = LLVMBool
    LLVMIsThreadLocal.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetThreadLocal = _libraries['llvm'].LLVMSetThreadLocal
    LLVMSetThreadLocal.restype = None
    LLVMSetThreadLocal.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMIsGlobalConstant = _libraries['llvm'].LLVMIsGlobalConstant
    LLVMIsGlobalConstant.restype = LLVMBool
    LLVMIsGlobalConstant.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetGlobalConstant = _libraries['llvm'].LLVMSetGlobalConstant
    LLVMSetGlobalConstant.restype = None
    LLVMSetGlobalConstant.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetThreadLocalMode = _libraries['llvm'].LLVMGetThreadLocalMode
    LLVMGetThreadLocalMode.restype = LLVMThreadLocalMode
    LLVMGetThreadLocalMode.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetThreadLocalMode = _libraries['llvm'].LLVMSetThreadLocalMode
    LLVMSetThreadLocalMode.restype = None
    LLVMSetThreadLocalMode.argtypes = [LLVMValueRef, LLVMThreadLocalMode]
except AttributeError:
    pass
try:
    LLVMIsExternallyInitialized = _libraries['llvm'].LLVMIsExternallyInitialized
    LLVMIsExternallyInitialized.restype = LLVMBool
    LLVMIsExternallyInitialized.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetExternallyInitialized = _libraries['llvm'].LLVMSetExternallyInitialized
    LLVMSetExternallyInitialized.restype = None
    LLVMSetExternallyInitialized.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMAddAlias = _libraries['llvm'].LLVMAddAlias
    LLVMAddAlias.restype = LLVMValueRef
    LLVMAddAlias.argtypes = [LLVMModuleRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAddAlias2 = _libraries['llvm'].LLVMAddAlias2
    LLVMAddAlias2.restype = LLVMValueRef
    LLVMAddAlias2.argtypes = [LLVMModuleRef, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetNamedGlobalAlias = _libraries['llvm'].LLVMGetNamedGlobalAlias
    LLVMGetNamedGlobalAlias.restype = LLVMValueRef
    LLVMGetNamedGlobalAlias.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetFirstGlobalAlias = _libraries['llvm'].LLVMGetFirstGlobalAlias
    LLVMGetFirstGlobalAlias.restype = LLVMValueRef
    LLVMGetFirstGlobalAlias.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetLastGlobalAlias = _libraries['llvm'].LLVMGetLastGlobalAlias
    LLVMGetLastGlobalAlias.restype = LLVMValueRef
    LLVMGetLastGlobalAlias.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetNextGlobalAlias = _libraries['llvm'].LLVMGetNextGlobalAlias
    LLVMGetNextGlobalAlias.restype = LLVMValueRef
    LLVMGetNextGlobalAlias.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousGlobalAlias = _libraries['llvm'].LLVMGetPreviousGlobalAlias
    LLVMGetPreviousGlobalAlias.restype = LLVMValueRef
    LLVMGetPreviousGlobalAlias.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAliasGetAliasee = _libraries['llvm'].LLVMAliasGetAliasee
    LLVMAliasGetAliasee.restype = LLVMValueRef
    LLVMAliasGetAliasee.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAliasSetAliasee = _libraries['llvm'].LLVMAliasSetAliasee
    LLVMAliasSetAliasee.restype = None
    LLVMAliasSetAliasee.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMDeleteFunction = _libraries['llvm'].LLVMDeleteFunction
    LLVMDeleteFunction.restype = None
    LLVMDeleteFunction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMHasPersonalityFn = _libraries['llvm'].LLVMHasPersonalityFn
    LLVMHasPersonalityFn.restype = LLVMBool
    LLVMHasPersonalityFn.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPersonalityFn = _libraries['llvm'].LLVMGetPersonalityFn
    LLVMGetPersonalityFn.restype = LLVMValueRef
    LLVMGetPersonalityFn.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetPersonalityFn = _libraries['llvm'].LLVMSetPersonalityFn
    LLVMSetPersonalityFn.restype = None
    LLVMSetPersonalityFn.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMLookupIntrinsicID = _libraries['llvm'].LLVMLookupIntrinsicID
    LLVMLookupIntrinsicID.restype = ctypes.c_uint32
    LLVMLookupIntrinsicID.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetIntrinsicID = _libraries['llvm'].LLVMGetIntrinsicID
    LLVMGetIntrinsicID.restype = ctypes.c_uint32
    LLVMGetIntrinsicID.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetIntrinsicDeclaration = _libraries['llvm'].LLVMGetIntrinsicDeclaration
    LLVMGetIntrinsicDeclaration.restype = LLVMValueRef
    LLVMGetIntrinsicDeclaration.argtypes = [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), size_t]
except AttributeError:
    pass
try:
    LLVMIntrinsicGetType = _libraries['llvm'].LLVMIntrinsicGetType
    LLVMIntrinsicGetType.restype = LLVMTypeRef
    LLVMIntrinsicGetType.argtypes = [LLVMContextRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), size_t]
except AttributeError:
    pass
try:
    LLVMIntrinsicGetName = _libraries['llvm'].LLVMIntrinsicGetName
    LLVMIntrinsicGetName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMIntrinsicGetName.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMIntrinsicCopyOverloadedName = _libraries['llvm'].LLVMIntrinsicCopyOverloadedName
    LLVMIntrinsicCopyOverloadedName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMIntrinsicCopyOverloadedName.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), size_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMIntrinsicCopyOverloadedName2 = _libraries['llvm'].LLVMIntrinsicCopyOverloadedName2
    LLVMIntrinsicCopyOverloadedName2.restype = ctypes.POINTER(ctypes.c_char)
    LLVMIntrinsicCopyOverloadedName2.argtypes = [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), size_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMIntrinsicIsOverloaded = _libraries['llvm'].LLVMIntrinsicIsOverloaded
    LLVMIntrinsicIsOverloaded.restype = LLVMBool
    LLVMIntrinsicIsOverloaded.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetFunctionCallConv = _libraries['llvm'].LLVMGetFunctionCallConv
    LLVMGetFunctionCallConv.restype = ctypes.c_uint32
    LLVMGetFunctionCallConv.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetFunctionCallConv = _libraries['llvm'].LLVMSetFunctionCallConv
    LLVMSetFunctionCallConv.restype = None
    LLVMSetFunctionCallConv.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetGC = _libraries['llvm'].LLVMGetGC
    LLVMGetGC.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetGC.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetGC = _libraries['llvm'].LLVMSetGC
    LLVMSetGC.restype = None
    LLVMSetGC.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAddAttributeAtIndex = _libraries['llvm'].LLVMAddAttributeAtIndex
    LLVMAddAttributeAtIndex.restype = None
    LLVMAddAttributeAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMGetAttributeCountAtIndex = _libraries['llvm'].LLVMGetAttributeCountAtIndex
    LLVMGetAttributeCountAtIndex.restype = ctypes.c_uint32
    LLVMGetAttributeCountAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex]
except AttributeError:
    pass
try:
    LLVMGetAttributesAtIndex = _libraries['llvm'].LLVMGetAttributesAtIndex
    LLVMGetAttributesAtIndex.restype = None
    LLVMGetAttributesAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueAttributeRef))]
except AttributeError:
    pass
try:
    LLVMGetEnumAttributeAtIndex = _libraries['llvm'].LLVMGetEnumAttributeAtIndex
    LLVMGetEnumAttributeAtIndex.restype = LLVMAttributeRef
    LLVMGetEnumAttributeAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetStringAttributeAtIndex = _libraries['llvm'].LLVMGetStringAttributeAtIndex
    LLVMGetStringAttributeAtIndex.restype = LLVMAttributeRef
    LLVMGetStringAttributeAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMRemoveEnumAttributeAtIndex = _libraries['llvm'].LLVMRemoveEnumAttributeAtIndex
    LLVMRemoveEnumAttributeAtIndex.restype = None
    LLVMRemoveEnumAttributeAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMRemoveStringAttributeAtIndex = _libraries['llvm'].LLVMRemoveStringAttributeAtIndex
    LLVMRemoveStringAttributeAtIndex.restype = None
    LLVMRemoveStringAttributeAtIndex.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddTargetDependentFunctionAttr = _libraries['llvm'].LLVMAddTargetDependentFunctionAttr
    LLVMAddTargetDependentFunctionAttr.restype = None
    LLVMAddTargetDependentFunctionAttr.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMCountParams = _libraries['llvm'].LLVMCountParams
    LLVMCountParams.restype = ctypes.c_uint32
    LLVMCountParams.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetParams = _libraries['llvm'].LLVMGetParams
    LLVMGetParams.restype = None
    LLVMGetParams.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))]
except AttributeError:
    pass
try:
    LLVMGetParam = _libraries['llvm'].LLVMGetParam
    LLVMGetParam.restype = LLVMValueRef
    LLVMGetParam.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetParamParent = _libraries['llvm'].LLVMGetParamParent
    LLVMGetParamParent.restype = LLVMValueRef
    LLVMGetParamParent.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetFirstParam = _libraries['llvm'].LLVMGetFirstParam
    LLVMGetFirstParam.restype = LLVMValueRef
    LLVMGetFirstParam.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetLastParam = _libraries['llvm'].LLVMGetLastParam
    LLVMGetLastParam.restype = LLVMValueRef
    LLVMGetLastParam.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNextParam = _libraries['llvm'].LLVMGetNextParam
    LLVMGetNextParam.restype = LLVMValueRef
    LLVMGetNextParam.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousParam = _libraries['llvm'].LLVMGetPreviousParam
    LLVMGetPreviousParam.restype = LLVMValueRef
    LLVMGetPreviousParam.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetParamAlignment = _libraries['llvm'].LLVMSetParamAlignment
    LLVMSetParamAlignment.restype = None
    LLVMSetParamAlignment.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddGlobalIFunc = _libraries['llvm'].LLVMAddGlobalIFunc
    LLVMAddGlobalIFunc.restype = LLVMValueRef
    LLVMAddGlobalIFunc.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNamedGlobalIFunc = _libraries['llvm'].LLVMGetNamedGlobalIFunc
    LLVMGetNamedGlobalIFunc.restype = LLVMValueRef
    LLVMGetNamedGlobalIFunc.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMGetFirstGlobalIFunc = _libraries['llvm'].LLVMGetFirstGlobalIFunc
    LLVMGetFirstGlobalIFunc.restype = LLVMValueRef
    LLVMGetFirstGlobalIFunc.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetLastGlobalIFunc = _libraries['llvm'].LLVMGetLastGlobalIFunc
    LLVMGetLastGlobalIFunc.restype = LLVMValueRef
    LLVMGetLastGlobalIFunc.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMGetNextGlobalIFunc = _libraries['llvm'].LLVMGetNextGlobalIFunc
    LLVMGetNextGlobalIFunc.restype = LLVMValueRef
    LLVMGetNextGlobalIFunc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousGlobalIFunc = _libraries['llvm'].LLVMGetPreviousGlobalIFunc
    LLVMGetPreviousGlobalIFunc.restype = LLVMValueRef
    LLVMGetPreviousGlobalIFunc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetGlobalIFuncResolver = _libraries['llvm'].LLVMGetGlobalIFuncResolver
    LLVMGetGlobalIFuncResolver.restype = LLVMValueRef
    LLVMGetGlobalIFuncResolver.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetGlobalIFuncResolver = _libraries['llvm'].LLVMSetGlobalIFuncResolver
    LLVMSetGlobalIFuncResolver.restype = None
    LLVMSetGlobalIFuncResolver.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMEraseGlobalIFunc = _libraries['llvm'].LLVMEraseGlobalIFunc
    LLVMEraseGlobalIFunc.restype = None
    LLVMEraseGlobalIFunc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMRemoveGlobalIFunc = _libraries['llvm'].LLVMRemoveGlobalIFunc
    LLVMRemoveGlobalIFunc.restype = None
    LLVMRemoveGlobalIFunc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMMDStringInContext2 = _libraries['llvm'].LLVMMDStringInContext2
    LLVMMDStringInContext2.restype = LLVMMetadataRef
    LLVMMDStringInContext2.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMMDNodeInContext2 = _libraries['llvm'].LLVMMDNodeInContext2
    LLVMMDNodeInContext2.restype = LLVMMetadataRef
    LLVMMDNodeInContext2.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), size_t]
except AttributeError:
    pass
try:
    LLVMMetadataAsValue = _libraries['llvm'].LLVMMetadataAsValue
    LLVMMetadataAsValue.restype = LLVMValueRef
    LLVMMetadataAsValue.argtypes = [LLVMContextRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMValueAsMetadata = _libraries['llvm'].LLVMValueAsMetadata
    LLVMValueAsMetadata.restype = LLVMMetadataRef
    LLVMValueAsMetadata.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetMDString = _libraries['llvm'].LLVMGetMDString
    LLVMGetMDString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetMDString.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMGetMDNodeNumOperands = _libraries['llvm'].LLVMGetMDNodeNumOperands
    LLVMGetMDNodeNumOperands.restype = ctypes.c_uint32
    LLVMGetMDNodeNumOperands.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetMDNodeOperands = _libraries['llvm'].LLVMGetMDNodeOperands
    LLVMGetMDNodeOperands.restype = None
    LLVMGetMDNodeOperands.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))]
except AttributeError:
    pass
try:
    LLVMMDStringInContext = _libraries['llvm'].LLVMMDStringInContext
    LLVMMDStringInContext.restype = LLVMValueRef
    LLVMMDStringInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMMDString = _libraries['llvm'].LLVMMDString
    LLVMMDString.restype = LLVMValueRef
    LLVMMDString.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMMDNodeInContext = _libraries['llvm'].LLVMMDNodeInContext
    LLVMMDNodeInContext.restype = LLVMValueRef
    LLVMMDNodeInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMMDNode = _libraries['llvm'].LLVMMDNode
    LLVMMDNode.restype = LLVMValueRef
    LLVMMDNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBasicBlockAsValue = _libraries['llvm'].LLVMBasicBlockAsValue
    LLVMBasicBlockAsValue.restype = LLVMValueRef
    LLVMBasicBlockAsValue.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMValueIsBasicBlock = _libraries['llvm'].LLVMValueIsBasicBlock
    LLVMValueIsBasicBlock.restype = LLVMBool
    LLVMValueIsBasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMValueAsBasicBlock = _libraries['llvm'].LLVMValueAsBasicBlock
    LLVMValueAsBasicBlock.restype = LLVMBasicBlockRef
    LLVMValueAsBasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetBasicBlockName = _libraries['llvm'].LLVMGetBasicBlockName
    LLVMGetBasicBlockName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetBasicBlockName.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetBasicBlockParent = _libraries['llvm'].LLVMGetBasicBlockParent
    LLVMGetBasicBlockParent.restype = LLVMValueRef
    LLVMGetBasicBlockParent.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetBasicBlockTerminator = _libraries['llvm'].LLVMGetBasicBlockTerminator
    LLVMGetBasicBlockTerminator.restype = LLVMValueRef
    LLVMGetBasicBlockTerminator.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMCountBasicBlocks = _libraries['llvm'].LLVMCountBasicBlocks
    LLVMCountBasicBlocks.restype = ctypes.c_uint32
    LLVMCountBasicBlocks.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetBasicBlocks = _libraries['llvm'].LLVMGetBasicBlocks
    LLVMGetBasicBlocks.restype = None
    LLVMGetBasicBlocks.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueBasicBlock))]
except AttributeError:
    pass
try:
    LLVMGetFirstBasicBlock = _libraries['llvm'].LLVMGetFirstBasicBlock
    LLVMGetFirstBasicBlock.restype = LLVMBasicBlockRef
    LLVMGetFirstBasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetLastBasicBlock = _libraries['llvm'].LLVMGetLastBasicBlock
    LLVMGetLastBasicBlock.restype = LLVMBasicBlockRef
    LLVMGetLastBasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNextBasicBlock = _libraries['llvm'].LLVMGetNextBasicBlock
    LLVMGetNextBasicBlock.restype = LLVMBasicBlockRef
    LLVMGetNextBasicBlock.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousBasicBlock = _libraries['llvm'].LLVMGetPreviousBasicBlock
    LLVMGetPreviousBasicBlock.restype = LLVMBasicBlockRef
    LLVMGetPreviousBasicBlock.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetEntryBasicBlock = _libraries['llvm'].LLVMGetEntryBasicBlock
    LLVMGetEntryBasicBlock.restype = LLVMBasicBlockRef
    LLVMGetEntryBasicBlock.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInsertExistingBasicBlockAfterInsertBlock = _libraries['llvm'].LLVMInsertExistingBasicBlockAfterInsertBlock
    LLVMInsertExistingBasicBlockAfterInsertBlock.restype = None
    LLVMInsertExistingBasicBlockAfterInsertBlock.argtypes = [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMAppendExistingBasicBlock = _libraries['llvm'].LLVMAppendExistingBasicBlock
    LLVMAppendExistingBasicBlock.restype = None
    LLVMAppendExistingBasicBlock.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMCreateBasicBlockInContext = _libraries['llvm'].LLVMCreateBasicBlockInContext
    LLVMCreateBasicBlockInContext.restype = LLVMBasicBlockRef
    LLVMCreateBasicBlockInContext.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAppendBasicBlockInContext = _libraries['llvm'].LLVMAppendBasicBlockInContext
    LLVMAppendBasicBlockInContext.restype = LLVMBasicBlockRef
    LLVMAppendBasicBlockInContext.argtypes = [LLVMContextRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAppendBasicBlock = _libraries['llvm'].LLVMAppendBasicBlock
    LLVMAppendBasicBlock.restype = LLVMBasicBlockRef
    LLVMAppendBasicBlock.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMInsertBasicBlockInContext = _libraries['llvm'].LLVMInsertBasicBlockInContext
    LLVMInsertBasicBlockInContext.restype = LLVMBasicBlockRef
    LLVMInsertBasicBlockInContext.argtypes = [LLVMContextRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMInsertBasicBlock = _libraries['llvm'].LLVMInsertBasicBlock
    LLVMInsertBasicBlock.restype = LLVMBasicBlockRef
    LLVMInsertBasicBlock.argtypes = [LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMDeleteBasicBlock = _libraries['llvm'].LLVMDeleteBasicBlock
    LLVMDeleteBasicBlock.restype = None
    LLVMDeleteBasicBlock.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMRemoveBasicBlockFromParent = _libraries['llvm'].LLVMRemoveBasicBlockFromParent
    LLVMRemoveBasicBlockFromParent.restype = None
    LLVMRemoveBasicBlockFromParent.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMMoveBasicBlockBefore = _libraries['llvm'].LLVMMoveBasicBlockBefore
    LLVMMoveBasicBlockBefore.restype = None
    LLVMMoveBasicBlockBefore.argtypes = [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMMoveBasicBlockAfter = _libraries['llvm'].LLVMMoveBasicBlockAfter
    LLVMMoveBasicBlockAfter.restype = None
    LLVMMoveBasicBlockAfter.argtypes = [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetFirstInstruction = _libraries['llvm'].LLVMGetFirstInstruction
    LLVMGetFirstInstruction.restype = LLVMValueRef
    LLVMGetFirstInstruction.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetLastInstruction = _libraries['llvm'].LLVMGetLastInstruction
    LLVMGetLastInstruction.restype = LLVMValueRef
    LLVMGetLastInstruction.argtypes = [LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMHasMetadata = _libraries['llvm'].LLVMHasMetadata
    LLVMHasMetadata.restype = ctypes.c_int32
    LLVMHasMetadata.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetMetadata = _libraries['llvm'].LLVMGetMetadata
    LLVMGetMetadata.restype = LLVMValueRef
    LLVMGetMetadata.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMSetMetadata = _libraries['llvm'].LLVMSetMetadata
    LLVMSetMetadata.restype = None
    LLVMSetMetadata.argtypes = [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInstructionGetAllMetadataOtherThanDebugLoc = _libraries['llvm'].LLVMInstructionGetAllMetadataOtherThanDebugLoc
    LLVMInstructionGetAllMetadataOtherThanDebugLoc.restype = ctypes.POINTER(struct_LLVMOpaqueValueMetadataEntry)
    LLVMInstructionGetAllMetadataOtherThanDebugLoc.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMGetInstructionParent = _libraries['llvm'].LLVMGetInstructionParent
    LLVMGetInstructionParent.restype = LLVMBasicBlockRef
    LLVMGetInstructionParent.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNextInstruction = _libraries['llvm'].LLVMGetNextInstruction
    LLVMGetNextInstruction.restype = LLVMValueRef
    LLVMGetNextInstruction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetPreviousInstruction = _libraries['llvm'].LLVMGetPreviousInstruction
    LLVMGetPreviousInstruction.restype = LLVMValueRef
    LLVMGetPreviousInstruction.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInstructionRemoveFromParent = _libraries['llvm'].LLVMInstructionRemoveFromParent
    LLVMInstructionRemoveFromParent.restype = None
    LLVMInstructionRemoveFromParent.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInstructionEraseFromParent = _libraries['llvm'].LLVMInstructionEraseFromParent
    LLVMInstructionEraseFromParent.restype = None
    LLVMInstructionEraseFromParent.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetInstructionOpcode = _libraries['llvm'].LLVMGetInstructionOpcode
    LLVMGetInstructionOpcode.restype = LLVMOpcode
    LLVMGetInstructionOpcode.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetICmpPredicate = _libraries['llvm'].LLVMGetICmpPredicate
    LLVMGetICmpPredicate.restype = LLVMIntPredicate
    LLVMGetICmpPredicate.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetFCmpPredicate = _libraries['llvm'].LLVMGetFCmpPredicate
    LLVMGetFCmpPredicate.restype = LLVMRealPredicate
    LLVMGetFCmpPredicate.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInstructionClone = _libraries['llvm'].LLVMInstructionClone
    LLVMInstructionClone.restype = LLVMValueRef
    LLVMInstructionClone.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsATerminatorInst = _libraries['llvm'].LLVMIsATerminatorInst
    LLVMIsATerminatorInst.restype = LLVMValueRef
    LLVMIsATerminatorInst.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetNumArgOperands = _libraries['llvm'].LLVMGetNumArgOperands
    LLVMGetNumArgOperands.restype = ctypes.c_uint32
    LLVMGetNumArgOperands.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetInstructionCallConv = _libraries['llvm'].LLVMSetInstructionCallConv
    LLVMSetInstructionCallConv.restype = None
    LLVMSetInstructionCallConv.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetInstructionCallConv = _libraries['llvm'].LLVMGetInstructionCallConv
    LLVMGetInstructionCallConv.restype = ctypes.c_uint32
    LLVMGetInstructionCallConv.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetInstrParamAlignment = _libraries['llvm'].LLVMSetInstrParamAlignment
    LLVMSetInstrParamAlignment.restype = None
    LLVMSetInstrParamAlignment.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddCallSiteAttribute = _libraries['llvm'].LLVMAddCallSiteAttribute
    LLVMAddCallSiteAttribute.restype = None
    LLVMAddCallSiteAttribute.argtypes = [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError:
    pass
try:
    LLVMGetCallSiteAttributeCount = _libraries['llvm'].LLVMGetCallSiteAttributeCount
    LLVMGetCallSiteAttributeCount.restype = ctypes.c_uint32
    LLVMGetCallSiteAttributeCount.argtypes = [LLVMValueRef, LLVMAttributeIndex]
except AttributeError:
    pass
try:
    LLVMGetCallSiteAttributes = _libraries['llvm'].LLVMGetCallSiteAttributes
    LLVMGetCallSiteAttributes.restype = None
    LLVMGetCallSiteAttributes.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueAttributeRef))]
except AttributeError:
    pass
try:
    LLVMGetCallSiteEnumAttribute = _libraries['llvm'].LLVMGetCallSiteEnumAttribute
    LLVMGetCallSiteEnumAttribute.restype = LLVMAttributeRef
    LLVMGetCallSiteEnumAttribute.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetCallSiteStringAttribute = _libraries['llvm'].LLVMGetCallSiteStringAttribute
    LLVMGetCallSiteStringAttribute.restype = LLVMAttributeRef
    LLVMGetCallSiteStringAttribute.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMRemoveCallSiteEnumAttribute = _libraries['llvm'].LLVMRemoveCallSiteEnumAttribute
    LLVMRemoveCallSiteEnumAttribute.restype = None
    LLVMRemoveCallSiteEnumAttribute.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMRemoveCallSiteStringAttribute = _libraries['llvm'].LLVMRemoveCallSiteStringAttribute
    LLVMRemoveCallSiteStringAttribute.restype = None
    LLVMRemoveCallSiteStringAttribute.argtypes = [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetCalledFunctionType = _libraries['llvm'].LLVMGetCalledFunctionType
    LLVMGetCalledFunctionType.restype = LLVMTypeRef
    LLVMGetCalledFunctionType.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetCalledValue = _libraries['llvm'].LLVMGetCalledValue
    LLVMGetCalledValue.restype = LLVMValueRef
    LLVMGetCalledValue.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsTailCall = _libraries['llvm'].LLVMIsTailCall
    LLVMIsTailCall.restype = LLVMBool
    LLVMIsTailCall.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetTailCall = _libraries['llvm'].LLVMSetTailCall
    LLVMSetTailCall.restype = None
    LLVMSetTailCall.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetNormalDest = _libraries['llvm'].LLVMGetNormalDest
    LLVMGetNormalDest.restype = LLVMBasicBlockRef
    LLVMGetNormalDest.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetUnwindDest = _libraries['llvm'].LLVMGetUnwindDest
    LLVMGetUnwindDest.restype = LLVMBasicBlockRef
    LLVMGetUnwindDest.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetNormalDest = _libraries['llvm'].LLVMSetNormalDest
    LLVMSetNormalDest.restype = None
    LLVMSetNormalDest.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMSetUnwindDest = _libraries['llvm'].LLVMSetUnwindDest
    LLVMSetUnwindDest.restype = None
    LLVMSetUnwindDest.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetNumSuccessors = _libraries['llvm'].LLVMGetNumSuccessors
    LLVMGetNumSuccessors.restype = ctypes.c_uint32
    LLVMGetNumSuccessors.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetSuccessor = _libraries['llvm'].LLVMGetSuccessor
    LLVMGetSuccessor.restype = LLVMBasicBlockRef
    LLVMGetSuccessor.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMSetSuccessor = _libraries['llvm'].LLVMSetSuccessor
    LLVMSetSuccessor.restype = None
    LLVMSetSuccessor.argtypes = [LLVMValueRef, ctypes.c_uint32, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMIsConditional = _libraries['llvm'].LLVMIsConditional
    LLVMIsConditional.restype = LLVMBool
    LLVMIsConditional.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetCondition = _libraries['llvm'].LLVMGetCondition
    LLVMGetCondition.restype = LLVMValueRef
    LLVMGetCondition.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetCondition = _libraries['llvm'].LLVMSetCondition
    LLVMSetCondition.restype = None
    LLVMSetCondition.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetSwitchDefaultDest = _libraries['llvm'].LLVMGetSwitchDefaultDest
    LLVMGetSwitchDefaultDest.restype = LLVMBasicBlockRef
    LLVMGetSwitchDefaultDest.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetAllocatedType = _libraries['llvm'].LLVMGetAllocatedType
    LLVMGetAllocatedType.restype = LLVMTypeRef
    LLVMGetAllocatedType.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsInBounds = _libraries['llvm'].LLVMIsInBounds
    LLVMIsInBounds.restype = LLVMBool
    LLVMIsInBounds.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetIsInBounds = _libraries['llvm'].LLVMSetIsInBounds
    LLVMSetIsInBounds.restype = None
    LLVMSetIsInBounds.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetGEPSourceElementType = _libraries['llvm'].LLVMGetGEPSourceElementType
    LLVMGetGEPSourceElementType.restype = LLVMTypeRef
    LLVMGetGEPSourceElementType.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAddIncoming = _libraries['llvm'].LLVMAddIncoming
    LLVMAddIncoming.restype = None
    LLVMAddIncoming.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueBasicBlock)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMCountIncoming = _libraries['llvm'].LLVMCountIncoming
    LLVMCountIncoming.restype = ctypes.c_uint32
    LLVMCountIncoming.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetIncomingValue = _libraries['llvm'].LLVMGetIncomingValue
    LLVMGetIncomingValue.restype = LLVMValueRef
    LLVMGetIncomingValue.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetIncomingBlock = _libraries['llvm'].LLVMGetIncomingBlock
    LLVMGetIncomingBlock.restype = LLVMBasicBlockRef
    LLVMGetIncomingBlock.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMGetNumIndices = _libraries['llvm'].LLVMGetNumIndices
    LLVMGetNumIndices.restype = ctypes.c_uint32
    LLVMGetNumIndices.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetIndices = _libraries['llvm'].LLVMGetIndices
    LLVMGetIndices.restype = ctypes.POINTER(ctypes.c_uint32)
    LLVMGetIndices.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMCreateBuilderInContext = _libraries['llvm'].LLVMCreateBuilderInContext
    LLVMCreateBuilderInContext.restype = LLVMBuilderRef
    LLVMCreateBuilderInContext.argtypes = [LLVMContextRef]
except AttributeError:
    pass
try:
    LLVMCreateBuilder = _libraries['llvm'].LLVMCreateBuilder
    LLVMCreateBuilder.restype = LLVMBuilderRef
    LLVMCreateBuilder.argtypes = []
except AttributeError:
    pass
try:
    LLVMPositionBuilder = _libraries['llvm'].LLVMPositionBuilder
    LLVMPositionBuilder.restype = None
    LLVMPositionBuilder.argtypes = [LLVMBuilderRef, LLVMBasicBlockRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMPositionBuilderBefore = _libraries['llvm'].LLVMPositionBuilderBefore
    LLVMPositionBuilderBefore.restype = None
    LLVMPositionBuilderBefore.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMPositionBuilderAtEnd = _libraries['llvm'].LLVMPositionBuilderAtEnd
    LLVMPositionBuilderAtEnd.restype = None
    LLVMPositionBuilderAtEnd.argtypes = [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetInsertBlock = _libraries['llvm'].LLVMGetInsertBlock
    LLVMGetInsertBlock.restype = LLVMBasicBlockRef
    LLVMGetInsertBlock.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMClearInsertionPosition = _libraries['llvm'].LLVMClearInsertionPosition
    LLVMClearInsertionPosition.restype = None
    LLVMClearInsertionPosition.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMInsertIntoBuilder = _libraries['llvm'].LLVMInsertIntoBuilder
    LLVMInsertIntoBuilder.restype = None
    LLVMInsertIntoBuilder.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInsertIntoBuilderWithName = _libraries['llvm'].LLVMInsertIntoBuilderWithName
    LLVMInsertIntoBuilderWithName.restype = None
    LLVMInsertIntoBuilderWithName.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMDisposeBuilder = _libraries['llvm'].LLVMDisposeBuilder
    LLVMDisposeBuilder.restype = None
    LLVMDisposeBuilder.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMGetCurrentDebugLocation2 = _libraries['llvm'].LLVMGetCurrentDebugLocation2
    LLVMGetCurrentDebugLocation2.restype = LLVMMetadataRef
    LLVMGetCurrentDebugLocation2.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMSetCurrentDebugLocation2 = _libraries['llvm'].LLVMSetCurrentDebugLocation2
    LLVMSetCurrentDebugLocation2.restype = None
    LLVMSetCurrentDebugLocation2.argtypes = [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMSetInstDebugLocation = _libraries['llvm'].LLVMSetInstDebugLocation
    LLVMSetInstDebugLocation.restype = None
    LLVMSetInstDebugLocation.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAddMetadataToInst = _libraries['llvm'].LLVMAddMetadataToInst
    LLVMAddMetadataToInst.restype = None
    LLVMAddMetadataToInst.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuilderGetDefaultFPMathTag = _libraries['llvm'].LLVMBuilderGetDefaultFPMathTag
    LLVMBuilderGetDefaultFPMathTag.restype = LLVMMetadataRef
    LLVMBuilderGetDefaultFPMathTag.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMBuilderSetDefaultFPMathTag = _libraries['llvm'].LLVMBuilderSetDefaultFPMathTag
    LLVMBuilderSetDefaultFPMathTag.restype = None
    LLVMBuilderSetDefaultFPMathTag.argtypes = [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMSetCurrentDebugLocation = _libraries['llvm'].LLVMSetCurrentDebugLocation
    LLVMSetCurrentDebugLocation.restype = None
    LLVMSetCurrentDebugLocation.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetCurrentDebugLocation = _libraries['llvm'].LLVMGetCurrentDebugLocation
    LLVMGetCurrentDebugLocation.restype = LLVMValueRef
    LLVMGetCurrentDebugLocation.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMBuildRetVoid = _libraries['llvm'].LLVMBuildRetVoid
    LLVMBuildRetVoid.restype = LLVMValueRef
    LLVMBuildRetVoid.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMBuildRet = _libraries['llvm'].LLVMBuildRet
    LLVMBuildRet.restype = LLVMValueRef
    LLVMBuildRet.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildAggregateRet = _libraries['llvm'].LLVMBuildAggregateRet
    LLVMBuildAggregateRet.restype = LLVMValueRef
    LLVMBuildAggregateRet.argtypes = [LLVMBuilderRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBuildBr = _libraries['llvm'].LLVMBuildBr
    LLVMBuildBr.restype = LLVMValueRef
    LLVMBuildBr.argtypes = [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMBuildCondBr = _libraries['llvm'].LLVMBuildCondBr
    LLVMBuildCondBr.restype = LLVMValueRef
    LLVMBuildCondBr.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMBuildSwitch = _libraries['llvm'].LLVMBuildSwitch
    LLVMBuildSwitch.restype = LLVMValueRef
    LLVMBuildSwitch.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBuildIndirectBr = _libraries['llvm'].LLVMBuildIndirectBr
    LLVMBuildIndirectBr.restype = LLVMValueRef
    LLVMBuildIndirectBr.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBuildInvoke = _libraries['llvm'].LLVMBuildInvoke
    LLVMBuildInvoke.restype = LLVMValueRef
    LLVMBuildInvoke.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildInvoke2 = _libraries['llvm'].LLVMBuildInvoke2
    LLVMBuildInvoke2.restype = LLVMValueRef
    LLVMBuildInvoke2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildUnreachable = _libraries['llvm'].LLVMBuildUnreachable
    LLVMBuildUnreachable.restype = LLVMValueRef
    LLVMBuildUnreachable.argtypes = [LLVMBuilderRef]
except AttributeError:
    pass
try:
    LLVMBuildResume = _libraries['llvm'].LLVMBuildResume
    LLVMBuildResume.restype = LLVMValueRef
    LLVMBuildResume.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildLandingPad = _libraries['llvm'].LLVMBuildLandingPad
    LLVMBuildLandingPad.restype = LLVMValueRef
    LLVMBuildLandingPad.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCleanupRet = _libraries['llvm'].LLVMBuildCleanupRet
    LLVMBuildCleanupRet.restype = LLVMValueRef
    LLVMBuildCleanupRet.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMBuildCatchRet = _libraries['llvm'].LLVMBuildCatchRet
    LLVMBuildCatchRet.restype = LLVMValueRef
    LLVMBuildCatchRet.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMBuildCatchPad = _libraries['llvm'].LLVMBuildCatchPad
    LLVMBuildCatchPad.restype = LLVMValueRef
    LLVMBuildCatchPad.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCleanupPad = _libraries['llvm'].LLVMBuildCleanupPad
    LLVMBuildCleanupPad.restype = LLVMValueRef
    LLVMBuildCleanupPad.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCatchSwitch = _libraries['llvm'].LLVMBuildCatchSwitch
    LLVMBuildCatchSwitch.restype = LLVMValueRef
    LLVMBuildCatchSwitch.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAddCase = _libraries['llvm'].LLVMAddCase
    LLVMAddCase.restype = None
    LLVMAddCase.argtypes = [LLVMValueRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMAddDestination = _libraries['llvm'].LLVMAddDestination
    LLVMAddDestination.restype = None
    LLVMAddDestination.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetNumClauses = _libraries['llvm'].LLVMGetNumClauses
    LLVMGetNumClauses.restype = ctypes.c_uint32
    LLVMGetNumClauses.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetClause = _libraries['llvm'].LLVMGetClause
    LLVMGetClause.restype = LLVMValueRef
    LLVMGetClause.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddClause = _libraries['llvm'].LLVMAddClause
    LLVMAddClause.restype = None
    LLVMAddClause.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMIsCleanup = _libraries['llvm'].LLVMIsCleanup
    LLVMIsCleanup.restype = LLVMBool
    LLVMIsCleanup.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetCleanup = _libraries['llvm'].LLVMSetCleanup
    LLVMSetCleanup.restype = None
    LLVMSetCleanup.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMAddHandler = _libraries['llvm'].LLVMAddHandler
    LLVMAddHandler.restype = None
    LLVMAddHandler.argtypes = [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMGetNumHandlers = _libraries['llvm'].LLVMGetNumHandlers
    LLVMGetNumHandlers.restype = ctypes.c_uint32
    LLVMGetNumHandlers.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetHandlers = _libraries['llvm'].LLVMGetHandlers
    LLVMGetHandlers.restype = None
    LLVMGetHandlers.argtypes = [LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueBasicBlock))]
except AttributeError:
    pass
try:
    LLVMGetArgOperand = _libraries['llvm'].LLVMGetArgOperand
    LLVMGetArgOperand.restype = LLVMValueRef
    LLVMGetArgOperand.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMSetArgOperand = _libraries['llvm'].LLVMSetArgOperand
    LLVMSetArgOperand.restype = None
    LLVMSetArgOperand.argtypes = [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetParentCatchSwitch = _libraries['llvm'].LLVMGetParentCatchSwitch
    LLVMGetParentCatchSwitch.restype = LLVMValueRef
    LLVMGetParentCatchSwitch.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetParentCatchSwitch = _libraries['llvm'].LLVMSetParentCatchSwitch
    LLVMSetParentCatchSwitch.restype = None
    LLVMSetParentCatchSwitch.argtypes = [LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildAdd = _libraries['llvm'].LLVMBuildAdd
    LLVMBuildAdd.restype = LLVMValueRef
    LLVMBuildAdd.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNSWAdd = _libraries['llvm'].LLVMBuildNSWAdd
    LLVMBuildNSWAdd.restype = LLVMValueRef
    LLVMBuildNSWAdd.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNUWAdd = _libraries['llvm'].LLVMBuildNUWAdd
    LLVMBuildNUWAdd.restype = LLVMValueRef
    LLVMBuildNUWAdd.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFAdd = _libraries['llvm'].LLVMBuildFAdd
    LLVMBuildFAdd.restype = LLVMValueRef
    LLVMBuildFAdd.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSub = _libraries['llvm'].LLVMBuildSub
    LLVMBuildSub.restype = LLVMValueRef
    LLVMBuildSub.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNSWSub = _libraries['llvm'].LLVMBuildNSWSub
    LLVMBuildNSWSub.restype = LLVMValueRef
    LLVMBuildNSWSub.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNUWSub = _libraries['llvm'].LLVMBuildNUWSub
    LLVMBuildNUWSub.restype = LLVMValueRef
    LLVMBuildNUWSub.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFSub = _libraries['llvm'].LLVMBuildFSub
    LLVMBuildFSub.restype = LLVMValueRef
    LLVMBuildFSub.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildMul = _libraries['llvm'].LLVMBuildMul
    LLVMBuildMul.restype = LLVMValueRef
    LLVMBuildMul.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNSWMul = _libraries['llvm'].LLVMBuildNSWMul
    LLVMBuildNSWMul.restype = LLVMValueRef
    LLVMBuildNSWMul.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNUWMul = _libraries['llvm'].LLVMBuildNUWMul
    LLVMBuildNUWMul.restype = LLVMValueRef
    LLVMBuildNUWMul.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFMul = _libraries['llvm'].LLVMBuildFMul
    LLVMBuildFMul.restype = LLVMValueRef
    LLVMBuildFMul.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildUDiv = _libraries['llvm'].LLVMBuildUDiv
    LLVMBuildUDiv.restype = LLVMValueRef
    LLVMBuildUDiv.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildExactUDiv = _libraries['llvm'].LLVMBuildExactUDiv
    LLVMBuildExactUDiv.restype = LLVMValueRef
    LLVMBuildExactUDiv.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSDiv = _libraries['llvm'].LLVMBuildSDiv
    LLVMBuildSDiv.restype = LLVMValueRef
    LLVMBuildSDiv.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildExactSDiv = _libraries['llvm'].LLVMBuildExactSDiv
    LLVMBuildExactSDiv.restype = LLVMValueRef
    LLVMBuildExactSDiv.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFDiv = _libraries['llvm'].LLVMBuildFDiv
    LLVMBuildFDiv.restype = LLVMValueRef
    LLVMBuildFDiv.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildURem = _libraries['llvm'].LLVMBuildURem
    LLVMBuildURem.restype = LLVMValueRef
    LLVMBuildURem.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSRem = _libraries['llvm'].LLVMBuildSRem
    LLVMBuildSRem.restype = LLVMValueRef
    LLVMBuildSRem.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFRem = _libraries['llvm'].LLVMBuildFRem
    LLVMBuildFRem.restype = LLVMValueRef
    LLVMBuildFRem.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildShl = _libraries['llvm'].LLVMBuildShl
    LLVMBuildShl.restype = LLVMValueRef
    LLVMBuildShl.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildLShr = _libraries['llvm'].LLVMBuildLShr
    LLVMBuildLShr.restype = LLVMValueRef
    LLVMBuildLShr.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildAShr = _libraries['llvm'].LLVMBuildAShr
    LLVMBuildAShr.restype = LLVMValueRef
    LLVMBuildAShr.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildAnd = _libraries['llvm'].LLVMBuildAnd
    LLVMBuildAnd.restype = LLVMValueRef
    LLVMBuildAnd.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildOr = _libraries['llvm'].LLVMBuildOr
    LLVMBuildOr.restype = LLVMValueRef
    LLVMBuildOr.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildXor = _libraries['llvm'].LLVMBuildXor
    LLVMBuildXor.restype = LLVMValueRef
    LLVMBuildXor.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildBinOp = _libraries['llvm'].LLVMBuildBinOp
    LLVMBuildBinOp.restype = LLVMValueRef
    LLVMBuildBinOp.argtypes = [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNeg = _libraries['llvm'].LLVMBuildNeg
    LLVMBuildNeg.restype = LLVMValueRef
    LLVMBuildNeg.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNSWNeg = _libraries['llvm'].LLVMBuildNSWNeg
    LLVMBuildNSWNeg.restype = LLVMValueRef
    LLVMBuildNSWNeg.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNUWNeg = _libraries['llvm'].LLVMBuildNUWNeg
    LLVMBuildNUWNeg.restype = LLVMValueRef
    LLVMBuildNUWNeg.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFNeg = _libraries['llvm'].LLVMBuildFNeg
    LLVMBuildFNeg.restype = LLVMValueRef
    LLVMBuildFNeg.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildNot = _libraries['llvm'].LLVMBuildNot
    LLVMBuildNot.restype = LLVMValueRef
    LLVMBuildNot.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildMalloc = _libraries['llvm'].LLVMBuildMalloc
    LLVMBuildMalloc.restype = LLVMValueRef
    LLVMBuildMalloc.argtypes = [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildArrayMalloc = _libraries['llvm'].LLVMBuildArrayMalloc
    LLVMBuildArrayMalloc.restype = LLVMValueRef
    LLVMBuildArrayMalloc.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildMemSet = _libraries['llvm'].LLVMBuildMemSet
    LLVMBuildMemSet.restype = LLVMValueRef
    LLVMBuildMemSet.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMBuildMemCpy = _libraries['llvm'].LLVMBuildMemCpy
    LLVMBuildMemCpy.restype = LLVMValueRef
    LLVMBuildMemCpy.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildMemMove = _libraries['llvm'].LLVMBuildMemMove
    LLVMBuildMemMove.restype = LLVMValueRef
    LLVMBuildMemMove.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildAlloca = _libraries['llvm'].LLVMBuildAlloca
    LLVMBuildAlloca.restype = LLVMValueRef
    LLVMBuildAlloca.argtypes = [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildArrayAlloca = _libraries['llvm'].LLVMBuildArrayAlloca
    LLVMBuildArrayAlloca.restype = LLVMValueRef
    LLVMBuildArrayAlloca.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFree = _libraries['llvm'].LLVMBuildFree
    LLVMBuildFree.restype = LLVMValueRef
    LLVMBuildFree.argtypes = [LLVMBuilderRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildLoad = _libraries['llvm'].LLVMBuildLoad
    LLVMBuildLoad.restype = LLVMValueRef
    LLVMBuildLoad.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildLoad2 = _libraries['llvm'].LLVMBuildLoad2
    LLVMBuildLoad2.restype = LLVMValueRef
    LLVMBuildLoad2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildStore = _libraries['llvm'].LLVMBuildStore
    LLVMBuildStore.restype = LLVMValueRef
    LLVMBuildStore.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMBuildGEP = _libraries['llvm'].LLVMBuildGEP
    LLVMBuildGEP.restype = LLVMValueRef
    LLVMBuildGEP.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildInBoundsGEP = _libraries['llvm'].LLVMBuildInBoundsGEP
    LLVMBuildInBoundsGEP.restype = LLVMValueRef
    LLVMBuildInBoundsGEP.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildStructGEP = _libraries['llvm'].LLVMBuildStructGEP
    LLVMBuildStructGEP.restype = LLVMValueRef
    LLVMBuildStructGEP.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildGEP2 = _libraries['llvm'].LLVMBuildGEP2
    LLVMBuildGEP2.restype = LLVMValueRef
    LLVMBuildGEP2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildInBoundsGEP2 = _libraries['llvm'].LLVMBuildInBoundsGEP2
    LLVMBuildInBoundsGEP2.restype = LLVMValueRef
    LLVMBuildInBoundsGEP2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildStructGEP2 = _libraries['llvm'].LLVMBuildStructGEP2
    LLVMBuildStructGEP2.restype = LLVMValueRef
    LLVMBuildStructGEP2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildGlobalString = _libraries['llvm'].LLVMBuildGlobalString
    LLVMBuildGlobalString.restype = LLVMValueRef
    LLVMBuildGlobalString.argtypes = [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildGlobalStringPtr = _libraries['llvm'].LLVMBuildGlobalStringPtr
    LLVMBuildGlobalStringPtr.restype = LLVMValueRef
    LLVMBuildGlobalStringPtr.argtypes = [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetVolatile = _libraries['llvm'].LLVMGetVolatile
    LLVMGetVolatile.restype = LLVMBool
    LLVMGetVolatile.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetVolatile = _libraries['llvm'].LLVMSetVolatile
    LLVMSetVolatile.restype = None
    LLVMSetVolatile.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetWeak = _libraries['llvm'].LLVMGetWeak
    LLVMGetWeak.restype = LLVMBool
    LLVMGetWeak.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetWeak = _libraries['llvm'].LLVMSetWeak
    LLVMSetWeak.restype = None
    LLVMSetWeak.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetOrdering = _libraries['llvm'].LLVMGetOrdering
    LLVMGetOrdering.restype = LLVMAtomicOrdering
    LLVMGetOrdering.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetOrdering = _libraries['llvm'].LLVMSetOrdering
    LLVMSetOrdering.restype = None
    LLVMSetOrdering.argtypes = [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError:
    pass
try:
    LLVMGetAtomicRMWBinOp = _libraries['llvm'].LLVMGetAtomicRMWBinOp
    LLVMGetAtomicRMWBinOp.restype = LLVMAtomicRMWBinOp
    LLVMGetAtomicRMWBinOp.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetAtomicRMWBinOp = _libraries['llvm'].LLVMSetAtomicRMWBinOp
    LLVMSetAtomicRMWBinOp.restype = None
    LLVMSetAtomicRMWBinOp.argtypes = [LLVMValueRef, LLVMAtomicRMWBinOp]
except AttributeError:
    pass
try:
    LLVMBuildTrunc = _libraries['llvm'].LLVMBuildTrunc
    LLVMBuildTrunc.restype = LLVMValueRef
    LLVMBuildTrunc.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildZExt = _libraries['llvm'].LLVMBuildZExt
    LLVMBuildZExt.restype = LLVMValueRef
    LLVMBuildZExt.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSExt = _libraries['llvm'].LLVMBuildSExt
    LLVMBuildSExt.restype = LLVMValueRef
    LLVMBuildSExt.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFPToUI = _libraries['llvm'].LLVMBuildFPToUI
    LLVMBuildFPToUI.restype = LLVMValueRef
    LLVMBuildFPToUI.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFPToSI = _libraries['llvm'].LLVMBuildFPToSI
    LLVMBuildFPToSI.restype = LLVMValueRef
    LLVMBuildFPToSI.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildUIToFP = _libraries['llvm'].LLVMBuildUIToFP
    LLVMBuildUIToFP.restype = LLVMValueRef
    LLVMBuildUIToFP.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSIToFP = _libraries['llvm'].LLVMBuildSIToFP
    LLVMBuildSIToFP.restype = LLVMValueRef
    LLVMBuildSIToFP.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFPTrunc = _libraries['llvm'].LLVMBuildFPTrunc
    LLVMBuildFPTrunc.restype = LLVMValueRef
    LLVMBuildFPTrunc.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFPExt = _libraries['llvm'].LLVMBuildFPExt
    LLVMBuildFPExt.restype = LLVMValueRef
    LLVMBuildFPExt.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildPtrToInt = _libraries['llvm'].LLVMBuildPtrToInt
    LLVMBuildPtrToInt.restype = LLVMValueRef
    LLVMBuildPtrToInt.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildIntToPtr = _libraries['llvm'].LLVMBuildIntToPtr
    LLVMBuildIntToPtr.restype = LLVMValueRef
    LLVMBuildIntToPtr.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildBitCast = _libraries['llvm'].LLVMBuildBitCast
    LLVMBuildBitCast.restype = LLVMValueRef
    LLVMBuildBitCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildAddrSpaceCast = _libraries['llvm'].LLVMBuildAddrSpaceCast
    LLVMBuildAddrSpaceCast.restype = LLVMValueRef
    LLVMBuildAddrSpaceCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildZExtOrBitCast = _libraries['llvm'].LLVMBuildZExtOrBitCast
    LLVMBuildZExtOrBitCast.restype = LLVMValueRef
    LLVMBuildZExtOrBitCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSExtOrBitCast = _libraries['llvm'].LLVMBuildSExtOrBitCast
    LLVMBuildSExtOrBitCast.restype = LLVMValueRef
    LLVMBuildSExtOrBitCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildTruncOrBitCast = _libraries['llvm'].LLVMBuildTruncOrBitCast
    LLVMBuildTruncOrBitCast.restype = LLVMValueRef
    LLVMBuildTruncOrBitCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCast = _libraries['llvm'].LLVMBuildCast
    LLVMBuildCast.restype = LLVMValueRef
    LLVMBuildCast.argtypes = [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildPointerCast = _libraries['llvm'].LLVMBuildPointerCast
    LLVMBuildPointerCast.restype = LLVMValueRef
    LLVMBuildPointerCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildIntCast2 = _libraries['llvm'].LLVMBuildIntCast2
    LLVMBuildIntCast2.restype = LLVMValueRef
    LLVMBuildIntCast2.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFPCast = _libraries['llvm'].LLVMBuildFPCast
    LLVMBuildFPCast.restype = LLVMValueRef
    LLVMBuildFPCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildIntCast = _libraries['llvm'].LLVMBuildIntCast
    LLVMBuildIntCast.restype = LLVMValueRef
    LLVMBuildIntCast.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildICmp = _libraries['llvm'].LLVMBuildICmp
    LLVMBuildICmp.restype = LLVMValueRef
    LLVMBuildICmp.argtypes = [LLVMBuilderRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFCmp = _libraries['llvm'].LLVMBuildFCmp
    LLVMBuildFCmp.restype = LLVMValueRef
    LLVMBuildFCmp.argtypes = [LLVMBuilderRef, LLVMRealPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildPhi = _libraries['llvm'].LLVMBuildPhi
    LLVMBuildPhi.restype = LLVMValueRef
    LLVMBuildPhi.argtypes = [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCall = _libraries['llvm'].LLVMBuildCall
    LLVMBuildCall.restype = LLVMValueRef
    LLVMBuildCall.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildCall2 = _libraries['llvm'].LLVMBuildCall2
    LLVMBuildCall2.restype = LLVMValueRef
    LLVMBuildCall2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildSelect = _libraries['llvm'].LLVMBuildSelect
    LLVMBuildSelect.restype = LLVMValueRef
    LLVMBuildSelect.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildVAArg = _libraries['llvm'].LLVMBuildVAArg
    LLVMBuildVAArg.restype = LLVMValueRef
    LLVMBuildVAArg.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildExtractElement = _libraries['llvm'].LLVMBuildExtractElement
    LLVMBuildExtractElement.restype = LLVMValueRef
    LLVMBuildExtractElement.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildInsertElement = _libraries['llvm'].LLVMBuildInsertElement
    LLVMBuildInsertElement.restype = LLVMValueRef
    LLVMBuildInsertElement.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildShuffleVector = _libraries['llvm'].LLVMBuildShuffleVector
    LLVMBuildShuffleVector.restype = LLVMValueRef
    LLVMBuildShuffleVector.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildExtractValue = _libraries['llvm'].LLVMBuildExtractValue
    LLVMBuildExtractValue.restype = LLVMValueRef
    LLVMBuildExtractValue.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildInsertValue = _libraries['llvm'].LLVMBuildInsertValue
    LLVMBuildInsertValue.restype = LLVMValueRef
    LLVMBuildInsertValue.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFreeze = _libraries['llvm'].LLVMBuildFreeze
    LLVMBuildFreeze.restype = LLVMValueRef
    LLVMBuildFreeze.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildIsNull = _libraries['llvm'].LLVMBuildIsNull
    LLVMBuildIsNull.restype = LLVMValueRef
    LLVMBuildIsNull.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildIsNotNull = _libraries['llvm'].LLVMBuildIsNotNull
    LLVMBuildIsNotNull.restype = LLVMValueRef
    LLVMBuildIsNotNull.argtypes = [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildPtrDiff = _libraries['llvm'].LLVMBuildPtrDiff
    LLVMBuildPtrDiff.restype = LLVMValueRef
    LLVMBuildPtrDiff.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildPtrDiff2 = _libraries['llvm'].LLVMBuildPtrDiff2
    LLVMBuildPtrDiff2.restype = LLVMValueRef
    LLVMBuildPtrDiff2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildFence = _libraries['llvm'].LLVMBuildFence
    LLVMBuildFence.restype = LLVMValueRef
    LLVMBuildFence.argtypes = [LLVMBuilderRef, LLVMAtomicOrdering, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMBuildAtomicRMW = _libraries['llvm'].LLVMBuildAtomicRMW
    LLVMBuildAtomicRMW.restype = LLVMValueRef
    LLVMBuildAtomicRMW.argtypes = [LLVMBuilderRef, LLVMAtomicRMWBinOp, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMBool]
except AttributeError:
    pass
try:
    LLVMBuildAtomicCmpXchg = _libraries['llvm'].LLVMBuildAtomicCmpXchg
    LLVMBuildAtomicCmpXchg.restype = LLVMValueRef
    LLVMBuildAtomicCmpXchg.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetNumMaskElements = _libraries['llvm'].LLVMGetNumMaskElements
    LLVMGetNumMaskElements.restype = ctypes.c_uint32
    LLVMGetNumMaskElements.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetUndefMaskElem = _libraries['llvm'].LLVMGetUndefMaskElem
    LLVMGetUndefMaskElem.restype = ctypes.c_int32
    LLVMGetUndefMaskElem.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetMaskValue = _libraries['llvm'].LLVMGetMaskValue
    LLVMGetMaskValue.restype = ctypes.c_int32
    LLVMGetMaskValue.argtypes = [LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMIsAtomicSingleThread = _libraries['llvm'].LLVMIsAtomicSingleThread
    LLVMIsAtomicSingleThread.restype = LLVMBool
    LLVMIsAtomicSingleThread.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetAtomicSingleThread = _libraries['llvm'].LLVMSetAtomicSingleThread
    LLVMSetAtomicSingleThread.restype = None
    LLVMSetAtomicSingleThread.argtypes = [LLVMValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGetCmpXchgSuccessOrdering = _libraries['llvm'].LLVMGetCmpXchgSuccessOrdering
    LLVMGetCmpXchgSuccessOrdering.restype = LLVMAtomicOrdering
    LLVMGetCmpXchgSuccessOrdering.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetCmpXchgSuccessOrdering = _libraries['llvm'].LLVMSetCmpXchgSuccessOrdering
    LLVMSetCmpXchgSuccessOrdering.restype = None
    LLVMSetCmpXchgSuccessOrdering.argtypes = [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError:
    pass
try:
    LLVMGetCmpXchgFailureOrdering = _libraries['llvm'].LLVMGetCmpXchgFailureOrdering
    LLVMGetCmpXchgFailureOrdering.restype = LLVMAtomicOrdering
    LLVMGetCmpXchgFailureOrdering.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetCmpXchgFailureOrdering = _libraries['llvm'].LLVMSetCmpXchgFailureOrdering
    LLVMSetCmpXchgFailureOrdering.restype = None
    LLVMSetCmpXchgFailureOrdering.argtypes = [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError:
    pass
try:
    LLVMCreateModuleProviderForExistingModule = _libraries['llvm'].LLVMCreateModuleProviderForExistingModule
    LLVMCreateModuleProviderForExistingModule.restype = LLVMModuleProviderRef
    LLVMCreateModuleProviderForExistingModule.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMDisposeModuleProvider = _libraries['llvm'].LLVMDisposeModuleProvider
    LLVMDisposeModuleProvider.restype = None
    LLVMDisposeModuleProvider.argtypes = [LLVMModuleProviderRef]
except AttributeError:
    pass
try:
    LLVMCreateMemoryBufferWithContentsOfFile = _libraries['llvm'].LLVMCreateMemoryBufferWithContentsOfFile
    LLVMCreateMemoryBufferWithContentsOfFile.restype = LLVMBool
    LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMCreateMemoryBufferWithSTDIN = _libraries['llvm'].LLVMCreateMemoryBufferWithSTDIN
    LLVMCreateMemoryBufferWithSTDIN.restype = LLVMBool
    LLVMCreateMemoryBufferWithSTDIN.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMCreateMemoryBufferWithMemoryRange = _libraries['llvm'].LLVMCreateMemoryBufferWithMemoryRange
    LLVMCreateMemoryBufferWithMemoryRange.restype = LLVMMemoryBufferRef
    LLVMCreateMemoryBufferWithMemoryRange.argtypes = [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), LLVMBool]
except AttributeError:
    pass
try:
    LLVMCreateMemoryBufferWithMemoryRangeCopy = _libraries['llvm'].LLVMCreateMemoryBufferWithMemoryRangeCopy
    LLVMCreateMemoryBufferWithMemoryRangeCopy.restype = LLVMMemoryBufferRef
    LLVMCreateMemoryBufferWithMemoryRangeCopy.argtypes = [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetBufferStart = _libraries['llvm'].LLVMGetBufferStart
    LLVMGetBufferStart.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetBufferStart.argtypes = [LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMGetBufferSize = _libraries['llvm'].LLVMGetBufferSize
    LLVMGetBufferSize.restype = size_t
    LLVMGetBufferSize.argtypes = [LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMDisposeMemoryBuffer = _libraries['llvm'].LLVMDisposeMemoryBuffer
    LLVMDisposeMemoryBuffer.restype = None
    LLVMDisposeMemoryBuffer.argtypes = [LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMGetGlobalPassRegistry = _libraries['llvm'].LLVMGetGlobalPassRegistry
    LLVMGetGlobalPassRegistry.restype = LLVMPassRegistryRef
    LLVMGetGlobalPassRegistry.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreatePassManager = _libraries['llvm'].LLVMCreatePassManager
    LLVMCreatePassManager.restype = LLVMPassManagerRef
    LLVMCreatePassManager.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreateFunctionPassManagerForModule = _libraries['llvm'].LLVMCreateFunctionPassManagerForModule
    LLVMCreateFunctionPassManagerForModule.restype = LLVMPassManagerRef
    LLVMCreateFunctionPassManagerForModule.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMCreateFunctionPassManager = _libraries['llvm'].LLVMCreateFunctionPassManager
    LLVMCreateFunctionPassManager.restype = LLVMPassManagerRef
    LLVMCreateFunctionPassManager.argtypes = [LLVMModuleProviderRef]
except AttributeError:
    pass
try:
    LLVMRunPassManager = _libraries['llvm'].LLVMRunPassManager
    LLVMRunPassManager.restype = LLVMBool
    LLVMRunPassManager.argtypes = [LLVMPassManagerRef, LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMInitializeFunctionPassManager = _libraries['llvm'].LLVMInitializeFunctionPassManager
    LLVMInitializeFunctionPassManager.restype = LLVMBool
    LLVMInitializeFunctionPassManager.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMRunFunctionPassManager = _libraries['llvm'].LLVMRunFunctionPassManager
    LLVMRunFunctionPassManager.restype = LLVMBool
    LLVMRunFunctionPassManager.argtypes = [LLVMPassManagerRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMFinalizeFunctionPassManager = _libraries['llvm'].LLVMFinalizeFunctionPassManager
    LLVMFinalizeFunctionPassManager.restype = LLVMBool
    LLVMFinalizeFunctionPassManager.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMDisposePassManager = _libraries['llvm'].LLVMDisposePassManager
    LLVMDisposePassManager.restype = None
    LLVMDisposePassManager.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMStartMultithreaded = _libraries['llvm'].LLVMStartMultithreaded
    LLVMStartMultithreaded.restype = LLVMBool
    LLVMStartMultithreaded.argtypes = []
except AttributeError:
    pass
try:
    LLVMStopMultithreaded = _libraries['llvm'].LLVMStopMultithreaded
    LLVMStopMultithreaded.restype = None
    LLVMStopMultithreaded.argtypes = []
except AttributeError:
    pass
try:
    LLVMIsMultithreaded = _libraries['llvm'].LLVMIsMultithreaded
    LLVMIsMultithreaded.restype = LLVMBool
    LLVMIsMultithreaded.argtypes = []
except AttributeError:
    pass
LLVM_C_DEBUGINFO_H = True # macro

# values for enumeration 'c__EA_LLVMDIFlags'
c__EA_LLVMDIFlags__enumvalues = {
    0: 'LLVMDIFlagZero',
    1: 'LLVMDIFlagPrivate',
    2: 'LLVMDIFlagProtected',
    3: 'LLVMDIFlagPublic',
    4: 'LLVMDIFlagFwdDecl',
    8: 'LLVMDIFlagAppleBlock',
    16: 'LLVMDIFlagReservedBit4',
    32: 'LLVMDIFlagVirtual',
    64: 'LLVMDIFlagArtificial',
    128: 'LLVMDIFlagExplicit',
    256: 'LLVMDIFlagPrototyped',
    512: 'LLVMDIFlagObjcClassComplete',
    1024: 'LLVMDIFlagObjectPointer',
    2048: 'LLVMDIFlagVector',
    4096: 'LLVMDIFlagStaticMember',
    8192: 'LLVMDIFlagLValueReference',
    16384: 'LLVMDIFlagRValueReference',
    32768: 'LLVMDIFlagReserved',
    65536: 'LLVMDIFlagSingleInheritance',
    131072: 'LLVMDIFlagMultipleInheritance',
    196608: 'LLVMDIFlagVirtualInheritance',
    262144: 'LLVMDIFlagIntroducedVirtual',
    524288: 'LLVMDIFlagBitField',
    1048576: 'LLVMDIFlagNoReturn',
    4194304: 'LLVMDIFlagTypePassByValue',
    8388608: 'LLVMDIFlagTypePassByReference',
    16777216: 'LLVMDIFlagEnumClass',
    16777216: 'LLVMDIFlagFixedEnum',
    33554432: 'LLVMDIFlagThunk',
    67108864: 'LLVMDIFlagNonTrivial',
    134217728: 'LLVMDIFlagBigEndian',
    268435456: 'LLVMDIFlagLittleEndian',
    36: 'LLVMDIFlagIndirectVirtualBase',
    3: 'LLVMDIFlagAccessibility',
    196608: 'LLVMDIFlagPtrToMemberRep',
}
LLVMDIFlagZero = 0
LLVMDIFlagPrivate = 1
LLVMDIFlagProtected = 2
LLVMDIFlagPublic = 3
LLVMDIFlagFwdDecl = 4
LLVMDIFlagAppleBlock = 8
LLVMDIFlagReservedBit4 = 16
LLVMDIFlagVirtual = 32
LLVMDIFlagArtificial = 64
LLVMDIFlagExplicit = 128
LLVMDIFlagPrototyped = 256
LLVMDIFlagObjcClassComplete = 512
LLVMDIFlagObjectPointer = 1024
LLVMDIFlagVector = 2048
LLVMDIFlagStaticMember = 4096
LLVMDIFlagLValueReference = 8192
LLVMDIFlagRValueReference = 16384
LLVMDIFlagReserved = 32768
LLVMDIFlagSingleInheritance = 65536
LLVMDIFlagMultipleInheritance = 131072
LLVMDIFlagVirtualInheritance = 196608
LLVMDIFlagIntroducedVirtual = 262144
LLVMDIFlagBitField = 524288
LLVMDIFlagNoReturn = 1048576
LLVMDIFlagTypePassByValue = 4194304
LLVMDIFlagTypePassByReference = 8388608
LLVMDIFlagEnumClass = 16777216
LLVMDIFlagFixedEnum = 16777216
LLVMDIFlagThunk = 33554432
LLVMDIFlagNonTrivial = 67108864
LLVMDIFlagBigEndian = 134217728
LLVMDIFlagLittleEndian = 268435456
LLVMDIFlagIndirectVirtualBase = 36
LLVMDIFlagAccessibility = 3
LLVMDIFlagPtrToMemberRep = 196608
c__EA_LLVMDIFlags = ctypes.c_uint32 # enum
LLVMDIFlags = c__EA_LLVMDIFlags
LLVMDIFlags__enumvalues = c__EA_LLVMDIFlags__enumvalues

# values for enumeration 'c__EA_LLVMDWARFSourceLanguage'
c__EA_LLVMDWARFSourceLanguage__enumvalues = {
    0: 'LLVMDWARFSourceLanguageC89',
    1: 'LLVMDWARFSourceLanguageC',
    2: 'LLVMDWARFSourceLanguageAda83',
    3: 'LLVMDWARFSourceLanguageC_plus_plus',
    4: 'LLVMDWARFSourceLanguageCobol74',
    5: 'LLVMDWARFSourceLanguageCobol85',
    6: 'LLVMDWARFSourceLanguageFortran77',
    7: 'LLVMDWARFSourceLanguageFortran90',
    8: 'LLVMDWARFSourceLanguagePascal83',
    9: 'LLVMDWARFSourceLanguageModula2',
    10: 'LLVMDWARFSourceLanguageJava',
    11: 'LLVMDWARFSourceLanguageC99',
    12: 'LLVMDWARFSourceLanguageAda95',
    13: 'LLVMDWARFSourceLanguageFortran95',
    14: 'LLVMDWARFSourceLanguagePLI',
    15: 'LLVMDWARFSourceLanguageObjC',
    16: 'LLVMDWARFSourceLanguageObjC_plus_plus',
    17: 'LLVMDWARFSourceLanguageUPC',
    18: 'LLVMDWARFSourceLanguageD',
    19: 'LLVMDWARFSourceLanguagePython',
    20: 'LLVMDWARFSourceLanguageOpenCL',
    21: 'LLVMDWARFSourceLanguageGo',
    22: 'LLVMDWARFSourceLanguageModula3',
    23: 'LLVMDWARFSourceLanguageHaskell',
    24: 'LLVMDWARFSourceLanguageC_plus_plus_03',
    25: 'LLVMDWARFSourceLanguageC_plus_plus_11',
    26: 'LLVMDWARFSourceLanguageOCaml',
    27: 'LLVMDWARFSourceLanguageRust',
    28: 'LLVMDWARFSourceLanguageC11',
    29: 'LLVMDWARFSourceLanguageSwift',
    30: 'LLVMDWARFSourceLanguageJulia',
    31: 'LLVMDWARFSourceLanguageDylan',
    32: 'LLVMDWARFSourceLanguageC_plus_plus_14',
    33: 'LLVMDWARFSourceLanguageFortran03',
    34: 'LLVMDWARFSourceLanguageFortran08',
    35: 'LLVMDWARFSourceLanguageRenderScript',
    36: 'LLVMDWARFSourceLanguageBLISS',
    37: 'LLVMDWARFSourceLanguageMips_Assembler',
    38: 'LLVMDWARFSourceLanguageGOOGLE_RenderScript',
    39: 'LLVMDWARFSourceLanguageBORLAND_Delphi',
}
LLVMDWARFSourceLanguageC89 = 0
LLVMDWARFSourceLanguageC = 1
LLVMDWARFSourceLanguageAda83 = 2
LLVMDWARFSourceLanguageC_plus_plus = 3
LLVMDWARFSourceLanguageCobol74 = 4
LLVMDWARFSourceLanguageCobol85 = 5
LLVMDWARFSourceLanguageFortran77 = 6
LLVMDWARFSourceLanguageFortran90 = 7
LLVMDWARFSourceLanguagePascal83 = 8
LLVMDWARFSourceLanguageModula2 = 9
LLVMDWARFSourceLanguageJava = 10
LLVMDWARFSourceLanguageC99 = 11
LLVMDWARFSourceLanguageAda95 = 12
LLVMDWARFSourceLanguageFortran95 = 13
LLVMDWARFSourceLanguagePLI = 14
LLVMDWARFSourceLanguageObjC = 15
LLVMDWARFSourceLanguageObjC_plus_plus = 16
LLVMDWARFSourceLanguageUPC = 17
LLVMDWARFSourceLanguageD = 18
LLVMDWARFSourceLanguagePython = 19
LLVMDWARFSourceLanguageOpenCL = 20
LLVMDWARFSourceLanguageGo = 21
LLVMDWARFSourceLanguageModula3 = 22
LLVMDWARFSourceLanguageHaskell = 23
LLVMDWARFSourceLanguageC_plus_plus_03 = 24
LLVMDWARFSourceLanguageC_plus_plus_11 = 25
LLVMDWARFSourceLanguageOCaml = 26
LLVMDWARFSourceLanguageRust = 27
LLVMDWARFSourceLanguageC11 = 28
LLVMDWARFSourceLanguageSwift = 29
LLVMDWARFSourceLanguageJulia = 30
LLVMDWARFSourceLanguageDylan = 31
LLVMDWARFSourceLanguageC_plus_plus_14 = 32
LLVMDWARFSourceLanguageFortran03 = 33
LLVMDWARFSourceLanguageFortran08 = 34
LLVMDWARFSourceLanguageRenderScript = 35
LLVMDWARFSourceLanguageBLISS = 36
LLVMDWARFSourceLanguageMips_Assembler = 37
LLVMDWARFSourceLanguageGOOGLE_RenderScript = 38
LLVMDWARFSourceLanguageBORLAND_Delphi = 39
c__EA_LLVMDWARFSourceLanguage = ctypes.c_uint32 # enum
LLVMDWARFSourceLanguage = c__EA_LLVMDWARFSourceLanguage
LLVMDWARFSourceLanguage__enumvalues = c__EA_LLVMDWARFSourceLanguage__enumvalues

# values for enumeration 'c__EA_LLVMDWARFEmissionKind'
c__EA_LLVMDWARFEmissionKind__enumvalues = {
    0: 'LLVMDWARFEmissionNone',
    1: 'LLVMDWARFEmissionFull',
    2: 'LLVMDWARFEmissionLineTablesOnly',
}
LLVMDWARFEmissionNone = 0
LLVMDWARFEmissionFull = 1
LLVMDWARFEmissionLineTablesOnly = 2
c__EA_LLVMDWARFEmissionKind = ctypes.c_uint32 # enum
LLVMDWARFEmissionKind = c__EA_LLVMDWARFEmissionKind
LLVMDWARFEmissionKind__enumvalues = c__EA_LLVMDWARFEmissionKind__enumvalues

# values for enumeration 'c__Ea_LLVMMDStringMetadataKind'
c__Ea_LLVMMDStringMetadataKind__enumvalues = {
    0: 'LLVMMDStringMetadataKind',
    1: 'LLVMConstantAsMetadataMetadataKind',
    2: 'LLVMLocalAsMetadataMetadataKind',
    3: 'LLVMDistinctMDOperandPlaceholderMetadataKind',
    4: 'LLVMMDTupleMetadataKind',
    5: 'LLVMDILocationMetadataKind',
    6: 'LLVMDIExpressionMetadataKind',
    7: 'LLVMDIGlobalVariableExpressionMetadataKind',
    8: 'LLVMGenericDINodeMetadataKind',
    9: 'LLVMDISubrangeMetadataKind',
    10: 'LLVMDIEnumeratorMetadataKind',
    11: 'LLVMDIBasicTypeMetadataKind',
    12: 'LLVMDIDerivedTypeMetadataKind',
    13: 'LLVMDICompositeTypeMetadataKind',
    14: 'LLVMDISubroutineTypeMetadataKind',
    15: 'LLVMDIFileMetadataKind',
    16: 'LLVMDICompileUnitMetadataKind',
    17: 'LLVMDISubprogramMetadataKind',
    18: 'LLVMDILexicalBlockMetadataKind',
    19: 'LLVMDILexicalBlockFileMetadataKind',
    20: 'LLVMDINamespaceMetadataKind',
    21: 'LLVMDIModuleMetadataKind',
    22: 'LLVMDITemplateTypeParameterMetadataKind',
    23: 'LLVMDITemplateValueParameterMetadataKind',
    24: 'LLVMDIGlobalVariableMetadataKind',
    25: 'LLVMDILocalVariableMetadataKind',
    26: 'LLVMDILabelMetadataKind',
    27: 'LLVMDIObjCPropertyMetadataKind',
    28: 'LLVMDIImportedEntityMetadataKind',
    29: 'LLVMDIMacroMetadataKind',
    30: 'LLVMDIMacroFileMetadataKind',
    31: 'LLVMDICommonBlockMetadataKind',
    32: 'LLVMDIStringTypeMetadataKind',
    33: 'LLVMDIGenericSubrangeMetadataKind',
    34: 'LLVMDIArgListMetadataKind',
}
LLVMMDStringMetadataKind = 0
LLVMConstantAsMetadataMetadataKind = 1
LLVMLocalAsMetadataMetadataKind = 2
LLVMDistinctMDOperandPlaceholderMetadataKind = 3
LLVMMDTupleMetadataKind = 4
LLVMDILocationMetadataKind = 5
LLVMDIExpressionMetadataKind = 6
LLVMDIGlobalVariableExpressionMetadataKind = 7
LLVMGenericDINodeMetadataKind = 8
LLVMDISubrangeMetadataKind = 9
LLVMDIEnumeratorMetadataKind = 10
LLVMDIBasicTypeMetadataKind = 11
LLVMDIDerivedTypeMetadataKind = 12
LLVMDICompositeTypeMetadataKind = 13
LLVMDISubroutineTypeMetadataKind = 14
LLVMDIFileMetadataKind = 15
LLVMDICompileUnitMetadataKind = 16
LLVMDISubprogramMetadataKind = 17
LLVMDILexicalBlockMetadataKind = 18
LLVMDILexicalBlockFileMetadataKind = 19
LLVMDINamespaceMetadataKind = 20
LLVMDIModuleMetadataKind = 21
LLVMDITemplateTypeParameterMetadataKind = 22
LLVMDITemplateValueParameterMetadataKind = 23
LLVMDIGlobalVariableMetadataKind = 24
LLVMDILocalVariableMetadataKind = 25
LLVMDILabelMetadataKind = 26
LLVMDIObjCPropertyMetadataKind = 27
LLVMDIImportedEntityMetadataKind = 28
LLVMDIMacroMetadataKind = 29
LLVMDIMacroFileMetadataKind = 30
LLVMDICommonBlockMetadataKind = 31
LLVMDIStringTypeMetadataKind = 32
LLVMDIGenericSubrangeMetadataKind = 33
LLVMDIArgListMetadataKind = 34
c__Ea_LLVMMDStringMetadataKind = ctypes.c_uint32 # enum
LLVMMetadataKind = ctypes.c_uint32
LLVMDWARFTypeEncoding = ctypes.c_uint32

# values for enumeration 'c__EA_LLVMDWARFMacinfoRecordType'
c__EA_LLVMDWARFMacinfoRecordType__enumvalues = {
    1: 'LLVMDWARFMacinfoRecordTypeDefine',
    2: 'LLVMDWARFMacinfoRecordTypeMacro',
    3: 'LLVMDWARFMacinfoRecordTypeStartFile',
    4: 'LLVMDWARFMacinfoRecordTypeEndFile',
    255: 'LLVMDWARFMacinfoRecordTypeVendorExt',
}
LLVMDWARFMacinfoRecordTypeDefine = 1
LLVMDWARFMacinfoRecordTypeMacro = 2
LLVMDWARFMacinfoRecordTypeStartFile = 3
LLVMDWARFMacinfoRecordTypeEndFile = 4
LLVMDWARFMacinfoRecordTypeVendorExt = 255
c__EA_LLVMDWARFMacinfoRecordType = ctypes.c_uint32 # enum
LLVMDWARFMacinfoRecordType = c__EA_LLVMDWARFMacinfoRecordType
LLVMDWARFMacinfoRecordType__enumvalues = c__EA_LLVMDWARFMacinfoRecordType__enumvalues
try:
    LLVMDebugMetadataVersion = _libraries['llvm'].LLVMDebugMetadataVersion
    LLVMDebugMetadataVersion.restype = ctypes.c_uint32
    LLVMDebugMetadataVersion.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetModuleDebugMetadataVersion = _libraries['llvm'].LLVMGetModuleDebugMetadataVersion
    LLVMGetModuleDebugMetadataVersion.restype = ctypes.c_uint32
    LLVMGetModuleDebugMetadataVersion.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMStripModuleDebugInfo = _libraries['llvm'].LLVMStripModuleDebugInfo
    LLVMStripModuleDebugInfo.restype = LLVMBool
    LLVMStripModuleDebugInfo.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMCreateDIBuilderDisallowUnresolved = _libraries['llvm'].LLVMCreateDIBuilderDisallowUnresolved
    LLVMCreateDIBuilderDisallowUnresolved.restype = LLVMDIBuilderRef
    LLVMCreateDIBuilderDisallowUnresolved.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMCreateDIBuilder = _libraries['llvm'].LLVMCreateDIBuilder
    LLVMCreateDIBuilder.restype = LLVMDIBuilderRef
    LLVMCreateDIBuilder.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMDisposeDIBuilder = _libraries['llvm'].LLVMDisposeDIBuilder
    LLVMDisposeDIBuilder.restype = None
    LLVMDisposeDIBuilder.argtypes = [LLVMDIBuilderRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderFinalize = _libraries['llvm'].LLVMDIBuilderFinalize
    LLVMDIBuilderFinalize.restype = None
    LLVMDIBuilderFinalize.argtypes = [LLVMDIBuilderRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderFinalizeSubprogram = _libraries['llvm'].LLVMDIBuilderFinalizeSubprogram
    LLVMDIBuilderFinalizeSubprogram.restype = None
    LLVMDIBuilderFinalizeSubprogram.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateCompileUnit = _libraries['llvm'].LLVMDIBuilderCreateCompileUnit
    LLVMDIBuilderCreateCompileUnit.restype = LLVMMetadataRef
    LLVMDIBuilderCreateCompileUnit.argtypes = [LLVMDIBuilderRef, LLVMDWARFSourceLanguage, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMDWARFEmissionKind, ctypes.c_uint32, LLVMBool, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateFile = _libraries['llvm'].LLVMDIBuilderCreateFile
    LLVMDIBuilderCreateFile.restype = LLVMMetadataRef
    LLVMDIBuilderCreateFile.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateModule = _libraries['llvm'].LLVMDIBuilderCreateModule
    LLVMDIBuilderCreateModule.restype = LLVMMetadataRef
    LLVMDIBuilderCreateModule.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateNameSpace = _libraries['llvm'].LLVMDIBuilderCreateNameSpace
    LLVMDIBuilderCreateNameSpace.restype = LLVMMetadataRef
    LLVMDIBuilderCreateNameSpace.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateFunction = _libraries['llvm'].LLVMDIBuilderCreateFunction
    LLVMDIBuilderCreateFunction.restype = LLVMMetadataRef
    LLVMDIBuilderCreateFunction.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMBool, ctypes.c_uint32, LLVMDIFlags, LLVMBool]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateLexicalBlock = _libraries['llvm'].LLVMDIBuilderCreateLexicalBlock
    LLVMDIBuilderCreateLexicalBlock.restype = LLVMMetadataRef
    LLVMDIBuilderCreateLexicalBlock.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateLexicalBlockFile = _libraries['llvm'].LLVMDIBuilderCreateLexicalBlockFile
    LLVMDIBuilderCreateLexicalBlockFile.restype = LLVMMetadataRef
    LLVMDIBuilderCreateLexicalBlockFile.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateImportedModuleFromNamespace = _libraries['llvm'].LLVMDIBuilderCreateImportedModuleFromNamespace
    LLVMDIBuilderCreateImportedModuleFromNamespace.restype = LLVMMetadataRef
    LLVMDIBuilderCreateImportedModuleFromNamespace.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateImportedModuleFromAlias = _libraries['llvm'].LLVMDIBuilderCreateImportedModuleFromAlias
    LLVMDIBuilderCreateImportedModuleFromAlias.restype = LLVMMetadataRef
    LLVMDIBuilderCreateImportedModuleFromAlias.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateImportedModuleFromModule = _libraries['llvm'].LLVMDIBuilderCreateImportedModuleFromModule
    LLVMDIBuilderCreateImportedModuleFromModule.restype = LLVMMetadataRef
    LLVMDIBuilderCreateImportedModuleFromModule.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateImportedDeclaration = _libraries['llvm'].LLVMDIBuilderCreateImportedDeclaration
    LLVMDIBuilderCreateImportedDeclaration.restype = LLVMMetadataRef
    LLVMDIBuilderCreateImportedDeclaration.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateDebugLocation = _libraries['llvm'].LLVMDIBuilderCreateDebugLocation
    LLVMDIBuilderCreateDebugLocation.restype = LLVMMetadataRef
    LLVMDIBuilderCreateDebugLocation.argtypes = [LLVMContextRef, ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDILocationGetLine = _libraries['llvm'].LLVMDILocationGetLine
    LLVMDILocationGetLine.restype = ctypes.c_uint32
    LLVMDILocationGetLine.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDILocationGetColumn = _libraries['llvm'].LLVMDILocationGetColumn
    LLVMDILocationGetColumn.restype = ctypes.c_uint32
    LLVMDILocationGetColumn.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDILocationGetScope = _libraries['llvm'].LLVMDILocationGetScope
    LLVMDILocationGetScope.restype = LLVMMetadataRef
    LLVMDILocationGetScope.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDILocationGetInlinedAt = _libraries['llvm'].LLVMDILocationGetInlinedAt
    LLVMDILocationGetInlinedAt.restype = LLVMMetadataRef
    LLVMDILocationGetInlinedAt.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIScopeGetFile = _libraries['llvm'].LLVMDIScopeGetFile
    LLVMDIScopeGetFile.restype = LLVMMetadataRef
    LLVMDIScopeGetFile.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIFileGetDirectory = _libraries['llvm'].LLVMDIFileGetDirectory
    LLVMDIFileGetDirectory.restype = ctypes.POINTER(ctypes.c_char)
    LLVMDIFileGetDirectory.argtypes = [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMDIFileGetFilename = _libraries['llvm'].LLVMDIFileGetFilename
    LLVMDIFileGetFilename.restype = ctypes.POINTER(ctypes.c_char)
    LLVMDIFileGetFilename.argtypes = [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMDIFileGetSource = _libraries['llvm'].LLVMDIFileGetSource
    LLVMDIFileGetSource.restype = ctypes.POINTER(ctypes.c_char)
    LLVMDIFileGetSource.argtypes = [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    LLVMDIBuilderGetOrCreateTypeArray = _libraries['llvm'].LLVMDIBuilderGetOrCreateTypeArray
    LLVMDIBuilderGetOrCreateTypeArray.restype = LLVMMetadataRef
    LLVMDIBuilderGetOrCreateTypeArray.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateSubroutineType = _libraries['llvm'].LLVMDIBuilderCreateSubroutineType
    LLVMDIBuilderCreateSubroutineType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateSubroutineType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32, LLVMDIFlags]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateMacro = _libraries['llvm'].LLVMDIBuilderCreateMacro
    LLVMDIBuilderCreateMacro.restype = LLVMMetadataRef
    LLVMDIBuilderCreateMacro.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMDWARFMacinfoRecordType, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateTempMacroFile = _libraries['llvm'].LLVMDIBuilderCreateTempMacroFile
    LLVMDIBuilderCreateTempMacroFile.restype = LLVMMetadataRef
    LLVMDIBuilderCreateTempMacroFile.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
int64_t = ctypes.c_int64
try:
    LLVMDIBuilderCreateEnumerator = _libraries['llvm'].LLVMDIBuilderCreateEnumerator
    LLVMDIBuilderCreateEnumerator.restype = LLVMMetadataRef
    LLVMDIBuilderCreateEnumerator.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, int64_t, LLVMBool]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    LLVMDIBuilderCreateEnumerationType = _libraries['llvm'].LLVMDIBuilderCreateEnumerationType
    LLVMDIBuilderCreateEnumerationType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateEnumerationType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateUnionType = _libraries['llvm'].LLVMDIBuilderCreateUnionType
    LLVMDIBuilderCreateUnionType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateUnionType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateArrayType = _libraries['llvm'].LLVMDIBuilderCreateArrayType
    LLVMDIBuilderCreateArrayType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateArrayType.argtypes = [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateVectorType = _libraries['llvm'].LLVMDIBuilderCreateVectorType
    LLVMDIBuilderCreateVectorType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateVectorType.argtypes = [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateUnspecifiedType = _libraries['llvm'].LLVMDIBuilderCreateUnspecifiedType
    LLVMDIBuilderCreateUnspecifiedType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateUnspecifiedType.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateBasicType = _libraries['llvm'].LLVMDIBuilderCreateBasicType
    LLVMDIBuilderCreateBasicType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateBasicType.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, uint64_t, LLVMDWARFTypeEncoding, LLVMDIFlags]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreatePointerType = _libraries['llvm'].LLVMDIBuilderCreatePointerType
    LLVMDIBuilderCreatePointerType.restype = LLVMMetadataRef
    LLVMDIBuilderCreatePointerType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, uint64_t, uint32_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateStructType = _libraries['llvm'].LLVMDIBuilderCreateStructType
    LLVMDIBuilderCreateStructType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateStructType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateMemberType = _libraries['llvm'].LLVMDIBuilderCreateMemberType
    LLVMDIBuilderCreateMemberType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateMemberType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateStaticMemberType = _libraries['llvm'].LLVMDIBuilderCreateStaticMemberType
    LLVMDIBuilderCreateStaticMemberType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateStaticMemberType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMDIFlags, LLVMValueRef, uint32_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateMemberPointerType = _libraries['llvm'].LLVMDIBuilderCreateMemberPointerType
    LLVMDIBuilderCreateMemberPointerType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateMemberPointerType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateObjCIVar = _libraries['llvm'].LLVMDIBuilderCreateObjCIVar
    LLVMDIBuilderCreateObjCIVar.restype = LLVMMetadataRef
    LLVMDIBuilderCreateObjCIVar.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateObjCProperty = _libraries['llvm'].LLVMDIBuilderCreateObjCProperty
    LLVMDIBuilderCreateObjCProperty.restype = LLVMMetadataRef
    LLVMDIBuilderCreateObjCProperty.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateObjectPointerType = _libraries['llvm'].LLVMDIBuilderCreateObjectPointerType
    LLVMDIBuilderCreateObjectPointerType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateObjectPointerType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateQualifiedType = _libraries['llvm'].LLVMDIBuilderCreateQualifiedType
    LLVMDIBuilderCreateQualifiedType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateQualifiedType.argtypes = [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateReferenceType = _libraries['llvm'].LLVMDIBuilderCreateReferenceType
    LLVMDIBuilderCreateReferenceType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateReferenceType.argtypes = [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateNullPtrType = _libraries['llvm'].LLVMDIBuilderCreateNullPtrType
    LLVMDIBuilderCreateNullPtrType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateNullPtrType.argtypes = [LLVMDIBuilderRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateTypedef = _libraries['llvm'].LLVMDIBuilderCreateTypedef
    LLVMDIBuilderCreateTypedef.restype = LLVMMetadataRef
    LLVMDIBuilderCreateTypedef.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, uint32_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateInheritance = _libraries['llvm'].LLVMDIBuilderCreateInheritance
    LLVMDIBuilderCreateInheritance.restype = LLVMMetadataRef
    LLVMDIBuilderCreateInheritance.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateForwardDecl = _libraries['llvm'].LLVMDIBuilderCreateForwardDecl
    LLVMDIBuilderCreateForwardDecl.restype = LLVMMetadataRef
    LLVMDIBuilderCreateForwardDecl.argtypes = [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateReplaceableCompositeType = _libraries['llvm'].LLVMDIBuilderCreateReplaceableCompositeType
    LLVMDIBuilderCreateReplaceableCompositeType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateReplaceableCompositeType.argtypes = [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateBitFieldMemberType = _libraries['llvm'].LLVMDIBuilderCreateBitFieldMemberType
    LLVMDIBuilderCreateBitFieldMemberType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateBitFieldMemberType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint64_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateClassType = _libraries['llvm'].LLVMDIBuilderCreateClassType
    LLVMDIBuilderCreateClassType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateClassType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateArtificialType = _libraries['llvm'].LLVMDIBuilderCreateArtificialType
    LLVMDIBuilderCreateArtificialType.restype = LLVMMetadataRef
    LLVMDIBuilderCreateArtificialType.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDITypeGetName = _libraries['llvm'].LLVMDITypeGetName
    LLVMDITypeGetName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMDITypeGetName.argtypes = [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMDITypeGetSizeInBits = _libraries['llvm'].LLVMDITypeGetSizeInBits
    LLVMDITypeGetSizeInBits.restype = uint64_t
    LLVMDITypeGetSizeInBits.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDITypeGetOffsetInBits = _libraries['llvm'].LLVMDITypeGetOffsetInBits
    LLVMDITypeGetOffsetInBits.restype = uint64_t
    LLVMDITypeGetOffsetInBits.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDITypeGetAlignInBits = _libraries['llvm'].LLVMDITypeGetAlignInBits
    LLVMDITypeGetAlignInBits.restype = uint32_t
    LLVMDITypeGetAlignInBits.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDITypeGetLine = _libraries['llvm'].LLVMDITypeGetLine
    LLVMDITypeGetLine.restype = ctypes.c_uint32
    LLVMDITypeGetLine.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDITypeGetFlags = _libraries['llvm'].LLVMDITypeGetFlags
    LLVMDITypeGetFlags.restype = LLVMDIFlags
    LLVMDITypeGetFlags.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderGetOrCreateSubrange = _libraries['llvm'].LLVMDIBuilderGetOrCreateSubrange
    LLVMDIBuilderGetOrCreateSubrange.restype = LLVMMetadataRef
    LLVMDIBuilderGetOrCreateSubrange.argtypes = [LLVMDIBuilderRef, int64_t, int64_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderGetOrCreateArray = _libraries['llvm'].LLVMDIBuilderGetOrCreateArray
    LLVMDIBuilderGetOrCreateArray.restype = LLVMMetadataRef
    LLVMDIBuilderGetOrCreateArray.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateExpression = _libraries['llvm'].LLVMDIBuilderCreateExpression
    LLVMDIBuilderCreateExpression.restype = LLVMMetadataRef
    LLVMDIBuilderCreateExpression.argtypes = [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_uint64), size_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateConstantValueExpression = _libraries['llvm'].LLVMDIBuilderCreateConstantValueExpression
    LLVMDIBuilderCreateConstantValueExpression.restype = LLVMMetadataRef
    LLVMDIBuilderCreateConstantValueExpression.argtypes = [LLVMDIBuilderRef, uint64_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateGlobalVariableExpression = _libraries['llvm'].LLVMDIBuilderCreateGlobalVariableExpression
    LLVMDIBuilderCreateGlobalVariableExpression.restype = LLVMMetadataRef
    LLVMDIBuilderCreateGlobalVariableExpression.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, LLVMMetadataRef, uint32_t]
except AttributeError:
    pass
try:
    LLVMDIGlobalVariableExpressionGetVariable = _libraries['llvm'].LLVMDIGlobalVariableExpressionGetVariable
    LLVMDIGlobalVariableExpressionGetVariable.restype = LLVMMetadataRef
    LLVMDIGlobalVariableExpressionGetVariable.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIGlobalVariableExpressionGetExpression = _libraries['llvm'].LLVMDIGlobalVariableExpressionGetExpression
    LLVMDIGlobalVariableExpressionGetExpression.restype = LLVMMetadataRef
    LLVMDIGlobalVariableExpressionGetExpression.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIVariableGetFile = _libraries['llvm'].LLVMDIVariableGetFile
    LLVMDIVariableGetFile.restype = LLVMMetadataRef
    LLVMDIVariableGetFile.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIVariableGetScope = _libraries['llvm'].LLVMDIVariableGetScope
    LLVMDIVariableGetScope.restype = LLVMMetadataRef
    LLVMDIVariableGetScope.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIVariableGetLine = _libraries['llvm'].LLVMDIVariableGetLine
    LLVMDIVariableGetLine.restype = ctypes.c_uint32
    LLVMDIVariableGetLine.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMTemporaryMDNode = _libraries['llvm'].LLVMTemporaryMDNode
    LLVMTemporaryMDNode.restype = LLVMMetadataRef
    LLVMTemporaryMDNode.argtypes = [LLVMContextRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMetadata)), size_t]
except AttributeError:
    pass
try:
    LLVMDisposeTemporaryMDNode = _libraries['llvm'].LLVMDisposeTemporaryMDNode
    LLVMDisposeTemporaryMDNode.restype = None
    LLVMDisposeTemporaryMDNode.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMMetadataReplaceAllUsesWith = _libraries['llvm'].LLVMMetadataReplaceAllUsesWith
    LLVMMetadataReplaceAllUsesWith.restype = None
    LLVMMetadataReplaceAllUsesWith.argtypes = [LLVMMetadataRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateTempGlobalVariableFwdDecl = _libraries['llvm'].LLVMDIBuilderCreateTempGlobalVariableFwdDecl
    LLVMDIBuilderCreateTempGlobalVariableFwdDecl.restype = LLVMMetadataRef
    LLVMDIBuilderCreateTempGlobalVariableFwdDecl.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, uint32_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderInsertDeclareBefore = _libraries['llvm'].LLVMDIBuilderInsertDeclareBefore
    LLVMDIBuilderInsertDeclareBefore.restype = LLVMValueRef
    LLVMDIBuilderInsertDeclareBefore.argtypes = [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderInsertDeclareAtEnd = _libraries['llvm'].LLVMDIBuilderInsertDeclareAtEnd
    LLVMDIBuilderInsertDeclareAtEnd.restype = LLVMValueRef
    LLVMDIBuilderInsertDeclareAtEnd.argtypes = [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderInsertDbgValueBefore = _libraries['llvm'].LLVMDIBuilderInsertDbgValueBefore
    LLVMDIBuilderInsertDbgValueBefore.restype = LLVMValueRef
    LLVMDIBuilderInsertDbgValueBefore.argtypes = [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderInsertDbgValueAtEnd = _libraries['llvm'].LLVMDIBuilderInsertDbgValueAtEnd
    LLVMDIBuilderInsertDbgValueAtEnd.restype = LLVMValueRef
    LLVMDIBuilderInsertDbgValueAtEnd.argtypes = [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateAutoVariable = _libraries['llvm'].LLVMDIBuilderCreateAutoVariable
    LLVMDIBuilderCreateAutoVariable.restype = LLVMMetadataRef
    LLVMDIBuilderCreateAutoVariable.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags, uint32_t]
except AttributeError:
    pass
try:
    LLVMDIBuilderCreateParameterVariable = _libraries['llvm'].LLVMDIBuilderCreateParameterVariable
    LLVMDIBuilderCreateParameterVariable.restype = LLVMMetadataRef
    LLVMDIBuilderCreateParameterVariable.argtypes = [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags]
except AttributeError:
    pass
try:
    LLVMGetSubprogram = _libraries['llvm'].LLVMGetSubprogram
    LLVMGetSubprogram.restype = LLVMMetadataRef
    LLVMGetSubprogram.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMSetSubprogram = _libraries['llvm'].LLVMSetSubprogram
    LLVMSetSubprogram.restype = None
    LLVMSetSubprogram.argtypes = [LLVMValueRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMDISubprogramGetLine = _libraries['llvm'].LLVMDISubprogramGetLine
    LLVMDISubprogramGetLine.restype = ctypes.c_uint32
    LLVMDISubprogramGetLine.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMInstructionGetDebugLoc = _libraries['llvm'].LLVMInstructionGetDebugLoc
    LLVMInstructionGetDebugLoc.restype = LLVMMetadataRef
    LLVMInstructionGetDebugLoc.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMInstructionSetDebugLoc = _libraries['llvm'].LLVMInstructionSetDebugLoc
    LLVMInstructionSetDebugLoc.restype = None
    LLVMInstructionSetDebugLoc.argtypes = [LLVMValueRef, LLVMMetadataRef]
except AttributeError:
    pass
try:
    LLVMGetMetadataKind = _libraries['llvm'].LLVMGetMetadataKind
    LLVMGetMetadataKind.restype = LLVMMetadataKind
    LLVMGetMetadataKind.argtypes = [LLVMMetadataRef]
except AttributeError:
    pass
LLVM_C_DISASSEMBLER_H = True # macro
LLVM_C_DISASSEMBLERTYPES_H = True # macro
LLVMDisassembler_VariantKind_None = 0 # macro
LLVMDisassembler_VariantKind_ARM_HI16 = 1 # macro
LLVMDisassembler_VariantKind_ARM_LO16 = 2 # macro
LLVMDisassembler_VariantKind_ARM64_PAGE = 1 # macro
LLVMDisassembler_VariantKind_ARM64_PAGEOFF = 2 # macro
LLVMDisassembler_VariantKind_ARM64_GOTPAGE = 3 # macro
LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF = 4 # macro
LLVMDisassembler_VariantKind_ARM64_TLVP = 5 # macro
LLVMDisassembler_VariantKind_ARM64_TLVOFF = 6 # macro
LLVMDisassembler_ReferenceType_InOut_None = 0 # macro
LLVMDisassembler_ReferenceType_In_Branch = 1 # macro
LLVMDisassembler_ReferenceType_In_PCrel_Load = 2 # macro
LLVMDisassembler_ReferenceType_In_ARM64_ADRP = 0x100000001 # macro
LLVMDisassembler_ReferenceType_In_ARM64_ADDXri = 0x100000002 # macro
LLVMDisassembler_ReferenceType_In_ARM64_LDRXui = 0x100000003 # macro
LLVMDisassembler_ReferenceType_In_ARM64_LDRXl = 0x100000004 # macro
LLVMDisassembler_ReferenceType_In_ARM64_ADR = 0x100000005 # macro
LLVMDisassembler_ReferenceType_Out_SymbolStub = 1 # macro
LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr = 2 # macro
LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr = 3 # macro
LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref = 4 # macro
LLVMDisassembler_ReferenceType_Out_Objc_Message = 5 # macro
LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref = 6 # macro
LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref = 7 # macro
LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref = 8 # macro
LLVMDisassembler_ReferenceType_DeMangled_Name = 9 # macro
LLVMDisassembler_Option_UseMarkup = 1 # macro
LLVMDisassembler_Option_PrintImmHex = 2 # macro
LLVMDisassembler_Option_AsmPrinterVariant = 4 # macro
LLVMDisassembler_Option_SetInstrComments = 8 # macro
LLVMDisassembler_Option_PrintLatency = 16 # macro
LLVMDisasmContextRef = ctypes.POINTER(None)
LLVMOpInfoCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, ctypes.POINTER(None))
class struct_LLVMOpInfoSymbol1(Structure):
    pass

struct_LLVMOpInfoSymbol1._pack_ = 1 # source:False
struct_LLVMOpInfoSymbol1._fields_ = [
    ('Present', ctypes.c_uint64),
    ('Name', ctypes.POINTER(ctypes.c_char)),
    ('Value', ctypes.c_uint64),
]

class struct_LLVMOpInfo1(Structure):
    pass

struct_LLVMOpInfo1._pack_ = 1 # source:False
struct_LLVMOpInfo1._fields_ = [
    ('AddSymbol', struct_LLVMOpInfoSymbol1),
    ('SubtractSymbol', struct_LLVMOpInfoSymbol1),
    ('Value', ctypes.c_uint64),
    ('VariantKind', ctypes.c_uint64),
]

LLVMSymbolLookupCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
try:
    LLVMCreateDisasm = _libraries['llvm'].LLVMCreateDisasm
    LLVMCreateDisasm.restype = LLVMDisasmContextRef
    LLVMCreateDisasm.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError:
    pass
try:
    LLVMCreateDisasmCPU = _libraries['llvm'].LLVMCreateDisasmCPU
    LLVMCreateDisasmCPU.restype = LLVMDisasmContextRef
    LLVMCreateDisasmCPU.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError:
    pass
try:
    LLVMCreateDisasmCPUFeatures = _libraries['llvm'].LLVMCreateDisasmCPUFeatures
    LLVMCreateDisasmCPUFeatures.restype = LLVMDisasmContextRef
    LLVMCreateDisasmCPUFeatures.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError:
    pass
try:
    LLVMSetDisasmOptions = _libraries['llvm'].LLVMSetDisasmOptions
    LLVMSetDisasmOptions.restype = ctypes.c_int32
    LLVMSetDisasmOptions.argtypes = [LLVMDisasmContextRef, uint64_t]
except AttributeError:
    pass
try:
    LLVMDisasmDispose = _libraries['llvm'].LLVMDisasmDispose
    LLVMDisasmDispose.restype = None
    LLVMDisasmDispose.argtypes = [LLVMDisasmContextRef]
except AttributeError:
    pass
try:
    LLVMDisasmInstruction = _libraries['llvm'].LLVMDisasmInstruction
    LLVMDisasmInstruction.restype = size_t
    LLVMDisasmInstruction.argtypes = [LLVMDisasmContextRef, ctypes.POINTER(ctypes.c_ubyte), uint64_t, uint64_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
LLVM_C_ERROR_H = True # macro
LLVMErrorSuccess = 0 # macro
class struct_LLVMOpaqueError(Structure):
    pass

LLVMErrorRef = ctypes.POINTER(struct_LLVMOpaqueError)
LLVMErrorTypeId = ctypes.POINTER(None)
try:
    LLVMGetErrorTypeId = _libraries['llvm'].LLVMGetErrorTypeId
    LLVMGetErrorTypeId.restype = LLVMErrorTypeId
    LLVMGetErrorTypeId.argtypes = [LLVMErrorRef]
except AttributeError:
    pass
try:
    LLVMConsumeError = _libraries['llvm'].LLVMConsumeError
    LLVMConsumeError.restype = None
    LLVMConsumeError.argtypes = [LLVMErrorRef]
except AttributeError:
    pass
try:
    LLVMGetErrorMessage = _libraries['llvm'].LLVMGetErrorMessage
    LLVMGetErrorMessage.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetErrorMessage.argtypes = [LLVMErrorRef]
except AttributeError:
    pass
try:
    LLVMDisposeErrorMessage = _libraries['llvm'].LLVMDisposeErrorMessage
    LLVMDisposeErrorMessage.restype = None
    LLVMDisposeErrorMessage.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetStringErrorTypeId = _libraries['llvm'].LLVMGetStringErrorTypeId
    LLVMGetStringErrorTypeId.restype = LLVMErrorTypeId
    LLVMGetStringErrorTypeId.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreateStringError = _libraries['llvm'].LLVMCreateStringError
    LLVMCreateStringError.restype = LLVMErrorRef
    LLVMCreateStringError.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
LLVM_C_EXECUTIONENGINE_H = True # macro
LLVM_C_TARGET_H = True # macro
# def LLVM_TARGET(TargetName):  # macro
#    return LLVMInitialize##TargetName##TargetMC;
# def LLVM_ASM_PRINTER(TargetName):  # macro
#    return LLVMInitialize##TargetName##AsmPrinter;
# def LLVM_ASM_PARSER(TargetName):  # macro
#    return LLVMInitialize##TargetName##AsmParser;
# def LLVM_DISASSEMBLER(TargetName):  # macro
#    return LLVMInitialize##TargetName##Disassembler;
LLVM_C_TARGETMACHINE_H = True # macro

# values for enumeration 'LLVMByteOrdering'
LLVMByteOrdering__enumvalues = {
    0: 'LLVMBigEndian',
    1: 'LLVMLittleEndian',
}
LLVMBigEndian = 0
LLVMLittleEndian = 1
LLVMByteOrdering = ctypes.c_uint32 # enum
class struct_LLVMOpaqueTargetData(Structure):
    pass

LLVMTargetDataRef = ctypes.POINTER(struct_LLVMOpaqueTargetData)
class struct_LLVMOpaqueTargetLibraryInfotData(Structure):
    pass

LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
try:
    LLVMInitializeAArch64TargetInfo = _libraries['llvm'].LLVMInitializeAArch64TargetInfo
    LLVMInitializeAArch64TargetInfo.restype = None
    LLVMInitializeAArch64TargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUTargetInfo = _libraries['llvm'].LLVMInitializeAMDGPUTargetInfo
    LLVMInitializeAMDGPUTargetInfo.restype = None
    LLVMInitializeAMDGPUTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMTargetInfo = _libraries['llvm'].LLVMInitializeARMTargetInfo
    LLVMInitializeARMTargetInfo.restype = None
    LLVMInitializeARMTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRTargetInfo = _libraries['llvm'].LLVMInitializeAVRTargetInfo
    LLVMInitializeAVRTargetInfo.restype = None
    LLVMInitializeAVRTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFTargetInfo = _libraries['llvm'].LLVMInitializeBPFTargetInfo
    LLVMInitializeBPFTargetInfo.restype = None
    LLVMInitializeBPFTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonTargetInfo = _libraries['llvm'].LLVMInitializeHexagonTargetInfo
    LLVMInitializeHexagonTargetInfo.restype = None
    LLVMInitializeHexagonTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiTargetInfo = _libraries['llvm'].LLVMInitializeLanaiTargetInfo
    LLVMInitializeLanaiTargetInfo.restype = None
    LLVMInitializeLanaiTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsTargetInfo = _libraries['llvm'].LLVMInitializeMipsTargetInfo
    LLVMInitializeMipsTargetInfo.restype = None
    LLVMInitializeMipsTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430TargetInfo = _libraries['llvm'].LLVMInitializeMSP430TargetInfo
    LLVMInitializeMSP430TargetInfo.restype = None
    LLVMInitializeMSP430TargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNVPTXTargetInfo = _libraries['llvm'].LLVMInitializeNVPTXTargetInfo
    LLVMInitializeNVPTXTargetInfo.restype = None
    LLVMInitializeNVPTXTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCTargetInfo = _libraries['llvm'].LLVMInitializePowerPCTargetInfo
    LLVMInitializePowerPCTargetInfo.restype = None
    LLVMInitializePowerPCTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVTargetInfo = _libraries['llvm'].LLVMInitializeRISCVTargetInfo
    LLVMInitializeRISCVTargetInfo.restype = None
    LLVMInitializeRISCVTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcTargetInfo = _libraries['llvm'].LLVMInitializeSparcTargetInfo
    LLVMInitializeSparcTargetInfo.restype = None
    LLVMInitializeSparcTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZTargetInfo = _libraries['llvm'].LLVMInitializeSystemZTargetInfo
    LLVMInitializeSystemZTargetInfo.restype = None
    LLVMInitializeSystemZTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVETargetInfo = _libraries['llvm'].LLVMInitializeVETargetInfo
    LLVMInitializeVETargetInfo.restype = None
    LLVMInitializeVETargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyTargetInfo = _libraries['llvm'].LLVMInitializeWebAssemblyTargetInfo
    LLVMInitializeWebAssemblyTargetInfo.restype = None
    LLVMInitializeWebAssemblyTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86TargetInfo = _libraries['llvm'].LLVMInitializeX86TargetInfo
    LLVMInitializeX86TargetInfo.restype = None
    LLVMInitializeX86TargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeXCoreTargetInfo = _libraries['llvm'].LLVMInitializeXCoreTargetInfo
    LLVMInitializeXCoreTargetInfo.restype = None
    LLVMInitializeXCoreTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kTargetInfo = _libraries['llvm'].LLVMInitializeM68kTargetInfo
    LLVMInitializeM68kTargetInfo.restype = None
    LLVMInitializeM68kTargetInfo.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAArch64Target = _libraries['llvm'].LLVMInitializeAArch64Target
    LLVMInitializeAArch64Target.restype = None
    LLVMInitializeAArch64Target.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUTarget = _libraries['llvm'].LLVMInitializeAMDGPUTarget
    LLVMInitializeAMDGPUTarget.restype = None
    LLVMInitializeAMDGPUTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMTarget = _libraries['llvm'].LLVMInitializeARMTarget
    LLVMInitializeARMTarget.restype = None
    LLVMInitializeARMTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRTarget = _libraries['llvm'].LLVMInitializeAVRTarget
    LLVMInitializeAVRTarget.restype = None
    LLVMInitializeAVRTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFTarget = _libraries['llvm'].LLVMInitializeBPFTarget
    LLVMInitializeBPFTarget.restype = None
    LLVMInitializeBPFTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonTarget = _libraries['llvm'].LLVMInitializeHexagonTarget
    LLVMInitializeHexagonTarget.restype = None
    LLVMInitializeHexagonTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiTarget = _libraries['llvm'].LLVMInitializeLanaiTarget
    LLVMInitializeLanaiTarget.restype = None
    LLVMInitializeLanaiTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsTarget = _libraries['llvm'].LLVMInitializeMipsTarget
    LLVMInitializeMipsTarget.restype = None
    LLVMInitializeMipsTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430Target = _libraries['llvm'].LLVMInitializeMSP430Target
    LLVMInitializeMSP430Target.restype = None
    LLVMInitializeMSP430Target.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNVPTXTarget = _libraries['llvm'].LLVMInitializeNVPTXTarget
    LLVMInitializeNVPTXTarget.restype = None
    LLVMInitializeNVPTXTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCTarget = _libraries['llvm'].LLVMInitializePowerPCTarget
    LLVMInitializePowerPCTarget.restype = None
    LLVMInitializePowerPCTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVTarget = _libraries['llvm'].LLVMInitializeRISCVTarget
    LLVMInitializeRISCVTarget.restype = None
    LLVMInitializeRISCVTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcTarget = _libraries['llvm'].LLVMInitializeSparcTarget
    LLVMInitializeSparcTarget.restype = None
    LLVMInitializeSparcTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZTarget = _libraries['llvm'].LLVMInitializeSystemZTarget
    LLVMInitializeSystemZTarget.restype = None
    LLVMInitializeSystemZTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVETarget = _libraries['llvm'].LLVMInitializeVETarget
    LLVMInitializeVETarget.restype = None
    LLVMInitializeVETarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyTarget = _libraries['llvm'].LLVMInitializeWebAssemblyTarget
    LLVMInitializeWebAssemblyTarget.restype = None
    LLVMInitializeWebAssemblyTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86Target = _libraries['llvm'].LLVMInitializeX86Target
    LLVMInitializeX86Target.restype = None
    LLVMInitializeX86Target.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeXCoreTarget = _libraries['llvm'].LLVMInitializeXCoreTarget
    LLVMInitializeXCoreTarget.restype = None
    LLVMInitializeXCoreTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kTarget = _libraries['llvm'].LLVMInitializeM68kTarget
    LLVMInitializeM68kTarget.restype = None
    LLVMInitializeM68kTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAArch64TargetMC = _libraries['llvm'].LLVMInitializeAArch64TargetMC
    LLVMInitializeAArch64TargetMC.restype = None
    LLVMInitializeAArch64TargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUTargetMC = _libraries['llvm'].LLVMInitializeAMDGPUTargetMC
    LLVMInitializeAMDGPUTargetMC.restype = None
    LLVMInitializeAMDGPUTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMTargetMC = _libraries['llvm'].LLVMInitializeARMTargetMC
    LLVMInitializeARMTargetMC.restype = None
    LLVMInitializeARMTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRTargetMC = _libraries['llvm'].LLVMInitializeAVRTargetMC
    LLVMInitializeAVRTargetMC.restype = None
    LLVMInitializeAVRTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFTargetMC = _libraries['llvm'].LLVMInitializeBPFTargetMC
    LLVMInitializeBPFTargetMC.restype = None
    LLVMInitializeBPFTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonTargetMC = _libraries['llvm'].LLVMInitializeHexagonTargetMC
    LLVMInitializeHexagonTargetMC.restype = None
    LLVMInitializeHexagonTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiTargetMC = _libraries['llvm'].LLVMInitializeLanaiTargetMC
    LLVMInitializeLanaiTargetMC.restype = None
    LLVMInitializeLanaiTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsTargetMC = _libraries['llvm'].LLVMInitializeMipsTargetMC
    LLVMInitializeMipsTargetMC.restype = None
    LLVMInitializeMipsTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430TargetMC = _libraries['llvm'].LLVMInitializeMSP430TargetMC
    LLVMInitializeMSP430TargetMC.restype = None
    LLVMInitializeMSP430TargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNVPTXTargetMC = _libraries['llvm'].LLVMInitializeNVPTXTargetMC
    LLVMInitializeNVPTXTargetMC.restype = None
    LLVMInitializeNVPTXTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCTargetMC = _libraries['llvm'].LLVMInitializePowerPCTargetMC
    LLVMInitializePowerPCTargetMC.restype = None
    LLVMInitializePowerPCTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVTargetMC = _libraries['llvm'].LLVMInitializeRISCVTargetMC
    LLVMInitializeRISCVTargetMC.restype = None
    LLVMInitializeRISCVTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcTargetMC = _libraries['llvm'].LLVMInitializeSparcTargetMC
    LLVMInitializeSparcTargetMC.restype = None
    LLVMInitializeSparcTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZTargetMC = _libraries['llvm'].LLVMInitializeSystemZTargetMC
    LLVMInitializeSystemZTargetMC.restype = None
    LLVMInitializeSystemZTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVETargetMC = _libraries['llvm'].LLVMInitializeVETargetMC
    LLVMInitializeVETargetMC.restype = None
    LLVMInitializeVETargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyTargetMC = _libraries['llvm'].LLVMInitializeWebAssemblyTargetMC
    LLVMInitializeWebAssemblyTargetMC.restype = None
    LLVMInitializeWebAssemblyTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86TargetMC = _libraries['llvm'].LLVMInitializeX86TargetMC
    LLVMInitializeX86TargetMC.restype = None
    LLVMInitializeX86TargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeXCoreTargetMC = _libraries['llvm'].LLVMInitializeXCoreTargetMC
    LLVMInitializeXCoreTargetMC.restype = None
    LLVMInitializeXCoreTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kTargetMC = _libraries['llvm'].LLVMInitializeM68kTargetMC
    LLVMInitializeM68kTargetMC.restype = None
    LLVMInitializeM68kTargetMC.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAArch64AsmPrinter = _libraries['llvm'].LLVMInitializeAArch64AsmPrinter
    LLVMInitializeAArch64AsmPrinter.restype = None
    LLVMInitializeAArch64AsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUAsmPrinter = _libraries['llvm'].LLVMInitializeAMDGPUAsmPrinter
    LLVMInitializeAMDGPUAsmPrinter.restype = None
    LLVMInitializeAMDGPUAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMAsmPrinter = _libraries['llvm'].LLVMInitializeARMAsmPrinter
    LLVMInitializeARMAsmPrinter.restype = None
    LLVMInitializeARMAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRAsmPrinter = _libraries['llvm'].LLVMInitializeAVRAsmPrinter
    LLVMInitializeAVRAsmPrinter.restype = None
    LLVMInitializeAVRAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFAsmPrinter = _libraries['llvm'].LLVMInitializeBPFAsmPrinter
    LLVMInitializeBPFAsmPrinter.restype = None
    LLVMInitializeBPFAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonAsmPrinter = _libraries['llvm'].LLVMInitializeHexagonAsmPrinter
    LLVMInitializeHexagonAsmPrinter.restype = None
    LLVMInitializeHexagonAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiAsmPrinter = _libraries['llvm'].LLVMInitializeLanaiAsmPrinter
    LLVMInitializeLanaiAsmPrinter.restype = None
    LLVMInitializeLanaiAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsAsmPrinter = _libraries['llvm'].LLVMInitializeMipsAsmPrinter
    LLVMInitializeMipsAsmPrinter.restype = None
    LLVMInitializeMipsAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430AsmPrinter = _libraries['llvm'].LLVMInitializeMSP430AsmPrinter
    LLVMInitializeMSP430AsmPrinter.restype = None
    LLVMInitializeMSP430AsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNVPTXAsmPrinter = _libraries['llvm'].LLVMInitializeNVPTXAsmPrinter
    LLVMInitializeNVPTXAsmPrinter.restype = None
    LLVMInitializeNVPTXAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCAsmPrinter = _libraries['llvm'].LLVMInitializePowerPCAsmPrinter
    LLVMInitializePowerPCAsmPrinter.restype = None
    LLVMInitializePowerPCAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVAsmPrinter = _libraries['llvm'].LLVMInitializeRISCVAsmPrinter
    LLVMInitializeRISCVAsmPrinter.restype = None
    LLVMInitializeRISCVAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcAsmPrinter = _libraries['llvm'].LLVMInitializeSparcAsmPrinter
    LLVMInitializeSparcAsmPrinter.restype = None
    LLVMInitializeSparcAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZAsmPrinter = _libraries['llvm'].LLVMInitializeSystemZAsmPrinter
    LLVMInitializeSystemZAsmPrinter.restype = None
    LLVMInitializeSystemZAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVEAsmPrinter = _libraries['llvm'].LLVMInitializeVEAsmPrinter
    LLVMInitializeVEAsmPrinter.restype = None
    LLVMInitializeVEAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyAsmPrinter = _libraries['llvm'].LLVMInitializeWebAssemblyAsmPrinter
    LLVMInitializeWebAssemblyAsmPrinter.restype = None
    LLVMInitializeWebAssemblyAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86AsmPrinter = _libraries['llvm'].LLVMInitializeX86AsmPrinter
    LLVMInitializeX86AsmPrinter.restype = None
    LLVMInitializeX86AsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeXCoreAsmPrinter = _libraries['llvm'].LLVMInitializeXCoreAsmPrinter
    LLVMInitializeXCoreAsmPrinter.restype = None
    LLVMInitializeXCoreAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kAsmPrinter = _libraries['llvm'].LLVMInitializeM68kAsmPrinter
    LLVMInitializeM68kAsmPrinter.restype = None
    LLVMInitializeM68kAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAArch64AsmParser = _libraries['llvm'].LLVMInitializeAArch64AsmParser
    LLVMInitializeAArch64AsmParser.restype = None
    LLVMInitializeAArch64AsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUAsmParser = _libraries['llvm'].LLVMInitializeAMDGPUAsmParser
    LLVMInitializeAMDGPUAsmParser.restype = None
    LLVMInitializeAMDGPUAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMAsmParser = _libraries['llvm'].LLVMInitializeARMAsmParser
    LLVMInitializeARMAsmParser.restype = None
    LLVMInitializeARMAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRAsmParser = _libraries['llvm'].LLVMInitializeAVRAsmParser
    LLVMInitializeAVRAsmParser.restype = None
    LLVMInitializeAVRAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFAsmParser = _libraries['llvm'].LLVMInitializeBPFAsmParser
    LLVMInitializeBPFAsmParser.restype = None
    LLVMInitializeBPFAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonAsmParser = _libraries['llvm'].LLVMInitializeHexagonAsmParser
    LLVMInitializeHexagonAsmParser.restype = None
    LLVMInitializeHexagonAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiAsmParser = _libraries['llvm'].LLVMInitializeLanaiAsmParser
    LLVMInitializeLanaiAsmParser.restype = None
    LLVMInitializeLanaiAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsAsmParser = _libraries['llvm'].LLVMInitializeMipsAsmParser
    LLVMInitializeMipsAsmParser.restype = None
    LLVMInitializeMipsAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430AsmParser = _libraries['llvm'].LLVMInitializeMSP430AsmParser
    LLVMInitializeMSP430AsmParser.restype = None
    LLVMInitializeMSP430AsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCAsmParser = _libraries['llvm'].LLVMInitializePowerPCAsmParser
    LLVMInitializePowerPCAsmParser.restype = None
    LLVMInitializePowerPCAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVAsmParser = _libraries['llvm'].LLVMInitializeRISCVAsmParser
    LLVMInitializeRISCVAsmParser.restype = None
    LLVMInitializeRISCVAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcAsmParser = _libraries['llvm'].LLVMInitializeSparcAsmParser
    LLVMInitializeSparcAsmParser.restype = None
    LLVMInitializeSparcAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZAsmParser = _libraries['llvm'].LLVMInitializeSystemZAsmParser
    LLVMInitializeSystemZAsmParser.restype = None
    LLVMInitializeSystemZAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVEAsmParser = _libraries['llvm'].LLVMInitializeVEAsmParser
    LLVMInitializeVEAsmParser.restype = None
    LLVMInitializeVEAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyAsmParser = _libraries['llvm'].LLVMInitializeWebAssemblyAsmParser
    LLVMInitializeWebAssemblyAsmParser.restype = None
    LLVMInitializeWebAssemblyAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86AsmParser = _libraries['llvm'].LLVMInitializeX86AsmParser
    LLVMInitializeX86AsmParser.restype = None
    LLVMInitializeX86AsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kAsmParser = _libraries['llvm'].LLVMInitializeM68kAsmParser
    LLVMInitializeM68kAsmParser.restype = None
    LLVMInitializeM68kAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAArch64Disassembler = _libraries['llvm'].LLVMInitializeAArch64Disassembler
    LLVMInitializeAArch64Disassembler.restype = None
    LLVMInitializeAArch64Disassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAMDGPUDisassembler = _libraries['llvm'].LLVMInitializeAMDGPUDisassembler
    LLVMInitializeAMDGPUDisassembler.restype = None
    LLVMInitializeAMDGPUDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeARMDisassembler = _libraries['llvm'].LLVMInitializeARMDisassembler
    LLVMInitializeARMDisassembler.restype = None
    LLVMInitializeARMDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAVRDisassembler = _libraries['llvm'].LLVMInitializeAVRDisassembler
    LLVMInitializeAVRDisassembler.restype = None
    LLVMInitializeAVRDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeBPFDisassembler = _libraries['llvm'].LLVMInitializeBPFDisassembler
    LLVMInitializeBPFDisassembler.restype = None
    LLVMInitializeBPFDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeHexagonDisassembler = _libraries['llvm'].LLVMInitializeHexagonDisassembler
    LLVMInitializeHexagonDisassembler.restype = None
    LLVMInitializeHexagonDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeLanaiDisassembler = _libraries['llvm'].LLVMInitializeLanaiDisassembler
    LLVMInitializeLanaiDisassembler.restype = None
    LLVMInitializeLanaiDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMipsDisassembler = _libraries['llvm'].LLVMInitializeMipsDisassembler
    LLVMInitializeMipsDisassembler.restype = None
    LLVMInitializeMipsDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeMSP430Disassembler = _libraries['llvm'].LLVMInitializeMSP430Disassembler
    LLVMInitializeMSP430Disassembler.restype = None
    LLVMInitializeMSP430Disassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializePowerPCDisassembler = _libraries['llvm'].LLVMInitializePowerPCDisassembler
    LLVMInitializePowerPCDisassembler.restype = None
    LLVMInitializePowerPCDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeRISCVDisassembler = _libraries['llvm'].LLVMInitializeRISCVDisassembler
    LLVMInitializeRISCVDisassembler.restype = None
    LLVMInitializeRISCVDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSparcDisassembler = _libraries['llvm'].LLVMInitializeSparcDisassembler
    LLVMInitializeSparcDisassembler.restype = None
    LLVMInitializeSparcDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeSystemZDisassembler = _libraries['llvm'].LLVMInitializeSystemZDisassembler
    LLVMInitializeSystemZDisassembler.restype = None
    LLVMInitializeSystemZDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeVEDisassembler = _libraries['llvm'].LLVMInitializeVEDisassembler
    LLVMInitializeVEDisassembler.restype = None
    LLVMInitializeVEDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeWebAssemblyDisassembler = _libraries['llvm'].LLVMInitializeWebAssemblyDisassembler
    LLVMInitializeWebAssemblyDisassembler.restype = None
    LLVMInitializeWebAssemblyDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeX86Disassembler = _libraries['llvm'].LLVMInitializeX86Disassembler
    LLVMInitializeX86Disassembler.restype = None
    LLVMInitializeX86Disassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeXCoreDisassembler = _libraries['llvm'].LLVMInitializeXCoreDisassembler
    LLVMInitializeXCoreDisassembler.restype = None
    LLVMInitializeXCoreDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeM68kDisassembler = _libraries['llvm'].LLVMInitializeM68kDisassembler
    LLVMInitializeM68kDisassembler.restype = None
    LLVMInitializeM68kDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllTargetInfos = _libraries['llvm'].LLVMInitializeAllTargetInfos
    LLVMInitializeAllTargetInfos.restype = None
    LLVMInitializeAllTargetInfos.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllTargets = _libraries['llvm'].LLVMInitializeAllTargets
    LLVMInitializeAllTargets.restype = None
    LLVMInitializeAllTargets.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllTargetMCs = _libraries['llvm'].LLVMInitializeAllTargetMCs
    LLVMInitializeAllTargetMCs.restype = None
    LLVMInitializeAllTargetMCs.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllAsmPrinters = _libraries['llvm'].LLVMInitializeAllAsmPrinters
    LLVMInitializeAllAsmPrinters.restype = None
    LLVMInitializeAllAsmPrinters.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllAsmParsers = _libraries['llvm'].LLVMInitializeAllAsmParsers
    LLVMInitializeAllAsmParsers.restype = None
    LLVMInitializeAllAsmParsers.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeAllDisassemblers = _libraries['llvm'].LLVMInitializeAllDisassemblers
    LLVMInitializeAllDisassemblers.restype = None
    LLVMInitializeAllDisassemblers.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNativeTarget = _libraries['llvm'].LLVMInitializeNativeTarget
    LLVMInitializeNativeTarget.restype = LLVMBool
    LLVMInitializeNativeTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNativeAsmParser = _libraries['llvm'].LLVMInitializeNativeAsmParser
    LLVMInitializeNativeAsmParser.restype = LLVMBool
    LLVMInitializeNativeAsmParser.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNativeAsmPrinter = _libraries['llvm'].LLVMInitializeNativeAsmPrinter
    LLVMInitializeNativeAsmPrinter.restype = LLVMBool
    LLVMInitializeNativeAsmPrinter.argtypes = []
except AttributeError:
    pass
try:
    LLVMInitializeNativeDisassembler = _libraries['llvm'].LLVMInitializeNativeDisassembler
    LLVMInitializeNativeDisassembler.restype = LLVMBool
    LLVMInitializeNativeDisassembler.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetModuleDataLayout = _libraries['llvm'].LLVMGetModuleDataLayout
    LLVMGetModuleDataLayout.restype = LLVMTargetDataRef
    LLVMGetModuleDataLayout.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMSetModuleDataLayout = _libraries['llvm'].LLVMSetModuleDataLayout
    LLVMSetModuleDataLayout.restype = None
    LLVMSetModuleDataLayout.argtypes = [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMCreateTargetData = _libraries['llvm'].LLVMCreateTargetData
    LLVMCreateTargetData.restype = LLVMTargetDataRef
    LLVMCreateTargetData.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMDisposeTargetData = _libraries['llvm'].LLVMDisposeTargetData
    LLVMDisposeTargetData.restype = None
    LLVMDisposeTargetData.argtypes = [LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMAddTargetLibraryInfo = _libraries['llvm'].LLVMAddTargetLibraryInfo
    LLVMAddTargetLibraryInfo.restype = None
    LLVMAddTargetLibraryInfo.argtypes = [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMCopyStringRepOfTargetData = _libraries['llvm'].LLVMCopyStringRepOfTargetData
    LLVMCopyStringRepOfTargetData.restype = ctypes.POINTER(ctypes.c_char)
    LLVMCopyStringRepOfTargetData.argtypes = [LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMByteOrder = _libraries['llvm'].LLVMByteOrder
    LLVMByteOrder.restype = LLVMByteOrdering
    LLVMByteOrder.argtypes = [LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMPointerSize = _libraries['llvm'].LLVMPointerSize
    LLVMPointerSize.restype = ctypes.c_uint32
    LLVMPointerSize.argtypes = [LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMPointerSizeForAS = _libraries['llvm'].LLVMPointerSizeForAS
    LLVMPointerSizeForAS.restype = ctypes.c_uint32
    LLVMPointerSizeForAS.argtypes = [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMIntPtrType = _libraries['llvm'].LLVMIntPtrType
    LLVMIntPtrType.restype = LLVMTypeRef
    LLVMIntPtrType.argtypes = [LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMIntPtrTypeForAS = _libraries['llvm'].LLVMIntPtrTypeForAS
    LLVMIntPtrTypeForAS.restype = LLVMTypeRef
    LLVMIntPtrTypeForAS.argtypes = [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMIntPtrTypeInContext = _libraries['llvm'].LLVMIntPtrTypeInContext
    LLVMIntPtrTypeInContext.restype = LLVMTypeRef
    LLVMIntPtrTypeInContext.argtypes = [LLVMContextRef, LLVMTargetDataRef]
except AttributeError:
    pass
try:
    LLVMIntPtrTypeForASInContext = _libraries['llvm'].LLVMIntPtrTypeForASInContext
    LLVMIntPtrTypeForASInContext.restype = LLVMTypeRef
    LLVMIntPtrTypeForASInContext.argtypes = [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMSizeOfTypeInBits = _libraries['llvm'].LLVMSizeOfTypeInBits
    LLVMSizeOfTypeInBits.restype = ctypes.c_uint64
    LLVMSizeOfTypeInBits.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMStoreSizeOfType = _libraries['llvm'].LLVMStoreSizeOfType
    LLVMStoreSizeOfType.restype = ctypes.c_uint64
    LLVMStoreSizeOfType.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMABISizeOfType = _libraries['llvm'].LLVMABISizeOfType
    LLVMABISizeOfType.restype = ctypes.c_uint64
    LLVMABISizeOfType.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMABIAlignmentOfType = _libraries['llvm'].LLVMABIAlignmentOfType
    LLVMABIAlignmentOfType.restype = ctypes.c_uint32
    LLVMABIAlignmentOfType.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMCallFrameAlignmentOfType = _libraries['llvm'].LLVMCallFrameAlignmentOfType
    LLVMCallFrameAlignmentOfType.restype = ctypes.c_uint32
    LLVMCallFrameAlignmentOfType.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMPreferredAlignmentOfType = _libraries['llvm'].LLVMPreferredAlignmentOfType
    LLVMPreferredAlignmentOfType.restype = ctypes.c_uint32
    LLVMPreferredAlignmentOfType.argtypes = [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    LLVMPreferredAlignmentOfGlobal = _libraries['llvm'].LLVMPreferredAlignmentOfGlobal
    LLVMPreferredAlignmentOfGlobal.restype = ctypes.c_uint32
    LLVMPreferredAlignmentOfGlobal.argtypes = [LLVMTargetDataRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMElementAtOffset = _libraries['llvm'].LLVMElementAtOffset
    LLVMElementAtOffset.restype = ctypes.c_uint32
    LLVMElementAtOffset.argtypes = [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError:
    pass
try:
    LLVMOffsetOfElement = _libraries['llvm'].LLVMOffsetOfElement
    LLVMOffsetOfElement.restype = ctypes.c_uint64
    LLVMOffsetOfElement.argtypes = [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError:
    pass
class struct_LLVMOpaqueTargetMachine(Structure):
    pass

LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
class struct_LLVMTarget(Structure):
    pass

LLVMTargetRef = ctypes.POINTER(struct_LLVMTarget)

# values for enumeration 'c__EA_LLVMCodeGenOptLevel'
c__EA_LLVMCodeGenOptLevel__enumvalues = {
    0: 'LLVMCodeGenLevelNone',
    1: 'LLVMCodeGenLevelLess',
    2: 'LLVMCodeGenLevelDefault',
    3: 'LLVMCodeGenLevelAggressive',
}
LLVMCodeGenLevelNone = 0
LLVMCodeGenLevelLess = 1
LLVMCodeGenLevelDefault = 2
LLVMCodeGenLevelAggressive = 3
c__EA_LLVMCodeGenOptLevel = ctypes.c_uint32 # enum
LLVMCodeGenOptLevel = c__EA_LLVMCodeGenOptLevel
LLVMCodeGenOptLevel__enumvalues = c__EA_LLVMCodeGenOptLevel__enumvalues

# values for enumeration 'c__EA_LLVMRelocMode'
c__EA_LLVMRelocMode__enumvalues = {
    0: 'LLVMRelocDefault',
    1: 'LLVMRelocStatic',
    2: 'LLVMRelocPIC',
    3: 'LLVMRelocDynamicNoPic',
    4: 'LLVMRelocROPI',
    5: 'LLVMRelocRWPI',
    6: 'LLVMRelocROPI_RWPI',
}
LLVMRelocDefault = 0
LLVMRelocStatic = 1
LLVMRelocPIC = 2
LLVMRelocDynamicNoPic = 3
LLVMRelocROPI = 4
LLVMRelocRWPI = 5
LLVMRelocROPI_RWPI = 6
c__EA_LLVMRelocMode = ctypes.c_uint32 # enum
LLVMRelocMode = c__EA_LLVMRelocMode
LLVMRelocMode__enumvalues = c__EA_LLVMRelocMode__enumvalues

# values for enumeration 'c__EA_LLVMCodeModel'
c__EA_LLVMCodeModel__enumvalues = {
    0: 'LLVMCodeModelDefault',
    1: 'LLVMCodeModelJITDefault',
    2: 'LLVMCodeModelTiny',
    3: 'LLVMCodeModelSmall',
    4: 'LLVMCodeModelKernel',
    5: 'LLVMCodeModelMedium',
    6: 'LLVMCodeModelLarge',
}
LLVMCodeModelDefault = 0
LLVMCodeModelJITDefault = 1
LLVMCodeModelTiny = 2
LLVMCodeModelSmall = 3
LLVMCodeModelKernel = 4
LLVMCodeModelMedium = 5
LLVMCodeModelLarge = 6
c__EA_LLVMCodeModel = ctypes.c_uint32 # enum
LLVMCodeModel = c__EA_LLVMCodeModel
LLVMCodeModel__enumvalues = c__EA_LLVMCodeModel__enumvalues

# values for enumeration 'c__EA_LLVMCodeGenFileType'
c__EA_LLVMCodeGenFileType__enumvalues = {
    0: 'LLVMAssemblyFile',
    1: 'LLVMObjectFile',
}
LLVMAssemblyFile = 0
LLVMObjectFile = 1
c__EA_LLVMCodeGenFileType = ctypes.c_uint32 # enum
LLVMCodeGenFileType = c__EA_LLVMCodeGenFileType
LLVMCodeGenFileType__enumvalues = c__EA_LLVMCodeGenFileType__enumvalues
try:
    LLVMGetFirstTarget = _libraries['llvm'].LLVMGetFirstTarget
    LLVMGetFirstTarget.restype = LLVMTargetRef
    LLVMGetFirstTarget.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetNextTarget = _libraries['llvm'].LLVMGetNextTarget
    LLVMGetNextTarget.restype = LLVMTargetRef
    LLVMGetNextTarget.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMGetTargetFromName = _libraries['llvm'].LLVMGetTargetFromName
    LLVMGetTargetFromName.restype = LLVMTargetRef
    LLVMGetTargetFromName.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetTargetFromTriple = _libraries['llvm'].LLVMGetTargetFromTriple
    LLVMGetTargetFromTriple.restype = LLVMBool
    LLVMGetTargetFromTriple.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_LLVMTarget)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMGetTargetName = _libraries['llvm'].LLVMGetTargetName
    LLVMGetTargetName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTargetName.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMGetTargetDescription = _libraries['llvm'].LLVMGetTargetDescription
    LLVMGetTargetDescription.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTargetDescription.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMTargetHasJIT = _libraries['llvm'].LLVMTargetHasJIT
    LLVMTargetHasJIT.restype = LLVMBool
    LLVMTargetHasJIT.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMTargetHasTargetMachine = _libraries['llvm'].LLVMTargetHasTargetMachine
    LLVMTargetHasTargetMachine.restype = LLVMBool
    LLVMTargetHasTargetMachine.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMTargetHasAsmBackend = _libraries['llvm'].LLVMTargetHasAsmBackend
    LLVMTargetHasAsmBackend.restype = LLVMBool
    LLVMTargetHasAsmBackend.argtypes = [LLVMTargetRef]
except AttributeError:
    pass
try:
    LLVMCreateTargetMachine = _libraries['llvm'].LLVMCreateTargetMachine
    LLVMCreateTargetMachine.restype = LLVMTargetMachineRef
    LLVMCreateTargetMachine.argtypes = [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError:
    pass
try:
    LLVMDisposeTargetMachine = _libraries['llvm'].LLVMDisposeTargetMachine
    LLVMDisposeTargetMachine.restype = None
    LLVMDisposeTargetMachine.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMGetTargetMachineTarget = _libraries['llvm'].LLVMGetTargetMachineTarget
    LLVMGetTargetMachineTarget.restype = LLVMTargetRef
    LLVMGetTargetMachineTarget.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMGetTargetMachineTriple = _libraries['llvm'].LLVMGetTargetMachineTriple
    LLVMGetTargetMachineTriple.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTargetMachineTriple.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMGetTargetMachineCPU = _libraries['llvm'].LLVMGetTargetMachineCPU
    LLVMGetTargetMachineCPU.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTargetMachineCPU.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMGetTargetMachineFeatureString = _libraries['llvm'].LLVMGetTargetMachineFeatureString
    LLVMGetTargetMachineFeatureString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetTargetMachineFeatureString.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMCreateTargetDataLayout = _libraries['llvm'].LLVMCreateTargetDataLayout
    LLVMCreateTargetDataLayout.restype = LLVMTargetDataRef
    LLVMCreateTargetDataLayout.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMSetTargetMachineAsmVerbosity = _libraries['llvm'].LLVMSetTargetMachineAsmVerbosity
    LLVMSetTargetMachineAsmVerbosity.restype = None
    LLVMSetTargetMachineAsmVerbosity.argtypes = [LLVMTargetMachineRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMTargetMachineEmitToFile = _libraries['llvm'].LLVMTargetMachineEmitToFile
    LLVMTargetMachineEmitToFile.restype = LLVMBool
    LLVMTargetMachineEmitToFile.argtypes = [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMTargetMachineEmitToMemoryBuffer = _libraries['llvm'].LLVMTargetMachineEmitToMemoryBuffer
    LLVMTargetMachineEmitToMemoryBuffer.restype = LLVMBool
    LLVMTargetMachineEmitToMemoryBuffer.argtypes = [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer))]
except AttributeError:
    pass
try:
    LLVMGetDefaultTargetTriple = _libraries['llvm'].LLVMGetDefaultTargetTriple
    LLVMGetDefaultTargetTriple.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetDefaultTargetTriple.argtypes = []
except AttributeError:
    pass
try:
    LLVMNormalizeTargetTriple = _libraries['llvm'].LLVMNormalizeTargetTriple
    LLVMNormalizeTargetTriple.restype = ctypes.POINTER(ctypes.c_char)
    LLVMNormalizeTargetTriple.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetHostCPUName = _libraries['llvm'].LLVMGetHostCPUName
    LLVMGetHostCPUName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetHostCPUName.argtypes = []
except AttributeError:
    pass
try:
    LLVMGetHostCPUFeatures = _libraries['llvm'].LLVMGetHostCPUFeatures
    LLVMGetHostCPUFeatures.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetHostCPUFeatures.argtypes = []
except AttributeError:
    pass
try:
    LLVMAddAnalysisPasses = _libraries['llvm'].LLVMAddAnalysisPasses
    LLVMAddAnalysisPasses.restype = None
    LLVMAddAnalysisPasses.argtypes = [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMLinkInMCJIT = _libraries['llvm'].LLVMLinkInMCJIT
    LLVMLinkInMCJIT.restype = None
    LLVMLinkInMCJIT.argtypes = []
except AttributeError:
    pass
try:
    LLVMLinkInInterpreter = _libraries['llvm'].LLVMLinkInInterpreter
    LLVMLinkInInterpreter.restype = None
    LLVMLinkInInterpreter.argtypes = []
except AttributeError:
    pass
class struct_LLVMOpaqueGenericValue(Structure):
    pass

LLVMGenericValueRef = ctypes.POINTER(struct_LLVMOpaqueGenericValue)
class struct_LLVMOpaqueExecutionEngine(Structure):
    pass

LLVMExecutionEngineRef = ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)
class struct_LLVMOpaqueMCJITMemoryManager(Structure):
    pass

LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
class struct_LLVMMCJITCompilerOptions(Structure):
    pass

struct_LLVMMCJITCompilerOptions._pack_ = 1 # source:False
struct_LLVMMCJITCompilerOptions._fields_ = [
    ('OptLevel', ctypes.c_uint32),
    ('CodeModel', LLVMCodeModel),
    ('NoFramePointerElim', ctypes.c_int32),
    ('EnableFastISel', ctypes.c_int32),
    ('MCJMM', ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)),
]

try:
    LLVMCreateGenericValueOfInt = _libraries['llvm'].LLVMCreateGenericValueOfInt
    LLVMCreateGenericValueOfInt.restype = LLVMGenericValueRef
    LLVMCreateGenericValueOfInt.argtypes = [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError:
    pass
try:
    LLVMCreateGenericValueOfPointer = _libraries['llvm'].LLVMCreateGenericValueOfPointer
    LLVMCreateGenericValueOfPointer.restype = LLVMGenericValueRef
    LLVMCreateGenericValueOfPointer.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMCreateGenericValueOfFloat = _libraries['llvm'].LLVMCreateGenericValueOfFloat
    LLVMCreateGenericValueOfFloat.restype = LLVMGenericValueRef
    LLVMCreateGenericValueOfFloat.argtypes = [LLVMTypeRef, ctypes.c_double]
except AttributeError:
    pass
try:
    LLVMGenericValueIntWidth = _libraries['llvm'].LLVMGenericValueIntWidth
    LLVMGenericValueIntWidth.restype = ctypes.c_uint32
    LLVMGenericValueIntWidth.argtypes = [LLVMGenericValueRef]
except AttributeError:
    pass
try:
    LLVMGenericValueToInt = _libraries['llvm'].LLVMGenericValueToInt
    LLVMGenericValueToInt.restype = ctypes.c_uint64
    LLVMGenericValueToInt.argtypes = [LLVMGenericValueRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMGenericValueToPointer = _libraries['llvm'].LLVMGenericValueToPointer
    LLVMGenericValueToPointer.restype = ctypes.POINTER(None)
    LLVMGenericValueToPointer.argtypes = [LLVMGenericValueRef]
except AttributeError:
    pass
try:
    LLVMGenericValueToFloat = _libraries['llvm'].LLVMGenericValueToFloat
    LLVMGenericValueToFloat.restype = ctypes.c_double
    LLVMGenericValueToFloat.argtypes = [LLVMTypeRef, LLVMGenericValueRef]
except AttributeError:
    pass
try:
    LLVMDisposeGenericValue = _libraries['llvm'].LLVMDisposeGenericValue
    LLVMDisposeGenericValue.restype = None
    LLVMDisposeGenericValue.argtypes = [LLVMGenericValueRef]
except AttributeError:
    pass
try:
    LLVMCreateExecutionEngineForModule = _libraries['llvm'].LLVMCreateExecutionEngineForModule
    LLVMCreateExecutionEngineForModule.restype = LLVMBool
    LLVMCreateExecutionEngineForModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMCreateInterpreterForModule = _libraries['llvm'].LLVMCreateInterpreterForModule
    LLVMCreateInterpreterForModule.restype = LLVMBool
    LLVMCreateInterpreterForModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMCreateJITCompilerForModule = _libraries['llvm'].LLVMCreateJITCompilerForModule
    LLVMCreateJITCompilerForModule.restype = LLVMBool
    LLVMCreateJITCompilerForModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)), LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMInitializeMCJITCompilerOptions = _libraries['llvm'].LLVMInitializeMCJITCompilerOptions
    LLVMInitializeMCJITCompilerOptions.restype = None
    LLVMInitializeMCJITCompilerOptions.argtypes = [ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t]
except AttributeError:
    pass
try:
    LLVMCreateMCJITCompilerForModule = _libraries['llvm'].LLVMCreateMCJITCompilerForModule
    LLVMCreateMCJITCompilerForModule.restype = LLVMBool
    LLVMCreateMCJITCompilerForModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)), LLVMModuleRef, ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMDisposeExecutionEngine = _libraries['llvm'].LLVMDisposeExecutionEngine
    LLVMDisposeExecutionEngine.restype = None
    LLVMDisposeExecutionEngine.argtypes = [LLVMExecutionEngineRef]
except AttributeError:
    pass
try:
    LLVMRunStaticConstructors = _libraries['llvm'].LLVMRunStaticConstructors
    LLVMRunStaticConstructors.restype = None
    LLVMRunStaticConstructors.argtypes = [LLVMExecutionEngineRef]
except AttributeError:
    pass
try:
    LLVMRunStaticDestructors = _libraries['llvm'].LLVMRunStaticDestructors
    LLVMRunStaticDestructors.restype = None
    LLVMRunStaticDestructors.argtypes = [LLVMExecutionEngineRef]
except AttributeError:
    pass
try:
    LLVMRunFunctionAsMain = _libraries['llvm'].LLVMRunFunctionAsMain
    LLVMRunFunctionAsMain.restype = ctypes.c_int32
    LLVMRunFunctionAsMain.argtypes = [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMRunFunction = _libraries['llvm'].LLVMRunFunction
    LLVMRunFunction.restype = LLVMGenericValueRef
    LLVMRunFunction.argtypes = [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueGenericValue))]
except AttributeError:
    pass
try:
    LLVMFreeMachineCodeForFunction = _libraries['llvm'].LLVMFreeMachineCodeForFunction
    LLVMFreeMachineCodeForFunction.restype = None
    LLVMFreeMachineCodeForFunction.argtypes = [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMAddModule = _libraries['llvm'].LLVMAddModule
    LLVMAddModule.restype = None
    LLVMAddModule.argtypes = [LLVMExecutionEngineRef, LLVMModuleRef]
except AttributeError:
    pass
try:
    LLVMRemoveModule = _libraries['llvm'].LLVMRemoveModule
    LLVMRemoveModule.restype = LLVMBool
    LLVMRemoveModule.argtypes = [LLVMExecutionEngineRef, LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMFindFunction = _libraries['llvm'].LLVMFindFunction
    LLVMFindFunction.restype = LLVMBool
    LLVMFindFunction.argtypes = [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))]
except AttributeError:
    pass
try:
    LLVMRecompileAndRelinkFunction = _libraries['llvm'].LLVMRecompileAndRelinkFunction
    LLVMRecompileAndRelinkFunction.restype = ctypes.POINTER(None)
    LLVMRecompileAndRelinkFunction.argtypes = [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetExecutionEngineTargetData = _libraries['llvm'].LLVMGetExecutionEngineTargetData
    LLVMGetExecutionEngineTargetData.restype = LLVMTargetDataRef
    LLVMGetExecutionEngineTargetData.argtypes = [LLVMExecutionEngineRef]
except AttributeError:
    pass
try:
    LLVMGetExecutionEngineTargetMachine = _libraries['llvm'].LLVMGetExecutionEngineTargetMachine
    LLVMGetExecutionEngineTargetMachine.restype = LLVMTargetMachineRef
    LLVMGetExecutionEngineTargetMachine.argtypes = [LLVMExecutionEngineRef]
except AttributeError:
    pass
try:
    LLVMAddGlobalMapping = _libraries['llvm'].LLVMAddGlobalMapping
    LLVMAddGlobalMapping.restype = None
    LLVMAddGlobalMapping.argtypes = [LLVMExecutionEngineRef, LLVMValueRef, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMGetPointerToGlobal = _libraries['llvm'].LLVMGetPointerToGlobal
    LLVMGetPointerToGlobal.restype = ctypes.POINTER(None)
    LLVMGetPointerToGlobal.argtypes = [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError:
    pass
try:
    LLVMGetGlobalValueAddress = _libraries['llvm'].LLVMGetGlobalValueAddress
    LLVMGetGlobalValueAddress.restype = uint64_t
    LLVMGetGlobalValueAddress.argtypes = [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMGetFunctionAddress = _libraries['llvm'].LLVMGetFunctionAddress
    LLVMGetFunctionAddress.restype = uint64_t
    LLVMGetFunctionAddress.argtypes = [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMExecutionEngineGetErrMsg = _libraries['llvm'].LLVMExecutionEngineGetErrMsg
    LLVMExecutionEngineGetErrMsg.restype = LLVMBool
    LLVMExecutionEngineGetErrMsg.argtypes = [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
LLVMMemoryManagerAllocateCodeSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char))
LLVMMemoryManagerAllocateDataSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32)
LLVMMemoryManagerFinalizeMemoryCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
LLVMMemoryManagerDestroyCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))
try:
    LLVMCreateSimpleMCJITMemoryManager = _libraries['llvm'].LLVMCreateSimpleMCJITMemoryManager
    LLVMCreateSimpleMCJITMemoryManager.restype = LLVMMCJITMemoryManagerRef
    LLVMCreateSimpleMCJITMemoryManager.argtypes = [ctypes.POINTER(None), LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError:
    pass
try:
    LLVMDisposeMCJITMemoryManager = _libraries['llvm'].LLVMDisposeMCJITMemoryManager
    LLVMDisposeMCJITMemoryManager.restype = None
    LLVMDisposeMCJITMemoryManager.argtypes = [LLVMMCJITMemoryManagerRef]
except AttributeError:
    pass
try:
    LLVMCreateGDBRegistrationListener = _libraries['llvm'].LLVMCreateGDBRegistrationListener
    LLVMCreateGDBRegistrationListener.restype = LLVMJITEventListenerRef
    LLVMCreateGDBRegistrationListener.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreateIntelJITEventListener = _libraries['llvm'].LLVMCreateIntelJITEventListener
    LLVMCreateIntelJITEventListener.restype = LLVMJITEventListenerRef
    LLVMCreateIntelJITEventListener.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreateOProfileJITEventListener = _libraries['llvm'].LLVMCreateOProfileJITEventListener
    LLVMCreateOProfileJITEventListener.restype = LLVMJITEventListenerRef
    LLVMCreateOProfileJITEventListener.argtypes = []
except AttributeError:
    pass
try:
    LLVMCreatePerfJITEventListener = _libraries['llvm'].LLVMCreatePerfJITEventListener
    LLVMCreatePerfJITEventListener.restype = LLVMJITEventListenerRef
    LLVMCreatePerfJITEventListener.argtypes = []
except AttributeError:
    pass
LLVM_C_IRREADER_H = True # macro
try:
    LLVMParseIRInContext = _libraries['llvm'].LLVMParseIRInContext
    LLVMParseIRInContext.restype = LLVMBool
    LLVMParseIRInContext.argtypes = [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueModule)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
LLVM_C_INITIALIZATION_H = True # macro
try:
    LLVMInitializeTransformUtils = _libraries['llvm'].LLVMInitializeTransformUtils
    LLVMInitializeTransformUtils.restype = None
    LLVMInitializeTransformUtils.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeScalarOpts = _libraries['llvm'].LLVMInitializeScalarOpts
    LLVMInitializeScalarOpts.restype = None
    LLVMInitializeScalarOpts.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeObjCARCOpts = _libraries['llvm'].LLVMInitializeObjCARCOpts
    LLVMInitializeObjCARCOpts.restype = None
    LLVMInitializeObjCARCOpts.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeVectorization = _libraries['llvm'].LLVMInitializeVectorization
    LLVMInitializeVectorization.restype = None
    LLVMInitializeVectorization.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeInstCombine = _libraries['llvm'].LLVMInitializeInstCombine
    LLVMInitializeInstCombine.restype = None
    LLVMInitializeInstCombine.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeAggressiveInstCombiner = _libraries['llvm'].LLVMInitializeAggressiveInstCombiner
    LLVMInitializeAggressiveInstCombiner.restype = None
    LLVMInitializeAggressiveInstCombiner.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeIPO = _libraries['llvm'].LLVMInitializeIPO
    LLVMInitializeIPO.restype = None
    LLVMInitializeIPO.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeInstrumentation = _libraries['llvm'].LLVMInitializeInstrumentation
    LLVMInitializeInstrumentation.restype = None
    LLVMInitializeInstrumentation.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeAnalysis = _libraries['llvm'].LLVMInitializeAnalysis
    LLVMInitializeAnalysis.restype = None
    LLVMInitializeAnalysis.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeIPA = _libraries['llvm'].LLVMInitializeIPA
    LLVMInitializeIPA.restype = None
    LLVMInitializeIPA.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeCodeGen = _libraries['llvm'].LLVMInitializeCodeGen
    LLVMInitializeCodeGen.restype = None
    LLVMInitializeCodeGen.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
try:
    LLVMInitializeTarget = _libraries['llvm'].LLVMInitializeTarget
    LLVMInitializeTarget.restype = None
    LLVMInitializeTarget.argtypes = [LLVMPassRegistryRef]
except AttributeError:
    pass
LLVM_C_LLJIT_H = True # macro
LLVM_C_ORC_H = True # macro
LLVMOrcJITTargetAddress = ctypes.c_uint64
LLVMOrcExecutorAddress = ctypes.c_uint64

# values for enumeration 'c__EA_LLVMJITSymbolGenericFlags'
c__EA_LLVMJITSymbolGenericFlags__enumvalues = {
    1: 'LLVMJITSymbolGenericFlagsExported',
    2: 'LLVMJITSymbolGenericFlagsWeak',
    4: 'LLVMJITSymbolGenericFlagsCallable',
    8: 'LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly',
}
LLVMJITSymbolGenericFlagsExported = 1
LLVMJITSymbolGenericFlagsWeak = 2
LLVMJITSymbolGenericFlagsCallable = 4
LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly = 8
c__EA_LLVMJITSymbolGenericFlags = ctypes.c_uint32 # enum
LLVMJITSymbolGenericFlags = c__EA_LLVMJITSymbolGenericFlags
LLVMJITSymbolGenericFlags__enumvalues = c__EA_LLVMJITSymbolGenericFlags__enumvalues
LLVMJITSymbolTargetFlags = ctypes.c_ubyte
class struct_c__SA_LLVMJITSymbolFlags(Structure):
    pass

struct_c__SA_LLVMJITSymbolFlags._pack_ = 1 # source:False
struct_c__SA_LLVMJITSymbolFlags._fields_ = [
    ('GenericFlags', ctypes.c_ubyte),
    ('TargetFlags', ctypes.c_ubyte),
]

LLVMJITSymbolFlags = struct_c__SA_LLVMJITSymbolFlags
class struct_c__SA_LLVMJITEvaluatedSymbol(Structure):
    pass

struct_c__SA_LLVMJITEvaluatedSymbol._pack_ = 1 # source:False
struct_c__SA_LLVMJITEvaluatedSymbol._fields_ = [
    ('Address', ctypes.c_uint64),
    ('Flags', LLVMJITSymbolFlags),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

LLVMJITEvaluatedSymbol = struct_c__SA_LLVMJITEvaluatedSymbol
class struct_LLVMOrcOpaqueExecutionSession(Structure):
    pass

LLVMOrcExecutionSessionRef = ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession)
LLVMOrcErrorReporterFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOpaqueError))
class struct_LLVMOrcOpaqueSymbolStringPool(Structure):
    pass

LLVMOrcSymbolStringPoolRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPool)
class struct_LLVMOrcOpaqueSymbolStringPoolEntry(Structure):
    pass

LLVMOrcSymbolStringPoolEntryRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)
class struct_c__SA_LLVMOrcCSymbolFlagsMapPair(Structure):
    pass

struct_c__SA_LLVMOrcCSymbolFlagsMapPair._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCSymbolFlagsMapPair._fields_ = [
    ('Name', ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)),
    ('Flags', LLVMJITSymbolFlags),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

LLVMOrcCSymbolFlagsMapPair = struct_c__SA_LLVMOrcCSymbolFlagsMapPair
LLVMOrcCSymbolFlagsMapPairs = ctypes.POINTER(struct_c__SA_LLVMOrcCSymbolFlagsMapPair)
class struct_c__SA_LLVMJITCSymbolMapPair(Structure):
    pass

struct_c__SA_LLVMJITCSymbolMapPair._pack_ = 1 # source:False
struct_c__SA_LLVMJITCSymbolMapPair._fields_ = [
    ('Name', ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)),
    ('Sym', LLVMJITEvaluatedSymbol),
]

LLVMJITCSymbolMapPair = struct_c__SA_LLVMJITCSymbolMapPair
LLVMOrcCSymbolMapPairs = ctypes.POINTER(struct_c__SA_LLVMJITCSymbolMapPair)
class struct_c__SA_LLVMOrcCSymbolAliasMapEntry(Structure):
    pass

struct_c__SA_LLVMOrcCSymbolAliasMapEntry._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCSymbolAliasMapEntry._fields_ = [
    ('Name', ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)),
    ('Flags', LLVMJITSymbolFlags),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

LLVMOrcCSymbolAliasMapEntry = struct_c__SA_LLVMOrcCSymbolAliasMapEntry
class struct_c__SA_LLVMOrcCSymbolAliasMapPair(Structure):
    pass

struct_c__SA_LLVMOrcCSymbolAliasMapPair._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCSymbolAliasMapPair._fields_ = [
    ('Name', ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)),
    ('Entry', LLVMOrcCSymbolAliasMapEntry),
]

LLVMOrcCSymbolAliasMapPair = struct_c__SA_LLVMOrcCSymbolAliasMapPair
LLVMOrcCSymbolAliasMapPairs = ctypes.POINTER(struct_c__SA_LLVMOrcCSymbolAliasMapPair)
class struct_LLVMOrcOpaqueJITDylib(Structure):
    pass

LLVMOrcJITDylibRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib)
class struct_c__SA_LLVMOrcCSymbolsList(Structure):
    pass

struct_c__SA_LLVMOrcCSymbolsList._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCSymbolsList._fields_ = [
    ('Symbols', ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))),
    ('Length', ctypes.c_uint64),
]

LLVMOrcCSymbolsList = struct_c__SA_LLVMOrcCSymbolsList
class struct_c__SA_LLVMOrcCDependenceMapPair(Structure):
    pass

struct_c__SA_LLVMOrcCDependenceMapPair._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCDependenceMapPair._fields_ = [
    ('JD', ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib)),
    ('Names', LLVMOrcCSymbolsList),
]

LLVMOrcCDependenceMapPair = struct_c__SA_LLVMOrcCDependenceMapPair
LLVMOrcCDependenceMapPairs = ctypes.POINTER(struct_c__SA_LLVMOrcCDependenceMapPair)

# values for enumeration 'c__EA_LLVMOrcLookupKind'
c__EA_LLVMOrcLookupKind__enumvalues = {
    0: 'LLVMOrcLookupKindStatic',
    1: 'LLVMOrcLookupKindDLSym',
}
LLVMOrcLookupKindStatic = 0
LLVMOrcLookupKindDLSym = 1
c__EA_LLVMOrcLookupKind = ctypes.c_uint32 # enum
LLVMOrcLookupKind = c__EA_LLVMOrcLookupKind
LLVMOrcLookupKind__enumvalues = c__EA_LLVMOrcLookupKind__enumvalues

# values for enumeration 'c__EA_LLVMOrcJITDylibLookupFlags'
c__EA_LLVMOrcJITDylibLookupFlags__enumvalues = {
    0: 'LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly',
    1: 'LLVMOrcJITDylibLookupFlagsMatchAllSymbols',
}
LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly = 0
LLVMOrcJITDylibLookupFlagsMatchAllSymbols = 1
c__EA_LLVMOrcJITDylibLookupFlags = ctypes.c_uint32 # enum
LLVMOrcJITDylibLookupFlags = c__EA_LLVMOrcJITDylibLookupFlags
LLVMOrcJITDylibLookupFlags__enumvalues = c__EA_LLVMOrcJITDylibLookupFlags__enumvalues

# values for enumeration 'c__EA_LLVMOrcSymbolLookupFlags'
c__EA_LLVMOrcSymbolLookupFlags__enumvalues = {
    0: 'LLVMOrcSymbolLookupFlagsRequiredSymbol',
    1: 'LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol',
}
LLVMOrcSymbolLookupFlagsRequiredSymbol = 0
LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol = 1
c__EA_LLVMOrcSymbolLookupFlags = ctypes.c_uint32 # enum
LLVMOrcSymbolLookupFlags = c__EA_LLVMOrcSymbolLookupFlags
LLVMOrcSymbolLookupFlags__enumvalues = c__EA_LLVMOrcSymbolLookupFlags__enumvalues
class struct_c__SA_LLVMOrcCLookupSetElement(Structure):
    pass

struct_c__SA_LLVMOrcCLookupSetElement._pack_ = 1 # source:False
struct_c__SA_LLVMOrcCLookupSetElement._fields_ = [
    ('Name', ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)),
    ('LookupFlags', LLVMOrcSymbolLookupFlags),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

LLVMOrcCLookupSetElement = struct_c__SA_LLVMOrcCLookupSetElement
LLVMOrcCLookupSet = ctypes.POINTER(struct_c__SA_LLVMOrcCLookupSetElement)
class struct_LLVMOrcOpaqueMaterializationUnit(Structure):
    pass

LLVMOrcMaterializationUnitRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationUnit)
class struct_LLVMOrcOpaqueMaterializationResponsibility(Structure):
    pass

LLVMOrcMaterializationResponsibilityRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility)
LLVMOrcMaterializationUnitMaterializeFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
LLVMOrcMaterializationUnitDiscardFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
LLVMOrcMaterializationUnitDestroyFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))
class struct_LLVMOrcOpaqueResourceTracker(Structure):
    pass

LLVMOrcResourceTrackerRef = ctypes.POINTER(struct_LLVMOrcOpaqueResourceTracker)
class struct_LLVMOrcOpaqueDefinitionGenerator(Structure):
    pass

LLVMOrcDefinitionGeneratorRef = ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)
class struct_LLVMOrcOpaqueLookupState(Structure):
    pass

LLVMOrcLookupStateRef = ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)
LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator), ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)), c__EA_LLVMOrcLookupKind, ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), c__EA_LLVMOrcJITDylibLookupFlags, ctypes.POINTER(struct_c__SA_LLVMOrcCLookupSetElement), ctypes.c_uint64)
LLVMOrcSymbolPredicate = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
class struct_LLVMOrcOpaqueThreadSafeContext(Structure):
    pass

LLVMOrcThreadSafeContextRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeContext)
class struct_LLVMOrcOpaqueThreadSafeModule(Structure):
    pass

LLVMOrcThreadSafeModuleRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)
LLVMOrcGenericIRModuleOperationFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOpaqueModule))
class struct_LLVMOrcOpaqueJITTargetMachineBuilder(Structure):
    pass

LLVMOrcJITTargetMachineBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITTargetMachineBuilder)
class struct_LLVMOrcOpaqueObjectLayer(Structure):
    pass

LLVMOrcObjectLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer)
class struct_LLVMOrcOpaqueObjectLinkingLayer(Structure):
    pass

LLVMOrcObjectLinkingLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLinkingLayer)
class struct_LLVMOrcOpaqueIRTransformLayer(Structure):
    pass

LLVMOrcIRTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIRTransformLayer)
LLVMOrcIRTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)), ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
class struct_LLVMOrcOpaqueObjectTransformLayer(Structure):
    pass

LLVMOrcObjectTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectTransformLayer)
LLVMOrcObjectTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)))
class struct_LLVMOrcOpaqueIndirectStubsManager(Structure):
    pass

LLVMOrcIndirectStubsManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIndirectStubsManager)
class struct_LLVMOrcOpaqueLazyCallThroughManager(Structure):
    pass

LLVMOrcLazyCallThroughManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueLazyCallThroughManager)
class struct_LLVMOrcOpaqueDumpObjects(Structure):
    pass

LLVMOrcDumpObjectsRef = ctypes.POINTER(struct_LLVMOrcOpaqueDumpObjects)
try:
    LLVMOrcExecutionSessionSetErrorReporter = _libraries['llvm'].LLVMOrcExecutionSessionSetErrorReporter
    LLVMOrcExecutionSessionSetErrorReporter.restype = None
    LLVMOrcExecutionSessionSetErrorReporter.argtypes = [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcExecutionSessionGetSymbolStringPool = _libraries['llvm'].LLVMOrcExecutionSessionGetSymbolStringPool
    LLVMOrcExecutionSessionGetSymbolStringPool.restype = LLVMOrcSymbolStringPoolRef
    LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = [LLVMOrcExecutionSessionRef]
except AttributeError:
    pass
try:
    LLVMOrcSymbolStringPoolClearDeadEntries = _libraries['llvm'].LLVMOrcSymbolStringPoolClearDeadEntries
    LLVMOrcSymbolStringPoolClearDeadEntries.restype = None
    LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = [LLVMOrcSymbolStringPoolRef]
except AttributeError:
    pass
try:
    LLVMOrcExecutionSessionIntern = _libraries['llvm'].LLVMOrcExecutionSessionIntern
    LLVMOrcExecutionSessionIntern.restype = LLVMOrcSymbolStringPoolEntryRef
    LLVMOrcExecutionSessionIntern.argtypes = [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcRetainSymbolStringPoolEntry = _libraries['llvm'].LLVMOrcRetainSymbolStringPoolEntry
    LLVMOrcRetainSymbolStringPoolEntry.restype = None
    LLVMOrcRetainSymbolStringPoolEntry.argtypes = [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError:
    pass
try:
    LLVMOrcReleaseSymbolStringPoolEntry = _libraries['llvm'].LLVMOrcReleaseSymbolStringPoolEntry
    LLVMOrcReleaseSymbolStringPoolEntry.restype = None
    LLVMOrcReleaseSymbolStringPoolEntry.argtypes = [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError:
    pass
try:
    LLVMOrcSymbolStringPoolEntryStr = _libraries['llvm'].LLVMOrcSymbolStringPoolEntryStr
    LLVMOrcSymbolStringPoolEntryStr.restype = ctypes.POINTER(ctypes.c_char)
    LLVMOrcSymbolStringPoolEntryStr.argtypes = [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError:
    pass
try:
    LLVMOrcReleaseResourceTracker = _libraries['llvm'].LLVMOrcReleaseResourceTracker
    LLVMOrcReleaseResourceTracker.restype = None
    LLVMOrcReleaseResourceTracker.argtypes = [LLVMOrcResourceTrackerRef]
except AttributeError:
    pass
try:
    LLVMOrcResourceTrackerTransferTo = _libraries['llvm'].LLVMOrcResourceTrackerTransferTo
    LLVMOrcResourceTrackerTransferTo.restype = None
    LLVMOrcResourceTrackerTransferTo.argtypes = [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError:
    pass
try:
    LLVMOrcResourceTrackerRemove = _libraries['llvm'].LLVMOrcResourceTrackerRemove
    LLVMOrcResourceTrackerRemove.restype = LLVMErrorRef
    LLVMOrcResourceTrackerRemove.argtypes = [LLVMOrcResourceTrackerRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeDefinitionGenerator = _libraries['llvm'].LLVMOrcDisposeDefinitionGenerator
    LLVMOrcDisposeDefinitionGenerator.restype = None
    LLVMOrcDisposeDefinitionGenerator.argtypes = [LLVMOrcDefinitionGeneratorRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeMaterializationUnit = _libraries['llvm'].LLVMOrcDisposeMaterializationUnit
    LLVMOrcDisposeMaterializationUnit.restype = None
    LLVMOrcDisposeMaterializationUnit.argtypes = [LLVMOrcMaterializationUnitRef]
except AttributeError:
    pass
try:
    LLVMOrcCreateCustomMaterializationUnit = _libraries['llvm'].LLVMOrcCreateCustomMaterializationUnit
    LLVMOrcCreateCustomMaterializationUnit.restype = LLVMOrcMaterializationUnitRef
    LLVMOrcCreateCustomMaterializationUnit.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError:
    pass
try:
    LLVMOrcAbsoluteSymbols = _libraries['llvm'].LLVMOrcAbsoluteSymbols
    LLVMOrcAbsoluteSymbols.restype = LLVMOrcMaterializationUnitRef
    LLVMOrcAbsoluteSymbols.argtypes = [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcLazyReexports = _libraries['llvm'].LLVMOrcLazyReexports
    LLVMOrcLazyReexports.restype = LLVMOrcMaterializationUnitRef
    LLVMOrcLazyReexports.argtypes = [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcDisposeMaterializationResponsibility = _libraries['llvm'].LLVMOrcDisposeMaterializationResponsibility
    LLVMOrcDisposeMaterializationResponsibility.restype = None
    LLVMOrcDisposeMaterializationResponsibility.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityGetTargetDylib = _libraries['llvm'].LLVMOrcMaterializationResponsibilityGetTargetDylib
    LLVMOrcMaterializationResponsibilityGetTargetDylib.restype = LLVMOrcJITDylibRef
    LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityGetExecutionSession = _libraries['llvm'].LLVMOrcMaterializationResponsibilityGetExecutionSession
    LLVMOrcMaterializationResponsibilityGetExecutionSession.restype = LLVMOrcExecutionSessionRef
    LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityGetSymbols = _libraries['llvm'].LLVMOrcMaterializationResponsibilityGetSymbols
    LLVMOrcMaterializationResponsibilityGetSymbols.restype = LLVMOrcCSymbolFlagsMapPairs
    LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMOrcDisposeCSymbolFlagsMap = _libraries['llvm'].LLVMOrcDisposeCSymbolFlagsMap
    LLVMOrcDisposeCSymbolFlagsMap.restype = None
    LLVMOrcDisposeCSymbolFlagsMap.argtypes = [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityGetInitializerSymbol = _libraries['llvm'].LLVMOrcMaterializationResponsibilityGetInitializerSymbol
    LLVMOrcMaterializationResponsibilityGetInitializerSymbol.restype = LLVMOrcSymbolStringPoolEntryRef
    LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityGetRequestedSymbols = _libraries['llvm'].LLVMOrcMaterializationResponsibilityGetRequestedSymbols
    LLVMOrcMaterializationResponsibilityGetRequestedSymbols.restype = ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
    LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    LLVMOrcDisposeSymbols = _libraries['llvm'].LLVMOrcDisposeSymbols
    LLVMOrcDisposeSymbols.restype = None
    LLVMOrcDisposeSymbols.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityNotifyResolved = _libraries['llvm'].LLVMOrcMaterializationResponsibilityNotifyResolved
    LLVMOrcMaterializationResponsibilityNotifyResolved.restype = LLVMErrorRef
    LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityNotifyEmitted = _libraries['llvm'].LLVMOrcMaterializationResponsibilityNotifyEmitted
    LLVMOrcMaterializationResponsibilityNotifyEmitted.restype = LLVMErrorRef
    LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityDefineMaterializing = _libraries['llvm'].LLVMOrcMaterializationResponsibilityDefineMaterializing
    LLVMOrcMaterializationResponsibilityDefineMaterializing.restype = LLVMErrorRef
    LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityFailMaterialization = _libraries['llvm'].LLVMOrcMaterializationResponsibilityFailMaterialization
    LLVMOrcMaterializationResponsibilityFailMaterialization.restype = None
    LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = [LLVMOrcMaterializationResponsibilityRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityReplace = _libraries['llvm'].LLVMOrcMaterializationResponsibilityReplace
    LLVMOrcMaterializationResponsibilityReplace.restype = LLVMErrorRef
    LLVMOrcMaterializationResponsibilityReplace.argtypes = [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityDelegate = _libraries['llvm'].LLVMOrcMaterializationResponsibilityDelegate
    LLVMOrcMaterializationResponsibilityDelegate.restype = LLVMErrorRef
    LLVMOrcMaterializationResponsibilityDelegate.argtypes = [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)), size_t, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityAddDependencies = _libraries['llvm'].LLVMOrcMaterializationResponsibilityAddDependencies
    LLVMOrcMaterializationResponsibilityAddDependencies.restype = None
    LLVMOrcMaterializationResponsibilityAddDependencies.argtypes = [LLVMOrcMaterializationResponsibilityRef, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcCDependenceMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcMaterializationResponsibilityAddDependenciesForAll = _libraries['llvm'].LLVMOrcMaterializationResponsibilityAddDependenciesForAll
    LLVMOrcMaterializationResponsibilityAddDependenciesForAll.restype = None
    LLVMOrcMaterializationResponsibilityAddDependenciesForAll.argtypes = [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCDependenceMapPairs, size_t]
except AttributeError:
    pass
try:
    LLVMOrcExecutionSessionCreateBareJITDylib = _libraries['llvm'].LLVMOrcExecutionSessionCreateBareJITDylib
    LLVMOrcExecutionSessionCreateBareJITDylib.restype = LLVMOrcJITDylibRef
    LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcExecutionSessionCreateJITDylib = _libraries['llvm'].LLVMOrcExecutionSessionCreateJITDylib
    LLVMOrcExecutionSessionCreateJITDylib.restype = LLVMErrorRef
    LLVMOrcExecutionSessionCreateJITDylib.argtypes = [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcExecutionSessionGetJITDylibByName = _libraries['llvm'].LLVMOrcExecutionSessionGetJITDylibByName
    LLVMOrcExecutionSessionGetJITDylibByName.restype = LLVMOrcJITDylibRef
    LLVMOrcExecutionSessionGetJITDylibByName.argtypes = [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcJITDylibCreateResourceTracker = _libraries['llvm'].LLVMOrcJITDylibCreateResourceTracker
    LLVMOrcJITDylibCreateResourceTracker.restype = LLVMOrcResourceTrackerRef
    LLVMOrcJITDylibCreateResourceTracker.argtypes = [LLVMOrcJITDylibRef]
except AttributeError:
    pass
try:
    LLVMOrcJITDylibGetDefaultResourceTracker = _libraries['llvm'].LLVMOrcJITDylibGetDefaultResourceTracker
    LLVMOrcJITDylibGetDefaultResourceTracker.restype = LLVMOrcResourceTrackerRef
    LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = [LLVMOrcJITDylibRef]
except AttributeError:
    pass
try:
    LLVMOrcJITDylibDefine = _libraries['llvm'].LLVMOrcJITDylibDefine
    LLVMOrcJITDylibDefine.restype = LLVMErrorRef
    LLVMOrcJITDylibDefine.argtypes = [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError:
    pass
try:
    LLVMOrcJITDylibClear = _libraries['llvm'].LLVMOrcJITDylibClear
    LLVMOrcJITDylibClear.restype = LLVMErrorRef
    LLVMOrcJITDylibClear.argtypes = [LLVMOrcJITDylibRef]
except AttributeError:
    pass
try:
    LLVMOrcJITDylibAddGenerator = _libraries['llvm'].LLVMOrcJITDylibAddGenerator
    LLVMOrcJITDylibAddGenerator.restype = None
    LLVMOrcJITDylibAddGenerator.argtypes = [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError:
    pass
try:
    LLVMOrcCreateCustomCAPIDefinitionGenerator = _libraries['llvm'].LLVMOrcCreateCustomCAPIDefinitionGenerator
    LLVMOrcCreateCustomCAPIDefinitionGenerator.restype = LLVMOrcDefinitionGeneratorRef
    LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess = _libraries['llvm'].LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.restype = LLVMErrorRef
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcCreateDynamicLibrarySearchGeneratorForPath = _libraries['llvm'].LLVMOrcCreateDynamicLibrarySearchGeneratorForPath
    LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.restype = LLVMErrorRef
    LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcCreateStaticLibrarySearchGeneratorForPath = _libraries['llvm'].LLVMOrcCreateStaticLibrarySearchGeneratorForPath
    LLVMOrcCreateStaticLibrarySearchGeneratorForPath.restype = LLVMErrorRef
    LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcCreateNewThreadSafeContext = _libraries['llvm'].LLVMOrcCreateNewThreadSafeContext
    LLVMOrcCreateNewThreadSafeContext.restype = LLVMOrcThreadSafeContextRef
    LLVMOrcCreateNewThreadSafeContext.argtypes = []
except AttributeError:
    pass
try:
    LLVMOrcThreadSafeContextGetContext = _libraries['llvm'].LLVMOrcThreadSafeContextGetContext
    LLVMOrcThreadSafeContextGetContext.restype = LLVMContextRef
    LLVMOrcThreadSafeContextGetContext.argtypes = [LLVMOrcThreadSafeContextRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeThreadSafeContext = _libraries['llvm'].LLVMOrcDisposeThreadSafeContext
    LLVMOrcDisposeThreadSafeContext.restype = None
    LLVMOrcDisposeThreadSafeContext.argtypes = [LLVMOrcThreadSafeContextRef]
except AttributeError:
    pass
try:
    LLVMOrcCreateNewThreadSafeModule = _libraries['llvm'].LLVMOrcCreateNewThreadSafeModule
    LLVMOrcCreateNewThreadSafeModule.restype = LLVMOrcThreadSafeModuleRef
    LLVMOrcCreateNewThreadSafeModule.argtypes = [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeThreadSafeModule = _libraries['llvm'].LLVMOrcDisposeThreadSafeModule
    LLVMOrcDisposeThreadSafeModule.restype = None
    LLVMOrcDisposeThreadSafeModule.argtypes = [LLVMOrcThreadSafeModuleRef]
except AttributeError:
    pass
try:
    LLVMOrcThreadSafeModuleWithModuleDo = _libraries['llvm'].LLVMOrcThreadSafeModuleWithModuleDo
    LLVMOrcThreadSafeModuleWithModuleDo.restype = LLVMErrorRef
    LLVMOrcThreadSafeModuleWithModuleDo.argtypes = [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcJITTargetMachineBuilderDetectHost = _libraries['llvm'].LLVMOrcJITTargetMachineBuilderDetectHost
    LLVMOrcJITTargetMachineBuilderDetectHost.restype = LLVMErrorRef
    LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueJITTargetMachineBuilder))]
except AttributeError:
    pass
try:
    LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine = _libraries['llvm'].LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine
    LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.restype = LLVMOrcJITTargetMachineBuilderRef
    LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = [LLVMTargetMachineRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeJITTargetMachineBuilder = _libraries['llvm'].LLVMOrcDisposeJITTargetMachineBuilder
    LLVMOrcDisposeJITTargetMachineBuilder.restype = None
    LLVMOrcDisposeJITTargetMachineBuilder.argtypes = [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError:
    pass
try:
    LLVMOrcJITTargetMachineBuilderGetTargetTriple = _libraries['llvm'].LLVMOrcJITTargetMachineBuilderGetTargetTriple
    LLVMOrcJITTargetMachineBuilderGetTargetTriple.restype = ctypes.POINTER(ctypes.c_char)
    LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError:
    pass
try:
    LLVMOrcJITTargetMachineBuilderSetTargetTriple = _libraries['llvm'].LLVMOrcJITTargetMachineBuilderSetTargetTriple
    LLVMOrcJITTargetMachineBuilderSetTargetTriple.restype = None
    LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcObjectLayerAddObjectFile = _libraries['llvm'].LLVMOrcObjectLayerAddObjectFile
    LLVMOrcObjectLayerAddObjectFile.restype = LLVMErrorRef
    LLVMOrcObjectLayerAddObjectFile.argtypes = [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMOrcObjectLayerAddObjectFileWithRT = _libraries['llvm'].LLVMOrcObjectLayerAddObjectFileWithRT
    LLVMOrcObjectLayerAddObjectFileWithRT.restype = LLVMErrorRef
    LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMOrcObjectLayerEmit = _libraries['llvm'].LLVMOrcObjectLayerEmit
    LLVMOrcObjectLayerEmit.restype = None
    LLVMOrcObjectLayerEmit.argtypes = [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeObjectLayer = _libraries['llvm'].LLVMOrcDisposeObjectLayer
    LLVMOrcDisposeObjectLayer.restype = None
    LLVMOrcDisposeObjectLayer.argtypes = [LLVMOrcObjectLayerRef]
except AttributeError:
    pass
try:
    LLVMOrcIRTransformLayerEmit = _libraries['llvm'].LLVMOrcIRTransformLayerEmit
    LLVMOrcIRTransformLayerEmit.restype = None
    LLVMOrcIRTransformLayerEmit.argtypes = [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError:
    pass
try:
    LLVMOrcIRTransformLayerSetTransform = _libraries['llvm'].LLVMOrcIRTransformLayerSetTransform
    LLVMOrcIRTransformLayerSetTransform.restype = None
    LLVMOrcIRTransformLayerSetTransform.argtypes = [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcObjectTransformLayerSetTransform = _libraries['llvm'].LLVMOrcObjectTransformLayerSetTransform
    LLVMOrcObjectTransformLayerSetTransform.restype = None
    LLVMOrcObjectTransformLayerSetTransform.argtypes = [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcCreateLocalIndirectStubsManager = _libraries['llvm'].LLVMOrcCreateLocalIndirectStubsManager
    LLVMOrcCreateLocalIndirectStubsManager.restype = LLVMOrcIndirectStubsManagerRef
    LLVMOrcCreateLocalIndirectStubsManager.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcDisposeIndirectStubsManager = _libraries['llvm'].LLVMOrcDisposeIndirectStubsManager
    LLVMOrcDisposeIndirectStubsManager.restype = None
    LLVMOrcDisposeIndirectStubsManager.argtypes = [LLVMOrcIndirectStubsManagerRef]
except AttributeError:
    pass
try:
    LLVMOrcCreateLocalLazyCallThroughManager = _libraries['llvm'].LLVMOrcCreateLocalLazyCallThroughManager
    LLVMOrcCreateLocalLazyCallThroughManager.restype = LLVMErrorRef
    LLVMOrcCreateLocalLazyCallThroughManager.argtypes = [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueLazyCallThroughManager))]
except AttributeError:
    pass
try:
    LLVMOrcDisposeLazyCallThroughManager = _libraries['llvm'].LLVMOrcDisposeLazyCallThroughManager
    LLVMOrcDisposeLazyCallThroughManager.restype = None
    LLVMOrcDisposeLazyCallThroughManager.argtypes = [LLVMOrcLazyCallThroughManagerRef]
except AttributeError:
    pass
try:
    LLVMOrcCreateDumpObjects = _libraries['llvm'].LLVMOrcCreateDumpObjects
    LLVMOrcCreateDumpObjects.restype = LLVMOrcDumpObjectsRef
    LLVMOrcCreateDumpObjects.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcDisposeDumpObjects = _libraries['llvm'].LLVMOrcDisposeDumpObjects
    LLVMOrcDisposeDumpObjects.restype = None
    LLVMOrcDisposeDumpObjects.argtypes = [LLVMOrcDumpObjectsRef]
except AttributeError:
    pass
try:
    LLVMOrcDumpObjects_CallOperator = _libraries['llvm'].LLVMOrcDumpObjects_CallOperator
    LLVMOrcDumpObjects_CallOperator.restype = LLVMErrorRef
    LLVMOrcDumpObjects_CallOperator.argtypes = [LLVMOrcDumpObjectsRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer))]
except AttributeError:
    pass
LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer), ctypes.POINTER(None), ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession), ctypes.POINTER(ctypes.c_char))
class struct_LLVMOrcOpaqueLLJITBuilder(Structure):
    pass

LLVMOrcLLJITBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJITBuilder)
class struct_LLVMOrcOpaqueLLJIT(Structure):
    pass

LLVMOrcLLJITRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJIT)
try:
    LLVMOrcCreateLLJITBuilder = _libraries['llvm'].LLVMOrcCreateLLJITBuilder
    LLVMOrcCreateLLJITBuilder.restype = LLVMOrcLLJITBuilderRef
    LLVMOrcCreateLLJITBuilder.argtypes = []
except AttributeError:
    pass
try:
    LLVMOrcDisposeLLJITBuilder = _libraries['llvm'].LLVMOrcDisposeLLJITBuilder
    LLVMOrcDisposeLLJITBuilder.restype = None
    LLVMOrcDisposeLLJITBuilder.argtypes = [LLVMOrcLLJITBuilderRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder = _libraries['llvm'].LLVMOrcLLJITBuilderSetJITTargetMachineBuilder
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.restype = None
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.argtypes = [LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator = _libraries['llvm'].LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator
    LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.restype = None
    LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.argtypes = [LLVMOrcLLJITBuilderRef, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    LLVMOrcCreateLLJIT = _libraries['llvm'].LLVMOrcCreateLLJIT
    LLVMOrcCreateLLJIT.restype = LLVMErrorRef
    LLVMOrcCreateLLJIT.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueLLJIT)), LLVMOrcLLJITBuilderRef]
except AttributeError:
    pass
try:
    LLVMOrcDisposeLLJIT = _libraries['llvm'].LLVMOrcDisposeLLJIT
    LLVMOrcDisposeLLJIT.restype = LLVMErrorRef
    LLVMOrcDisposeLLJIT.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetExecutionSession = _libraries['llvm'].LLVMOrcLLJITGetExecutionSession
    LLVMOrcLLJITGetExecutionSession.restype = LLVMOrcExecutionSessionRef
    LLVMOrcLLJITGetExecutionSession.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetMainJITDylib = _libraries['llvm'].LLVMOrcLLJITGetMainJITDylib
    LLVMOrcLLJITGetMainJITDylib.restype = LLVMOrcJITDylibRef
    LLVMOrcLLJITGetMainJITDylib.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetTripleString = _libraries['llvm'].LLVMOrcLLJITGetTripleString
    LLVMOrcLLJITGetTripleString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMOrcLLJITGetTripleString.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetGlobalPrefix = _libraries['llvm'].LLVMOrcLLJITGetGlobalPrefix
    LLVMOrcLLJITGetGlobalPrefix.restype = ctypes.c_char
    LLVMOrcLLJITGetGlobalPrefix.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITMangleAndIntern = _libraries['llvm'].LLVMOrcLLJITMangleAndIntern
    LLVMOrcLLJITMangleAndIntern.restype = LLVMOrcSymbolStringPoolEntryRef
    LLVMOrcLLJITMangleAndIntern.argtypes = [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcLLJITAddObjectFile = _libraries['llvm'].LLVMOrcLLJITAddObjectFile
    LLVMOrcLLJITAddObjectFile.restype = LLVMErrorRef
    LLVMOrcLLJITAddObjectFile.argtypes = [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITAddObjectFileWithRT = _libraries['llvm'].LLVMOrcLLJITAddObjectFileWithRT
    LLVMOrcLLJITAddObjectFileWithRT.restype = LLVMErrorRef
    LLVMOrcLLJITAddObjectFileWithRT.argtypes = [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITAddLLVMIRModule = _libraries['llvm'].LLVMOrcLLJITAddLLVMIRModule
    LLVMOrcLLJITAddLLVMIRModule.restype = LLVMErrorRef
    LLVMOrcLLJITAddLLVMIRModule.argtypes = [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITAddLLVMIRModuleWithRT = _libraries['llvm'].LLVMOrcLLJITAddLLVMIRModuleWithRT
    LLVMOrcLLJITAddLLVMIRModuleWithRT.restype = LLVMErrorRef
    LLVMOrcLLJITAddLLVMIRModuleWithRT.argtypes = [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMOrcThreadSafeModuleRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITLookup = _libraries['llvm'].LLVMOrcLLJITLookup
    LLVMOrcLLJITLookup.restype = LLVMErrorRef
    LLVMOrcLLJITLookup.argtypes = [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetObjLinkingLayer = _libraries['llvm'].LLVMOrcLLJITGetObjLinkingLayer
    LLVMOrcLLJITGetObjLinkingLayer.restype = LLVMOrcObjectLayerRef
    LLVMOrcLLJITGetObjLinkingLayer.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetObjTransformLayer = _libraries['llvm'].LLVMOrcLLJITGetObjTransformLayer
    LLVMOrcLLJITGetObjTransformLayer.restype = LLVMOrcObjectTransformLayerRef
    LLVMOrcLLJITGetObjTransformLayer.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetIRTransformLayer = _libraries['llvm'].LLVMOrcLLJITGetIRTransformLayer
    LLVMOrcLLJITGetIRTransformLayer.restype = LLVMOrcIRTransformLayerRef
    LLVMOrcLLJITGetIRTransformLayer.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
try:
    LLVMOrcLLJITGetDataLayoutStr = _libraries['llvm'].LLVMOrcLLJITGetDataLayoutStr
    LLVMOrcLLJITGetDataLayoutStr.restype = ctypes.POINTER(ctypes.c_char)
    LLVMOrcLLJITGetDataLayoutStr.argtypes = [LLVMOrcLLJITRef]
except AttributeError:
    pass
LLVM_C_LINKER_H = True # macro

# values for enumeration 'c__EA_LLVMLinkerMode'
c__EA_LLVMLinkerMode__enumvalues = {
    0: 'LLVMLinkerDestroySource',
    1: 'LLVMLinkerPreserveSource_Removed',
}
LLVMLinkerDestroySource = 0
LLVMLinkerPreserveSource_Removed = 1
c__EA_LLVMLinkerMode = ctypes.c_uint32 # enum
LLVMLinkerMode = c__EA_LLVMLinkerMode
LLVMLinkerMode__enumvalues = c__EA_LLVMLinkerMode__enumvalues
try:
    LLVMLinkModules2 = _libraries['llvm'].LLVMLinkModules2
    LLVMLinkModules2.restype = LLVMBool
    LLVMLinkModules2.argtypes = [LLVMModuleRef, LLVMModuleRef]
except AttributeError:
    pass
LLVM_C_OBJECT_H = True # macro
class struct_LLVMOpaqueSectionIterator(Structure):
    pass

LLVMSectionIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSectionIterator)
class struct_LLVMOpaqueSymbolIterator(Structure):
    pass

LLVMSymbolIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSymbolIterator)
class struct_LLVMOpaqueRelocationIterator(Structure):
    pass

LLVMRelocationIteratorRef = ctypes.POINTER(struct_LLVMOpaqueRelocationIterator)

# values for enumeration 'c__EA_LLVMBinaryType'
c__EA_LLVMBinaryType__enumvalues = {
    0: 'LLVMBinaryTypeArchive',
    1: 'LLVMBinaryTypeMachOUniversalBinary',
    2: 'LLVMBinaryTypeCOFFImportFile',
    3: 'LLVMBinaryTypeIR',
    4: 'LLVMBinaryTypeWinRes',
    5: 'LLVMBinaryTypeCOFF',
    6: 'LLVMBinaryTypeELF32L',
    7: 'LLVMBinaryTypeELF32B',
    8: 'LLVMBinaryTypeELF64L',
    9: 'LLVMBinaryTypeELF64B',
    10: 'LLVMBinaryTypeMachO32L',
    11: 'LLVMBinaryTypeMachO32B',
    12: 'LLVMBinaryTypeMachO64L',
    13: 'LLVMBinaryTypeMachO64B',
    14: 'LLVMBinaryTypeWasm',
}
LLVMBinaryTypeArchive = 0
LLVMBinaryTypeMachOUniversalBinary = 1
LLVMBinaryTypeCOFFImportFile = 2
LLVMBinaryTypeIR = 3
LLVMBinaryTypeWinRes = 4
LLVMBinaryTypeCOFF = 5
LLVMBinaryTypeELF32L = 6
LLVMBinaryTypeELF32B = 7
LLVMBinaryTypeELF64L = 8
LLVMBinaryTypeELF64B = 9
LLVMBinaryTypeMachO32L = 10
LLVMBinaryTypeMachO32B = 11
LLVMBinaryTypeMachO64L = 12
LLVMBinaryTypeMachO64B = 13
LLVMBinaryTypeWasm = 14
c__EA_LLVMBinaryType = ctypes.c_uint32 # enum
LLVMBinaryType = c__EA_LLVMBinaryType
LLVMBinaryType__enumvalues = c__EA_LLVMBinaryType__enumvalues
try:
    LLVMCreateBinary = _libraries['llvm'].LLVMCreateBinary
    LLVMCreateBinary.restype = LLVMBinaryRef
    LLVMCreateBinary.argtypes = [LLVMMemoryBufferRef, LLVMContextRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMDisposeBinary = _libraries['llvm'].LLVMDisposeBinary
    LLVMDisposeBinary.restype = None
    LLVMDisposeBinary.argtypes = [LLVMBinaryRef]
except AttributeError:
    pass
try:
    LLVMBinaryCopyMemoryBuffer = _libraries['llvm'].LLVMBinaryCopyMemoryBuffer
    LLVMBinaryCopyMemoryBuffer.restype = LLVMMemoryBufferRef
    LLVMBinaryCopyMemoryBuffer.argtypes = [LLVMBinaryRef]
except AttributeError:
    pass
try:
    LLVMBinaryGetType = _libraries['llvm'].LLVMBinaryGetType
    LLVMBinaryGetType.restype = LLVMBinaryType
    LLVMBinaryGetType.argtypes = [LLVMBinaryRef]
except AttributeError:
    pass
try:
    LLVMMachOUniversalBinaryCopyObjectForArch = _libraries['llvm'].LLVMMachOUniversalBinaryCopyObjectForArch
    LLVMMachOUniversalBinaryCopyObjectForArch.restype = LLVMBinaryRef
    LLVMMachOUniversalBinaryCopyObjectForArch.argtypes = [LLVMBinaryRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    LLVMObjectFileCopySectionIterator = _libraries['llvm'].LLVMObjectFileCopySectionIterator
    LLVMObjectFileCopySectionIterator.restype = LLVMSectionIteratorRef
    LLVMObjectFileCopySectionIterator.argtypes = [LLVMBinaryRef]
except AttributeError:
    pass
try:
    LLVMObjectFileIsSectionIteratorAtEnd = _libraries['llvm'].LLVMObjectFileIsSectionIteratorAtEnd
    LLVMObjectFileIsSectionIteratorAtEnd.restype = LLVMBool
    LLVMObjectFileIsSectionIteratorAtEnd.argtypes = [LLVMBinaryRef, LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMObjectFileCopySymbolIterator = _libraries['llvm'].LLVMObjectFileCopySymbolIterator
    LLVMObjectFileCopySymbolIterator.restype = LLVMSymbolIteratorRef
    LLVMObjectFileCopySymbolIterator.argtypes = [LLVMBinaryRef]
except AttributeError:
    pass
try:
    LLVMObjectFileIsSymbolIteratorAtEnd = _libraries['llvm'].LLVMObjectFileIsSymbolIteratorAtEnd
    LLVMObjectFileIsSymbolIteratorAtEnd.restype = LLVMBool
    LLVMObjectFileIsSymbolIteratorAtEnd.argtypes = [LLVMBinaryRef, LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMDisposeSectionIterator = _libraries['llvm'].LLVMDisposeSectionIterator
    LLVMDisposeSectionIterator.restype = None
    LLVMDisposeSectionIterator.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMMoveToNextSection = _libraries['llvm'].LLVMMoveToNextSection
    LLVMMoveToNextSection.restype = None
    LLVMMoveToNextSection.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMMoveToContainingSection = _libraries['llvm'].LLVMMoveToContainingSection
    LLVMMoveToContainingSection.restype = None
    LLVMMoveToContainingSection.argtypes = [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMDisposeSymbolIterator = _libraries['llvm'].LLVMDisposeSymbolIterator
    LLVMDisposeSymbolIterator.restype = None
    LLVMDisposeSymbolIterator.argtypes = [LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMMoveToNextSymbol = _libraries['llvm'].LLVMMoveToNextSymbol
    LLVMMoveToNextSymbol.restype = None
    LLVMMoveToNextSymbol.argtypes = [LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSectionName = _libraries['llvm'].LLVMGetSectionName
    LLVMGetSectionName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetSectionName.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSectionSize = _libraries['llvm'].LLVMGetSectionSize
    LLVMGetSectionSize.restype = uint64_t
    LLVMGetSectionSize.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSectionContents = _libraries['llvm'].LLVMGetSectionContents
    LLVMGetSectionContents.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetSectionContents.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSectionAddress = _libraries['llvm'].LLVMGetSectionAddress
    LLVMGetSectionAddress.restype = uint64_t
    LLVMGetSectionAddress.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSectionContainsSymbol = _libraries['llvm'].LLVMGetSectionContainsSymbol
    LLVMGetSectionContainsSymbol.restype = LLVMBool
    LLVMGetSectionContainsSymbol.argtypes = [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocations = _libraries['llvm'].LLVMGetRelocations
    LLVMGetRelocations.restype = LLVMRelocationIteratorRef
    LLVMGetRelocations.argtypes = [LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMDisposeRelocationIterator = _libraries['llvm'].LLVMDisposeRelocationIterator
    LLVMDisposeRelocationIterator.restype = None
    LLVMDisposeRelocationIterator.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMIsRelocationIteratorAtEnd = _libraries['llvm'].LLVMIsRelocationIteratorAtEnd
    LLVMIsRelocationIteratorAtEnd.restype = LLVMBool
    LLVMIsRelocationIteratorAtEnd.argtypes = [LLVMSectionIteratorRef, LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMMoveToNextRelocation = _libraries['llvm'].LLVMMoveToNextRelocation
    LLVMMoveToNextRelocation.restype = None
    LLVMMoveToNextRelocation.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSymbolName = _libraries['llvm'].LLVMGetSymbolName
    LLVMGetSymbolName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetSymbolName.argtypes = [LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSymbolAddress = _libraries['llvm'].LLVMGetSymbolAddress
    LLVMGetSymbolAddress.restype = uint64_t
    LLVMGetSymbolAddress.argtypes = [LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSymbolSize = _libraries['llvm'].LLVMGetSymbolSize
    LLVMGetSymbolSize.restype = uint64_t
    LLVMGetSymbolSize.argtypes = [LLVMSymbolIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocationOffset = _libraries['llvm'].LLVMGetRelocationOffset
    LLVMGetRelocationOffset.restype = uint64_t
    LLVMGetRelocationOffset.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocationSymbol = _libraries['llvm'].LLVMGetRelocationSymbol
    LLVMGetRelocationSymbol.restype = LLVMSymbolIteratorRef
    LLVMGetRelocationSymbol.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocationType = _libraries['llvm'].LLVMGetRelocationType
    LLVMGetRelocationType.restype = uint64_t
    LLVMGetRelocationType.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocationTypeName = _libraries['llvm'].LLVMGetRelocationTypeName
    LLVMGetRelocationTypeName.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetRelocationTypeName.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetRelocationValueString = _libraries['llvm'].LLVMGetRelocationValueString
    LLVMGetRelocationValueString.restype = ctypes.POINTER(ctypes.c_char)
    LLVMGetRelocationValueString.argtypes = [LLVMRelocationIteratorRef]
except AttributeError:
    pass
class struct_LLVMOpaqueObjectFile(Structure):
    pass

LLVMObjectFileRef = ctypes.POINTER(struct_LLVMOpaqueObjectFile)
try:
    LLVMCreateObjectFile = _libraries['llvm'].LLVMCreateObjectFile
    LLVMCreateObjectFile.restype = LLVMObjectFileRef
    LLVMCreateObjectFile.argtypes = [LLVMMemoryBufferRef]
except AttributeError:
    pass
try:
    LLVMDisposeObjectFile = _libraries['llvm'].LLVMDisposeObjectFile
    LLVMDisposeObjectFile.restype = None
    LLVMDisposeObjectFile.argtypes = [LLVMObjectFileRef]
except AttributeError:
    pass
try:
    LLVMGetSections = _libraries['llvm'].LLVMGetSections
    LLVMGetSections.restype = LLVMSectionIteratorRef
    LLVMGetSections.argtypes = [LLVMObjectFileRef]
except AttributeError:
    pass
try:
    LLVMIsSectionIteratorAtEnd = _libraries['llvm'].LLVMIsSectionIteratorAtEnd
    LLVMIsSectionIteratorAtEnd.restype = LLVMBool
    LLVMIsSectionIteratorAtEnd.argtypes = [LLVMObjectFileRef, LLVMSectionIteratorRef]
except AttributeError:
    pass
try:
    LLVMGetSymbols = _libraries['llvm'].LLVMGetSymbols
    LLVMGetSymbols.restype = LLVMSymbolIteratorRef
    LLVMGetSymbols.argtypes = [LLVMObjectFileRef]
except AttributeError:
    pass
try:
    LLVMIsSymbolIteratorAtEnd = _libraries['llvm'].LLVMIsSymbolIteratorAtEnd
    LLVMIsSymbolIteratorAtEnd.restype = LLVMBool
    LLVMIsSymbolIteratorAtEnd.argtypes = [LLVMObjectFileRef, LLVMSymbolIteratorRef]
except AttributeError:
    pass
LLVM_C_ORCEE_H = True # macro
try:
    LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager = _libraries['llvm'].LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager
    LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager.restype = LLVMOrcObjectLayerRef
    LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager.argtypes = [LLVMOrcExecutionSessionRef]
except AttributeError:
    pass
try:
    LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener = _libraries['llvm'].LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener
    LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener.restype = None
    LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener.argtypes = [LLVMOrcObjectLayerRef, LLVMJITEventListenerRef]
except AttributeError:
    pass
LLVM_C_REMARKS_H = True # macro
REMARKS_API_VERSION = 1 # macro

# values for enumeration 'LLVMRemarkType'
LLVMRemarkType__enumvalues = {
    0: 'LLVMRemarkTypeUnknown',
    1: 'LLVMRemarkTypePassed',
    2: 'LLVMRemarkTypeMissed',
    3: 'LLVMRemarkTypeAnalysis',
    4: 'LLVMRemarkTypeAnalysisFPCommute',
    5: 'LLVMRemarkTypeAnalysisAliasing',
    6: 'LLVMRemarkTypeFailure',
}
LLVMRemarkTypeUnknown = 0
LLVMRemarkTypePassed = 1
LLVMRemarkTypeMissed = 2
LLVMRemarkTypeAnalysis = 3
LLVMRemarkTypeAnalysisFPCommute = 4
LLVMRemarkTypeAnalysisAliasing = 5
LLVMRemarkTypeFailure = 6
LLVMRemarkType = ctypes.c_uint32 # enum
class struct_LLVMRemarkOpaqueString(Structure):
    pass

LLVMRemarkStringRef = ctypes.POINTER(struct_LLVMRemarkOpaqueString)
try:
    LLVMRemarkStringGetData = _libraries['llvm'].LLVMRemarkStringGetData
    LLVMRemarkStringGetData.restype = ctypes.POINTER(ctypes.c_char)
    LLVMRemarkStringGetData.argtypes = [LLVMRemarkStringRef]
except AttributeError:
    pass
try:
    LLVMRemarkStringGetLen = _libraries['llvm'].LLVMRemarkStringGetLen
    LLVMRemarkStringGetLen.restype = uint32_t
    LLVMRemarkStringGetLen.argtypes = [LLVMRemarkStringRef]
except AttributeError:
    pass
class struct_LLVMRemarkOpaqueDebugLoc(Structure):
    pass

LLVMRemarkDebugLocRef = ctypes.POINTER(struct_LLVMRemarkOpaqueDebugLoc)
try:
    LLVMRemarkDebugLocGetSourceFilePath = _libraries['llvm'].LLVMRemarkDebugLocGetSourceFilePath
    LLVMRemarkDebugLocGetSourceFilePath.restype = LLVMRemarkStringRef
    LLVMRemarkDebugLocGetSourceFilePath.argtypes = [LLVMRemarkDebugLocRef]
except AttributeError:
    pass
try:
    LLVMRemarkDebugLocGetSourceLine = _libraries['llvm'].LLVMRemarkDebugLocGetSourceLine
    LLVMRemarkDebugLocGetSourceLine.restype = uint32_t
    LLVMRemarkDebugLocGetSourceLine.argtypes = [LLVMRemarkDebugLocRef]
except AttributeError:
    pass
try:
    LLVMRemarkDebugLocGetSourceColumn = _libraries['llvm'].LLVMRemarkDebugLocGetSourceColumn
    LLVMRemarkDebugLocGetSourceColumn.restype = uint32_t
    LLVMRemarkDebugLocGetSourceColumn.argtypes = [LLVMRemarkDebugLocRef]
except AttributeError:
    pass
class struct_LLVMRemarkOpaqueArg(Structure):
    pass

LLVMRemarkArgRef = ctypes.POINTER(struct_LLVMRemarkOpaqueArg)
try:
    LLVMRemarkArgGetKey = _libraries['llvm'].LLVMRemarkArgGetKey
    LLVMRemarkArgGetKey.restype = LLVMRemarkStringRef
    LLVMRemarkArgGetKey.argtypes = [LLVMRemarkArgRef]
except AttributeError:
    pass
try:
    LLVMRemarkArgGetValue = _libraries['llvm'].LLVMRemarkArgGetValue
    LLVMRemarkArgGetValue.restype = LLVMRemarkStringRef
    LLVMRemarkArgGetValue.argtypes = [LLVMRemarkArgRef]
except AttributeError:
    pass
try:
    LLVMRemarkArgGetDebugLoc = _libraries['llvm'].LLVMRemarkArgGetDebugLoc
    LLVMRemarkArgGetDebugLoc.restype = LLVMRemarkDebugLocRef
    LLVMRemarkArgGetDebugLoc.argtypes = [LLVMRemarkArgRef]
except AttributeError:
    pass
class struct_LLVMRemarkOpaqueEntry(Structure):
    pass

LLVMRemarkEntryRef = ctypes.POINTER(struct_LLVMRemarkOpaqueEntry)
try:
    LLVMRemarkEntryDispose = _libraries['llvm'].LLVMRemarkEntryDispose
    LLVMRemarkEntryDispose.restype = None
    LLVMRemarkEntryDispose.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetType = _libraries['llvm'].LLVMRemarkEntryGetType
    LLVMRemarkEntryGetType.restype = LLVMRemarkType
    LLVMRemarkEntryGetType.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetPassName = _libraries['llvm'].LLVMRemarkEntryGetPassName
    LLVMRemarkEntryGetPassName.restype = LLVMRemarkStringRef
    LLVMRemarkEntryGetPassName.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetRemarkName = _libraries['llvm'].LLVMRemarkEntryGetRemarkName
    LLVMRemarkEntryGetRemarkName.restype = LLVMRemarkStringRef
    LLVMRemarkEntryGetRemarkName.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetFunctionName = _libraries['llvm'].LLVMRemarkEntryGetFunctionName
    LLVMRemarkEntryGetFunctionName.restype = LLVMRemarkStringRef
    LLVMRemarkEntryGetFunctionName.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetDebugLoc = _libraries['llvm'].LLVMRemarkEntryGetDebugLoc
    LLVMRemarkEntryGetDebugLoc.restype = LLVMRemarkDebugLocRef
    LLVMRemarkEntryGetDebugLoc.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetHotness = _libraries['llvm'].LLVMRemarkEntryGetHotness
    LLVMRemarkEntryGetHotness.restype = uint64_t
    LLVMRemarkEntryGetHotness.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetNumArgs = _libraries['llvm'].LLVMRemarkEntryGetNumArgs
    LLVMRemarkEntryGetNumArgs.restype = uint32_t
    LLVMRemarkEntryGetNumArgs.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetFirstArg = _libraries['llvm'].LLVMRemarkEntryGetFirstArg
    LLVMRemarkEntryGetFirstArg.restype = LLVMRemarkArgRef
    LLVMRemarkEntryGetFirstArg.argtypes = [LLVMRemarkEntryRef]
except AttributeError:
    pass
try:
    LLVMRemarkEntryGetNextArg = _libraries['llvm'].LLVMRemarkEntryGetNextArg
    LLVMRemarkEntryGetNextArg.restype = LLVMRemarkArgRef
    LLVMRemarkEntryGetNextArg.argtypes = [LLVMRemarkArgRef, LLVMRemarkEntryRef]
except AttributeError:
    pass
class struct_LLVMRemarkOpaqueParser(Structure):
    pass

LLVMRemarkParserRef = ctypes.POINTER(struct_LLVMRemarkOpaqueParser)
try:
    LLVMRemarkParserCreateYAML = _libraries['llvm'].LLVMRemarkParserCreateYAML
    LLVMRemarkParserCreateYAML.restype = LLVMRemarkParserRef
    LLVMRemarkParserCreateYAML.argtypes = [ctypes.POINTER(None), uint64_t]
except AttributeError:
    pass
try:
    LLVMRemarkParserCreateBitstream = _libraries['llvm'].LLVMRemarkParserCreateBitstream
    LLVMRemarkParserCreateBitstream.restype = LLVMRemarkParserRef
    LLVMRemarkParserCreateBitstream.argtypes = [ctypes.POINTER(None), uint64_t]
except AttributeError:
    pass
try:
    LLVMRemarkParserGetNext = _libraries['llvm'].LLVMRemarkParserGetNext
    LLVMRemarkParserGetNext.restype = LLVMRemarkEntryRef
    LLVMRemarkParserGetNext.argtypes = [LLVMRemarkParserRef]
except AttributeError:
    pass
try:
    LLVMRemarkParserHasError = _libraries['llvm'].LLVMRemarkParserHasError
    LLVMRemarkParserHasError.restype = LLVMBool
    LLVMRemarkParserHasError.argtypes = [LLVMRemarkParserRef]
except AttributeError:
    pass
try:
    LLVMRemarkParserGetErrorMessage = _libraries['llvm'].LLVMRemarkParserGetErrorMessage
    LLVMRemarkParserGetErrorMessage.restype = ctypes.POINTER(ctypes.c_char)
    LLVMRemarkParserGetErrorMessage.argtypes = [LLVMRemarkParserRef]
except AttributeError:
    pass
try:
    LLVMRemarkParserDispose = _libraries['llvm'].LLVMRemarkParserDispose
    LLVMRemarkParserDispose.restype = None
    LLVMRemarkParserDispose.argtypes = [LLVMRemarkParserRef]
except AttributeError:
    pass
try:
    LLVMRemarkVersion = _libraries['llvm'].LLVMRemarkVersion
    LLVMRemarkVersion.restype = uint32_t
    LLVMRemarkVersion.argtypes = []
except AttributeError:
    pass
LLVM_C_SUPPORT_H = True # macro
try:
    LLVMLoadLibraryPermanently = _libraries['llvm'].LLVMLoadLibraryPermanently
    LLVMLoadLibraryPermanently.restype = LLVMBool
    LLVMLoadLibraryPermanently.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMParseCommandLineOptions = _libraries['llvm'].LLVMParseCommandLineOptions
    LLVMParseCommandLineOptions.restype = None
    LLVMParseCommandLineOptions.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMSearchForAddressOfSymbol = _libraries['llvm'].LLVMSearchForAddressOfSymbol
    LLVMSearchForAddressOfSymbol.restype = ctypes.POINTER(None)
    LLVMSearchForAddressOfSymbol.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    LLVMAddSymbol = _libraries['llvm'].LLVMAddSymbol
    LLVMAddSymbol.restype = None
    LLVMAddSymbol.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_AGGRESSIVEINSTCOMBINE_H = True # macro
try:
    LLVMAddAggressiveInstCombinerPass = _libraries['llvm'].LLVMAddAggressiveInstCombinerPass
    LLVMAddAggressiveInstCombinerPass.restype = None
    LLVMAddAggressiveInstCombinerPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_COROUTINES_H = True # macro
LLVM_C_TRANSFORMS_PASSMANAGERBUILDER_H = True # macro
class struct_LLVMOpaquePassManagerBuilder(Structure):
    pass

LLVMPassManagerBuilderRef = ctypes.POINTER(struct_LLVMOpaquePassManagerBuilder)
try:
    LLVMPassManagerBuilderCreate = _libraries['llvm'].LLVMPassManagerBuilderCreate
    LLVMPassManagerBuilderCreate.restype = LLVMPassManagerBuilderRef
    LLVMPassManagerBuilderCreate.argtypes = []
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderDispose = _libraries['llvm'].LLVMPassManagerBuilderDispose
    LLVMPassManagerBuilderDispose.restype = None
    LLVMPassManagerBuilderDispose.argtypes = [LLVMPassManagerBuilderRef]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderSetOptLevel = _libraries['llvm'].LLVMPassManagerBuilderSetOptLevel
    LLVMPassManagerBuilderSetOptLevel.restype = None
    LLVMPassManagerBuilderSetOptLevel.argtypes = [LLVMPassManagerBuilderRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderSetSizeLevel = _libraries['llvm'].LLVMPassManagerBuilderSetSizeLevel
    LLVMPassManagerBuilderSetSizeLevel.restype = None
    LLVMPassManagerBuilderSetSizeLevel.argtypes = [LLVMPassManagerBuilderRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderSetDisableUnitAtATime = _libraries['llvm'].LLVMPassManagerBuilderSetDisableUnitAtATime
    LLVMPassManagerBuilderSetDisableUnitAtATime.restype = None
    LLVMPassManagerBuilderSetDisableUnitAtATime.argtypes = [LLVMPassManagerBuilderRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderSetDisableUnrollLoops = _libraries['llvm'].LLVMPassManagerBuilderSetDisableUnrollLoops
    LLVMPassManagerBuilderSetDisableUnrollLoops.restype = None
    LLVMPassManagerBuilderSetDisableUnrollLoops.argtypes = [LLVMPassManagerBuilderRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderSetDisableSimplifyLibCalls = _libraries['llvm'].LLVMPassManagerBuilderSetDisableSimplifyLibCalls
    LLVMPassManagerBuilderSetDisableSimplifyLibCalls.restype = None
    LLVMPassManagerBuilderSetDisableSimplifyLibCalls.argtypes = [LLVMPassManagerBuilderRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderUseInlinerWithThreshold = _libraries['llvm'].LLVMPassManagerBuilderUseInlinerWithThreshold
    LLVMPassManagerBuilderUseInlinerWithThreshold.restype = None
    LLVMPassManagerBuilderUseInlinerWithThreshold.argtypes = [LLVMPassManagerBuilderRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderPopulateFunctionPassManager = _libraries['llvm'].LLVMPassManagerBuilderPopulateFunctionPassManager
    LLVMPassManagerBuilderPopulateFunctionPassManager.restype = None
    LLVMPassManagerBuilderPopulateFunctionPassManager.argtypes = [LLVMPassManagerBuilderRef, LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderPopulateModulePassManager = _libraries['llvm'].LLVMPassManagerBuilderPopulateModulePassManager
    LLVMPassManagerBuilderPopulateModulePassManager.restype = None
    LLVMPassManagerBuilderPopulateModulePassManager.argtypes = [LLVMPassManagerBuilderRef, LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderPopulateLTOPassManager = _libraries['llvm'].LLVMPassManagerBuilderPopulateLTOPassManager
    LLVMPassManagerBuilderPopulateLTOPassManager.restype = None
    LLVMPassManagerBuilderPopulateLTOPassManager.argtypes = [LLVMPassManagerBuilderRef, LLVMPassManagerRef, LLVMBool, LLVMBool]
except AttributeError:
    pass
try:
    LLVMAddCoroEarlyPass = _libraries['llvm'].LLVMAddCoroEarlyPass
    LLVMAddCoroEarlyPass.restype = None
    LLVMAddCoroEarlyPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCoroSplitPass = _libraries['llvm'].LLVMAddCoroSplitPass
    LLVMAddCoroSplitPass.restype = None
    LLVMAddCoroSplitPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCoroElidePass = _libraries['llvm'].LLVMAddCoroElidePass
    LLVMAddCoroElidePass.restype = None
    LLVMAddCoroElidePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCoroCleanupPass = _libraries['llvm'].LLVMAddCoroCleanupPass
    LLVMAddCoroCleanupPass.restype = None
    LLVMAddCoroCleanupPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints = _libraries['llvm'].LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints
    LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints.restype = None
    LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints.argtypes = [LLVMPassManagerBuilderRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_IPO_H = True # macro
try:
    LLVMAddArgumentPromotionPass = _libraries['llvm'].LLVMAddArgumentPromotionPass
    LLVMAddArgumentPromotionPass.restype = None
    LLVMAddArgumentPromotionPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddConstantMergePass = _libraries['llvm'].LLVMAddConstantMergePass
    LLVMAddConstantMergePass.restype = None
    LLVMAddConstantMergePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddMergeFunctionsPass = _libraries['llvm'].LLVMAddMergeFunctionsPass
    LLVMAddMergeFunctionsPass.restype = None
    LLVMAddMergeFunctionsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCalledValuePropagationPass = _libraries['llvm'].LLVMAddCalledValuePropagationPass
    LLVMAddCalledValuePropagationPass.restype = None
    LLVMAddCalledValuePropagationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddDeadArgEliminationPass = _libraries['llvm'].LLVMAddDeadArgEliminationPass
    LLVMAddDeadArgEliminationPass.restype = None
    LLVMAddDeadArgEliminationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddFunctionAttrsPass = _libraries['llvm'].LLVMAddFunctionAttrsPass
    LLVMAddFunctionAttrsPass.restype = None
    LLVMAddFunctionAttrsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddFunctionInliningPass = _libraries['llvm'].LLVMAddFunctionInliningPass
    LLVMAddFunctionInliningPass.restype = None
    LLVMAddFunctionInliningPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddAlwaysInlinerPass = _libraries['llvm'].LLVMAddAlwaysInlinerPass
    LLVMAddAlwaysInlinerPass.restype = None
    LLVMAddAlwaysInlinerPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddGlobalDCEPass = _libraries['llvm'].LLVMAddGlobalDCEPass
    LLVMAddGlobalDCEPass.restype = None
    LLVMAddGlobalDCEPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddGlobalOptimizerPass = _libraries['llvm'].LLVMAddGlobalOptimizerPass
    LLVMAddGlobalOptimizerPass.restype = None
    LLVMAddGlobalOptimizerPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddPruneEHPass = _libraries['llvm'].LLVMAddPruneEHPass
    LLVMAddPruneEHPass.restype = None
    LLVMAddPruneEHPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddIPSCCPPass = _libraries['llvm'].LLVMAddIPSCCPPass
    LLVMAddIPSCCPPass.restype = None
    LLVMAddIPSCCPPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddInternalizePass = _libraries['llvm'].LLVMAddInternalizePass
    LLVMAddInternalizePass.restype = None
    LLVMAddInternalizePass.argtypes = [LLVMPassManagerRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMAddInternalizePassWithMustPreservePredicate = _libraries['llvm'].LLVMAddInternalizePassWithMustPreservePredicate
    LLVMAddInternalizePassWithMustPreservePredicate.restype = None
    LLVMAddInternalizePassWithMustPreservePredicate.argtypes = [LLVMPassManagerRef, ctypes.POINTER(None), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    LLVMAddStripDeadPrototypesPass = _libraries['llvm'].LLVMAddStripDeadPrototypesPass
    LLVMAddStripDeadPrototypesPass.restype = None
    LLVMAddStripDeadPrototypesPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddStripSymbolsPass = _libraries['llvm'].LLVMAddStripSymbolsPass
    LLVMAddStripSymbolsPass.restype = None
    LLVMAddStripSymbolsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_INSTCOMBINE_H = True # macro
try:
    LLVMAddInstructionCombiningPass = _libraries['llvm'].LLVMAddInstructionCombiningPass
    LLVMAddInstructionCombiningPass.restype = None
    LLVMAddInstructionCombiningPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_PASSBUILDER_H = True # macro
class struct_LLVMOpaquePassBuilderOptions(Structure):
    pass

LLVMPassBuilderOptionsRef = ctypes.POINTER(struct_LLVMOpaquePassBuilderOptions)
try:
    LLVMRunPasses = _libraries['llvm'].LLVMRunPasses
    LLVMRunPasses.restype = LLVMErrorRef
    LLVMRunPasses.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineRef, LLVMPassBuilderOptionsRef]
except AttributeError:
    pass
try:
    LLVMCreatePassBuilderOptions = _libraries['llvm'].LLVMCreatePassBuilderOptions
    LLVMCreatePassBuilderOptions.restype = LLVMPassBuilderOptionsRef
    LLVMCreatePassBuilderOptions.argtypes = []
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetVerifyEach = _libraries['llvm'].LLVMPassBuilderOptionsSetVerifyEach
    LLVMPassBuilderOptionsSetVerifyEach.restype = None
    LLVMPassBuilderOptionsSetVerifyEach.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetDebugLogging = _libraries['llvm'].LLVMPassBuilderOptionsSetDebugLogging
    LLVMPassBuilderOptionsSetDebugLogging.restype = None
    LLVMPassBuilderOptionsSetDebugLogging.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetLoopInterleaving = _libraries['llvm'].LLVMPassBuilderOptionsSetLoopInterleaving
    LLVMPassBuilderOptionsSetLoopInterleaving.restype = None
    LLVMPassBuilderOptionsSetLoopInterleaving.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetLoopVectorization = _libraries['llvm'].LLVMPassBuilderOptionsSetLoopVectorization
    LLVMPassBuilderOptionsSetLoopVectorization.restype = None
    LLVMPassBuilderOptionsSetLoopVectorization.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetSLPVectorization = _libraries['llvm'].LLVMPassBuilderOptionsSetSLPVectorization
    LLVMPassBuilderOptionsSetSLPVectorization.restype = None
    LLVMPassBuilderOptionsSetSLPVectorization.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetLoopUnrolling = _libraries['llvm'].LLVMPassBuilderOptionsSetLoopUnrolling
    LLVMPassBuilderOptionsSetLoopUnrolling.restype = None
    LLVMPassBuilderOptionsSetLoopUnrolling.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll = _libraries['llvm'].LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll
    LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll.restype = None
    LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetLicmMssaOptCap = _libraries['llvm'].LLVMPassBuilderOptionsSetLicmMssaOptCap
    LLVMPassBuilderOptionsSetLicmMssaOptCap.restype = None
    LLVMPassBuilderOptionsSetLicmMssaOptCap.argtypes = [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap = _libraries['llvm'].LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap
    LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap.restype = None
    LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap.argtypes = [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetCallGraphProfile = _libraries['llvm'].LLVMPassBuilderOptionsSetCallGraphProfile
    LLVMPassBuilderOptionsSetCallGraphProfile.restype = None
    LLVMPassBuilderOptionsSetCallGraphProfile.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMPassBuilderOptionsSetMergeFunctions = _libraries['llvm'].LLVMPassBuilderOptionsSetMergeFunctions
    LLVMPassBuilderOptionsSetMergeFunctions.restype = None
    LLVMPassBuilderOptionsSetMergeFunctions.argtypes = [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError:
    pass
try:
    LLVMDisposePassBuilderOptions = _libraries['llvm'].LLVMDisposePassBuilderOptions
    LLVMDisposePassBuilderOptions.restype = None
    LLVMDisposePassBuilderOptions.argtypes = [LLVMPassBuilderOptionsRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_SCALAR_H = True # macro
try:
    LLVMAddAggressiveDCEPass = _libraries['llvm'].LLVMAddAggressiveDCEPass
    LLVMAddAggressiveDCEPass.restype = None
    LLVMAddAggressiveDCEPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddDCEPass = _libraries['llvm'].LLVMAddDCEPass
    LLVMAddDCEPass.restype = None
    LLVMAddDCEPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddBitTrackingDCEPass = _libraries['llvm'].LLVMAddBitTrackingDCEPass
    LLVMAddBitTrackingDCEPass.restype = None
    LLVMAddBitTrackingDCEPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddAlignmentFromAssumptionsPass = _libraries['llvm'].LLVMAddAlignmentFromAssumptionsPass
    LLVMAddAlignmentFromAssumptionsPass.restype = None
    LLVMAddAlignmentFromAssumptionsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCFGSimplificationPass = _libraries['llvm'].LLVMAddCFGSimplificationPass
    LLVMAddCFGSimplificationPass.restype = None
    LLVMAddCFGSimplificationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddDeadStoreEliminationPass = _libraries['llvm'].LLVMAddDeadStoreEliminationPass
    LLVMAddDeadStoreEliminationPass.restype = None
    LLVMAddDeadStoreEliminationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddScalarizerPass = _libraries['llvm'].LLVMAddScalarizerPass
    LLVMAddScalarizerPass.restype = None
    LLVMAddScalarizerPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddMergedLoadStoreMotionPass = _libraries['llvm'].LLVMAddMergedLoadStoreMotionPass
    LLVMAddMergedLoadStoreMotionPass.restype = None
    LLVMAddMergedLoadStoreMotionPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddGVNPass = _libraries['llvm'].LLVMAddGVNPass
    LLVMAddGVNPass.restype = None
    LLVMAddGVNPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddNewGVNPass = _libraries['llvm'].LLVMAddNewGVNPass
    LLVMAddNewGVNPass.restype = None
    LLVMAddNewGVNPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddIndVarSimplifyPass = _libraries['llvm'].LLVMAddIndVarSimplifyPass
    LLVMAddIndVarSimplifyPass.restype = None
    LLVMAddIndVarSimplifyPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddInstructionSimplifyPass = _libraries['llvm'].LLVMAddInstructionSimplifyPass
    LLVMAddInstructionSimplifyPass.restype = None
    LLVMAddInstructionSimplifyPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddJumpThreadingPass = _libraries['llvm'].LLVMAddJumpThreadingPass
    LLVMAddJumpThreadingPass.restype = None
    LLVMAddJumpThreadingPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLICMPass = _libraries['llvm'].LLVMAddLICMPass
    LLVMAddLICMPass.restype = None
    LLVMAddLICMPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopDeletionPass = _libraries['llvm'].LLVMAddLoopDeletionPass
    LLVMAddLoopDeletionPass.restype = None
    LLVMAddLoopDeletionPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopIdiomPass = _libraries['llvm'].LLVMAddLoopIdiomPass
    LLVMAddLoopIdiomPass.restype = None
    LLVMAddLoopIdiomPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopRotatePass = _libraries['llvm'].LLVMAddLoopRotatePass
    LLVMAddLoopRotatePass.restype = None
    LLVMAddLoopRotatePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopRerollPass = _libraries['llvm'].LLVMAddLoopRerollPass
    LLVMAddLoopRerollPass.restype = None
    LLVMAddLoopRerollPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopUnrollPass = _libraries['llvm'].LLVMAddLoopUnrollPass
    LLVMAddLoopUnrollPass.restype = None
    LLVMAddLoopUnrollPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopUnrollAndJamPass = _libraries['llvm'].LLVMAddLoopUnrollAndJamPass
    LLVMAddLoopUnrollAndJamPass.restype = None
    LLVMAddLoopUnrollAndJamPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLoopUnswitchPass = _libraries['llvm'].LLVMAddLoopUnswitchPass
    LLVMAddLoopUnswitchPass.restype = None
    LLVMAddLoopUnswitchPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLowerAtomicPass = _libraries['llvm'].LLVMAddLowerAtomicPass
    LLVMAddLowerAtomicPass.restype = None
    LLVMAddLowerAtomicPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddMemCpyOptPass = _libraries['llvm'].LLVMAddMemCpyOptPass
    LLVMAddMemCpyOptPass.restype = None
    LLVMAddMemCpyOptPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddPartiallyInlineLibCallsPass = _libraries['llvm'].LLVMAddPartiallyInlineLibCallsPass
    LLVMAddPartiallyInlineLibCallsPass.restype = None
    LLVMAddPartiallyInlineLibCallsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddReassociatePass = _libraries['llvm'].LLVMAddReassociatePass
    LLVMAddReassociatePass.restype = None
    LLVMAddReassociatePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddSCCPPass = _libraries['llvm'].LLVMAddSCCPPass
    LLVMAddSCCPPass.restype = None
    LLVMAddSCCPPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddScalarReplAggregatesPass = _libraries['llvm'].LLVMAddScalarReplAggregatesPass
    LLVMAddScalarReplAggregatesPass.restype = None
    LLVMAddScalarReplAggregatesPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddScalarReplAggregatesPassSSA = _libraries['llvm'].LLVMAddScalarReplAggregatesPassSSA
    LLVMAddScalarReplAggregatesPassSSA.restype = None
    LLVMAddScalarReplAggregatesPassSSA.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddScalarReplAggregatesPassWithThreshold = _libraries['llvm'].LLVMAddScalarReplAggregatesPassWithThreshold
    LLVMAddScalarReplAggregatesPassWithThreshold.restype = None
    LLVMAddScalarReplAggregatesPassWithThreshold.argtypes = [LLVMPassManagerRef, ctypes.c_int32]
except AttributeError:
    pass
try:
    LLVMAddSimplifyLibCallsPass = _libraries['llvm'].LLVMAddSimplifyLibCallsPass
    LLVMAddSimplifyLibCallsPass.restype = None
    LLVMAddSimplifyLibCallsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddTailCallEliminationPass = _libraries['llvm'].LLVMAddTailCallEliminationPass
    LLVMAddTailCallEliminationPass.restype = None
    LLVMAddTailCallEliminationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddDemoteMemoryToRegisterPass = _libraries['llvm'].LLVMAddDemoteMemoryToRegisterPass
    LLVMAddDemoteMemoryToRegisterPass.restype = None
    LLVMAddDemoteMemoryToRegisterPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddVerifierPass = _libraries['llvm'].LLVMAddVerifierPass
    LLVMAddVerifierPass.restype = None
    LLVMAddVerifierPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddCorrelatedValuePropagationPass = _libraries['llvm'].LLVMAddCorrelatedValuePropagationPass
    LLVMAddCorrelatedValuePropagationPass.restype = None
    LLVMAddCorrelatedValuePropagationPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddEarlyCSEPass = _libraries['llvm'].LLVMAddEarlyCSEPass
    LLVMAddEarlyCSEPass.restype = None
    LLVMAddEarlyCSEPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddEarlyCSEMemSSAPass = _libraries['llvm'].LLVMAddEarlyCSEMemSSAPass
    LLVMAddEarlyCSEMemSSAPass.restype = None
    LLVMAddEarlyCSEMemSSAPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLowerExpectIntrinsicPass = _libraries['llvm'].LLVMAddLowerExpectIntrinsicPass
    LLVMAddLowerExpectIntrinsicPass.restype = None
    LLVMAddLowerExpectIntrinsicPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddLowerConstantIntrinsicsPass = _libraries['llvm'].LLVMAddLowerConstantIntrinsicsPass
    LLVMAddLowerConstantIntrinsicsPass.restype = None
    LLVMAddLowerConstantIntrinsicsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddTypeBasedAliasAnalysisPass = _libraries['llvm'].LLVMAddTypeBasedAliasAnalysisPass
    LLVMAddTypeBasedAliasAnalysisPass.restype = None
    LLVMAddTypeBasedAliasAnalysisPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddScopedNoAliasAAPass = _libraries['llvm'].LLVMAddScopedNoAliasAAPass
    LLVMAddScopedNoAliasAAPass.restype = None
    LLVMAddScopedNoAliasAAPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddBasicAliasAnalysisPass = _libraries['llvm'].LLVMAddBasicAliasAnalysisPass
    LLVMAddBasicAliasAnalysisPass.restype = None
    LLVMAddBasicAliasAnalysisPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddUnifyFunctionExitNodesPass = _libraries['llvm'].LLVMAddUnifyFunctionExitNodesPass
    LLVMAddUnifyFunctionExitNodesPass.restype = None
    LLVMAddUnifyFunctionExitNodesPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_UTILS_H = True # macro
try:
    LLVMAddLowerSwitchPass = _libraries['llvm'].LLVMAddLowerSwitchPass
    LLVMAddLowerSwitchPass.restype = None
    LLVMAddLowerSwitchPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddPromoteMemoryToRegisterPass = _libraries['llvm'].LLVMAddPromoteMemoryToRegisterPass
    LLVMAddPromoteMemoryToRegisterPass.restype = None
    LLVMAddPromoteMemoryToRegisterPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddAddDiscriminatorsPass = _libraries['llvm'].LLVMAddAddDiscriminatorsPass
    LLVMAddAddDiscriminatorsPass.restype = None
    LLVMAddAddDiscriminatorsPass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_TRANSFORMS_VECTORIZE_H = True # macro
try:
    LLVMAddLoopVectorizePass = _libraries['llvm'].LLVMAddLoopVectorizePass
    LLVMAddLoopVectorizePass.restype = None
    LLVMAddLoopVectorizePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
try:
    LLVMAddSLPVectorizePass = _libraries['llvm'].LLVMAddSLPVectorizePass
    LLVMAddSLPVectorizePass.restype = None
    LLVMAddSLPVectorizePass.argtypes = [LLVMPassManagerRef]
except AttributeError:
    pass
LLVM_C_LTO_H = True # macro
LTO_API_VERSION = 29 # macro
lto_bool_t = ctypes.c_bool

# values for enumeration 'c__EA_lto_symbol_attributes'
c__EA_lto_symbol_attributes__enumvalues = {
    31: 'LTO_SYMBOL_ALIGNMENT_MASK',
    224: 'LTO_SYMBOL_PERMISSIONS_MASK',
    160: 'LTO_SYMBOL_PERMISSIONS_CODE',
    192: 'LTO_SYMBOL_PERMISSIONS_DATA',
    128: 'LTO_SYMBOL_PERMISSIONS_RODATA',
    1792: 'LTO_SYMBOL_DEFINITION_MASK',
    256: 'LTO_SYMBOL_DEFINITION_REGULAR',
    512: 'LTO_SYMBOL_DEFINITION_TENTATIVE',
    768: 'LTO_SYMBOL_DEFINITION_WEAK',
    1024: 'LTO_SYMBOL_DEFINITION_UNDEFINED',
    1280: 'LTO_SYMBOL_DEFINITION_WEAKUNDEF',
    14336: 'LTO_SYMBOL_SCOPE_MASK',
    2048: 'LTO_SYMBOL_SCOPE_INTERNAL',
    4096: 'LTO_SYMBOL_SCOPE_HIDDEN',
    8192: 'LTO_SYMBOL_SCOPE_PROTECTED',
    6144: 'LTO_SYMBOL_SCOPE_DEFAULT',
    10240: 'LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN',
    16384: 'LTO_SYMBOL_COMDAT',
    32768: 'LTO_SYMBOL_ALIAS',
}
LTO_SYMBOL_ALIGNMENT_MASK = 31
LTO_SYMBOL_PERMISSIONS_MASK = 224
LTO_SYMBOL_PERMISSIONS_CODE = 160
LTO_SYMBOL_PERMISSIONS_DATA = 192
LTO_SYMBOL_PERMISSIONS_RODATA = 128
LTO_SYMBOL_DEFINITION_MASK = 1792
LTO_SYMBOL_DEFINITION_REGULAR = 256
LTO_SYMBOL_DEFINITION_TENTATIVE = 512
LTO_SYMBOL_DEFINITION_WEAK = 768
LTO_SYMBOL_DEFINITION_UNDEFINED = 1024
LTO_SYMBOL_DEFINITION_WEAKUNDEF = 1280
LTO_SYMBOL_SCOPE_MASK = 14336
LTO_SYMBOL_SCOPE_INTERNAL = 2048
LTO_SYMBOL_SCOPE_HIDDEN = 4096
LTO_SYMBOL_SCOPE_PROTECTED = 8192
LTO_SYMBOL_SCOPE_DEFAULT = 6144
LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN = 10240
LTO_SYMBOL_COMDAT = 16384
LTO_SYMBOL_ALIAS = 32768
c__EA_lto_symbol_attributes = ctypes.c_uint32 # enum
lto_symbol_attributes = c__EA_lto_symbol_attributes
lto_symbol_attributes__enumvalues = c__EA_lto_symbol_attributes__enumvalues

# values for enumeration 'c__EA_lto_debug_model'
c__EA_lto_debug_model__enumvalues = {
    0: 'LTO_DEBUG_MODEL_NONE',
    1: 'LTO_DEBUG_MODEL_DWARF',
}
LTO_DEBUG_MODEL_NONE = 0
LTO_DEBUG_MODEL_DWARF = 1
c__EA_lto_debug_model = ctypes.c_uint32 # enum
lto_debug_model = c__EA_lto_debug_model
lto_debug_model__enumvalues = c__EA_lto_debug_model__enumvalues

# values for enumeration 'c__EA_lto_codegen_model'
c__EA_lto_codegen_model__enumvalues = {
    0: 'LTO_CODEGEN_PIC_MODEL_STATIC',
    1: 'LTO_CODEGEN_PIC_MODEL_DYNAMIC',
    2: 'LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC',
    3: 'LTO_CODEGEN_PIC_MODEL_DEFAULT',
}
LTO_CODEGEN_PIC_MODEL_STATIC = 0
LTO_CODEGEN_PIC_MODEL_DYNAMIC = 1
LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC = 2
LTO_CODEGEN_PIC_MODEL_DEFAULT = 3
c__EA_lto_codegen_model = ctypes.c_uint32 # enum
lto_codegen_model = c__EA_lto_codegen_model
lto_codegen_model__enumvalues = c__EA_lto_codegen_model__enumvalues
class struct_LLVMOpaqueLTOModule(Structure):
    pass

lto_module_t = ctypes.POINTER(struct_LLVMOpaqueLTOModule)
class struct_LLVMOpaqueLTOCodeGenerator(Structure):
    pass

lto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueLTOCodeGenerator)
class struct_LLVMOpaqueThinLTOCodeGenerator(Structure):
    pass

thinlto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueThinLTOCodeGenerator)
try:
    lto_get_version = _libraries['llvm'].lto_get_version
    lto_get_version.restype = ctypes.POINTER(ctypes.c_char)
    lto_get_version.argtypes = []
except AttributeError:
    pass
try:
    lto_get_error_message = _libraries['llvm'].lto_get_error_message
    lto_get_error_message.restype = ctypes.POINTER(ctypes.c_char)
    lto_get_error_message.argtypes = []
except AttributeError:
    pass
try:
    lto_module_is_object_file = _libraries['llvm'].lto_module_is_object_file
    lto_module_is_object_file.restype = lto_bool_t
    lto_module_is_object_file.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_is_object_file_for_target = _libraries['llvm'].lto_module_is_object_file_for_target
    lto_module_is_object_file_for_target.restype = lto_bool_t
    lto_module_is_object_file_for_target.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_has_objc_category = _libraries['llvm'].lto_module_has_objc_category
    lto_module_has_objc_category.restype = lto_bool_t
    lto_module_has_objc_category.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    lto_module_is_object_file_in_memory = _libraries['llvm'].lto_module_is_object_file_in_memory
    lto_module_is_object_file_in_memory.restype = lto_bool_t
    lto_module_is_object_file_in_memory.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    lto_module_is_object_file_in_memory_for_target = _libraries['llvm'].lto_module_is_object_file_in_memory_for_target
    lto_module_is_object_file_in_memory_for_target.restype = lto_bool_t
    lto_module_is_object_file_in_memory_for_target.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_create = _libraries['llvm'].lto_module_create
    lto_module_create.restype = lto_module_t
    lto_module_create.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_create_from_memory = _libraries['llvm'].lto_module_create_from_memory
    lto_module_create_from_memory.restype = lto_module_t
    lto_module_create_from_memory.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    lto_module_create_from_memory_with_path = _libraries['llvm'].lto_module_create_from_memory_with_path
    lto_module_create_from_memory_with_path.restype = lto_module_t
    lto_module_create_from_memory_with_path.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_create_in_local_context = _libraries['llvm'].lto_module_create_in_local_context
    lto_module_create_in_local_context.restype = lto_module_t
    lto_module_create_in_local_context.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_create_in_codegen_context = _libraries['llvm'].lto_module_create_in_codegen_context
    lto_module_create_in_codegen_context.restype = lto_module_t
    lto_module_create_in_codegen_context.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), lto_code_gen_t]
except AttributeError:
    pass
try:
    lto_module_create_from_fd = _libraries['llvm'].lto_module_create_from_fd
    lto_module_create_from_fd.restype = lto_module_t
    lto_module_create_from_fd.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
off_t = ctypes.c_int64
try:
    lto_module_create_from_fd_at_offset = _libraries['llvm'].lto_module_create_from_fd_at_offset
    lto_module_create_from_fd_at_offset.restype = lto_module_t
    lto_module_create_from_fd_at_offset.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t, size_t, off_t]
except AttributeError:
    pass
try:
    lto_module_dispose = _libraries['llvm'].lto_module_dispose
    lto_module_dispose.restype = None
    lto_module_dispose.argtypes = [lto_module_t]
except AttributeError:
    pass
try:
    lto_module_get_target_triple = _libraries['llvm'].lto_module_get_target_triple
    lto_module_get_target_triple.restype = ctypes.POINTER(ctypes.c_char)
    lto_module_get_target_triple.argtypes = [lto_module_t]
except AttributeError:
    pass
try:
    lto_module_set_target_triple = _libraries['llvm'].lto_module_set_target_triple
    lto_module_set_target_triple.restype = None
    lto_module_set_target_triple.argtypes = [lto_module_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_module_get_num_symbols = _libraries['llvm'].lto_module_get_num_symbols
    lto_module_get_num_symbols.restype = ctypes.c_uint32
    lto_module_get_num_symbols.argtypes = [lto_module_t]
except AttributeError:
    pass
try:
    lto_module_get_symbol_name = _libraries['llvm'].lto_module_get_symbol_name
    lto_module_get_symbol_name.restype = ctypes.POINTER(ctypes.c_char)
    lto_module_get_symbol_name.argtypes = [lto_module_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lto_module_get_symbol_attribute = _libraries['llvm'].lto_module_get_symbol_attribute
    lto_module_get_symbol_attribute.restype = lto_symbol_attributes
    lto_module_get_symbol_attribute.argtypes = [lto_module_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lto_module_get_linkeropts = _libraries['llvm'].lto_module_get_linkeropts
    lto_module_get_linkeropts.restype = ctypes.POINTER(ctypes.c_char)
    lto_module_get_linkeropts.argtypes = [lto_module_t]
except AttributeError:
    pass
try:
    lto_module_get_macho_cputype = _libraries['llvm'].lto_module_get_macho_cputype
    lto_module_get_macho_cputype.restype = lto_bool_t
    lto_module_get_macho_cputype.argtypes = [lto_module_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    lto_module_has_ctor_dtor = _libraries['llvm'].lto_module_has_ctor_dtor
    lto_module_has_ctor_dtor.restype = lto_bool_t
    lto_module_has_ctor_dtor.argtypes = [lto_module_t]
except AttributeError:
    pass

# values for enumeration 'c__EA_lto_codegen_diagnostic_severity_t'
c__EA_lto_codegen_diagnostic_severity_t__enumvalues = {
    0: 'LTO_DS_ERROR',
    1: 'LTO_DS_WARNING',
    3: 'LTO_DS_REMARK',
    2: 'LTO_DS_NOTE',
}
LTO_DS_ERROR = 0
LTO_DS_WARNING = 1
LTO_DS_REMARK = 3
LTO_DS_NOTE = 2
c__EA_lto_codegen_diagnostic_severity_t = ctypes.c_uint32 # enum
lto_codegen_diagnostic_severity_t = c__EA_lto_codegen_diagnostic_severity_t
lto_codegen_diagnostic_severity_t__enumvalues = c__EA_lto_codegen_diagnostic_severity_t__enumvalues
lto_diagnostic_handler_t = ctypes.CFUNCTYPE(None, c__EA_lto_codegen_diagnostic_severity_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None))
try:
    lto_codegen_set_diagnostic_handler = _libraries['llvm'].lto_codegen_set_diagnostic_handler
    lto_codegen_set_diagnostic_handler.restype = None
    lto_codegen_set_diagnostic_handler.argtypes = [lto_code_gen_t, lto_diagnostic_handler_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    lto_codegen_create = _libraries['llvm'].lto_codegen_create
    lto_codegen_create.restype = lto_code_gen_t
    lto_codegen_create.argtypes = []
except AttributeError:
    pass
try:
    lto_codegen_create_in_local_context = _libraries['llvm'].lto_codegen_create_in_local_context
    lto_codegen_create_in_local_context.restype = lto_code_gen_t
    lto_codegen_create_in_local_context.argtypes = []
except AttributeError:
    pass
try:
    lto_codegen_dispose = _libraries['llvm'].lto_codegen_dispose
    lto_codegen_dispose.restype = None
    lto_codegen_dispose.argtypes = [lto_code_gen_t]
except AttributeError:
    pass
try:
    lto_codegen_add_module = _libraries['llvm'].lto_codegen_add_module
    lto_codegen_add_module.restype = lto_bool_t
    lto_codegen_add_module.argtypes = [lto_code_gen_t, lto_module_t]
except AttributeError:
    pass
try:
    lto_codegen_set_module = _libraries['llvm'].lto_codegen_set_module
    lto_codegen_set_module.restype = None
    lto_codegen_set_module.argtypes = [lto_code_gen_t, lto_module_t]
except AttributeError:
    pass
try:
    lto_codegen_set_debug_model = _libraries['llvm'].lto_codegen_set_debug_model
    lto_codegen_set_debug_model.restype = lto_bool_t
    lto_codegen_set_debug_model.argtypes = [lto_code_gen_t, lto_debug_model]
except AttributeError:
    pass
try:
    lto_codegen_set_pic_model = _libraries['llvm'].lto_codegen_set_pic_model
    lto_codegen_set_pic_model.restype = lto_bool_t
    lto_codegen_set_pic_model.argtypes = [lto_code_gen_t, lto_codegen_model]
except AttributeError:
    pass
try:
    lto_codegen_set_cpu = _libraries['llvm'].lto_codegen_set_cpu
    lto_codegen_set_cpu.restype = None
    lto_codegen_set_cpu.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_codegen_set_assembler_path = _libraries['llvm'].lto_codegen_set_assembler_path
    lto_codegen_set_assembler_path.restype = None
    lto_codegen_set_assembler_path.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_codegen_set_assembler_args = _libraries['llvm'].lto_codegen_set_assembler_args
    lto_codegen_set_assembler_args.restype = None
    lto_codegen_set_assembler_args.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError:
    pass
try:
    lto_codegen_add_must_preserve_symbol = _libraries['llvm'].lto_codegen_add_must_preserve_symbol
    lto_codegen_add_must_preserve_symbol.restype = None
    lto_codegen_add_must_preserve_symbol.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_codegen_write_merged_modules = _libraries['llvm'].lto_codegen_write_merged_modules
    lto_codegen_write_merged_modules.restype = lto_bool_t
    lto_codegen_write_merged_modules.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_codegen_compile = _libraries['llvm'].lto_codegen_compile
    lto_codegen_compile.restype = ctypes.POINTER(None)
    lto_codegen_compile.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    lto_codegen_compile_to_file = _libraries['llvm'].lto_codegen_compile_to_file
    lto_codegen_compile_to_file.restype = lto_bool_t
    lto_codegen_compile_to_file.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    lto_codegen_optimize = _libraries['llvm'].lto_codegen_optimize
    lto_codegen_optimize.restype = lto_bool_t
    lto_codegen_optimize.argtypes = [lto_code_gen_t]
except AttributeError:
    pass
try:
    lto_codegen_compile_optimized = _libraries['llvm'].lto_codegen_compile_optimized
    lto_codegen_compile_optimized.restype = ctypes.POINTER(None)
    lto_codegen_compile_optimized.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    lto_api_version = _libraries['llvm'].lto_api_version
    lto_api_version.restype = ctypes.c_uint32
    lto_api_version.argtypes = []
except AttributeError:
    pass
try:
    lto_set_debug_options = _libraries['llvm'].lto_set_debug_options
    lto_set_debug_options.restype = None
    lto_set_debug_options.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError:
    pass
try:
    lto_codegen_debug_options = _libraries['llvm'].lto_codegen_debug_options
    lto_codegen_debug_options.restype = None
    lto_codegen_debug_options.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_codegen_debug_options_array = _libraries['llvm'].lto_codegen_debug_options_array
    lto_codegen_debug_options_array.restype = None
    lto_codegen_debug_options_array.argtypes = [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError:
    pass
try:
    lto_initialize_disassembler = _libraries['llvm'].lto_initialize_disassembler
    lto_initialize_disassembler.restype = None
    lto_initialize_disassembler.argtypes = []
except AttributeError:
    pass
try:
    lto_codegen_set_should_internalize = _libraries['llvm'].lto_codegen_set_should_internalize
    lto_codegen_set_should_internalize.restype = None
    lto_codegen_set_should_internalize.argtypes = [lto_code_gen_t, lto_bool_t]
except AttributeError:
    pass
try:
    lto_codegen_set_should_embed_uselists = _libraries['llvm'].lto_codegen_set_should_embed_uselists
    lto_codegen_set_should_embed_uselists.restype = None
    lto_codegen_set_should_embed_uselists.argtypes = [lto_code_gen_t, lto_bool_t]
except AttributeError:
    pass
class struct_LLVMOpaqueLTOInput(Structure):
    pass

lto_input_t = ctypes.POINTER(struct_LLVMOpaqueLTOInput)
try:
    lto_input_create = _libraries['llvm'].lto_input_create
    lto_input_create.restype = lto_input_t
    lto_input_create.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lto_input_dispose = _libraries['llvm'].lto_input_dispose
    lto_input_dispose.restype = None
    lto_input_dispose.argtypes = [lto_input_t]
except AttributeError:
    pass
try:
    lto_input_get_num_dependent_libraries = _libraries['llvm'].lto_input_get_num_dependent_libraries
    lto_input_get_num_dependent_libraries.restype = ctypes.c_uint32
    lto_input_get_num_dependent_libraries.argtypes = [lto_input_t]
except AttributeError:
    pass
try:
    lto_input_get_dependent_library = _libraries['llvm'].lto_input_get_dependent_library
    lto_input_get_dependent_library.restype = ctypes.POINTER(ctypes.c_char)
    lto_input_get_dependent_library.argtypes = [lto_input_t, size_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    lto_runtime_lib_symbols_list = _libraries['llvm'].lto_runtime_lib_symbols_list
    lto_runtime_lib_symbols_list.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
    lto_runtime_lib_symbols_list.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
class struct_c__SA_LTOObjectBuffer(Structure):
    pass

struct_c__SA_LTOObjectBuffer._pack_ = 1 # source:False
struct_c__SA_LTOObjectBuffer._fields_ = [
    ('Buffer', ctypes.POINTER(ctypes.c_char)),
    ('Size', ctypes.c_uint64),
]

LTOObjectBuffer = struct_c__SA_LTOObjectBuffer
try:
    thinlto_create_codegen = _libraries['llvm'].thinlto_create_codegen
    thinlto_create_codegen.restype = thinlto_code_gen_t
    thinlto_create_codegen.argtypes = []
except AttributeError:
    pass
try:
    thinlto_codegen_dispose = _libraries['llvm'].thinlto_codegen_dispose
    thinlto_codegen_dispose.restype = None
    thinlto_codegen_dispose.argtypes = [thinlto_code_gen_t]
except AttributeError:
    pass
try:
    thinlto_codegen_add_module = _libraries['llvm'].thinlto_codegen_add_module
    thinlto_codegen_add_module.restype = None
    thinlto_codegen_add_module.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    thinlto_codegen_process = _libraries['llvm'].thinlto_codegen_process
    thinlto_codegen_process.restype = None
    thinlto_codegen_process.argtypes = [thinlto_code_gen_t]
except AttributeError:
    pass
try:
    thinlto_module_get_num_objects = _libraries['llvm'].thinlto_module_get_num_objects
    thinlto_module_get_num_objects.restype = ctypes.c_uint32
    thinlto_module_get_num_objects.argtypes = [thinlto_code_gen_t]
except AttributeError:
    pass
try:
    thinlto_module_get_object = _libraries['llvm'].thinlto_module_get_object
    thinlto_module_get_object.restype = LTOObjectBuffer
    thinlto_module_get_object.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_module_get_num_object_files = _libraries['llvm'].thinlto_module_get_num_object_files
    thinlto_module_get_num_object_files.restype = ctypes.c_uint32
    thinlto_module_get_num_object_files.argtypes = [thinlto_code_gen_t]
except AttributeError:
    pass
try:
    thinlto_module_get_object_file = _libraries['llvm'].thinlto_module_get_object_file
    thinlto_module_get_object_file.restype = ctypes.POINTER(ctypes.c_char)
    thinlto_module_get_object_file.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_pic_model = _libraries['llvm'].thinlto_codegen_set_pic_model
    thinlto_codegen_set_pic_model.restype = lto_bool_t
    thinlto_codegen_set_pic_model.argtypes = [thinlto_code_gen_t, lto_codegen_model]
except AttributeError:
    pass
try:
    thinlto_codegen_set_savetemps_dir = _libraries['llvm'].thinlto_codegen_set_savetemps_dir
    thinlto_codegen_set_savetemps_dir.restype = None
    thinlto_codegen_set_savetemps_dir.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    thinlto_set_generated_objects_dir = _libraries['llvm'].thinlto_set_generated_objects_dir
    thinlto_set_generated_objects_dir.restype = None
    thinlto_set_generated_objects_dir.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cpu = _libraries['llvm'].thinlto_codegen_set_cpu
    thinlto_codegen_set_cpu.restype = None
    thinlto_codegen_set_cpu.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    thinlto_codegen_disable_codegen = _libraries['llvm'].thinlto_codegen_disable_codegen
    thinlto_codegen_disable_codegen.restype = None
    thinlto_codegen_disable_codegen.argtypes = [thinlto_code_gen_t, lto_bool_t]
except AttributeError:
    pass
try:
    thinlto_codegen_set_codegen_only = _libraries['llvm'].thinlto_codegen_set_codegen_only
    thinlto_codegen_set_codegen_only.restype = None
    thinlto_codegen_set_codegen_only.argtypes = [thinlto_code_gen_t, lto_bool_t]
except AttributeError:
    pass
try:
    thinlto_debug_options = _libraries['llvm'].thinlto_debug_options
    thinlto_debug_options.restype = None
    thinlto_debug_options.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError:
    pass
try:
    lto_module_is_thinlto = _libraries['llvm'].lto_module_is_thinlto
    lto_module_is_thinlto.restype = lto_bool_t
    lto_module_is_thinlto.argtypes = [lto_module_t]
except AttributeError:
    pass
try:
    thinlto_codegen_add_must_preserve_symbol = _libraries['llvm'].thinlto_codegen_add_must_preserve_symbol
    thinlto_codegen_add_must_preserve_symbol.restype = None
    thinlto_codegen_add_must_preserve_symbol.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    thinlto_codegen_add_cross_referenced_symbol = _libraries['llvm'].thinlto_codegen_add_cross_referenced_symbol
    thinlto_codegen_add_cross_referenced_symbol.restype = None
    thinlto_codegen_add_cross_referenced_symbol.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_dir = _libraries['llvm'].thinlto_codegen_set_cache_dir
    thinlto_codegen_set_cache_dir.restype = None
    thinlto_codegen_set_cache_dir.argtypes = [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_pruning_interval = _libraries['llvm'].thinlto_codegen_set_cache_pruning_interval
    thinlto_codegen_set_cache_pruning_interval.restype = None
    thinlto_codegen_set_cache_pruning_interval.argtypes = [thinlto_code_gen_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_final_cache_size_relative_to_available_space = _libraries['llvm'].thinlto_codegen_set_final_cache_size_relative_to_available_space
    thinlto_codegen_set_final_cache_size_relative_to_available_space.restype = None
    thinlto_codegen_set_final_cache_size_relative_to_available_space.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_entry_expiration = _libraries['llvm'].thinlto_codegen_set_cache_entry_expiration
    thinlto_codegen_set_cache_entry_expiration.restype = None
    thinlto_codegen_set_cache_entry_expiration.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_size_bytes = _libraries['llvm'].thinlto_codegen_set_cache_size_bytes
    thinlto_codegen_set_cache_size_bytes.restype = None
    thinlto_codegen_set_cache_size_bytes.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_size_megabytes = _libraries['llvm'].thinlto_codegen_set_cache_size_megabytes
    thinlto_codegen_set_cache_size_megabytes.restype = None
    thinlto_codegen_set_cache_size_megabytes.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    thinlto_codegen_set_cache_size_files = _libraries['llvm'].thinlto_codegen_set_cache_size_files
    thinlto_codegen_set_cache_size_files.restype = None
    thinlto_codegen_set_cache_size_files.argtypes = [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError:
    pass
__all__ = \
    ['LLVMABIAlignmentOfType', 'LLVMABISizeOfType',
    'LLVMAMDGPUCSCallConv', 'LLVMAMDGPUESCallConv',
    'LLVMAMDGPUGSCallConv', 'LLVMAMDGPUHSCallConv',
    'LLVMAMDGPUKERNELCallConv', 'LLVMAMDGPULSCallConv',
    'LLVMAMDGPUPSCallConv', 'LLVMAMDGPUVSCallConv',
    'LLVMARMAAPCSCallConv', 'LLVMARMAAPCSVFPCallConv',
    'LLVMARMAPCSCallConv', 'LLVMAShr', 'LLVMAVRBUILTINCallConv',
    'LLVMAVRINTRCallConv', 'LLVMAVRSIGNALCallConv',
    'LLVMAbortProcessAction', 'LLVMAdd',
    'LLVMAddAddDiscriminatorsPass', 'LLVMAddAggressiveDCEPass',
    'LLVMAddAggressiveInstCombinerPass', 'LLVMAddAlias',
    'LLVMAddAlias2', 'LLVMAddAlignmentFromAssumptionsPass',
    'LLVMAddAlwaysInlinerPass', 'LLVMAddAnalysisPasses',
    'LLVMAddArgumentPromotionPass', 'LLVMAddAttributeAtIndex',
    'LLVMAddBasicAliasAnalysisPass', 'LLVMAddBitTrackingDCEPass',
    'LLVMAddCFGSimplificationPass', 'LLVMAddCallSiteAttribute',
    'LLVMAddCalledValuePropagationPass', 'LLVMAddCase',
    'LLVMAddClause', 'LLVMAddConstantMergePass',
    'LLVMAddCoroCleanupPass', 'LLVMAddCoroEarlyPass',
    'LLVMAddCoroElidePass', 'LLVMAddCoroSplitPass',
    'LLVMAddCorrelatedValuePropagationPass', 'LLVMAddDCEPass',
    'LLVMAddDeadArgEliminationPass',
    'LLVMAddDeadStoreEliminationPass',
    'LLVMAddDemoteMemoryToRegisterPass', 'LLVMAddDestination',
    'LLVMAddEarlyCSEMemSSAPass', 'LLVMAddEarlyCSEPass',
    'LLVMAddFunction', 'LLVMAddFunctionAttrsPass',
    'LLVMAddFunctionInliningPass', 'LLVMAddGVNPass', 'LLVMAddGlobal',
    'LLVMAddGlobalDCEPass', 'LLVMAddGlobalIFunc',
    'LLVMAddGlobalInAddressSpace', 'LLVMAddGlobalMapping',
    'LLVMAddGlobalOptimizerPass', 'LLVMAddHandler',
    'LLVMAddIPSCCPPass', 'LLVMAddIncoming',
    'LLVMAddIndVarSimplifyPass', 'LLVMAddInstructionCombiningPass',
    'LLVMAddInstructionSimplifyPass', 'LLVMAddInternalizePass',
    'LLVMAddInternalizePassWithMustPreservePredicate',
    'LLVMAddJumpThreadingPass', 'LLVMAddLICMPass',
    'LLVMAddLoopDeletionPass', 'LLVMAddLoopIdiomPass',
    'LLVMAddLoopRerollPass', 'LLVMAddLoopRotatePass',
    'LLVMAddLoopUnrollAndJamPass', 'LLVMAddLoopUnrollPass',
    'LLVMAddLoopUnswitchPass', 'LLVMAddLoopVectorizePass',
    'LLVMAddLowerAtomicPass', 'LLVMAddLowerConstantIntrinsicsPass',
    'LLVMAddLowerExpectIntrinsicPass', 'LLVMAddLowerSwitchPass',
    'LLVMAddMemCpyOptPass', 'LLVMAddMergeFunctionsPass',
    'LLVMAddMergedLoadStoreMotionPass', 'LLVMAddMetadataToInst',
    'LLVMAddModule', 'LLVMAddModuleFlag',
    'LLVMAddNamedMetadataOperand', 'LLVMAddNewGVNPass',
    'LLVMAddPartiallyInlineLibCallsPass',
    'LLVMAddPromoteMemoryToRegisterPass', 'LLVMAddPruneEHPass',
    'LLVMAddReassociatePass', 'LLVMAddSCCPPass',
    'LLVMAddSLPVectorizePass', 'LLVMAddScalarReplAggregatesPass',
    'LLVMAddScalarReplAggregatesPassSSA',
    'LLVMAddScalarReplAggregatesPassWithThreshold',
    'LLVMAddScalarizerPass', 'LLVMAddScopedNoAliasAAPass',
    'LLVMAddSimplifyLibCallsPass', 'LLVMAddStripDeadPrototypesPass',
    'LLVMAddStripSymbolsPass', 'LLVMAddSymbol',
    'LLVMAddTailCallEliminationPass',
    'LLVMAddTargetDependentFunctionAttr', 'LLVMAddTargetLibraryInfo',
    'LLVMAddTypeBasedAliasAnalysisPass',
    'LLVMAddUnifyFunctionExitNodesPass', 'LLVMAddVerifierPass',
    'LLVMAddrSpaceCast', 'LLVMAliasGetAliasee', 'LLVMAliasSetAliasee',
    'LLVMAlignOf', 'LLVMAlloca', 'LLVMAnd',
    'LLVMAnyComdatSelectionKind', 'LLVMAnyRegCallConv',
    'LLVMAppendBasicBlock', 'LLVMAppendBasicBlockInContext',
    'LLVMAppendExistingBasicBlock', 'LLVMAppendModuleInlineAsm',
    'LLVMAppendingLinkage', 'LLVMArgumentValueKind', 'LLVMArrayType',
    'LLVMArrayTypeKind', 'LLVMAssemblyFile', 'LLVMAtomicCmpXchg',
    'LLVMAtomicOrdering', 'LLVMAtomicOrderingAcquire',
    'LLVMAtomicOrderingAcquireRelease', 'LLVMAtomicOrderingMonotonic',
    'LLVMAtomicOrderingNotAtomic', 'LLVMAtomicOrderingRelease',
    'LLVMAtomicOrderingSequentiallyConsistent',
    'LLVMAtomicOrderingUnordered', 'LLVMAtomicOrdering__enumvalues',
    'LLVMAtomicRMW', 'LLVMAtomicRMWBinOp', 'LLVMAtomicRMWBinOpAdd',
    'LLVMAtomicRMWBinOpAnd', 'LLVMAtomicRMWBinOpFAdd',
    'LLVMAtomicRMWBinOpFSub', 'LLVMAtomicRMWBinOpMax',
    'LLVMAtomicRMWBinOpMin', 'LLVMAtomicRMWBinOpNand',
    'LLVMAtomicRMWBinOpOr', 'LLVMAtomicRMWBinOpSub',
    'LLVMAtomicRMWBinOpUMax', 'LLVMAtomicRMWBinOpUMin',
    'LLVMAtomicRMWBinOpXchg', 'LLVMAtomicRMWBinOpXor',
    'LLVMAtomicRMWBinOp__enumvalues', 'LLVMAttributeFunctionIndex',
    'LLVMAttributeIndex', 'LLVMAttributeRef',
    'LLVMAttributeReturnIndex', 'LLVMAvailableExternallyLinkage',
    'LLVMBFloatType', 'LLVMBFloatTypeInContext', 'LLVMBFloatTypeKind',
    'LLVMBasicBlockAsValue', 'LLVMBasicBlockRef',
    'LLVMBasicBlockValueKind', 'LLVMBigEndian',
    'LLVMBinaryCopyMemoryBuffer', 'LLVMBinaryGetType',
    'LLVMBinaryRef', 'LLVMBinaryType', 'LLVMBinaryTypeArchive',
    'LLVMBinaryTypeCOFF', 'LLVMBinaryTypeCOFFImportFile',
    'LLVMBinaryTypeELF32B', 'LLVMBinaryTypeELF32L',
    'LLVMBinaryTypeELF64B', 'LLVMBinaryTypeELF64L',
    'LLVMBinaryTypeIR', 'LLVMBinaryTypeMachO32B',
    'LLVMBinaryTypeMachO32L', 'LLVMBinaryTypeMachO64B',
    'LLVMBinaryTypeMachO64L', 'LLVMBinaryTypeMachOUniversalBinary',
    'LLVMBinaryTypeWasm', 'LLVMBinaryTypeWinRes',
    'LLVMBinaryType__enumvalues', 'LLVMBitCast', 'LLVMBlockAddress',
    'LLVMBlockAddressValueKind', 'LLVMBool', 'LLVMBr',
    'LLVMBuildAShr', 'LLVMBuildAdd', 'LLVMBuildAddrSpaceCast',
    'LLVMBuildAggregateRet', 'LLVMBuildAlloca', 'LLVMBuildAnd',
    'LLVMBuildArrayAlloca', 'LLVMBuildArrayMalloc',
    'LLVMBuildAtomicCmpXchg', 'LLVMBuildAtomicRMW', 'LLVMBuildBinOp',
    'LLVMBuildBitCast', 'LLVMBuildBr', 'LLVMBuildCall',
    'LLVMBuildCall2', 'LLVMBuildCast', 'LLVMBuildCatchPad',
    'LLVMBuildCatchRet', 'LLVMBuildCatchSwitch',
    'LLVMBuildCleanupPad', 'LLVMBuildCleanupRet', 'LLVMBuildCondBr',
    'LLVMBuildExactSDiv', 'LLVMBuildExactUDiv',
    'LLVMBuildExtractElement', 'LLVMBuildExtractValue',
    'LLVMBuildFAdd', 'LLVMBuildFCmp', 'LLVMBuildFDiv',
    'LLVMBuildFMul', 'LLVMBuildFNeg', 'LLVMBuildFPCast',
    'LLVMBuildFPExt', 'LLVMBuildFPToSI', 'LLVMBuildFPToUI',
    'LLVMBuildFPTrunc', 'LLVMBuildFRem', 'LLVMBuildFSub',
    'LLVMBuildFence', 'LLVMBuildFree', 'LLVMBuildFreeze',
    'LLVMBuildGEP', 'LLVMBuildGEP2', 'LLVMBuildGlobalString',
    'LLVMBuildGlobalStringPtr', 'LLVMBuildICmp',
    'LLVMBuildInBoundsGEP', 'LLVMBuildInBoundsGEP2',
    'LLVMBuildIndirectBr', 'LLVMBuildInsertElement',
    'LLVMBuildInsertValue', 'LLVMBuildIntCast', 'LLVMBuildIntCast2',
    'LLVMBuildIntToPtr', 'LLVMBuildInvoke', 'LLVMBuildInvoke2',
    'LLVMBuildIsNotNull', 'LLVMBuildIsNull', 'LLVMBuildLShr',
    'LLVMBuildLandingPad', 'LLVMBuildLoad', 'LLVMBuildLoad2',
    'LLVMBuildMalloc', 'LLVMBuildMemCpy', 'LLVMBuildMemMove',
    'LLVMBuildMemSet', 'LLVMBuildMul', 'LLVMBuildNSWAdd',
    'LLVMBuildNSWMul', 'LLVMBuildNSWNeg', 'LLVMBuildNSWSub',
    'LLVMBuildNUWAdd', 'LLVMBuildNUWMul', 'LLVMBuildNUWNeg',
    'LLVMBuildNUWSub', 'LLVMBuildNeg', 'LLVMBuildNot', 'LLVMBuildOr',
    'LLVMBuildPhi', 'LLVMBuildPointerCast', 'LLVMBuildPtrDiff',
    'LLVMBuildPtrDiff2', 'LLVMBuildPtrToInt', 'LLVMBuildResume',
    'LLVMBuildRet', 'LLVMBuildRetVoid', 'LLVMBuildSDiv',
    'LLVMBuildSExt', 'LLVMBuildSExtOrBitCast', 'LLVMBuildSIToFP',
    'LLVMBuildSRem', 'LLVMBuildSelect', 'LLVMBuildShl',
    'LLVMBuildShuffleVector', 'LLVMBuildStore', 'LLVMBuildStructGEP',
    'LLVMBuildStructGEP2', 'LLVMBuildSub', 'LLVMBuildSwitch',
    'LLVMBuildTrunc', 'LLVMBuildTruncOrBitCast', 'LLVMBuildUDiv',
    'LLVMBuildUIToFP', 'LLVMBuildURem', 'LLVMBuildUnreachable',
    'LLVMBuildVAArg', 'LLVMBuildXor', 'LLVMBuildZExt',
    'LLVMBuildZExtOrBitCast', 'LLVMBuilderGetDefaultFPMathTag',
    'LLVMBuilderRef', 'LLVMBuilderSetDefaultFPMathTag',
    'LLVMByteOrder', 'LLVMByteOrdering', 'LLVMCCallConv',
    'LLVMCXXFASTTLSCallConv', 'LLVMCall', 'LLVMCallBr',
    'LLVMCallConv', 'LLVMCallConv__enumvalues',
    'LLVMCallFrameAlignmentOfType', 'LLVMCatchPad', 'LLVMCatchRet',
    'LLVMCatchSwitch', 'LLVMCleanupPad', 'LLVMCleanupRet',
    'LLVMClearInsertionPosition', 'LLVMCloneModule',
    'LLVMCodeGenFileType', 'LLVMCodeGenFileType__enumvalues',
    'LLVMCodeGenLevelAggressive', 'LLVMCodeGenLevelDefault',
    'LLVMCodeGenLevelLess', 'LLVMCodeGenLevelNone',
    'LLVMCodeGenOptLevel', 'LLVMCodeGenOptLevel__enumvalues',
    'LLVMCodeModel', 'LLVMCodeModelDefault',
    'LLVMCodeModelJITDefault', 'LLVMCodeModelKernel',
    'LLVMCodeModelLarge', 'LLVMCodeModelMedium', 'LLVMCodeModelSmall',
    'LLVMCodeModelTiny', 'LLVMCodeModel__enumvalues',
    'LLVMColdCallConv', 'LLVMComdatRef', 'LLVMComdatSelectionKind',
    'LLVMComdatSelectionKind__enumvalues', 'LLVMCommonLinkage',
    'LLVMConstAShr', 'LLVMConstAdd', 'LLVMConstAddrSpaceCast',
    'LLVMConstAllOnes', 'LLVMConstAnd', 'LLVMConstArray',
    'LLVMConstBitCast', 'LLVMConstExactSDiv', 'LLVMConstExactUDiv',
    'LLVMConstExtractElement', 'LLVMConstExtractValue',
    'LLVMConstFAdd', 'LLVMConstFCmp', 'LLVMConstFDiv',
    'LLVMConstFMul', 'LLVMConstFNeg', 'LLVMConstFPCast',
    'LLVMConstFPExt', 'LLVMConstFPToSI', 'LLVMConstFPToUI',
    'LLVMConstFPTrunc', 'LLVMConstFRem', 'LLVMConstFSub',
    'LLVMConstGEP', 'LLVMConstGEP2', 'LLVMConstICmp',
    'LLVMConstInBoundsGEP', 'LLVMConstInBoundsGEP2',
    'LLVMConstInlineAsm', 'LLVMConstInsertElement',
    'LLVMConstInsertValue', 'LLVMConstInt', 'LLVMConstIntCast',
    'LLVMConstIntGetSExtValue', 'LLVMConstIntGetZExtValue',
    'LLVMConstIntOfArbitraryPrecision', 'LLVMConstIntOfString',
    'LLVMConstIntOfStringAndSize', 'LLVMConstIntToPtr',
    'LLVMConstLShr', 'LLVMConstMul', 'LLVMConstNSWAdd',
    'LLVMConstNSWMul', 'LLVMConstNSWNeg', 'LLVMConstNSWSub',
    'LLVMConstNUWAdd', 'LLVMConstNUWMul', 'LLVMConstNUWNeg',
    'LLVMConstNUWSub', 'LLVMConstNamedStruct', 'LLVMConstNeg',
    'LLVMConstNot', 'LLVMConstNull', 'LLVMConstOr',
    'LLVMConstPointerCast', 'LLVMConstPointerNull',
    'LLVMConstPtrToInt', 'LLVMConstReal', 'LLVMConstRealGetDouble',
    'LLVMConstRealOfString', 'LLVMConstRealOfStringAndSize',
    'LLVMConstSDiv', 'LLVMConstSExt', 'LLVMConstSExtOrBitCast',
    'LLVMConstSIToFP', 'LLVMConstSRem', 'LLVMConstSelect',
    'LLVMConstShl', 'LLVMConstShuffleVector', 'LLVMConstString',
    'LLVMConstStringInContext', 'LLVMConstStruct',
    'LLVMConstStructInContext', 'LLVMConstSub', 'LLVMConstTrunc',
    'LLVMConstTruncOrBitCast', 'LLVMConstUDiv', 'LLVMConstUIToFP',
    'LLVMConstURem', 'LLVMConstVector', 'LLVMConstXor',
    'LLVMConstZExt', 'LLVMConstZExtOrBitCast',
    'LLVMConstantAggregateZeroValueKind',
    'LLVMConstantArrayValueKind',
    'LLVMConstantAsMetadataMetadataKind',
    'LLVMConstantDataArrayValueKind',
    'LLVMConstantDataVectorValueKind', 'LLVMConstantExprValueKind',
    'LLVMConstantFPValueKind', 'LLVMConstantIntValueKind',
    'LLVMConstantPointerNullValueKind', 'LLVMConstantStructValueKind',
    'LLVMConstantTokenNoneValueKind', 'LLVMConstantVectorValueKind',
    'LLVMConsumeError', 'LLVMContextCreate', 'LLVMContextDispose',
    'LLVMContextGetDiagnosticContext',
    'LLVMContextGetDiagnosticHandler', 'LLVMContextRef',
    'LLVMContextSetDiagnosticHandler',
    'LLVMContextSetDiscardValueNames', 'LLVMContextSetYieldCallback',
    'LLVMContextShouldDiscardValueNames',
    'LLVMCopyModuleFlagsMetadata', 'LLVMCopyStringRepOfTargetData',
    'LLVMCountBasicBlocks', 'LLVMCountIncoming',
    'LLVMCountParamTypes', 'LLVMCountParams',
    'LLVMCountStructElementTypes', 'LLVMCreateBasicBlockInContext',
    'LLVMCreateBinary', 'LLVMCreateBuilder',
    'LLVMCreateBuilderInContext', 'LLVMCreateDIBuilder',
    'LLVMCreateDIBuilderDisallowUnresolved', 'LLVMCreateDisasm',
    'LLVMCreateDisasmCPU', 'LLVMCreateDisasmCPUFeatures',
    'LLVMCreateEnumAttribute', 'LLVMCreateExecutionEngineForModule',
    'LLVMCreateFunctionPassManager',
    'LLVMCreateFunctionPassManagerForModule',
    'LLVMCreateGDBRegistrationListener',
    'LLVMCreateGenericValueOfFloat', 'LLVMCreateGenericValueOfInt',
    'LLVMCreateGenericValueOfPointer',
    'LLVMCreateIntelJITEventListener',
    'LLVMCreateInterpreterForModule',
    'LLVMCreateJITCompilerForModule',
    'LLVMCreateMCJITCompilerForModule',
    'LLVMCreateMemoryBufferWithContentsOfFile',
    'LLVMCreateMemoryBufferWithMemoryRange',
    'LLVMCreateMemoryBufferWithMemoryRangeCopy',
    'LLVMCreateMemoryBufferWithSTDIN', 'LLVMCreateMessage',
    'LLVMCreateModuleProviderForExistingModule',
    'LLVMCreateOProfileJITEventListener', 'LLVMCreateObjectFile',
    'LLVMCreatePassBuilderOptions', 'LLVMCreatePassManager',
    'LLVMCreatePerfJITEventListener',
    'LLVMCreateSimpleMCJITMemoryManager', 'LLVMCreateStringAttribute',
    'LLVMCreateStringError', 'LLVMCreateTargetData',
    'LLVMCreateTargetDataLayout', 'LLVMCreateTargetMachine',
    'LLVMCreateTypeAttribute', 'LLVMDIArgListMetadataKind',
    'LLVMDIBasicTypeMetadataKind', 'LLVMDIBuilderCreateArrayType',
    'LLVMDIBuilderCreateArtificialType',
    'LLVMDIBuilderCreateAutoVariable', 'LLVMDIBuilderCreateBasicType',
    'LLVMDIBuilderCreateBitFieldMemberType',
    'LLVMDIBuilderCreateClassType', 'LLVMDIBuilderCreateCompileUnit',
    'LLVMDIBuilderCreateConstantValueExpression',
    'LLVMDIBuilderCreateDebugLocation',
    'LLVMDIBuilderCreateEnumerationType',
    'LLVMDIBuilderCreateEnumerator', 'LLVMDIBuilderCreateExpression',
    'LLVMDIBuilderCreateFile', 'LLVMDIBuilderCreateForwardDecl',
    'LLVMDIBuilderCreateFunction',
    'LLVMDIBuilderCreateGlobalVariableExpression',
    'LLVMDIBuilderCreateImportedDeclaration',
    'LLVMDIBuilderCreateImportedModuleFromAlias',
    'LLVMDIBuilderCreateImportedModuleFromModule',
    'LLVMDIBuilderCreateImportedModuleFromNamespace',
    'LLVMDIBuilderCreateInheritance',
    'LLVMDIBuilderCreateLexicalBlock',
    'LLVMDIBuilderCreateLexicalBlockFile', 'LLVMDIBuilderCreateMacro',
    'LLVMDIBuilderCreateMemberPointerType',
    'LLVMDIBuilderCreateMemberType', 'LLVMDIBuilderCreateModule',
    'LLVMDIBuilderCreateNameSpace', 'LLVMDIBuilderCreateNullPtrType',
    'LLVMDIBuilderCreateObjCIVar', 'LLVMDIBuilderCreateObjCProperty',
    'LLVMDIBuilderCreateObjectPointerType',
    'LLVMDIBuilderCreateParameterVariable',
    'LLVMDIBuilderCreatePointerType',
    'LLVMDIBuilderCreateQualifiedType',
    'LLVMDIBuilderCreateReferenceType',
    'LLVMDIBuilderCreateReplaceableCompositeType',
    'LLVMDIBuilderCreateStaticMemberType',
    'LLVMDIBuilderCreateStructType',
    'LLVMDIBuilderCreateSubroutineType',
    'LLVMDIBuilderCreateTempGlobalVariableFwdDecl',
    'LLVMDIBuilderCreateTempMacroFile', 'LLVMDIBuilderCreateTypedef',
    'LLVMDIBuilderCreateUnionType',
    'LLVMDIBuilderCreateUnspecifiedType',
    'LLVMDIBuilderCreateVectorType', 'LLVMDIBuilderFinalize',
    'LLVMDIBuilderFinalizeSubprogram',
    'LLVMDIBuilderGetOrCreateArray',
    'LLVMDIBuilderGetOrCreateSubrange',
    'LLVMDIBuilderGetOrCreateTypeArray',
    'LLVMDIBuilderInsertDbgValueAtEnd',
    'LLVMDIBuilderInsertDbgValueBefore',
    'LLVMDIBuilderInsertDeclareAtEnd',
    'LLVMDIBuilderInsertDeclareBefore', 'LLVMDIBuilderRef',
    'LLVMDICommonBlockMetadataKind', 'LLVMDICompileUnitMetadataKind',
    'LLVMDICompositeTypeMetadataKind',
    'LLVMDIDerivedTypeMetadataKind', 'LLVMDIEnumeratorMetadataKind',
    'LLVMDIExpressionMetadataKind', 'LLVMDIFileGetDirectory',
    'LLVMDIFileGetFilename', 'LLVMDIFileGetSource',
    'LLVMDIFileMetadataKind', 'LLVMDIFlagAccessibility',
    'LLVMDIFlagAppleBlock', 'LLVMDIFlagArtificial',
    'LLVMDIFlagBigEndian', 'LLVMDIFlagBitField',
    'LLVMDIFlagEnumClass', 'LLVMDIFlagExplicit',
    'LLVMDIFlagFixedEnum', 'LLVMDIFlagFwdDecl',
    'LLVMDIFlagIndirectVirtualBase', 'LLVMDIFlagIntroducedVirtual',
    'LLVMDIFlagLValueReference', 'LLVMDIFlagLittleEndian',
    'LLVMDIFlagMultipleInheritance', 'LLVMDIFlagNoReturn',
    'LLVMDIFlagNonTrivial', 'LLVMDIFlagObjcClassComplete',
    'LLVMDIFlagObjectPointer', 'LLVMDIFlagPrivate',
    'LLVMDIFlagProtected', 'LLVMDIFlagPrototyped',
    'LLVMDIFlagPtrToMemberRep', 'LLVMDIFlagPublic',
    'LLVMDIFlagRValueReference', 'LLVMDIFlagReserved',
    'LLVMDIFlagReservedBit4', 'LLVMDIFlagSingleInheritance',
    'LLVMDIFlagStaticMember', 'LLVMDIFlagThunk',
    'LLVMDIFlagTypePassByReference', 'LLVMDIFlagTypePassByValue',
    'LLVMDIFlagVector', 'LLVMDIFlagVirtual',
    'LLVMDIFlagVirtualInheritance', 'LLVMDIFlagZero', 'LLVMDIFlags',
    'LLVMDIFlags__enumvalues', 'LLVMDIGenericSubrangeMetadataKind',
    'LLVMDIGlobalVariableExpressionGetExpression',
    'LLVMDIGlobalVariableExpressionGetVariable',
    'LLVMDIGlobalVariableExpressionMetadataKind',
    'LLVMDIGlobalVariableMetadataKind',
    'LLVMDIImportedEntityMetadataKind', 'LLVMDILabelMetadataKind',
    'LLVMDILexicalBlockFileMetadataKind',
    'LLVMDILexicalBlockMetadataKind',
    'LLVMDILocalVariableMetadataKind', 'LLVMDILocationGetColumn',
    'LLVMDILocationGetInlinedAt', 'LLVMDILocationGetLine',
    'LLVMDILocationGetScope', 'LLVMDILocationMetadataKind',
    'LLVMDIMacroFileMetadataKind', 'LLVMDIMacroMetadataKind',
    'LLVMDIModuleMetadataKind', 'LLVMDINamespaceMetadataKind',
    'LLVMDIObjCPropertyMetadataKind', 'LLVMDIScopeGetFile',
    'LLVMDIStringTypeMetadataKind', 'LLVMDISubprogramGetLine',
    'LLVMDISubprogramMetadataKind', 'LLVMDISubrangeMetadataKind',
    'LLVMDISubroutineTypeMetadataKind',
    'LLVMDITemplateTypeParameterMetadataKind',
    'LLVMDITemplateValueParameterMetadataKind',
    'LLVMDITypeGetAlignInBits', 'LLVMDITypeGetFlags',
    'LLVMDITypeGetLine', 'LLVMDITypeGetName',
    'LLVMDITypeGetOffsetInBits', 'LLVMDITypeGetSizeInBits',
    'LLVMDIVariableGetFile', 'LLVMDIVariableGetLine',
    'LLVMDIVariableGetScope', 'LLVMDLLExportLinkage',
    'LLVMDLLExportStorageClass', 'LLVMDLLImportLinkage',
    'LLVMDLLImportStorageClass', 'LLVMDLLStorageClass',
    'LLVMDLLStorageClass__enumvalues', 'LLVMDSError', 'LLVMDSNote',
    'LLVMDSRemark', 'LLVMDSWarning', 'LLVMDWARFEmissionFull',
    'LLVMDWARFEmissionKind', 'LLVMDWARFEmissionKind__enumvalues',
    'LLVMDWARFEmissionLineTablesOnly', 'LLVMDWARFEmissionNone',
    'LLVMDWARFMacinfoRecordType', 'LLVMDWARFMacinfoRecordTypeDefine',
    'LLVMDWARFMacinfoRecordTypeEndFile',
    'LLVMDWARFMacinfoRecordTypeMacro',
    'LLVMDWARFMacinfoRecordTypeStartFile',
    'LLVMDWARFMacinfoRecordTypeVendorExt',
    'LLVMDWARFMacinfoRecordType__enumvalues',
    'LLVMDWARFSourceLanguage', 'LLVMDWARFSourceLanguageAda83',
    'LLVMDWARFSourceLanguageAda95', 'LLVMDWARFSourceLanguageBLISS',
    'LLVMDWARFSourceLanguageBORLAND_Delphi',
    'LLVMDWARFSourceLanguageC', 'LLVMDWARFSourceLanguageC11',
    'LLVMDWARFSourceLanguageC89', 'LLVMDWARFSourceLanguageC99',
    'LLVMDWARFSourceLanguageC_plus_plus',
    'LLVMDWARFSourceLanguageC_plus_plus_03',
    'LLVMDWARFSourceLanguageC_plus_plus_11',
    'LLVMDWARFSourceLanguageC_plus_plus_14',
    'LLVMDWARFSourceLanguageCobol74',
    'LLVMDWARFSourceLanguageCobol85', 'LLVMDWARFSourceLanguageD',
    'LLVMDWARFSourceLanguageDylan',
    'LLVMDWARFSourceLanguageFortran03',
    'LLVMDWARFSourceLanguageFortran08',
    'LLVMDWARFSourceLanguageFortran77',
    'LLVMDWARFSourceLanguageFortran90',
    'LLVMDWARFSourceLanguageFortran95',
    'LLVMDWARFSourceLanguageGOOGLE_RenderScript',
    'LLVMDWARFSourceLanguageGo', 'LLVMDWARFSourceLanguageHaskell',
    'LLVMDWARFSourceLanguageJava', 'LLVMDWARFSourceLanguageJulia',
    'LLVMDWARFSourceLanguageMips_Assembler',
    'LLVMDWARFSourceLanguageModula2',
    'LLVMDWARFSourceLanguageModula3', 'LLVMDWARFSourceLanguageOCaml',
    'LLVMDWARFSourceLanguageObjC',
    'LLVMDWARFSourceLanguageObjC_plus_plus',
    'LLVMDWARFSourceLanguageOpenCL', 'LLVMDWARFSourceLanguagePLI',
    'LLVMDWARFSourceLanguagePascal83',
    'LLVMDWARFSourceLanguagePython',
    'LLVMDWARFSourceLanguageRenderScript',
    'LLVMDWARFSourceLanguageRust', 'LLVMDWARFSourceLanguageSwift',
    'LLVMDWARFSourceLanguageUPC',
    'LLVMDWARFSourceLanguage__enumvalues', 'LLVMDWARFTypeEncoding',
    'LLVMDebugMetadataVersion', 'LLVMDefaultStorageClass',
    'LLVMDefaultVisibility', 'LLVMDeleteBasicBlock',
    'LLVMDeleteFunction', 'LLVMDeleteGlobal', 'LLVMDiagnosticHandler',
    'LLVMDiagnosticInfoRef', 'LLVMDiagnosticSeverity',
    'LLVMDiagnosticSeverity__enumvalues', 'LLVMDisasmContextRef',
    'LLVMDisasmDispose', 'LLVMDisasmInstruction',
    'LLVMDisassembler_Option_AsmPrinterVariant',
    'LLVMDisassembler_Option_PrintImmHex',
    'LLVMDisassembler_Option_PrintLatency',
    'LLVMDisassembler_Option_SetInstrComments',
    'LLVMDisassembler_Option_UseMarkup',
    'LLVMDisassembler_ReferenceType_DeMangled_Name',
    'LLVMDisassembler_ReferenceType_InOut_None',
    'LLVMDisassembler_ReferenceType_In_ARM64_ADDXri',
    'LLVMDisassembler_ReferenceType_In_ARM64_ADR',
    'LLVMDisassembler_ReferenceType_In_ARM64_ADRP',
    'LLVMDisassembler_ReferenceType_In_ARM64_LDRXl',
    'LLVMDisassembler_ReferenceType_In_ARM64_LDRXui',
    'LLVMDisassembler_ReferenceType_In_Branch',
    'LLVMDisassembler_ReferenceType_In_PCrel_Load',
    'LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr',
    'LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr',
    'LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref',
    'LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref',
    'LLVMDisassembler_ReferenceType_Out_Objc_Message',
    'LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref',
    'LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref',
    'LLVMDisassembler_ReferenceType_Out_SymbolStub',
    'LLVMDisassembler_VariantKind_ARM64_GOTPAGE',
    'LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF',
    'LLVMDisassembler_VariantKind_ARM64_PAGE',
    'LLVMDisassembler_VariantKind_ARM64_PAGEOFF',
    'LLVMDisassembler_VariantKind_ARM64_TLVOFF',
    'LLVMDisassembler_VariantKind_ARM64_TLVP',
    'LLVMDisassembler_VariantKind_ARM_HI16',
    'LLVMDisassembler_VariantKind_ARM_LO16',
    'LLVMDisassembler_VariantKind_None', 'LLVMDisposeBinary',
    'LLVMDisposeBuilder', 'LLVMDisposeDIBuilder',
    'LLVMDisposeErrorMessage', 'LLVMDisposeExecutionEngine',
    'LLVMDisposeGenericValue', 'LLVMDisposeMCJITMemoryManager',
    'LLVMDisposeMemoryBuffer', 'LLVMDisposeMessage',
    'LLVMDisposeModule', 'LLVMDisposeModuleFlagsMetadata',
    'LLVMDisposeModuleProvider', 'LLVMDisposeObjectFile',
    'LLVMDisposePassBuilderOptions', 'LLVMDisposePassManager',
    'LLVMDisposeRelocationIterator', 'LLVMDisposeSectionIterator',
    'LLVMDisposeSymbolIterator', 'LLVMDisposeTargetData',
    'LLVMDisposeTargetMachine', 'LLVMDisposeTemporaryMDNode',
    'LLVMDisposeValueMetadataEntries',
    'LLVMDistinctMDOperandPlaceholderMetadataKind', 'LLVMDoubleType',
    'LLVMDoubleTypeInContext', 'LLVMDoubleTypeKind', 'LLVMDumpModule',
    'LLVMDumpType', 'LLVMDumpValue', 'LLVMElementAtOffset',
    'LLVMEnablePrettyStackTrace', 'LLVMEraseGlobalIFunc',
    'LLVMErrorRef', 'LLVMErrorSuccess', 'LLVMErrorTypeId',
    'LLVMExactMatchComdatSelectionKind',
    'LLVMExecutionEngineGetErrMsg', 'LLVMExecutionEngineRef',
    'LLVMExternalLinkage', 'LLVMExternalWeakLinkage',
    'LLVMExtractElement', 'LLVMExtractValue', 'LLVMFAdd', 'LLVMFCmp',
    'LLVMFDiv', 'LLVMFMul', 'LLVMFNeg', 'LLVMFP128Type',
    'LLVMFP128TypeInContext', 'LLVMFP128TypeKind', 'LLVMFPExt',
    'LLVMFPToSI', 'LLVMFPToUI', 'LLVMFPTrunc', 'LLVMFRem', 'LLVMFSub',
    'LLVMFastCallConv', 'LLVMFatalErrorHandler', 'LLVMFence',
    'LLVMFinalizeFunctionPassManager', 'LLVMFindFunction',
    'LLVMFloatType', 'LLVMFloatTypeInContext', 'LLVMFloatTypeKind',
    'LLVMFreeMachineCodeForFunction', 'LLVMFreeze',
    'LLVMFunctionType', 'LLVMFunctionTypeKind',
    'LLVMFunctionValueKind', 'LLVMGHCCallConv',
    'LLVMGeneralDynamicTLSModel', 'LLVMGenericDINodeMetadataKind',
    'LLVMGenericValueIntWidth', 'LLVMGenericValueRef',
    'LLVMGenericValueToFloat', 'LLVMGenericValueToInt',
    'LLVMGenericValueToPointer', 'LLVMGetAlignment',
    'LLVMGetAllocatedType', 'LLVMGetArgOperand', 'LLVMGetArrayLength',
    'LLVMGetAsString', 'LLVMGetAtomicRMWBinOp',
    'LLVMGetAttributeCountAtIndex', 'LLVMGetAttributesAtIndex',
    'LLVMGetBasicBlockName', 'LLVMGetBasicBlockParent',
    'LLVMGetBasicBlockTerminator', 'LLVMGetBasicBlocks',
    'LLVMGetBitcodeModule', 'LLVMGetBitcodeModule2',
    'LLVMGetBitcodeModuleInContext', 'LLVMGetBitcodeModuleInContext2',
    'LLVMGetBufferSize', 'LLVMGetBufferStart',
    'LLVMGetCallSiteAttributeCount', 'LLVMGetCallSiteAttributes',
    'LLVMGetCallSiteEnumAttribute', 'LLVMGetCallSiteStringAttribute',
    'LLVMGetCalledFunctionType', 'LLVMGetCalledValue',
    'LLVMGetClause', 'LLVMGetCmpXchgFailureOrdering',
    'LLVMGetCmpXchgSuccessOrdering', 'LLVMGetComdat',
    'LLVMGetComdatSelectionKind', 'LLVMGetCondition',
    'LLVMGetConstOpcode', 'LLVMGetCurrentDebugLocation',
    'LLVMGetCurrentDebugLocation2', 'LLVMGetDLLStorageClass',
    'LLVMGetDataLayout', 'LLVMGetDataLayoutStr',
    'LLVMGetDebugLocColumn', 'LLVMGetDebugLocDirectory',
    'LLVMGetDebugLocFilename', 'LLVMGetDebugLocLine',
    'LLVMGetDefaultTargetTriple', 'LLVMGetDiagInfoDescription',
    'LLVMGetDiagInfoSeverity', 'LLVMGetElementAsConstant',
    'LLVMGetElementPtr', 'LLVMGetElementType',
    'LLVMGetEntryBasicBlock', 'LLVMGetEnumAttributeAtIndex',
    'LLVMGetEnumAttributeKind', 'LLVMGetEnumAttributeKindForName',
    'LLVMGetEnumAttributeValue', 'LLVMGetErrorMessage',
    'LLVMGetErrorTypeId', 'LLVMGetExecutionEngineTargetData',
    'LLVMGetExecutionEngineTargetMachine', 'LLVMGetFCmpPredicate',
    'LLVMGetFirstBasicBlock', 'LLVMGetFirstFunction',
    'LLVMGetFirstGlobal', 'LLVMGetFirstGlobalAlias',
    'LLVMGetFirstGlobalIFunc', 'LLVMGetFirstInstruction',
    'LLVMGetFirstNamedMetadata', 'LLVMGetFirstParam',
    'LLVMGetFirstTarget', 'LLVMGetFirstUse', 'LLVMGetFunctionAddress',
    'LLVMGetFunctionCallConv', 'LLVMGetGC',
    'LLVMGetGEPSourceElementType', 'LLVMGetGlobalContext',
    'LLVMGetGlobalIFuncResolver', 'LLVMGetGlobalParent',
    'LLVMGetGlobalPassRegistry', 'LLVMGetGlobalValueAddress',
    'LLVMGetHandlers', 'LLVMGetHostCPUFeatures', 'LLVMGetHostCPUName',
    'LLVMGetICmpPredicate', 'LLVMGetIncomingBlock',
    'LLVMGetIncomingValue', 'LLVMGetIndices', 'LLVMGetInitializer',
    'LLVMGetInlineAsm', 'LLVMGetInsertBlock',
    'LLVMGetInstructionCallConv', 'LLVMGetInstructionOpcode',
    'LLVMGetInstructionParent', 'LLVMGetIntTypeWidth',
    'LLVMGetIntrinsicDeclaration', 'LLVMGetIntrinsicID',
    'LLVMGetLastBasicBlock', 'LLVMGetLastEnumAttributeKind',
    'LLVMGetLastFunction', 'LLVMGetLastGlobal',
    'LLVMGetLastGlobalAlias', 'LLVMGetLastGlobalIFunc',
    'LLVMGetLastInstruction', 'LLVMGetLastNamedMetadata',
    'LLVMGetLastParam', 'LLVMGetLinkage', 'LLVMGetMDKindID',
    'LLVMGetMDKindIDInContext', 'LLVMGetMDNodeNumOperands',
    'LLVMGetMDNodeOperands', 'LLVMGetMDString', 'LLVMGetMaskValue',
    'LLVMGetMetadata', 'LLVMGetMetadataKind', 'LLVMGetModuleContext',
    'LLVMGetModuleDataLayout', 'LLVMGetModuleDebugMetadataVersion',
    'LLVMGetModuleFlag', 'LLVMGetModuleIdentifier',
    'LLVMGetModuleInlineAsm', 'LLVMGetNamedFunction',
    'LLVMGetNamedGlobal', 'LLVMGetNamedGlobalAlias',
    'LLVMGetNamedGlobalIFunc', 'LLVMGetNamedMetadata',
    'LLVMGetNamedMetadataName', 'LLVMGetNamedMetadataNumOperands',
    'LLVMGetNamedMetadataOperands', 'LLVMGetNextBasicBlock',
    'LLVMGetNextFunction', 'LLVMGetNextGlobal',
    'LLVMGetNextGlobalAlias', 'LLVMGetNextGlobalIFunc',
    'LLVMGetNextInstruction', 'LLVMGetNextNamedMetadata',
    'LLVMGetNextParam', 'LLVMGetNextTarget', 'LLVMGetNextUse',
    'LLVMGetNormalDest', 'LLVMGetNumArgOperands', 'LLVMGetNumClauses',
    'LLVMGetNumContainedTypes', 'LLVMGetNumHandlers',
    'LLVMGetNumIndices', 'LLVMGetNumMaskElements',
    'LLVMGetNumOperands', 'LLVMGetNumSuccessors', 'LLVMGetOperand',
    'LLVMGetOperandUse', 'LLVMGetOrInsertComdat',
    'LLVMGetOrInsertNamedMetadata', 'LLVMGetOrdering', 'LLVMGetParam',
    'LLVMGetParamParent', 'LLVMGetParamTypes', 'LLVMGetParams',
    'LLVMGetParentCatchSwitch', 'LLVMGetPersonalityFn',
    'LLVMGetPointerAddressSpace', 'LLVMGetPointerToGlobal',
    'LLVMGetPoison', 'LLVMGetPreviousBasicBlock',
    'LLVMGetPreviousFunction', 'LLVMGetPreviousGlobal',
    'LLVMGetPreviousGlobalAlias', 'LLVMGetPreviousGlobalIFunc',
    'LLVMGetPreviousInstruction', 'LLVMGetPreviousNamedMetadata',
    'LLVMGetPreviousParam', 'LLVMGetRelocationOffset',
    'LLVMGetRelocationSymbol', 'LLVMGetRelocationType',
    'LLVMGetRelocationTypeName', 'LLVMGetRelocationValueString',
    'LLVMGetRelocations', 'LLVMGetReturnType', 'LLVMGetSection',
    'LLVMGetSectionAddress', 'LLVMGetSectionContainsSymbol',
    'LLVMGetSectionContents', 'LLVMGetSectionName',
    'LLVMGetSectionSize', 'LLVMGetSections', 'LLVMGetSourceFileName',
    'LLVMGetStringAttributeAtIndex', 'LLVMGetStringAttributeKind',
    'LLVMGetStringAttributeValue', 'LLVMGetStringErrorTypeId',
    'LLVMGetStructElementTypes', 'LLVMGetStructName',
    'LLVMGetSubprogram', 'LLVMGetSubtypes', 'LLVMGetSuccessor',
    'LLVMGetSwitchDefaultDest', 'LLVMGetSymbolAddress',
    'LLVMGetSymbolName', 'LLVMGetSymbolSize', 'LLVMGetSymbols',
    'LLVMGetTarget', 'LLVMGetTargetDescription',
    'LLVMGetTargetFromName', 'LLVMGetTargetFromTriple',
    'LLVMGetTargetMachineCPU', 'LLVMGetTargetMachineFeatureString',
    'LLVMGetTargetMachineTarget', 'LLVMGetTargetMachineTriple',
    'LLVMGetTargetName', 'LLVMGetThreadLocalMode',
    'LLVMGetTypeAttributeValue', 'LLVMGetTypeByName',
    'LLVMGetTypeByName2', 'LLVMGetTypeContext', 'LLVMGetTypeKind',
    'LLVMGetUndef', 'LLVMGetUndefMaskElem', 'LLVMGetUnnamedAddress',
    'LLVMGetUnwindDest', 'LLVMGetUsedValue', 'LLVMGetUser',
    'LLVMGetValueKind', 'LLVMGetValueName', 'LLVMGetValueName2',
    'LLVMGetVectorSize', 'LLVMGetVisibility', 'LLVMGetVolatile',
    'LLVMGetWeak', 'LLVMGhostLinkage', 'LLVMGlobalAliasValueKind',
    'LLVMGlobalClearMetadata', 'LLVMGlobalCopyAllMetadata',
    'LLVMGlobalEraseMetadata', 'LLVMGlobalGetValueType',
    'LLVMGlobalIFuncValueKind', 'LLVMGlobalSetMetadata',
    'LLVMGlobalUnnamedAddr', 'LLVMGlobalVariableValueKind',
    'LLVMHHVMCCallConv', 'LLVMHHVMCallConv', 'LLVMHalfType',
    'LLVMHalfTypeInContext', 'LLVMHalfTypeKind', 'LLVMHasMetadata',
    'LLVMHasPersonalityFn', 'LLVMHasUnnamedAddr', 'LLVMHiPECallConv',
    'LLVMHiddenVisibility', 'LLVMICmp', 'LLVMIndirectBr',
    'LLVMInitialExecTLSModel', 'LLVMInitializeAArch64AsmParser',
    'LLVMInitializeAArch64AsmPrinter',
    'LLVMInitializeAArch64Disassembler',
    'LLVMInitializeAArch64Target', 'LLVMInitializeAArch64TargetInfo',
    'LLVMInitializeAArch64TargetMC', 'LLVMInitializeAMDGPUAsmParser',
    'LLVMInitializeAMDGPUAsmPrinter',
    'LLVMInitializeAMDGPUDisassembler', 'LLVMInitializeAMDGPUTarget',
    'LLVMInitializeAMDGPUTargetInfo', 'LLVMInitializeAMDGPUTargetMC',
    'LLVMInitializeARMAsmParser', 'LLVMInitializeARMAsmPrinter',
    'LLVMInitializeARMDisassembler', 'LLVMInitializeARMTarget',
    'LLVMInitializeARMTargetInfo', 'LLVMInitializeARMTargetMC',
    'LLVMInitializeAVRAsmParser', 'LLVMInitializeAVRAsmPrinter',
    'LLVMInitializeAVRDisassembler', 'LLVMInitializeAVRTarget',
    'LLVMInitializeAVRTargetInfo', 'LLVMInitializeAVRTargetMC',
    'LLVMInitializeAggressiveInstCombiner',
    'LLVMInitializeAllAsmParsers', 'LLVMInitializeAllAsmPrinters',
    'LLVMInitializeAllDisassemblers', 'LLVMInitializeAllTargetInfos',
    'LLVMInitializeAllTargetMCs', 'LLVMInitializeAllTargets',
    'LLVMInitializeAnalysis', 'LLVMInitializeBPFAsmParser',
    'LLVMInitializeBPFAsmPrinter', 'LLVMInitializeBPFDisassembler',
    'LLVMInitializeBPFTarget', 'LLVMInitializeBPFTargetInfo',
    'LLVMInitializeBPFTargetMC', 'LLVMInitializeCodeGen',
    'LLVMInitializeCore', 'LLVMInitializeFunctionPassManager',
    'LLVMInitializeHexagonAsmParser',
    'LLVMInitializeHexagonAsmPrinter',
    'LLVMInitializeHexagonDisassembler',
    'LLVMInitializeHexagonTarget', 'LLVMInitializeHexagonTargetInfo',
    'LLVMInitializeHexagonTargetMC', 'LLVMInitializeIPA',
    'LLVMInitializeIPO', 'LLVMInitializeInstCombine',
    'LLVMInitializeInstrumentation', 'LLVMInitializeLanaiAsmParser',
    'LLVMInitializeLanaiAsmPrinter',
    'LLVMInitializeLanaiDisassembler', 'LLVMInitializeLanaiTarget',
    'LLVMInitializeLanaiTargetInfo', 'LLVMInitializeLanaiTargetMC',
    'LLVMInitializeM68kAsmParser', 'LLVMInitializeM68kAsmPrinter',
    'LLVMInitializeM68kDisassembler', 'LLVMInitializeM68kTarget',
    'LLVMInitializeM68kTargetInfo', 'LLVMInitializeM68kTargetMC',
    'LLVMInitializeMCJITCompilerOptions',
    'LLVMInitializeMSP430AsmParser', 'LLVMInitializeMSP430AsmPrinter',
    'LLVMInitializeMSP430Disassembler', 'LLVMInitializeMSP430Target',
    'LLVMInitializeMSP430TargetInfo', 'LLVMInitializeMSP430TargetMC',
    'LLVMInitializeMipsAsmParser', 'LLVMInitializeMipsAsmPrinter',
    'LLVMInitializeMipsDisassembler', 'LLVMInitializeMipsTarget',
    'LLVMInitializeMipsTargetInfo', 'LLVMInitializeMipsTargetMC',
    'LLVMInitializeNVPTXAsmPrinter', 'LLVMInitializeNVPTXTarget',
    'LLVMInitializeNVPTXTargetInfo', 'LLVMInitializeNVPTXTargetMC',
    'LLVMInitializeNativeAsmParser', 'LLVMInitializeNativeAsmPrinter',
    'LLVMInitializeNativeDisassembler', 'LLVMInitializeNativeTarget',
    'LLVMInitializeObjCARCOpts', 'LLVMInitializePowerPCAsmParser',
    'LLVMInitializePowerPCAsmPrinter',
    'LLVMInitializePowerPCDisassembler',
    'LLVMInitializePowerPCTarget', 'LLVMInitializePowerPCTargetInfo',
    'LLVMInitializePowerPCTargetMC', 'LLVMInitializeRISCVAsmParser',
    'LLVMInitializeRISCVAsmPrinter',
    'LLVMInitializeRISCVDisassembler', 'LLVMInitializeRISCVTarget',
    'LLVMInitializeRISCVTargetInfo', 'LLVMInitializeRISCVTargetMC',
    'LLVMInitializeScalarOpts', 'LLVMInitializeSparcAsmParser',
    'LLVMInitializeSparcAsmPrinter',
    'LLVMInitializeSparcDisassembler', 'LLVMInitializeSparcTarget',
    'LLVMInitializeSparcTargetInfo', 'LLVMInitializeSparcTargetMC',
    'LLVMInitializeSystemZAsmParser',
    'LLVMInitializeSystemZAsmPrinter',
    'LLVMInitializeSystemZDisassembler',
    'LLVMInitializeSystemZTarget', 'LLVMInitializeSystemZTargetInfo',
    'LLVMInitializeSystemZTargetMC', 'LLVMInitializeTarget',
    'LLVMInitializeTransformUtils', 'LLVMInitializeVEAsmParser',
    'LLVMInitializeVEAsmPrinter', 'LLVMInitializeVEDisassembler',
    'LLVMInitializeVETarget', 'LLVMInitializeVETargetInfo',
    'LLVMInitializeVETargetMC', 'LLVMInitializeVectorization',
    'LLVMInitializeWebAssemblyAsmParser',
    'LLVMInitializeWebAssemblyAsmPrinter',
    'LLVMInitializeWebAssemblyDisassembler',
    'LLVMInitializeWebAssemblyTarget',
    'LLVMInitializeWebAssemblyTargetInfo',
    'LLVMInitializeWebAssemblyTargetMC', 'LLVMInitializeX86AsmParser',
    'LLVMInitializeX86AsmPrinter', 'LLVMInitializeX86Disassembler',
    'LLVMInitializeX86Target', 'LLVMInitializeX86TargetInfo',
    'LLVMInitializeX86TargetMC', 'LLVMInitializeXCoreAsmPrinter',
    'LLVMInitializeXCoreDisassembler', 'LLVMInitializeXCoreTarget',
    'LLVMInitializeXCoreTargetInfo', 'LLVMInitializeXCoreTargetMC',
    'LLVMInlineAsmDialect', 'LLVMInlineAsmDialectATT',
    'LLVMInlineAsmDialectIntel', 'LLVMInlineAsmDialect__enumvalues',
    'LLVMInlineAsmValueKind', 'LLVMInsertBasicBlock',
    'LLVMInsertBasicBlockInContext', 'LLVMInsertElement',
    'LLVMInsertExistingBasicBlockAfterInsertBlock',
    'LLVMInsertIntoBuilder', 'LLVMInsertIntoBuilderWithName',
    'LLVMInsertValue', 'LLVMInstallFatalErrorHandler',
    'LLVMInstructionClone', 'LLVMInstructionEraseFromParent',
    'LLVMInstructionGetAllMetadataOtherThanDebugLoc',
    'LLVMInstructionGetDebugLoc', 'LLVMInstructionRemoveFromParent',
    'LLVMInstructionSetDebugLoc', 'LLVMInstructionValueKind',
    'LLVMInt128Type', 'LLVMInt128TypeInContext', 'LLVMInt16Type',
    'LLVMInt16TypeInContext', 'LLVMInt1Type', 'LLVMInt1TypeInContext',
    'LLVMInt32Type', 'LLVMInt32TypeInContext', 'LLVMInt64Type',
    'LLVMInt64TypeInContext', 'LLVMInt8Type', 'LLVMInt8TypeInContext',
    'LLVMIntEQ', 'LLVMIntNE', 'LLVMIntPredicate',
    'LLVMIntPredicate__enumvalues', 'LLVMIntPtrType',
    'LLVMIntPtrTypeForAS', 'LLVMIntPtrTypeForASInContext',
    'LLVMIntPtrTypeInContext', 'LLVMIntSGE', 'LLVMIntSGT',
    'LLVMIntSLE', 'LLVMIntSLT', 'LLVMIntToPtr', 'LLVMIntType',
    'LLVMIntTypeInContext', 'LLVMIntUGE', 'LLVMIntUGT', 'LLVMIntULE',
    'LLVMIntULT', 'LLVMIntegerTypeKind', 'LLVMIntelOCLBICallConv',
    'LLVMInternalLinkage', 'LLVMIntrinsicCopyOverloadedName',
    'LLVMIntrinsicCopyOverloadedName2', 'LLVMIntrinsicGetName',
    'LLVMIntrinsicGetType', 'LLVMIntrinsicIsOverloaded', 'LLVMInvoke',
    'LLVMIsAAddrSpaceCastInst', 'LLVMIsAAllocaInst',
    'LLVMIsAArgument', 'LLVMIsAAtomicCmpXchgInst',
    'LLVMIsAAtomicRMWInst', 'LLVMIsABasicBlock',
    'LLVMIsABinaryOperator', 'LLVMIsABitCastInst',
    'LLVMIsABlockAddress', 'LLVMIsABranchInst', 'LLVMIsACallBrInst',
    'LLVMIsACallInst', 'LLVMIsACastInst', 'LLVMIsACatchPadInst',
    'LLVMIsACatchReturnInst', 'LLVMIsACatchSwitchInst',
    'LLVMIsACleanupPadInst', 'LLVMIsACleanupReturnInst',
    'LLVMIsACmpInst', 'LLVMIsAConstant',
    'LLVMIsAConstantAggregateZero', 'LLVMIsAConstantArray',
    'LLVMIsAConstantDataArray', 'LLVMIsAConstantDataSequential',
    'LLVMIsAConstantDataVector', 'LLVMIsAConstantExpr',
    'LLVMIsAConstantFP', 'LLVMIsAConstantInt',
    'LLVMIsAConstantPointerNull', 'LLVMIsAConstantStruct',
    'LLVMIsAConstantTokenNone', 'LLVMIsAConstantVector',
    'LLVMIsADbgDeclareInst', 'LLVMIsADbgInfoIntrinsic',
    'LLVMIsADbgLabelInst', 'LLVMIsADbgVariableIntrinsic',
    'LLVMIsAExtractElementInst', 'LLVMIsAExtractValueInst',
    'LLVMIsAFCmpInst', 'LLVMIsAFPExtInst', 'LLVMIsAFPToSIInst',
    'LLVMIsAFPToUIInst', 'LLVMIsAFPTruncInst', 'LLVMIsAFenceInst',
    'LLVMIsAFreezeInst', 'LLVMIsAFuncletPadInst', 'LLVMIsAFunction',
    'LLVMIsAGetElementPtrInst', 'LLVMIsAGlobalAlias',
    'LLVMIsAGlobalIFunc', 'LLVMIsAGlobalObject', 'LLVMIsAGlobalValue',
    'LLVMIsAGlobalVariable', 'LLVMIsAICmpInst',
    'LLVMIsAIndirectBrInst', 'LLVMIsAInlineAsm',
    'LLVMIsAInsertElementInst', 'LLVMIsAInsertValueInst',
    'LLVMIsAInstruction', 'LLVMIsAIntToPtrInst',
    'LLVMIsAIntrinsicInst', 'LLVMIsAInvokeInst',
    'LLVMIsALandingPadInst', 'LLVMIsALoadInst', 'LLVMIsAMDNode',
    'LLVMIsAMDString', 'LLVMIsAMemCpyInst', 'LLVMIsAMemIntrinsic',
    'LLVMIsAMemMoveInst', 'LLVMIsAMemSetInst', 'LLVMIsAPHINode',
    'LLVMIsAPoisonValue', 'LLVMIsAPtrToIntInst', 'LLVMIsAResumeInst',
    'LLVMIsAReturnInst', 'LLVMIsASExtInst', 'LLVMIsASIToFPInst',
    'LLVMIsASelectInst', 'LLVMIsAShuffleVectorInst',
    'LLVMIsAStoreInst', 'LLVMIsASwitchInst', 'LLVMIsATerminatorInst',
    'LLVMIsATruncInst', 'LLVMIsAUIToFPInst',
    'LLVMIsAUnaryInstruction', 'LLVMIsAUnaryOperator',
    'LLVMIsAUndefValue', 'LLVMIsAUnreachableInst', 'LLVMIsAUser',
    'LLVMIsAVAArgInst', 'LLVMIsAZExtInst', 'LLVMIsAtomicSingleThread',
    'LLVMIsCleanup', 'LLVMIsConditional', 'LLVMIsConstant',
    'LLVMIsConstantString', 'LLVMIsDeclaration',
    'LLVMIsEnumAttribute', 'LLVMIsExternallyInitialized',
    'LLVMIsFunctionVarArg', 'LLVMIsGlobalConstant', 'LLVMIsInBounds',
    'LLVMIsLiteralStruct', 'LLVMIsMultithreaded', 'LLVMIsNull',
    'LLVMIsOpaqueStruct', 'LLVMIsPackedStruct', 'LLVMIsPoison',
    'LLVMIsRelocationIteratorAtEnd', 'LLVMIsSectionIteratorAtEnd',
    'LLVMIsStringAttribute', 'LLVMIsSymbolIteratorAtEnd',
    'LLVMIsTailCall', 'LLVMIsThreadLocal', 'LLVMIsTypeAttribute',
    'LLVMIsUndef', 'LLVMJITCSymbolMapPair', 'LLVMJITEvaluatedSymbol',
    'LLVMJITEventListenerRef', 'LLVMJITSymbolFlags',
    'LLVMJITSymbolGenericFlags', 'LLVMJITSymbolGenericFlagsCallable',
    'LLVMJITSymbolGenericFlagsExported',
    'LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly',
    'LLVMJITSymbolGenericFlagsWeak',
    'LLVMJITSymbolGenericFlags__enumvalues',
    'LLVMJITSymbolTargetFlags', 'LLVMLShr', 'LLVMLabelType',
    'LLVMLabelTypeInContext', 'LLVMLabelTypeKind', 'LLVMLandingPad',
    'LLVMLandingPadCatch', 'LLVMLandingPadClauseTy',
    'LLVMLandingPadClauseTy__enumvalues', 'LLVMLandingPadFilter',
    'LLVMLargestComdatSelectionKind', 'LLVMLinkInInterpreter',
    'LLVMLinkInMCJIT', 'LLVMLinkModules2', 'LLVMLinkOnceAnyLinkage',
    'LLVMLinkOnceODRAutoHideLinkage', 'LLVMLinkOnceODRLinkage',
    'LLVMLinkage', 'LLVMLinkage__enumvalues',
    'LLVMLinkerDestroySource', 'LLVMLinkerMode',
    'LLVMLinkerMode__enumvalues', 'LLVMLinkerPreserveSource_Removed',
    'LLVMLinkerPrivateLinkage', 'LLVMLinkerPrivateWeakLinkage',
    'LLVMLittleEndian', 'LLVMLoad', 'LLVMLoadLibraryPermanently',
    'LLVMLocalAsMetadataMetadataKind', 'LLVMLocalDynamicTLSModel',
    'LLVMLocalExecTLSModel', 'LLVMLocalUnnamedAddr',
    'LLVMLookupIntrinsicID', 'LLVMMCJITMemoryManagerRef',
    'LLVMMDNode', 'LLVMMDNodeInContext', 'LLVMMDNodeInContext2',
    'LLVMMDString', 'LLVMMDStringInContext', 'LLVMMDStringInContext2',
    'LLVMMDStringMetadataKind', 'LLVMMDTupleMetadataKind',
    'LLVMMSP430BUILTINCallConv', 'LLVMMSP430INTRCallConv',
    'LLVMMachOUniversalBinaryCopyObjectForArch',
    'LLVMMemoryBufferRef', 'LLVMMemoryDefValueKind',
    'LLVMMemoryManagerAllocateCodeSectionCallback',
    'LLVMMemoryManagerAllocateDataSectionCallback',
    'LLVMMemoryManagerDestroyCallback',
    'LLVMMemoryManagerFinalizeMemoryCallback',
    'LLVMMemoryPhiValueKind', 'LLVMMemoryUseValueKind',
    'LLVMMetadataAsValue', 'LLVMMetadataAsValueValueKind',
    'LLVMMetadataKind', 'LLVMMetadataRef',
    'LLVMMetadataReplaceAllUsesWith', 'LLVMMetadataTypeInContext',
    'LLVMMetadataTypeKind', 'LLVMModuleCreateWithName',
    'LLVMModuleCreateWithNameInContext', 'LLVMModuleFlagBehavior',
    'LLVMModuleFlagBehaviorAppend',
    'LLVMModuleFlagBehaviorAppendUnique',
    'LLVMModuleFlagBehaviorError', 'LLVMModuleFlagBehaviorOverride',
    'LLVMModuleFlagBehaviorRequire', 'LLVMModuleFlagBehaviorWarning',
    'LLVMModuleFlagBehavior__enumvalues',
    'LLVMModuleFlagEntriesGetFlagBehavior',
    'LLVMModuleFlagEntriesGetKey', 'LLVMModuleFlagEntriesGetMetadata',
    'LLVMModuleFlagEntry', 'LLVMModuleProviderRef', 'LLVMModuleRef',
    'LLVMMoveBasicBlockAfter', 'LLVMMoveBasicBlockBefore',
    'LLVMMoveToContainingSection', 'LLVMMoveToNextRelocation',
    'LLVMMoveToNextSection', 'LLVMMoveToNextSymbol', 'LLVMMul',
    'LLVMNamedMDNodeRef', 'LLVMNoDeduplicateComdatSelectionKind',
    'LLVMNoUnnamedAddr', 'LLVMNormalizeTargetTriple',
    'LLVMNotThreadLocal', 'LLVMObjectFile',
    'LLVMObjectFileCopySectionIterator',
    'LLVMObjectFileCopySymbolIterator',
    'LLVMObjectFileIsSectionIteratorAtEnd',
    'LLVMObjectFileIsSymbolIteratorAtEnd', 'LLVMObjectFileRef',
    'LLVMOffsetOfElement', 'LLVMOpInfoCallback', 'LLVMOpcode',
    'LLVMOpcode__enumvalues', 'LLVMOr', 'LLVMOrcAbsoluteSymbols',
    'LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction',
    'LLVMOrcCDependenceMapPair', 'LLVMOrcCDependenceMapPairs',
    'LLVMOrcCLookupSet', 'LLVMOrcCLookupSetElement',
    'LLVMOrcCSymbolAliasMapEntry', 'LLVMOrcCSymbolAliasMapPair',
    'LLVMOrcCSymbolAliasMapPairs', 'LLVMOrcCSymbolFlagsMapPair',
    'LLVMOrcCSymbolFlagsMapPairs', 'LLVMOrcCSymbolMapPairs',
    'LLVMOrcCSymbolsList',
    'LLVMOrcCreateCustomCAPIDefinitionGenerator',
    'LLVMOrcCreateCustomMaterializationUnit',
    'LLVMOrcCreateDumpObjects',
    'LLVMOrcCreateDynamicLibrarySearchGeneratorForPath',
    'LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess',
    'LLVMOrcCreateLLJIT', 'LLVMOrcCreateLLJITBuilder',
    'LLVMOrcCreateLocalIndirectStubsManager',
    'LLVMOrcCreateLocalLazyCallThroughManager',
    'LLVMOrcCreateNewThreadSafeContext',
    'LLVMOrcCreateNewThreadSafeModule',
    'LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager',
    'LLVMOrcCreateStaticLibrarySearchGeneratorForPath',
    'LLVMOrcDefinitionGeneratorRef', 'LLVMOrcDisposeCSymbolFlagsMap',
    'LLVMOrcDisposeDefinitionGenerator', 'LLVMOrcDisposeDumpObjects',
    'LLVMOrcDisposeIndirectStubsManager',
    'LLVMOrcDisposeJITTargetMachineBuilder', 'LLVMOrcDisposeLLJIT',
    'LLVMOrcDisposeLLJITBuilder',
    'LLVMOrcDisposeLazyCallThroughManager',
    'LLVMOrcDisposeMaterializationResponsibility',
    'LLVMOrcDisposeMaterializationUnit', 'LLVMOrcDisposeObjectLayer',
    'LLVMOrcDisposeSymbols', 'LLVMOrcDisposeThreadSafeContext',
    'LLVMOrcDisposeThreadSafeModule', 'LLVMOrcDumpObjectsRef',
    'LLVMOrcDumpObjects_CallOperator', 'LLVMOrcErrorReporterFunction',
    'LLVMOrcExecutionSessionCreateBareJITDylib',
    'LLVMOrcExecutionSessionCreateJITDylib',
    'LLVMOrcExecutionSessionGetJITDylibByName',
    'LLVMOrcExecutionSessionGetSymbolStringPool',
    'LLVMOrcExecutionSessionIntern', 'LLVMOrcExecutionSessionRef',
    'LLVMOrcExecutionSessionSetErrorReporter',
    'LLVMOrcExecutorAddress',
    'LLVMOrcGenericIRModuleOperationFunction',
    'LLVMOrcIRTransformLayerEmit', 'LLVMOrcIRTransformLayerRef',
    'LLVMOrcIRTransformLayerSetTransform',
    'LLVMOrcIRTransformLayerTransformFunction',
    'LLVMOrcIndirectStubsManagerRef', 'LLVMOrcJITDylibAddGenerator',
    'LLVMOrcJITDylibClear', 'LLVMOrcJITDylibCreateResourceTracker',
    'LLVMOrcJITDylibDefine',
    'LLVMOrcJITDylibGetDefaultResourceTracker',
    'LLVMOrcJITDylibLookupFlags',
    'LLVMOrcJITDylibLookupFlagsMatchAllSymbols',
    'LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly',
    'LLVMOrcJITDylibLookupFlags__enumvalues', 'LLVMOrcJITDylibRef',
    'LLVMOrcJITTargetAddress',
    'LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine',
    'LLVMOrcJITTargetMachineBuilderDetectHost',
    'LLVMOrcJITTargetMachineBuilderGetTargetTriple',
    'LLVMOrcJITTargetMachineBuilderRef',
    'LLVMOrcJITTargetMachineBuilderSetTargetTriple',
    'LLVMOrcLLJITAddLLVMIRModule',
    'LLVMOrcLLJITAddLLVMIRModuleWithRT', 'LLVMOrcLLJITAddObjectFile',
    'LLVMOrcLLJITAddObjectFileWithRT',
    'LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction',
    'LLVMOrcLLJITBuilderRef',
    'LLVMOrcLLJITBuilderSetJITTargetMachineBuilder',
    'LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator',
    'LLVMOrcLLJITGetDataLayoutStr', 'LLVMOrcLLJITGetExecutionSession',
    'LLVMOrcLLJITGetGlobalPrefix', 'LLVMOrcLLJITGetIRTransformLayer',
    'LLVMOrcLLJITGetMainJITDylib', 'LLVMOrcLLJITGetObjLinkingLayer',
    'LLVMOrcLLJITGetObjTransformLayer', 'LLVMOrcLLJITGetTripleString',
    'LLVMOrcLLJITLookup', 'LLVMOrcLLJITMangleAndIntern',
    'LLVMOrcLLJITRef', 'LLVMOrcLazyCallThroughManagerRef',
    'LLVMOrcLazyReexports', 'LLVMOrcLookupKind',
    'LLVMOrcLookupKindDLSym', 'LLVMOrcLookupKindStatic',
    'LLVMOrcLookupKind__enumvalues', 'LLVMOrcLookupStateRef',
    'LLVMOrcMaterializationResponsibilityAddDependencies',
    'LLVMOrcMaterializationResponsibilityAddDependenciesForAll',
    'LLVMOrcMaterializationResponsibilityDefineMaterializing',
    'LLVMOrcMaterializationResponsibilityDelegate',
    'LLVMOrcMaterializationResponsibilityFailMaterialization',
    'LLVMOrcMaterializationResponsibilityGetExecutionSession',
    'LLVMOrcMaterializationResponsibilityGetInitializerSymbol',
    'LLVMOrcMaterializationResponsibilityGetRequestedSymbols',
    'LLVMOrcMaterializationResponsibilityGetSymbols',
    'LLVMOrcMaterializationResponsibilityGetTargetDylib',
    'LLVMOrcMaterializationResponsibilityNotifyEmitted',
    'LLVMOrcMaterializationResponsibilityNotifyResolved',
    'LLVMOrcMaterializationResponsibilityRef',
    'LLVMOrcMaterializationResponsibilityReplace',
    'LLVMOrcMaterializationUnitDestroyFunction',
    'LLVMOrcMaterializationUnitDiscardFunction',
    'LLVMOrcMaterializationUnitMaterializeFunction',
    'LLVMOrcMaterializationUnitRef',
    'LLVMOrcObjectLayerAddObjectFile',
    'LLVMOrcObjectLayerAddObjectFileWithRT', 'LLVMOrcObjectLayerEmit',
    'LLVMOrcObjectLayerRef', 'LLVMOrcObjectLinkingLayerRef',
    'LLVMOrcObjectTransformLayerRef',
    'LLVMOrcObjectTransformLayerSetTransform',
    'LLVMOrcObjectTransformLayerTransformFunction',
    'LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener',
    'LLVMOrcReleaseResourceTracker',
    'LLVMOrcReleaseSymbolStringPoolEntry',
    'LLVMOrcResourceTrackerRef', 'LLVMOrcResourceTrackerRemove',
    'LLVMOrcResourceTrackerTransferTo',
    'LLVMOrcRetainSymbolStringPoolEntry', 'LLVMOrcSymbolLookupFlags',
    'LLVMOrcSymbolLookupFlagsRequiredSymbol',
    'LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol',
    'LLVMOrcSymbolLookupFlags__enumvalues', 'LLVMOrcSymbolPredicate',
    'LLVMOrcSymbolStringPoolClearDeadEntries',
    'LLVMOrcSymbolStringPoolEntryRef',
    'LLVMOrcSymbolStringPoolEntryStr', 'LLVMOrcSymbolStringPoolRef',
    'LLVMOrcThreadSafeContextGetContext',
    'LLVMOrcThreadSafeContextRef', 'LLVMOrcThreadSafeModuleRef',
    'LLVMOrcThreadSafeModuleWithModuleDo', 'LLVMPHI',
    'LLVMPPCFP128Type', 'LLVMPPCFP128TypeInContext',
    'LLVMPPC_FP128TypeKind', 'LLVMPTXDeviceCallConv',
    'LLVMPTXKernelCallConv', 'LLVMParseBitcode', 'LLVMParseBitcode2',
    'LLVMParseBitcodeInContext', 'LLVMParseBitcodeInContext2',
    'LLVMParseCommandLineOptions', 'LLVMParseIRInContext',
    'LLVMPassBuilderOptionsRef',
    'LLVMPassBuilderOptionsSetCallGraphProfile',
    'LLVMPassBuilderOptionsSetDebugLogging',
    'LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll',
    'LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap',
    'LLVMPassBuilderOptionsSetLicmMssaOptCap',
    'LLVMPassBuilderOptionsSetLoopInterleaving',
    'LLVMPassBuilderOptionsSetLoopUnrolling',
    'LLVMPassBuilderOptionsSetLoopVectorization',
    'LLVMPassBuilderOptionsSetMergeFunctions',
    'LLVMPassBuilderOptionsSetSLPVectorization',
    'LLVMPassBuilderOptionsSetVerifyEach',
    'LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints',
    'LLVMPassManagerBuilderCreate', 'LLVMPassManagerBuilderDispose',
    'LLVMPassManagerBuilderPopulateFunctionPassManager',
    'LLVMPassManagerBuilderPopulateLTOPassManager',
    'LLVMPassManagerBuilderPopulateModulePassManager',
    'LLVMPassManagerBuilderRef',
    'LLVMPassManagerBuilderSetDisableSimplifyLibCalls',
    'LLVMPassManagerBuilderSetDisableUnitAtATime',
    'LLVMPassManagerBuilderSetDisableUnrollLoops',
    'LLVMPassManagerBuilderSetOptLevel',
    'LLVMPassManagerBuilderSetSizeLevel',
    'LLVMPassManagerBuilderUseInlinerWithThreshold',
    'LLVMPassManagerRef', 'LLVMPassRegistryRef', 'LLVMPointerSize',
    'LLVMPointerSizeForAS', 'LLVMPointerType', 'LLVMPointerTypeKind',
    'LLVMPoisonValueValueKind', 'LLVMPositionBuilder',
    'LLVMPositionBuilderAtEnd', 'LLVMPositionBuilderBefore',
    'LLVMPreferredAlignmentOfGlobal', 'LLVMPreferredAlignmentOfType',
    'LLVMPreserveAllCallConv', 'LLVMPreserveMostCallConv',
    'LLVMPrintMessageAction', 'LLVMPrintModuleToFile',
    'LLVMPrintModuleToString', 'LLVMPrintTypeToString',
    'LLVMPrintValueToString', 'LLVMPrivateLinkage',
    'LLVMProtectedVisibility', 'LLVMPtrToInt', 'LLVMRealOEQ',
    'LLVMRealOGE', 'LLVMRealOGT', 'LLVMRealOLE', 'LLVMRealOLT',
    'LLVMRealONE', 'LLVMRealORD', 'LLVMRealPredicate',
    'LLVMRealPredicateFalse', 'LLVMRealPredicateTrue',
    'LLVMRealPredicate__enumvalues', 'LLVMRealUEQ', 'LLVMRealUGE',
    'LLVMRealUGT', 'LLVMRealULE', 'LLVMRealULT', 'LLVMRealUNE',
    'LLVMRealUNO', 'LLVMRecompileAndRelinkFunction',
    'LLVMRelocDefault', 'LLVMRelocDynamicNoPic', 'LLVMRelocMode',
    'LLVMRelocMode__enumvalues', 'LLVMRelocPIC', 'LLVMRelocROPI',
    'LLVMRelocROPI_RWPI', 'LLVMRelocRWPI', 'LLVMRelocStatic',
    'LLVMRelocationIteratorRef', 'LLVMRemarkArgGetDebugLoc',
    'LLVMRemarkArgGetKey', 'LLVMRemarkArgGetValue',
    'LLVMRemarkArgRef', 'LLVMRemarkDebugLocGetSourceColumn',
    'LLVMRemarkDebugLocGetSourceFilePath',
    'LLVMRemarkDebugLocGetSourceLine', 'LLVMRemarkDebugLocRef',
    'LLVMRemarkEntryDispose', 'LLVMRemarkEntryGetDebugLoc',
    'LLVMRemarkEntryGetFirstArg', 'LLVMRemarkEntryGetFunctionName',
    'LLVMRemarkEntryGetHotness', 'LLVMRemarkEntryGetNextArg',
    'LLVMRemarkEntryGetNumArgs', 'LLVMRemarkEntryGetPassName',
    'LLVMRemarkEntryGetRemarkName', 'LLVMRemarkEntryGetType',
    'LLVMRemarkEntryRef', 'LLVMRemarkParserCreateBitstream',
    'LLVMRemarkParserCreateYAML', 'LLVMRemarkParserDispose',
    'LLVMRemarkParserGetErrorMessage', 'LLVMRemarkParserGetNext',
    'LLVMRemarkParserHasError', 'LLVMRemarkParserRef',
    'LLVMRemarkStringGetData', 'LLVMRemarkStringGetLen',
    'LLVMRemarkStringRef', 'LLVMRemarkType', 'LLVMRemarkTypeAnalysis',
    'LLVMRemarkTypeAnalysisAliasing',
    'LLVMRemarkTypeAnalysisFPCommute', 'LLVMRemarkTypeFailure',
    'LLVMRemarkTypeMissed', 'LLVMRemarkTypePassed',
    'LLVMRemarkTypeUnknown', 'LLVMRemarkVersion',
    'LLVMRemoveBasicBlockFromParent',
    'LLVMRemoveCallSiteEnumAttribute',
    'LLVMRemoveCallSiteStringAttribute',
    'LLVMRemoveEnumAttributeAtIndex', 'LLVMRemoveGlobalIFunc',
    'LLVMRemoveModule', 'LLVMRemoveStringAttributeAtIndex',
    'LLVMReplaceAllUsesWith', 'LLVMResetFatalErrorHandler',
    'LLVMResume', 'LLVMRet', 'LLVMReturnStatusAction',
    'LLVMRunFunction', 'LLVMRunFunctionAsMain',
    'LLVMRunFunctionPassManager', 'LLVMRunPassManager',
    'LLVMRunPasses', 'LLVMRunStaticConstructors',
    'LLVMRunStaticDestructors', 'LLVMSDiv', 'LLVMSExt', 'LLVMSIToFP',
    'LLVMSPIRFUNCCallConv', 'LLVMSPIRKERNELCallConv', 'LLVMSRem',
    'LLVMSameSizeComdatSelectionKind', 'LLVMScalableVectorType',
    'LLVMScalableVectorTypeKind', 'LLVMSearchForAddressOfSymbol',
    'LLVMSectionIteratorRef', 'LLVMSelect', 'LLVMSetAlignment',
    'LLVMSetArgOperand', 'LLVMSetAtomicRMWBinOp',
    'LLVMSetAtomicSingleThread', 'LLVMSetCleanup',
    'LLVMSetCmpXchgFailureOrdering', 'LLVMSetCmpXchgSuccessOrdering',
    'LLVMSetComdat', 'LLVMSetComdatSelectionKind', 'LLVMSetCondition',
    'LLVMSetCurrentDebugLocation', 'LLVMSetCurrentDebugLocation2',
    'LLVMSetDLLStorageClass', 'LLVMSetDataLayout',
    'LLVMSetDisasmOptions', 'LLVMSetExternallyInitialized',
    'LLVMSetFunctionCallConv', 'LLVMSetGC', 'LLVMSetGlobalConstant',
    'LLVMSetGlobalIFuncResolver', 'LLVMSetInitializer',
    'LLVMSetInstDebugLocation', 'LLVMSetInstrParamAlignment',
    'LLVMSetInstructionCallConv', 'LLVMSetIsInBounds',
    'LLVMSetLinkage', 'LLVMSetMetadata', 'LLVMSetModuleDataLayout',
    'LLVMSetModuleIdentifier', 'LLVMSetModuleInlineAsm',
    'LLVMSetModuleInlineAsm2', 'LLVMSetNormalDest', 'LLVMSetOperand',
    'LLVMSetOrdering', 'LLVMSetParamAlignment',
    'LLVMSetParentCatchSwitch', 'LLVMSetPersonalityFn',
    'LLVMSetSection', 'LLVMSetSourceFileName', 'LLVMSetSubprogram',
    'LLVMSetSuccessor', 'LLVMSetTailCall', 'LLVMSetTarget',
    'LLVMSetTargetMachineAsmVerbosity', 'LLVMSetThreadLocal',
    'LLVMSetThreadLocalMode', 'LLVMSetUnnamedAddr',
    'LLVMSetUnnamedAddress', 'LLVMSetUnwindDest', 'LLVMSetValueName',
    'LLVMSetValueName2', 'LLVMSetVisibility', 'LLVMSetVolatile',
    'LLVMSetWeak', 'LLVMShl', 'LLVMShuffleVector', 'LLVMShutdown',
    'LLVMSizeOf', 'LLVMSizeOfTypeInBits', 'LLVMStartMultithreaded',
    'LLVMStopMultithreaded', 'LLVMStore', 'LLVMStoreSizeOfType',
    'LLVMStripModuleDebugInfo', 'LLVMStructCreateNamed',
    'LLVMStructGetTypeAtIndex', 'LLVMStructSetBody', 'LLVMStructType',
    'LLVMStructTypeInContext', 'LLVMStructTypeKind', 'LLVMSub',
    'LLVMSwiftCallConv', 'LLVMSwitch', 'LLVMSymbolIteratorRef',
    'LLVMSymbolLookupCallback', 'LLVMTargetDataRef',
    'LLVMTargetHasAsmBackend', 'LLVMTargetHasJIT',
    'LLVMTargetHasTargetMachine', 'LLVMTargetLibraryInfoRef',
    'LLVMTargetMachineEmitToFile',
    'LLVMTargetMachineEmitToMemoryBuffer', 'LLVMTargetMachineRef',
    'LLVMTargetRef', 'LLVMTemporaryMDNode', 'LLVMThreadLocalMode',
    'LLVMThreadLocalMode__enumvalues', 'LLVMTokenTypeInContext',
    'LLVMTokenTypeKind', 'LLVMTrunc', 'LLVMTypeIsSized',
    'LLVMTypeKind', 'LLVMTypeKind__enumvalues', 'LLVMTypeOf',
    'LLVMTypeRef', 'LLVMUDiv', 'LLVMUIToFP', 'LLVMURem',
    'LLVMUndefValueValueKind', 'LLVMUnnamedAddr',
    'LLVMUnnamedAddr__enumvalues', 'LLVMUnreachable', 'LLVMUseRef',
    'LLVMUserOp1', 'LLVMUserOp2', 'LLVMVAArg',
    'LLVMValueAsBasicBlock', 'LLVMValueAsMetadata',
    'LLVMValueIsBasicBlock', 'LLVMValueKind',
    'LLVMValueKind__enumvalues', 'LLVMValueMetadataEntriesGetKind',
    'LLVMValueMetadataEntriesGetMetadata', 'LLVMValueMetadataEntry',
    'LLVMValueRef', 'LLVMVectorType', 'LLVMVectorTypeKind',
    'LLVMVerifierFailureAction',
    'LLVMVerifierFailureAction__enumvalues', 'LLVMVerifyFunction',
    'LLVMVerifyModule', 'LLVMViewFunctionCFG',
    'LLVMViewFunctionCFGOnly', 'LLVMVisibility',
    'LLVMVisibility__enumvalues', 'LLVMVoidType',
    'LLVMVoidTypeInContext', 'LLVMVoidTypeKind', 'LLVMWeakAnyLinkage',
    'LLVMWeakODRLinkage', 'LLVMWebKitJSCallConv', 'LLVMWin64CallConv',
    'LLVMWriteBitcodeToFD', 'LLVMWriteBitcodeToFile',
    'LLVMWriteBitcodeToFileHandle', 'LLVMWriteBitcodeToMemoryBuffer',
    'LLVMX8664SysVCallConv', 'LLVMX86AMXType',
    'LLVMX86AMXTypeInContext', 'LLVMX86FP80Type',
    'LLVMX86FP80TypeInContext', 'LLVMX86FastcallCallConv',
    'LLVMX86INTRCallConv', 'LLVMX86MMXType',
    'LLVMX86MMXTypeInContext', 'LLVMX86RegCallCallConv',
    'LLVMX86StdcallCallConv', 'LLVMX86ThisCallCallConv',
    'LLVMX86VectorCallCallConv', 'LLVMX86_AMXTypeKind',
    'LLVMX86_FP80TypeKind', 'LLVMX86_MMXTypeKind', 'LLVMXor',
    'LLVMYieldCallback', 'LLVMZExt', 'LLVM_C_ANALYSIS_H',
    'LLVM_C_BITREADER_H', 'LLVM_C_BITWRITER_H', 'LLVM_C_COMDAT_H',
    'LLVM_C_CORE_H', 'LLVM_C_DATATYPES_H', 'LLVM_C_DEBUGINFO_H',
    'LLVM_C_DEPRECATED_H', 'LLVM_C_DISASSEMBLERTYPES_H',
    'LLVM_C_DISASSEMBLER_H', 'LLVM_C_ERRORHANDLING_H',
    'LLVM_C_ERROR_H', 'LLVM_C_EXECUTIONENGINE_H', 'LLVM_C_EXTERNC_H',
    'LLVM_C_INITIALIZATION_H', 'LLVM_C_IRREADER_H', 'LLVM_C_LINKER_H',
    'LLVM_C_LLJIT_H', 'LLVM_C_LTO_H', 'LLVM_C_OBJECT_H',
    'LLVM_C_ORCEE_H', 'LLVM_C_ORC_H', 'LLVM_C_REMARKS_H',
    'LLVM_C_SUPPORT_H', 'LLVM_C_TARGETMACHINE_H', 'LLVM_C_TARGET_H',
    'LLVM_C_TRANSFORMS_AGGRESSIVEINSTCOMBINE_H',
    'LLVM_C_TRANSFORMS_COROUTINES_H',
    'LLVM_C_TRANSFORMS_INSTCOMBINE_H', 'LLVM_C_TRANSFORMS_IPO_H',
    'LLVM_C_TRANSFORMS_PASSBUILDER_H',
    'LLVM_C_TRANSFORMS_PASSMANAGERBUILDER_H',
    'LLVM_C_TRANSFORMS_SCALAR_H', 'LLVM_C_TRANSFORMS_UTILS_H',
    'LLVM_C_TRANSFORMS_VECTORIZE_H', 'LLVM_C_TYPES_H',
    'LTOObjectBuffer', 'LTO_API_VERSION',
    'LTO_CODEGEN_PIC_MODEL_DEFAULT', 'LTO_CODEGEN_PIC_MODEL_DYNAMIC',
    'LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC',
    'LTO_CODEGEN_PIC_MODEL_STATIC', 'LTO_DEBUG_MODEL_DWARF',
    'LTO_DEBUG_MODEL_NONE', 'LTO_DS_ERROR', 'LTO_DS_NOTE',
    'LTO_DS_REMARK', 'LTO_DS_WARNING', 'LTO_SYMBOL_ALIAS',
    'LTO_SYMBOL_ALIGNMENT_MASK', 'LTO_SYMBOL_COMDAT',
    'LTO_SYMBOL_DEFINITION_MASK', 'LTO_SYMBOL_DEFINITION_REGULAR',
    'LTO_SYMBOL_DEFINITION_TENTATIVE',
    'LTO_SYMBOL_DEFINITION_UNDEFINED', 'LTO_SYMBOL_DEFINITION_WEAK',
    'LTO_SYMBOL_DEFINITION_WEAKUNDEF', 'LTO_SYMBOL_PERMISSIONS_CODE',
    'LTO_SYMBOL_PERMISSIONS_DATA', 'LTO_SYMBOL_PERMISSIONS_MASK',
    'LTO_SYMBOL_PERMISSIONS_RODATA', 'LTO_SYMBOL_SCOPE_DEFAULT',
    'LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN',
    'LTO_SYMBOL_SCOPE_HIDDEN', 'LTO_SYMBOL_SCOPE_INTERNAL',
    'LTO_SYMBOL_SCOPE_MASK', 'LTO_SYMBOL_SCOPE_PROTECTED',
    'REMARKS_API_VERSION', 'c__EA_LLVMAtomicOrdering',
    'c__EA_LLVMAtomicRMWBinOp', 'c__EA_LLVMBinaryType',
    'c__EA_LLVMCallConv', 'c__EA_LLVMCodeGenFileType',
    'c__EA_LLVMCodeGenOptLevel', 'c__EA_LLVMCodeModel',
    'c__EA_LLVMComdatSelectionKind', 'c__EA_LLVMDIFlags',
    'c__EA_LLVMDLLStorageClass', 'c__EA_LLVMDWARFEmissionKind',
    'c__EA_LLVMDWARFMacinfoRecordType',
    'c__EA_LLVMDWARFSourceLanguage', 'c__EA_LLVMDiagnosticSeverity',
    'c__EA_LLVMInlineAsmDialect', 'c__EA_LLVMIntPredicate',
    'c__EA_LLVMJITSymbolGenericFlags', 'c__EA_LLVMLandingPadClauseTy',
    'c__EA_LLVMLinkage', 'c__EA_LLVMLinkerMode',
    'c__EA_LLVMModuleFlagBehavior', 'c__EA_LLVMOpcode',
    'c__EA_LLVMOrcJITDylibLookupFlags', 'c__EA_LLVMOrcLookupKind',
    'c__EA_LLVMOrcSymbolLookupFlags', 'c__EA_LLVMRealPredicate',
    'c__EA_LLVMRelocMode', 'c__EA_LLVMThreadLocalMode',
    'c__EA_LLVMTypeKind', 'c__EA_LLVMUnnamedAddr',
    'c__EA_LLVMValueKind', 'c__EA_LLVMVerifierFailureAction',
    'c__EA_LLVMVisibility', 'c__EA_lto_codegen_diagnostic_severity_t',
    'c__EA_lto_codegen_model', 'c__EA_lto_debug_model',
    'c__EA_lto_symbol_attributes', 'c__Ea_LLVMAttributeReturnIndex',
    'c__Ea_LLVMMDStringMetadataKind', 'int64_t', 'lto_api_version',
    'lto_bool_t', 'lto_code_gen_t', 'lto_codegen_add_module',
    'lto_codegen_add_must_preserve_symbol', 'lto_codegen_compile',
    'lto_codegen_compile_optimized', 'lto_codegen_compile_to_file',
    'lto_codegen_create', 'lto_codegen_create_in_local_context',
    'lto_codegen_debug_options', 'lto_codegen_debug_options_array',
    'lto_codegen_diagnostic_severity_t',
    'lto_codegen_diagnostic_severity_t__enumvalues',
    'lto_codegen_dispose', 'lto_codegen_model',
    'lto_codegen_model__enumvalues', 'lto_codegen_optimize',
    'lto_codegen_set_assembler_args',
    'lto_codegen_set_assembler_path', 'lto_codegen_set_cpu',
    'lto_codegen_set_debug_model',
    'lto_codegen_set_diagnostic_handler', 'lto_codegen_set_module',
    'lto_codegen_set_pic_model',
    'lto_codegen_set_should_embed_uselists',
    'lto_codegen_set_should_internalize',
    'lto_codegen_write_merged_modules', 'lto_debug_model',
    'lto_debug_model__enumvalues', 'lto_diagnostic_handler_t',
    'lto_get_error_message', 'lto_get_version',
    'lto_initialize_disassembler', 'lto_input_create',
    'lto_input_dispose', 'lto_input_get_dependent_library',
    'lto_input_get_num_dependent_libraries', 'lto_input_t',
    'lto_module_create', 'lto_module_create_from_fd',
    'lto_module_create_from_fd_at_offset',
    'lto_module_create_from_memory',
    'lto_module_create_from_memory_with_path',
    'lto_module_create_in_codegen_context',
    'lto_module_create_in_local_context', 'lto_module_dispose',
    'lto_module_get_linkeropts', 'lto_module_get_macho_cputype',
    'lto_module_get_num_symbols', 'lto_module_get_symbol_attribute',
    'lto_module_get_symbol_name', 'lto_module_get_target_triple',
    'lto_module_has_ctor_dtor', 'lto_module_has_objc_category',
    'lto_module_is_object_file',
    'lto_module_is_object_file_for_target',
    'lto_module_is_object_file_in_memory',
    'lto_module_is_object_file_in_memory_for_target',
    'lto_module_is_thinlto', 'lto_module_set_target_triple',
    'lto_module_t', 'lto_runtime_lib_symbols_list',
    'lto_set_debug_options', 'lto_symbol_attributes',
    'lto_symbol_attributes__enumvalues', 'off_t', 'size_t',
    'struct_LLVMComdat', 'struct_LLVMMCJITCompilerOptions',
    'struct_LLVMOpInfo1', 'struct_LLVMOpInfoSymbol1',
    'struct_LLVMOpaqueAttributeRef', 'struct_LLVMOpaqueBasicBlock',
    'struct_LLVMOpaqueBinary', 'struct_LLVMOpaqueBuilder',
    'struct_LLVMOpaqueContext', 'struct_LLVMOpaqueDIBuilder',
    'struct_LLVMOpaqueDiagnosticInfo', 'struct_LLVMOpaqueError',
    'struct_LLVMOpaqueExecutionEngine',
    'struct_LLVMOpaqueGenericValue',
    'struct_LLVMOpaqueJITEventListener',
    'struct_LLVMOpaqueLTOCodeGenerator', 'struct_LLVMOpaqueLTOInput',
    'struct_LLVMOpaqueLTOModule',
    'struct_LLVMOpaqueMCJITMemoryManager',
    'struct_LLVMOpaqueMemoryBuffer', 'struct_LLVMOpaqueMetadata',
    'struct_LLVMOpaqueModule', 'struct_LLVMOpaqueModuleFlagEntry',
    'struct_LLVMOpaqueModuleProvider', 'struct_LLVMOpaqueNamedMDNode',
    'struct_LLVMOpaqueObjectFile',
    'struct_LLVMOpaquePassBuilderOptions',
    'struct_LLVMOpaquePassManager',
    'struct_LLVMOpaquePassManagerBuilder',
    'struct_LLVMOpaquePassRegistry',
    'struct_LLVMOpaqueRelocationIterator',
    'struct_LLVMOpaqueSectionIterator',
    'struct_LLVMOpaqueSymbolIterator', 'struct_LLVMOpaqueTargetData',
    'struct_LLVMOpaqueTargetLibraryInfotData',
    'struct_LLVMOpaqueTargetMachine',
    'struct_LLVMOpaqueThinLTOCodeGenerator', 'struct_LLVMOpaqueType',
    'struct_LLVMOpaqueUse', 'struct_LLVMOpaqueValue',
    'struct_LLVMOpaqueValueMetadataEntry',
    'struct_LLVMOrcOpaqueDefinitionGenerator',
    'struct_LLVMOrcOpaqueDumpObjects',
    'struct_LLVMOrcOpaqueExecutionSession',
    'struct_LLVMOrcOpaqueIRTransformLayer',
    'struct_LLVMOrcOpaqueIndirectStubsManager',
    'struct_LLVMOrcOpaqueJITDylib',
    'struct_LLVMOrcOpaqueJITTargetMachineBuilder',
    'struct_LLVMOrcOpaqueLLJIT', 'struct_LLVMOrcOpaqueLLJITBuilder',
    'struct_LLVMOrcOpaqueLazyCallThroughManager',
    'struct_LLVMOrcOpaqueLookupState',
    'struct_LLVMOrcOpaqueMaterializationResponsibility',
    'struct_LLVMOrcOpaqueMaterializationUnit',
    'struct_LLVMOrcOpaqueObjectLayer',
    'struct_LLVMOrcOpaqueObjectLinkingLayer',
    'struct_LLVMOrcOpaqueObjectTransformLayer',
    'struct_LLVMOrcOpaqueResourceTracker',
    'struct_LLVMOrcOpaqueSymbolStringPool',
    'struct_LLVMOrcOpaqueSymbolStringPoolEntry',
    'struct_LLVMOrcOpaqueThreadSafeContext',
    'struct_LLVMOrcOpaqueThreadSafeModule',
    'struct_LLVMRemarkOpaqueArg', 'struct_LLVMRemarkOpaqueDebugLoc',
    'struct_LLVMRemarkOpaqueEntry', 'struct_LLVMRemarkOpaqueParser',
    'struct_LLVMRemarkOpaqueString', 'struct_LLVMTarget',
    'struct_c__SA_LLVMJITCSymbolMapPair',
    'struct_c__SA_LLVMJITEvaluatedSymbol',
    'struct_c__SA_LLVMJITSymbolFlags',
    'struct_c__SA_LLVMOrcCDependenceMapPair',
    'struct_c__SA_LLVMOrcCLookupSetElement',
    'struct_c__SA_LLVMOrcCSymbolAliasMapEntry',
    'struct_c__SA_LLVMOrcCSymbolAliasMapPair',
    'struct_c__SA_LLVMOrcCSymbolFlagsMapPair',
    'struct_c__SA_LLVMOrcCSymbolsList',
    'struct_c__SA_LTOObjectBuffer', 'thinlto_code_gen_t',
    'thinlto_codegen_add_cross_referenced_symbol',
    'thinlto_codegen_add_module',
    'thinlto_codegen_add_must_preserve_symbol',
    'thinlto_codegen_disable_codegen', 'thinlto_codegen_dispose',
    'thinlto_codegen_process', 'thinlto_codegen_set_cache_dir',
    'thinlto_codegen_set_cache_entry_expiration',
    'thinlto_codegen_set_cache_pruning_interval',
    'thinlto_codegen_set_cache_size_bytes',
    'thinlto_codegen_set_cache_size_files',
    'thinlto_codegen_set_cache_size_megabytes',
    'thinlto_codegen_set_codegen_only', 'thinlto_codegen_set_cpu',
    'thinlto_codegen_set_final_cache_size_relative_to_available_space',
    'thinlto_codegen_set_pic_model',
    'thinlto_codegen_set_savetemps_dir', 'thinlto_create_codegen',
    'thinlto_debug_options', 'thinlto_module_get_num_object_files',
    'thinlto_module_get_num_objects', 'thinlto_module_get_object',
    'thinlto_module_get_object_file',
    'thinlto_set_generated_objects_dir', 'uint32_t', 'uint64_t',
    'uint8_t']
