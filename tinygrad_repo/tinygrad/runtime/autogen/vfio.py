# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


from tinygrad.runtime.support.hcq import HWInterface
import functools

def _do_ioctl_io(__idir, __base, __nr, __fd:HWInterface, val=0, __len=0):
  return __fd.ioctl((__idir<<30) | (__len<<16) | (__base<<8) | __nr, val)

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd:HWInterface, __val=None, **kwargs):
  ret = __fd.ioctl((__idir<<30) | (ctypes.sizeof(made := (__made or __user_struct(**kwargs)))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl_io, 0, ord(base) if isinstance(base, str) else base, nr)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)

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





VFIO_H = True # macro
VFIO_API_VERSION = 0 # macro
VFIO_TYPE1_IOMMU = 1 # macro
VFIO_SPAPR_TCE_IOMMU = 2 # macro
VFIO_TYPE1v2_IOMMU = 3 # macro
VFIO_DMA_CC_IOMMU = 4 # macro
VFIO_EEH = 5 # macro
VFIO_TYPE1_NESTING_IOMMU = 6 # macro
VFIO_SPAPR_TCE_v2_IOMMU = 7 # macro
VFIO_NOIOMMU_IOMMU = 8 # macro
VFIO_UNMAP_ALL = 9 # macro
VFIO_UPDATE_VADDR = 10 # macro
VFIO_TYPE = (';') # macro
VFIO_BASE = 100 # macro
VFIO_GET_API_VERSION = _IO ( ( ';' ) , 100 + 0 ) # macro (from list)
VFIO_CHECK_EXTENSION = _IO ( ( ';' ) , 100 + 1 ) # macro (from list)
VFIO_SET_IOMMU = _IO ( ( ';' ) , 100 + 2 ) # macro (from list)
VFIO_GROUP_FLAGS_VIABLE = (1<<0) # macro
VFIO_GROUP_FLAGS_CONTAINER_SET = (1<<1) # macro
VFIO_GROUP_GET_STATUS = _IO ( ( ';' ) , 100 + 3 ) # macro (from list)
VFIO_GROUP_SET_CONTAINER = _IO ( ( ';' ) , 100 + 4 ) # macro (from list)
VFIO_GROUP_UNSET_CONTAINER = _IO ( ( ';' ) , 100 + 5 ) # macro (from list)
VFIO_GROUP_GET_DEVICE_FD = _IO ( ( ';' ) , 100 + 6 ) # macro (from list)
VFIO_DEVICE_FLAGS_RESET = (1<<0) # macro
VFIO_DEVICE_FLAGS_PCI = (1<<1) # macro
VFIO_DEVICE_FLAGS_PLATFORM = (1<<2) # macro
VFIO_DEVICE_FLAGS_AMBA = (1<<3) # macro
VFIO_DEVICE_FLAGS_CCW = (1<<4) # macro
VFIO_DEVICE_FLAGS_AP = (1<<5) # macro
VFIO_DEVICE_FLAGS_FSL_MC = (1<<6) # macro
VFIO_DEVICE_FLAGS_CAPS = (1<<7) # macro
VFIO_DEVICE_GET_INFO = _IO ( ( ';' ) , 100 + 7 ) # macro (from list)
VFIO_DEVICE_API_PCI_STRING = "vfio-pci" # macro
VFIO_DEVICE_API_PLATFORM_STRING = "vfio-platform" # macro
VFIO_DEVICE_API_AMBA_STRING = "vfio-amba" # macro
VFIO_DEVICE_API_CCW_STRING = "vfio-ccw" # macro
VFIO_DEVICE_API_AP_STRING = "vfio-ap" # macro
VFIO_DEVICE_INFO_CAP_ZPCI_BASE = 1 # macro
VFIO_DEVICE_INFO_CAP_ZPCI_GROUP = 2 # macro
VFIO_DEVICE_INFO_CAP_ZPCI_UTIL = 3 # macro
VFIO_DEVICE_INFO_CAP_ZPCI_PFIP = 4 # macro
VFIO_REGION_INFO_FLAG_READ = (1<<0) # macro
VFIO_REGION_INFO_FLAG_WRITE = (1<<1) # macro
VFIO_REGION_INFO_FLAG_MMAP = (1<<2) # macro
VFIO_REGION_INFO_FLAG_CAPS = (1<<3) # macro
VFIO_DEVICE_GET_REGION_INFO = _IO ( ( ';' ) , 100 + 8 ) # macro (from list)
VFIO_REGION_INFO_CAP_SPARSE_MMAP = 1 # macro
VFIO_REGION_INFO_CAP_TYPE = 2 # macro
VFIO_REGION_TYPE_PCI_VENDOR_TYPE = (1<<31) # macro
VFIO_REGION_TYPE_PCI_VENDOR_MASK = (0xffff) # macro
VFIO_REGION_TYPE_GFX = (1) # macro
VFIO_REGION_TYPE_CCW = (2) # macro
VFIO_REGION_TYPE_MIGRATION = (3) # macro
VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION = (1) # macro
VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG = (2) # macro
VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG = (3) # macro
VFIO_REGION_SUBTYPE_NVIDIA_NVLINK2_RAM = (1) # macro
VFIO_REGION_SUBTYPE_IBM_NVLINK2_ATSD = (1) # macro
VFIO_REGION_SUBTYPE_GFX_EDID = (1) # macro
VFIO_DEVICE_GFX_LINK_STATE_UP = 1 # macro
VFIO_DEVICE_GFX_LINK_STATE_DOWN = 2 # macro
VFIO_REGION_SUBTYPE_CCW_ASYNC_CMD = (1) # macro
VFIO_REGION_SUBTYPE_CCW_SCHIB = (2) # macro
VFIO_REGION_SUBTYPE_CCW_CRW = (3) # macro
VFIO_REGION_SUBTYPE_MIGRATION = (1) # macro
VFIO_DEVICE_STATE_STOP = (0) # macro
VFIO_DEVICE_STATE_RUNNING = (1<<0) # macro
VFIO_DEVICE_STATE_SAVING = (1<<1) # macro
VFIO_DEVICE_STATE_RESUMING = (1<<2) # macro
VFIO_DEVICE_STATE_MASK = ((1<<0)|(1<<1)|(1<<2)) # macro
# def VFIO_DEVICE_STATE_VALID(state):  # macro
#    return (state&(1<<2)?(state&((1<<0)|(1<<1)|(1<<2)))==(1<<2):1)
def VFIO_DEVICE_STATE_IS_ERROR(state):  # macro
   return ((state&((1<<0)|(1<<1)|(1<<2)))==((1<<1)|(1<<2)))
# def VFIO_DEVICE_STATE_SET_ERROR(state):  # macro
#    return ((state&~((1<<0)|(1<<1)|(1<<2)))|VFIO_DEVICE_SATE_SAVING|(1<<2))
VFIO_REGION_INFO_CAP_MSIX_MAPPABLE = 3 # macro
VFIO_REGION_INFO_CAP_NVLINK2_SSATGT = 4 # macro
VFIO_REGION_INFO_CAP_NVLINK2_LNKSPD = 5 # macro
VFIO_IRQ_INFO_EVENTFD = (1<<0) # macro
VFIO_IRQ_INFO_MASKABLE = (1<<1) # macro
VFIO_IRQ_INFO_AUTOMASKED = (1<<2) # macro
VFIO_IRQ_INFO_NORESIZE = (1<<3) # macro
VFIO_DEVICE_GET_IRQ_INFO = _IO ( ( ';' ) , 100 + 9 ) # macro (from list)
VFIO_IRQ_SET_DATA_NONE = (1<<0) # macro
VFIO_IRQ_SET_DATA_BOOL = (1<<1) # macro
VFIO_IRQ_SET_DATA_EVENTFD = (1<<2) # macro
VFIO_IRQ_SET_ACTION_MASK = (1<<3) # macro
VFIO_IRQ_SET_ACTION_UNMASK = (1<<4) # macro
VFIO_IRQ_SET_ACTION_TRIGGER = (1<<5) # macro
VFIO_DEVICE_SET_IRQS = _IO ( ( ';' ) , 100 + 10 ) # macro (from list)
VFIO_IRQ_SET_DATA_TYPE_MASK = ((1<<0)|(1<<1)|(1<<2)) # macro
VFIO_IRQ_SET_ACTION_TYPE_MASK = ((1<<3)|(1<<4)|(1<<5)) # macro
VFIO_DEVICE_RESET = _IO ( ( ';' ) , 100 + 11 ) # macro (from list)
VFIO_DEVICE_GET_PCI_HOT_RESET_INFO = _IO ( ( ';' ) , 100 + 12 ) # macro (from list)
VFIO_DEVICE_PCI_HOT_RESET = _IO ( ( ';' ) , 100 + 13 ) # macro (from list)
VFIO_GFX_PLANE_TYPE_PROBE = (1<<0) # macro
VFIO_GFX_PLANE_TYPE_DMABUF = (1<<1) # macro
VFIO_GFX_PLANE_TYPE_REGION = (1<<2) # macro
VFIO_DEVICE_QUERY_GFX_PLANE = _IO ( ( ';' ) , 100 + 14 ) # macro (from list)
VFIO_DEVICE_GET_GFX_DMABUF = _IO ( ( ';' ) , 100 + 15 ) # macro (from list)
VFIO_DEVICE_IOEVENTFD_8 = (1<<0) # macro
VFIO_DEVICE_IOEVENTFD_16 = (1<<1) # macro
VFIO_DEVICE_IOEVENTFD_32 = (1<<2) # macro
VFIO_DEVICE_IOEVENTFD_64 = (1<<3) # macro
VFIO_DEVICE_IOEVENTFD_SIZE_MASK = (0xf) # macro
VFIO_DEVICE_IOEVENTFD = _IO ( ( ';' ) , 100 + 16 ) # macro (from list)
VFIO_DEVICE_FEATURE_MASK = (0xffff) # macro
VFIO_DEVICE_FEATURE_GET = (1<<16) # macro
VFIO_DEVICE_FEATURE_SET = (1<<17) # macro
VFIO_DEVICE_FEATURE_PROBE = (1<<18) # macro
VFIO_DEVICE_FEATURE = _IO ( ( ';' ) , 100 + 17 ) # macro (from list)
VFIO_DEVICE_FEATURE_PCI_VF_TOKEN = (0) # macro
VFIO_IOMMU_INFO_PGSIZES = (1<<0) # macro
VFIO_IOMMU_INFO_CAPS = (1<<1) # macro
VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE = 1 # macro
VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION = 2 # macro
VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL = 3 # macro
VFIO_IOMMU_GET_INFO = _IO ( ( ';' ) , 100 + 12 ) # macro (from list)
VFIO_DMA_MAP_FLAG_READ = (1<<0) # macro
VFIO_DMA_MAP_FLAG_WRITE = (1<<1) # macro
VFIO_DMA_MAP_FLAG_VADDR = (1<<2) # macro
VFIO_IOMMU_MAP_DMA = _IO ( ( ';' ) , 100 + 13 ) # macro (from list)
VFIO_DMA_UNMAP_FLAG_GET_DIRTY_BITMAP = (1<<0) # macro
VFIO_DMA_UNMAP_FLAG_ALL = (1<<1) # macro
VFIO_DMA_UNMAP_FLAG_VADDR = (1<<2) # macro
VFIO_IOMMU_UNMAP_DMA = _IO ( ( ';' ) , 100 + 14 ) # macro (from list)
VFIO_IOMMU_ENABLE = _IO ( ( ';' ) , 100 + 15 ) # macro (from list)
VFIO_IOMMU_DISABLE = _IO ( ( ';' ) , 100 + 16 ) # macro (from list)
VFIO_IOMMU_DIRTY_PAGES_FLAG_START = (1<<0) # macro
VFIO_IOMMU_DIRTY_PAGES_FLAG_STOP = (1<<1) # macro
VFIO_IOMMU_DIRTY_PAGES_FLAG_GET_BITMAP = (1<<2) # macro
VFIO_IOMMU_DIRTY_PAGES = _IO ( ( ';' ) , 100 + 17 ) # macro (from list)
VFIO_IOMMU_SPAPR_INFO_DDW = (1<<0) # macro
VFIO_IOMMU_SPAPR_TCE_GET_INFO = _IO ( ( ';' ) , 100 + 12 ) # macro (from list)
VFIO_EEH_PE_DISABLE = 0 # macro
VFIO_EEH_PE_ENABLE = 1 # macro
VFIO_EEH_PE_UNFREEZE_IO = 2 # macro
VFIO_EEH_PE_UNFREEZE_DMA = 3 # macro
VFIO_EEH_PE_GET_STATE = 4 # macro
VFIO_EEH_PE_STATE_NORMAL = 0 # macro
VFIO_EEH_PE_STATE_RESET = 1 # macro
VFIO_EEH_PE_STATE_STOPPED = 2 # macro
VFIO_EEH_PE_STATE_STOPPED_DMA = 4 # macro
VFIO_EEH_PE_STATE_UNAVAIL = 5 # macro
VFIO_EEH_PE_RESET_DEACTIVATE = 5 # macro
VFIO_EEH_PE_RESET_HOT = 6 # macro
VFIO_EEH_PE_RESET_FUNDAMENTAL = 7 # macro
VFIO_EEH_PE_CONFIGURE = 8 # macro
VFIO_EEH_PE_INJECT_ERR = 9 # macro
VFIO_EEH_PE_OP = _IO ( ( ';' ) , 100 + 21 ) # macro (from list)
VFIO_IOMMU_SPAPR_REGISTER_MEMORY = _IO ( ( ';' ) , 100 + 17 ) # macro (from list)
VFIO_IOMMU_SPAPR_UNREGISTER_MEMORY = _IO ( ( ';' ) , 100 + 18 ) # macro (from list)
VFIO_IOMMU_SPAPR_TCE_CREATE = _IO ( ( ';' ) , 100 + 19 ) # macro (from list)
VFIO_IOMMU_SPAPR_TCE_REMOVE = _IO ( ( ';' ) , 100 + 20 ) # macro (from list)
class struct_vfio_info_cap_header(Structure):
    pass

struct_vfio_info_cap_header._pack_ = 1 # source:False
struct_vfio_info_cap_header._fields_ = [
    ('id', ctypes.c_uint16),
    ('version', ctypes.c_uint16),
    ('next', ctypes.c_uint32),
]

class struct_vfio_group_status(Structure):
    pass

struct_vfio_group_status._pack_ = 1 # source:False
struct_vfio_group_status._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_vfio_device_info(Structure):
    pass

struct_vfio_device_info._pack_ = 1 # source:False
struct_vfio_device_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('num_regions', ctypes.c_uint32),
    ('num_irqs', ctypes.c_uint32),
    ('cap_offset', ctypes.c_uint32),
]

class struct_vfio_region_info(Structure):
    pass

struct_vfio_region_info._pack_ = 1 # source:False
struct_vfio_region_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('cap_offset', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

class struct_vfio_region_sparse_mmap_area(Structure):
    pass

struct_vfio_region_sparse_mmap_area._pack_ = 1 # source:False
struct_vfio_region_sparse_mmap_area._fields_ = [
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_vfio_region_info_cap_sparse_mmap(Structure):
    pass

struct_vfio_region_info_cap_sparse_mmap._pack_ = 1 # source:False
struct_vfio_region_info_cap_sparse_mmap._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('nr_areas', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('areas', struct_vfio_region_sparse_mmap_area * 0),
]

class struct_vfio_region_info_cap_type(Structure):
    pass

struct_vfio_region_info_cap_type._pack_ = 1 # source:False
struct_vfio_region_info_cap_type._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('type', ctypes.c_uint32),
    ('subtype', ctypes.c_uint32),
]

class struct_vfio_region_gfx_edid(Structure):
    pass

struct_vfio_region_gfx_edid._pack_ = 1 # source:False
struct_vfio_region_gfx_edid._fields_ = [
    ('edid_offset', ctypes.c_uint32),
    ('edid_max_size', ctypes.c_uint32),
    ('edid_size', ctypes.c_uint32),
    ('max_xres', ctypes.c_uint32),
    ('max_yres', ctypes.c_uint32),
    ('link_state', ctypes.c_uint32),
]

class struct_vfio_device_migration_info(Structure):
    pass

struct_vfio_device_migration_info._pack_ = 1 # source:False
struct_vfio_device_migration_info._fields_ = [
    ('device_state', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('pending_bytes', ctypes.c_uint64),
    ('data_offset', ctypes.c_uint64),
    ('data_size', ctypes.c_uint64),
]

class struct_vfio_region_info_cap_nvlink2_ssatgt(Structure):
    pass

struct_vfio_region_info_cap_nvlink2_ssatgt._pack_ = 1 # source:False
struct_vfio_region_info_cap_nvlink2_ssatgt._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('tgt', ctypes.c_uint64),
]

class struct_vfio_region_info_cap_nvlink2_lnkspd(Structure):
    pass

struct_vfio_region_info_cap_nvlink2_lnkspd._pack_ = 1 # source:False
struct_vfio_region_info_cap_nvlink2_lnkspd._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('link_speed', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32),
]

class struct_vfio_irq_info(Structure):
    pass

struct_vfio_irq_info._pack_ = 1 # source:False
struct_vfio_irq_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

class struct_vfio_irq_set(Structure):
    pass

struct_vfio_irq_set._pack_ = 1 # source:False
struct_vfio_irq_set._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('start', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('data', ctypes.c_int * 1),
]


# values for enumeration 'c__Ea_VFIO_PCI_BAR0_REGION_INDEX'
c__Ea_VFIO_PCI_BAR0_REGION_INDEX__enumvalues = {
    0: 'VFIO_PCI_BAR0_REGION_INDEX',
    1: 'VFIO_PCI_BAR1_REGION_INDEX',
    2: 'VFIO_PCI_BAR2_REGION_INDEX',
    3: 'VFIO_PCI_BAR3_REGION_INDEX',
    4: 'VFIO_PCI_BAR4_REGION_INDEX',
    5: 'VFIO_PCI_BAR5_REGION_INDEX',
    6: 'VFIO_PCI_ROM_REGION_INDEX',
    7: 'VFIO_PCI_CONFIG_REGION_INDEX',
    8: 'VFIO_PCI_VGA_REGION_INDEX',
    9: 'VFIO_PCI_NUM_REGIONS',
}
VFIO_PCI_BAR0_REGION_INDEX = 0
VFIO_PCI_BAR1_REGION_INDEX = 1
VFIO_PCI_BAR2_REGION_INDEX = 2
VFIO_PCI_BAR3_REGION_INDEX = 3
VFIO_PCI_BAR4_REGION_INDEX = 4
VFIO_PCI_BAR5_REGION_INDEX = 5
VFIO_PCI_ROM_REGION_INDEX = 6
VFIO_PCI_CONFIG_REGION_INDEX = 7
VFIO_PCI_VGA_REGION_INDEX = 8
VFIO_PCI_NUM_REGIONS = 9
c__Ea_VFIO_PCI_BAR0_REGION_INDEX = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_VFIO_PCI_INTX_IRQ_INDEX'
c__Ea_VFIO_PCI_INTX_IRQ_INDEX__enumvalues = {
    0: 'VFIO_PCI_INTX_IRQ_INDEX',
    1: 'VFIO_PCI_MSI_IRQ_INDEX',
    2: 'VFIO_PCI_MSIX_IRQ_INDEX',
    3: 'VFIO_PCI_ERR_IRQ_INDEX',
    4: 'VFIO_PCI_REQ_IRQ_INDEX',
    5: 'VFIO_PCI_NUM_IRQS',
}
VFIO_PCI_INTX_IRQ_INDEX = 0
VFIO_PCI_MSI_IRQ_INDEX = 1
VFIO_PCI_MSIX_IRQ_INDEX = 2
VFIO_PCI_ERR_IRQ_INDEX = 3
VFIO_PCI_REQ_IRQ_INDEX = 4
VFIO_PCI_NUM_IRQS = 5
c__Ea_VFIO_PCI_INTX_IRQ_INDEX = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_VFIO_CCW_CONFIG_REGION_INDEX'
c__Ea_VFIO_CCW_CONFIG_REGION_INDEX__enumvalues = {
    0: 'VFIO_CCW_CONFIG_REGION_INDEX',
    1: 'VFIO_CCW_NUM_REGIONS',
}
VFIO_CCW_CONFIG_REGION_INDEX = 0
VFIO_CCW_NUM_REGIONS = 1
c__Ea_VFIO_CCW_CONFIG_REGION_INDEX = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_VFIO_CCW_IO_IRQ_INDEX'
c__Ea_VFIO_CCW_IO_IRQ_INDEX__enumvalues = {
    0: 'VFIO_CCW_IO_IRQ_INDEX',
    1: 'VFIO_CCW_CRW_IRQ_INDEX',
    2: 'VFIO_CCW_REQ_IRQ_INDEX',
    3: 'VFIO_CCW_NUM_IRQS',
}
VFIO_CCW_IO_IRQ_INDEX = 0
VFIO_CCW_CRW_IRQ_INDEX = 1
VFIO_CCW_REQ_IRQ_INDEX = 2
VFIO_CCW_NUM_IRQS = 3
c__Ea_VFIO_CCW_IO_IRQ_INDEX = ctypes.c_uint32 # enum
class struct_vfio_pci_dependent_device(Structure):
    pass

struct_vfio_pci_dependent_device._pack_ = 1 # source:False
struct_vfio_pci_dependent_device._fields_ = [
    ('group_id', ctypes.c_uint32),
    ('segment', ctypes.c_uint16),
    ('bus', ctypes.c_ubyte),
    ('devfn', ctypes.c_ubyte),
]

class struct_vfio_pci_hot_reset_info(Structure):
    pass

struct_vfio_pci_hot_reset_info._pack_ = 1 # source:False
struct_vfio_pci_hot_reset_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('devices', struct_vfio_pci_dependent_device * 0),
]

class struct_vfio_pci_hot_reset(Structure):
    pass

struct_vfio_pci_hot_reset._pack_ = 1 # source:False
struct_vfio_pci_hot_reset._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('group_fds', ctypes.c_int32 * 0),
]

class struct_vfio_device_gfx_plane_info(Structure):
    pass

class union_vfio_device_gfx_plane_info_0(Union):
    pass

union_vfio_device_gfx_plane_info_0._pack_ = 1 # source:False
union_vfio_device_gfx_plane_info_0._fields_ = [
    ('region_index', ctypes.c_uint32),
    ('dmabuf_id', ctypes.c_uint32),
]

struct_vfio_device_gfx_plane_info._pack_ = 1 # source:False
struct_vfio_device_gfx_plane_info._anonymous_ = ('_0',)
struct_vfio_device_gfx_plane_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('drm_plane_type', ctypes.c_uint32),
    ('drm_format', ctypes.c_uint32),
    ('drm_format_mod', ctypes.c_uint64),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('stride', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('x_pos', ctypes.c_uint32),
    ('y_pos', ctypes.c_uint32),
    ('x_hot', ctypes.c_uint32),
    ('y_hot', ctypes.c_uint32),
    ('_0', union_vfio_device_gfx_plane_info_0),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_vfio_device_ioeventfd(Structure):
    pass

struct_vfio_device_ioeventfd._pack_ = 1 # source:False
struct_vfio_device_ioeventfd._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
    ('data', ctypes.c_uint64),
    ('fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_vfio_device_feature(Structure):
    pass

struct_vfio_device_feature._pack_ = 1 # source:False
struct_vfio_device_feature._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 0),
]

class struct_vfio_iommu_type1_info(Structure):
    pass

struct_vfio_iommu_type1_info._pack_ = 1 # source:False
struct_vfio_iommu_type1_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('iova_pgsizes', ctypes.c_uint64),
    ('cap_offset', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_vfio_iova_range(Structure):
    pass

struct_vfio_iova_range._pack_ = 1 # source:False
struct_vfio_iova_range._fields_ = [
    ('start', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

class struct_vfio_iommu_type1_info_cap_iova_range(Structure):
    pass

struct_vfio_iommu_type1_info_cap_iova_range._pack_ = 1 # source:False
struct_vfio_iommu_type1_info_cap_iova_range._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('nr_iovas', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('iova_ranges', struct_vfio_iova_range * 0),
]

class struct_vfio_iommu_type1_info_cap_migration(Structure):
    pass

struct_vfio_iommu_type1_info_cap_migration._pack_ = 1 # source:False
struct_vfio_iommu_type1_info_cap_migration._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pgsize_bitmap', ctypes.c_uint64),
    ('max_dirty_bitmap_size', ctypes.c_uint64),
]

class struct_vfio_iommu_type1_info_dma_avail(Structure):
    pass

struct_vfio_iommu_type1_info_dma_avail._pack_ = 1 # source:False
struct_vfio_iommu_type1_info_dma_avail._fields_ = [
    ('header', struct_vfio_info_cap_header),
    ('avail', ctypes.c_uint32),
]

class struct_vfio_iommu_type1_dma_map(Structure):
    pass

struct_vfio_iommu_type1_dma_map._pack_ = 1 # source:False
struct_vfio_iommu_type1_dma_map._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('vaddr', ctypes.c_uint64),
    ('iova', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_vfio_bitmap(Structure):
    pass

struct_vfio_bitmap._pack_ = 1 # source:False
struct_vfio_bitmap._fields_ = [
    ('pgsize', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('data', ctypes.POINTER(ctypes.c_uint64)),
]

class struct_vfio_iommu_type1_dma_unmap(Structure):
    pass

struct_vfio_iommu_type1_dma_unmap._pack_ = 1 # source:False
struct_vfio_iommu_type1_dma_unmap._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('iova', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('data', ctypes.c_ubyte * 0),
]

class struct_vfio_iommu_type1_dirty_bitmap(Structure):
    pass

struct_vfio_iommu_type1_dirty_bitmap._pack_ = 1 # source:False
struct_vfio_iommu_type1_dirty_bitmap._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 0),
]

class struct_vfio_iommu_type1_dirty_bitmap_get(Structure):
    pass

struct_vfio_iommu_type1_dirty_bitmap_get._pack_ = 1 # source:False
struct_vfio_iommu_type1_dirty_bitmap_get._fields_ = [
    ('iova', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('bitmap', struct_vfio_bitmap),
]

class struct_vfio_iommu_spapr_tce_ddw_info(Structure):
    pass

struct_vfio_iommu_spapr_tce_ddw_info._pack_ = 1 # source:False
struct_vfio_iommu_spapr_tce_ddw_info._fields_ = [
    ('pgsizes', ctypes.c_uint64),
    ('max_dynamic_windows_supported', ctypes.c_uint32),
    ('levels', ctypes.c_uint32),
]

class struct_vfio_iommu_spapr_tce_info(Structure):
    pass

struct_vfio_iommu_spapr_tce_info._pack_ = 1 # source:False
struct_vfio_iommu_spapr_tce_info._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('dma32_window_start', ctypes.c_uint32),
    ('dma32_window_size', ctypes.c_uint32),
    ('ddw', struct_vfio_iommu_spapr_tce_ddw_info),
]

class struct_vfio_eeh_pe_err(Structure):
    pass

struct_vfio_eeh_pe_err._pack_ = 1 # source:False
struct_vfio_eeh_pe_err._fields_ = [
    ('type', ctypes.c_uint32),
    ('func', ctypes.c_uint32),
    ('addr', ctypes.c_uint64),
    ('mask', ctypes.c_uint64),
]

class struct_vfio_eeh_pe_op(Structure):
    pass

class union_vfio_eeh_pe_op_0(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('err', struct_vfio_eeh_pe_err),
     ]

struct_vfio_eeh_pe_op._pack_ = 1 # source:False
struct_vfio_eeh_pe_op._anonymous_ = ('_0',)
struct_vfio_eeh_pe_op._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_vfio_eeh_pe_op_0),
]

class struct_vfio_iommu_spapr_register_memory(Structure):
    pass

struct_vfio_iommu_spapr_register_memory._pack_ = 1 # source:False
struct_vfio_iommu_spapr_register_memory._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('vaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_vfio_iommu_spapr_tce_create(Structure):
    pass

struct_vfio_iommu_spapr_tce_create._pack_ = 1 # source:False
struct_vfio_iommu_spapr_tce_create._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('page_shift', ctypes.c_uint32),
    ('__resv1', ctypes.c_uint32),
    ('window_size', ctypes.c_uint64),
    ('levels', ctypes.c_uint32),
    ('__resv2', ctypes.c_uint32),
    ('start_addr', ctypes.c_uint64),
]

class struct_vfio_iommu_spapr_tce_remove(Structure):
    pass

struct_vfio_iommu_spapr_tce_remove._pack_ = 1 # source:False
struct_vfio_iommu_spapr_tce_remove._fields_ = [
    ('argsz', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('start_addr', ctypes.c_uint64),
]

__all__ = \
    ['VFIO_API_VERSION', 'VFIO_BASE', 'VFIO_CCW_CONFIG_REGION_INDEX',
    'VFIO_CCW_CRW_IRQ_INDEX', 'VFIO_CCW_IO_IRQ_INDEX',
    'VFIO_CCW_NUM_IRQS', 'VFIO_CCW_NUM_REGIONS',
    'VFIO_CCW_REQ_IRQ_INDEX', 'VFIO_DEVICE_API_AMBA_STRING',
    'VFIO_DEVICE_API_AP_STRING', 'VFIO_DEVICE_API_CCW_STRING',
    'VFIO_DEVICE_API_PCI_STRING', 'VFIO_DEVICE_API_PLATFORM_STRING',
    'VFIO_DEVICE_FEATURE_GET', 'VFIO_DEVICE_FEATURE_MASK',
    'VFIO_DEVICE_FEATURE_PCI_VF_TOKEN', 'VFIO_DEVICE_FEATURE_PROBE',
    'VFIO_DEVICE_FEATURE_SET', 'VFIO_DEVICE_FLAGS_AMBA',
    'VFIO_DEVICE_FLAGS_AP', 'VFIO_DEVICE_FLAGS_CAPS',
    'VFIO_DEVICE_FLAGS_CCW', 'VFIO_DEVICE_FLAGS_FSL_MC',
    'VFIO_DEVICE_FLAGS_PCI', 'VFIO_DEVICE_FLAGS_PLATFORM',
    'VFIO_DEVICE_FLAGS_RESET', 'VFIO_DEVICE_GFX_LINK_STATE_DOWN',
    'VFIO_DEVICE_GFX_LINK_STATE_UP', 'VFIO_DEVICE_INFO_CAP_ZPCI_BASE',
    'VFIO_DEVICE_INFO_CAP_ZPCI_GROUP',
    'VFIO_DEVICE_INFO_CAP_ZPCI_PFIP',
    'VFIO_DEVICE_INFO_CAP_ZPCI_UTIL', 'VFIO_DEVICE_IOEVENTFD_16',
    'VFIO_DEVICE_IOEVENTFD_32', 'VFIO_DEVICE_IOEVENTFD_64',
    'VFIO_DEVICE_IOEVENTFD_8', 'VFIO_DEVICE_IOEVENTFD_SIZE_MASK',
    'VFIO_DEVICE_STATE_MASK', 'VFIO_DEVICE_STATE_RESUMING',
    'VFIO_DEVICE_STATE_RUNNING', 'VFIO_DEVICE_STATE_SAVING',
    'VFIO_DEVICE_STATE_STOP', 'VFIO_DMA_CC_IOMMU',
    'VFIO_DMA_MAP_FLAG_READ', 'VFIO_DMA_MAP_FLAG_VADDR',
    'VFIO_DMA_MAP_FLAG_WRITE', 'VFIO_DMA_UNMAP_FLAG_ALL',
    'VFIO_DMA_UNMAP_FLAG_GET_DIRTY_BITMAP',
    'VFIO_DMA_UNMAP_FLAG_VADDR', 'VFIO_EEH', 'VFIO_EEH_PE_CONFIGURE',
    'VFIO_EEH_PE_DISABLE', 'VFIO_EEH_PE_ENABLE',
    'VFIO_EEH_PE_GET_STATE', 'VFIO_EEH_PE_INJECT_ERR',
    'VFIO_EEH_PE_RESET_DEACTIVATE', 'VFIO_EEH_PE_RESET_FUNDAMENTAL',
    'VFIO_EEH_PE_RESET_HOT', 'VFIO_EEH_PE_STATE_NORMAL',
    'VFIO_EEH_PE_STATE_RESET', 'VFIO_EEH_PE_STATE_STOPPED',
    'VFIO_EEH_PE_STATE_STOPPED_DMA', 'VFIO_EEH_PE_STATE_UNAVAIL',
    'VFIO_EEH_PE_UNFREEZE_DMA', 'VFIO_EEH_PE_UNFREEZE_IO',
    'VFIO_GFX_PLANE_TYPE_DMABUF', 'VFIO_GFX_PLANE_TYPE_PROBE',
    'VFIO_GFX_PLANE_TYPE_REGION', 'VFIO_GROUP_FLAGS_CONTAINER_SET',
    'VFIO_GROUP_FLAGS_VIABLE', 'VFIO_H',
    'VFIO_IOMMU_DIRTY_PAGES_FLAG_GET_BITMAP',
    'VFIO_IOMMU_DIRTY_PAGES_FLAG_START',
    'VFIO_IOMMU_DIRTY_PAGES_FLAG_STOP', 'VFIO_IOMMU_INFO_CAPS',
    'VFIO_IOMMU_INFO_PGSIZES', 'VFIO_IOMMU_SPAPR_INFO_DDW',
    'VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE',
    'VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION',
    'VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL', 'VFIO_IRQ_INFO_AUTOMASKED',
    'VFIO_IRQ_INFO_EVENTFD', 'VFIO_IRQ_INFO_MASKABLE',
    'VFIO_IRQ_INFO_NORESIZE', 'VFIO_IRQ_SET_ACTION_MASK',
    'VFIO_IRQ_SET_ACTION_TRIGGER', 'VFIO_IRQ_SET_ACTION_TYPE_MASK',
    'VFIO_IRQ_SET_ACTION_UNMASK', 'VFIO_IRQ_SET_DATA_BOOL',
    'VFIO_IRQ_SET_DATA_EVENTFD', 'VFIO_IRQ_SET_DATA_NONE',
    'VFIO_IRQ_SET_DATA_TYPE_MASK', 'VFIO_NOIOMMU_IOMMU',
    'VFIO_PCI_BAR0_REGION_INDEX', 'VFIO_PCI_BAR1_REGION_INDEX',
    'VFIO_PCI_BAR2_REGION_INDEX', 'VFIO_PCI_BAR3_REGION_INDEX',
    'VFIO_PCI_BAR4_REGION_INDEX', 'VFIO_PCI_BAR5_REGION_INDEX',
    'VFIO_PCI_CONFIG_REGION_INDEX', 'VFIO_PCI_ERR_IRQ_INDEX',
    'VFIO_PCI_INTX_IRQ_INDEX', 'VFIO_PCI_MSIX_IRQ_INDEX',
    'VFIO_PCI_MSI_IRQ_INDEX', 'VFIO_PCI_NUM_IRQS',
    'VFIO_PCI_NUM_REGIONS', 'VFIO_PCI_REQ_IRQ_INDEX',
    'VFIO_PCI_ROM_REGION_INDEX', 'VFIO_PCI_VGA_REGION_INDEX',
    'VFIO_REGION_INFO_CAP_MSIX_MAPPABLE',
    'VFIO_REGION_INFO_CAP_NVLINK2_LNKSPD',
    'VFIO_REGION_INFO_CAP_NVLINK2_SSATGT',
    'VFIO_REGION_INFO_CAP_SPARSE_MMAP', 'VFIO_REGION_INFO_CAP_TYPE',
    'VFIO_REGION_INFO_FLAG_CAPS', 'VFIO_REGION_INFO_FLAG_MMAP',
    'VFIO_REGION_INFO_FLAG_READ', 'VFIO_REGION_INFO_FLAG_WRITE',
    'VFIO_REGION_SUBTYPE_CCW_ASYNC_CMD',
    'VFIO_REGION_SUBTYPE_CCW_CRW', 'VFIO_REGION_SUBTYPE_CCW_SCHIB',
    'VFIO_REGION_SUBTYPE_GFX_EDID',
    'VFIO_REGION_SUBTYPE_IBM_NVLINK2_ATSD',
    'VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG',
    'VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG',
    'VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION',
    'VFIO_REGION_SUBTYPE_MIGRATION',
    'VFIO_REGION_SUBTYPE_NVIDIA_NVLINK2_RAM', 'VFIO_REGION_TYPE_CCW',
    'VFIO_REGION_TYPE_GFX', 'VFIO_REGION_TYPE_MIGRATION',
    'VFIO_REGION_TYPE_PCI_VENDOR_MASK',
    'VFIO_REGION_TYPE_PCI_VENDOR_TYPE', 'VFIO_SPAPR_TCE_IOMMU',
    'VFIO_SPAPR_TCE_v2_IOMMU', 'VFIO_TYPE', 'VFIO_TYPE1_IOMMU',
    'VFIO_TYPE1_NESTING_IOMMU', 'VFIO_TYPE1v2_IOMMU',
    'VFIO_UNMAP_ALL', 'VFIO_UPDATE_VADDR', '_IO', '_IOR', '_IOW',
    '_IOWR', 'c__Ea_VFIO_CCW_CONFIG_REGION_INDEX',
    'c__Ea_VFIO_CCW_IO_IRQ_INDEX', 'c__Ea_VFIO_PCI_BAR0_REGION_INDEX',
    'c__Ea_VFIO_PCI_INTX_IRQ_INDEX', 'struct_vfio_bitmap',
    'struct_vfio_device_feature', 'struct_vfio_device_gfx_plane_info',
    'struct_vfio_device_info', 'struct_vfio_device_ioeventfd',
    'struct_vfio_device_migration_info', 'struct_vfio_eeh_pe_err',
    'struct_vfio_eeh_pe_op', 'struct_vfio_group_status',
    'struct_vfio_info_cap_header',
    'struct_vfio_iommu_spapr_register_memory',
    'struct_vfio_iommu_spapr_tce_create',
    'struct_vfio_iommu_spapr_tce_ddw_info',
    'struct_vfio_iommu_spapr_tce_info',
    'struct_vfio_iommu_spapr_tce_remove',
    'struct_vfio_iommu_type1_dirty_bitmap',
    'struct_vfio_iommu_type1_dirty_bitmap_get',
    'struct_vfio_iommu_type1_dma_map',
    'struct_vfio_iommu_type1_dma_unmap',
    'struct_vfio_iommu_type1_info',
    'struct_vfio_iommu_type1_info_cap_iova_range',
    'struct_vfio_iommu_type1_info_cap_migration',
    'struct_vfio_iommu_type1_info_dma_avail',
    'struct_vfio_iova_range', 'struct_vfio_irq_info',
    'struct_vfio_irq_set', 'struct_vfio_pci_dependent_device',
    'struct_vfio_pci_hot_reset', 'struct_vfio_pci_hot_reset_info',
    'struct_vfio_region_gfx_edid', 'struct_vfio_region_info',
    'struct_vfio_region_info_cap_nvlink2_lnkspd',
    'struct_vfio_region_info_cap_nvlink2_ssatgt',
    'struct_vfio_region_info_cap_sparse_mmap',
    'struct_vfio_region_info_cap_type',
    'struct_vfio_region_sparse_mmap_area',
    'union_vfio_device_gfx_plane_info_0', 'union_vfio_eeh_pe_op_0']
