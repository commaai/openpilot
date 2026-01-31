# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_vfio_info_cap_header(c.Struct):
  SIZE = 8
  id: Annotated[Annotated[int, ctypes.c_uint16], 0]
  version: Annotated[Annotated[int, ctypes.c_uint16], 2]
  next: Annotated[Annotated[int, ctypes.c_uint32], 4]
__u16: TypeAlias = Annotated[int, ctypes.c_uint16]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_vfio_group_status(c.Struct):
  SIZE = 8
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_vfio_device_info(c.Struct):
  SIZE = 24
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  num_regions: Annotated[Annotated[int, ctypes.c_uint32], 8]
  num_irqs: Annotated[Annotated[int, ctypes.c_uint32], 12]
  cap_offset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_vfio_device_info_cap_pci_atomic_comp(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_region_info(c.Struct):
  SIZE = 32
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  index: Annotated[Annotated[int, ctypes.c_uint32], 8]
  cap_offset: Annotated[Annotated[int, ctypes.c_uint32], 12]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 24]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_vfio_region_sparse_mmap_area(c.Struct):
  SIZE = 16
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_vfio_region_info_cap_sparse_mmap(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  nr_areas: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  areas: Annotated[c.Array[struct_vfio_region_sparse_mmap_area, Literal[0]], 16]
@c.record
class struct_vfio_region_info_cap_type(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 8]
  subtype: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_region_gfx_edid(c.Struct):
  SIZE = 24
  edid_offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  edid_max_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  edid_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_xres: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_yres: Annotated[Annotated[int, ctypes.c_uint32], 16]
  link_state: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_vfio_device_migration_info(c.Struct):
  SIZE = 32
  device_state: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  pending_bytes: Annotated[Annotated[int, ctypes.c_uint64], 8]
  data_offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  data_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_vfio_region_info_cap_nvlink2_ssatgt(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  tgt: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_vfio_region_info_cap_nvlink2_lnkspd(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  link_speed: Annotated[Annotated[int, ctypes.c_uint32], 8]
  __pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_irq_info(c.Struct):
  SIZE = 16
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  index: Annotated[Annotated[int, ctypes.c_uint32], 8]
  count: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_irq_set(c.Struct):
  SIZE = 20
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  index: Annotated[Annotated[int, ctypes.c_uint32], 8]
  start: Annotated[Annotated[int, ctypes.c_uint32], 12]
  count: Annotated[Annotated[int, ctypes.c_uint32], 16]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 20]
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_PCI_BAR0_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR0_REGION_INDEX', 0)
VFIO_PCI_BAR1_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR1_REGION_INDEX', 1)
VFIO_PCI_BAR2_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR2_REGION_INDEX', 2)
VFIO_PCI_BAR3_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR3_REGION_INDEX', 3)
VFIO_PCI_BAR4_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR4_REGION_INDEX', 4)
VFIO_PCI_BAR5_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR5_REGION_INDEX', 5)
VFIO_PCI_ROM_REGION_INDEX = _anonenum0.define('VFIO_PCI_ROM_REGION_INDEX', 6)
VFIO_PCI_CONFIG_REGION_INDEX = _anonenum0.define('VFIO_PCI_CONFIG_REGION_INDEX', 7)
VFIO_PCI_VGA_REGION_INDEX = _anonenum0.define('VFIO_PCI_VGA_REGION_INDEX', 8)
VFIO_PCI_NUM_REGIONS = _anonenum0.define('VFIO_PCI_NUM_REGIONS', 9)

class _anonenum1(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_PCI_INTX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_INTX_IRQ_INDEX', 0)
VFIO_PCI_MSI_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSI_IRQ_INDEX', 1)
VFIO_PCI_MSIX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSIX_IRQ_INDEX', 2)
VFIO_PCI_ERR_IRQ_INDEX = _anonenum1.define('VFIO_PCI_ERR_IRQ_INDEX', 3)
VFIO_PCI_REQ_IRQ_INDEX = _anonenum1.define('VFIO_PCI_REQ_IRQ_INDEX', 4)
VFIO_PCI_NUM_IRQS = _anonenum1.define('VFIO_PCI_NUM_IRQS', 5)

class _anonenum2(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_CCW_CONFIG_REGION_INDEX = _anonenum2.define('VFIO_CCW_CONFIG_REGION_INDEX', 0)
VFIO_CCW_NUM_REGIONS = _anonenum2.define('VFIO_CCW_NUM_REGIONS', 1)

class _anonenum3(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_CCW_IO_IRQ_INDEX = _anonenum3.define('VFIO_CCW_IO_IRQ_INDEX', 0)
VFIO_CCW_CRW_IRQ_INDEX = _anonenum3.define('VFIO_CCW_CRW_IRQ_INDEX', 1)
VFIO_CCW_REQ_IRQ_INDEX = _anonenum3.define('VFIO_CCW_REQ_IRQ_INDEX', 2)
VFIO_CCW_NUM_IRQS = _anonenum3.define('VFIO_CCW_NUM_IRQS', 3)

class _anonenum4(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_AP_REQ_IRQ_INDEX = _anonenum4.define('VFIO_AP_REQ_IRQ_INDEX', 0)
VFIO_AP_NUM_IRQS = _anonenum4.define('VFIO_AP_NUM_IRQS', 1)

@c.record
class struct_vfio_pci_dependent_device(c.Struct):
  SIZE = 8
  group_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  devid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  segment: Annotated[Annotated[int, ctypes.c_uint16], 4]
  bus: Annotated[Annotated[int, ctypes.c_ubyte], 6]
  devfn: Annotated[Annotated[int, ctypes.c_ubyte], 7]
@c.record
class struct_vfio_pci_hot_reset_info(c.Struct):
  SIZE = 12
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  count: Annotated[Annotated[int, ctypes.c_uint32], 8]
  devices: Annotated[c.Array[struct_vfio_pci_dependent_device, Literal[0]], 12]
@c.record
class struct_vfio_pci_hot_reset(c.Struct):
  SIZE = 12
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  count: Annotated[Annotated[int, ctypes.c_uint32], 8]
  group_fds: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[0]], 12]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class struct_vfio_device_gfx_plane_info(c.Struct):
  SIZE = 64
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  drm_plane_type: Annotated[Annotated[int, ctypes.c_uint32], 8]
  drm_format: Annotated[Annotated[int, ctypes.c_uint32], 12]
  drm_format_mod: Annotated[Annotated[int, ctypes.c_uint64], 16]
  width: Annotated[Annotated[int, ctypes.c_uint32], 24]
  height: Annotated[Annotated[int, ctypes.c_uint32], 28]
  stride: Annotated[Annotated[int, ctypes.c_uint32], 32]
  size: Annotated[Annotated[int, ctypes.c_uint32], 36]
  x_pos: Annotated[Annotated[int, ctypes.c_uint32], 40]
  y_pos: Annotated[Annotated[int, ctypes.c_uint32], 44]
  x_hot: Annotated[Annotated[int, ctypes.c_uint32], 48]
  y_hot: Annotated[Annotated[int, ctypes.c_uint32], 52]
  region_index: Annotated[Annotated[int, ctypes.c_uint32], 56]
  dmabuf_id: Annotated[Annotated[int, ctypes.c_uint32], 56]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 60]
@c.record
class struct_vfio_device_ioeventfd(c.Struct):
  SIZE = 32
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  data: Annotated[Annotated[int, ctypes.c_uint64], 16]
  fd: Annotated[Annotated[int, ctypes.c_int32], 24]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_vfio_device_feature(c.Struct):
  SIZE = 8
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 8]
@c.record
class struct_vfio_device_bind_iommufd(c.Struct):
  SIZE = 16
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  iommufd: Annotated[Annotated[int, ctypes.c_int32], 8]
  out_devid: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_device_attach_iommufd_pt(c.Struct):
  SIZE = 12
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  pt_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_vfio_device_detach_iommufd_pt(c.Struct):
  SIZE = 8
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_vfio_device_feature_migration(c.Struct):
  SIZE = 8
  flags: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_vfio_device_feature_mig_state(c.Struct):
  SIZE = 8
  device_state: Annotated[Annotated[int, ctypes.c_uint32], 0]
  data_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
class enum_vfio_device_mig_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
VFIO_DEVICE_STATE_ERROR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_ERROR', 0)
VFIO_DEVICE_STATE_STOP = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP', 1)
VFIO_DEVICE_STATE_RUNNING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING', 2)
VFIO_DEVICE_STATE_STOP_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP_COPY', 3)
VFIO_DEVICE_STATE_RESUMING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RESUMING', 4)
VFIO_DEVICE_STATE_RUNNING_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING_P2P', 5)
VFIO_DEVICE_STATE_PRE_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY', 6)
VFIO_DEVICE_STATE_PRE_COPY_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY_P2P', 7)
VFIO_DEVICE_STATE_NR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_NR', 8)

@c.record
class struct_vfio_precopy_info(c.Struct):
  SIZE = 24
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  initial_bytes: Annotated[Annotated[int, ctypes.c_uint64], 8]
  dirty_bytes: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_vfio_device_low_power_entry_with_wakeup(c.Struct):
  SIZE = 8
  wakeup_eventfd: Annotated[Annotated[int, ctypes.c_int32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_vfio_device_feature_dma_logging_control(c.Struct):
  SIZE = 24
  page_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_ranges: Annotated[Annotated[int, ctypes.c_uint32], 8]
  __reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  ranges: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_vfio_device_feature_dma_logging_range(c.Struct):
  SIZE = 16
  iova: Annotated[Annotated[int, ctypes.c_uint64], 0]
  length: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_vfio_device_feature_dma_logging_report(c.Struct):
  SIZE = 32
  iova: Annotated[Annotated[int, ctypes.c_uint64], 0]
  length: Annotated[Annotated[int, ctypes.c_uint64], 8]
  page_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  bitmap: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_vfio_device_feature_mig_data_size(c.Struct):
  SIZE = 8
  stop_copy_length: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_vfio_device_feature_bus_master(c.Struct):
  SIZE = 4
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_vfio_iommu_type1_info(c.Struct):
  SIZE = 24
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  iova_pgsizes: Annotated[Annotated[int, ctypes.c_uint64], 8]
  cap_offset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_vfio_iova_range(c.Struct):
  SIZE = 16
  start: Annotated[Annotated[int, ctypes.c_uint64], 0]
  end: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_vfio_iommu_type1_info_cap_iova_range(c.Struct):
  SIZE = 16
  header: Annotated[struct_vfio_info_cap_header, 0]
  nr_iovas: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  iova_ranges: Annotated[c.Array[struct_vfio_iova_range, Literal[0]], 16]
@c.record
class struct_vfio_iommu_type1_info_cap_migration(c.Struct):
  SIZE = 32
  header: Annotated[struct_vfio_info_cap_header, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pgsize_bitmap: Annotated[Annotated[int, ctypes.c_uint64], 16]
  max_dirty_bitmap_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_vfio_iommu_type1_info_dma_avail(c.Struct):
  SIZE = 12
  header: Annotated[struct_vfio_info_cap_header, 0]
  avail: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_vfio_iommu_type1_dma_map(c.Struct):
  SIZE = 32
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  vaddr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  iova: Annotated[Annotated[int, ctypes.c_uint64], 16]
  size: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_vfio_bitmap(c.Struct):
  SIZE = 24
  pgsize: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  data: Annotated[c.POINTER[Annotated[int, ctypes.c_uint64]], 16]
@c.record
class struct_vfio_iommu_type1_dma_unmap(c.Struct):
  SIZE = 24
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  iova: Annotated[Annotated[int, ctypes.c_uint64], 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 24]
@c.record
class struct_vfio_iommu_type1_dirty_bitmap(c.Struct):
  SIZE = 8
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 8]
@c.record
class struct_vfio_iommu_type1_dirty_bitmap_get(c.Struct):
  SIZE = 40
  iova: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  bitmap: Annotated[struct_vfio_bitmap, 16]
@c.record
class struct_vfio_iommu_spapr_tce_ddw_info(c.Struct):
  SIZE = 16
  pgsizes: Annotated[Annotated[int, ctypes.c_uint64], 0]
  max_dynamic_windows_supported: Annotated[Annotated[int, ctypes.c_uint32], 8]
  levels: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_vfio_iommu_spapr_tce_info(c.Struct):
  SIZE = 32
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  dma32_window_start: Annotated[Annotated[int, ctypes.c_uint32], 8]
  dma32_window_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  ddw: Annotated[struct_vfio_iommu_spapr_tce_ddw_info, 16]
@c.record
class struct_vfio_eeh_pe_err(c.Struct):
  SIZE = 24
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  func: Annotated[Annotated[int, ctypes.c_uint32], 4]
  addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  mask: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_vfio_eeh_pe_op(c.Struct):
  SIZE = 40
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  op: Annotated[Annotated[int, ctypes.c_uint32], 8]
  err: Annotated[struct_vfio_eeh_pe_err, 16]
@c.record
class struct_vfio_iommu_spapr_register_memory(c.Struct):
  SIZE = 24
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  vaddr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_vfio_iommu_spapr_tce_create(c.Struct):
  SIZE = 40
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  page_shift: Annotated[Annotated[int, ctypes.c_uint32], 8]
  __resv1: Annotated[Annotated[int, ctypes.c_uint32], 12]
  window_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  levels: Annotated[Annotated[int, ctypes.c_uint32], 24]
  __resv2: Annotated[Annotated[int, ctypes.c_uint32], 28]
  start_addr: Annotated[Annotated[int, ctypes.c_uint64], 32]
@c.record
class struct_vfio_iommu_spapr_tce_remove(c.Struct):
  SIZE = 16
  argsz: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  start_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
c.init_records()
VFIO_API_VERSION = 0 # type: ignore
VFIO_TYPE1_IOMMU = 1 # type: ignore
VFIO_SPAPR_TCE_IOMMU = 2 # type: ignore
VFIO_TYPE1v2_IOMMU = 3 # type: ignore
VFIO_DMA_CC_IOMMU = 4 # type: ignore
VFIO_EEH = 5 # type: ignore
VFIO_TYPE1_NESTING_IOMMU = 6 # type: ignore
VFIO_SPAPR_TCE_v2_IOMMU = 7 # type: ignore
VFIO_NOIOMMU_IOMMU = 8 # type: ignore
VFIO_UNMAP_ALL = 9 # type: ignore
VFIO_UPDATE_VADDR = 10 # type: ignore
VFIO_TYPE = (';') # type: ignore
VFIO_BASE = 100 # type: ignore
VFIO_GET_API_VERSION = _IO(VFIO_TYPE, VFIO_BASE + 0) # type: ignore
VFIO_CHECK_EXTENSION = _IO(VFIO_TYPE, VFIO_BASE + 1) # type: ignore
VFIO_SET_IOMMU = _IO(VFIO_TYPE, VFIO_BASE + 2) # type: ignore
VFIO_GROUP_FLAGS_VIABLE = (1 << 0) # type: ignore
VFIO_GROUP_FLAGS_CONTAINER_SET = (1 << 1) # type: ignore
VFIO_GROUP_GET_STATUS = _IO(VFIO_TYPE, VFIO_BASE + 3) # type: ignore
VFIO_GROUP_SET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 4) # type: ignore
VFIO_GROUP_UNSET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 5) # type: ignore
VFIO_GROUP_GET_DEVICE_FD = _IO(VFIO_TYPE, VFIO_BASE + 6) # type: ignore
VFIO_DEVICE_FLAGS_RESET = (1 << 0) # type: ignore
VFIO_DEVICE_FLAGS_PCI = (1 << 1) # type: ignore
VFIO_DEVICE_FLAGS_PLATFORM = (1 << 2) # type: ignore
VFIO_DEVICE_FLAGS_AMBA = (1 << 3) # type: ignore
VFIO_DEVICE_FLAGS_CCW = (1 << 4) # type: ignore
VFIO_DEVICE_FLAGS_AP = (1 << 5) # type: ignore
VFIO_DEVICE_FLAGS_FSL_MC = (1 << 6) # type: ignore
VFIO_DEVICE_FLAGS_CAPS = (1 << 7) # type: ignore
VFIO_DEVICE_FLAGS_CDX = (1 << 8) # type: ignore
VFIO_DEVICE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 7) # type: ignore
VFIO_DEVICE_API_PCI_STRING = "vfio-pci" # type: ignore
VFIO_DEVICE_API_PLATFORM_STRING = "vfio-platform" # type: ignore
VFIO_DEVICE_API_AMBA_STRING = "vfio-amba" # type: ignore
VFIO_DEVICE_API_CCW_STRING = "vfio-ccw" # type: ignore
VFIO_DEVICE_API_AP_STRING = "vfio-ap" # type: ignore
VFIO_DEVICE_INFO_CAP_ZPCI_BASE = 1 # type: ignore
VFIO_DEVICE_INFO_CAP_ZPCI_GROUP = 2 # type: ignore
VFIO_DEVICE_INFO_CAP_ZPCI_UTIL = 3 # type: ignore
VFIO_DEVICE_INFO_CAP_ZPCI_PFIP = 4 # type: ignore
VFIO_DEVICE_INFO_CAP_PCI_ATOMIC_COMP = 5 # type: ignore
VFIO_PCI_ATOMIC_COMP32 = (1 << 0) # type: ignore
VFIO_PCI_ATOMIC_COMP64 = (1 << 1) # type: ignore
VFIO_PCI_ATOMIC_COMP128 = (1 << 2) # type: ignore
VFIO_REGION_INFO_FLAG_READ = (1 << 0) # type: ignore
VFIO_REGION_INFO_FLAG_WRITE = (1 << 1) # type: ignore
VFIO_REGION_INFO_FLAG_MMAP = (1 << 2) # type: ignore
VFIO_REGION_INFO_FLAG_CAPS = (1 << 3) # type: ignore
VFIO_DEVICE_GET_REGION_INFO = _IO(VFIO_TYPE, VFIO_BASE + 8) # type: ignore
VFIO_REGION_INFO_CAP_SPARSE_MMAP = 1 # type: ignore
VFIO_REGION_INFO_CAP_TYPE = 2 # type: ignore
VFIO_REGION_TYPE_PCI_VENDOR_TYPE = (1 << 31) # type: ignore
VFIO_REGION_TYPE_PCI_VENDOR_MASK = (0xffff) # type: ignore
VFIO_REGION_TYPE_GFX = (1) # type: ignore
VFIO_REGION_TYPE_CCW = (2) # type: ignore
VFIO_REGION_TYPE_MIGRATION_DEPRECATED = (3) # type: ignore
VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION = (1) # type: ignore
VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG = (2) # type: ignore
VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG = (3) # type: ignore
VFIO_REGION_SUBTYPE_NVIDIA_NVLINK2_RAM = (1) # type: ignore
VFIO_REGION_SUBTYPE_IBM_NVLINK2_ATSD = (1) # type: ignore
VFIO_REGION_SUBTYPE_GFX_EDID = (1) # type: ignore
VFIO_DEVICE_GFX_LINK_STATE_UP = 1 # type: ignore
VFIO_DEVICE_GFX_LINK_STATE_DOWN = 2 # type: ignore
VFIO_REGION_SUBTYPE_CCW_ASYNC_CMD = (1) # type: ignore
VFIO_REGION_SUBTYPE_CCW_SCHIB = (2) # type: ignore
VFIO_REGION_SUBTYPE_CCW_CRW = (3) # type: ignore
VFIO_REGION_SUBTYPE_MIGRATION_DEPRECATED = (1) # type: ignore
VFIO_DEVICE_STATE_V1_STOP = (0) # type: ignore
VFIO_DEVICE_STATE_V1_RUNNING = (1 << 0) # type: ignore
VFIO_DEVICE_STATE_V1_SAVING = (1 << 1) # type: ignore
VFIO_DEVICE_STATE_V1_RESUMING = (1 << 2) # type: ignore
VFIO_DEVICE_STATE_MASK = (VFIO_DEVICE_STATE_V1_RUNNING | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING) # type: ignore
VFIO_DEVICE_STATE_IS_ERROR = lambda state: ((state & VFIO_DEVICE_STATE_MASK) == (VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING)) # type: ignore
VFIO_DEVICE_STATE_SET_ERROR = lambda state: ((state & ~VFIO_DEVICE_STATE_MASK) | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING) # type: ignore
VFIO_REGION_INFO_CAP_MSIX_MAPPABLE = 3 # type: ignore
VFIO_REGION_INFO_CAP_NVLINK2_SSATGT = 4 # type: ignore
VFIO_REGION_INFO_CAP_NVLINK2_LNKSPD = 5 # type: ignore
VFIO_IRQ_INFO_EVENTFD = (1 << 0) # type: ignore
VFIO_IRQ_INFO_MASKABLE = (1 << 1) # type: ignore
VFIO_IRQ_INFO_AUTOMASKED = (1 << 2) # type: ignore
VFIO_IRQ_INFO_NORESIZE = (1 << 3) # type: ignore
VFIO_DEVICE_GET_IRQ_INFO = _IO(VFIO_TYPE, VFIO_BASE + 9) # type: ignore
VFIO_IRQ_SET_DATA_NONE = (1 << 0) # type: ignore
VFIO_IRQ_SET_DATA_BOOL = (1 << 1) # type: ignore
VFIO_IRQ_SET_DATA_EVENTFD = (1 << 2) # type: ignore
VFIO_IRQ_SET_ACTION_MASK = (1 << 3) # type: ignore
VFIO_IRQ_SET_ACTION_UNMASK = (1 << 4) # type: ignore
VFIO_IRQ_SET_ACTION_TRIGGER = (1 << 5) # type: ignore
VFIO_DEVICE_SET_IRQS = _IO(VFIO_TYPE, VFIO_BASE + 10) # type: ignore
VFIO_IRQ_SET_DATA_TYPE_MASK = (VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_DATA_BOOL | VFIO_IRQ_SET_DATA_EVENTFD) # type: ignore
VFIO_IRQ_SET_ACTION_TYPE_MASK = (VFIO_IRQ_SET_ACTION_MASK | VFIO_IRQ_SET_ACTION_UNMASK | VFIO_IRQ_SET_ACTION_TRIGGER) # type: ignore
VFIO_DEVICE_RESET = _IO(VFIO_TYPE, VFIO_BASE + 11) # type: ignore
VFIO_PCI_DEVID_OWNED = 0 # type: ignore
VFIO_PCI_DEVID_NOT_OWNED = -1 # type: ignore
VFIO_PCI_HOT_RESET_FLAG_DEV_ID = (1 << 0) # type: ignore
VFIO_PCI_HOT_RESET_FLAG_DEV_ID_OWNED = (1 << 1) # type: ignore
VFIO_DEVICE_GET_PCI_HOT_RESET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12) # type: ignore
VFIO_DEVICE_PCI_HOT_RESET = _IO(VFIO_TYPE, VFIO_BASE + 13) # type: ignore
VFIO_GFX_PLANE_TYPE_PROBE = (1 << 0) # type: ignore
VFIO_GFX_PLANE_TYPE_DMABUF = (1 << 1) # type: ignore
VFIO_GFX_PLANE_TYPE_REGION = (1 << 2) # type: ignore
VFIO_DEVICE_QUERY_GFX_PLANE = _IO(VFIO_TYPE, VFIO_BASE + 14) # type: ignore
VFIO_DEVICE_GET_GFX_DMABUF = _IO(VFIO_TYPE, VFIO_BASE + 15) # type: ignore
VFIO_DEVICE_IOEVENTFD_8 = (1 << 0) # type: ignore
VFIO_DEVICE_IOEVENTFD_16 = (1 << 1) # type: ignore
VFIO_DEVICE_IOEVENTFD_32 = (1 << 2) # type: ignore
VFIO_DEVICE_IOEVENTFD_64 = (1 << 3) # type: ignore
VFIO_DEVICE_IOEVENTFD_SIZE_MASK = (0xf) # type: ignore
VFIO_DEVICE_IOEVENTFD = _IO(VFIO_TYPE, VFIO_BASE + 16) # type: ignore
VFIO_DEVICE_FEATURE_MASK = (0xffff) # type: ignore
VFIO_DEVICE_FEATURE_GET = (1 << 16) # type: ignore
VFIO_DEVICE_FEATURE_SET = (1 << 17) # type: ignore
VFIO_DEVICE_FEATURE_PROBE = (1 << 18) # type: ignore
VFIO_DEVICE_FEATURE = _IO(VFIO_TYPE, VFIO_BASE + 17) # type: ignore
VFIO_DEVICE_BIND_IOMMUFD = _IO(VFIO_TYPE, VFIO_BASE + 18) # type: ignore
VFIO_DEVICE_ATTACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 19) # type: ignore
VFIO_DEVICE_DETACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 20) # type: ignore
VFIO_DEVICE_FEATURE_PCI_VF_TOKEN = (0) # type: ignore
VFIO_MIGRATION_STOP_COPY = (1 << 0) # type: ignore
VFIO_MIGRATION_P2P = (1 << 1) # type: ignore
VFIO_MIGRATION_PRE_COPY = (1 << 2) # type: ignore
VFIO_DEVICE_FEATURE_MIGRATION = 1 # type: ignore
VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE = 2 # type: ignore
VFIO_MIG_GET_PRECOPY_INFO = _IO(VFIO_TYPE, VFIO_BASE + 21) # type: ignore
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY = 3 # type: ignore
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY_WITH_WAKEUP = 4 # type: ignore
VFIO_DEVICE_FEATURE_LOW_POWER_EXIT = 5 # type: ignore
VFIO_DEVICE_FEATURE_DMA_LOGGING_START = 6 # type: ignore
VFIO_DEVICE_FEATURE_DMA_LOGGING_STOP = 7 # type: ignore
VFIO_DEVICE_FEATURE_DMA_LOGGING_REPORT = 8 # type: ignore
VFIO_DEVICE_FEATURE_MIG_DATA_SIZE = 9 # type: ignore
VFIO_DEVICE_FEATURE_CLEAR_MASTER = 0 # type: ignore
VFIO_DEVICE_FEATURE_SET_MASTER = 1 # type: ignore
VFIO_DEVICE_FEATURE_BUS_MASTER = 10 # type: ignore
VFIO_IOMMU_INFO_PGSIZES = (1 << 0) # type: ignore
VFIO_IOMMU_INFO_CAPS = (1 << 1) # type: ignore
VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE = 1 # type: ignore
VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION = 2 # type: ignore
VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL = 3 # type: ignore
VFIO_IOMMU_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12) # type: ignore
VFIO_DMA_MAP_FLAG_READ = (1 << 0) # type: ignore
VFIO_DMA_MAP_FLAG_WRITE = (1 << 1) # type: ignore
VFIO_DMA_MAP_FLAG_VADDR = (1 << 2) # type: ignore
VFIO_IOMMU_MAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 13) # type: ignore
VFIO_DMA_UNMAP_FLAG_GET_DIRTY_BITMAP = (1 << 0) # type: ignore
VFIO_DMA_UNMAP_FLAG_ALL = (1 << 1) # type: ignore
VFIO_DMA_UNMAP_FLAG_VADDR = (1 << 2) # type: ignore
VFIO_IOMMU_UNMAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 14) # type: ignore
VFIO_IOMMU_ENABLE = _IO(VFIO_TYPE, VFIO_BASE + 15) # type: ignore
VFIO_IOMMU_DISABLE = _IO(VFIO_TYPE, VFIO_BASE + 16) # type: ignore
VFIO_IOMMU_DIRTY_PAGES_FLAG_START = (1 << 0) # type: ignore
VFIO_IOMMU_DIRTY_PAGES_FLAG_STOP = (1 << 1) # type: ignore
VFIO_IOMMU_DIRTY_PAGES_FLAG_GET_BITMAP = (1 << 2) # type: ignore
VFIO_IOMMU_DIRTY_PAGES = _IO(VFIO_TYPE, VFIO_BASE + 17) # type: ignore
VFIO_IOMMU_SPAPR_INFO_DDW = (1 << 0) # type: ignore
VFIO_IOMMU_SPAPR_TCE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12) # type: ignore
VFIO_EEH_PE_DISABLE = 0 # type: ignore
VFIO_EEH_PE_ENABLE = 1 # type: ignore
VFIO_EEH_PE_UNFREEZE_IO = 2 # type: ignore
VFIO_EEH_PE_UNFREEZE_DMA = 3 # type: ignore
VFIO_EEH_PE_GET_STATE = 4 # type: ignore
VFIO_EEH_PE_STATE_NORMAL = 0 # type: ignore
VFIO_EEH_PE_STATE_RESET = 1 # type: ignore
VFIO_EEH_PE_STATE_STOPPED = 2 # type: ignore
VFIO_EEH_PE_STATE_STOPPED_DMA = 4 # type: ignore
VFIO_EEH_PE_STATE_UNAVAIL = 5 # type: ignore
VFIO_EEH_PE_RESET_DEACTIVATE = 5 # type: ignore
VFIO_EEH_PE_RESET_HOT = 6 # type: ignore
VFIO_EEH_PE_RESET_FUNDAMENTAL = 7 # type: ignore
VFIO_EEH_PE_CONFIGURE = 8 # type: ignore
VFIO_EEH_PE_INJECT_ERR = 9 # type: ignore
VFIO_EEH_PE_OP = _IO(VFIO_TYPE, VFIO_BASE + 21) # type: ignore
VFIO_IOMMU_SPAPR_REGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 17) # type: ignore
VFIO_IOMMU_SPAPR_UNREGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 18) # type: ignore
VFIO_IOMMU_SPAPR_TCE_CREATE = _IO(VFIO_TYPE, VFIO_BASE + 19) # type: ignore
VFIO_IOMMU_SPAPR_TCE_REMOVE = _IO(VFIO_TYPE, VFIO_BASE + 20) # type: ignore