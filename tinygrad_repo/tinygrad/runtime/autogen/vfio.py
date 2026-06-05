# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_vfio_info_cap_header(c.Struct):
  SIZE = 8
  id: int
  version: int
  next: int
__u16: TypeAlias = ctypes.c_uint16
__u32: TypeAlias = ctypes.c_uint32
struct_vfio_info_cap_header.register_fields([('id', ctypes.c_uint16, 0), ('version', ctypes.c_uint16, 2), ('next', ctypes.c_uint32, 4)])
@c.record
class struct_vfio_group_status(c.Struct):
  SIZE = 8
  argsz: int
  flags: int
struct_vfio_group_status.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class struct_vfio_device_info(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  num_regions: int
  num_irqs: int
  cap_offset: int
  pad: int
struct_vfio_device_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('num_regions', ctypes.c_uint32, 8), ('num_irqs', ctypes.c_uint32, 12), ('cap_offset', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_vfio_device_info_cap_pci_atomic_comp(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  flags: int
  reserved: int
struct_vfio_device_info_cap_pci_atomic_comp.register_fields([('header', struct_vfio_info_cap_header, 0), ('flags', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_region_info(c.Struct):
  SIZE = 32
  argsz: int
  flags: int
  index: int
  cap_offset: int
  size: int
  offset: int
__u64: TypeAlias = ctypes.c_uint64
struct_vfio_region_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('index', ctypes.c_uint32, 8), ('cap_offset', ctypes.c_uint32, 12), ('size', ctypes.c_uint64, 16), ('offset', ctypes.c_uint64, 24)])
@c.record
class struct_vfio_region_sparse_mmap_area(c.Struct):
  SIZE = 16
  offset: int
  size: int
struct_vfio_region_sparse_mmap_area.register_fields([('offset', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8)])
@c.record
class struct_vfio_region_info_cap_sparse_mmap(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  nr_areas: int
  reserved: int
  areas: c.Array[struct_vfio_region_sparse_mmap_area, Literal[0]]
struct_vfio_region_info_cap_sparse_mmap.register_fields([('header', struct_vfio_info_cap_header, 0), ('nr_areas', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12), ('areas', c.Array[struct_vfio_region_sparse_mmap_area, Literal[0]], 16)])
@c.record
class struct_vfio_region_info_cap_type(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  type: int
  subtype: int
struct_vfio_region_info_cap_type.register_fields([('header', struct_vfio_info_cap_header, 0), ('type', ctypes.c_uint32, 8), ('subtype', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_region_gfx_edid(c.Struct):
  SIZE = 24
  edid_offset: int
  edid_max_size: int
  edid_size: int
  max_xres: int
  max_yres: int
  link_state: int
struct_vfio_region_gfx_edid.register_fields([('edid_offset', ctypes.c_uint32, 0), ('edid_max_size', ctypes.c_uint32, 4), ('edid_size', ctypes.c_uint32, 8), ('max_xres', ctypes.c_uint32, 12), ('max_yres', ctypes.c_uint32, 16), ('link_state', ctypes.c_uint32, 20)])
@c.record
class struct_vfio_device_migration_info(c.Struct):
  SIZE = 32
  device_state: int
  reserved: int
  pending_bytes: int
  data_offset: int
  data_size: int
struct_vfio_device_migration_info.register_fields([('device_state', ctypes.c_uint32, 0), ('reserved', ctypes.c_uint32, 4), ('pending_bytes', ctypes.c_uint64, 8), ('data_offset', ctypes.c_uint64, 16), ('data_size', ctypes.c_uint64, 24)])
@c.record
class struct_vfio_region_info_cap_nvlink2_ssatgt(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  tgt: int
struct_vfio_region_info_cap_nvlink2_ssatgt.register_fields([('header', struct_vfio_info_cap_header, 0), ('tgt', ctypes.c_uint64, 8)])
@c.record
class struct_vfio_region_info_cap_nvlink2_lnkspd(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  link_speed: int
  __pad: int
struct_vfio_region_info_cap_nvlink2_lnkspd.register_fields([('header', struct_vfio_info_cap_header, 0), ('link_speed', ctypes.c_uint32, 8), ('__pad', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_irq_info(c.Struct):
  SIZE = 16
  argsz: int
  flags: int
  index: int
  count: int
struct_vfio_irq_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('index', ctypes.c_uint32, 8), ('count', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_irq_set(c.Struct):
  SIZE = 20
  argsz: int
  flags: int
  index: int
  start: int
  count: int
  data: c.Array[ctypes.c_ubyte, Literal[0]]
__u8: TypeAlias = ctypes.c_ubyte
struct_vfio_irq_set.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('index', ctypes.c_uint32, 8), ('start', ctypes.c_uint32, 12), ('count', ctypes.c_uint32, 16), ('data', c.Array[ctypes.c_ubyte, Literal[0]], 20)])
_anonenum0: dict[int, str] = {(VFIO_PCI_BAR0_REGION_INDEX:=0): 'VFIO_PCI_BAR0_REGION_INDEX', (VFIO_PCI_BAR1_REGION_INDEX:=1): 'VFIO_PCI_BAR1_REGION_INDEX', (VFIO_PCI_BAR2_REGION_INDEX:=2): 'VFIO_PCI_BAR2_REGION_INDEX', (VFIO_PCI_BAR3_REGION_INDEX:=3): 'VFIO_PCI_BAR3_REGION_INDEX', (VFIO_PCI_BAR4_REGION_INDEX:=4): 'VFIO_PCI_BAR4_REGION_INDEX', (VFIO_PCI_BAR5_REGION_INDEX:=5): 'VFIO_PCI_BAR5_REGION_INDEX', (VFIO_PCI_ROM_REGION_INDEX:=6): 'VFIO_PCI_ROM_REGION_INDEX', (VFIO_PCI_CONFIG_REGION_INDEX:=7): 'VFIO_PCI_CONFIG_REGION_INDEX', (VFIO_PCI_VGA_REGION_INDEX:=8): 'VFIO_PCI_VGA_REGION_INDEX', (VFIO_PCI_NUM_REGIONS:=9): 'VFIO_PCI_NUM_REGIONS'}
_anonenum1: dict[int, str] = {(VFIO_PCI_INTX_IRQ_INDEX:=0): 'VFIO_PCI_INTX_IRQ_INDEX', (VFIO_PCI_MSI_IRQ_INDEX:=1): 'VFIO_PCI_MSI_IRQ_INDEX', (VFIO_PCI_MSIX_IRQ_INDEX:=2): 'VFIO_PCI_MSIX_IRQ_INDEX', (VFIO_PCI_ERR_IRQ_INDEX:=3): 'VFIO_PCI_ERR_IRQ_INDEX', (VFIO_PCI_REQ_IRQ_INDEX:=4): 'VFIO_PCI_REQ_IRQ_INDEX', (VFIO_PCI_NUM_IRQS:=5): 'VFIO_PCI_NUM_IRQS'}
_anonenum2: dict[int, str] = {(VFIO_CCW_CONFIG_REGION_INDEX:=0): 'VFIO_CCW_CONFIG_REGION_INDEX', (VFIO_CCW_NUM_REGIONS:=1): 'VFIO_CCW_NUM_REGIONS'}
_anonenum3: dict[int, str] = {(VFIO_CCW_IO_IRQ_INDEX:=0): 'VFIO_CCW_IO_IRQ_INDEX', (VFIO_CCW_CRW_IRQ_INDEX:=1): 'VFIO_CCW_CRW_IRQ_INDEX', (VFIO_CCW_REQ_IRQ_INDEX:=2): 'VFIO_CCW_REQ_IRQ_INDEX', (VFIO_CCW_NUM_IRQS:=3): 'VFIO_CCW_NUM_IRQS'}
_anonenum4: dict[int, str] = {(VFIO_AP_REQ_IRQ_INDEX:=0): 'VFIO_AP_REQ_IRQ_INDEX', (VFIO_AP_CFG_CHG_IRQ_INDEX:=1): 'VFIO_AP_CFG_CHG_IRQ_INDEX', (VFIO_AP_NUM_IRQS:=2): 'VFIO_AP_NUM_IRQS'}
@c.record
class struct_vfio_pci_dependent_device(c.Struct):
  SIZE = 8
  group_id: int
  devid: int
  segment: int
  bus: int
  devfn: int
struct_vfio_pci_dependent_device.register_fields([('group_id', ctypes.c_uint32, 0), ('devid', ctypes.c_uint32, 0), ('segment', ctypes.c_uint16, 4), ('bus', ctypes.c_ubyte, 6), ('devfn', ctypes.c_ubyte, 7)])
@c.record
class struct_vfio_pci_hot_reset_info(c.Struct):
  SIZE = 12
  argsz: int
  flags: int
  count: int
  devices: c.Array[struct_vfio_pci_dependent_device, Literal[0]]
struct_vfio_pci_hot_reset_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('count', ctypes.c_uint32, 8), ('devices', c.Array[struct_vfio_pci_dependent_device, Literal[0]], 12)])
@c.record
class struct_vfio_pci_hot_reset(c.Struct):
  SIZE = 12
  argsz: int
  flags: int
  count: int
  group_fds: c.Array[ctypes.c_int32, Literal[0]]
__s32: TypeAlias = ctypes.c_int32
struct_vfio_pci_hot_reset.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('count', ctypes.c_uint32, 8), ('group_fds', c.Array[ctypes.c_int32, Literal[0]], 12)])
@c.record
class struct_vfio_device_gfx_plane_info(c.Struct):
  SIZE = 64
  argsz: int
  flags: int
  drm_plane_type: int
  drm_format: int
  drm_format_mod: int
  width: int
  height: int
  stride: int
  size: int
  x_pos: int
  y_pos: int
  x_hot: int
  y_hot: int
  region_index: int
  dmabuf_id: int
  reserved: int
struct_vfio_device_gfx_plane_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('drm_plane_type', ctypes.c_uint32, 8), ('drm_format', ctypes.c_uint32, 12), ('drm_format_mod', ctypes.c_uint64, 16), ('width', ctypes.c_uint32, 24), ('height', ctypes.c_uint32, 28), ('stride', ctypes.c_uint32, 32), ('size', ctypes.c_uint32, 36), ('x_pos', ctypes.c_uint32, 40), ('y_pos', ctypes.c_uint32, 44), ('x_hot', ctypes.c_uint32, 48), ('y_hot', ctypes.c_uint32, 52), ('region_index', ctypes.c_uint32, 56), ('dmabuf_id', ctypes.c_uint32, 56), ('reserved', ctypes.c_uint32, 60)])
@c.record
class struct_vfio_device_ioeventfd(c.Struct):
  SIZE = 32
  argsz: int
  flags: int
  offset: int
  data: int
  fd: int
  reserved: int
struct_vfio_device_ioeventfd.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('offset', ctypes.c_uint64, 8), ('data', ctypes.c_uint64, 16), ('fd', ctypes.c_int32, 24), ('reserved', ctypes.c_uint32, 28)])
@c.record
class struct_vfio_device_feature(c.Struct):
  SIZE = 8
  argsz: int
  flags: int
  data: c.Array[ctypes.c_ubyte, Literal[0]]
struct_vfio_device_feature.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('data', c.Array[ctypes.c_ubyte, Literal[0]], 8)])
@c.record
class struct_vfio_device_bind_iommufd(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  iommufd: int
  out_devid: int
  token_uuid_ptr: int
struct_vfio_device_bind_iommufd.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('iommufd', ctypes.c_int32, 8), ('out_devid', ctypes.c_uint32, 12), ('token_uuid_ptr', ctypes.c_uint64, 16)])
@c.record
class struct_vfio_device_attach_iommufd_pt(c.Struct):
  SIZE = 16
  argsz: int
  flags: int
  pt_id: int
  pasid: int
struct_vfio_device_attach_iommufd_pt.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('pt_id', ctypes.c_uint32, 8), ('pasid', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_device_detach_iommufd_pt(c.Struct):
  SIZE = 12
  argsz: int
  flags: int
  pasid: int
struct_vfio_device_detach_iommufd_pt.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('pasid', ctypes.c_uint32, 8)])
@c.record
class struct_vfio_device_feature_migration(c.Struct):
  SIZE = 8
  flags: int
struct_vfio_device_feature_migration.register_fields([('flags', ctypes.c_uint64, 0)])
@c.record
class struct_vfio_device_feature_mig_state(c.Struct):
  SIZE = 8
  device_state: int
  data_fd: int
struct_vfio_device_feature_mig_state.register_fields([('device_state', ctypes.c_uint32, 0), ('data_fd', ctypes.c_int32, 4)])
enum_vfio_device_mig_state: dict[int, str] = {(VFIO_DEVICE_STATE_ERROR:=0): 'VFIO_DEVICE_STATE_ERROR', (VFIO_DEVICE_STATE_STOP:=1): 'VFIO_DEVICE_STATE_STOP', (VFIO_DEVICE_STATE_RUNNING:=2): 'VFIO_DEVICE_STATE_RUNNING', (VFIO_DEVICE_STATE_STOP_COPY:=3): 'VFIO_DEVICE_STATE_STOP_COPY', (VFIO_DEVICE_STATE_RESUMING:=4): 'VFIO_DEVICE_STATE_RESUMING', (VFIO_DEVICE_STATE_RUNNING_P2P:=5): 'VFIO_DEVICE_STATE_RUNNING_P2P', (VFIO_DEVICE_STATE_PRE_COPY:=6): 'VFIO_DEVICE_STATE_PRE_COPY', (VFIO_DEVICE_STATE_PRE_COPY_P2P:=7): 'VFIO_DEVICE_STATE_PRE_COPY_P2P', (VFIO_DEVICE_STATE_NR:=8): 'VFIO_DEVICE_STATE_NR'}
@c.record
class struct_vfio_precopy_info(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  initial_bytes: int
  dirty_bytes: int
struct_vfio_precopy_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('initial_bytes', ctypes.c_uint64, 8), ('dirty_bytes', ctypes.c_uint64, 16)])
@c.record
class struct_vfio_device_low_power_entry_with_wakeup(c.Struct):
  SIZE = 8
  wakeup_eventfd: int
  reserved: int
struct_vfio_device_low_power_entry_with_wakeup.register_fields([('wakeup_eventfd', ctypes.c_int32, 0), ('reserved', ctypes.c_uint32, 4)])
@c.record
class struct_vfio_device_feature_dma_logging_control(c.Struct):
  SIZE = 24
  page_size: int
  num_ranges: int
  __reserved: int
  ranges: int
struct_vfio_device_feature_dma_logging_control.register_fields([('page_size', ctypes.c_uint64, 0), ('num_ranges', ctypes.c_uint32, 8), ('__reserved', ctypes.c_uint32, 12), ('ranges', ctypes.c_uint64, 16)])
@c.record
class struct_vfio_device_feature_dma_logging_range(c.Struct):
  SIZE = 16
  iova: int
  length: int
struct_vfio_device_feature_dma_logging_range.register_fields([('iova', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8)])
@c.record
class struct_vfio_device_feature_dma_logging_report(c.Struct):
  SIZE = 32
  iova: int
  length: int
  page_size: int
  bitmap: int
struct_vfio_device_feature_dma_logging_report.register_fields([('iova', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8), ('page_size', ctypes.c_uint64, 16), ('bitmap', ctypes.c_uint64, 24)])
@c.record
class struct_vfio_device_feature_mig_data_size(c.Struct):
  SIZE = 8
  stop_copy_length: int
struct_vfio_device_feature_mig_data_size.register_fields([('stop_copy_length', ctypes.c_uint64, 0)])
@c.record
class struct_vfio_device_feature_bus_master(c.Struct):
  SIZE = 4
  op: int
struct_vfio_device_feature_bus_master.register_fields([('op', ctypes.c_uint32, 0)])
@c.record
class struct_vfio_iommu_type1_info(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  iova_pgsizes: int
  cap_offset: int
  pad: int
struct_vfio_iommu_type1_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('iova_pgsizes', ctypes.c_uint64, 8), ('cap_offset', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_vfio_iova_range(c.Struct):
  SIZE = 16
  start: int
  end: int
struct_vfio_iova_range.register_fields([('start', ctypes.c_uint64, 0), ('end', ctypes.c_uint64, 8)])
@c.record
class struct_vfio_iommu_type1_info_cap_iova_range(c.Struct):
  SIZE = 16
  header: struct_vfio_info_cap_header
  nr_iovas: int
  reserved: int
  iova_ranges: c.Array[struct_vfio_iova_range, Literal[0]]
struct_vfio_iommu_type1_info_cap_iova_range.register_fields([('header', struct_vfio_info_cap_header, 0), ('nr_iovas', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12), ('iova_ranges', c.Array[struct_vfio_iova_range, Literal[0]], 16)])
@c.record
class struct_vfio_iommu_type1_info_cap_migration(c.Struct):
  SIZE = 32
  header: struct_vfio_info_cap_header
  flags: int
  pgsize_bitmap: int
  max_dirty_bitmap_size: int
struct_vfio_iommu_type1_info_cap_migration.register_fields([('header', struct_vfio_info_cap_header, 0), ('flags', ctypes.c_uint32, 8), ('pgsize_bitmap', ctypes.c_uint64, 16), ('max_dirty_bitmap_size', ctypes.c_uint64, 24)])
@c.record
class struct_vfio_iommu_type1_info_dma_avail(c.Struct):
  SIZE = 12
  header: struct_vfio_info_cap_header
  avail: int
struct_vfio_iommu_type1_info_dma_avail.register_fields([('header', struct_vfio_info_cap_header, 0), ('avail', ctypes.c_uint32, 8)])
@c.record
class struct_vfio_iommu_type1_dma_map(c.Struct):
  SIZE = 32
  argsz: int
  flags: int
  vaddr: int
  iova: int
  size: int
struct_vfio_iommu_type1_dma_map.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('vaddr', ctypes.c_uint64, 8), ('iova', ctypes.c_uint64, 16), ('size', ctypes.c_uint64, 24)])
@c.record
class struct_vfio_bitmap(c.Struct):
  SIZE = 24
  pgsize: int
  size: int
  data: c.POINTER[ctypes.c_uint64]
struct_vfio_bitmap.register_fields([('pgsize', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('data', c.POINTER[ctypes.c_uint64], 16)])
@c.record
class struct_vfio_iommu_type1_dma_unmap(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  iova: int
  size: int
  data: c.Array[ctypes.c_ubyte, Literal[0]]
struct_vfio_iommu_type1_dma_unmap.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('iova', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16), ('data', c.Array[ctypes.c_ubyte, Literal[0]], 24)])
@c.record
class struct_vfio_iommu_type1_dirty_bitmap(c.Struct):
  SIZE = 8
  argsz: int
  flags: int
  data: c.Array[ctypes.c_ubyte, Literal[0]]
struct_vfio_iommu_type1_dirty_bitmap.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('data', c.Array[ctypes.c_ubyte, Literal[0]], 8)])
@c.record
class struct_vfio_iommu_type1_dirty_bitmap_get(c.Struct):
  SIZE = 40
  iova: int
  size: int
  bitmap: struct_vfio_bitmap
struct_vfio_iommu_type1_dirty_bitmap_get.register_fields([('iova', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('bitmap', struct_vfio_bitmap, 16)])
@c.record
class struct_vfio_iommu_spapr_tce_ddw_info(c.Struct):
  SIZE = 16
  pgsizes: int
  max_dynamic_windows_supported: int
  levels: int
struct_vfio_iommu_spapr_tce_ddw_info.register_fields([('pgsizes', ctypes.c_uint64, 0), ('max_dynamic_windows_supported', ctypes.c_uint32, 8), ('levels', ctypes.c_uint32, 12)])
@c.record
class struct_vfio_iommu_spapr_tce_info(c.Struct):
  SIZE = 32
  argsz: int
  flags: int
  dma32_window_start: int
  dma32_window_size: int
  ddw: struct_vfio_iommu_spapr_tce_ddw_info
struct_vfio_iommu_spapr_tce_info.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('dma32_window_start', ctypes.c_uint32, 8), ('dma32_window_size', ctypes.c_uint32, 12), ('ddw', struct_vfio_iommu_spapr_tce_ddw_info, 16)])
@c.record
class struct_vfio_eeh_pe_err(c.Struct):
  SIZE = 24
  type: int
  func: int
  addr: int
  mask: int
struct_vfio_eeh_pe_err.register_fields([('type', ctypes.c_uint32, 0), ('func', ctypes.c_uint32, 4), ('addr', ctypes.c_uint64, 8), ('mask', ctypes.c_uint64, 16)])
@c.record
class struct_vfio_eeh_pe_op(c.Struct):
  SIZE = 40
  argsz: int
  flags: int
  op: int
  err: struct_vfio_eeh_pe_err
struct_vfio_eeh_pe_op.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('op', ctypes.c_uint32, 8), ('err', struct_vfio_eeh_pe_err, 16)])
@c.record
class struct_vfio_iommu_spapr_register_memory(c.Struct):
  SIZE = 24
  argsz: int
  flags: int
  vaddr: int
  size: int
struct_vfio_iommu_spapr_register_memory.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('vaddr', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16)])
@c.record
class struct_vfio_iommu_spapr_tce_create(c.Struct):
  SIZE = 40
  argsz: int
  flags: int
  page_shift: int
  __resv1: int
  window_size: int
  levels: int
  __resv2: int
  start_addr: int
struct_vfio_iommu_spapr_tce_create.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('page_shift', ctypes.c_uint32, 8), ('__resv1', ctypes.c_uint32, 12), ('window_size', ctypes.c_uint64, 16), ('levels', ctypes.c_uint32, 24), ('__resv2', ctypes.c_uint32, 28), ('start_addr', ctypes.c_uint64, 32)])
@c.record
class struct_vfio_iommu_spapr_tce_remove(c.Struct):
  SIZE = 16
  argsz: int
  flags: int
  start_addr: int
struct_vfio_iommu_spapr_tce_remove.register_fields([('argsz', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('start_addr', ctypes.c_uint64, 8)])
VFIO_API_VERSION = 0
VFIO_TYPE1_IOMMU = 1
VFIO_SPAPR_TCE_IOMMU = 2
VFIO_TYPE1v2_IOMMU = 3
VFIO_DMA_CC_IOMMU = 4
VFIO_EEH = 5
__VFIO_RESERVED_TYPE1_NESTING_IOMMU = 6
VFIO_SPAPR_TCE_v2_IOMMU = 7
VFIO_NOIOMMU_IOMMU = 8
VFIO_UNMAP_ALL = 9
VFIO_UPDATE_VADDR = 10
VFIO_TYPE = (';')
VFIO_BASE = 100
VFIO_GET_API_VERSION = _IO(VFIO_TYPE, VFIO_BASE + 0)
VFIO_CHECK_EXTENSION = _IO(VFIO_TYPE, VFIO_BASE + 1)
VFIO_SET_IOMMU = _IO(VFIO_TYPE, VFIO_BASE + 2)
VFIO_GROUP_FLAGS_VIABLE = (1 << 0)
VFIO_GROUP_FLAGS_CONTAINER_SET = (1 << 1)
VFIO_GROUP_GET_STATUS = _IO(VFIO_TYPE, VFIO_BASE + 3)
VFIO_GROUP_SET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 4)
VFIO_GROUP_UNSET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 5)
VFIO_GROUP_GET_DEVICE_FD = _IO(VFIO_TYPE, VFIO_BASE + 6)
VFIO_DEVICE_FLAGS_RESET = (1 << 0)
VFIO_DEVICE_FLAGS_PCI = (1 << 1)
VFIO_DEVICE_FLAGS_PLATFORM = (1 << 2)
VFIO_DEVICE_FLAGS_AMBA = (1 << 3)
VFIO_DEVICE_FLAGS_CCW = (1 << 4)
VFIO_DEVICE_FLAGS_AP = (1 << 5)
VFIO_DEVICE_FLAGS_FSL_MC = (1 << 6)
VFIO_DEVICE_FLAGS_CAPS = (1 << 7)
VFIO_DEVICE_FLAGS_CDX = (1 << 8)
VFIO_DEVICE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 7)
VFIO_DEVICE_API_PCI_STRING = "vfio-pci"
VFIO_DEVICE_API_PLATFORM_STRING = "vfio-platform"
VFIO_DEVICE_API_AMBA_STRING = "vfio-amba"
VFIO_DEVICE_API_CCW_STRING = "vfio-ccw"
VFIO_DEVICE_API_AP_STRING = "vfio-ap"
VFIO_DEVICE_INFO_CAP_ZPCI_BASE = 1
VFIO_DEVICE_INFO_CAP_ZPCI_GROUP = 2
VFIO_DEVICE_INFO_CAP_ZPCI_UTIL = 3
VFIO_DEVICE_INFO_CAP_ZPCI_PFIP = 4
VFIO_DEVICE_INFO_CAP_PCI_ATOMIC_COMP = 5
VFIO_PCI_ATOMIC_COMP32 = (1 << 0)
VFIO_PCI_ATOMIC_COMP64 = (1 << 1)
VFIO_PCI_ATOMIC_COMP128 = (1 << 2)
VFIO_REGION_INFO_FLAG_READ = (1 << 0)
VFIO_REGION_INFO_FLAG_WRITE = (1 << 1)
VFIO_REGION_INFO_FLAG_MMAP = (1 << 2)
VFIO_REGION_INFO_FLAG_CAPS = (1 << 3)
VFIO_DEVICE_GET_REGION_INFO = _IO(VFIO_TYPE, VFIO_BASE + 8)
VFIO_REGION_INFO_CAP_SPARSE_MMAP = 1
VFIO_REGION_INFO_CAP_TYPE = 2
VFIO_REGION_TYPE_PCI_VENDOR_TYPE = (1 << 31)
VFIO_REGION_TYPE_PCI_VENDOR_MASK = (0xffff)
VFIO_REGION_TYPE_GFX = (1)
VFIO_REGION_TYPE_CCW = (2)
VFIO_REGION_TYPE_MIGRATION_DEPRECATED = (3)
VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION = (1)
VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG = (2)
VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG = (3)
VFIO_REGION_SUBTYPE_NVIDIA_NVLINK2_RAM = (1)
VFIO_REGION_SUBTYPE_IBM_NVLINK2_ATSD = (1)
VFIO_REGION_SUBTYPE_GFX_EDID = (1)
VFIO_DEVICE_GFX_LINK_STATE_UP = 1
VFIO_DEVICE_GFX_LINK_STATE_DOWN = 2
VFIO_REGION_SUBTYPE_CCW_ASYNC_CMD = (1)
VFIO_REGION_SUBTYPE_CCW_SCHIB = (2)
VFIO_REGION_SUBTYPE_CCW_CRW = (3)
VFIO_REGION_SUBTYPE_MIGRATION_DEPRECATED = (1)
VFIO_DEVICE_STATE_V1_STOP = (0)
VFIO_DEVICE_STATE_V1_RUNNING = (1 << 0)
VFIO_DEVICE_STATE_V1_SAVING = (1 << 1)
VFIO_DEVICE_STATE_V1_RESUMING = (1 << 2)
VFIO_DEVICE_STATE_MASK = (VFIO_DEVICE_STATE_V1_RUNNING | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING)
VFIO_DEVICE_STATE_IS_ERROR = lambda state: ((state & VFIO_DEVICE_STATE_MASK) == (VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING)) # type: ignore
VFIO_DEVICE_STATE_SET_ERROR = lambda state: ((state & ~VFIO_DEVICE_STATE_MASK) | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING) # type: ignore
VFIO_REGION_INFO_CAP_MSIX_MAPPABLE = 3
VFIO_REGION_INFO_CAP_NVLINK2_SSATGT = 4
VFIO_REGION_INFO_CAP_NVLINK2_LNKSPD = 5
VFIO_IRQ_INFO_EVENTFD = (1 << 0)
VFIO_IRQ_INFO_MASKABLE = (1 << 1)
VFIO_IRQ_INFO_AUTOMASKED = (1 << 2)
VFIO_IRQ_INFO_NORESIZE = (1 << 3)
VFIO_DEVICE_GET_IRQ_INFO = _IO(VFIO_TYPE, VFIO_BASE + 9)
VFIO_IRQ_SET_DATA_NONE = (1 << 0)
VFIO_IRQ_SET_DATA_BOOL = (1 << 1)
VFIO_IRQ_SET_DATA_EVENTFD = (1 << 2)
VFIO_IRQ_SET_ACTION_MASK = (1 << 3)
VFIO_IRQ_SET_ACTION_UNMASK = (1 << 4)
VFIO_IRQ_SET_ACTION_TRIGGER = (1 << 5)
VFIO_DEVICE_SET_IRQS = _IO(VFIO_TYPE, VFIO_BASE + 10)
VFIO_IRQ_SET_DATA_TYPE_MASK = (VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_DATA_BOOL | VFIO_IRQ_SET_DATA_EVENTFD)
VFIO_IRQ_SET_ACTION_TYPE_MASK = (VFIO_IRQ_SET_ACTION_MASK | VFIO_IRQ_SET_ACTION_UNMASK | VFIO_IRQ_SET_ACTION_TRIGGER)
VFIO_DEVICE_RESET = _IO(VFIO_TYPE, VFIO_BASE + 11)
VFIO_PCI_DEVID_OWNED = 0
VFIO_PCI_DEVID_NOT_OWNED = -1
VFIO_PCI_HOT_RESET_FLAG_DEV_ID = (1 << 0)
VFIO_PCI_HOT_RESET_FLAG_DEV_ID_OWNED = (1 << 1)
VFIO_DEVICE_GET_PCI_HOT_RESET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_DEVICE_PCI_HOT_RESET = _IO(VFIO_TYPE, VFIO_BASE + 13)
VFIO_GFX_PLANE_TYPE_PROBE = (1 << 0)
VFIO_GFX_PLANE_TYPE_DMABUF = (1 << 1)
VFIO_GFX_PLANE_TYPE_REGION = (1 << 2)
VFIO_DEVICE_QUERY_GFX_PLANE = _IO(VFIO_TYPE, VFIO_BASE + 14)
VFIO_DEVICE_GET_GFX_DMABUF = _IO(VFIO_TYPE, VFIO_BASE + 15)
VFIO_DEVICE_IOEVENTFD_8 = (1 << 0)
VFIO_DEVICE_IOEVENTFD_16 = (1 << 1)
VFIO_DEVICE_IOEVENTFD_32 = (1 << 2)
VFIO_DEVICE_IOEVENTFD_64 = (1 << 3)
VFIO_DEVICE_IOEVENTFD_SIZE_MASK = (0xf)
VFIO_DEVICE_IOEVENTFD = _IO(VFIO_TYPE, VFIO_BASE + 16)
VFIO_DEVICE_FEATURE_MASK = (0xffff)
VFIO_DEVICE_FEATURE_GET = (1 << 16)
VFIO_DEVICE_FEATURE_SET = (1 << 17)
VFIO_DEVICE_FEATURE_PROBE = (1 << 18)
VFIO_DEVICE_FEATURE = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_DEVICE_BIND_FLAG_TOKEN = (1 << 0)
VFIO_DEVICE_BIND_IOMMUFD = _IO(VFIO_TYPE, VFIO_BASE + 18)
VFIO_DEVICE_ATTACH_PASID = (1 << 0)
VFIO_DEVICE_ATTACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 19)
VFIO_DEVICE_DETACH_PASID = (1 << 0)
VFIO_DEVICE_DETACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 20)
VFIO_DEVICE_FEATURE_PCI_VF_TOKEN = (0)
VFIO_MIGRATION_STOP_COPY = (1 << 0)
VFIO_MIGRATION_P2P = (1 << 1)
VFIO_MIGRATION_PRE_COPY = (1 << 2)
VFIO_DEVICE_FEATURE_MIGRATION = 1
VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE = 2
VFIO_MIG_GET_PRECOPY_INFO = _IO(VFIO_TYPE, VFIO_BASE + 21)
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY = 3
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY_WITH_WAKEUP = 4
VFIO_DEVICE_FEATURE_LOW_POWER_EXIT = 5
VFIO_DEVICE_FEATURE_DMA_LOGGING_START = 6
VFIO_DEVICE_FEATURE_DMA_LOGGING_STOP = 7
VFIO_DEVICE_FEATURE_DMA_LOGGING_REPORT = 8
VFIO_DEVICE_FEATURE_MIG_DATA_SIZE = 9
VFIO_DEVICE_FEATURE_CLEAR_MASTER = 0
VFIO_DEVICE_FEATURE_SET_MASTER = 1
VFIO_DEVICE_FEATURE_BUS_MASTER = 10
VFIO_IOMMU_INFO_PGSIZES = (1 << 0)
VFIO_IOMMU_INFO_CAPS = (1 << 1)
VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE = 1
VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION = 2
VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL = 3
VFIO_IOMMU_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_DMA_MAP_FLAG_READ = (1 << 0)
VFIO_DMA_MAP_FLAG_WRITE = (1 << 1)
VFIO_DMA_MAP_FLAG_VADDR = (1 << 2)
VFIO_IOMMU_MAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 13)
VFIO_DMA_UNMAP_FLAG_GET_DIRTY_BITMAP = (1 << 0)
VFIO_DMA_UNMAP_FLAG_ALL = (1 << 1)
VFIO_DMA_UNMAP_FLAG_VADDR = (1 << 2)
VFIO_IOMMU_UNMAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 14)
VFIO_IOMMU_ENABLE = _IO(VFIO_TYPE, VFIO_BASE + 15)
VFIO_IOMMU_DISABLE = _IO(VFIO_TYPE, VFIO_BASE + 16)
VFIO_IOMMU_DIRTY_PAGES_FLAG_START = (1 << 0)
VFIO_IOMMU_DIRTY_PAGES_FLAG_STOP = (1 << 1)
VFIO_IOMMU_DIRTY_PAGES_FLAG_GET_BITMAP = (1 << 2)
VFIO_IOMMU_DIRTY_PAGES = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_IOMMU_SPAPR_INFO_DDW = (1 << 0)
VFIO_IOMMU_SPAPR_TCE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_EEH_PE_DISABLE = 0
VFIO_EEH_PE_ENABLE = 1
VFIO_EEH_PE_UNFREEZE_IO = 2
VFIO_EEH_PE_UNFREEZE_DMA = 3
VFIO_EEH_PE_GET_STATE = 4
VFIO_EEH_PE_STATE_NORMAL = 0
VFIO_EEH_PE_STATE_RESET = 1
VFIO_EEH_PE_STATE_STOPPED = 2
VFIO_EEH_PE_STATE_STOPPED_DMA = 4
VFIO_EEH_PE_STATE_UNAVAIL = 5
VFIO_EEH_PE_RESET_DEACTIVATE = 5
VFIO_EEH_PE_RESET_HOT = 6
VFIO_EEH_PE_RESET_FUNDAMENTAL = 7
VFIO_EEH_PE_CONFIGURE = 8
VFIO_EEH_PE_INJECT_ERR = 9
VFIO_EEH_PE_OP = _IO(VFIO_TYPE, VFIO_BASE + 21)
VFIO_IOMMU_SPAPR_REGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_IOMMU_SPAPR_UNREGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 18)
VFIO_IOMMU_SPAPR_TCE_CREATE = _IO(VFIO_TYPE, VFIO_BASE + 19)
VFIO_IOMMU_SPAPR_TCE_REMOVE = _IO(VFIO_TYPE, VFIO_BASE + 20)