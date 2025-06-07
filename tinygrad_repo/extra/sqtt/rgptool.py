#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, struct, hashlib, pickle, code, typing, functools
import tinygrad.runtime.autogen.sqtt as sqtt
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.helpers import round_up, flatten, all_same
from dataclasses import dataclass

CHUNK_CLASSES = {
  sqtt.SQTT_FILE_CHUNK_TYPE_ASIC_INFO: sqtt.struct_sqtt_file_chunk_asic_info,
  sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DESC: sqtt.struct_sqtt_file_chunk_sqtt_desc,
  sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DATA: sqtt.struct_sqtt_file_chunk_sqtt_data,
  sqtt.SQTT_FILE_CHUNK_TYPE_API_INFO: sqtt.struct_sqtt_file_chunk_api_info,
  sqtt.SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS: sqtt.struct_sqtt_file_chunk_queue_event_timings,
  sqtt.SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION: sqtt.struct_sqtt_file_chunk_clock_calibration,
  sqtt.SQTT_FILE_CHUNK_TYPE_CPU_INFO: sqtt.struct_sqtt_file_chunk_cpu_info,
  sqtt.SQTT_FILE_CHUNK_TYPE_SPM_DB: sqtt.struct_sqtt_file_chunk_spm_db,
  sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE: sqtt.struct_sqtt_file_chunk_code_object_database,
  sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS: sqtt.struct_sqtt_file_chunk_code_object_loader_events,
  sqtt.SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION: sqtt.struct_sqtt_file_chunk_pso_correlation,
}

def pretty(val, pad=0) -> str:
  if isinstance(val, (ctypes.Structure, ctypes.Union)):
    nl = '\n' # old python versions don't support \ in f-strings
    return f"{val.__class__.__name__}({nl}{' '*(pad+2)}{(f', {nl}'+' '*(pad+2)).join([f'{field[0]}={pretty(getattr(val, field[0]), pad=pad+2)}' for field in val._fields_])}{nl}{' '*pad})"
  if isinstance(val, ctypes.Array):
    return f"[{', '.join(map(pretty, val))}]"
  if isinstance(val, int) and val >= 1024: return hex(val)
  return repr(val)

@dataclass(frozen=True)
class RGPChunk:
  header: sqtt.Structure
  data: list[typing.Any]|list[tuple[typing.Any, bytes]]|bytes|None = None
  def print(self):
    print(pretty(self.header))
    # if isinstance(self.data, bytes): print(repr(self.data))
    if isinstance(self.data, list):
      for dchunk in self.data:
        if isinstance(dchunk, tuple):
          print(pretty(dchunk[0]))
          # print(repr(dchunk[1]))
        else:
          print(pretty(dchunk))
  # TODO: `def fixup` and true immutability
  def to_bytes(self, offset:int) -> bytes:
    cid = self.header.header.chunk_id.type
    match cid:
      case _ if cid in {sqtt.SQTT_FILE_CHUNK_TYPE_ASIC_INFO, sqtt.SQTT_FILE_CHUNK_TYPE_CPU_INFO, sqtt.SQTT_FILE_CHUNK_TYPE_API_INFO, sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DESC}:
        self.header.header.size_in_bytes = ctypes.sizeof(self.header)
        return bytes(self.header)
      case sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DATA:
        assert isinstance(self.data, bytes)
        self.header.header.size_in_bytes = ctypes.sizeof(self.header) + len(self.data)
        self.header.offset = offset+ctypes.sizeof(self.header)
        self.header.size = len(self.data)
        return bytes(self.header) + self.data
      case sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE:
        assert isinstance(self.data, list)
        data_codb = typing.cast(list[tuple[sqtt.struct_sqtt_code_object_database_record, bytes]], self.data)
        ret = bytearray()
        sz = ctypes.sizeof(self.header)+sum([ctypes.sizeof(record_hdr)+round_up(len(record_blob), 4) for record_hdr,record_blob in data_codb])
        self.header.header.size_in_bytes = sz
        self.header.offset = offset
        self.header.record_count = len(data_codb)
        self.header.size = sz
        ret += self.header
        for record_hdr,record_blob in data_codb:
          record_hdr.size = round_up(len(record_blob), 4)
          ret += record_hdr
          ret += record_blob.ljust(4, b'\x00')
        return ret
      case sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS:
        assert isinstance(self.data, list)
        data_lev = typing.cast(list[tuple[sqtt.struct_sqtt_code_object_loader_events_record]], self.data)
        self.header.header.size_in_bytes = ctypes.sizeof(self.header)+ctypes.sizeof(sqtt.struct_sqtt_code_object_loader_events_record)*len(data_lev)
        self.header.offset = offset
        self.header.record_size = ctypes.sizeof(sqtt.struct_sqtt_code_object_loader_events_record)
        self.header.record_count = len(data_lev)
        return bytes(self.header) + b''.join(map(bytes, data_lev))
      case sqtt.SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION:
        assert isinstance(self.data, list)
        data_pso = typing.cast(list[tuple[sqtt.struct_sqtt_pso_correlation_record]], self.data)
        self.header.header.size_in_bytes = ctypes.sizeof(self.header)+ctypes.sizeof(sqtt.struct_sqtt_pso_correlation_record)*len(data_pso)
        self.header.offset = offset
        self.header.record_size = ctypes.sizeof(sqtt.struct_sqtt_pso_correlation_record)
        self.header.record_count = len(data_pso)
        return bytes(self.header) + b''.join(map(bytes, data_pso))
      case _: raise NotImplementedError(pretty(self.header))

@dataclass(frozen=True)
class RGP:
  header: sqtt.struct_sqtt_file_header
  chunks: list[RGPChunk]
  @staticmethod
  def from_bytes(blob: bytes) -> RGP:
    file_header = sqtt.struct_sqtt_file_header.from_buffer_copy(blob)
    assert file_header.magic_number == sqtt.SQTT_FILE_MAGIC_NUMBER and file_header.version_major == sqtt.SQTT_FILE_VERSION_MAJOR
    i = file_header.chunk_offset
    chunks = []
    while i < len(blob):
      assert i%4==0, hex(i)
      hdr = sqtt.struct_sqtt_file_chunk_header.from_buffer_copy(blob, i)
      cid = hdr.chunk_id.type
      header: ctypes.Structure
      match cid:
        case _ if cid in {sqtt.SQTT_FILE_CHUNK_TYPE_RESERVED, sqtt.SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS, sqtt.SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION, sqtt.SQTT_FILE_CHUNK_TYPE_SPM_DB}:
          chunk = None
        case sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE:
          header = sqtt.struct_sqtt_file_chunk_code_object_database.from_buffer_copy(blob, i)
          j = header.offset + ctypes.sizeof(header)
          data: list = []
          while j < header.offset + header.size:
            rec_hdr: ctypes.Structure = sqtt.struct_sqtt_code_object_database_record.from_buffer_copy(blob, j)
            data.append((rec_hdr, elf:=blob[j+ctypes.sizeof(rec_hdr):j+ctypes.sizeof(rec_hdr)+rec_hdr.size]))
            assert elf[:4] == b'\x7fELF', repr(elf[:16])
            j += ctypes.sizeof(rec_hdr)+rec_hdr.size
          assert len(data) == header.record_count
          chunk = RGPChunk(header, data)
        case sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS:
          header = sqtt.struct_sqtt_file_chunk_code_object_loader_events.from_buffer_copy(blob, i)
          data = [sqtt.struct_sqtt_code_object_loader_events_record.from_buffer_copy(blob, header.offset+ctypes.sizeof(header)+j*header.record_size)
                                                                                                                for j in range(header.record_count)]
          chunk = RGPChunk(header, data)
        case sqtt.SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION:
          header = sqtt.struct_sqtt_file_chunk_pso_correlation.from_buffer_copy(blob, i)
          data = [sqtt.struct_sqtt_pso_correlation_record.from_buffer_copy(blob, header.offset+ctypes.sizeof(header)+j*header.record_size)
                                                                                                                for j in range(header.record_count)]
          chunk = RGPChunk(header, data)
        case sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DATA:
          header = sqtt.struct_sqtt_file_chunk_sqtt_data.from_buffer_copy(blob, i)
          chunk = RGPChunk(header, blob[header.offset:header.offset+header.size])
        case _ if cid in {sqtt.SQTT_FILE_CHUNK_TYPE_ASIC_INFO, sqtt.SQTT_FILE_CHUNK_TYPE_CPU_INFO, sqtt.SQTT_FILE_CHUNK_TYPE_API_INFO,
                          sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DESC}:
          chunk = RGPChunk(CHUNK_CLASSES[cid].from_buffer_copy(blob, i))
        case _:
          chunk = None
          print(f"unknown chunk id {cid}")
      if chunk is not None: chunks.append(chunk)
      i += hdr.size_in_bytes
    assert i == len(blob), f'{i} != {len(blob)}'
    return RGP(file_header, chunks)
  @staticmethod
  def from_profile(profile_pickled, device:str|None=None):
    profile: list[ProfileEvent] = pickle.loads(profile_pickled)
    device_events = {x.device:x for x in profile if isinstance(x, ProfileDeviceEvent) and x.device.startswith('AMD')}
    if device is None:
      if len(device_events) == 0: raise RuntimeError('No supported devices found in profile')
      if len(device_events) > 1: raise RuntimeError(f"More than one supported device found, select which one to export: {', '.join(device_events.keys())}")
      _, device_event = device_events.popitem()
    else:
      if device not in device_events: raise RuntimeError(f"Device {device} not found in profile, devices in profile: {', '.join(device_events.keys())} ")
      device_event = device_events[device]
    sqtt_events = [x for x in profile if isinstance(x, ProfileSQTTEvent) and x.device == device_event.device]
    if len(sqtt_events) == 0: raise RuntimeError(f"Device {device_event.device} doesn't contain SQTT data")
    sqtt_itrace_enabled = any([event.itrace for event in sqtt_events])
    sqtt_itrace_masked = not all_same([event.itrace for event in sqtt_events])
    sqtt_itrace_se_mask = functools.reduce(lambda a,b: a|b, [int(event.itrace) << event.se for event in sqtt_events], 0) if sqtt_itrace_masked else 0
    load_events = [x for x in profile if isinstance(x, ProfileProgramEvent) and x.device == device_event.device]
    loads = [(event.base, struct.unpack('<Q', hashlib.md5(event.lib).digest()[:8])*2) for event in load_events if event.base is not None and event.lib is not None]
    code_objects = list(dict.fromkeys([x.lib for x in load_events if x.lib is not None]).keys())
    if len(loads) == 0: raise RuntimeError('No load events in profile')
    # TODO: tons of stuff hardcoded for 7900xtx
    file_header = sqtt.struct_sqtt_file_header(
      magic_number=sqtt.SQTT_FILE_MAGIC_NUMBER,
      version_major=sqtt.SQTT_FILE_VERSION_MAJOR,
      version_minor=sqtt.SQTT_FILE_VERSION_MINOR,
      flags=sqtt.struct_sqtt_file_header_flags(
        _0=sqtt.union_sqtt_file_header_flags_0(value=1),
      ),
      chunk_offset=ctypes.sizeof(sqtt.struct_sqtt_file_header),
    )
    chunks = [
      RGPChunk(sqtt.struct_sqtt_file_chunk_cpu_info(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_CPU_INFO),
          major_version=0, minor_version=0,
        ),
        cpu_timestamp_freq=1000000000,
        clock_speed=2994, # in mhz???
        num_logical_cores=64,
        num_physical_cores=32,
        system_ram_size=256*1024, # in mb???
      )),
      RGPChunk(sqtt.struct_sqtt_file_chunk_asic_info(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_ASIC_INFO),
          major_version=0, minor_version=5,
        ),
        flags=0,
        trace_shader_core_clock=0x93f05080,
        trace_memory_clock=0x4a723a40,
        device_id=0x744c,
        device_revision_id=0xc8,
        vgprs_per_simd=1536,
        sgprs_per_simd=128*16,
        shader_engines=6,
        compute_unit_per_shader_engine=16,
        simd_per_compute_unit=2,
        wavefronts_per_simd=16,
        minimum_vgpr_alloc=4,
        vgpr_alloc_granularity=8,
        minimum_sgpr_alloc=128,
        sgpr_alloc_granularity=128,
        hardware_contexts=8,
        gpu_type=sqtt.SQTT_GPU_TYPE_DISCRETE,
        gfxip_level=sqtt.SQTT_GFXIP_LEVEL_GFXIP_11_0,
        gpu_index=0,
        gds_size=0,
        gds_per_shader_engine=0,
        ce_ram_size=0,
        ce_ram_size_graphics=0,
        ce_ram_size_compute=0,
        max_number_of_dedicated_cus=0,
        vram_size=24 * 1024 * 1024 * 1024,  # 24 GB
        vram_bus_width=384, # 384-bit
        l2_cache_size=6 * 1024 * 1024, # 6 MB
        l1_cache_size=32 * 1024, # 32 KB per SIMD (?)
        lds_size=65536, # 64 KB per CU
        gpu_name=b'NAVI31',
        alu_per_clock=0,
        texture_per_clock=0,
        prims_per_clock=6,
        pixels_per_clock=0,
        gpu_timestamp_frequency=100000000, # 100 MHz
        max_shader_core_clock=2500000000, # 2.5 GHz (boost clock)
        max_memory_clock=1250000000,  # 1.25 GHz
        memory_ops_per_clock=16,
        memory_chip_type=sqtt.SQTT_MEMORY_TYPE_GDDR6,
        lds_granularity=512,
        cu_mask=((255, 255),)*6 + ((0,0),)*(32-6),
        gl1_cache_size=256 * 1024, # 256 KB
        instruction_cache_size=32 * 1024, # 32 KB
        scalar_cache_size=16 * 1024, # 16 KB
        mall_cache_size=96 * 1024 * 1024, # 96 MB
      )),
      RGPChunk(sqtt.struct_sqtt_file_chunk_api_info(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_API_INFO),
          major_version=0,
          minor_version=2,
        ),
        api_type=5, # HIP, not in enum
        major_version=12, minor_version=0,
        profiling_mode=sqtt.SQTT_PROFILING_MODE_PRESENT,
        instruction_trace_mode=sqtt.SQTT_INSTRUCTION_TRACE_FULL_FRAME if sqtt_itrace_enabled else sqtt.SQTT_INSTRUCTION_TRACE_DISABLED,
        instruction_trace_data=sqtt.union_sqtt_instruction_trace_data(
          shader_engine_filter=sqtt.struct_sqtt_instruction_trace_data_shader_engine_filter(mask=sqtt_itrace_se_mask),
        ),
      )),
      *flatten([(
        RGPChunk(sqtt.struct_sqtt_file_chunk_sqtt_desc(
          header=sqtt.struct_sqtt_file_chunk_header(
            chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DESC, index=sqtt_event.se),
            major_version=0, minor_version=2,
          ),
          shader_engine_index=sqtt_event.se,
          sqtt_version=sqtt.SQTT_VERSION_3_2,
          _0=sqtt.union_sqtt_file_chunk_sqtt_desc_0(
            v1=sqtt.struct_sqtt_file_chunk_sqtt_desc_0_v1(
              instrumentation_spec_version=1,
              instrumentation_api_version=0,
              compute_unit_index=0,
            )
          ),
        )),
        RGPChunk(sqtt.struct_sqtt_file_chunk_sqtt_data(
          header=sqtt.struct_sqtt_file_chunk_header(
            chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_SQTT_DATA, index=sqtt_event.se),
            major_version=0, minor_version=0,
          ),
        ), sqtt_event.blob),
      ) for sqtt_event in sqtt_events]),
      RGPChunk(sqtt.struct_sqtt_file_chunk_code_object_database(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE),
          major_version=0, minor_version=0,
        ),
      ), [(sqtt.struct_sqtt_code_object_database_record(), lib) for lib in code_objects]),
      RGPChunk(sqtt.struct_sqtt_file_chunk_code_object_loader_events(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS),
          major_version=1, minor_version=0,
        ),
      ), [sqtt.struct_sqtt_code_object_loader_events_record(base_address=base, code_object_hash=hash) for base,hash in loads]),
      RGPChunk(sqtt.struct_sqtt_file_chunk_pso_correlation(
        header=sqtt.struct_sqtt_file_chunk_header(
          chunk_id=sqtt.struct_sqtt_file_chunk_id(type=sqtt.SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION),
          major_version=0, minor_version=0,
        ),
      ), [sqtt.struct_sqtt_pso_correlation_record(api_pso_hash=hash[0], pipeline_hash=hash) for _,hash in loads])
    ]
    return RGP(file_header, chunks)
  def to_bytes(self) -> bytes:
    ret = bytearray()
    ret += self.header
    for chunk in self.chunks:
      ret += chunk.to_bytes(len(ret))
    return bytes(ret)
  def print(self):
    print(pretty(self.header))
    for chunk in self.chunks: chunk.print()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='rgptool', description='A tool to create (from pickled tinygrad profile), inspect and modify Radeon GPU Profiler files')
  parser.add_argument('command')
  parser.add_argument('input')
  parser.add_argument('-d', '--device')
  parser.add_argument('-o', '--output')
  args = parser.parse_args()

  with open(args.input, 'rb') as fd: input_bytes = fd.read()

  match args.command:
    case 'print':
      rgp = RGP.from_bytes(input_bytes)
      rgp.print()
    case 'create':
      rgp = RGP.from_profile(input_bytes, device=args.device)
      # rgp.to_bytes() # fixup
      # rgp.print()
    case 'repl':
      rgp = RGP.from_bytes(input_bytes)
      code.interact(local=locals())
    case _: raise RuntimeError(args.command)

  if args.output is not None:
    with open(args.output, 'wb+') as fd: fd.write(rgp.to_bytes())
