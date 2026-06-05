from collections import defaultdict
from tinygrad.device import Device
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  return all(not d.startswith(("DISK", "TINYFS")) and hasattr(Device[d].allocator, "_offset") for d in devs)

LaneKey = tuple[str, int]

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None) -> UOp:
  if NO_MEMORY_PLANNER: return linear
  if held_bufs is None: held_bufs = set()

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  copy_bufs: set[UOp] = set()
  for i, si in enumerate(linear.src):
    si_bufs = [b for src in si.src[1:] for b in _collect_bufs(src) if _can_plan(b, held_bufs)]
    for b in si_bufs:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
    if si.src[0].op is Ops.COPY: copy_bufs.update(si_bufs)
  if not first_appearance: return linear

  # separate copy and compute buffers into different lanes to avoid introducing dependencies (copy->compute->copy)
  def _key(b:UOp): return (b.device, 1 if b in copy_bufs else 0)
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}

  # suballocation: build sorted open/close events, then alloc/free in order
  block_size = 256
  nbytes = {b: round_up(b.arg * b.dtype.itemsize, block_size) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1 + buf_hold.get(b, 0), False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2

  offsets:dict[UOp, int] = {}
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    if is_open: offsets[buf] = peaks[_key(buf)][1].alloc(nbytes[buf])
    else: peaks[_key(buf)][1].free(offsets[buf])
    peaks[_key(buf)] = (max(peaks[_key(buf)][0], offsets[buf] + buf.arg * buf.dtype.itemsize), peaks[_key(buf)][1])
  arena_sizes = {key: round_up(peak, block_size) for key, (peak, _) in peaks.items()}

  # build replace_map: each buffer becomes a BUFFER_VIEW into a shared per-device-lane arena
  arenas = {key: UOp.new_buffer(key[0], sz, dtypes.int8) for key, sz in arena_sizes.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[_key(buf_uop)],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1 and (omem:=sum(nbytes.values()) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)
