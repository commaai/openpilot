from functools import cache, partial
import math
from tinygrad import Tensor, Device, UOp, dtypes
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer import Estimates


@cache
def _program(M, N, K, device):
  name = f'worldmodel_fp8_linear_{M}_{N}_{K}'
  # Eight waves share 128 x 128 x 64 tiles; padding spreads LDS reads across memory banks.
  stride = 72
  size = 128 * stride
  lines = [
    f'@lds = internal addrspace(3) global [{size * 2} x i8] undef, align 16',
    'declare i32 @llvm.amdgcn.workgroup.id.x()',
    'declare i32 @llvm.amdgcn.workgroup.id.y()',
    'declare i32 @llvm.amdgcn.workitem.id.x()',
    'declare void @llvm.amdgcn.s.barrier()',
    'declare float @llvm.fma.f32(float, float, float)',
    'declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32(<2 x i32>, <2 x i32>, <8 x float>)',
    f'define amdgpu_kernel void @{name}(ptr addrspace(1) %c, ptr addrspace(1) %a, ptr addrspace(1) %b, ' +
    'ptr addrspace(1) %scale_a, ptr addrspace(1) %scale_b, ptr addrspace(1) %bias) #0 {',
    'entry:',
    '%as = load float, ptr addrspace(1) %scale_a, align 4',
    '%bs = load float, ptr addrspace(1) %scale_b, align 4',
    '%scale = fmul float %as, %bs',
    '%bx = call i32 @llvm.amdgcn.workgroup.id.x()',
    '%by = call i32 @llvm.amdgcn.workgroup.id.y()',
    '%tid = call i32 @llvm.amdgcn.workitem.id.x()',
    '%lane = and i32 %tid, 31',
    '%wave = lshr i32 %tid, 5',
    '%wm = lshr i32 %wave, 2',
    '%wn = and i32 %wave, 3',
    '%wm64 = shl i32 %wm, 6',
    '%wn32 = shl i32 %wn, 5',
    '%lm = and i32 %lane, 15',
    '%kh = lshr i32 %lane, 4',
    '%kh8 = shl i32 %kh, 3',
    '%mr = add i32 %wm64, %lm',
    '%nr = add i32 %wn32, %lm',
    f'%a_shared_row = mul i32 %mr, {stride}',
    f'%b_shared_row = mul i32 %nr, {stride}',
    '%a_shared = add i32 %a_shared_row, %kh8',
    f'%b_shared_0 = add i32 %b_shared_row, {size}',
    '%b_shared = add i32 %b_shared_0, %kh8',
    '%m_base = shl i32 %by, 7',
    '%n_base = shl i32 %bx, 7',
    '%load_row = lshr i32 %tid, 2',
    '%load_col_0 = and i32 %tid, 3',
    '%load_col = shl i32 %load_col_0, 4',
    '%load_ar = add i32 %m_base, %load_row',
    '%load_br = add i32 %n_base, %load_row',
    f'%load_ab = mul i32 %load_ar, {K}',
    f'%load_bb = mul i32 %load_br, {K}',
    '%load_ai = add i32 %load_ab, %load_col',
    '%load_bi = add i32 %load_bb, %load_col',
    f'%lds_row = mul i32 %load_row, {stride}',
    '%lds_off = add i32 %lds_row, %load_col',
    'br label %loop',
    'loop:',
    '%kk = phi i32 [0, %entry], [%next, %loop]',
  ]

  def emit(s):
    lines.append(s)

  for tm in range(4):
    for tn in range(2):
      t = f'{tm}_{tn}'
      emit(f'%acc{t} = phi <8 x float> [zeroinitializer, %entry], [%f3_{t}, %loop]')
  for ab in ['a', 'b']:
    emit(f'%g{ab} = add i32 %load_{ab}i, %kk')
    for r in range(2):
      emit(f'%gi{ab}{r} = add i32 %g{ab}, {r * 64 * K}')
      emit(f'%gp{ab}{r} = getelementptr i8, ptr addrspace(1) %{ab}, i32 %gi{ab}{r}')
      emit(f'%gv{ab}{r} = load <4 x i32>, ptr addrspace(1) %gp{ab}{r}, align 16')
      emit(f'%li{ab}{r} = add i32 %lds_off, {r * 64 * stride + (size if ab == "b" else 0)}')
      emit(f'%lp{ab}{r} = getelementptr i8, ptr addrspace(3) @lds, i32 %li{ab}{r}')
      emit(f'store <4 x i32> %gv{ab}{r}, ptr addrspace(3) %lp{ab}{r}, align 8')
  emit('call void @llvm.amdgcn.s.barrier()')
  for ik in range(4):
    for ab, nt in [('a', 4), ('b', 2)]:
      for t in range(nt):
        emit(f'%si{ab}{ik}_{t} = add i32 %{ab}_shared, {t * 16 * stride + ik * 16}')
        emit(f'%sp{ab}{ik}_{t} = getelementptr i8, ptr addrspace(3) @lds, i32 %si{ab}{ik}_{t}')
        emit(f'%sv{ab}{ik}_{t} = load <2 x i32>, ptr addrspace(3) %sp{ab}{ik}_{t}, align 8')
    for tm in range(4):
      for tn in range(2):
        t = f'{tm}_{tn}'
        prev = f'%f{ik - 1}_{t}' if ik else f'%acc{t}'
        emit(
          f'%f{ik}_{t} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32(' +
          f'<2 x i32> %sva{ik}_{tm}, <2 x i32> %svb{ik}_{tn}, <8 x float> {prev})'
        )
  emit('call void @llvm.amdgcn.s.barrier()')
  emit('%next = add i32 %kk, 64')
  emit(f'%continue = icmp ult i32 %next, {K}')
  emit('br i1 %continue, label %loop, label %exit')
  emit('exit:')
  emit('%row_0 = add i32 %m_base, %wm64')
  emit('%row = add i32 %row_0, %kh8')
  emit('%col_0 = add i32 %n_base, %wn32')
  emit('%col = add i32 %col_0, %lm')
  emit(f'%coff_0 = mul i32 %row, {N}')
  emit('%coff = add i32 %coff_0, %col')
  for tn in range(2):
    emit(f'%bcol{tn} = add i32 %col, {tn * 16}')
    emit(f'%bptr{tn} = getelementptr i16, ptr addrspace(1) %bias, i32 %bcol{tn}')
    emit(f'%bval{tn} = load i16, ptr addrspace(1) %bptr{tn}, align 2')
    emit(f'%bext{tn} = zext i16 %bval{tn} to i32')
    emit(f'%bbits{tn} = shl i32 %bext{tn}, 16')
    emit(f'%bias{tn} = bitcast i32 %bbits{tn} to float')
  for tm in range(4):
    for tn in range(2):
      for e in range(8):
        t = f'{tm}_{tn}_{e}'
        emit(f'%ci{t} = add i32 %coff, {(tm * 16 + e) * N + tn * 16}')
        emit(f'%cp{t} = getelementptr i16, ptr addrspace(1) %c, i32 %ci{t}')
        emit(f'%cv{t} = extractelement <8 x float> %f3_{tm}_{tn}, i32 {e}')
        emit(f'%scaled{t} = call float @llvm.fma.f32(float %cv{t}, float %scale, float %bias{tn})')
        emit(f'%bits{t} = bitcast float %scaled{t} to i32')
        emit(f'%top{t} = lshr i32 %bits{t}, 16')
        emit(f'%odd{t} = and i32 %top{t}, 1')
        emit(f'%round{t} = add i32 %bits{t}, 32767')
        emit(f'%rounded{t} = add i32 %round{t}, %odd{t}')
        emit(f'%tr{t} = lshr i32 %rounded{t}, 16')
        emit(f'%bf{t} = trunc i32 %tr{t} to i16')
        emit(f'store i16 %bf{t}, ptr addrspace(1) %cp{t}, align 2')
  lines += ['ret void', '}', 'attributes #0 = { nounwind "amdgpu-flat-work-group-size"="256,256" "no-trapping-math"="true" }']
  src = '\n'.join(lines)
  return name, src, Device[device].renderer.compiler.compile_cached(src)


def _kernel(c, a, b, scale_a, scale_b, bias):
  M, K = a.shape
  N = b.shape[0]
  assert M % 128 == N % 128 == K % 64 == 0
  name, src, lib = _program(M, N, K, a.device)
  sink = UOp.sink(
    c.base,
    a.base,
    b.base,
    scale_a.base,
    scale_b.base,
    bias.base,
    UOp.special(N // 128, 'gidx0'),
    UOp.special(M // 128, 'gidx1'),
    UOp.special(256, 'lidx0'),
    arg=KernelInfo(name=name, estimates=Estimates(ops=2 * M * N * K, mem=M * K + N * K + M * N * 2)),
  )
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))


def fp8_linear(x, weight, scale, weight_scale, bias):
  out = Tensor.empty(x.shape[0], weight.shape[0], dtype=dtypes.bfloat16, device=x.device)
  return out.custom_kernel(x, weight, scale.reshape(1), weight_scale.reshape(1), bias, fxn=_kernel)[0]


@cache
def _attention_program(seq, total, start_frame, device):
  name = f'worldmodel_attention_{seq}_{total}_{start_frame}'
  stride, area = 72, 64 * 72
  lines = [
    f'@lds = internal addrspace(3) global [{2 * area} x i16] undef, align 16',
    'declare i32 @llvm.amdgcn.workgroup.id.x()',
    'declare i32 @llvm.amdgcn.workgroup.id.y()',
    'declare i32 @llvm.amdgcn.workitem.id.x()',
    'declare i32 @llvm.amdgcn.ds.swizzle(i32, i32)',
    'declare void @llvm.amdgcn.s.barrier()',
    'declare float @llvm.maxnum.f32(float, float)',
    'declare float @llvm.exp2.f32(float)',
    'declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8bf16(<8 x i16>, <8 x i16>, <8 x float>)',
    f'define amdgpu_kernel void @{name}(ptr addrspace(1) %o, ptr addrspace(1) %q, ptr addrspace(1) %k, ptr addrspace(1) %v) #0 {{',
    'entry:',
    '%block = call i32 @llvm.amdgcn.workgroup.id.x()',
    '%head = call i32 @llvm.amdgcn.workgroup.id.y()',
    '%tid = call i32 @llvm.amdgcn.workitem.id.x()',
    '%lane = and i32 %tid, 31',
    '%wave = lshr i32 %tid, 5',
    '%lm = and i32 %lane, 15',
    '%half = lshr i32 %lane, 4',
    '%kh = shl i32 %half, 3',
    '%khalf = shl i32 %half, 2',
    '%qm0 = shl i32 %block, 6',
    '%qm1 = shl i32 %wave, 4',
    '%qm2 = add i32 %qm0, %qm1',
    '%qm = add i32 %qm2, %lm',
    f'%qbase = mul i32 %head, {seq * 64}',
    '%qrow = mul i32 %qm, 64',
    '%qoff0 = add i32 %qbase, %qrow',
    '%qoff = add i32 %qoff0, %khalf',
    f'%kvbase = mul i32 %head, {total * 64}',
    '%frame = lshr i32 %block, 1',
    f'%frame_end = add i32 %frame, {start_frame + 1}',
    '%kend = shl i32 %frame_end, 7',
    '%loadrow = lshr i32 %tid, 3',
    '%loadcol0 = and i32 %tid, 7',
    '%loadcol = shl i32 %loadcol0, 3',
    '%outrow0 = add i32 %qm2, %kh',
    '%outrow1 = mul i32 %outrow0, 64',
    '%outoff0 = add i32 %qbase, %outrow1',
    '%outoff = add i32 %outoff0, %lm',
    '%pm0 = add i32 %qm1, %kh',
    f'%pbase0 = mul i32 %pm0, {stride}',
    '%pbase = add i32 %pbase0, %lm',
  ]
  emit = lines.append

  def fragment(key, pointer, space):
    emit(f'%{key}lo = load <4 x i16>, ptr addrspace({space}) {pointer}, align 8')
    emit(f'%{key}ptr = getelementptr i16, ptr addrspace({space}) {pointer}, i32 8')
    emit(f'%{key}hi = load <4 x i16>, ptr addrspace({space}) %{key}ptr, align 8')
    emit(f'%{key} = shufflevector <4 x i16> %{key}lo, <4 x i16> %{key}hi, ' +
         '<8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>')

  for ik in range(4):
    emit(f'%qi{ik} = add i32 %qoff, {ik * 16}')
    emit(f'%qp{ik} = getelementptr i16, ptr addrspace(1) %q, i32 %qi{ik}')
    fragment(f'qv{ik}', f'%qp{ik}', 1)

  def bf16(key, value):
    emit(f'%{key}bits = bitcast float {value} to i32')
    emit(f'%{key}top = lshr i32 %{key}bits, 16')
    emit(f'%{key}odd = and i32 %{key}top, 1')
    emit(f'%{key}rnd0 = add i32 %{key}bits, 32767')
    emit(f'%{key}rnd = add i32 %{key}rnd0, %{key}odd')
    emit(f'%{key}tr = lshr i32 %{key}rnd, 16')
    emit(f'%{key}bf = trunc i32 %{key}tr to i16')
    return f'%{key}bf'

  def reduce_max(key, val):
    for delta in (8, 4, 2, 1):
      n = f'{key}_{delta}'
      emit(f'%{n}bits = bitcast float {val} to i32')
      emit(f'%{n}sh = call i32 @llvm.amdgcn.ds.swizzle(i32 %{n}bits, i32 {31 | delta << 10})')
      emit(f'%{n}f = bitcast i32 %{n}sh to float')
      emit(f'%{n} = call float @llvm.maxnum.f32(float {val}, float %{n}f)')
      val = f'%{n}'
    return val

  def shuffle(key, value, mask):
    emit(f'%{key}bits = bitcast float {value} to i32')
    emit(f'%{key}sh = call i32 @llvm.amdgcn.ds.swizzle(i32 %{key}bits, i32 {mask})')
    emit(f'%{key} = bitcast i32 %{key}sh to float')
    return f'%{key}'

  # Find row maxima, sum exponentials, then round normalized probabilities to BF16 for PV.
  # Match the original softmax's ordered sums of four and PV's separate WMMA tile accumulation.
  emit('br label %pass1')
  for phase in (1, 2, 3):
    tag = f't{phase}'
    emit(f'pass{phase}:')
    emit(f'%{tag}kk = phi i32 [0, %' + ('entry' if phase == 1 else f'between{phase - 1}') + f'], [%{tag}next, %pass{phase}]')
    if phase == 1:
      for e in range(8):
        emit(f'%m{e} = phi float [0xFFF0000000000000, %entry], [%mnew{e}, %pass1]')
    elif phase == 2:
      for e in range(8):
        emit(f'%l{e} = phi float [0.0, %between1], [%lnew{e}, %pass2]')
    else:
      for n in range(4):
        emit(f'%acc{n} = phi <8 x float> [zeroinitializer, %between2], [%pv3_{n}, %pass3]')

    emit(f'%{tag}kr0 = add i32 %{tag}kk, %loadrow')
    emit(f'%{tag}kr1 = mul i32 %{tag}kr0, 64')
    emit(f'%{tag}kb = add i32 %kvbase, %{tag}kr1')
    emit(f'%{tag}ki = add i32 %{tag}kb, %loadcol')
    emit(f'%{tag}ls0 = mul i32 %loadrow, {stride}')
    emit(f'%{tag}ls = add i32 %{tag}ls0, %loadcol')
    for r in range(4):
      emit(f'%{tag}kgi{r} = add i32 %{tag}ki, {r * 16 * 64}')
      emit(f'%{tag}kgp{r} = getelementptr i16, ptr addrspace(1) %k, i32 %{tag}kgi{r}')
      emit(f'%{tag}kgv{r} = load <8 x i16>, ptr addrspace(1) %{tag}kgp{r}, align 16')
      emit(f'%{tag}kli{r} = add i32 %{tag}ls, {r * 16 * stride + area}')
      emit(f'%{tag}klp{r} = getelementptr i16, ptr addrspace(3) @lds, i32 %{tag}kli{r}')
      emit(f'store <8 x i16> %{tag}kgv{r}, ptr addrspace(3) %{tag}klp{r}, align 16')
    emit('call void @llvm.amdgcn.s.barrier()')
    emit(f'%{tag}kn0 = mul i32 %lm, {stride}')
    emit(f'%{tag}kn = add i32 %{tag}kn0, %khalf')
    for ik in range(4):
      for n in range(4):
        key = f'{tag}qk{ik}_{n}'
        emit(f'%{key}idx = add i32 %{tag}kn, {area + n * 16 * stride + ik * 16}')
        emit(f'%{key}ptr = getelementptr i16, ptr addrspace(3) @lds, i32 %{key}idx')
        fragment(f'{key}kv', f'%{key}ptr', 3)
        prev = 'zeroinitializer' if ik == 0 else f'%{tag}qk{ik - 1}_{n}'
        emit(f'%{key} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8bf16(' +
             f'<8 x i16> %qv{ik}, <8 x i16> %{key}kv, <8 x float> {prev})')
    emit('call void @llvm.amdgcn.s.barrier()')
    for e in range(8):
      for n in range(4):
        key = f'{tag}s{e}_{n}'
        emit(f'%{key}raw = extractelement <8 x float> %{tag}qk3_{n}, i32 {e}')
        emit(f'%{key} = fmul float %{key}raw, 0.125')
      if phase == 1:
        val = f'%{tag}s{e}_0'
        for n in range(1, 4):
          emit(f'%max{e}_{n} = call float @llvm.maxnum.f32(float {val}, float %{tag}s{e}_{n})')
          val = f'%max{e}_{n}'
        maximum = reduce_max(f'maxwave{e}', val)
        emit(f'%mnew{e} = call float @llvm.maxnum.f32(float %m{e}, float {maximum})')
        continue
      for n in range(4):
        key = f'{tag}p{e}_{n}'
        emit(f'%{key}diff = fsub float %{tag}s{e}_{n}, %mnew{e}')
        emit(f'%{key}log = fmul float %{key}diff, 0x3FF7154760000000')
        emit(f'%{key}exp = call float @llvm.exp2.f32(float %{key}log)')
        if phase == 3:
          emit(f'%{key}prob = fdiv nsz arcp contract afn float %{key}exp, %lnew{e}')
          prob = bf16(key, f'%{key}prob')
          emit(f'%{key}idx = add i32 %pbase, {e * stride + n * 16}')
          emit(f'%{key}ptr = getelementptr i16, ptr addrspace(3) @lds, i32 %{key}idx')
          emit(f'store i16 {prob}, ptr addrspace(3) %{key}ptr, align 2')
      if phase == 2:
        val = f'%l{e}'
        for n in range(4):
          parts = [shuffle(f'quad{e}_{n}_{j}', f'%{tag}p{e}_{n}exp', 28 | (j << 5)) for j in range(4)]
          total_sum = parts[0]
          for j in range(1, 4):
            emit(f'%qsum{e}_{n}_{j} = fadd float {total_sum}, {parts[j]}')
            total_sum = f'%qsum{e}_{n}_{j}'
          for j in range(4):
            part = shuffle(f'part{e}_{n}_{j}', total_sum, 16 | (j * 4 << 5))
            key = f'lnew{e}' if n == j == 3 else f'ordered{e}_{n}_{j}'
            emit(f'%{key} = fadd float {val}, {part}')
            val = f'%{key}'
    if phase == 3:
      for r in range(4):
        emit(f'%vi{r} = add i32 %t3ki, {r * 16 * 64}')
        emit(f'%vp{r} = getelementptr i16, ptr addrspace(1) %v, i32 %vi{r}')
        emit(f'%vv{r} = load <8 x i16>, ptr addrspace(1) %vp{r}, align 16')
        for e in range(8):
          key = f'vt{r}_{e}'
          emit(f'%{key}col = add i32 %loadcol, {e}')
          emit(f'%{key}off0 = mul i32 %{key}col, {stride}')
          emit(f'%{key}off1 = add i32 %{key}off0, %loadrow')
          emit(f'%{key}off = add i32 %{key}off1, {area + r * 16}')
          emit(f'%{key}ptr = getelementptr i16, ptr addrspace(3) @lds, i32 %{key}off')
          emit(f'%{key}val = extractelement <8 x i16> %vv{r}, i32 {e}')
          emit(f'store i16 %{key}val, ptr addrspace(3) %{key}ptr, align 2')
      emit('call void @llvm.amdgcn.s.barrier()')
      emit('%pmrow = add i32 %qm1, %lm')
      emit(f'%pmoff0 = mul i32 %pmrow, {stride}')
      emit('%pmoff = add i32 %pmoff0, %khalf')
      for ik in range(4):
        emit(f'%pi{ik} = add i32 %pmoff, {ik * 16}')
        emit(f'%pp{ik} = getelementptr i16, ptr addrspace(3) @lds, i32 %pi{ik}')
        fragment(f'pfrag{ik}', f'%pp{ik}', 3)
        for n in range(4):
          key = f'pv{ik}_{n}'
          emit(f'%{key}idx = add i32 %t3kn, {area + n * 16 * stride + ik * 16}')
          emit(f'%{key}ptr = getelementptr i16, ptr addrspace(3) @lds, i32 %{key}idx')
          fragment(f'{key}val', f'%{key}ptr', 3)
          prev = f'%acc{n}' if ik == 0 else f'%pv{ik - 1}_{n}'
          emit(f'%{key}dot = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8bf16(' +
               f'<8 x i16> %pfrag{ik}, <8 x i16> %{key}val, <8 x float> zeroinitializer)')
          emit(f'%{key} = fadd <8 x float> {prev}, %{key}dot')
    emit('call void @llvm.amdgcn.s.barrier()')
    emit(f'%{tag}next = add i32 %{tag}kk, 64')
    emit(f'%{tag}more = icmp ult i32 %{tag}next, %kend')
    emit(f'br i1 %{tag}more, label %pass{phase}, label %' + (f'between{phase}' if phase < 3 else 'exit'))
    if phase < 3:
      emit(f'between{phase}:')
      emit(f'br label %pass{phase + 1}')
  emit('exit:')
  for n in range(4):
    for e in range(8):
      key = f'out{n}_{e}'
      emit(f'%{key}val = extractelement <8 x float> %pv3_{n}, i32 {e}')
      result = bf16(key, f'%{key}val')
      emit(f'%{key}idx = add i32 %outoff, {n * 16 + e * 64}')
      emit(f'%{key}ptr = getelementptr i16, ptr addrspace(1) %o, i32 %{key}idx')
      emit(f'store i16 {result}, ptr addrspace(1) %{key}ptr, align 2')
  lines += ['ret void', '}', 'attributes #0 = { nounwind "amdgpu-flat-work-group-size"="128,128" "no-trapping-math"="true" }']
  src = '\n'.join(lines)
  return name, src, Device[device].renderer.compiler.compile_cached(src)


def _attention_kernel(out, q, k, v, start_frame):
  heads, seq, dim = q.shape
  total = k.shape[1]
  assert dim == 64 and seq % 128 == total % 128 == 0 and total == seq + start_frame * 128
  name, src, lib = _attention_program(seq, total, start_frame, q.device)
  flops = 8 * heads * dim * sum(128 * (128 * (start_frame + i + 1)) for i in range(seq // 128))
  sink = UOp.sink(out.base, q.base, k.base, v.base, UOp.special(seq // 64, 'gidx0'),
                  UOp.special(heads, 'gidx1'), UOp.special(128, 'lidx0'),
                  arg=KernelInfo(name=name, estimates=Estimates(ops=flops, mem=sum(math.prod(a.shape) for a in (q, k, v, out)) * 2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))


def block_causal_attention(q, k, v, start_frame):
  batch, heads, seq, dim = q.shape
  out = Tensor.empty(batch * heads, seq, dim, dtype=dtypes.bfloat16, device=q.device)
  return out.custom_kernel(q.reshape(batch * heads, seq, dim), k.reshape(batch * heads, -1, dim), v.reshape(batch * heads, -1, dim),
                          fxn=partial(_attention_kernel, start_frame=start_frame))[0].reshape(batch, heads, seq, dim)
