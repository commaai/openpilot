from functools import cache
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
