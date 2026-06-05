import math, unittest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, graph_rewrite

_strip_unique_pm = PatternMatcher([
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE, name="d")), name="b"), lambda b,d: b.replace(src=(d,))),
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE, name="d")), name="b"), lambda b,d: b.replace(src=(UOp.unique(0), d))),
])
def _strip_unique(u: UOp) -> UOp: return graph_rewrite(u, _strip_unique_pm)

def _t(*shape):
  return Tensor.arange(math.prod(shape)).reshape(*shape)

# Tensor().func().uop should be the same as UOp.func()
def _check(tc: unittest.TestCase, t: Tensor, fn):
  tc.assertIs(fn(t).uop, fn(t.uop), f"\ntensor.uop = {fn(t).uop}\nuop = {fn(t.uop)}")

class TestTensorUOpBinop(unittest.TestCase):
  # Tensor's binop upcasts mixed dtypes via least_upper_dtype + explicit CAST; UOp should match.
  def test_mul_float_int(self):
    t = _t(3).float()
    self.assertIs(_strip_unique((t * Tensor.arange(3)).uop), _strip_unique(t.uop * UOp.arange(3)))
  def test_mul_bool_int(self):
    t = _t(3)
    self.assertIs(_strip_unique((t.eq(1) * Tensor.arange(3)).uop), _strip_unique(t.uop.eq(1) * UOp.arange(3)))
  # Tensor's ufix picks float dtype when scalar is float and self is int; UOp should match.
  def test_add_scalar_float_on_int(self): _check(self, _t(3), lambda x: x + 1.5)
  # div: Tensor.div (default case) delegates to ElementwiseMixin.div; trees must match for Tensor and UOp.
  def test_div_tensor_by_tensor(self):
    a, b = _t(4).float(), _t(4).float() + 1
    self.assertIs(_strip_unique((a/b).uop), _strip_unique(a.uop/b.uop))
  def test_div_int_by_int(self):                 _check(self, _t(4), lambda x: x / 3)
  def test_div_sum_by_sum(self):                 _check(self, _t(4).float(), lambda x: x.sum() / (x + 1).sum())
  def test_div_broadcast_tensor_by_tensor(self):
    a, b = _t(3, 4).float(), _t(4).float() + 1
    self.assertIs(_strip_unique((a/b).uop), _strip_unique(a.uop/b.uop))
  # isclose used `self == other` which is Python identity on UOp (not elementwise); now uses .eq().
  def test_isclose(self):
    t = _t(4).float()
    self.assertIs(_strip_unique(t.isclose(t).uop), _strip_unique(t.uop.isclose(t.uop)))
  # __floordiv__/mod/fmod and div(rounding_mode=...) dispatch on dtype in mixin
  def test_floordiv_int(self):   _check(self, _t(4), lambda x: x // 3)
  def test_floordiv_float(self): _check(self, _t(4).float() + 1.5, lambda x: x // 2.0)
  def test_rfloordiv_int(self):  _check(self, _t(4)+1, lambda x: 7 // x)
  def test_mod_int(self):        _check(self, _t(4), lambda x: x % 3)
  def test_mod_float(self):      _check(self, _t(4).float() + 1.5, lambda x: x % 2.0)
  def test_div_trunc_int(self):  _check(self, _t(4), lambda x: x.div(3, rounding_mode="trunc"))
  def test_div_trunc_float(self):_check(self, _t(4).float() + 1.5, lambda x: x.div(2.0, rounding_mode="trunc"))
  def test_fmod_int(self):       _check(self, _t(4), lambda x: x.fmod(3))
  def test_fmod_float(self):     _check(self, _t(4).float() + 1.5, lambda x: x.fmod(2.0))
  def test_floordiv_bool(self):  _check(self, _t(4).cast(dtypes.bool), lambda x: x // True)
  def test_mod_bool(self):       _check(self, _t(4).cast(dtypes.bool), lambda x: x % True)
  def test_fmod_bool(self):      _check(self, _t(4).cast(dtypes.bool), lambda x: x.fmod(True))

class TestTensorUOpClone(unittest.TestCase):
  def test_clone(self):
    t = _t(3, 4).float()
    self.assertIs(_strip_unique(t.clone().uop), _strip_unique(t.uop.clone()))
  def test_clone_deviceless_const(self):
    u = UOp.const(dtypes.float, 2.0)
    self.assertIs(_strip_unique(Tensor(u).clone().uop), _strip_unique(u.clone()))

class TestTensorUOpGetitem(unittest.TestCase):
  # ---- pure slice patterns ----
  def test_slice_full(self):           _check(self, _t(4), lambda x: x[slice(None)])
  def test_slice_positive(self):       _check(self, _t(8), lambda x: x[1:5])
  def test_slice_open_start(self):     _check(self, _t(8), lambda x: x[:5])
  def test_slice_open_stop(self):      _check(self, _t(8), lambda x: x[3:])
  def test_slice_negative_start(self): _check(self, _t(8), lambda x: x[-3:])
  def test_slice_negative_stop(self):  _check(self, _t(8), lambda x: x[:-2])
  def test_slice_both_negative(self):  _check(self, _t(8), lambda x: x[-5:-1])

  # ---- slice with stride ----
  def test_slice_stride(self):                  _check(self, _t(6), lambda x: x[::2])
  def test_slice_start_stop_stride(self):       _check(self, _t(6), lambda x: x[1:5:2])
  def test_slice_reverse(self):                 _check(self, _t(6), lambda x: x[::-1])
  def test_slice_singleton_negative_step(self): _check(self, _t(8), lambda x: x[3:2:-1])

  # ---- empty / out-of-bounds slice ----
  def test_slice_empty(self):    _check(self, _t(6), lambda x: x[3:1])
  def test_slice_oob_stop(self): _check(self, _t(6), lambda x: x[0:100])

  # ---- single int (reduces a dim) ----
  def test_int_positive(self): _check(self, _t(8), lambda x: x[3])
  def test_int_negative(self): _check(self, _t(8), lambda x: x[-1])

  # ---- ellipsis ----
  def test_ellipsis_only(self):       _check(self, _t(2, 3, 4), lambda x: x[...])
  def test_ellipsis_then_int(self):   _check(self, _t(2, 3, 4), lambda x: x[..., -1])
  def test_ellipsis_then_slice(self): _check(self, _t(2, 3, 4), lambda x: x[..., 1:3])
  def test_ellipsis_then_none(self):  _check(self, _t(2, 3), lambda x: x[..., None])

  # ---- None (unsqueeze) ----
  def test_none_front(self):    _check(self, _t(4), lambda x: x[None])
  def test_none_back(self):     _check(self, _t(4), lambda x: x[:, None])
  def test_none_middle(self):   _check(self, _t(2, 3), lambda x: x[:, None, :])
  def test_multiple_none(self): _check(self, _t(2, 3), lambda x: x[None, :, None])

  # ---- mixed multi-dim ----
  def test_int_then_slice(self):    _check(self, _t(2, 3), lambda x: x[1, :])
  def test_multi_int(self):         _check(self, _t(2, 3, 4), lambda x: x[1, 2])
  def test_mixed_slice_int(self):   _check(self, _t(2, 3, 4), lambda x: x[0:2, -1, 1:3])
  def test_mixed_slice_slice(self): _check(self, _t(3, 4, 5), lambda x: x[1:3, :, 0:2])
  def test_high_rank_combo(self):   _check(self, _t(4, 5, 6), lambda x: x[1:3, :, -1, None])

class TestTensorUOpCumalu(unittest.TestCase):
  def test_cumsum_1d(self):       _check(self, _t(5), lambda x: x.cumsum())
  def test_cumsum_2d(self):       _check(self, _t(3, 4), lambda x: x.cumsum(1))
  def test_cumsum_non_last(self): _check(self, _t(3, 4), lambda x: x.cumsum(0))
  def test_cumsum_large(self):    _check(self, _t(600), lambda x: x.cumsum())  # exercises _split_cumalu
  def test_cumprod(self):         _check(self, _t(4), lambda x: x.cumprod(0))

class TestTensorUOpCumMinMax(unittest.TestCase):
  def _check_pair(self, t, fn):
    vt, it = fn(t)
    vu, iu = fn(t.uop)
    self.assertIs(_strip_unique(vt.uop), _strip_unique(vu))
    self.assertIs(_strip_unique(it.uop), _strip_unique(iu))
  def test_cummax_1d(self):    self._check_pair(_t(5), lambda x: x.cummax(0))
  def test_cummax_2d(self):    self._check_pair(_t(3, 4), lambda x: x.cummax(1))
  def test_cummax_0d(self):    self._check_pair(_t(1).reshape(()), lambda x: x.cummax(0))
  def test_cummin_1d(self):    self._check_pair(_t(5), lambda x: x.cummin(0))
  def test_cummin_2d(self):    self._check_pair(_t(3, 4), lambda x: x.cummin(1))

class TestTensorUOpArgMinMax(unittest.TestCase):
  def _check_stripped(self, t, fn): self.assertIs(_strip_unique(fn(t).uop), _strip_unique(fn(t.uop)))
  def test_argmax(self):       self._check_stripped(_t(3, 4), lambda x: x.argmax(axis=1))
  def test_argmax_flat(self):  self._check_stripped(_t(3, 4), lambda x: x.argmax())
  def test_argmin(self):       self._check_stripped(_t(3, 4), lambda x: x.argmin(axis=0))

class TestTensorUOpSequential(unittest.TestCase):
  def test_sequential(self): _check(self, _t(4), lambda x: x.sequential([lambda y: y * 2, lambda y: y + 1]))

class TestTensorUOpOneHot(unittest.TestCase):
  def test_one_hot(self):
    t = _t(5)
    self.assertIs(_strip_unique(t.one_hot(5).uop), _strip_unique(t.uop.one_hot(5)))

class TestTensorUOpSort(unittest.TestCase):
  def _check(self, t, **kw):
    tv, ti = t.sort(**kw)
    uv, ui = t.uop.sort(**kw)
    self.assertIs(_strip_unique(tv.uop), _strip_unique(uv))
    self.assertIs(_strip_unique(ti.uop), _strip_unique(ui))
  def test_sort_1d(self):         self._check(Tensor([0.5, 0.1, 0.3]).float())
  def test_sort_descending(self): self._check(Tensor([0.5, 0.1, 0.3]).float(), descending=True)
  def test_sort_2d(self):         self._check(_t(2, 4).float())
  def test_sort_single(self):     self._check(Tensor([1.0]).float())
  def test_argsort(self):
    t = Tensor([0.5, 0.1, 0.3]).float()
    self.assertIs(_strip_unique(t.argsort().uop), _strip_unique(t.uop.argsort()))
  def test_topk(self):
    t = _t(2, 4).float()
    tv, ti = t.topk(2)
    uv, ui = t.uop.topk(2)
    self.assertIs(_strip_unique(tv.uop), _strip_unique(uv))
    self.assertIs(_strip_unique(ti.uop), _strip_unique(ui))

class TestTensorUOpAllclose(unittest.TestCase):
  def test_allclose(self):
    a, b = _t(4).float(), _t(4).float()
    self.assertIs(_strip_unique(a.allclose(b).uop), _strip_unique(a.uop.allclose(b.uop)))

class TestTensorUOpBitcast(unittest.TestCase):
  def test_bitcast_same_dtype(self): _check(self, _t(4).float(), lambda x: x.bitcast(dtypes.float32))

class TestTensorUOpRand(unittest.TestCase):
  def test_random_bits(self):
    k = UOp.empty((2,), dtype=dtypes.uint32)
    c = UOp.zeros(2, dtype=dtypes.uint32)
    for num in (1, 4, 7, 1024):
      self.assertIs(_strip_unique(Tensor.random_bits(Tensor(k), Tensor(c), num).uop),
                    _strip_unique(UOp.random_bits(k, c, num)))
  def test_bits_to_rand_float32(self):
    bits_uop = UOp.empty((8,), dtype=dtypes.uint32)
    for shape in ((8,), (2, 4), (5,)):
      self.assertIs(_strip_unique(Tensor._bits_to_rand(Tensor(bits_uop), shape, dtypes.float32).uop),
                    _strip_unique(UOp._bits_to_rand(bits_uop, shape, dtypes.float32)))

class TestTensorUOpGather(unittest.TestCase):
  def _check(self, t, dim, idx):
    self.assertIs(_strip_unique(t.gather(dim, idx).uop), _strip_unique(t.uop.gather(dim, idx.uop)))
  def test_gather_1d(self):  self._check(_t(5), 0, Tensor([2, 1, 0, 1, 2], dtype=dtypes.int32))
  def test_gather_dim0(self): self._check(_t(3, 4), 0, Tensor([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], dtype=dtypes.int32))
  def test_gather_dim1(self): self._check(_t(3, 4), 1, Tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]], dtype=dtypes.int32))

class TestTensorUOpInterpolate(unittest.TestCase):
  def _check(self, t, mode):
    self.assertIs(_strip_unique(t.interpolate(size=(2, 2), mode=mode).uop),
                  _strip_unique(t.uop.interpolate(size=(2, 2), mode=mode)))
  def test_interpolate_nearest(self):       self._check(_t(1, 1, 4, 4).float(), "nearest")
  def test_interpolate_nearest_exact(self): self._check(_t(1, 1, 4, 4).float(), "nearest-exact")
  def test_interpolate_linear(self):        self._check(_t(1, 1, 4, 4).float(), "linear")

class TestTensorUOpLoss(unittest.TestCase):
  def test_cross_entropy(self):
    t, Y = _t(2, 3).float(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.cross_entropy(Y).uop), _strip_unique(t.uop.cross_entropy(Y.uop)))
  def test_sparse_categorical_crossentropy(self):
    t, Y = _t(2, 3).float(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.sparse_categorical_crossentropy(Y).uop), _strip_unique(t.uop.sparse_categorical_crossentropy(Y.uop)))
  def test_sparse_categorical_crossentropy_ignore_index(self):
    t, Y = _t(2, 3).float(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.sparse_categorical_crossentropy(Y, ignore_index=0).uop),
                  _strip_unique(t.uop.sparse_categorical_crossentropy(Y.uop, ignore_index=0)))
  def test_nll_loss(self):
    t, Y = _t(2, 3).float().log_softmax(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.nll_loss(Y).uop), _strip_unique(t.uop.nll_loss(Y.uop)))
  def test_nll_loss_weight(self):
    t, Y, w = _t(2, 3).float().log_softmax(), Tensor([1, 2], dtype=dtypes.int32), _t(3).float()
    self.assertIs(_strip_unique(t.nll_loss(Y, weight=w).uop), _strip_unique(t.uop.nll_loss(Y.uop, weight=w.uop)))
  def test_nll_loss_ignore_index(self):
    t, Y = _t(2, 3).float().log_softmax(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.nll_loss(Y, ignore_index=1).uop), _strip_unique(t.uop.nll_loss(Y.uop, ignore_index=1)))
  def test_nll_loss_none_reduction(self):
    t, Y = _t(2, 3).float().log_softmax(), Tensor([1, 2], dtype=dtypes.int32)
    self.assertIs(_strip_unique(t.nll_loss(Y, reduction="none").uop), _strip_unique(t.uop.nll_loss(Y.uop, reduction="none")))
  def test_nll_loss_weight_ignore_index(self):
    t, Y, w = _t(2, 3).float().log_softmax(), Tensor([1, 2], dtype=dtypes.int32), _t(3).float()
    self.assertIs(_strip_unique(t.nll_loss(Y, weight=w, ignore_index=1).uop),
                  _strip_unique(t.uop.nll_loss(Y.uop, weight=w.uop, ignore_index=1)))

class TestTensorUOpScatter(unittest.TestCase):
  def test_scatter(self):
    x, idx, src = _t(3, 4).float(), Tensor([[0, 1, 2, 0]], dtype=dtypes.int32), _t(1, 4).float()
    self.assertIs(_strip_unique(x.scatter(0, idx, src).uop), _strip_unique(x.uop.scatter(0, idx.uop, src.uop)))
  def test_scatter_scalar_src(self):
    x, idx = _t(3, 4).float(), Tensor([[0, 1]], dtype=dtypes.int32)
    self.assertIs(_strip_unique(x.scatter(1, idx, 3.14).uop), _strip_unique(x.uop.scatter(1, idx.uop, 3.14)))
  # inf cannot be cast to int — this regresses if scalar src is routed through index.dtype first
  def test_scatter_inf_src(self):
    x, idx = _t(3, 4).float(), Tensor([[0, 1]], dtype=dtypes.int32)
    self.assertIs(_strip_unique(x.scatter(1, idx, float("inf")).uop),
                  _strip_unique(x.uop.scatter(1, idx.uop, float("inf"))))
  def test_scatter_add(self):
    x, idx = _t(3, 4).float(), Tensor([[0, 1]], dtype=dtypes.int32)
    self.assertIs(_strip_unique(x.scatter(1, idx, 3.14, reduce="add").uop),
                  _strip_unique(x.uop.scatter(1, idx.uop, 3.14, reduce="add")))
  def test_scatter_multiply(self):
    x, idx = _t(3, 4).float(), Tensor([[0, 1]], dtype=dtypes.int32)
    self.assertIs(_strip_unique(x.scatter(1, idx, 3.14, reduce="multiply").uop),
                  _strip_unique(x.uop.scatter(1, idx.uop, 3.14, reduce="multiply")))
  # tensor src with reduce hits the "elif reduce: raise" branch in both Tensor and UOp paths
  def test_scatter_tensor_src_with_reduce_raises(self):
    x, idx, src = _t(3, 4).float(), Tensor([[0, 1]], dtype=dtypes.int32), _t(1, 2).float()
    with self.assertRaises(TypeError): x.scatter(1, idx, src, reduce="add")
    with self.assertRaises(TypeError): x.uop.scatter(1, idx.uop, src.uop, reduce="add")

class TestTensorUOpScatterReduce(unittest.TestCase):
  def _check(self, x, idx, src, **kw):
    self.assertIs(_strip_unique(x.scatter_reduce(0, idx, src, **kw).uop),
                  _strip_unique(x.uop.scatter_reduce(0, idx.uop, src.uop, **kw)))
  def test_sum(self):  self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="sum")
  def test_prod(self): self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="prod")
  def test_mean(self): self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="mean")
  def test_amax(self): self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="amax")
  def test_amin(self): self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="amin")
  def test_mean_exclude_self(self):
    self._check(_t(3, 4).float(), Tensor([[0, 1, 0, 1]]*3, dtype=dtypes.int32), Tensor.ones(3, 4).float(), reduce="mean", include_self=False)

class TestTensorUOpPool(unittest.TestCase):
  def test_avg_pool2d(self):                _check(self, _t(1, 1, 5, 5).float(), lambda x: x.avg_pool2d())
  def test_avg_pool2d_padding(self):        _check(self, _t(1, 1, 5, 5).float(), lambda x: x.avg_pool2d(padding=1))
  def test_avg_pool2d_ceil(self):           _check(self, _t(1, 1, 5, 5).float(), lambda x: x.avg_pool2d(ceil_mode=True))
  def test_avg_pool2d_no_count_pad(self):   _check(self, _t(1, 1, 5, 5).float(), lambda x: x.avg_pool2d(padding=1, count_include_pad=False))
  def test_max_pool2d(self):                _check(self, _t(1, 1, 5, 5).float(), lambda x: x.max_pool2d())
  def test_max_pool2d_padding(self):        _check(self, _t(1, 1, 5, 5).float(), lambda x: x.max_pool2d(padding=1))
  def test_max_pool2d_ceil(self):           _check(self, _t(1, 1, 5, 5).float(), lambda x: x.max_pool2d(ceil_mode=True))
  def test_max_pool2d_return_indices(self):
    t = _t(1, 1, 5, 5).float()
    vt, it = t.max_pool2d(return_indices=True)
    vu, iu = t.uop.max_pool2d(return_indices=True)
    self.assertIs(_strip_unique(vt.uop), _strip_unique(vu))
    self.assertIs(_strip_unique(it.uop), _strip_unique(iu))
  def test_max_unpool2d(self):
    t = _t(1, 1, 4, 4).float()
    out, idx = t.max_pool2d(return_indices=True)
    self.assertIs(_strip_unique(out.max_unpool2d(idx).uop), _strip_unique(out.uop.max_unpool2d(idx.uop)))

class TestTensorUOpCat(unittest.TestCase):
  def test_cat_dim0(self):     _check(self, _t(2, 3), lambda x: x.cat(x, dim=0))
  def test_cat_dim1(self):     _check(self, _t(2, 3), lambda x: x.cat(x, dim=1))
  def test_cat_3tensors(self): _check(self, _t(2, 3), lambda x: x.cat(x, x, dim=0))
  def test_cat_neg_dim(self):  _check(self, _t(2, 3, 4), lambda x: x.cat(x, dim=-1))

class TestTensorUOpPad(unittest.TestCase):
  def test_pad_flat(self):               _check(self, _t(4, 5), lambda x: x.pad((1, 2, 0, 3)))
  def test_pad_flat_negative(self):      _check(self, _t(4, 5), lambda x: x.pad((1, -1, 0, 2), value=-1.0))
  def test_pad_grouped_none(self):       _check(self, _t(4, 5), lambda x: x.pad((None, (0, 3))))
  def test_pad_circular(self):           _check(self, _t(4, 5), lambda x: x.pad(((1, 2), (0, 3)), mode="circular"))
  def test_pad_circular_zero_after(self):_check(self, _t(4, 5), lambda x: x.pad(((1, 0), (2, 0)), mode="circular"))
  def test_pad_reflect(self):            _check(self, _t(4, 5), lambda x: x.pad(((1, 2), (0, 3)), mode="reflect"))
  def test_pad_reflect_negative(self):   _check(self, _t(4, 5), lambda x: x.pad(((1, -1), (0, 2)), mode="reflect"))
  def test_pad_replicate(self):          _check(self, _t(4, 5), lambda x: x.pad(((1, 2), (0, 3)), mode="replicate"))
  def test_pad_replicate_negative(self): _check(self, _t(4, 5), lambda x: x.pad(((1, -1), (0, 2)), mode="replicate"))

class TestTensorUOpStack(unittest.TestCase):
  def test_stack_dim0(self):     _check(self, _t(2, 3), lambda x: x.stack(x, dim=0))
  def test_stack_dim1(self):     _check(self, _t(2, 3), lambda x: x.stack(x, dim=1))
  def test_stack_3tensors(self): _check(self, _t(2, 3), lambda x: x.stack(x, x, dim=0))
  def test_stack_new_last(self): _check(self, _t(2, 3), lambda x: x.stack(x, dim=-1))

class TestTensorUOpConv2d(unittest.TestCase):
  def test_conv2d_basic(self):
    w = _t(1, 1, 2, 2).float()
    _check(self, _t(1, 1, 3, 3).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop))
  def test_conv2d_padded(self):
    w = _t(1, 1, 2, 2).float()
    _check(self, _t(1, 1, 3, 3).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop, padding=1))
  def test_conv2d_negative_padding(self):
    w = _t(1, 1, 3, 3).float()
    _check(self, _t(1, 1, 5, 5).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop, padding=(-1,-1,-1,-1)))
  def test_conv2d_multichannel_bias(self):
    w, b = _t(4, 2, 3, 3).float(), _t(4).float()
    _check(self, _t(2, 2, 5, 5).float(), lambda x: x.conv2d(*(y if isinstance(x, Tensor) else y.uop for y in (w, b))))
  def test_conv2d_stride_dilation(self):
    w = _t(2, 2, 2, 2).float()
    _check(self, _t(1, 2, 6, 6).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop, stride=2, dilation=2))
  def test_conv2d_groups(self):
    w = _t(4, 1, 2, 2).float()
    _check(self, _t(1, 4, 4, 4).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop, groups=4))
  def test_conv2d_3d(self):
    w = _t(1, 1, 2, 2, 2).float()
    _check(self, _t(1, 1, 3, 3, 3).float(), lambda x: x.conv2d(w if isinstance(x, Tensor) else w.uop))
  def test_conv_transpose2d_basic(self):
    w = _t(1, 1, 2, 2).float()
    _check(self, _t(1, 1, 3, 3).float(), lambda x: x.conv_transpose2d(w if isinstance(x, Tensor) else w.uop))
  def test_conv_transpose2d_stride(self):
    w = _t(1, 1, 2, 2).float()
    _check(self, _t(1, 1, 3, 3).float(), lambda x: x.conv_transpose2d(w if isinstance(x, Tensor) else w.uop, stride=2))

class TestTensorUOpEinsum(unittest.TestCase):
  def test_einsum_dot(self):       _check(self, _t(2, 3), lambda x: type(x).einsum("ij,ij->", x, x))
  def test_einsum_transpose(self): _check(self, _t(2, 3), lambda x: type(x).einsum("ij->ji", x))

class TestTensorUOpSoftmax(unittest.TestCase):
  def test_softmax_default(self):     _check(self, _t(2, 3).float(), lambda x: x.softmax())
  def test_softmax_axis0(self):       _check(self, _t(2, 3).float(), lambda x: x.softmax(axis=0))
  def test_log_softmax_default(self): _check(self, _t(2, 3).float(), lambda x: x.log_softmax())
  def test_log_softmax_axis0(self):   _check(self, _t(2, 3).float(), lambda x: x.log_softmax(axis=0))

class TestTensorUOpQR(unittest.TestCase):
  def _check(self, t):
    qt, rt = t.qr()
    qu, ru = t.uop.qr()
    self.assertIs(_strip_unique(qt.uop), _strip_unique(qu))
    self.assertIs(_strip_unique(rt.uop), _strip_unique(ru))
  def test_qr_square(self):   self._check(_t(3, 3).float())
  def test_qr_tall(self):     self._check(_t(4, 3).float())
  def test_qr_wide(self):     self._check(_t(3, 4).float())
  def test_qr_zero_col(self): self._check(Tensor([[0.0, 1.0], [0.0, 2.0]]))
  def test_qr_batched(self):  self._check(_t(2, 3, 3).float())

class TestTensorUOpSVD(unittest.TestCase):
  def _check(self, t, **kw):
    ut, st, vt = t.svd(**kw)
    uu, su, vu = t.uop.svd(**kw)
    self.assertIs(_strip_unique(ut.uop), _strip_unique(uu))
    self.assertIs(_strip_unique(st.uop), _strip_unique(su))
    self.assertIs(_strip_unique(vt.uop), _strip_unique(vu))
  def test_svd_square(self):    self._check(_t(2, 2).float())
  def test_svd_tall(self):      self._check(_t(3, 2).float())
  def test_svd_wide(self):      self._check(_t(2, 3).float())
  def test_svd_odd_num(self):   self._check(_t(3, 3).float())  # exercises odd-num runoff path
  def test_svd_batched(self):   self._check(_t(2, 2, 2).float())
  def test_svd_nonfull(self):   self._check(_t(3, 2).float(), full_matrices=False)

# UOp.empty / UOp.empty_like are the canonical buffer allocators; Tensor.empty / Tensor.empty_like just forward.
class TestUOpEmpty(unittest.TestCase):
  def test_empty_dtype_string(self):
    self.assertEqual(UOp.empty((3, 4), dtype="float32").dtype, dtypes.float32)

  def test_empty_like_dtype_override(self):
    u = Tensor.ones(3, 4).uop.empty_like(dtype=dtypes.int8)
    self.assertEqual((u.shape, u.dtype), ((3, 4), dtypes.int8))
    self.assertTrue(u.has_buffer_identity())

  def test_empty_like_sharded_to_single_device(self):
    # regression: sharded source, override to single device must yield full logical shape with no axis
    t = Tensor.ones(8, 4).shard(("NULL:0", "NULL:1"), axis=0)
    for dev in ("NULL:2", ("NULL:2",)):  # singleton tuple also canonicalizes to single device
      u = t.uop.empty_like(device=dev, dtype=dtypes.int32)
      self.assertEqual((u.shape, u.device, u.dtype, u.axis), ((8, 4), "NULL:2", dtypes.int32, None))
      self.assertTrue(u.has_buffer_identity())

  def test_empty_direct_singleton_tuple_device(self):
    # regression: direct UOp.empty with a singleton-tuple device + axis must not trip .multi()'s tuple assert
    u = UOp.empty((4,), dtype=dtypes.float32, device=("NULL:0",), axis=0)
    self.assertEqual((u.shape, u.device, u.axis), ((4,), "NULL", None))

class TestTensorUOpCreation(unittest.TestCase):
  def test_full(self):
    self.assertIs(_strip_unique(Tensor.full((2, 3), 42).uop), _strip_unique(UOp.full((2, 3), 42)))
  def test_full_kwargs(self):
    self.assertIs(_strip_unique(Tensor.full((2, 3), 42, dtype=dtypes.int8, device="NULL").uop),
                  _strip_unique(UOp.full((2, 3), 42, dtype=dtypes.int8, device="NULL")))
  def test_full_symbolic_fill(self):
    # bound symbolic variable — flows through Tensor.__init__'s UOp branch, no UNIQUE added
    t = Tensor.full((2, 3), UOp.variable("x", 1, 10).bind(5))
    self.assertEqual(t.shape, (2, 3))
    self.assertFalse(t.uop.op_in_backward_slice_with_self(Ops.UNIQUE))
  def test_zeros(self):
    self.assertIs(_strip_unique(Tensor.zeros(2, 3).uop), _strip_unique(UOp.zeros(2, 3)))
  def test_ones(self):
    self.assertIs(_strip_unique(Tensor.ones(2, 3).uop), _strip_unique(UOp.ones(2, 3)))
  def test_invalids(self):
    self.assertIs(_strip_unique(Tensor.invalids(2, 3, dtype=dtypes.int8).uop), _strip_unique(UOp.invalids(2, 3, dtype=dtypes.int8)))
  def test_arange(self):
    self.assertIs(_strip_unique(Tensor.arange(5).uop), _strip_unique(UOp.arange(5)))
  def test_arange_empty(self):
    self.assertIs(_strip_unique(Tensor.arange(5, 5).uop), _strip_unique(UOp.arange(5, 5)))
  def test_arange_step(self):
    self.assertIs(_strip_unique(Tensor.arange(5, 10, 2).uop), _strip_unique(UOp.arange(5, 10, 2)))
  def test_linspace(self):
    self.assertIs(_strip_unique(Tensor.linspace(0, 10, 5).uop), _strip_unique(UOp.linspace(0, 10, 5)))
  def test_linspace_one_step(self):
    self.assertIs(_strip_unique(Tensor.linspace(5, 10, 1).uop), _strip_unique(UOp.linspace(5, 10, 1)))
  def test_eye(self):
    self.assertIs(_strip_unique(Tensor.eye(3).uop), _strip_unique(UOp.eye(3)))
  def test_eye_rect(self):
    self.assertIs(_strip_unique(Tensor.eye(2, 4).uop), _strip_unique(UOp.eye(2, 4)))
  def test_triu(self):
    t = _t(3, 4)
    self.assertIs(_strip_unique(t.triu().uop), _strip_unique(t.uop.triu()))
  def test_triu_diagonal(self):
    t = _t(3, 4)
    self.assertIs(_strip_unique(t.triu(diagonal=1).uop), _strip_unique(t.uop.triu(diagonal=1)))
  def test_tril(self):
    t = _t(3, 4)
    self.assertIs(_strip_unique(t.tril().uop), _strip_unique(t.uop.tril()))
  def test_tril_diagonal(self):
    t = _t(3, 4)
    self.assertIs(_strip_unique(t.tril(diagonal=-1).uop), _strip_unique(t.uop.tril(diagonal=-1)))

if __name__ == "__main__":
  unittest.main()
