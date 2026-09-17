import math
import numpy as np


def load_weights(path, layers, return_plan):
  from tinygrad import Device, Tensor, UOp, dtypes
  from tinygrad.nn.state import safe_load

  state = {k: v for k, v in safe_load(path).items()
           if not k.startswith("final_layer.") and (return_plan or not k.startswith("plan_head.")) and
           not (layers is not None and k.startswith("blocks.") and int(k.split(".")[1]) >= layers)}
  offsets, size = {}, 0
  for name, value in state.items():
    offsets[name] = size
    size += (value.nbytes() + 255) // 256 * 256
  # One allocation avoids rounding hundreds of individual weights up to GPU page sizes.
  packed = np.empty(size, dtype=np.uint8)
  for name, value in state.items():
    raw = value.reshape(-1).bitcast(dtypes.uint8).numpy()
    packed[offsets[name]:offsets[name] + value.nbytes()] = raw.ravel()
  data = Tensor.empty(size, device=Device.DEFAULT, dtype=dtypes.uint8).realize()
  chunk_size = 32 * 1024 * 1024
  for start in range(0, size, chunk_size):
    chunk = packed[start:start + chunk_size]
    value = Tensor(chunk, device="CPU").to(Device.DEFAULT)
    data[start:start + len(chunk)].assign(value).realize()
    if (start // chunk_size + 1) % 32 == 0:
      print(f"Uploaded {(start + len(chunk)) / 2**30:.0f} / {size / 2**30:.2f} GiB", flush=True)
  weights = {}
  for name, value in state.items():
    # Typed views let custom kernels use the packed allocation without copying weights.
    view = Tensor(UOp.from_buffer(data.uop.buffer.view(value.numel(), value.dtype, offsets[name])))
    weights[name] = view.reshape(value.shape)
  return weights


def fp8_mlp_projection(out, x, weight, scale, weight_scale, bias):
  from tinygrad import UOp, dtypes
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.uop.ops import AxisType, KernelInfo, Ops

  m, n = UOp.range(out.shape[0], 0), UOp.range(out.shape[1], 1)
  k = UOp.range(x.shape[1], 2, AxisType.REDUCE)
  value = (x[m, k] * weight[k, n]).cast(dtypes.float32).reduce(k, arg=Ops.ADD)
  value = value * (scale.reshape(1)[0] * weight_scale.reshape(1)[0]) + bias[n].cast(dtypes.float32)
  opts = (Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),
          Opt(OptOps.SPLIT, 1, (2, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (2, AxisType.LOCAL)),
          Opt(OptOps.SPLIT, 9, (2, AxisType.UNROLL)))
  return out[m, n].store(value.cast(out.dtype)).end(m, n).sink(
    arg=KernelInfo(name="worldmodel_fp8_mlp_projection", opts_to_apply=opts))


class WorldModel:
  def __init__(self, config, weights, layers=None, return_plan=True):
    from tinygrad import Tensor

    self.w, self.config, self.return_plan = weights, config, return_plan
    self.layers = config["transformer"]["n_layer"] if layers is None else layers
    assert config["patch_size"] == [1, 2, 2]
    assert config["transformer"]["norm"] == "RMSNorm" and not config["transformer"]["prenorm"]
    assert config["transformer"]["attention_mask"] == "BLOCKWISE_LOWER_TRIANGLE"
    assert not config["experimental_pose_only_xy"]
    frames, h, w = config["input_size"]
    self.spatial, self.frames = h * w // 4, frames
    self.kv_cache = None
    half = weights["t_embedder.mlp.0.weight"].shape[1] // 2
    self.freqs = (Tensor.arange(half).float() * (-math.log(10000) / half)).exp().realize()

  def linear(self, x, name):
    from tinygrad import Device, Tensor, dtypes

    # Materialize boundaries so concatenations and separate matmuls don't hide WMMA patterns.
    x = x.contiguous().realize()
    weight = self.w[name + ".weight"]
    if weight.dtype == dtypes.fp8e4m3:
      scale = (x.float().abs().max().clamp(min_=1e-12) / 448.0).realize()
      quantized = (x.float() / scale).clamp(-448, 448).cast(weight.dtype).realize()
      if (x.ndim == 3 and x.shape[:2] in {(1, 640), (1, 1280)} and name.startswith("blocks.") and
          getattr(Device[x.device], "arch", "") in {"gfx1200", "gfx1201"}):
        from openpilot.selfdrive.modeld.worldmodel_kernels import fp8_linear

        rows = x.shape[1]
        out = fp8_linear(quantized.reshape(rows, -1), weight, scale, self.w[name + ".weight_scale"], self.w[name + ".bias"])
        return out.reshape(1, rows, weight.shape[0]).realize()
      if (x.shape == (1, 128, 9216) and weight.shape == (2304, 9216) and
          getattr(Device[x.device], "arch", "") in {"gfx1200", "gfx1201"}):
        out = Tensor.empty(128, 2304, dtype=x.dtype, device=x.device)
        out = out.custom_kernel(quantized.reshape(128, 9216), weight.T, scale,
                                self.w[name + ".weight_scale"], self.w[name + ".bias"], fxn=fp8_mlp_projection)[0]
        return out.reshape(1, 128, 2304).realize()
      out = quantized.matmul(weight.T, dtype=dtypes.float32) * (scale * self.w[name + ".weight_scale"])
    else:
      out = x.matmul(weight.T, dtype=dtypes.float32)
    if (bias := self.w.get(name + ".bias")) is not None:
      out = out + bias.float()
    return out.cast(x.dtype).realize()

  def setup_cache(self, batch, frames=5):
    from tinygrad import Tensor, dtypes

    heads, width = self.config["transformer"]["n_head"], self.config["transformer"]["n_embd"]
    self.kv_cache = Tensor.empty(self.layers, 2, batch, heads, frames * self.spatial,
                                 width // heads, dtype=dtypes.fp8e4m3).realize()

  def norm(self, x, name=None, layernorm=False):
    y = x.float()
    if layernorm:
      y = y - y.mean(-1, keepdim=True)
    y = y * (y.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    if layernorm:
      return (y * self.w[name + ".weight"].float() + self.w[name + ".bias"].float()).cast(x.dtype)
    y = y.cast(x.dtype)
    return y if name is None else y * self.w[name + ".weight"]

  def embed(self, x, name, discrete=False):
    x = self.w[name + ".mlp.0.weight"][x] if discrete else self.linear(x, name + ".mlp.0")
    x = self.linear(x.float().silu().cast(x.dtype), name + ".mlp.2")
    x = x.float().silu().cast(x.dtype)
    return self.linear(x, name + ".to_t6.1"), None

  def modulate(self, x, shift, scale):
    return (x.reshape(x.shape[0], -1, self.spatial, x.shape[-1]) * (1 + scale) + shift).reshape(x.shape)

  def gate(self, x, value):
    return (x.reshape(x.shape[0], -1, self.spatial, x.shape[-1]) * value).reshape(x.shape)

  def mlp(self, x, name):
    x = self.linear(x, name + ".c_fc")
    return self.linear(x.float().gelu().cast(x.dtype), name + ".c_proj")

  def attention(self, x, name, layer, start_frame):
    from tinygrad import dtypes

    batch, seq, width = x.shape
    heads = self.config["transformer"]["n_head"]
    qkv = self.linear(x, name + ".c_attn").reshape(batch, seq, 3, heads, width // heads)
    q, k, v = (qkv[:, :, i] for i in range(3))
    q, k = self.norm(q, name + ".q_norm"), self.norm(k, name + ".k_norm")
    q, k, v = (a.transpose(1, 2) for a in (q, k, v))
    q, k, v = (a.contiguous().realize() for a in (q, k, v))
    if self.kv_cache is not None:
      if start_frame == 0:
        self.kv_cache[layer, 0].assign(k.cast(dtypes.fp8e4m3)).realize()
        self.kv_cache[layer, 1].assign(v.cast(dtypes.fp8e4m3)).realize()
        k, v = (self.kv_cache[layer, i] for i in range(2))
      else:
        k, v = (self.kv_cache[layer, i].cat(a.cast(dtypes.fp8e4m3), dim=2) for i, a in enumerate((k, v)))
      k, v = (a.cast(x.dtype).contiguous().realize() for a in (k, v))
    chunks = []
    # All spatial tokens in a frame attend to that frame and every earlier frame.
    query_chunk = 32 if self.kv_cache is not None and start_frame == 0 else self.spatial
    for start in range(0, seq, query_chunk):
      end = start + query_chunk
      kv_end = (start_frame + start // self.spatial + 1) * self.spatial
      scores = q[:, :, start:end].matmul(k[:, :, :kv_end].transpose(-1, -2), dtype=dtypes.float32)
      probs = (scores / math.sqrt(width // heads)).softmax(-1).cast(x.dtype)
      chunks.append(probs.matmul(v[:, :, :kv_end], dtype=dtypes.float32).cast(x.dtype).realize())
    y = chunks[0].cat(*chunks[1:], dim=2).transpose(1, 2).reshape(batch, seq, width)
    return self.linear(y, name + ".c_proj")

  def __call__(self, x, t, augments_pos_ref_augment, ref_augment_from_augments_euler, pose_mask, fidx,
               start_frame=0, return_plan=None):
    from tinygrad import Tensor

    batch, frames, channels, height, width = x.shape
    x = x.reshape(batch, frames, channels, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(batch, frames * self.spatial, channels * 4)
    pos = self.w["pos_embed"][:, start_frame * self.spatial:(start_frame + frames) * self.spatial]
    x = (self.norm(self.linear(x, "x_embedder.1"), "x_embedder.2") + pos).realize()
    args = t.float().unsqueeze(-1) * self.config["time_factor"] * self.freqs
    t6, _ = self.embed(args.cos().cat(args.sin(), dim=-1).cast(x.dtype), "t_embedder")
    for value, name, discrete in (
      (augments_pos_ref_augment * self.w["position_scale.scale"], "augments_pos_ref_augment_embedder", False),
      (ref_augment_from_augments_euler * self.w["euler_scale.scale"],
       "ref_augment_from_augments_euler_embedder", False),
      (pose_mask, "pose_mask_embedder", True),
      (fidx, "fidx_embedder", True),
    ):
      c6, _ = self.embed(value, name, discrete)
      t6 = t6 + c6
    t6.realize()
    for i in range(self.layers):
      name = f"blocks.{i}"
      shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
        self.w[name + ".scale_shift_table"][:, start_frame:start_frame + frames] + t6.reshape(batch, frames, 6, -1)
      ).chunk(6, dim=2)
      attn = self.attention(self.modulate(self.norm(x), shift_a, scale_a), name + ".attn", i, start_frame)
      x = (x + self.gate(attn, gate_a)).realize()
      x = (x + self.gate(self.mlp(self.modulate(self.norm(x), shift_m, scale_m), name + ".mlp"), gate_m)).realize()
    outputs = {}
    if (self.return_plan if return_plan is None else return_plan):
      plan = x[:, -1]
      for i in range(self.config["plan_head"]["n_layer"]):
        name = f"plan_head.mlps.{i}"
        plan = plan + self.mlp(self.norm(plan, name + ".layer_norm", layernorm=True), name)
      outputs["plan"] = self.linear(plan, "plan_head.head") * self.w["plan_head.scale_layer.scale"]
    if outputs:
      Tensor.realize(*outputs.values())
    return outputs
