import json
import math
from pathlib import Path

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.nn.state import safe_load
from runtime import upload


class EdgeModel:
  def __init__(self, checkpoint: str, dtype=dtypes.bfloat16):
    self.config = json.loads((Path(checkpoint) / 'config.json').read_text())
    index = json.loads((Path(checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
    self.weights = {}
    for shard in sorted(set(index.values())):
      for name, weight in safe_load(Path(checkpoint) / shard).items():
        self.weights[name] = upload(weight, dtype)
        self.synchronize()
      print(f'Loaded {shard}', flush=True)
    self.dtype = dtype
    self.vision_jits = {}
    self.language_jits = {}
    self.expert_jit = TinyJit(self.expert_step)

  def linear(self, x, name):
    return x.linear(self.weights[f'{name}.weight'].T, self.weights.get(f'{name}.bias'))

  def norm(self, x, name, rms=False):
    y = x.float().rmsnorm(eps=1e-5) if rms else x.float().layernorm(eps=1e-6)
    if rms:
      return y.cast(x.dtype) * self.weights[f'{name}.weight']
    return (y * self.weights[f'{name}.weight'].float() + self.weights[f'{name}.bias'].float()).cast(x.dtype)

  @staticmethod
  def attention(q, k, v, mask=None):
    if q.shape[1] != k.shape[1]:
      k, v = (t.repeat_interleave(q.shape[1] // k.shape[1], dim=1) for t in (k, v))
    scores = (q.matmul(k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)).float()
    if mask is not None:
      scores = scores + mask
    return scores.softmax(-1).cast(q.dtype).matmul(v)

  def vision_layer(self, x, i):
    prefix = f'vlm.model.visual.encoder.layers.{i}'
    h = self.norm(x, f'{prefix}.layer_norm1')
    q, k, v = (self.linear(h, f'{prefix}.self_attn.{n}_proj').reshape(*h.shape[:2], 16, 72).transpose(1, 2) for n in ('q', 'k', 'v'))
    h = self.attention(q, k, v).transpose(1, 2).reshape(x.shape)
    x = x + self.linear(h, f'{prefix}.self_attn.out_proj')
    h = self.linear(self.norm(x, f'{prefix}.layer_norm2'), f'{prefix}.mlp.fc1').gelu()
    return (x + self.linear(h, f'{prefix}.mlp.fc2')).realize()

  def vision(self, pixels, positions):
    x = (self.linear(pixels, 'vlm.model.visual.embeddings.patch_embedding') + positions).realize()
    for i in range(self.config['vlm_config']['vision_config']['num_hidden_layers']):
      key = (i, x.shape)
      if key not in self.vision_jits:
        self.vision_jits[key] = TinyJit(lambda h, i=i: self.vision_layer(h, i))
      x = self.vision_jits[key](x)
    x = self.norm(x, 'vlm.model.visual.post_layernorm')
    x = self.norm(x, 'vlm.model.projector.norm').reshape(-1, 4608)
    x = self.linear(x, 'vlm.model.projector.linear_fc1').gelu(approximate='none')
    return self.linear(x, 'vlm.model.projector.linear_fc2').realize()

  @staticmethod
  def rope(q, k, cos, sin):
    def rotate(x):
      half = x.shape[-1] // 2
      return (-x[..., half:]).cat(x[..., :half], dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin

  def text_layer(self, x, cos, sin, prefix, cache=None, mask=None):
    h = self.norm(x, f'{prefix}.input_layernorm', rms=True)
    q, k, v = (self.linear(h, f'{prefix}.self_attn.{n}_proj').reshape(1, h.shape[1], heads, 128).transpose(1, 2)
               for n, heads in (('q', 16), ('k', 8), ('v', 8)))
    q, k = self.rope(q, k, cos, sin)
    if cache is not None:
      k, v = (old.cat(new, dim=2) for old, new in zip(cache, (k, v), strict=True))
    h = self.attention(q, k, v, mask).transpose(1, 2).reshape(1, x.shape[1], 2048)
    x = x + self.linear(h, f'{prefix}.self_attn.o_proj')
    h = self.linear(self.norm(x, f'{prefix}.post_attention_layernorm', rms=True), f'{prefix}.mlp.fc1').relu().square()
    return (x + self.linear(h, f'{prefix}.mlp.fc2')).realize(), (k.realize(), v.realize())

  def language(self, x, cos, sin, caches=None):
    key = (x.shape[1], 0 if caches is None else caches[0][0].shape[2])
    if key not in self.language_jits:
      self.language_jits[key] = TinyJit(self._language)
    return self.language_jits[key](x, cos, sin, *([] if caches is None else [t for pair in caches for t in pair]))

  def _language(self, x, cos, sin, *flat_cache):
    caches = [flat_cache[i:i+2] for i in range(0, len(flat_cache), 2)] if flat_cache else None
    past = 0 if caches is None else caches[0][0].shape[2]
    length = x.shape[1]
    mask = (Tensor.arange(past + length).reshape(1, -1) > Tensor.arange(past, past + length).reshape(-1, 1)).where(-math.inf, 0)
    new_cache = []
    for i in range(self.config['vlm_config']['text_config']['num_hidden_layers']):
      x, cache = self.text_layer(x, cos, sin, f'vlm.model.language_model.layers.{i}', None if caches is None else caches[i], mask)
      new_cache.append(cache)
    x = self.norm(x[:, -1:], 'vlm.model.language_model.norm', rms=True)
    return self.linear(x, 'vlm.lm_head').realize(), new_cache

  def action_projection(self, actions, timestep):
    freqs = (Tensor.arange(10, dtype=dtypes.float32) * (math.log(100) / 9)).exp()
    def fourier(x):
      angles = x.unsqueeze(-1) * freqs * (2 * math.pi)
      return angles.sin().cat(angles.cos(), dim=-1) * math.sqrt(2)
    x = fourier(actions[..., 0]).cat(fourier(actions[..., 1]), fourier(timestep).reshape(1, 1, 20).expand(1, 64, 20), dim=-1)
    prefix = 'expert.action_in_proj.encoder.trunk'
    x = self.linear(x.cast(self.dtype), f'{prefix}.0').silu()
    x = self.linear(self.norm(x, f'{prefix}.2', rms=True), f'{prefix}.3').silu()
    x = self.linear(self.norm(x, f'{prefix}.5', rms=True), f'{prefix}.6')
    # The action projection uses torch LayerNorm's default epsilon.
    x = x.float().layernorm(eps=1e-5)
    return (x * self.weights['expert.action_in_proj.norm.weight'].float() + self.weights['expert.action_in_proj.norm.bias'].float()).cast(self.dtype)

  def expert_step(self, actions, timestep, cos, sin, *flat_cache):
    x = self.action_projection(actions, timestep)
    for i in range(self.config['expert_config']['llm_config']['num_hidden_layers']):
      x, _ = self.text_layer(x, cos, sin, f'expert.expert.layers.{i}', flat_cache[2*i:2*i+2])
    x = self.norm(x, 'expert.expert.norm', rms=True)
    return self.linear(x, 'expert.action_out_proj').float().realize()

  def synchronize(self):
    Device[Device.DEFAULT].synchronize()
    dev = Device[Device.DEFAULT]
    if hasattr(dev, 'error_state') and dev.error_state.host.view(fmt='q')[0]:
      raise RuntimeError('eGPU reported an execution error')
