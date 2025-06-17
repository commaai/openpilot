import json, logging, math, re, sys, time, wave, argparse, numpy as np
from phonemizer.phonemize import default_separator, _phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from functools import reduce
from pathlib import Path
from typing import List
from tinygrad import nn, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn.state import torch_load
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from unidecode import unidecode

LRELU_SLOPE = 0.1

class Synthesizer:
  def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, use_sdp=True, emotion_embedding=False, **kwargs):
    self.n_vocab, self.spec_channels, self.inter_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.segment_size, self.n_speakers, self.gin_channels, self.use_sdp = n_vocab, spec_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, segment_size, n_speakers, gin_channels, use_sdp
    self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, emotion_embedding)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels) if use_sdp else DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
    if n_speakers > 1: self.emb_g = nn.Embedding(n_speakers, gin_channels)
  def infer(self, x, x_lengths, sid=None, noise_scale=1.0, length_scale=1, noise_scale_w=1., max_len=None, emotion_embedding=None, max_y_length_estimate_scale=None, pad_length=-1):
    x, m_p, logs_p, x_mask = self.enc_p.forward(x.realize(), x_lengths.realize(), emotion_embedding.realize() if emotion_embedding is not None else emotion_embedding)
    g = self.emb_g(sid.reshape(1, 1)).squeeze(1).unsqueeze(-1) if self.n_speakers > 0 else None
    logw = self.dp.forward(x, x_mask.realize(), g=g.realize(), reverse=self.use_sdp, noise_scale=noise_scale_w if self.use_sdp else 1.0)
    w_ceil = Tensor.ceil(logw.exp() * x_mask * length_scale)
    y_lengths = Tensor.maximum(w_ceil.sum([1, 2]), 1).cast(dtypes.int64)
    return self.generate(g, logs_p, m_p, max_len, max_y_length_estimate_scale, noise_scale, w_ceil, x, x_mask, y_lengths, pad_length)
  def generate(self, g, logs_p, m_p, max_len, max_y_length_estimate_scale, noise_scale, w_ceil, x, x_mask, y_lengths, pad_length):
    max_y_length = y_lengths.max().item() if max_y_length_estimate_scale is None else max(15, x.shape[-1]) * max_y_length_estimate_scale
    y_mask = sequence_mask(y_lengths, max_y_length).unsqueeze(1).cast(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
    attn = generate_path(w_ceil, attn_mask)
    m_p_2 = attn.squeeze(1).matmul(m_p.transpose(1, 2)).transpose(1, 2)        # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p_2 = attn.squeeze(1).matmul(logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
    z_p = m_p_2 + Tensor.randn(*m_p_2.shape, dtype=m_p_2.dtype) * logs_p_2.exp() * noise_scale
    row_len = y_mask.shape[2]
    if pad_length > -1:
      # Pad flow forward inputs to enable JIT
      assert pad_length > row_len, "pad length is too small"
      y_mask = y_mask.pad(((0, 0), (0, 0), (0, pad_length - row_len))).cast(z_p.dtype)
      # New y_mask tensor to remove sts mask
      y_mask = Tensor(y_mask.numpy(), device=y_mask.device, dtype=y_mask.dtype, requires_grad=y_mask.requires_grad)
      z_p = z_p.squeeze(0).pad(((0, 0), (0, pad_length - z_p.shape[2])), value=1).unsqueeze(0)
    z = self.flow.forward(z_p.realize(), y_mask.realize(), g=g.realize(), reverse=True)
    result_length = reduce(lambda x, y: x * y, self.dec.upsample_rates, row_len)
    o = self.dec.forward((z * y_mask)[:, :, :max_len], g=g)[:, :, :result_length]
    if max_y_length_estimate_scale is not None:
      length_scaler = o.shape[-1] / max_y_length
      o.realize()
      real_max_y_length = y_lengths.max().numpy()
      if real_max_y_length > max_y_length:
        logging.warning(f"Underestimated max length by {(((real_max_y_length / max_y_length) * 100) - 100):.2f}%, recomputing inference without estimate...")
        return self.generate(g, logs_p, m_p, max_len, None, noise_scale, w_ceil, x, x_mask, y_lengths)
      if real_max_y_length < max_y_length:
        overestimation = ((max_y_length / real_max_y_length) * 100) - 100
        logging.info(f"Overestimated max length by {overestimation:.2f}%")
        if overestimation > 10: logging.warning("Warning: max length overestimated by more than 10%")
      o = o[:, :, :(real_max_y_length * length_scaler).astype(np.int32)]
    return o

class StochasticDurationPredictor:
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.n_flows, self.gin_channels = in_channels, filter_channels, kernel_size, p_dropout, n_flows, gin_channels
    self.log_flow, self.flows = Log(), [ElementwiseAffine(2)]
    for _ in range(n_flows):
      self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(Flip())
    self.post_pre, self.post_proj = nn.Conv1d(1, filter_channels, 1), nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = [ElementwiseAffine(2)]
    for _ in range(4):
      self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(Flip())
    self.pre, self.proj = nn.Conv1d(in_channels, filter_channels, 1), nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
  @TinyJit
  def forward(self, x: Tensor, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = self.pre(x.detach())
    if g is not None: x = x + self.cond(g.detach())
    x = self.convs.forward(x, x_mask)
    x = self.proj(x) * x_mask
    if not reverse:
      flows = self.flows
      assert w is not None
      log_det_tot_q = 0
      h_w = self.post_proj(self.post_convs.forward(self.post_pre(w), x_mask)) * x_mask
      e_q = Tensor.randn(w.size(0), 2, w.size(2), dtype=x.dtype).to(device=x.device) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, log_det_q = flow.forward(z_q, x_mask, g=(x + h_w))
        log_det_tot_q += log_det_q
      z_u, z1 = z_q.split([1, 1], 1)
      u = z_u.sigmoid() * x_mask
      z0 = (w - u) * x_mask
      log_det_tot_q += Tensor.sum((z_u.logsigmoid() + (-z_u).logsigmoid()) * x_mask, [1,2])
      log_q = Tensor.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - log_det_tot_q
      log_det_tot = 0
      z0, log_det = self.log_flow.forward(z0, x_mask)
      log_det_tot += log_det
      z = z0.cat(z1, 1)
      for flow in flows:
        z, log_det = flow.forward(z, x_mask, g=x, reverse=reverse)
        log_det_tot = log_det_tot + log_det
      nll = Tensor.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - log_det_tot
      return (nll + log_q).realize() # [b]
    flows = list(reversed(self.flows))
    flows = flows[:-2] + [flows[-1]] # remove a useless vflow
    z = Tensor.randn(x.shape[0], 2, x.shape[2], dtype=x.dtype).to(device=x.device) * noise_scale
    for flow in flows: z = flow.forward(z, x_mask, g=x, reverse=reverse)
    z0, z1 = z.split([1, 1], 1)
    return z0.realize()

class DurationPredictor:
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    self.in_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.gin_channels = in_channels, filter_channels, kernel_size, p_dropout, gin_channels
    self.conv_1, self.norm_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2), LayerNorm(filter_channels)
    self.conv_2, self.norm_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2), LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, in_channels, 1)
  def forward(self, x: Tensor, x_mask, g=None):
    x = x.detach()
    if g is not None: x = x + self.cond(g.detach())
    x = self.conv_1(x * x_mask).relu()
    x = self.norm_1(x).dropout(self.p_dropout)
    x = self.conv_2(x * x_mask).relu(x)
    x = self.norm_2(x).dropout(self.p_dropout)
    return self.proj(x * x_mask) * x_mask

class TextEncoder:
  def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, emotion_embedding):
    self.n_vocab, self.out_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout = n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
    if n_vocab!=0:self.emb = nn.Embedding(n_vocab, hidden_channels)
    if emotion_embedding: self.emo_proj = nn.Linear(1024, hidden_channels)
    self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
  @TinyJit
  def forward(self, x: Tensor, x_lengths: Tensor, emotion_embedding=None):
    if self.n_vocab!=0: x = (self.emb(x) * math.sqrt(self.hidden_channels))
    if emotion_embedding: x = x + self.emo_proj(emotion_embedding).unsqueeze(1)
    x = x.transpose(1, -1)  # [b, t, h] -transpose-> [b, h, t]
    x_mask = sequence_mask(x_lengths, x.shape[2]).unsqueeze(1).cast(x.dtype)
    x = self.encoder.forward(x * x_mask, x_mask)
    m, logs = (self.proj(x) * x_mask).split(self.out_channels, dim=1)
    return x.realize(), m.realize(), logs.realize(), x_mask.realize()

class ResidualCouplingBlock:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.n_flows, self.gin_channels = channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows, gin_channels
    self.flows = []
    for _ in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(Flip())
  @TinyJit
  def forward(self, x, x_mask, g=None, reverse=False):
    for flow in reversed(self.flows) if reverse else self.flows: x = flow.forward(x, x_mask, g=g, reverse=reverse)
    return x.realize()

class PosteriorEncoder:
  def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
    self.in_channels, self.out_channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels = in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels
    self.pre, self.proj = nn.Conv1d(in_channels, hidden_channels, 1), nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
  def forward(self, x, x_lengths, g=None):
    x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).cast(x.dtype)
    stats = self.proj(self.enc.forward(self.pre(x) * x_mask, x_mask, g=g)) * x_mask
    m, logs = stats.split(self.out_channels, dim=1)
    z = (m + Tensor.randn(m.shape, m.dtype) * logs.exp()) * x_mask
    return z, m, logs, x_mask

class Generator:
  def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
    self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
    self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups = [nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2) for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))]
    self.resblocks = []
    self.upsample_rates = upsample_rates
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))
    self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
  @TinyJit
  def forward(self, x: Tensor, g=None):
    x = self.conv_pre(x)
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x = self.ups[i](x.leaky_relu(LRELU_SLOPE))
      xs = sum(self.resblocks[i * self.num_kernels + j].forward(x) for j in range(self.num_kernels))
      x = (xs / self.num_kernels).realize()
    res = self.conv_post(x.leaky_relu()).tanh().realize()
    return res

class LayerNorm(nn.LayerNorm):
  def __init__(self, channels, eps=1e-5): super().__init__(channels, eps, elementwise_affine=True)
  def forward(self, x: Tensor): return self.__call__(x.transpose(1, -1)).transpose(1, -1)

class WN:
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    assert (kernel_size % 2 == 1)
    self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels, self.p_dropout = hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout
    self.in_layers, self.res_skip_layers = [], []
    if gin_channels != 0: self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
    for i in range(n_layers):
      dilation = dilation_rate ** i
      self.in_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=int((kernel_size * dilation - dilation) / 2)))
      self.res_skip_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1))
  def forward(self, x, x_mask, g=None, **kwargs):
    output = Tensor.zeros_like(x)
    if g is not None: g = self.cond_layer(g)
    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = Tensor.zeros_like(x_in)
      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts
    return output * x_mask

class ResBlock1:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
    self.convs1 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(3)]
    self.convs2 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in range(3)]
  def forward(self, x: Tensor, x_mask=None):
    for c1, c2 in zip(self.convs1, self.convs2):
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c1(xt if x_mask is None else xt * x_mask).leaky_relu(LRELU_SLOPE)
      x = c2(xt if x_mask is None else xt * x_mask) + x
    return x if x_mask is None else x * x_mask

class ResBlock2:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
    self.convs = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(2)]
  def forward(self, x, x_mask=None):
    for c in self.convs:
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c(xt if x_mask is None else xt * x_mask)
      x = xt + x
    return x if x_mask is None else x * x_mask

class DDSConv: # Dilated and Depth-Separable Convolution
  def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
    self.channels, self.kernel_size, self.n_layers, self.p_dropout = channels, kernel_size, n_layers, p_dropout
    self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2 = [], [], [], []
    for i in range(n_layers):
      dilation = kernel_size ** i
      padding = (kernel_size * dilation - dilation) // 2
      self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
      self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
      self.norms_1.append(LayerNorm(channels))
      self.norms_2.append(LayerNorm(channels))
  def forward(self, x, x_mask, g=None):
    if g is not None: x = x + g
    for i in range(self.n_layers):
      y = self.convs_sep[i](x * x_mask)
      y = self.norms_1[i].forward(y).gelu()
      y = self.convs_1x1[i](y)
      y = self.norms_2[i].forward(y).gelu()
      x = x + y.dropout(self.p_dropout)
    return x * x_mask

class ConvFlow:
  def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
    self.in_channels, self.filter_channels, self.kernel_size, self.n_layers, self.num_bins, self.tail_bound = in_channels, filter_channels, kernel_size, n_layers, num_bins, tail_bound
    self.half_channels = in_channels // 2
    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = x.split([self.half_channels] * 2, 1)
    h = self.proj(self.convs.forward(self.pre(x0), x_mask, g=g)) * x_mask
    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2) # [b, cx?, t] -> [b, c, t, ?]
    un_normalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    un_normalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
    un_normalized_derivatives = h[..., 2 * self.num_bins:]
    x1, log_abs_det = piecewise_rational_quadratic_transform(x1, un_normalized_widths, un_normalized_heights, un_normalized_derivatives, inverse=reverse, tails='linear', tail_bound=self.tail_bound)
    x = x0.cat(x1, dim=1) * x_mask
    return x if reverse else (x, Tensor.sum(log_abs_det * x_mask, [1,2]))

class ResidualCouplingLayer:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.mean_only = channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only
    self.half_channels = channels // 2
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = x.split([self.half_channels] * 2, 1)
    stats = self.post(self.enc.forward(self.pre(x0) * x_mask, x_mask, g=g)) * x_mask
    if not self.mean_only:
      m, logs = stats.split([self.half_channels] * 2, 1)
    else:
      m = stats
      logs = Tensor.zeros_like(m)
    if not reverse: return x0.cat((m + x1 * logs.exp() * x_mask), dim=1)
    return x0.cat(((x1 - m) * (-logs).exp() * x_mask), dim=1)

class Log:
  def forward(self, x : Tensor, x_mask, reverse=False):
    if not reverse:
      y = x.maximum(1e-5).log() * x_mask
      return y, (-y).sum([1, 2])
    return x.exp() * x_mask

class Flip:
  def forward(self, x: Tensor, *args, reverse=False, **kwargs):
    return x.flip([1]) if reverse else (x.flip([1]), Tensor.zeros(x.shape[0], dtype=x.dtype).to(device=x.device))

class ElementwiseAffine:
  def __init__(self, channels): self.m, self.logs = Tensor.zeros(channels, 1), Tensor.zeros(channels, 1)
  def forward(self, x, x_mask, reverse=False, **kwargs): # x if reverse else y, logdet
    return (x - self.m) * Tensor.exp(-self.logs) * x_mask if reverse \
      else ((self.m + Tensor.exp(self.logs) * x) * x_mask, Tensor.sum(self.logs * x_mask, [1, 2]))

class MultiHeadAttention:
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    assert channels % n_heads == 0
    self.channels, self.out_channels, self.n_heads, self.p_dropout, self.window_size, self.heads_share, self.block_length, self.proximal_bias, self.proximal_init = channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init
    self.attn, self.k_channels  = None, channels // n_heads
    self.conv_q, self.conv_k, self.conv_v = [nn.Conv1d(channels, channels, 1) for _ in range(3)]
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    if window_size is not None: self.emb_rel_k, self.emb_rel_v = [Tensor.randn(1 if heads_share else n_heads, window_size * 2 + 1, self.k_channels) * (self.k_channels ** -0.5) for _ in range(2)]
  def forward(self, x, c, attn_mask=None):
    q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    return self.conv_o(x)
  def attention(self, query: Tensor, key: Tensor, value: Tensor, mask=None):# reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = key.shape[0], key.shape[1], key.shape[2], query.shape[2]
    query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
      scores = scores + self._relative_position_to_absolute_position(rel_logits)
    if mask is not None:
      scores = Tensor.where(mask, scores, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        scores = Tensor.where(Tensor.ones_like(scores).triu(-self.block_length).tril(self.block_length), scores, -1e4)
    p_attn = scores.softmax(axis=-1)  # [b, n_h, t_t, t_s]
    output = p_attn.matmul(value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().reshape(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn
  def _matmul_with_relative_values(self, x, y): return x.matmul(y.unsqueeze(0))                 # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]
  def _matmul_with_relative_keys(self, x, y): return x.matmul(y.unsqueeze(0).transpose(-2, -1)) # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]
  def _get_relative_embeddings(self, relative_embeddings, length):
    pad_length, slice_start_position = max(length - (self.window_size + 1), 0), max((self.window_size + 1) - length, 0)
    padded_relative_embeddings = relative_embeddings if pad_length <= 0\
      else relative_embeddings.pad(convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    return padded_relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)] #used_relative_embeddings
  def _relative_position_to_absolute_position(self, x: Tensor): # x: [b, h, l, 2*l-1] -> [b, h, l, l]
    batch, heads, length, _ = x.shape
    x = x.pad(convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
    x_flat = x.reshape([batch, heads, length * 2 * length]).pad(convert_pad_shape([[0,0],[0,0],[0,length-1]]))
    return x_flat.reshape([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
  def _absolute_position_to_relative_position(self, x: Tensor): # x: [b, h, l, l] -> [b, h, l, 2*l-1]
    batch, heads, length, _ = x.shape
    x = x.pad(convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.reshape([batch, heads, length**2 + length*(length -1)]).pad(convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    return x_flat.reshape([batch, heads, length, 2*length])[:,:,:,1:]

class FFN:
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    self.in_channels, self.out_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.activation, self.causal = in_channels, out_channels, filter_channels, kernel_size, p_dropout, activation, causal
    self.padding = self._causal_padding if causal else self._same_padding
    self.conv_1, self.conv_2 = nn.Conv1d(in_channels, filter_channels, kernel_size), nn.Conv1d(filter_channels, out_channels, kernel_size)
  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    x = x * (1.702 * x).sigmoid() if self.activation == "gelu" else x.relu()
    return self.conv_2(self.padding(x.dropout(self.p_dropout) * x_mask)) * x_mask
  def _causal_padding(self, x):return x if self.kernel_size == 1 else x.pad(convert_pad_shape([[0, 0], [0, 0], [self.kernel_size - 1, 0]]))
  def _same_padding(self, x): return x if self.kernel_size == 1 else x.pad(convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1) // 2, self.kernel_size // 2]]))

class Encoder:
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.window_size = hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size
    self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2 = [], [], [], []
    for _ in range(n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))
  def forward(self, x, x_mask):
    attn_mask, x = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1), x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i].forward(x, x, attn_mask).dropout(self.p_dropout)
      x = self.norm_layers_1[i].forward(x + y)
      y = self.ffn_layers[i].forward(x, x_mask).dropout(self.p_dropout)
      x = self.norm_layers_2[i].forward(x + y)
    return x * x_mask

DEFAULT_MIN_BIN_WIDTH, DEFAULT_MIN_BIN_HEIGHT, DEFAULT_MIN_DERIVATIVE = 1e-3, 1e-3, 1e-3
def piecewise_rational_quadratic_transform(inputs, un_normalized_widths, un_normalized_heights, un_normalized_derivatives, inverse=False, tails=None, tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
  if tails is None: spline_fn, spline_kwargs = rational_quadratic_spline, {}
  else: spline_fn, spline_kwargs = unconstrained_rational_quadratic_spline, {'tails': tails, 'tail_bound': tail_bound}
  return spline_fn(inputs=inputs, un_normalized_widths=un_normalized_widths, un_normalized_heights=un_normalized_heights, un_normalized_derivatives=un_normalized_derivatives, inverse=inverse, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative, **spline_kwargs)
def unconstrained_rational_quadratic_spline(inputs, un_normalized_widths, un_normalized_heights, un_normalized_derivatives, inverse=False, tails='linear', tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
  if not tails == 'linear': raise RuntimeError('{} tails are not implemented.'.format(tails))
  constant = np.log(np.exp(1 - min_derivative) - 1).item()
  un_normalized_derivatives = cat_lr(un_normalized_derivatives, constant, constant)
  output, log_abs_det = rational_quadratic_spline(inputs=inputs.squeeze(dim=0).squeeze(dim=0), unnormalized_widths=un_normalized_widths.squeeze(dim=0).squeeze(dim=0), unnormalized_heights=un_normalized_heights.squeeze(dim=0).squeeze(dim=0), unnormalized_derivatives=un_normalized_derivatives.squeeze(dim=0).squeeze(dim=0), inverse=inverse, left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)
  return output.unsqueeze(dim=0).unsqueeze(dim=0), log_abs_det.unsqueeze(dim=0).unsqueeze(dim=0)
def rational_quadratic_spline(inputs: Tensor, unnormalized_widths: Tensor, unnormalized_heights: Tensor, unnormalized_derivatives: Tensor, inverse=False, left=0., right=1., bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
  num_bins = unnormalized_widths.shape[-1]
  if min_bin_width * num_bins > 1.0: raise ValueError('Minimal bin width too large for the number of bins')
  if min_bin_height * num_bins > 1.0: raise ValueError('Minimal bin height too large for the number of bins')
  widths = min_bin_width + (1 - min_bin_width * num_bins) * unnormalized_widths.softmax(axis=-1)
  cum_widths = cat_lr(((right - left) * widths[..., :-1].cumsum(axis=1) + left), left, right + 1e-6 if not inverse else right)
  widths = cum_widths[..., 1:] - cum_widths[..., :-1]
  derivatives = min_derivative + (unnormalized_derivatives.exp()+1).log()
  heights = min_bin_height + (1 - min_bin_height * num_bins) * unnormalized_heights.softmax(axis=-1)
  cum_heights = cat_lr(((top - bottom) * heights[..., :-1].cumsum(axis=1) + bottom), bottom, top + 1e-6 if inverse else top)
  heights = cum_heights[..., 1:] - cum_heights[..., :-1]
  bin_idx = ((inputs[..., None] >= (cum_heights if inverse else cum_widths)).sum(axis=-1) - 1)[..., None]
  input_cum_widths = gather(cum_widths, bin_idx, axis=-1)[..., 0]
  input_bin_widths = gather(widths, bin_idx, axis=-1)[..., 0]
  input_cum_heights = gather(cum_heights, bin_idx, axis=-1)[..., 0]
  input_delta = gather(heights / widths, bin_idx, axis=-1)[..., 0]
  input_derivatives = gather(derivatives, bin_idx, axis=-1)[..., 0]
  input_derivatives_plus_one = gather(derivatives[..., 1:], bin_idx, axis=-1)[..., 0]
  input_heights = gather(heights, bin_idx, axis=-1)[..., 0]
  if inverse:
    a = ((inputs - input_cum_heights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives))
    b = (input_heights * input_derivatives - (inputs - input_cum_heights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
    c = - input_delta * (inputs - input_cum_heights)
    discriminant = b.square() - 4 * a * c
    # assert (discriminant.numpy() >= 0).all()
    root = (2 * c) / (-b - discriminant.sqrt())
    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
    derivative_numerator = input_delta.square() * (input_derivatives_plus_one * root.square() + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - root).square())
    return root * input_bin_widths + input_cum_widths, -(derivative_numerator.log() - 2 * denominator.log())
  theta = (inputs - input_cum_widths) / input_bin_widths
  theta_one_minus_theta = theta * (1 - theta)
  numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
  denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
  derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
  return input_cum_heights + numerator / denominator, derivative_numerator.log() - 2 * denominator.log()

def sequence_mask(length: Tensor, max_length): return Tensor.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0) < length.unsqueeze(1)
def generate_path(duration: Tensor, mask: Tensor):  # duration: [b, 1, t_x], mask: [b, 1, t_y, t_x]
  b, _, t_y, t_x = mask.shape
  path = sequence_mask(duration.cumsum(axis=2).reshape(b * t_x), t_y).cast(mask.dtype).reshape(b, t_x, t_y)
  path = path - path.pad(convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  return path.unsqueeze(1).transpose(2, 3) * mask
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: int):
  n_channels_int, in_act = n_channels, input_a + input_b
  t_act, s_act = in_act[:, :n_channels_int, :].tanh(), in_act[:, n_channels_int:, :].sigmoid()
  return t_act * s_act

def cat_lr(t, left, right): return Tensor.full(get_shape(t), left).cat(t, dim=-1).cat(Tensor.full(get_shape(t), right), dim=-1)
def get_shape(tensor):
  (shape := list(tensor.shape))[-1] = 1
  return tuple(shape)
def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)
def get_padding(kernel_size, dilation=1): return int((kernel_size*dilation - dilation)/2)

def gather(x, indices, axis):
  indices = (indices < 0).where(indices + x.shape[axis], indices).transpose(0, axis)
  permute_args = list(range(x.ndim))
  permute_args[0], permute_args[axis] = permute_args[axis], permute_args[0]
  permute_args.append(permute_args.pop(0))
  x = x.permute(*permute_args)
  reshape_arg = [1] * x.ndim + [x.shape[-1]]
  return ((indices.unsqueeze(indices.ndim).expand(*indices.shape, x.shape[-1]) ==
           Tensor.arange(x.shape[-1]).reshape(*reshape_arg).expand(*indices.shape, x.shape[-1])) * x).sum(indices.ndim).transpose(0, axis)

def norm_except_dim(v, dim):
  if dim == -1: return np.linalg.norm(v)
  if dim == 0:
    (output_shape := [1] * v.ndim)[0] = v.shape[0]
    return np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape(output_shape)
  if dim == v.ndim - 1:
    (output_shape := [1] * v.ndim)[-1] = v.shape[-1]
    return np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=0).reshape(output_shape)
  transposed_v = np.transpose(v, (dim,) + tuple(i for i in range(v.ndim) if i != dim))
  return np.transpose(norm_except_dim(transposed_v, 0), (dim,) + tuple(i for i in range(v.ndim) if i != dim))
def weight_norm(v: Tensor, g: Tensor, dim):
  v, g = v.numpy(), g.numpy()
  return Tensor(v * (g / norm_except_dim(v, dim)))

# HPARAMS LOADING
def get_hparams_from_file(path):
  with open(path, "r") as f:
    data = f.read()
  return HParams(**json.loads(data))
class HParams:
  def __init__(self, **kwargs):
    for k, v in kwargs.items(): self[k] = v if type(v) != dict else HParams(**v)
  def keys(self): return self.__dict__.keys()
  def items(self): return self.__dict__.items()
  def values(self): return self.__dict__.values()
  def __len__(self): return len(self.__dict__)
  def __getitem__(self, key): return getattr(self, key)
  def __setitem__(self, key, value): return setattr(self, key, value)
  def __contains__(self, key): return key in self.__dict__
  def __repr__(self): return self.__dict__.__repr__()

# MODEL LOADING
def load_model(symbols, hps, model) -> Synthesizer:
  net_g = Synthesizer(len(symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers = hps.data.n_speakers, **hps.model)
  _ = load_checkpoint(fetch(model[1]), net_g, None)
  return net_g
def load_checkpoint(checkpoint_path, model: Synthesizer, optimizer=None, skip_list=[]):
  assert Path(checkpoint_path).is_file()
  start_time = time.time()
  checkpoint_dict = torch_load(checkpoint_path)
  iteration, learning_rate = checkpoint_dict['iteration'], checkpoint_dict['learning_rate']
  if optimizer: optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  weight_g, weight_v, parent = None, None, None
  for key, v in saved_state_dict.items():
    if any(layer in key for layer in skip_list): continue
    try:
      obj, skip = model, False
      for k in key.split('.'):
        if k.isnumeric(): obj = obj[int(k)]
        elif isinstance(obj, dict): obj = obj[k]
        else:
          if isinstance(obj, (LayerNorm, nn.LayerNorm)) and k in ["gamma", "beta"]:
            k = "weight" if k == "gamma" else "bias"
          elif k in ["weight_g", "weight_v"]:
            parent, skip = obj, True
            if k == "weight_g": weight_g = v
            else: weight_v = v
          if not skip: obj = getattr(obj, k)
      if weight_g is not None and weight_v is not None:
        setattr(obj, "weight_g", weight_g.numpy())
        setattr(obj, "weight_v", weight_v.numpy())
        obj, v = getattr(parent, "weight"), weight_norm(weight_v, weight_g, 0)
        weight_g, weight_v, parent, skip = None, None, None, False
      if not skip and obj.shape == v.shape: obj.assign(v.to(obj.device))
      elif not skip: logging.error(f"MISMATCH SHAPE IN {key}, {obj.shape} {v.shape}")
    except Exception as e: raise e
  logging.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration}) in {time.time() - start_time:.4f}s")
  return model, optimizer, learning_rate, iteration

# Used for cleaning input text and mapping to symbols
class TextMapper: # Based on https://github.com/keithito/tacotron
  def __init__(self, symbols, apply_cleaners=True):
    self.apply_cleaners, self.symbols, self._inflect = apply_cleaners, symbols, None
    self._symbol_to_id, _id_to_symbol = {s: i for i, s in enumerate(symbols)}, {i: s for i, s in enumerate(symbols)}
    self._whitespace_re, self._abbreviations = re.compile(r'\s+'), [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [('mrs', 'misess'), ('mr', 'mister'), ('dr', 'doctor'), ('st', 'saint'), ('co', 'company'), ('jr', 'junior'), ('maj', 'major'), ('gen', 'general'), ('drs', 'doctors'), ('rev', 'reverend'), ('lt', 'lieutenant'), ('hon', 'honorable'), ('sgt', 'sergeant'), ('capt', 'captain'), ('esq', 'esquire'), ('ltd', 'limited'), ('col', 'colonel'), ('ft', 'fort'), ]]
    self.phonemizer = EspeakBackend(
        language="en-us", punctuation_marks=Punctuation.default_marks(), preserve_punctuation=True, with_stress=True,
    )
  def text_to_sequence(self, text, cleaner_names):
    if self.apply_cleaners:
      for name in cleaner_names:
        cleaner = getattr(self, name)
        if not cleaner: raise ModuleNotFoundError('Unknown cleaner: %s' % name)
        text = cleaner(text)
    else: text = text.strip()
    return [self._symbol_to_id[symbol] for symbol in text]
  def get_text(self, text, add_blank=False, cleaners=('english_cleaners2',)):
    text_norm = self.text_to_sequence(text, cleaners)
    return Tensor(self.intersperse(text_norm, 0) if add_blank else text_norm, dtype=dtypes.int64)
  def intersperse(self, lst, item):
    (result := [item] * (len(lst) * 2 + 1))[1::2] = lst
    return result
  def phonemize(self, text, strip=True): return _phonemize(self.phonemizer, text, default_separator, strip, 1, False, False)
  def filter_oov(self, text): return "".join(list(filter(lambda x: x in self._symbol_to_id, text)))
  def base_english_cleaners(self, text): return self.collapse_whitespace(self.phonemize(self.expand_abbreviations(unidecode(text.lower()))))
  def english_cleaners2(self, text): return self.base_english_cleaners(text)
  def transliteration_cleaners(self, text): return self.collapse_whitespace(unidecode(text.lower()))
  def cjke_cleaners(self, text): return re.sub(r'([^\.,!\?\-…~])$', r'\1.', re.sub(r'\s+$', '', self.english_to_ipa2(text).replace('ɑ', 'a').replace('ɔ', 'o').replace('ɛ', 'e').replace('ɪ', 'i').replace('ʊ', 'u')))
  def cjke_cleaners2(self, text): return re.sub(r'([^\.,!\?\-…~])$', r'\1.', re.sub(r'\s+$', '', self.english_to_ipa2(text)))
  def cjks_cleaners(self, text): return re.sub(r'([^\.,!\?\-…~])$', r'\1.', re.sub(r'\s+$', '', self.english_to_lazy_ipa(text)))
  def english_to_ipa2(self, text):
    _ipa_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [ ('r', 'ɹ'), ('ʤ', 'dʒ'), ('ʧ', 'tʃ')]]
    return reduce(lambda t, rx: re.sub(rx[0], rx[1], t), _ipa_to_ipa2, self.mark_dark_l(self.english_to_ipa(text))).replace('...', '…')
  def mark_dark_l(self, text): return re.sub(r'l([^aeiouæɑɔəɛɪʊ ]*(?: |$))', lambda x: 'ɫ' + x.group(1), text)
  def english_to_ipa(self, text):
    import eng_to_ipa as ipa
    return self.collapse_whitespace(ipa.convert(self.normalize_numbers(self.expand_abbreviations(unidecode(text).lower()))))
  def english_to_lazy_ipa(self, text):
    _lazy_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [('r', 'ɹ'), ('æ', 'e'), ('ɑ', 'a'), ('ɔ', 'o'), ('ð', 'z'), ('θ', 's'), ('ɛ', 'e'), ('ɪ', 'i'), ('ʊ', 'u'), ('ʒ', 'ʥ'), ('ʤ', 'ʥ'), ('ˈ', '↓')]]
    return reduce(lambda t, rx: re.sub(rx[0], rx[1], t), _lazy_ipa, self.english_to_ipa(text))
  def expand_abbreviations(self, text): return reduce(lambda t, abbr: re.sub(abbr[0], abbr[1], t), self._abbreviations, text)
  def collapse_whitespace(self, text): return re.sub(self._whitespace_re, ' ', text)
  def normalize_numbers(self, text):
    import inflect
    self._inflect = inflect.engine()
    text = re.sub(re.compile(r'([0-9][0-9\,]+[0-9])'), self._remove_commas, text)
    text = re.sub(re.compile(r'£([0-9\,]*[0-9]+)'), r'\1 pounds', text)
    text = re.sub(re.compile(r'\$([0-9\.\,]*[0-9]+)'), self._expand_dollars, text)
    text = re.sub(re.compile(r'([0-9]+\.[0-9]+)'), self._expand_decimal_point, text)
    text = re.sub(re.compile(r'[0-9]+(st|nd|rd|th)'), self._expand_ordinal, text)
    text = re.sub(re.compile(r'[0-9]+'), self._expand_number, text)
    return text
  def _remove_commas(self, m): return m.group(1).replace(',', '') # george won't like this
  def _expand_dollars(self, m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2: return match + ' dollars'  # Unexpected format
    dollars, cents = int(parts[0]) if parts[0] else 0, int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents: return '%s %s, %s %s' % (dollars, 'dollar' if dollars == 1 else 'dollars', cents, 'cent' if cents == 1 else 'cents')
    if dollars: return '%s %s' % (dollars, 'dollar' if dollars == 1 else 'dollars')
    if cents: return '%s %s' % (cents, 'cent' if cents == 1 else 'cents')
    return 'zero dollars'
  def _expand_decimal_point(self, m): return m.group(1).replace('.', ' point ')
  def _expand_ordinal(self, m): return self._inflect.number_to_words(m.group(0))
  def _expand_number(self, _inflect, m):
    num = int(m.group(0))
    if 1000 < num < 3000:
      if num == 2000: return 'two thousand'
      if 2000 < num < 2010: return 'two thousand ' + self._inflect.number_to_words(num % 100)
      if num % 100 == 0: return self._inflect.number_to_words(num // 100) + ' hundred'
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    return self._inflect.number_to_words(num, andword='')

#########################################################################################
# PAPER: https://arxiv.org/abs/2106.06103
# CODE: https://github.com/jaywalnut310/vits/tree/main
#########################################################################################
# INSTALLATION: this is based on default config, dependencies are for preprocessing.
# vctk, ljs                      | pip3 install unidecode phonemizer          | phonemizer requires [eSpeak](https://espeak.sourceforge.net) backend to be installed on your system
# mmts-tts                       | pip3 install unidecode                     |
# uma_trilingual, cjks, voistock | pip3 install unidecode inflect eng_to_ipa  |
#########################################################################################
# Some good speakers to try out, there may be much better ones, I only tried out a few:
# male vctk 1  | --model_to_use vctk --speaker_id 2
# male vctk 2  | --model_to_use vctk --speaker_id 6
# anime lady 1 | --model_to_use uma_trilingual --speaker_id 36
# anime lady 2 | --model_to_use uma_trilingual --speaker_id 121
#########################################################################################
VITS_PATH = Path(__file__).parents[1] / "weights/VITS/"
MODELS = { # config_url, weights_url
  "ljs": ("https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/ljs_base.json", "https://drive.google.com/uc?export=download&id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT&confirm=t"),
  "vctk": ("https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/vctk_base.json", "https://drive.google.com/uc?export=download&id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru&confirm=t"),
  "mmts-tts": ("https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/config.json", "https://huggingface.co/facebook/mms-tts/resolve/main/full_models/eng/G_100000.pth"),
  "uma_trilingual": ("https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/raw/main/configs/uma_trilingual.json", "https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth"),
  "cjks": ("https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/model.pth"),
  "voistock": ("https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/model.pth"),
}
Y_LENGTH_ESTIMATE_SCALARS = {"ljs": 2.8, "vctk": 1.74, "mmts-tts": 1.9, "uma_trilingual": 2.3, "cjks": 3.3, "voistock": 3.1}
if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_to_use", default="vctk", help="Specify the model to use. Default is 'vctk'.")
  parser.add_argument("--speaker_id", type=int, default=6, help="Specify the speaker ID. Default is 6.")
  parser.add_argument("--out_path", default=None, help="Specify the full output path. Overrides the --out_dir and --name parameter.")
  parser.add_argument("--out_dir", default=str(Path(__file__).parents[1] / "temp"), help="Specify the output path.")
  parser.add_argument("--base_name", default="test", help="Specify the base of the output file name. Default is 'test'.")
  parser.add_argument("--text_to_synthesize", default="""Hello person. If the code you are contributing isn't some of the highest quality code you've written in your life, either put in the effort to make it great, or don't bother.""", help="Specify the text to synthesize. Default is a greeting message.")
  parser.add_argument("--noise_scale", type=float, default=0.667, help="Specify the noise scale. Default is 0.667.")
  parser.add_argument("--noise_scale_w", type=float, default=0.8, help="Specify the noise scale w. Default is 0.8.")
  parser.add_argument("--length_scale", type=float, default=1, help="Specify the length scale. Default is 1.")
  parser.add_argument("--seed", type=int, default=1337, help="Specify the seed (set to None if no seed). Default is 1337.")
  parser.add_argument("--num_channels", type=int, default=1, help="Specify the number of audio output channels. Default is 1.")
  parser.add_argument("--sample_width", type=int, default=2, help="Specify the number of bytes per sample, adjust if necessary. Default is 2.")
  parser.add_argument("--emotion_path", type=str, default=None, help="Specify the path to emotion reference.")
  parser.add_argument("--estimate_max_y_length", type=str, default=False, help="If true, overestimate the output length and then trim it to the correct length, to prevent premature realization, much more performant for larger inputs, for smaller inputs not so much. Default is False.")
  args = parser.parse_args()

  model_config = MODELS[args.model_to_use]

  # Load the hyperparameters from the config file.
  hps = get_hparams_from_file(fetch(model_config[0]))

  # If model has multiple speakers, validate speaker id and retrieve name if available.
  model_has_multiple_speakers = hps.data.n_speakers > 0
  if model_has_multiple_speakers:
    logging.info(f"Model has {hps.data.n_speakers} speakers")
    if args.speaker_id >= hps.data.n_speakers: raise ValueError(f"Speaker ID {args.speaker_id} is invalid for this model.")
    speaker_name = "?"
    if hps.__contains__("speakers"): # maps speaker ids to names
      speakers = hps.speakers
      if isinstance(speakers, List): speakers = {speaker: i for i, speaker in enumerate(speakers)}
      speaker_name = next((key for key, value in speakers.items() if value == args.speaker_id), None)
    logging.info(f"You selected speaker {args.speaker_id} (name: {speaker_name})")

  # Load emotions if any. TODO: find an english model with emotions, this is untested atm.
  emotion_embedding = None
  if args.emotion_path is not None:
    if args.emotion_path.endswith(".npy"): emotion_embedding = Tensor(np.load(args.emotion_path), dtype=dtypes.int64).unsqueeze(0)
    else: raise ValueError("Emotion path must be a .npy file.")

  # Load symbols, instantiate TextMapper and clean the text.
  if hps.__contains__("symbols"): symbols = hps.symbols
  elif args.model_to_use == "mmts-tts": symbols = [x.replace("\n", "") for x in fetch("https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt").open(encoding="utf-8").readlines()]
  else: symbols = ['_'] + list(';:,.!?¡¿—…"«»“” ') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)

  # Load the model.
  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)
    np.random.seed(args.seed)
  net_g = load_model(text_mapper.symbols, hps, model_config)
  logging.debug(f"Loaded model with hps: {hps}")

  # Convert the input text to a tensor.
  text_to_synthesize = args.text_to_synthesize
  if args.model_to_use == "mmts-tts": text_to_synthesize = text_mapper.filter_oov(text_to_synthesize.lower())
  stn_tst = text_mapper.get_text(text_to_synthesize, hps.data.add_blank, hps.data.text_cleaners)
  logging.debug(f"Converted input text to tensor \"{text_to_synthesize}\" -> Tensor({stn_tst.shape}): {stn_tst.numpy()}")
  x_tst, x_tst_lengths = stn_tst.unsqueeze(0), Tensor([stn_tst.shape[0]], dtype=dtypes.int64)
  sid = Tensor([args.speaker_id], dtype=dtypes.int64) if model_has_multiple_speakers else None

  # Perform inference.
  start_time = time.time()
  audio_tensor = net_g.infer(x_tst, x_tst_lengths, sid, args.noise_scale, args.length_scale, args.noise_scale_w, emotion_embedding=emotion_embedding,
                             max_y_length_estimate_scale=Y_LENGTH_ESTIMATE_SCALARS[args.model_to_use] if args.estimate_max_y_length else None)[0, 0].realize()
  logging.info(f"Inference took {(time.time() - start_time):.2f}s")

  # Save the audio output.
  audio_data = (np.clip(audio_tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)
  out_path = Path(args.out_path or Path(args.out_dir)/f"{args.model_to_use}{f'_sid_{args.speaker_id}' if model_has_multiple_speakers else ''}_{args.base_name}.wav")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(out_path), 'wb') as wav_file:
    wav_file.setnchannels(args.num_channels)
    wav_file.setsampwidth(args.sample_width)
    wav_file.setframerate(hps.data.sampling_rate)
    wav_file.setnframes(len(audio_data))
    wav_file.writeframes(audio_data.tobytes())
  logging.info(f"Saved audio output to {out_path}")
