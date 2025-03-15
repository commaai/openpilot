# pip3 install sentencepiece

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tinygrad T5 model."""
from tinygrad import nn, Tensor, dtypes

import math
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
from pathlib import Path
from sentencepiece import SentencePieceProcessor

# default config is t5-xxl
@dataclass
class T5Config:
  d_ff:int = 10240
  d_kv:int = 64
  d_model:int = 4096
  layer_norm_epsilon:float = 1e-6
  num_decoder_layers:int = 24
  num_heads:int = 64
  num_layers:int = 24
  relative_attention_num_buckets:int = 32
  relative_attention_max_distance:int = 128
  vocab_size:int = 32128

class T5Tokenizer:
  def __init__(self, spiece_path):
    self.spp = SentencePieceProcessor(str(spiece_path))

  def encode(self, text:str, max_length:int) -> List[int]:
    encoded = self.spp.Encode(text)
    if len(encoded) > max_length - 1: encoded = encoded[:max_length - 1]
    return encoded + [1] + [0]*(max_length - len(encoded) - 1)

class T5LayerNorm:
  def __init__(self, hidden_size:int, eps:float=1e-6):
    """
    Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
    """
    self.weight = Tensor.ones(hidden_size)
    self.variance_epsilon = eps

  def __call__(self, hidden_states:Tensor) -> Tensor:
    # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    variance = hidden_states.cast(dtypes.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * Tensor.rsqrt(variance + self.variance_epsilon)

    # convert into half-precision if necessary
    if self.weight.dtype in [dtypes.float16, dtypes.bfloat16]:
      hidden_states = hidden_states.cast(self.weight.dtype)

    return self.weight * hidden_states


class T5DenseGatedActDense:
  def __init__(self, config:T5Config):
    self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

  def __call__(self, hidden_states:Tensor) -> Tensor:
    hidden_gelu = self.wi_0(hidden_states).gelu()
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.wo(hidden_states)
    return hidden_states


class T5LayerFF:
  def __init__(self, config:T5Config):
    self.DenseReluDense = T5DenseGatedActDense(config)
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def __call__(self, hidden_states:Tensor) -> Tensor:
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + forwarded_states
    return hidden_states


class T5Attention:
  def __init__(self, config:T5Config, has_relative_attention_bias:bool=False):
    self.has_relative_attention_bias = has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = config.relative_attention_max_distance
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    # Mesh TensorFlow initialization to avoid scaling before softmax
    self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    if self.has_relative_attention_bias:
      self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

  @staticmethod
  def _relative_position_bucket(relative_position:Tensor, num_buckets:int=32, max_distance:int=128) -> Tensor:
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = Tensor.zeros_like(relative_position)
    num_buckets //= 2
    relative_buckets += (relative_position > 0).cast(dtypes.long) * num_buckets
    relative_position = Tensor.abs(relative_position)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        Tensor.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).cast(dtypes.long)

    relative_position_if_large = Tensor.min(
        Tensor.stack(
            relative_position_if_large, Tensor.full_like(relative_position_if_large, num_buckets - 1)
        ),
        axis=0,
    )
    relative_buckets += Tensor.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None) -> Tensor:
    """Compute binned relative position bias"""
    if device is None:
      device = self.relative_attention_bias.weight.device
    context_position = Tensor.arange(query_length, dtype=dtypes.long, device=device)[:, None]
    memory_position = Tensor.arange(key_length, dtype=dtypes.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

  def __call__(self, hidden_states:Tensor, position_bias:Optional[Tensor]=None) -> Tuple[Tensor,Tensor]:
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    batch_size, key_length = hidden_states.shape[:2]

    def shape(states):
      """projection"""
      return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
      """reshape"""
      return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer):
      """projects hidden states correctly to key/query states"""
      # self-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      return shape(proj_layer(hidden_states))

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(hidden_states, self.k)
    value_states = project(hidden_states, self.v)

    # compute scores
    scores = Tensor.matmul(query_states, key_states.transpose(3, 2))

    if position_bias is None:
      position_bias = self.compute_bias(key_length, key_length, device=scores.device)

    scores += position_bias
    attn_weights = Tensor.softmax(scores.float(), axis=-1).cast(scores.dtype)  # (batch_size, n_heads, seq_length, key_length)

    attn_output = unshape(Tensor.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    return attn_output, position_bias


class T5LayerSelfAttention:
  def __init__(self, config:T5Config, has_relative_attention_bias:bool=False):
    self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def __call__(self, hidden_states:Tensor, position_bias:Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
    normed_hidden_states = self.layer_norm(hidden_states)
    attention_output, position_bias = self.SelfAttention(normed_hidden_states, position_bias=position_bias)
    return hidden_states + attention_output, position_bias


class T5Block:
  def __init__(self, config:T5Config, has_relative_attention_bias:bool=False):
    self.layer = (T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias),
                  T5LayerFF(config))

  def __call__(self, hidden_states:Tensor, position_bias:Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
    self_attention_outputs, position_bias = self.layer[0](hidden_states, position_bias=position_bias)
    hidden_states = self_attention_outputs

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    return hidden_states, position_bias


class T5Stack:
  def __init__(self, config:T5Config, embed_tokens:nn.Embedding):
    self.config = config
    self.embed_tokens = embed_tokens
    self.block = [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
    self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def __call__(self, input_ids:Tensor) -> Tensor:
    input_ids = input_ids.view(-1, input_ids.shape[-1])

    hidden_states, position_bias = self.embed_tokens(input_ids), None

    for layer_module in self.block:
      hidden_states, position_bias = layer_module(hidden_states, position_bias=position_bias)

    return self.final_layer_norm(hidden_states)


class T5EncoderModel:
  def __init__(self, config:T5Config):
    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    self.encoder = T5Stack(config, self.shared)

  def __call__(self, input_ids:Tensor) -> Tensor:
    return self.encoder(input_ids)

class T5Embedder:
  def __init__(self, max_length:int, spiece_path:Union[str, Path]):
    self.tokenizer = T5Tokenizer(spiece_path)
    self.max_length = max_length
    config = T5Config()
    self.encoder = T5EncoderModel(config)

  def __call__(self, texts:Union[str, List[str]]) -> Tensor:
    if isinstance(texts, str): texts = [texts]
    toks = Tensor.cat(*[Tensor(self.tokenizer.encode(text, self.max_length)) for text in texts], dim=0)
    return self.encoder(toks)