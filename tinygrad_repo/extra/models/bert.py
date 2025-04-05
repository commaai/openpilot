import re, os
from pathlib import Path
from tinygrad.tensor import Tensor, cast
from tinygrad import nn, dtypes
from tinygrad.helpers import fetch, get_child
from tinygrad.nn.state import get_parameters

# allow for monkeypatching
Embedding = nn.Embedding
Linear = nn.Linear
LayerNorm = nn.LayerNorm

class BertForQuestionAnswering:
  def __init__(self, hidden_size=1024, intermediate_size=4096, max_position_embeddings=512, num_attention_heads=16, num_hidden_layers=24, type_vocab_size=2, vocab_size=30522, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
    self.bert = Bert(hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob)
    self.qa_outputs = Linear(hidden_size, 2)

  def load_from_pretrained(self):
    fn = Path(__file__).parents[1] / "weights/bert_for_qa.pt"
    fetch("https://zenodo.org/record/3733896/files/model.pytorch?download=1", fn)
    fn_vocab = Path(__file__).parents[1] / "weights/bert_vocab.txt"
    fetch("https://zenodo.org/record/3733896/files/vocab.txt?download=1", fn_vocab)

    import torch
    with open(fn, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")

    for k, v in state_dict.items():
      if "dropout" in k: continue # skip dropout
      if "pooler" in k: continue # skip pooler
      get_child(self, k).assign(v.numpy()).realize()

  def __call__(self, input_ids:Tensor, attention_mask:Tensor, token_type_ids:Tensor):
    sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.chunk(2, dim=-1)
    start_logits = start_logits.reshape(-1, 1)
    end_logits = end_logits.reshape(-1, 1)

    return Tensor.stack(start_logits, end_logits)

class BertForPretraining:
  def __init__(self, hidden_size:int=1024, intermediate_size:int=4096, max_position_embeddings:int=512, num_attention_heads:int=16, num_hidden_layers:int=24, type_vocab_size:int=2, vocab_size:int=30522, attention_probs_dropout_prob:float=0.1, hidden_dropout_prob:float=0.1):
    """Default is BERT-large"""
    self.bert = Bert(hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob)
    self.cls = BertPreTrainingHeads(hidden_size, vocab_size, self.bert.embeddings.word_embeddings.weight)

  def __call__(self, input_ids:Tensor, attention_mask:Tensor, masked_lm_positions:Tensor, token_type_ids:Tensor):
    output = self.bert(input_ids, attention_mask, token_type_ids)
    return self.cls(output, masked_lm_positions)

  # Reference has residual on denominator: https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/run_pretraining.py#L315
  def sparse_categorical_crossentropy(self, predictions:Tensor, labels:Tensor, ignore_index=-1):
    log_probs, loss_mask = predictions.log_softmax(dtype=dtypes.float), (labels != ignore_index)
    y_counter = Tensor.arange(predictions.shape[-1], requires_grad=False, device=predictions.device).unsqueeze(0).expand(labels.numel(), predictions.shape[-1])
    y = ((y_counter == labels.flatten().reshape(-1, 1)) * loss_mask.reshape(-1, 1)).reshape(*labels.shape, predictions.shape[-1])
    return -((log_probs * y).sum()) / (loss_mask.sum() + 1e-5) # Small constant to avoid division by zero

  def loss(self, prediction_logits:Tensor, seq_relationship_logits:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
    masked_lm_loss = self.sparse_categorical_crossentropy(prediction_logits, masked_lm_ids, ignore_index=masked_lm_weights)
    next_sentence_loss = seq_relationship_logits.binary_crossentropy_logits(next_sentence_labels)
    return masked_lm_loss + next_sentence_loss

  def accuracy(self, prediction_logits:Tensor, seq_relationship_logits:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
    valid = masked_lm_ids != 0
    masked_lm_predictions = prediction_logits.argmax(-1)
    masked_lm_correct = (masked_lm_predictions == masked_lm_ids) * valid
    masked_lm_loss = self.sparse_categorical_crossentropy(prediction_logits, masked_lm_ids, ignore_index=masked_lm_weights)

    seq_relationship_predictions = seq_relationship_logits.argmax(-1)
    seq_relationship_correct = (seq_relationship_predictions == next_sentence_labels)
    next_sentence_loss = seq_relationship_logits.binary_crossentropy_logits(next_sentence_labels)

    # TODO: is it okay that next_sentence_loss is half here?
    return masked_lm_correct.sum().float() / valid.sum(), seq_relationship_correct.mean(), masked_lm_loss, next_sentence_loss.float()

  def load_from_pretrained(self, tf_weight_path:str=Path(__file__).parent.parent / "datasets" / "wiki"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Mute tf flag info
    # load from tensorflow
    import tensorflow as tf
    import numpy as np

    state_dict = {}
    for name, _ in tf.train.list_variables(str(tf_weight_path)):
      state_dict[name] = tf.train.load_variable(str(tf_weight_path), name)

    for k, v in state_dict.items():
      m = k.split("/")
      if any(n in ["adam_v", "adam_m", "global_step", "LAMB", "LAMB_1", "beta1_power", "beta2_power"] for n in m):
        continue

      pointer = self
      n = m[-1] # this is just to stop python from complaining about possibly unbound local variable
      for i, n in enumerate(m):
        if re.fullmatch(r'[A-Za-z]+_\d+', n):
          l = re.split(r'_(\d+)', n)[:-1]
        else:
          l = [n]
        if l[0] in ["kernel", "gamma", "output_weights"]:
          pointer = getattr(pointer, "weight")
        elif l[0] in ["output_bias", "beta"]:
          pointer = getattr(pointer, "bias")
        elif l[0] == "pooler":
          pointer = getattr(getattr(self, "cls"), "pooler")
        else:
          pointer = getattr(pointer, l[0])
        if len(l) == 2: # layers
          pointer = pointer[int(l[1])]
      if n[-11:] == "_embeddings":
        pointer = getattr(pointer, "weight")
      elif n == "kernel":
        v = np.transpose(v)
      cast(Tensor, pointer).assign(v).realize()

    params = get_parameters(self)
    count = 0
    for p in params:
      param_count = 1
      for s in p.shape:
        param_count *= s
      count += param_count
    print(f"Total parameters: {count / 1000 / 1000}M")
    return self

class BertPreTrainingHeads:
  def __init__(self, hidden_size:int, vocab_size:int, embeddings_weight:Tensor):
    self.predictions = BertLMPredictionHead(hidden_size, vocab_size, embeddings_weight)
    self.pooler = BertPooler(hidden_size)
    self.seq_relationship = Linear(hidden_size, 2)

  def __call__(self, sequence_output:Tensor, masked_lm_positions:Tensor):
    prediction_logits = self.predictions(gather(sequence_output, masked_lm_positions))
    seq_relationship_logits = self.seq_relationship(self.pooler(sequence_output))
    return prediction_logits, seq_relationship_logits

class BertLMPredictionHead:
  def __init__(self, hidden_size:int, vocab_size:int, embeddings_weight:Tensor):
    self.transform = BertPredictionHeadTransform(hidden_size)
    self.embedding_weight = embeddings_weight
    self.bias = Tensor.zeros(vocab_size, dtype=dtypes.float32)

  def __call__(self, hidden_states:Tensor):
    return self.transform(hidden_states) @ self.embedding_weight.T + self.bias

class BertPredictionHeadTransform:
  def __init__(self, hidden_size:int):
    self.dense = Linear(hidden_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

  def __call__(self, hidden_states:Tensor):
    return self.LayerNorm(gelu(self.dense(hidden_states)))

class BertPooler:
  def __init__(self, hidden_size:int):
    self.dense = Linear(hidden_size, hidden_size)

  def __call__(self, hidden_states:Tensor):
    return self.dense(hidden_states[:, 0]).tanh()

def gather(prediction_logits:Tensor, masked_lm_positions:Tensor):
  counter = Tensor.arange(prediction_logits.shape[1], device=prediction_logits.device, requires_grad=False).reshape(1, 1, prediction_logits.shape[1]).expand(*masked_lm_positions.shape, prediction_logits.shape[1])
  onehot = counter == masked_lm_positions.unsqueeze(2).expand(*masked_lm_positions.shape, prediction_logits.shape[1])
  return onehot @ prediction_logits

class Bert:
  def __init__(self, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob):
    self.embeddings = BertEmbeddings(hidden_size, max_position_embeddings, type_vocab_size, vocab_size, hidden_dropout_prob)
    self.encoder = BertEncoder(hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob)

  def __call__(self, input_ids, attention_mask, token_type_ids):
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output = self.embeddings(input_ids, token_type_ids)
    encoder_outputs = self.encoder(embedding_output, extended_attention_mask)

    return encoder_outputs

class BertEmbeddings:
  def __init__(self, hidden_size, max_position_embeddings, type_vocab_size, vocab_size,  hidden_dropout_prob):
    self.word_embeddings = Embedding(vocab_size, hidden_size)
    self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
    self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, input_ids, token_type_ids):
    input_shape = input_ids.shape
    seq_length = input_shape[1]

    position_ids = Tensor.arange(seq_length, requires_grad=False, device=input_ids.device).unsqueeze(0).expand(*input_shape)
    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = embeddings.dropout(self.dropout)
    return embeddings

class BertEncoder:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob):
    self.layer = [BertLayer(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(num_hidden_layers)]

  def __call__(self, hidden_states, attention_mask):
    for layer in self.layer:
      hidden_states = layer(hidden_states, attention_mask)
    return hidden_states

class BertLayer:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
    self.intermediate = BertIntermediate(hidden_size, intermediate_size)
    self.output = BertOutput(hidden_size, intermediate_size, hidden_dropout_prob)

  def __call__(self, hidden_states, attention_mask):
    attention_output = self.attention(hidden_states, attention_mask)
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output

class BertOutput:
  def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
    self.dense = Linear(intermediate_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def gelu(x):
  return x * 0.5 * (1.0 + (x / 1.41421).erf())

class BertIntermediate:
  def __init__(self, hidden_size, intermediate_size):
    self.dense = Linear(hidden_size, intermediate_size)

  def __call__(self, hidden_states):
    x = self.dense(hidden_states)
    # tinygrad gelu is openai gelu but we need the original bert gelu
    return gelu(x)

class BertAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
    self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

  def __call__(self, hidden_states, attention_mask):
    self_output = self.self(hidden_states, attention_mask)
    attention_output = self.output(self_output, hidden_states)
    return attention_output

class BertSelfAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
    self.num_attention_heads = num_attention_heads
    self.attention_head_size = int(hidden_size / num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(hidden_size, self.all_head_size)
    self.key = Linear(hidden_size, self.all_head_size)
    self.value = Linear(hidden_size, self.all_head_size)

    self.dropout = attention_probs_dropout_prob

  def __call__(self, hidden_states, attention_mask):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    context_layer = Tensor.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, self.dropout)

    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(context_layer.shape[0], context_layer.shape[1], self.all_head_size)

    return context_layer

  def transpose_for_scores(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.num_attention_heads, self.attention_head_size)
    return x.transpose(1, 2)

class BertSelfOutput:
  def __init__(self, hidden_size, hidden_dropout_prob):
    self.dense = Linear(hidden_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states
