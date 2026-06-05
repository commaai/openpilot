import re, string
from collections import Counter
from tinygrad import Tensor

def levenshtein(a, b):
  n, m = len(a), len(b)
  if n > m:
    a, b, n, m = b, a, m, n

  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)

  return current[n]

def word_error_rate(x, y):
  scores = words = 0
  for h, r in zip(x, y):
    h_list = h.split()
    r_list = r.split()
    words += len(r_list)
    scores += levenshtein(h_list, r_list)
  return float(scores) / words, float(scores), words

def one_hot(x):
  return x.one_hot(3).squeeze(1).permute(0, 4, 1, 2, 3)

def dice_score(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6, argmax=True, to_one_hot_x=True):
  channel_axis, reduce_axis = 1, tuple(range(2, len(prediction.shape)))
  if argmax: prediction = prediction.argmax(axis=channel_axis)
  else: prediction = prediction.softmax(axis=channel_axis)
  if to_one_hot_x: prediction = one_hot(prediction)
  target = one_hot(target)
  prediction, target = prediction[:, 1:], target[:, 1:]
  assert prediction.shape == target.shape, f"prediction ({prediction.shape}) and target ({target.shape}) shapes do not match"
  intersection = (prediction * target).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return result

def normalize_string(s):
  s = "".join(c for c in s.lower() if c not in string.punctuation)
  s = re.sub(r'\b(a|an|the)\b', ' ', s)
  return " ".join(s.split())

def f1_score(x, y):
  xt = normalize_string(x).split()
  yt = normalize_string(y).split()
  ct = Counter(xt) & Counter(yt)
  if (ns := sum(ct.values())) == 0:
    return 0.0
  p = ns / len(xt)
  r = ns / len(yt)
  return 2 * p * r / (p + r)

def log_perplexity(logit:Tensor, target:Tensor, ignore_index:int|None=None):
  # logit has shape (n_samples, seq_len, vocab_size), target has shape (n_samples, seq_len)
  assert logit.ndim == 3, logit.ndim
  assert target.ndim == 2, target.ndim
  assert logit.shape[:2] == target.shape, f"{logit.shape[:2]=}, {target.shape=}"
  log_prob = logit.log_softmax(axis=-1)
  return log_prob.transpose(1, 2).nll_loss(target, ignore_index=ignore_index)