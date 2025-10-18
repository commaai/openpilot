from collections import OrderedDict
import unicodedata
from typing import Optional
import math
import numpy as np
from tinygrad.nn import state
from tinygrad.tensor import Tensor, dtypes
from tinygrad.helpers import getenv

#
# checkpointing utils
#

def invert_dict(d): return {v: k for k, v in reversed(d.items())}
def dedup_dict(d): return invert_dict(invert_dict(d))
# store each tensor into the first key it appears in
def get_training_state(model, optimizer, scheduler):
  # hack: let get_state_dict walk the tree starting with model, so that the checkpoint keys are
  # readable and can be loaded as a model for eval
  train_state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
  return dedup_dict(state.get_state_dict(train_state))
def load_training_state(model, optimizer, scheduler, state_dict):
  # use fresh model to restore duplicate keys
  train_state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
  big_dict = state.get_state_dict(train_state)
  # hack: put back the dupes
  dupe_names = {}
  for k, v in big_dict.items():
    if v not in dupe_names:
      dupe_names[v] = k
      assert k in state_dict
    state_dict[k] = state_dict[dupe_names[v]]
  # scheduler contains optimizer and all params, load each weight only once
  scheduler_state = {'scheduler': scheduler}
  state.load_state_dict(scheduler_state, state_dict)

def gaussian_kernel(n, std):
  from scipy import signal
  gaussian_1d = signal.windows.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d

def prepare_arrays(image, roi_shape=(128, 128, 128)):
  assert len(roi_shape) == 3 and any(roi_shape)
  image_shape = list(image.shape[2:])
  result = np.zeros((1, 3, *image_shape), dtype=image.dtype)
  norm_map = np.zeros_like(result)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)
  return result, norm_map, norm_patch

def get_slice(image, roi_shape=(128, 128, 128), overlap_factor=0.5):
  assert len(roi_shape) == 3 and any(roi_shape)
  assert 0 < overlap_factor < 1
  image_shape, dim = list(image.shape[2:]), len(image.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap_factor)) for i in range(dim)]
  size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        yield i, j, k

def _get_best_indices(logits, n_best_size):
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
  return list(map(lambda x: x[0], index_and_score))[:n_best_size]

def _is_punctuation(char):
  if (cp := ord(char)) in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
    return True
  return unicodedata.category(char).startswith("P")

def _is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _run_split_on_punc(text):
  if text in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
    return [text]
  start_new_word = True
  output = []
  for i in range(len(text)):
    if _is_punctuation(char := text[i]):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
  return ["".join(x) for x in output]

def _run_strip_accents(text):
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text):
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)

def _get_final_text(pred_text, orig_text):
  def _strip_spaces(text):
    ns_text = ""
    ns_to_s_map = OrderedDict()
    for i, c in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_text)] = i
      ns_text += c
    return ns_text, ns_to_s_map

  orig_tokens = _clean_text(orig_text).strip().split()
  split_tokens = []
  for token in orig_tokens:
    if token not in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
      token = token.lower()
      token = _run_strip_accents(token)
    split_tokens.extend(_run_split_on_punc(token))

  tok_text = " ".join(" ".join(split_tokens).strip().split())
  start_position = tok_text.find(pred_text)
  if start_position == -1:
    return orig_text
  end_position = start_position + len(pred_text) - 1

  orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
  tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)
  if len(orig_ns_text) != len(tok_ns_text):
    return orig_text
  tok_s_to_ns_map = {v: k for k, v in tok_ns_to_s_map.items()}

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    if (ns_start_position := tok_s_to_ns_map[start_position]) in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]
  if orig_start_position is None:
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    if (ns_end_position := tok_s_to_ns_map[end_position]) in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]
  if orig_end_position is None:
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text

def get_bert_qa_prediction(features, example, start_end_logits):
  prelim_predictions = []
  for i, feature in enumerate(features):
    for start_index in _get_best_indices(start_end_logits[i][0], 20):
      for end_index in _get_best_indices(start_end_logits[i][1], 20):
        if start_index >= len(feature["tokens"]) or end_index >= len(feature["tokens"]):
          continue
        if start_index not in feature["token_to_orig_map"] or end_index not in feature["token_to_orig_map"]:
          continue
        if not feature["token_is_max_context"].get(start_index, False):
          continue
        if end_index < start_index or end_index - start_index + 1 > 30:
          continue

        prelim_predictions.append({
          "feature_index": i,
          "start_index": start_index,
          "end_index": end_index,
          "start_logit": start_end_logits[i][0, start_index],
          "end_logit": start_end_logits[i][1, end_index]
        })
  predictions = sorted(prelim_predictions, key=lambda x: (x["start_logit"] + x["end_logit"]), reverse=True)

  if len(predictions) > 0:
    feature = features[predictions[0]["feature_index"]]
    tok_tokens = feature["tokens"][predictions[0]["start_index"]:(predictions[0]["end_index"] + 1)]
    orig_doc_start = feature["token_to_orig_map"][predictions[0]["start_index"]]
    orig_doc_end = feature["token_to_orig_map"][predictions[0]["end_index"]]
    orig_tokens = example["context"][orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens).replace(" ##", "").replace("##", "")
    tok_text = " ".join(tok_text.strip().split())
    orig_text = " ".join(orig_tokens)
    return _get_final_text(tok_text, orig_text)
  return "empty"

def get_mlperf_bert_config():
  """benchmark is BERT-large"""
  ret = {"attention_probs_dropout_prob": 0.1, "hidden_dropout_prob": 0.1, "vocab_size": 30522, "type_vocab_size": 2, "max_position_embeddings": 512}

  match (bert_size:=getenv("BERT_SIZE", "large")):
    case "large": ret.update({"hidden_size": 1024, "intermediate_size": 4096, "num_attention_heads": 16, "num_hidden_layers": 24})
    case "tiny": ret.update({"hidden_size": 128, "intermediate_size": 512, "num_attention_heads": 2, "num_hidden_layers": 2})
    case _: raise RuntimeError(f"unhandled {bert_size=}")

  if (bert_layers:=getenv("BERT_LAYERS")): ret["num_hidden_layers"] = bert_layers
  return ret

def get_mlperf_bert_model():
  from extra.models import bert
  from examples.mlperf.initializers import LinearBert, EmbeddingBert, LayerNormBert

  bert.Linear = LinearBert
  bert.Embedding = EmbeddingBert
  bert.LayerNorm = LayerNormBert

  from extra.models.bert import BertForPretraining
  config = get_mlperf_bert_config()
  if getenv("DISABLE_DROPOUT", 0):
    config["hidden_dropout_prob"] = config["attention_probs_dropout_prob"] = 0.0
  return BertForPretraining(**config)

def get_fake_data_bert(BS:int):
  return {
    "input_ids": Tensor.empty((BS, 512), dtype=dtypes.int32, device="CPU"),
    "input_mask": Tensor.empty((BS, 512), dtype=dtypes.int32, device="CPU"),
    "segment_ids": Tensor.empty((BS, 512), dtype=dtypes.int32, device="CPU"),
    "masked_lm_positions": Tensor.empty((BS, 76), dtype=dtypes.int32, device="CPU"),
    "masked_lm_ids": Tensor.empty((BS, 76), dtype=dtypes.int32, device="CPU"),
    "masked_lm_weights": Tensor.empty((BS, 76), dtype=dtypes.float32, device="CPU"),
    "next_sentence_labels": Tensor.empty((BS, 1), dtype=dtypes.int32, device="CPU"),
  }

def find_matches(match_quality_matrix:np.ndarray, high_threshold:float=0.5, low_threshold:float=0.4, allow_low_quality_matches:bool=False) -> np.ndarray:
  BELOW_LOW_THRESHOLD, BETWEEN_THRESHOLDS = -1, -2

  def _set_low_quality_matches_(matches:np.ndarray, all_matches:np.ndarray, match_quality_matrix:np.ndarray):
    highest_quality_foreach_gt = np.max(match_quality_matrix, axis=1)
    pred_inds_to_update = np.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])[1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

  assert low_threshold <= high_threshold

  matched_vals, matches = match_quality_matrix.max(axis=0), match_quality_matrix.argmax(axis=0)
  all_matches = np.copy(matches) if allow_low_quality_matches else None
  below_low_threshold = matched_vals < low_threshold
  between_thresholds = (matched_vals >= low_threshold) & (matched_vals < high_threshold)
  matches[below_low_threshold] = BELOW_LOW_THRESHOLD
  matches[between_thresholds] = BETWEEN_THRESHOLDS

  if allow_low_quality_matches:
    assert all_matches is not None
    _set_low_quality_matches_(matches, all_matches, match_quality_matrix)

  return matches

def box_iou(boxes1:np.ndarray, boxes2:np.ndarray) -> np.ndarray:
  def _box_area(boxes:np.ndarray) -> np.ndarray: return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

  def _box_inter_union(boxes1:np.ndarray, boxes2:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    area1, area2 = _box_area(boxes1), _box_area(boxes2)
    lt, rb = np.maximum(boxes1[:, None, :2], boxes2[:, :2]), np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter, union

  inter, union = _box_inter_union(boxes1, boxes2)
  return inter / union

def generate_anchors(input_size:tuple[int, int], scales:Optional[tuple[Tensor, ...]]=None, aspect_ratios:Optional[tuple[Tensor, ...]]=None) -> list[np.ndarray]:
  def _compute_grid_sizes(input_size:tuple[int, int]) -> np.ndarray:
    return np.ceil(np.array(input_size)[None, :] / 2 ** np.arange(3, 8)[:, None])

  scales = tuple((i, int(i * 2 ** (1/3)), int(i * 2 ** (2/3))) for i in 2 ** np.arange(5, 10)) if scales is None else scales
  aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales) if aspect_ratios is None else aspect_ratios
  aspect_ratios = tuple(ar for ar in aspect_ratios)
  grid_sizes = _compute_grid_sizes(input_size)

  assert len(scales) == len(aspect_ratios) == len(grid_sizes), "scales, aspect_ratios, and grid_sizes must have the same length"

  anchors = []
  for s, ar, gs in zip(scales, aspect_ratios, grid_sizes):
    s, ar = np.array(s), np.array(ar)
    h_ratios = np.sqrt(ar)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[:, None] * s[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * s[None, :]).reshape(-1)
    base_anchors = (np.stack([-ws, -hs, ws, hs], axis=1) / 2).round()
    stride_h, stride_w = input_size[0] // gs[0], input_size[1] // gs[1]
    shifts_x, shifts_y = np.meshgrid(np.arange(gs[1]) * stride_w, np.arange(gs[0]) * stride_h)
    shifts_x, shifts_y = shifts_x.reshape(-1), shifts_y.reshape(-1)
    shifts = np.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis=1, dtype=np.float32)
    anchors.append((shifts[:, None] + base_anchors[None, :]).reshape(-1, 4))

  return anchors


class BoxCoder(object):
  def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16), apply_to_remove=True):
    self.weights = weights
    self.bbox_xform_clip = bbox_xform_clip
    self.apply_to_remove = apply_to_remove

  def encode(self, reference_boxes, proposals):
    TO_REMOVE = self.apply_to_remove  # TODO remove
    ex_widths = proposals[..., 2] - proposals[..., 0] + TO_REMOVE
    ex_heights = proposals[..., 3] - proposals[..., 1] + TO_REMOVE
    ex_ctr_x = proposals[..., 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[..., 1] + 0.5 * ex_heights

    gt_widths = reference_boxes[..., 2] - reference_boxes[..., 0] + TO_REMOVE
    gt_heights = reference_boxes[..., 3] - reference_boxes[..., 1] + TO_REMOVE
    gt_ctr_x = reference_boxes[..., 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[..., 1] + 0.5 * gt_heights

    wx, wy, ww, wh = self.weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * Tensor.log(gt_widths / ex_widths)
    targets_dh = wh * Tensor.log(gt_heights / ex_heights)

    targets = Tensor.stack(targets_dx, targets_dy, targets_dw, targets_dh, dim=-1)
    return targets

  def decode(self, rel_codes, boxes):
    boxes = boxes.cast(rel_codes.dtype)
    rel_codes = rel_codes

    TO_REMOVE = self.apply_to_remove  # TODO remove
    widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
    heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = self.weights
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into Tensor.exp()
    dw = dw.clip(min_=dw.min(), max_=self.bbox_xform_clip)
    dh = dh.clip(min_=dh.min(), max_=self.bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = dw.exp() * widths[:, None]
    pred_h = dh.exp() * heights[:, None]
    x = pred_ctr_x - 0.5 * pred_w
    y = pred_ctr_y - 0.5 * pred_h
    w = pred_ctr_x + 0.5 * pred_w - 1
    h = pred_ctr_y + 0.5 * pred_h - 1
    pred_boxes = Tensor.stack(x, y, w, h).permute(1,2,0).reshape(rel_codes.shape[0], rel_codes.shape[1])
    return pred_boxes
