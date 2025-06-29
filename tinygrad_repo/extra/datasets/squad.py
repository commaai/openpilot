import json
import os
from pathlib import Path
from transformers import BertTokenizer
import numpy as np
from tinygrad.helpers import fetch

BASEDIR = Path(__file__).parent / "squad"
def init_dataset():
  os.makedirs(BASEDIR, exist_ok=True)
  fetch("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json", BASEDIR / "dev-v1.1.json")
  with open(BASEDIR / "dev-v1.1.json") as f:
    data = json.load(f)["data"]

  examples = []
  for article in data:
    for paragraph in article["paragraphs"]:
      text = paragraph["context"]
      doc_tokens = []
      prev_is_whitespace = True
      for c in text:
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False

      for qa in paragraph["qas"]:
        qa_id = qa["id"]
        q_text = qa["question"]

        examples.append({
          "id": qa_id,
          "question": q_text,
          "context": doc_tokens,
          "answers": list(map(lambda x: x["text"], qa["answers"]))
        })
  return examples

def _check_is_max_context(doc_spans, cur_span_index, position):
  best_score, best_span_index = None, None
  for di, (doc_start, doc_length) in enumerate(doc_spans):
    end = doc_start + doc_length - 1
    if position < doc_start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = di
  return cur_span_index == best_span_index

def convert_example_to_features(example, tokenizer):
  query_tokens = tokenizer.tokenize(example["question"])

  if len(query_tokens) > 64:
    query_tokens = query_tokens[:64]

  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  for i, token in enumerate(example["context"]):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)

  max_tokens_for_doc = 384 - len(query_tokens) - 3

  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append((start_offset, length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, 128)

  outputs = []
  for di, (doc_start, doc_length) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_length):
      split_token_index = doc_start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
      token_is_max_context[len(tokens)] = _check_is_max_context(doc_spans, di, split_token_index)
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < 384:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == 384
    assert len(input_mask) == 384
    assert len(segment_ids) == 384

    outputs.append({
      "input_ids": np.expand_dims(np.array(input_ids), 0).astype(np.float32),
      "input_mask": np.expand_dims(np.array(input_mask), 0).astype(np.float32),
      "segment_ids": np.expand_dims(np.array(segment_ids), 0).astype(np.float32),
      "token_to_orig_map": token_to_orig_map,
      "token_is_max_context": token_is_max_context,
      "tokens": tokens,
    })

  return outputs

def iterate(tokenizer, start=0):
  examples = init_dataset()
  print(f"there are {len(examples)} pairs in the dataset")

  for i in range(start, len(examples)):
    example = examples[i]
    features = convert_example_to_features(example, tokenizer)
    # we need to yield all features here as the f1 score is the maximum over all features
    yield features, example

if __name__ == "__main__":
  tokenizer = BertTokenizer(str(Path(__file__).parents[2] / "weights" / "bert_vocab.txt"))

  X, Y = next(iterate(tokenizer))
  print(" ".join(X[0]["tokens"]))
  print(X[0]["input_ids"].shape, Y)
