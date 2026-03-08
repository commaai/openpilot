# Preprocessing of downloaded text from Wikipedia for MLPerf BERT training
# This is a modified version of the original script:
# https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/cleanup_scripts/create_pretraining_data.py
# ENV VARS:
# MAX_SEQ_LENGTH          - Maximum sequence length
# MAX_PREDICTIONS_PER_SEQ - Maximum number of masked LM predictions per sequence
# RANDOM_SEED             - Random seed
# DUPE_FACTOR             - Number of times to duplicate the input data with different masks
# MASKED_LM_PROB          - Probability of masking a token
# SHORT_SEQ_PROB          - Probability of picking a sequence shorter than MAX_SEQ_LENGTH

import os, sys, pickle, random, unicodedata
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinygrad.helpers import diskcache, getenv

BASEDIR = getenv('BASEDIR', Path(__file__).parent / "wiki")

################### Tokenization #####################

def _is_whitespace(char:str) -> bool:
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char:str) -> bool:
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _is_punctuation(char:str) -> bool:
  # range(33, 48) -> ! " # $ % & ' ( ) * + , - . /
  # range(58, 65) -> : ; < = > ? @
  # range(91, 97) -> [ \ ] ^ _
  # range(123, 127) -> { | } ~
  if (cp := ord(char)) in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
    return True
  return unicodedata.category(char).startswith("P")

def _is_chinese_char(cp:int) -> bool:
  if ((cp >= 0x4E00 and cp <= 0x9FFF) or
      (cp >= 0x3400 and cp <= 0x4DBF) or
      (cp >= 0x20000 and cp <= 0x2A6DF) or
      (cp >= 0x2A700 and cp <= 0x2B73F) or
      (cp >= 0x2B740 and cp <= 0x2B81F) or
      (cp >= 0x2B820 and cp <= 0x2CEAF) or
      (cp >= 0xF900 and cp <= 0xFAFF) or
      (cp >= 0x2F800 and cp <= 0x2FA1F)):
    return True
  return False

def _run_split_on_punc(text:str) -> list[str]:
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

def _run_strip_accents(text:str) -> str:
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text:str) -> str:
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)

def _tokenize_chinese_chars(text:str) -> str:
  output = []
  for char in text:
    cp = ord(char)
    if _is_chinese_char(cp):
      output.append(" ")
      output.append(char)
      output.append(" ")
    else:
      output.append(char)
  return "".join(output)

def whitespace_tokenize(text):
  if not (text := text.strip()): return []
  return text.split()

def _wordpiece_tokenize(text:str, vocab:dict[str, int]) -> list[str]:
  text = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
  output_tokens = []
  for token in text.strip().split():
    chars = list(token)
    if len(chars) > 200:
      output_tokens.append("[UNK]")
      continue

    is_bad = False
    start = 0
    sub_tokens = []
    while start < len(chars):
      end = len(chars)
      cur_substr = None
      while start < end:
        substr = "".join(chars[start:end])
        if start > 0: substr = "##" + substr
        if substr in vocab:
          cur_substr = substr
          break
        end -= 1
      if cur_substr is None:
        is_bad = True
        break
      sub_tokens.append(cur_substr)
      start = end

    if is_bad: output_tokens.append("[UNK]")
    else: output_tokens.extend(sub_tokens)
  return output_tokens

class Tokenizer:
  def __init__(self, vocab_file):
    self.vocab = {}
    with open(vocab_file) as f:
      for line in f:
        line = line.decode("utf-8", "ignore") if isinstance(line, bytes) else line
        if (token := line.strip()) and token not in self.vocab: self.vocab[token] = len(self.vocab)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text:str) -> list[str]:
    # BasicTokenizer
    split_tokens = []
    for token in whitespace_tokenize(_tokenize_chinese_chars(_clean_text(text.decode("utf-8", "ignore") if isinstance(text, bytes) else text))):
      split_tokens.extend(_run_split_on_punc(_run_strip_accents(token.lower())))
    split_tokens = " ".join(split_tokens).strip().split()
    # WordpieceTokenizer
    tokens = []
    for token in split_tokens:
      tokens.extend(_wordpiece_tokenize(token, self.vocab))
    return tokens

  def convert_tokens_to_ids(self, tokens:list[str]) -> list[int]: return [self.vocab[token] for token in tokens]
  def convert_ids_to_tokens(self, ids:list[int]) -> list[str]: return [self.inv_vocab[id] for id in ids]

##################### Feature transformation #####################

def truncate_seq_pair(tokens_a:list[str], tokens_b:list[str], max_num_tokens:int, rng:random.Random) -> None:
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_masked_lm_predictions(tokens:list[str], tokenizer:Tokenizer, rng:random.Random, vocab_words:list[str]) -> tuple[list[str], list[int], list[str]]:
  cand_indices = []
  for i, token in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indices.append(i)

  rng.shuffle(cand_indices)
  output_tokens = list(tokens)
  num_to_predict = min(getenv('MAX_PREDICTIONS_PER_SEQ', 76), max(1, int(round(len(tokens) * getenv("MASKED_LM_PROB", 0.15)))))

  masked_lms = []
  covered_indices = set()
  for index in cand_indices:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indices:
      continue
    covered_indices.add(index)

    masked_token = None
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      if rng.random() < 0.5:
        masked_token = tokens[index]
      else:
        masked_token = vocab_words[rng.randint(0, len(tokenizer.vocab) - 1)]

    output_tokens[index] = masked_token
    masked_lms.append((index, tokens[index]))
  masked_lms = sorted(masked_lms, key=lambda x: x[0])

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p[0])
    masked_lm_labels.append(p[1])

  return output_tokens, masked_lm_positions, masked_lm_labels

def create_instances_from_document(rng:random.Random, tokenizer:Tokenizer, doc:list[str], di:int, documents:list[list[str]]) -> list[dict]:
  max_num_tokens = getenv('MAX_SEQ_LENGTH', 512) - 3 # [CLS] + 2 * [SEP]

  target_seq_length = max_num_tokens
  if rng.random() < getenv("SHORT_SEQ_PROB", 0.1):
    target_seq_length = rng.randint(2, max_num_tokens)

  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(doc):
    segment = doc[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(doc) - 1 or current_length >= target_seq_length:
      if current_chunk:
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          for _ in range(10):
            random_document_index = rng.randint(0, len(documents) - 1)
            if random_document_index != di:
              break

          random_document = documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break

          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, tokenizer, rng, list(tokenizer.vocab.keys()))
        instances.append({
          "tokens": tokens,
          "segment_ids": segment_ids,
          "masked_lm_positions": masked_lm_positions,
          "masked_lm_labels": masked_lm_labels,
          "is_random_next": is_random_next
        })
      current_chunk = []
      current_length = 0
    i += 1
  return instances

def get_documents(rng:random.Random, tokenizer:Tokenizer, fn:str) -> list[list[str]]:
  documents = [[]]
  with open(BASEDIR / fn) as f:
    for line in f.readlines():
      if not (line := line.decode("utf-8", "ignore") if isinstance(line, bytes) else line): break
      if not (line := line.strip()): documents.append([])
      if (tokens := tokenizer.tokenize(line)): documents[-1].append(tokens)
  documents = [x for x in documents if x]
  rng.shuffle(documents)
  return documents

def get_instances(rng:random.Random, tokenizer:Tokenizer, documents:list[list[str]]) -> list[dict]:
  instances = []
  for _ in range(getenv('DUPE_FACTOR', 10)):
    for di, doc in enumerate(documents):
      instances.extend(create_instances_from_document(rng, tokenizer, doc, di, documents))
  rng.shuffle(instances)
  return instances

def instance_to_features(instance:dict, tokenizer:Tokenizer) -> dict:
  input_ids = tokenizer.convert_tokens_to_ids(instance["tokens"])
  input_mask = [1] * len(input_ids)
  segment_ids = instance["segment_ids"]

  max_seq_length = getenv('MAX_SEQ_LENGTH', 512)

  assert len(input_ids) <= max_seq_length
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  masked_lm_positions = instance["masked_lm_positions"]
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance["masked_lm_labels"])
  masked_lm_weights = [1.0] * len(masked_lm_ids)

  while len(masked_lm_positions) < getenv("MAX_PREDICTIONS_PER_SEQ", 76):
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  next_sentence_label = 1 if instance["is_random_next"] else 0

  return {
    "input_ids": np.expand_dims(np.array(input_ids, dtype=np.int32), 0),
    "input_mask": np.expand_dims(np.array(input_mask, dtype=np.int32), 0),
    "segment_ids": np.expand_dims(np.array(segment_ids, dtype=np.int32), 0),
    "masked_lm_positions": np.expand_dims(np.array(masked_lm_positions, dtype=np.int32), 0),
    "masked_lm_ids": np.expand_dims(np.array(masked_lm_ids, dtype=np.int32), 0),
    "masked_lm_weights": np.expand_dims(np.array(masked_lm_weights, dtype=np.float32), 0),
    "next_sentence_labels": np.expand_dims(np.array([next_sentence_label], dtype=np.int32), 0),
  }

def process_part(part:int):
  tokenizer = Tokenizer(getenv("BASEDIR", Path(__file__).parent / "wiki") / "vocab.txt")
  os.makedirs(BASEDIR / "train", exist_ok=True)

  if os.path.exists(BASEDIR / f"train/{str(part)}.pkl"): return
  features = get_features_from_part(tokenizer, val=False, part=part)
  with open(BASEDIR / f"train/{str(part)}.pkl", "wb") as f:
    pickle.dump(features, f)

def get_features_from_part(tokenizer:Tokenizer, val:bool=False, part:int=0) -> list[dict]: # Convert raw text to masked NSP samples
  rng = random.Random(getenv('RANDOM_SEED', 12345))

  if val:
    tqdm.write("Getting samples from dataset")
    documents = get_documents(rng, tokenizer, "results4/eval.txt")
    instances = get_instances(rng, tokenizer, documents)

    tqdm.write(f"There are {len(instances)} samples in the dataset")
    tqdm.write(f"Picking 10000 samples")

    pick_ratio = len(instances) / 10000
    return [instance_to_features(instances[int(inst*pick_ratio)], tokenizer) for inst in range(10000)]
  else:
    documents = get_documents(rng, tokenizer, f"results4/part-{part:05d}-of-00500")
    instances = get_instances(rng, tokenizer, documents)
    return [instance_to_features(instance, tokenizer) for instance in instances]

##################### Load files #####################

@diskcache
def get_wiki_train_files(): return sorted(list((BASEDIR / "train/").glob("*.pkl")))

if __name__ == "__main__":
  tokenizer = Tokenizer(getenv("BASEDIR", Path(__file__).parent / "wiki") / "vocab.txt")

  assert len(sys.argv) > 1, "Usage: python wikipedia.py pre-eval|pre-train [part]|all"

  if sys.argv[1] == "pre-eval": # Generate 10000 eval samples
    with open(BASEDIR / "eval.pkl", "wb") as f:
      pickle.dump(get_features_from_part(tokenizer, val=True), f)
  elif sys.argv[1] == "pre-train":
    if sys.argv[2] == "all": # Use all 500 parts for training generation
      process_map(process_part, [part for part in range(500)], max_workers=getenv('NUM_WORKERS', min(os.cpu_count(), 32)), chunksize=1)
    else: # Use a specific part for training generation
      part = sys.argv[2]
      print(f"Processing part {part}...")
      process_part(int(part))
