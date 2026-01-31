import functools, multiprocessing
from transformers import AutoTokenizer
from datasets import load_dataset
from tinygrad.apps.llm import SimpleTokenizer
from tinygrad.helpers import tqdm, getenv, partition

@functools.cache
def get_tokenizers():
  print("getting tokenizers")
  base_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
  special_tokens, normal_tokens = partition(((t, tid) for t, tid in base_tokenizer.vocab.items()), lambda e: e[1] in base_tokenizer.all_special_ids)
  simple_tokenizer = SimpleTokenizer(dict(normal_tokens), dict(special_tokens))
  return base_tokenizer, simple_tokenizer

def test_tokenize(samp) -> bool:
  base_tokenizer, simple_tokenizer = get_tokenizers()
  idx, txt = samp
  try: simple_tokens = tuple(simple_tokenizer.encode(txt))
  except RuntimeError: simple_tokens = ()
  base_tokens = tuple(base_tokenizer.encode(txt, add_special_tokens=False))
  if simple_tokens != base_tokens:
    print(f"tokens mismatch at index: {idx}.\n")
    color_codes = [91, 92, 94, 93, 95]
    def color_tokens(tids):
      return "".join(f"\033[{color_codes[i%len(color_codes)]}m{base_tokenizer.decode([t])}" for i, t in enumerate(tids)) + "\033[0m"
    print("simple:  ", color_tokens(simple_tokens))
    print("official:", color_tokens(base_tokens) + "\n")
    return False
  if simple_tokenizer.decode(simple_tokens) != txt:
    print(f"decode mismatch at {idx}")
    return False
  return True

# use ALLOW_FAILED=-1 to go over the entire dataset without printing.
if __name__ == "__main__":
  print("loading datasets")
  ds = load_dataset("OpenAssistant/oasst1")
  loaded_ds = [(idx, el["text"]) for idx, el in enumerate(ds["train"])]
  print(f"loaded {len(loaded_ds)}")

  allow_failed = getenv("ALLOW_FAILED", 10)
  fail_count, total = 0, 0
  with multiprocessing.Pool(16) as pool:
    for good in tqdm(pool.imap_unordered(test_tokenize, loaded_ds), total=len(loaded_ds)):
      total += 1
      if not good:
        fail_count += 1
        allow_failed -= 1
        if allow_failed == 0: break
    print(f"{fail_count}/{total} samples are inconsistent with the official tokenizer.")
