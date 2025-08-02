from transformers import AutoTokenizer
from datasets import load_dataset
from tinygrad.apps.llm import SimpleTokenizer
from tinygrad.helpers import tqdm, getenv

# use ALLOW_FAILED=-1 to go over the entire dataset without printing.
if __name__ == "__main__":
  base_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
  vocab_words = [ word for word, _ in sorted(base_tokenizer.get_vocab().items(), key=lambda t: t[1]) ]
  inv_vocab = { tid: word for word, tid in base_tokenizer.get_vocab().items() }
  simple_tokenizer = SimpleTokenizer(vocab_words)

  color_codes = [ 91, 92, 94, 93, 95 ]
  def color_tokens(tids): return "".join(f"\033[{color_codes[i%len(color_codes)]}m{inv_vocab[t]}" for i, t in enumerate(tids)) + "\033[0m"

  ds = load_dataset("OpenAssistant/oasst1")
  allow_failed = getenv("ALLOW_FAILED", 10)

  fail_count, total = 0, 0

  for idx, el in enumerate(tqdm(ds["train"])):
    total += 1

    try: simple_tokens = tuple(simple_tokenizer.encode(el["text"]))
    except RuntimeError: simple_tokens = ()
    base_tokens = tuple(base_tokenizer.encode(el["text"], add_special_tokens=False))

    if simple_tokens != base_tokens:
      fail_count += 1
      allow_failed -= 1

      if allow_failed >= 0:
        print(f"tokens mismatch at index: {idx}.\n")

        print("simple:  ", color_tokens(simple_tokens))
        print("official:", color_tokens(base_tokens) + "\n")

      if allow_failed == 0: break
  print(f"{fail_count}/{total} samples are inconsistent with the official tokenizer.")
