import unittest, base64, functools, re, sys, time, unicodedata
from tinygrad.llm.cli import SimpleTokenizer, FallbackTemplate
from tinygrad.helpers import fetch

@unittest.skipIf(sys.platform == 'win32', "fetch race condition on Windows")
class TestLLMTokenizer(unittest.TestCase):
  @functools.cached_property
  def llama_tok(self):
    # from https://github.com/tinygrad/tinygrad/blob/e0106b6b257ebc003eb3694144e3e198f7d8cc37/examples/llama3.py#L14
    model_file = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model")
    with open(model_file, "rt") as fd:
      str_vocab = [line.split(maxsplit=1) for line in fd.read().splitlines() if line]

      # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
      bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
      _byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}
      _byte_encoder = {v:k for k,v in _byte_decoder.items()}
      normal_tokens = {''.join([_byte_encoder[x] for x in base64.b64decode(stok)]): int(srank) for stok, srank in str_vocab}

    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [ f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5) ]
    return SimpleTokenizer(normal_tokens, {token: len(normal_tokens) + i for i, token in enumerate(special_tokens)})

  def _test_coding(self, tok: SimpleTokenizer, text: str, expected_tokens: list[int]):
    self.assertEqual(tok.encode(text), expected_tokens)
    self.assertEqual(tok.decode(expected_tokens), text)

  # NOTE: the correct tokenization for this can only be found by looking up the text chunk in the vocab, not by applying merges
  def test_llama_early_tokenize(self): self._test_coding(self.llama_tok, " например", [ 111797 ])

  def test_llama_basic(self): self._test_coding(self.llama_tok, "hello world", [ 15339, 1917 ])
  def test_llama_control_char(self): self._test_coding(self.llama_tok, " \x850", [ 220, 116360, 15 ])
  def test_llama_bytes(self): self._test_coding(self.llama_tok, " \xec\x8b\xa4\xed", [ 1717, 105, 116174, 82638, 2483 ])
  def test_llama_special1(self): self._test_coding(self.llama_tok, "hello <|end_of_text|>", [ 15339, 220, 128001 ])
  def test_llama_special2(self): self._test_coding(self.llama_tok, "<|start_header_id|>user<|end_header_id|>\n\n", [ 128006, 882, 128007, 271 ])
  def test_llama_repeat(self): self._test_coding(self.llama_tok, "00000000000000000", [ 931, 931, 931, 931, 931, 410 ])
  def test_llama_pat(self): self._test_coding(self.llama_tok, "today\n  \n", [ 31213, 14211 ])

  def test_split_regex_matches_naive_listing(self):
    # the compacted codepoint ranges must match the same text as listing every codepoint
    def naive(pre): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + naive("Z"), naive("N"), naive("L")
    naive_re = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" +
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    sample = "hello world 한국어 中文 текст ١٢٣ 123 😊\n  \ttoday\n'équivalent ²³№ "
    self.assertEqual(SimpleTokenizer({}, {})._split_to_word.findall(sample), naive_re.findall(sample))

  def test_split_regex_speed(self):
    # the naive listing compiles a 429KB pattern that takes 10+s to match a 225KB prompt; ranges keep it small and fast
    tok = SimpleTokenizer({}, {})
    self.assertLess(len(tok._split_to_word.pattern), 100_000)
    text = "The quick brown fox jumps over the lazy dog. " * 5000
    tok._split_to_word.findall(text)  # warmup
    tms = []
    for _ in range(5):
      st = time.perf_counter()
      words = tok._split_to_word.findall(text)
      tms.append(time.perf_counter() - st)
    self.assertLess(min(tms), 4)  # best-of-5 is robust to CI scheduling pauses; new code takes ~60ms
    self.assertEqual(len(words), 50001)

  def test_llama_continued_conversation(self):
    self._test_coding(self.llama_tok, "hello <|eot_id|>world", [15339, 220, 128009, 14957])
    self._test_coding(self.llama_tok, "hello <|eot_id|>world again", [15339, 220, 128009, 14957, 1578])
    self._test_coding(self.llama_tok, "hello changed <|eot_id|>world again", [15339, 5614, 220, 128009, 14957, 1578])

  def test_long_cached_prompt_matches_fresh_tokenization(self):
    prefix = "system tools\n" * 700 + "<|eot_id|>"
    first, changed = prefix + "run tower of hanoi", prefix + "run ls /"
    expected = self.llama_tok.encode(changed)
    self.llama_tok.encode(first)
    self.assertEqual(self.llama_tok.encode(changed), expected)

  def test_tekken_from_gguf_kv(self):
    kv = {
      "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "hello"],
      "tokenizer.ggml.token_type": [3, 3, 3, 3, 3, 1],
      "tokenizer.ggml.pre": "tekken",
      "tokenizer.ggml.eos_token_id": 2,
    }
    tok = SimpleTokenizer.from_gguf_kv(kv)
    template = FallbackTemplate(tok)
    self.assertEqual(template.role("user"), "[INST]")
    self.assertEqual(tok.encode("hello"), [5])
    self.assertEqual(template.end_turn(), "[/INST]")
    self.assertEqual(template.role("assistant"), "")

  def test_stream_decoder(self):
    """stream_decoder buffers incomplete UTF-8: token 25677 has 3/4 of emoji, token 138 completes it."""
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    be = {b: chr(b) for b in bs} | {b: chr(256+i) for i,b in enumerate(b for b in range(256) if b not in bs)}
    token_bytes = {25677: b'\x20\xf0\x9f\x98', 138: b'\x8a'}  # ' ' + 3/4 emoji | 1/4 emoji (qwen3.5)
    tok = SimpleTokenizer({"".join(be[b] for b in v): k for k, v in token_bytes.items()}, {})
    dec = tok.stream_decoder()
    self.assertEqual(dec(25677) + dec(138) + dec(), " 😊")

if __name__ == '__main__':
  unittest.main()
