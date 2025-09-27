import unittest, base64, functools, sys
from tinygrad.apps.llm import SimpleTokenizer, get_llama_re
from tinygrad.helpers import fetch

@unittest.skipIf(sys.platform == 'win32', "fetch race condition on Windows")
class TestLLMTokenizer(unittest.TestCase):
  @functools.cached_property
  def basic_tok(self): return SimpleTokenizer(".*", { b"a": 0, b"b": 1, b"c": 2, b"ab": 3, b"bc": 4 }, { "<x>": 5, "<y>": 6, "<z>": 7 })

  @functools.cached_property
  def llama_tok(self):
    # from https://github.com/tinygrad/tinygrad/blob/e0106b6b257ebc003eb3694144e3e198f7d8cc37/examples/llama3.py#L14
    model_file = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model")
    with open(model_file, "rt") as fd:
      str_vocab = [ line.split(maxsplit=1) for line in fd.read().splitlines() if line ]
      normal_tokens = { base64.b64decode(stok): int(srank) for stok, srank in str_vocab }

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
    return SimpleTokenizer(get_llama_re(), normal_tokens, { token: len(normal_tokens) + i for i, token in enumerate(special_tokens) })

  def _test_coding(self, tok: SimpleTokenizer, text: str, expected_tokens: list[int]):
    self.assertEqual(tok.encode(text), expected_tokens)
    self.assertEqual(tok.decode(expected_tokens), text)

  def test_abc(self): self._test_coding(self.basic_tok, "abc", [ 3, 2 ])
  def test_abbc(self): self._test_coding(self.basic_tok, "abbc", [ 3, 4 ])
  def test_aabbbcc(self): self._test_coding(self.basic_tok, "aabbbcc", [ 0, 3, 1, 4, 2 ])
  def test_specials1(self): self._test_coding(self.basic_tok, "a<x>a<y>a<z>a", [ 0, 5, 0, 6, 0, 7, 0 ])
  def test_specials2(self): self._test_coding(self.basic_tok, "<x>a<y>a<z>", [ 5, 0, 6, 0, 7 ])
  def test_invalid_token(self):
    with self.assertRaises(RuntimeError): self._test_coding(self.basic_tok, "L", [])

  def test_no_specials(self): self._test_coding(SimpleTokenizer(".*", { bytes([i]): i for i in range(256) }, {}), "abc", [97, 98, 99])

  # NOTE: the correct tokenization for this can only be found by looking up the text chunk in the vocab, not by applying merges
  def test_llama_early_tokenize(self): self._test_coding(self.llama_tok, " например", [ 111797 ])

  def test_llama_basic(self): self._test_coding(self.llama_tok, "hello world", [ 15339, 1917 ])
  def test_llama_control_char(self): self._test_coding(self.llama_tok, " \x850", [ 220, 116360, 15 ])
  def test_llama_bytes(self): self._test_coding(self.llama_tok, " \xec\x8b\xa4\xed", [ 1717, 105, 116174, 82638, 2483 ])
  def test_llama_special1(self): self._test_coding(self.llama_tok, "hello <|end_of_text|>", [ 15339, 220, 128001 ])
  def test_llama_special2(self): self._test_coding(self.llama_tok, "<|start_header_id|>user<|end_header_id|>\n\n", [ 128006, 882, 128007, 271 ])
  def test_llama_repeat(self): self._test_coding(self.llama_tok, "00000000000000000", [ 931, 931, 931, 931, 931, 410 ])
  def test_llama_pat(self): self._test_coding(self.llama_tok, "today\n  \n", [ 31213, 14211 ])

if __name__ == '__main__':
  unittest.main()
