import unittest, base64, functools, sys
from tinygrad.apps.llm import SimpleTokenizer
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

if __name__ == '__main__':
  unittest.main()
