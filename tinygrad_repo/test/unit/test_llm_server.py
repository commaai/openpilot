import unittest, threading, time
from unittest.mock import Mock, patch
from tinygrad import Tensor

class TestTransformerGenerate(unittest.TestCase):
  def test_start_pos_parameter_is_used(self):
    """Test that start_pos parameter is not ignored (regression test for always resetting to 0)."""
    from tinygrad.apps.llm import Transformer
    # Create a minimal transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.bind_val))
      return Tensor([[42]])  # return a fake next token

    with patch.object(Transformer, '__call__', mock_call):
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens, start_pos=3)
      next(gen)  # get first token

    # With start_pos=3, the initial tensor should only have tokens[3:] = [4, 5] (length 2)
    # If the bug existed (start_pos always reset to 0), it would have all 5 tokens
    self.assertEqual(captured_inputs[0][0][-1], 2)  # shape should be (1, 2)
    self.assertEqual(captured_inputs[0][1], 3)  # start_pos should be 3, not 0

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.role = Mock(return_value=[100, 101])
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.end_turn = Mock(return_value=[998])

    cls.mock_model = Mock()
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))

    cls.bos_id = 1
    cls.eos_id = 999

    import tinygrad.apps.llm as llm_module
    llm_module.model = cls.mock_model
    llm_module.tok = cls.mock_tok
    llm_module.bos_id = cls.bos_id
    llm_module.eos_id = cls.eos_id

    from tinygrad.apps.llm import Handler
    from tinygrad.viz.serve import TCPServerWithReuse

    cls.server = TCPServerWithReuse(('127.0.0.1', 0), Handler)
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)

    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def test_chat_completion_stream(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_openai_response_structure(self):
    stream = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Test"}],
      stream=True
    )

    for chunk in stream:
      self.assertTrue(chunk.id.startswith("chatcmpl-"))
      self.assertEqual(chunk.object, "chat.completion.chunk")
      self.assertIsNotNone(chunk.choices)
      self.assertIsNotNone(chunk.created)
      self.assertIsInstance(chunk.created, int)
      self.assertEqual(chunk.model, "test-model")

  def test_stream_with_usage(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True,
      stream_options={"include_usage": True}
    )

    chunks = list(stream)
    last_chunk = chunks[-1]

    self.assertIsNotNone(last_chunk.usage)
    self.assertIsNotNone(last_chunk.usage.prompt_tokens)
    self.assertIsNotNone(last_chunk.usage.completion_tokens)
    self.assertIsNotNone(last_chunk.usage.total_tokens)

  def test_multi_turn_conversation(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
      ],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_content_is_streamed(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    contents = []
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        contents.append(chunk.choices[0].delta.content)

    self.assertGreater(len(contents), 0)

  def test_non_streaming(self):
    resp = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Hello"}],
      stream=False
    )

    self.assertTrue(resp.id.startswith("chatcmpl-"))
    self.assertEqual(resp.object, "chat.completion")
    self.assertEqual(resp.model, "test-model")
    self.assertIsNotNone(resp.created)
    self.assertEqual(len(resp.choices), 1)
    self.assertEqual(resp.choices[0].message.role, "assistant")
    self.assertIsNotNone(resp.choices[0].message.content)
    self.assertEqual(resp.choices[0].finish_reason, "stop")
    self.assertIsNotNone(resp.usage)
    self.assertIsNotNone(resp.usage.prompt_tokens)
    self.assertIsNotNone(resp.usage.completion_tokens)

if __name__ == '__main__':
  unittest.main()
