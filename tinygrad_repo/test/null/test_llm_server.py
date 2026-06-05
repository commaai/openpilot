import unittest, threading, time
from unittest.mock import Mock

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.role = Mock(return_value=[100, 101])
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.end_turn = Mock(return_value=[998])
    cls.mock_tok.prefix = Mock(return_value=[1])
    cls.mock_tok.preset = "llama3"
    cls.mock_tok.bos_id = 1
    cls.mock_tok.eos_id = 999
    cls.mock_tok.eot_id = None
    cls.mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    cls.mock_model = Mock()
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_model.get_start_pos = Mock(return_value=0)

    from tinygrad.llm.cli import LLMServer

    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "test-model", cls.mock_tok)
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

  def test_max_tokens_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, max_tokens=2
    )
    chunks = list(stream)
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    self.assertEqual(len(content_chunks), 2)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

  def test_max_tokens_non_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=False, max_tokens=2
    )
    self.assertEqual(resp.choices[0].finish_reason, "length")
    self.assertEqual(resp.usage.completion_tokens, 2)

  def test_assistant_prefill(self):
    """Last assistant message should be treated as prefill (not a completed turn)."""
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 999]))
    captured_ids = []
    orig_generate = self.mock_model.generate.side_effect
    def capture_generate(ids, **kwargs):
      captured_ids.extend(ids)
      return orig_generate(ids, **kwargs)
    self.mock_model.generate = Mock(side_effect=capture_generate)

    resp = self.client.chat.completions.create(
      model="test", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Sure"}
      ], stream=False
    )
    # prefill tokens should be in ids: role("assistant") + encode("Sure") but NO end_turn after it
    # and NO extra role("assistant") appended
    role_tokens = self.mock_tok.role.call_args_list
    # last role() call should be for "assistant" (the prefill message), not an extra one
    self.assertEqual(role_tokens[-1], unittest.mock.call("assistant"))
    # end_turn should be called once less than role() — the prefill assistant msg doesn't get end_turn
    self.assertEqual(self.mock_tok.end_turn.call_count, self.mock_tok.role.call_count - 1)
    self.assertIsNotNone(resp.choices[0].message.content)

  def test_assistant_prefill_not_last(self):
    """Assistant message that's NOT last should be a normal completed turn."""
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 999]))
    self.mock_tok.role.reset_mock()
    self.mock_tok.end_turn.reset_mock()
    self.client.chat.completions.create(
      model="test", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Sure"},
        {"role": "user", "content": "Continue"}
      ], stream=False
    )
    # all messages get end_turn, plus an extra role("assistant") at the end
    # roles: user, assistant, user, assistant(generation prompt) = 4 role calls
    # end_turns: user, assistant, user = 3 end_turn calls (one per message)
    self.assertEqual(self.mock_tok.end_turn.call_count, 3)
    self.assertEqual(self.mock_tok.role.call_count, 4)

  def test_models_endpoint(self):
    import requests as req
    resp = req.get(f"http://127.0.0.1:{self.port}/v1/models")
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(data["object"], "list")
    self.assertEqual(len(data["data"]), 1)
    self.assertEqual(data["data"][0]["id"], "test-model")
    self.assertEqual(data["data"][0]["object"], "model")

if __name__ == '__main__':
  unittest.main()
