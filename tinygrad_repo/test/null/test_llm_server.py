import unittest, threading, time, json
from unittest.mock import Mock, patch

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.preset = "llama3"
    cls.mock_tok.bos_id = 1
    cls.mock_tok.eos_id = 999
    cls.mock_tok.eot_id = None
    cls.mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    cls.mock_model = Mock()
    cls.mock_model.max_context = 4
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_model.get_start_pos = Mock(return_value=0)

    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer

    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "test-model", cls.mock_tok, FallbackTemplate(cls.mock_tok))
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
    def generate(ids, **kwargs):
      for token in (300, 301, 999):
        ids.append(token)
        yield token
    with patch.object(self.mock_model, "generate", side_effect=generate):
      chunks = list(self.client.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, stream_options={"include_usage": True}))
    last_chunk = chunks[-1]

    self.assertEqual(last_chunk.usage.prompt_tokens, 3)
    self.assertEqual(last_chunk.usage.completion_tokens, 2)
    self.assertEqual(last_chunk.usage.total_tokens, 5)

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

  def test_context_length_error(self):
    from openai import BadRequestError
    self.mock_tok.encode.return_value = [200, 201, 202, 203]
    try:
      with self.assertRaises(BadRequestError) as err:
        self.client.chat.completions.create(model="test-model", messages=[{"role":"user", "content":"too long"}])
      self.assertEqual(err.exception.code, "context_length_exceeded")
    finally:
      self.mock_tok.encode.return_value = [200, 201, 202]

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

  def test_models_endpoint(self):
    import requests as req
    resp = req.get(f"http://127.0.0.1:{self.port}/v1/models")
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(data["object"], "list")
    self.assertEqual(len(data["data"]), 1)
    self.assertEqual(data["data"][0]["id"], "test-model")
    self.assertEqual(data["data"][0]["object"], "model")

class TestLLMToolCalls(unittest.TestCase):
  """Tool calling through the OpenAI-compatible HTTP API."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="")
    cls.mock_tok.preset = "qwen2"
    cls.mock_tok.bos_id, cls.mock_tok.eos_id, cls.mock_tok.eot_id = None, 999, None
    cls.mock_tok.is_end = Mock(return_value=False)

    cls.mock_model = Mock()
    cls.mock_model.max_context = 4
    cls.mock_model.get_start_pos = Mock(return_value=0)

    from tinygrad.llm.serve import LLMServer
    import jinja2
    # .items() matches tool-aware templates and ensures OpenAI JSON argument strings are normalized before rendering the next turn.
    template = jinja2.Template("""{% for m in messages %}{{ m.content or '' }}{% for tc in m.tool_calls or [] %}
      {% for key, value in tc.function.arguments.items() %}{{ key }}={{ value }}{% endfor %}{% endfor %}{% endfor %}""")
    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "tool-model", cls.mock_tok, template)
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

  def set_output(self, text:str):
    pieces = dict(enumerate(text, 1))
    self.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: pieces[tid] if tid is not None else "")
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter(pieces))

  @staticmethod
  def tools():
    return [{"type":"function", "function":{"name":"read", "description":"Read a file",
      "parameters":{"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"]}}}]

  def test_streaming_tool_call(self):
    self.set_output('before<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>')
    chunks = list(self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read README.md"}],
                                                     tools=self.tools(), stream=True))
    self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "before")
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0].function.name, "read")
    self.assertEqual(json.loads(calls[0].function.arguments), {"path":"README.md"})
    self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

  def test_multiple_xml_tool_calls(self):
    self.set_output("<tool_call><function=read><parameter=path>\"a\"</parameter></function></tool_call>"
                    "<tool_call><function=read><parameter=path>\"b\"</parameter></function></tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read a and b"}],
                                                   tools=self.tools())
    self.assertEqual([json.loads(tc.function.arguments)["path"] for tc in response.choices[0].message.tool_calls], ["a", "b"])
    self.assertEqual(response.choices[0].finish_reason, "tool_calls")

  def test_multiline_tool_argument_preserves_trailing_newline(self):
    self.set_output("<tool_call>\n<function=write>\n<parameter=content>\nfirst\nsecond\n\n</parameter>\n"
                    "<parameter=filePath>\nout.txt\n</parameter>\n</function>\n</tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Write out.txt"}], tools=self.tools())
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    self.assertEqual(args, {"content":"first\nsecond\n", "filePath":"out.txt"})

  def test_invalid_tool_call_becomes_content(self):
    self.set_output("<tool_call>not a call</tool_call>")
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Hello"}], tools=self.tools())
    self.assertEqual(response.choices[0].message.content, "<tool_call>not a call</tool_call>")
    self.assertIsNone(response.choices[0].message.tool_calls)
    self.assertEqual(response.choices[0].finish_reason, "stop")

  def test_tool_call_in_reasoning_is_not_executed(self):
    self.set_output('<think>draft <tool_call>{"name":"wrong","arguments":{}}</tool_call></think>answer')
    response = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Hello"}], tools=self.tools())
    self.assertEqual(response.choices[0].message.content, "answer")
    self.assertIsNone(response.choices[0].message.tool_calls)
    self.assertEqual(response.choices[0].finish_reason, "stop")

  def test_tool_result_round_trip(self):
    self.set_output('<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>')
    first = self.client.chat.completions.create(model="tool-model", messages=[{"role":"user", "content":"Read README.md"}], tools=self.tools())
    call = first.choices[0].message.tool_calls[0]
    self.set_output("done")
    second = self.client.chat.completions.create(model="tool-model", messages=[
      {"role":"user", "content":"Read README.md"},
      {"role":"assistant", "content":None, "tool_calls":[call.model_dump()]},
      {"role":"tool", "tool_call_id":call.id, "content":"file contents"},
    ], tools=self.tools())
    self.assertEqual(second.choices[0].message.content, "done")
    self.assertEqual(second.choices[0].finish_reason, "stop")

if __name__ == '__main__':
  unittest.main()
