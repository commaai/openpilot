from __future__ import annotations
import functools, json, pathlib, queue, re, socket, socketserver, threading, time, traceback, typing, uuid
from tinygrad.helpers import DEBUG, colored, stderr_log
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.llm.model import Transformer

DSML = "\uff5cDSML\uff5c"  # the DeepSeek tool call tags, between fullwidth bars

def parse_tool_call(s:str) -> tuple[str, typing.Any]|None:
  s = s.strip()
  # Kimi-K3 format: tool="name" index="1"<|sep|><|open|>argument key="key" type="string"<|sep|>value<|close|>argument<|sep|>...
  if (m := re.fullmatch(r'tool="([^"]+)"(?: index="\d+")?<\|sep\|>(.*)', s, re.DOTALL)):
    pattern = r'<\|open\|>argument key="([^"]+)" type="([^"]+)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>'
    if re.sub(pattern, '', m[2], flags=re.DOTALL).strip(): return None
    def unescape(v:str) -> str: return v.replace('&quot;', '"').replace('&amp;', '&')
    try: return unescape(m[1]), {unescape(a[1]): a[3] if a[2] == "string" else json.loads(a[3]) for a in re.finditer(pattern, m[2], re.DOTALL)}
    except json.JSONDecodeError: return None
  # DeepSeek DSML format: name="name">\n<{DSML}parameter name="key" string="true|false">value</{DSML}parameter>..., V4.1 puts a space after DSML
  if (m := re.fullmatch(r'name="([^"]+)">(.*)', s, re.DOTALL)):
    pattern = rf'<{DSML} ?parameter name="([^"]+)" string="(true|false)">(.*?)</{DSML} ?parameter>'
    if re.sub(pattern, '', m[2], flags=re.DOTALL).strip(): return None
    try: return m[1], {a[1]: a[3] if a[2] == "true" else json.loads(a[3]) for a in re.finditer(pattern, m[2], re.DOTALL)}
    except json.JSONDecodeError: return None
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  # GLM format: name<arg_key>key</arg_key><arg_value>value</arg_value>...
  patterns = (
    (r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", r"<parameter=([^>]+)>(.*?)</parameter>"),
    (r"([\w.-]+)\s*((?:<arg_key>[^<]+</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)$", r"<arg_key>([^<]+)</arg_key>\s*<arg_value>(.*?)</arg_value>"))
  for call_pattern, arg_pattern in patterns:
    if (fm := re.match(call_pattern, s, re.DOTALL)):
      args = {}
      for pm in re.finditer(arg_pattern, fm.group(2), re.DOTALL):
        value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
        try: args[pm.group(1)] = json.loads(value)
        except json.JSONDecodeError: args[pm.group(1)] = value
      return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # string-only templates need text parts joined and explicit null content emptied
    if m.get("content", "") is None: m["content"] = ""
    elif isinstance(c := m.get("content"), list) and all(p.get("type") == "text" for p in c): m["content"] = "".join(p["text"] for p in c)

def disconnected(conn:socket.socket) -> bool:
  # non-blocking peek, b'' means the client closed (or half-closed). assumes no socket timeout is set.
  try: return conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
  except BlockingIOError: return False
  except OSError: return True

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse
  # (start think, end think, end content, tool call, special token that selects the format)
  FORMATS = (("<think>", "</think>", "<tool_call>", r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", "<tool_call>"),
             ("<|open|>think<|sep|>", "<|close|>think<|sep|><|open|>response<|sep|>", "<|close|>response<|sep|>",
              r"<\|open\|>call \s*(.*?)\s*(?:<\|close\|>call<\|sep\|>|$)", "<|close|>"),
             # the calls block is function_calls (V3.2), tool_calls (V4) or " calls" (V4.1)
             ("<think>", "</think>", f"\n\n<{DSML}", rf"<{DSML} ?invoke (.*?)\s*(?:</{DSML} ?invoke>|$)", DSML))
  def __init__(self, special_tokens:dict[str, int], reasoning:bool=False):
    self.think, self.end_think, self.end_content, self.call = next((f for f in self.FORMATS if f[-1] in special_tokens),
                                                                   self.FORMATS[0])[:-1]
    self.buf = ""
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
  def split(self, tag:str, final:bool) -> tuple[str, bool]:
    # split buf on the first full tag, holding back a partial tag at the end unless final
    if tag in self.buf:
      before, self.buf = self.buf.split(tag, 1)
      return before, True
    hold = max((i for i in range(1, min(len(self.buf), len(tag))+1) if tag.startswith(self.buf[-i:])), default=0) if not final else 0
    emit, self.buf = self.buf[:len(self.buf)-hold], self.buf[len(self.buf)-hold:]
    return emit, False
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len(self.think) and self.think.startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len(self.think):]) if self.buf.startswith(self.think) else ("content", self.buf)
    if self.mode == "reasoning":
      emit, done = self.split(self.end_think, final)
      if emit: yield "reasoning_content", emit
      if not done: return
      self.mode = "content"
    if self.mode == "tool": return
    emit, found = self.split(self.end_content, final)
    if emit: yield "content", emit
    if found: self.mode, self.buf = "tool", self.end_content + self.buf

class Request:
  def __init__(self, ids:list[int], disconnected:typing.Callable[[], bool], max_tokens:int|None=None, temperature:float=0.0, log_prefix:str=""):
    self.ids, self.disconnected, self.max_tokens, self.temperature, self.log_prefix = ids, disconnected, max_tokens, temperature, log_prefix
    self.finish_reason = "stop"
    self.cached_tokens = 0
    self.error: Exception|None = None
    self.out: queue.Queue[int|None] = queue.Queue()

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, req:Request, model_name:str, include_usage:bool=False, reasoning:bool=False, continuous_usage:bool=False):
    tok = self.server.tok
    prompt_tokens = len(req.ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    out: list[int] = []
    def usage():
      return {"prompt_tokens": prompt_tokens, "completion_tokens": len(out), "total_tokens": prompt_tokens + len(out),
              "prompt_tokens_details": {"cached_tokens": req.cached_tokens}}
    def chunk(d:dict, finish_reason:str|None=None):
      ret = {"choices": [{"index":0, "delta":d, "finish_reason":finish_reason}], **tmpl}
      if continuous_usage: ret["usage"] = usage()
      return ret
    dec = tok.stream_decoder()
    router = StreamRouter(tok._special_tokens, reasoning)
    self.server.waiting.put(req)
    yield chunk({"role":"assistant", "content":""})
    while (next_id := req.out.get()) is not None:
      out.append(next_id)
      for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
    if req.error is not None:
      yield {"error":{"message":f"{type(req.error).__name__}: {req.error}", "type":"server_error", "code":"generation_failed"}}
      return
    finish_reason = req.finish_reason
    for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
    tool_calls: list[dict] = []
    for m in re.finditer(router.call, router.buf, re.DOTALL):
      if (parsed := parse_tool_call(m.group(1))) is None:
        stderr_log(f"{colored('failed to parse tool call', 'red')}: {m.group(1)[:200]}\n")
        yield chunk({"content":m.group(0)})  # don't silently drop output the client can't use
      else:
        name, args = parsed
        tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                           "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
    if tool_calls:
      yield chunk({"tool_calls":tool_calls})
      if finish_reason == "stop": finish_reason = "tool_calls"
    yield chunk({}, finish_reason)
    if include_usage: yield {"choices": [], "usage": usage(), **tmpl}
  def do_POST(self):
    request_st = time.perf_counter()
    log_prefix = f"{self.path}  {colored('--', 'BLACK')}  "
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      normalize_messages(body["messages"])
      rendered = self.server.template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, preserve_thinking=True)
      ids: list[int] = self.server.tok.encode(rendered)
      log_prefix += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
      if len(ids) >= self.server.model.max_context:
        stderr_log(log_prefix + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      req = Request(ids, functools.partial(disconnected, self.connection), max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                    log_prefix=log_prefix)
      stream_options = body.get("stream_options", {})
      include_usage = not body.get("stream") or stream_options.get("include_usage", False)
      continuous_usage = include_usage and stream_options.get("continuous_usage_stats", False)
      chunks = self.run_model(req, body["model"], include_usage, continuous_usage=continuous_usage,
                              reasoning=rendered.rstrip().endswith(tuple(f[0] for f in StreamRouter.FORMATS)))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, reasoning, tool_calls, finish_reason = [], [], [], "stop"
        for c in chunks:
          if "error" in c: return self.send_data(json.dumps(c).encode(), status_code=500)
          if not c["choices"]: continue
          choice = c["choices"][0]
          if (delta := choice.get("delta", {})):
            if delta.get("content"): out.append(delta["content"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            tool_calls += [{k:v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", [])]
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out) or None}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        try:
          self.send_data(json.dumps({**c, "object":"chat.completion",
            "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
        except (BrokenPipeError, ConnectionResetError): pass
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  daemon_threads = True
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    self.waiting: queue.Queue[Request|None] = queue.Queue()
    super().__init__(server_address, Handler)

  def serve_forever(self, poll_interval:float=0.5):
    # inference runs on the caller's thread: BEAM with PARALLEL=0 uses SIGALRM, which only works on the main thread
    threading.Thread(target=super().serve_forever, args=(poll_interval,), daemon=True, name="HTTPServer").start()
    while (req := self.waiting.get()) is not None:
      try:
        if req.disconnected(): stderr_log(req.log_prefix + f"{colored('cancelled while queued', 'red')}\n")
        else:
          for next_id in self.generate(req): req.out.put(next_id)  # TODO: batch decodes across requests (batched_step)
      except Exception as e:
        traceback.print_exc()
        req.error = e
      req.out.put(None)

  def shutdown(self):
    # a request queued after the sentinel is never picked up, so its handler thread waits forever. fine since handlers are daemon threads
    super().shutdown()
    self.waiting.put(None)

  def generate(self, req:Request):
    n, finished = 0, False
    req.cached_tokens = cache_start_pos = self.model.get_start_pos(req.ids)
    stderr_log(req.log_prefix + f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(req.ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    prompt_tokens = len(req.ids)
    st = pt = time.perf_counter()
    try:
      for next_id in self.model.generate(req.ids, temperature=req.temperature):
        if n == 0: stderr_log(f"prefill:{(prompt_tokens-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if self.tok.is_end(next_id): break
        if req.disconnected(): return
        yield next_id
        n += 1
        if req.max_tokens is not None and n >= req.max_tokens:
          req.finish_reason = "length"
          break
      finished = True
    finally:
      et = time.perf_counter()
      total = f"total:{et-st:6.2f}s"
      stderr_log(f"gen:{n/(et-pt) if n > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
                 f"out:{n:5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if not finished else total}\n")
