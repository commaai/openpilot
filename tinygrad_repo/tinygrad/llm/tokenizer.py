import sys, codecs, itertools, typing, re, string, unicodedata, json, pathlib
from tinygrad.helpers import partition

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, add_bos:bool=True, eos_id:int=0, eot_id:int|None=None, eom_id:int|None=None, chat_template:str|None=None):
    marks = preset in ("qwen35", "qwen35moe")
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2","minimax-m2":"gpt-4o","inkling2":"gpt-4o","deepseek-v3":"joyai-llm"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4","gpt-4o","joyai-llm","bailingmoe2"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # each limit is one past the category's max codepoint (Z→U+3000, N→U+1FBF9, L→U+323AF, M→U+E01EF, P→U+1E95F, S→U+1FBCA)
    # compact adjacent codepoints into ranges: listing them all makes re spend seconds on large prompts
    def ucat_range(pre:str|tuple[str, ...]) -> str:
      limits = {"Z": 0x3001, "N": 0x1fbfa, "L": 0x323b0, "M": 0xe01f0, "P": 0x1e960, "S": 0x1fbcb}
      limit = max(limits[p if p in limits else p[0]] for p in (pre if isinstance(pre, tuple) else (pre,)))
      cps = enumerate(cp for cp in range(limit) if unicodedata.category(chr(cp)).startswith(pre))
      runs = [list(g) for _, g in itertools.groupby(cps, lambda e: e[1]-e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range(("L", "M") if marks else "L")
    contr, r_l, r_n = "(?i:'s|'t|'re|'ve|'m|'ll|'d)", f"[^\\r\\n{r_p_N}{r_p_L}]?", f"[{r_p_N}]" if preset == "tekken" else f"[{r_p_N}]{{1,3}}"
    r_p, r_w, r_t = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*", f"{contr}|{r_l}[{r_p_L}]+", f"[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    if preset == "olmo":
      r_w, r_n = f"'s|'t|'re|'ve|'m|'ll|'d| ?[{r_p_L}]+", f" ?[{r_p_N}]+"
      r_p, r_t = f" ?[^{r_ws}{r_p_N}{r_p_L}]+", f"[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    if preset in ("tekken", "gpt-4o", "kimi-k2"):
      r_up, r_lo = ucat_range(("Lu","Lt","Lm","Lo","M")), ucat_range(("Ll","Lm","Lo","M"))
      sfx = "" if preset == "tekken" else f"{contr}?"
      r_w = f"{r_l}[{r_up}]*[{r_lo}]+{sfx}|{r_l}[{r_up}]+[{r_lo}]*{sfx}"
      if preset != "kimi-k2": r_p = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n/]*"
    self._pre_split = None
    if preset == "joyai-llm":
      r_ps = ucat_range(("P", "S"))
      r_p, r_w = f" ?[{r_ps}]+[\\r\\n]*", f"[{re.escape(string.punctuation)}][A-Za-z]+|[^\\r\\n{r_p_L}{r_ps}]?[{r_p_L}{ucat_range('M')}]+"
      # numbers and CJK runs are split off before the words
      self._pre_split = re.compile(f"({r_n}|[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+)")
    self._split_to_word = re.compile(f"({r_w}|{r_n}|{r_p}|{r_t})")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset, self.chat_template = preset, chat_template
    self.bos_id, self.add_bos, self.eos_id, self.eot_id, self.eom_id = bos_id, add_bos, eos_id, eot_id, eom_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] not in (2, 3, 4)) # UNKNOWN|CONTROL|USER_DEFINED
    special_tokens_dict = dict(special_tokens)
    return SimpleTokenizer(dict(normal_tokens), special_tokens_dict, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id'), add_bos=kv.get('tokenizer.ggml.add_bos_token', True),
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_tokens_dict.get('<|im_end|>')),
      eom_id=kv.get('tokenizer.ggml.eom_token_id'), chat_template=kv.get('tokenizer.chat_template'))

  @staticmethod
  def from_hf(path:str|pathlib.Path, preset:str):
    path = pathlib.Path(path)
    cfg, tok = (json.loads((path / name).read_text()) for name in ("config.json", "tokenizer.json"))
    tok_cfg = json.loads((path / "tokenizer_config.json").read_text()) if (path / "tokenizer_config.json").exists() else {}
    added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    vocab = tok["model"]["vocab"]
    eos = cfg.get("eos_token_id", 0)
    eos = eos if isinstance(eos, list) else [eos]
    return SimpleTokenizer({t: i for t, i in vocab.items() if t not in added}, added, preset,
      bos_id=vocab.get(tok_cfg.get("bos_token")), eos_id=eos[0],
      eot_id=eos[1] if len(eos) > 1 else None, eom_id=eos[2] if len(eos) > 2 else None,
      chat_template=template.read_text() if (template:=path / "chat_template.jinja").exists() else tok_cfg.get("chat_template"))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    if self._pre_split is None: words = self._split_to_word.findall(chunk)
    else: words = [w for p in self._pre_split.split(chunk) for w in self._split_to_word.split(p) if w]
    return [tok for word in words for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id, self.eom_id)
