import os, json, hashlib, math
from extra.export_model import export_model
from examples.llama3 import build_transformer, Tokenizer
from tinygrad.nn.state import get_state_dict, load_state_dict
from tinygrad import Device, Variable, Tensor, dtypes, TinyJit
from tinygrad.helpers import fetch, Context
from tiktoken.load import load_tiktoken_bpe, dump_tiktoken_bpe

def prepare_browser_chunks(model):
  # split weights into browser-friendly chunks
  state_dict = get_state_dict(model)
  del state_dict['output.weight'], state_dict['output.scale'] # same as tok_embeddings; ensures consistency with model export
  chunk_size = 16 * 1024 * 1024 # small chunks based on iphone browser constraints
  metadata = {}
  # We won't export cache_kv bytes (because we start inference on client at start_pos=0), but we will tell the client how big cache_kv needs to be
  t_infos = [(v.uop.base.realized.nbytes, k, v.dtype) for k,v in state_dict.items() if "cache_kv" not in k]
  empty_t_infos = [(v.uop.base.realized.nbytes, k, v.dtype) for k,v in state_dict.items() if "cache_kv" in k]

  split_t_infos = []
  for size, name, dtype in t_infos:
    if size <= chunk_size:
      split_t_infos.append((size, name, dtype, ()))
    else: # split large weights into multiple parts
      for i in range(0, size, chunk_size):
        split_t_infos.append((min(chunk_size, size-i), f"{name}_part{math.ceil(i/chunk_size)}", dtype, (i, min(i+chunk_size, size))))

  files = []
  # pack weights into files with FFD bin packing
  split_t_infos = sorted(split_t_infos, reverse=True)
  for info in split_t_infos:
    placed = False
    for file in files:
      if sum(i[0] for i in file) + info[0] <= chunk_size:
        if info[3] and any(i[3] for i in file): continue # no two split tensors can touch the same file, due to wasm loading constraints
        file.append(info)
        placed = True
        break
    if not placed:
      files.append([info])

  tinygrad_dtypes = {dtypes.float32: "float32", dtypes.float16: "float16", dtypes.int8: "int8", dtypes.int32: "int32"}
  for i, file in enumerate(files):
    cursor = 0
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.chunk'), "wb+") as writer:
      for size, name, dtype, offsets in file:
        name, part_num = (name, 0) if "_part" not in name else (name.split("_part")[0], int(name.split("_part")[1]))
        default = {"parts": {}, "dtype": tinygrad_dtypes[dtype]}
        weight_metadata = metadata.get(name, default)
        weight_metadata["parts"][part_num] = {"file": i, "file_start_pos": cursor, "size": size}
        metadata[name] = weight_metadata
        data = bytes(state_dict[name].uop.base.realized.as_buffer())
        data = data if not offsets else data[offsets[0]:offsets[1]]
        writer.write(data)
        cursor += size

  metadata.update({name: {"parts": {0: {"empty": True, "size": size}}, "dtype": tinygrad_dtypes[dtype]} for size, name, dtype in empty_t_infos})

  for k in metadata:
    metadata[k]["parts"] = [part for part_num, part in sorted(metadata[k]["parts"].items(), key = lambda x: x[0])]
    cursor = 0
    for i, part in enumerate(metadata[k]["parts"]):
      metadata[k]["parts"][i]["target_start_pos"] = cursor
      cursor += part["size"]
    metadata[k]["size"] = cursor

  # compute hashes, which client app will check to determine whether to update with new weights and/or detect integrity issues
  state_dict_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"state_dict": metadata, "state_dict_hash": state_dict_hash, "files": []}
  hashes = set()
  for i in range(len(files)):
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.chunk'), "rb") as reader:
      hash = hashlib.sha256(reader.read()).hexdigest()
      hashes.add(hash)
      metadata["files"].append({"name": f'net_part{i}.chunk', "hash": hash})
  if len(hashes) != len(files): print(f"WARNING: {len(files)} files were exported, but only {len(hashes)} are unique: something may have gone wrong")
  metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"metadata": metadata, "metadata_hash": metadata_hash}

  with open(os.path.join(os.path.dirname(__file__), f'./net_metadata.json'), "w") as writer: json.dump(metadata, writer, indent=4)
  return metadata

def validate_model(model, tokenizer):
  prompt = "yo"
  toks = [tokenizer.bos_id]
  toks += [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode("user") + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  toks += tokenizer.encode(prompt) + [tokenizer.special_tokens["<|eot_id|>"]]
  toks += [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode("assistant") + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  start_pos = 0
  run = TinyJit(model.forward)
  for tok in toks[:-1]:
    run(Tensor([[tok]]), Variable("start_pos", 0, model.max_context).bind(start_pos), 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
  tok = toks[-1]
  result = ""
  expected = "How's it going?"
  while True:
    tok = run(Tensor([[tok]]), Variable("start_pos", 0, model.max_context).bind(start_pos), 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1
    if tok in tokenizer.stop_tokens or len(result) > len(expected): break
    result += tokenizer.decode([tok])
  assert result == expected, f"Model validation failed, expected output: {expected}, actual output: {result}"

if __name__=="__main__":
  # Export BPE data for use with tiktoken.js
  tokenizer_path = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
  mergeable_ranks = load_tiktoken_bpe(str(tokenizer_path))
  bpe_path = os.path.join(os.path.dirname(__file__), "llama3-2.tiktoken")
  dump_tiktoken_bpe(mergeable_ranks, bpe_path)
  tokenizer = Tokenizer(str(tokenizer_path))

  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf", "Llama-3.2-1B-Instruct-f16.gguf", subdir="llama3-1b-instruct")
  max_context=1024
  tok = 128000
  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0
  start_pos = Variable("start_pos", 0, max_context).bind(0)
  model_input = lambda: [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P]

  Device.DEFAULT="CPU"
  model = build_transformer(model_path, model_size="1B", quantize="int8", scale_dtype=dtypes.float32, device=Device.DEFAULT, max_context=max_context)
  state_dict = get_state_dict(model)
  validate_model(model, tokenizer)
  model_name = "transformer"

  with Context(BEAM=3):
    cprog, js_wrapper = export_model(model, "wasm", *model_input(), model_name=model_name)
    # ensure consistency with exported weights
    js_wrapper = js_wrapper.replace("output.weight", "tok_embeddings.weight").replace("output.scale", "tok_embeddings.scale")

  with open(os.path.join(os.path.dirname(__file__), f"{model_name}.c"), "w") as f: f.write(cprog)
  with open(os.path.join(os.path.dirname(__file__), "net_clang.js"), "w") as f: f.write(js_wrapper)

  Device.DEFAULT="WEBGPU"
  # float16 is not yet supported for dawn/Vulkan/NVIDIA stack, see: https://issues.chromium.org/issues/42251215
  # therefore for now, we used CLANG to quantize the float16 llama to int8 with float32 scales, then load to WEBGPU
  model = build_transformer(model_path, model_size="1B", quantize="int8", max_context=max_context, load_weights=False)
  load_state_dict(model, state_dict)
  # these were the same before load_state_dict
  model.output.weight, model.output.scale = model.tok_embeddings.weight, model.tok_embeddings.scale

  validate_model(model, tokenizer)
  metadata = prepare_browser_chunks(model) # export weights to disk

  with Context(BEAM=3):
    prg, input_sizes, output_sizes, state = export_model(model, "webgpu", *model_input(), model_name=model_name, stream_weights=True)
    # ensure consistency with exported weights
    prg = prg.replace("output.weight", "tok_embeddings.weight").replace("output.scale", "tok_embeddings.scale")

  with open(os.path.join(os.path.dirname(__file__), "net.js"), "w") as f: f.write(prg)