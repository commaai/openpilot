from typing import Tuple, Dict, List, Optional
from tinygrad.dtype import DType
from tinygrad.renderer import ProgramSpec
from tinygrad.tensor import Device, Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_state_dict
from tinygrad.helpers import Context, to_mv
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops
import json
from collections import OrderedDict

EXPORT_SUPPORTED_DEVICE = ["WEBGPU", "CPU", "CUDA", "GPU"]

def compile_net(run:TinyJit, special_names:Dict[int,str]) -> Tuple[Dict[str,str],List[Tuple[str,List[str],List[int]]],Dict[str,Tuple[int,DType,int]],Dict[str,Tensor]]:
  functions, bufs, bufs_to_save, statements, bufnum = {}, {}, {}, [], 0
  for ji in run.jit_cache:
    fxn: ProgramSpec = ji.prg.p
    functions[fxn.function_name] = fxn.src   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(ji.bufs):
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], arg.size*arg.dtype.itemsize, arg.dtype, key)
        else:
          bufs[key] = (f"buf_{bufnum}", arg.size*arg.dtype.itemsize, arg.dtype, key)
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an output, and it's not a special name
      cargs.append(bufs[key][0])
    cargs += [var for var in fxn.vars if getattr(var, "op", None) is Ops.DEFINE_VAR] # symbolic vars; is it necessary or sufficient to check for DEFINE_VAR?
    statements.append((fxn.function_name, cargs, fxn.global_size, fxn.local_size))

  return functions, statements, {name:(size, dtype, key) for (name,size,dtype,key) in bufs.values()}, bufs_to_save

def jit_model(model, *args) -> Tuple[TinyJit,Dict[int,str]]:
  assert hasattr(model, "forward") or callable(model), "model needs a forward function"
  @TinyJit
  def run(*x):
    out = model.forward(*x) if hasattr(model, "forward") else model(*x)
    assert isinstance(out, (tuple, list, Tensor)), "model output must be a Tensor, tuple, or a list of Tensors for export"
    out = [out] if isinstance(out, Tensor) else out
    return [o.realize() for o in out]

  # twice to run the JIT
  for _ in range(2): the_output = run(*args)
  special_names = {}

  # hack to put the inputs back
  for (j,i),idx in run.input_replace.items():
    realized_input = args[idx].uop.base.realized
    run.jit_cache[j].bufs[i] = realized_input
    special_names[id(realized_input)] = f'input{idx}'

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  for i, output in enumerate(the_output):
    special_names[id(output.uop.base.realized)] = f'output{i}'
  return run, special_names

def export_model_clang(functions:Dict[str,str], statements:Dict[str,Tuple[str,int,int]], bufs:Dict[str,Tuple[str,int,int]],
  bufs_to_save:Dict[str,Tensor], input_names:List[str], output_names:List[str], weight_names={}, model_name="model", symbolic_vars={}, wasm=False) -> str:
  headers = ["#include <tgmath.h>"]
  cprog = list(functions.values())
  dtype_map = {dtypes.int: "int", dtypes.float: "float", dtypes.uchar: "unsigned char", dtypes.char: "signed char", dtypes.half: "__fp16", dtypes.uint: "unsigned int"}
  inputs = [(name, dtype_map[bufs[name][1]], bufs[name][0]) for name in input_names + list(symbolic_vars.values())]
  outputs = [(name, dtype_map[bufs[name][1]], bufs[name][0]) for name in output_names]
  forward_args = ",".join(f"{dtype}{'*' if name not in symbolic_vars.values() else ''} {name}" for name,dtype,_ in (outputs+inputs if wasm else inputs+outputs))

  if not wasm:
    for name,cl in bufs_to_save.items():
      weight = ''.join(["\\x%02X"%x for x in bytes(to_mv(cl._buf.va_addr, cl._buf.size))])
      cprog.append(f"unsigned char {name}_data[] = \"{weight}\";")
    cprog += [f"{dtype_map[dtype]} {name}[{len}];" if name not in bufs_to_save else f"{dtype_map[dtype]} *{name} = ({dtype_map[dtype]} *){name}_data;" for name,(len,dtype,_key) in bufs.items() if name not in input_names+output_names]
    cprog += [f"void net({forward_args}) {{"] + [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
    return '\n'.join(headers + cprog)
  else:
    if bufs_to_save:
      headers += ["#include <stddef.h>"]
      bufs_to_save = {k:v for k,v in bufs.items() if v[2] in weight_names} # causes random seeds to be set as zeroes, not exported as a model weight
      buf_to_name = OrderedDict((buf_name, {"name": weight_names[data[2]], "idx": i}) for i, (buf_name, data) in enumerate(bufs_to_save.items()))
      cprog.append(f"void* bufs[{len(buf_to_name)}];")
      cprog.append(f"""void set_buf(size_t index, void* ptr) {{\n  bufs[index] = ptr;\n}}""")

    for name in set(bufs.keys()) - set(bufs_to_save.keys()) - set(input_names + output_names):
      n_bytes, dtype, _ = bufs[name]
      cprog += [f"{dtype_map[dtype]} {name}[{n_bytes // dtype.itemsize}];"]

    cprog += [f"void net({forward_args})"] + ["{"]
    get_weight_ptr = lambda x: f"({dtype_map[bufs_to_save[x][1]]} *)bufs[{buf_to_name[x]['idx']}]" if x in bufs_to_save else x
    cprog += [f"  {name}({', '.join(map(get_weight_ptr, args))});" for (name, args, _global_size, _local_size) in statements] + ["}"]
    weightMapping = "" if not bufs_to_save else f"""\nconst weightNames = [{", ".join([f'"{weight_name}"' for weight_name in [v["name"] for v in buf_to_name.values()]])}];
const {model_name}_name_to_id = Object.fromEntries(weightNames.map((name, index) => [name, index]));\n"""
    top = f"""import {model_name}Module from './{model_name}.js'{weightMapping}"""
    whitespace = "\n  "
    js_wrapper = f"""{top}\nvar {model_name} = async function() {{
  const wasm = await {model_name}Module();
  {whitespace.join(f"const {name}Ptr = wasm._malloc({n_bytes});" for name, _, n_bytes in outputs+inputs if name not in symbolic_vars.values())}
  return {{
    run: ({",".join(name for name,_,_ in inputs)}) => {{
      {(whitespace + "    ").join(f"wasm.HEAPU8.set({name}, {name}Ptr);" for name,_,_ in inputs if name not in symbolic_vars.values())}
      wasm._net({", ".join(f"{name}{'Ptr' if name not in symbolic_vars.values() else ''}" for name,_,_ in outputs+inputs)});
      {(whitespace + "    ").join(f"const {name} = wasm.HEAPU8.slice({name}Ptr, {name}Ptr + {n_bytes});" for name,_,n_bytes in outputs)}
      return [{", ".join(f"{name}" for name,_,_ in outputs)}];
    }},
    wasm: wasm
  }}
}}\nexport {{ {model_name}, {model_name}_name_to_id }};"""

    return '\n'.join(headers + cprog), js_wrapper

def dtype_to_js_type(dtype: DType) -> str:
  return f"{'Uint' if dtype in dtypes.uints else 'Int' if (dtype in dtypes.sints or dtype == dtypes.bool) else 'Float'}{8*dtype.itemsize}Array"

def export_model_webgpu(functions, statements, bufs, weight_names, input_names, output_names, model_name, symbolic_vars={}, stream_weights=False) -> Tuple[str,int,int]:
  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  kernel_names = ', '.join([name for (name, _, _, _) in statements])
  input_names += list(symbolic_vars.values())
  input_buffer_types = [dtype_to_js_type(bufs[inp_name][1]) for inp_name in input_names]
  output_buffer_types = [dtype_to_js_type(bufs[out_name][1]) for out_name in output_names]

  buf_type = lambda x: "uniform" if x in set(symbolic_vars.values()) else "storage"
  create_bind_group_layouts = ",".join([
    "device.createBindGroupLayout({{entries: [{{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {{ type: 'uniform' }}}}, {}]}})".format(
        ",".join([f"{{binding: {argIdx+1}, visibility: GPUShaderStage.COMPUTE, buffer: {{ type: '{buf_type(argName)}' }} }}" for argIdx, argName in enumerate(args)])
    )
    for _, (_, args, _, _) in enumerate(statements)
  ])
  layouts = f"const layouts=[{create_bind_group_layouts}]"
  kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, pipelines[{i}], layouts[{i}], infinityBuf, [{', '.join(args)}], [{', '.join(str(x) for x in global_size)}]);" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])

  buf_type = lambda x: "createUniformBuf" if x in set(uop.arg[0] for uop in symbolic_vars) else "createEmptyBuf"
  map_to_external_weight = lambda _key: f"state_dict['{weight_names[_key]}']" if stream_weights else f"getTensorBuffer(safetensor, metadata['{weight_names[_key]}'])"
  _bufs =  '\n    '.join([f"const {name} = " + (f"{buf_type(_key)}(device, {size});" if _key not in weight_names else f"createWeightBuf(device, {size}, {map_to_external_weight(_key)})") + ";"  for name,(size,dtype,_key) in bufs.items()])
  gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:{input_name}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,input_name in enumerate(input_names)])
  input_writers = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n        new {input_buffer_types[i]}(gpuWriteBuffer{i}.getMappedRange()).set(" + f'_{inp_name});' + f"\n        gpuWriteBuffer{i}.unmap();\n        commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {inp_name}, 0, gpuWriteBuffer{i}.size);"  for i,inp_name in enumerate(input_names)])
  gpu_read_bufs = '\n    '.join([f"const gpuReadBuffer{i} = device.createBuffer({{size:{output_name}.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});" for i,output_name in enumerate(output_names)])
  outbuf_copies = '\n        '.join([f"commandEncoder.copyBufferToBuffer({output_name}, 0, gpuReadBuffer{i}, 0, output{i}.size);" for i,output_name in enumerate(output_names)])
  output_readers = '\n        '.join([f"await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);\n        const resultBuffer{i} = new {output_buffer_types[i]}(gpuReadBuffer{i}.size/{bufs[output_names[i]][1].itemsize});\n        resultBuffer{i}.set(new {output_buffer_types[i]}(gpuReadBuffer{i}.getMappedRange()));\n        gpuReadBuffer{i}.unmap();" for i in range(len(output_names))])
  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(output_names))]))
  getTensorMetadata = f"""\nconst getTensorMetadata = (safetensorBuffer) => {{
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
}};\n""" if not stream_weights else ""
  return f"""
const {model_name} = (() => {{
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {{
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
}};
{getTensorMetadata}
const createEmptyBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
}};

const createUniformBuf = (device, size) => {{
  return device.createBuffer({{size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}})
}}

const createInfinityUniformBuf = (device) => {{
  const size = 4;
  const buf = device.createBuffer({{
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  }});
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
}};

const createWeightBuf = (device, size, data) => {{
  const buf = device.createBuffer({{ size, usage: GPUBufferUsage.STORAGE{" | GPUBufferUsage.COPY_DST" if stream_weights else ", mappedAtCreation: true"} }});
  {"data.bytes = buf;" if stream_weights else "new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();"}
  return buf;
}};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {{
  const bindGroup = device.createBindGroup({{
    layout: layout,
    entries: [
      {{ binding: 0, resource: {{ buffer: infinityUniformBuf }} }},
      ...bufs.map((buffer, index) => ({{ binding: index + 1, resource: {{ buffer }} }}))
    ]
  }});

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
}};

{kernel_code}

const setupNet = async (device, {"state_dict" if stream_weights else "safetensor"}) => {{
    {"const metadata = getTensorMetadata(safetensor);" if not stream_weights else ""}
    const infinityBuf = createInfinityUniformBuf(device);

    {layouts}

    {_bufs}

    {gpu_write_bufs}

    {gpu_read_bufs}

    const kernels = [{kernel_names}];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {{
      return await device.createComputePipelineAsync({{
          layout: device.createPipelineLayout({{
              bindGroupLayouts: [layouts[i]],
          }}),
          compute: {{
              module: device.createShaderModule({{
                  code: name,
              }}),
              entryPoint: "main",
          }},
      }});
  }}))

    return async ({",".join([f"_{input_name}" for input_name in input_names])}) => {{
        const commandEncoder = device.createCommandEncoder();
        {input_writers}
        {kernel_calls}
        {outbuf_copies}
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        {output_readers}
        return {output_return};
    }}
}}
const load = async (device, weight_path) => {{ return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }}
return {{ load, setupNet }};
}})();
export default {model_name};
"""

def export_model(model, target:str, *inputs, model_name: Optional[str] = "model", stream_weights=False):
  assert Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, f"only {', '.join(EXPORT_SUPPORTED_DEVICE)} are supported"
  with Context(JIT=2): run,special_names = jit_model(model, *inputs)
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
  state = get_state_dict(model)
  weight_names = {id(x.uop.base.realized): name for name, x in state.items()}
  input_names = [name for _,name in special_names.items() if "input" in name]
  output_names = [name for _,name in special_names.items() if "output" in name]

  # handle symbolic variables; TODO: refactor to fix some of this stuff upstream in tinygrad
  symbolic_vars = OrderedDict()
  for i, (_, args, global_size, _) in enumerate(statements):
    for j, var in enumerate(args):
      if getattr(var, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(var, "arg", None), tuple) and isinstance(var.arg[0], str):
        if var not in symbolic_vars:
          symbolic_vars[var] = var.arg[0]
          bufs[symbolic_vars[var]] = (var.dtype.itemsize, var.dtype, symbolic_vars[var])
        statements[i][1][j] = symbolic_vars[var]

    if global_size:
      for j, dim in enumerate(global_size):
        if getattr(dim, "op", None) is Ops.ADD and len(dim.src) == 2 and {dim.src[0].op, dim.src[1].op} == {Ops.DEFINE_VAR, Ops.CONST}:
          name, val = dim.src if dim.src[1].op is Ops.CONST else reversed(dim.src)
          global_size[j] = f"_{name.arg[0]}[0] + {val.arg}"

  prg = ""
  if target == "clang":
    prg = export_model_clang(functions, statements, bufs, bufs_to_save, input_names, output_names)
  elif target == "wasm":
    return export_model_clang(functions, statements, bufs, bufs_to_save, input_names, output_names, weight_names, model_name, symbolic_vars, wasm=True)
  elif target == "webgpu":
    prg = export_model_webgpu(functions, statements, bufs, weight_names, input_names, output_names, model_name, symbolic_vars, stream_weights)
  else:
    prg = json.dumps({
      "backend": Device.DEFAULT,
      "inputs": [{
        "size": bufs[name][0],
        "dtype": bufs[name][1].name
      } for name in input_names],
      "outputs": [{
        "size": bufs[name][0],
        "dtype": bufs[name][1].name
      } for name in output_names],
      "functions": functions,
      "statements": [{
        "kernel": kernel,
        "args": args,
        "global_size": global_size,
        "local_size": local_size
      } for (kernel, args, global_size, local_size) in statements],
      "buffers": {
        name: {
          "size": size,
          "dtype": dtype.name,
          "id": weight_names[_key] if _key in weight_names else ""
        } for name, (size,dtype,_key) in bufs.items() if name not in ["input", "outputs"]
      }
    })

  return prg, {input:bufs[input][0] for input in input_names}, {output:bufs[output][0] for output in output_names}, state
