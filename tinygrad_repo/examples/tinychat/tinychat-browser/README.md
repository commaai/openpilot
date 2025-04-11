# How to build and run tinychat in browser (WebGPU and WASM)
- `PYTHONPATH=. python examples/tinychat/tinychat-browser/compile.py`
- `./examples/tinychat/tinychat-browser/compile_wasm.sh`
    - Prerequisite: [install emscripten](https://emscripten.org/docs/getting_started/downloads.html). This script looks for `~/emsdk/emsdk_env.sh`, adjust this based on your installation.
- `./examples/tinychat/tinychat-browser/make_tiktoken_js.sh`
    - Prerequisite: install `npm`, `webpack`.
- `cd examples/tinychat && python -m http.server 7776`
- In browser: open either `localhost:7776/tinychat-browser` (WebGPU), or `localhost:7776/tinychat-browser/?backend=wasm` (WASM)