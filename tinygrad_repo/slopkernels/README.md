# slopkernels

Optional AMD RDNA3/RDNA4/CDNA static templated UIR kernel generator for `tinygrad.llm` (`SLOP=0` disables custom kernels and falls back to pure `tinygrad`).

## How It Works (Dev vs Release)

Generated `.uir` files live in `tinygrad/llm/kernels/` (gitignored, never committed to git):

- **Development (`slopkernels` importable)**: `tinygrad.llm.model._load_kernel(name)` imports `from slopkernels import get` and calls `get(name)`, which **regenerates `tinygrad/llm/kernels/<name>.uir` fresh on disk every time** and parses it back through `parse_uir`.
- **Release (only `tinygrad/llm/kernels/*.uir` distributed, no `slopkernels` package)**: when `from slopkernels import get` raises `ImportError`, `tinygrad.llm.model._load_kernel(name)` loads and parses `tinygrad/llm/kernels/<name>.uir` directly if present (otherwise falls back to pure `tinygrad`).

Because release builds ship **only** the static `.uir` files—without `slopkernels`—everything required at runtime must live inside the `.uir` templates themselves or in `tinygrad/llm/model.py`.

## Rules

- **No dispatch or dynamic stuff in `slopkernels`**: `slopkernels` is strictly an offline/dev `.uir` generator. It exposes **only** `{get, generate_all}` (`tinygrad` may only import `from slopkernels import get` inside `tinygrad/llm/model.py`). Never put dispatch logic, shape routing, runtime helpers, or dynamic UIR generation in `slopkernels`.
- **No runtime UIR editing**: every kernel is a static templated `.uir` file in `tinygrad/llm/kernels/<name>.uir`. Callers may **only** call `.specialize(**template_params)` on its `addrspace=TEMPLATE` parameters—never rewrite, patch, or edit UIR at runtime.
- **Pure UOp builders only**: `slopkernels/__init__.py` and `slopkernels/src/*.py` must stay pure `UOp` builders (no `Tensor`, `Buffer`, `Device`, `getenv`, `nn`, or runtime dispatch modules).
- **Compact & gitignored UIR**: `tinygrad/llm/kernels/` is gitignored and `.uir` files are never tracked in git (enforced by `test_narrow_api_and_purity_boundary`). Total generated UIR size must stay under 1 MB.

## API

```python
import slopkernels

k = slopkernels.get("rdna4_gemm_f16")       # regenerates tinygrad/llm/kernels/<name>.uir, returns parsed templated UOp
uirs = slopkernels.generate_all()           # regenerates all static templated .uir files in tinygrad/llm/kernels/
```

## CLI & Tests

```bash
DEV=NULL python3 -m slopkernels
DEV="MOCKKFD+AMD::gfx1201" PARALLEL=0 python3 -m pytest slopkernels/test/ -n12
DEV=AMD python3 -m slopkernels.bench --fast --warmup 2 --iters 5
```
