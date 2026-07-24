# SQTT Profiling

## Getting SQ Thread Trace

`VIZ=2` to enable SQTT profiling.

`SQTT_ITRACE_SE_MASK=X` to select shader engines for instruction tracing, -1 = all, 0 = disabled, >0 = SE bitmask, default 0b11.

`SQTT_BUFFER_SIZE=X` to change size of SQTT buffer (per shader engine, 6 SEs on 7900xtx) in megabytes, default 256.

## Viewing the traces

- Web UI: `tinygrad/viz/serve.py`
- Command line: `python -m tinygrad.renderer.amd.sqtt`
