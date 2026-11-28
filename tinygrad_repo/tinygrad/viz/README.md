VIZ is a tool for inspecting tinygrad's rewrites and runtime profiling.

to use:
1. Run tinygrad with `VIZ=1` (this saves the pkls and launches the server in interactive shells)
2. That's it!

# VIZ in the command line

Use `python -m tinygrad.viz.cli` (add --json for scripting) to view the full timeline of events.

### Profiling examples

Get kernel times and ASTs

```bash
DEBUG=3 python -m tinygrad.viz.cli --json > /tmp/events.jsonl
```

Search kernels or rewrite passes

```
python -m tinygrad.viz.cli -q 'schedule|linear'
```

Select events between two markers

Markers are set using the `profile_marker` helper in user code. To list them:
```
python -m tinygrad.viz.cli | rg MARKER
```
Then:
```
python -m tinygrad.viz.cli --interval "train @ 2" "train @ 3"
```

Set `-t` to aggregate events.

### Rewrites Debugging example

Query rewrites:

```bash
python -m tinygrad.viz.cli -q 'tensor graph'
python -m tinygrad.viz.cli -q 'devectorize2'
python -m tinygrad.viz.cli -q '^E_'
```

Select specific rewrites:

```bash
python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1"
```

### Environment variables

Combine `DEBUG` with the CLI command to include additional information

| DEBUG | Includes |
|------:|----------|
| 3 | Base AST |
| 4 | Generated source |
| 5 | Rewrite steps and kernel graph |
| 6 | All UOp graphs |
| 7 | All rewrites |

Set `NO_COLOR=1` to disable colors.

# SQTT / PMC profiling (DEV=AMD only)

PMC profiling is included by default, select using:

```bash
python -m tinygrad.viz.cli -s "gemm PMC"
```

SQTT has additional overhead. Set VIZ=2 to include it in pkls.

```bash
python -m tinygrad.viz.cli -s "kernel SQTT SE:0 PKTS" --json
```
