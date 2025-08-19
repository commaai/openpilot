# tinygrad directory layout

This explains the flow of a big graph down to programs.

Directories are listed in order of how they are processed.

---

## tinygrad/schedule

Group UOps into kernels.

::: tinygrad.schedule.kernelize.get_kernelize_map
    options:
        members: false
        show_labels: false
        show_source: false

---

## tinygrad/codegen/opt

Transforms the ast into an optimized ast. This is where BEAM search and heuristics live.

::: tinygrad.codegen.opt.get_optimized_ast
    options:
        members: false
        show_labels: false
        show_source: false

---

## tinygrad/codegen

Transform the optimized ast into a linearized list of UOps.

::: tinygrad.codegen.full_rewrite
    options:
        members: false
        show_labels: false
        show_source: false

---

## tinygrad/renderer

Transform the linearized list of UOps into a program, represented as a string.

::: tinygrad.renderer.Renderer
    options:
        members:
            - render
        show_labels: false
        show_source: false

---

## tinygrad/engine

Abstracted high level interface to the runtimes.

::: tinygrad.engine.realize.get_program
    options:
        members: false
        show_labels: false
        show_source: false
