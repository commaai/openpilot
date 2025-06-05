The tinygrad framework has four pieces

* a PyTorch like <b>frontend</b>.
* a <b>scheduler</b> which breaks the compute into kernels.
* a <b>lowering</b> engine which converts ASTs into code that can run on the accelerator.
* an <b>execution</b> engine which can run that code.

There is a good [bunch of tutorials](https://mesozoic-egg.github.io/tinygrad-notes/) by Di Zhu that go over tinygrad internals.

There's also a [doc describing speed](../developer/speed.md)

## Frontend

Everything in [Tensor](../tensor/index.md) is syntactic sugar around constructing a graph of [UOps](../developer/uop.md).

The `UOp` graph specifies the compute in terms of low level tinygrad ops. Not all UOps will actually become realized. There's two types of UOps, base and view. base contains compute into a contiguous buffer, and view is a view (specified by a ShapeTracker). Inputs to a base can be either base or view, inputs to a view can only be a single base.

## Scheduling

The [scheduler](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/engine/schedule.py) converts the graph of UOps into a list of `ScheduleItem`. One `ScheduleItem` is one kernel on the GPU, and the scheduler is responsible for breaking the large compute graph into subgraphs that can fit in a kernel. `ast` specifies what compute to run, and `bufs` specifies what buffers to run it on.

::: tinygrad.engine.schedule.ScheduleItem

## Lowering

The code in [realize](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/engine/realize.py) lowers `ScheduleItem` to `ExecItem` with

::: tinygrad.engine.realize.lower_schedule

There's a ton of complexity hidden behind this, see the `codegen/` directory.

First we lower the AST to UOps, which is a linear list of the compute to be run. This is where the BEAM search happens.

Then we render the UOps into code with a `Renderer`, then we compile the code to binary with a `Compiler`.

## Execution

Creating `ExecItem`, which has a run method

::: tinygrad.engine.realize.ExecItem
    options:
        members: true

Lists of `ExecItem` can be condensed into a single ExecItem with the Graph API (rename to Queue?)

## Runtime

Runtimes are responsible for device-specific interactions. They handle tasks such as initializing devices, allocating memory, loading/launching programs, and more. You can find more information about the runtimes API on the [runtime overview page](runtime.md).

All runtime implementations can be found in the [runtime directory](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime).

### HCQ Compatible Runtimes

HCQ API is a lower-level API for defining runtimes. Interaction with HCQ-compatible devices occurs at a lower level, with commands issued directly to hardware queues. Some examples of such backends are [NV](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_nv.py) and [AMD](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_amd.py), which are userspace drivers for NVIDIA and AMD devices respectively. You can find more information about the API on [HCQ overview page](hcq.md)
