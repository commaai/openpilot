# speed in tinygrad

## Overview

Speed refers to many different things. To break it down to four, there's:

- Compile Speed (Python)
- Execution Speed (driver)
- Model Speed (scheduler)
- Kernel Speed (codegen)

## Compile Speed (Python)

This is how long the first run of your model takes. It's limited largely by the runtime of the Python doing UOp rewrites. Currently it's a bit slow, but on par with torch.compile. It gets even slower if you are using BEAM, since that's compiling many variants of each kernel.

This will be improved by writing faster graph_rewrite, doing less graph_rewrite, and better parallelization.

## Execution Speed (driver)

After your model is compiled, you are often using the `TinyJIT`. tinygrad has the best execution speed of any framework because it usually bypasses the GPU driver and prebuilds the command queue. It's tons faster than normal CUDA, and often even faster than CUDA Graph.

There's very little to improve here, as this is almost never the bottleneck.

## Model Speed (scheduler)

The scheduler determines how operations are grouped into kernels and which Tensors are written to memory. This is currently a big bottleneck of training speed.

The decisions are often not obvious. For example, when is it worth recomputing an arithmetic operation instead of storing and loading from memory? Example:

```python
from tinygrad import Tensor
a = Tensor.rand(100)
b = Tensor.rand(100)
c = Tensor.rand(100)
d = Tensor.rand(100)
out1 = a+b+c
out2 = a+b+d
Tensor.realize(out1, out2)
```

The real answer is obvious, compute both `out1` and `out2` in the same kernel. But you can't always do that. If you can't, should `a+b` first be saved to a subbuffer? Or should both the `out1` and `out2` kernels recompute `a+b`?

In this case: with recompute (6 reads + 2 writes), no recompute (6 reads + 3 writes), so we should probably recompute. However, once you add movement ops and casts this is even harder to figure out. tinygrad doesn't yet have a systematic way to do it.

## Kernel Speed (codegen)

Given that you have decided how the model ops will be grouped and what will be written to memory, kernel speed determines how fast that operation is done. This is what BEAM changes, it searches over a set of equivalent kernels which all perform the same operation and finds the one which performs the task the fastest.

In `kernel.py` we have a set of `OptOps`, these control the parameters of the speed optimizations applied to the kernel.

### Memory

The main bottleneck in most kernels is accessing memory. In a freshman algorithms class, you'll learn about cache aware matrix multiplication, and this is all forms of that. While the same math is run, the order in which you run it can have large impacts on the speed depending on if the data you are loading. OptOps will change this order.

Memory, even cache, is often much slower than accessing the register file. The amount of times data is used in math is called the "arithmetic intensity". For operations like BS=1 GEMV, the arithmetic intensity is 1, but for GEMMs and convs it can be much higher. OptOps like UPCAST and UNROLL can increase this, but be careful of making them too large, as if there's too much register pressure on the GPU the warp scheduler may not be able to fit many warps, or even worse, it could be spilling to local memory.

4090s have 1 TB/s of ram bandwidth and ~160 TFLOPS of compute, so you need to use each loaded value ~100 times. The L1 cache has around 40 TB/s of bandwidth, so in order to get full compute utilization you need to use each value ~4 times.

A lot of work can still be done here. For example, we never copy the inputs to on chip SRAM, but this is often quite helpful for kernel speed. Also, we aren't doing a good job with L2 cache awareness (the locals handle L1 quite well)

### Tensor Cores

Many accelerators have Tensor Cores / MAC arrays / systolic arrays. The main value of these is that, since they are 2-D, they create an n^2 ratio between the compute and the input data.

GPUs use Tensor Cores instead of MAC arrays to fit better in the GPU warp paradigm. This is because the output of Tensor Cores is O(n) wrt the input, while the output of MAC arrays like the AMX is O(n^2)

We have a simple framework in tinygrad for adding these ALU blocks and achieving good performance from them.

### Indexing

Indexing determines the address of the memory we need to load. GPUs often have less integer math resources than floating point math, so this can sometimes be the bottleneck. We have a symbolic math engine in our rewrite rules to simplifiy indexing before it's emitted to the kernel. Newer NVIDIA GPUs have a "Tensor Memory Accelerator" to assist with fast indexing, however, this is not supported in tinygrad yet.
