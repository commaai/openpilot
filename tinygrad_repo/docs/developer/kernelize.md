# Kernel Creation

Tinygrad lazily builds up a graph of Tensor operations. The Tensor graph includes a mix of:

- Buffer and Assignment Ops: `BUFFER`, `BUFFER_VIEW`, `COPY`, `ASSIGN`
- Movement Ops: `RESHAPE`, `EXPAND`, `PERMUTE`, `PAD`, `SHRINK`, `FLIP`
- Compute Ops: `ADD`, `MUL`, `REDUCE_AXIS`, ...

`Tensor.kernelize` creates the kernels and buffers needed to realize the output Tensor(s).

## Kernelize flow

Let's see how a multiply add Tensor graph becomes a fused elementwise kernel.

```py
# initialize 3 input buffers on the device
a = Tensor([1]).realize()
b = Tensor([2]).realize()
c = Tensor([3]).realize()

# create the Tensor graph
mul = a*b
out = mul+c

print(mul) # <Tensor <UOp METAL (1,) int (<Ops.MUL: 48>, None)> on METAL with grad None>
print(out) # <Tensor <UOp METAL (1,) int (<Ops.ADD: 52>, None)> on METAL with grad None>

out.kernelize()

print(mul) # <Tensor <UOp METAL (1,) int (<Ops.MUL: 48>, None)> on METAL with grad None>
print(out) # <Tensor <UOp METAL (1,) int (<Ops.ASSIGN: 66>, None)> on METAL with grad None>
```

The multiply Tensor stays the same because it is fused. The output Tensor's UOp becomes a new ASSIGN UOp:

```py
print(out.uop)
```

The first source is the output BUFFER:

```
UOp(Ops.BUFFER, dtypes.int, arg=1, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),
  UOp(Ops.UNIQUE, dtypes.void, arg=6, src=()),))
```

And the second source is the KERNEL and its 4 buffer edges (output_buffer, a, b, c):

```
UOp(Ops.KERNEL, dtypes.void, arg=<Kernel 12 SINK(<Ops.STORE: 45>,) (__add__, __mul__)>, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=1, src=(
    x1:=UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),
    UOp(Ops.UNIQUE, dtypes.void, arg=6, src=()),)),
  UOp(Ops.BUFFER, dtypes.int, arg=1, src=(
     x1,
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),)),
  UOp(Ops.BUFFER, dtypes.int, arg=1, src=(
     x1,
    UOp(Ops.UNIQUE, dtypes.void, arg=3, src=()),)),
  UOp(Ops.BUFFER, dtypes.int, arg=1, src=(
     x1,
    UOp(Ops.UNIQUE, dtypes.void, arg=5, src=()),)),))
```

KERNEL describes the compute AST, metadata and memory dependencies.

BUFFER holds a reference to the device memory where the output will be stored.

Once a Tensor is kernelized, all children will LOAD its BUFFER, instead of fusing it:

```py
child = out+2
child.kernelize()
print(child.uop.src[1].arg.ast)
```

```
UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=0, src=()),
    x2:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=1, src=()),
         x2,)),
      UOp(Ops.CONST, dtypes.int, arg=2, src=(
         x2,)),)),)),))
```

`Tensor.realize` will execute the kernels and write outputs to memory:

```py
Tensor.realize(out)
print(out)        # <Tensor <UOp METAL (1,) int (<Ops.BUFFER: 23>, <buf real:True device:METAL size:1 dtype:dtypes.int offset:0>)> on METAL with grad None>
print(out.item()) # 5
```

<hr />

**Summary**

- The large Tensor graph is built from a mix of data, compute and movement Ops.

- `Tensor.kernelize` splits the Tensor graph into data (BUFFER), compute (KERNEL) and links dependencies with ASSIGN.

- `Tensor.realize` executes KERNELs on device and replaces the Tensor graph with just a BUFFER.

- Kernelize can be called multiple times on a Tensor. This allows for incrementally building the kernel fusion layout of a large Tensor graph, without having to call `realize` or `schedule`.
