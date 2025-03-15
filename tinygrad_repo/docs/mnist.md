# MNIST Tutorial

After you have installed tinygrad, this is a great first tutorial.

Start up a notebook locally, or use [colab](https://colab.research.google.com/). tinygrad is very lightweight, so it's easy to install anywhere and doesn't need a special colab image, but for speed we recommend a T4 GPU image.

### One-liner to install tinygrad in colab

```python
!pip install git+https://github.com/tinygrad/tinygrad.git
```

### What's the default device?

```python
from tinygrad import Device
print(Device.DEFAULT)
```

You will see `CUDA` here on a GPU instance, or `CLANG` here on a CPU instance.

## A simple model

We'll use the model from [the Keras tutorial](https://keras.io/examples/vision/mnist_convnet/).

```python
from tinygrad import Tensor, nn

class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))
```

Two key differences from PyTorch:

* Only the stateful layers are declared in `__init__`
* There's no `nn.Module` class or `forward` function, just a normal class and `__call__`

### Getting the dataset

```python
from tinygrad.nn.datasets import mnist
X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar
```

tinygrad includes MNIST, it only adds four lines. Feel free to read the [function](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/datasets.py).

## Using the model

MNIST is small enough that the `mnist()` function copies the dataset to the default device.

So creating the model and evaluating it is a matter of:

```python
model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model
```

### Training the model

We'll use the Adam optimizer. The `nn.state.get_parameters` will walk the model class and pull out the parameters for the optimizer. Also, in tinygrad, it's typical to write a function to do the training step so it can be jitted.

```python
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss
```

You can time a step with:

```python
import timeit
timeit.repeat(step, repeat=5, number=1)
#[0.08268719699981375,
# 0.07478952900009972,
# 0.07714716600003158,
# 0.07785399599970333,
# 0.07605237000007037]
```

So around 75 ms on T4 colab.

If you want to see a breakdown of the time by kernel:

```python
from tinygrad import GlobalCounters, Context
GlobalCounters.reset()
with Context(DEBUG=2): step()
```

### Why so slow?

Unlike PyTorch, tinygrad isn't designed to be fast like that. While 75 ms for one step is plenty fast for debugging, it's not great for training. Here, we introduce the first quintessentially tinygrad concept, the `TinyJit`.

```python
from tinygrad import TinyJit
jit_step = TinyJit(step)
```

NOTE: It can also be used as a decorator `@TinyJit`

Now when we time it:

```python
import timeit
timeit.repeat(jit_step, repeat=5, number=1)
# [0.2596786549997887,
#  0.08989566299987928,
#  0.0012115650001760514,
#  0.001010227999813651,
#  0.0012164899999334011]
```

1.0 ms is 75x faster! Note that we aren't syncing the GPU, so GPU time may be slower.

The slowness the first two times is the JIT capturing the kernels. And this JIT will not run any Python in the function, it will just replay the tinygrad kernels that were run, so be aware that non tinygrad Python operations won't work. Randomness functions work as expected.

Unlike other JITs, we JIT everything, including the optimizer. Think of it as a dumb replay on different data.

## Putting it together

Since we are just randomly sampling from the dataset, there's no real concept of an epoch. We have a batch size of 128, so the Keras example is taking about 7000 steps.

```python
for step in range(7000):
  loss = jit_step()
  if step%100 == 0:
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
```

It doesn't take long to reach 98%, and it usually reaches 99%.

```
step    0, loss 4.03, acc 71.43%
step  100, loss 0.34, acc 93.86%
step  200, loss 0.23, acc 95.97%
step  300, loss 0.18, acc 96.32%
step  400, loss 0.18, acc 96.76%
step  500, loss 0.13, acc 97.46%
step  600, loss 0.14, acc 97.45%
step  700, loss 0.10, acc 97.27%
step  800, loss 0.23, acc 97.49%
step  900, loss 0.13, acc 97.51%
step 1000, loss 0.13, acc 97.88%
step 1100, loss 0.11, acc 97.72%
step 1200, loss 0.14, acc 97.65%
step 1300, loss 0.12, acc 98.04%
step 1400, loss 0.25, acc 98.17%
step 1500, loss 0.11, acc 97.86%
step 1600, loss 0.21, acc 98.21%
step 1700, loss 0.14, acc 98.34%
...
```

## From here?

tinygrad is yours to play with now. It's pure Python and short, so unlike PyTorch, fixing library bugs is well within your abilities.

- It's two lines to add multiGPU support to this example (can you find them?). You have to `.shard` the model to all GPUs, and `.shard` the dataset by batch.
- `with Context(DEBUG=2)` shows the running kernels, `DEBUG=4` shows the code. All `Context` variables can also be environment variables.
- `with Context(BEAM=2)` will do a BEAM search on the kernels, searching many possible implementations for what runs the fastest on your hardware. After this search, tinygrad is usually speed competitive with PyTorch, and the results are cached so you won't have to search next time.

[Join our Discord](https://discord.gg/ZjZadyC7PK) for help, and if you want to be a tinygrad developer. Please read the Discord rules when you get there.

[Follow us on Twitter](https://twitter.com/__tinygrad__) to keep up with the project.
