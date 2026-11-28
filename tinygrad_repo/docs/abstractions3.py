# abstractions2 goes from back to front, here we will go from front to back

# *****
# 0. Load mnist on the device

from tinygrad.nn.datasets import mnist
X_train, Y_train, _, _ = mnist()
X_train = X_train.float()
X_train -= X_train.mean()

# *****
# 1. Define an MNIST model.

from tinygrad import Tensor, Context

l1 = Tensor.kaiming_uniform(128, 784)
l2 = Tensor.kaiming_uniform(10, 128)
def model(x): return x.flatten(1).dot(l1.T).relu().dot(l2.T)
l1n, l2n = l1.numpy(), l2.numpy()

# *****
# 2. Choose a batch for training and do the backward pass.

from tinygrad.nn.optim import SGD
optim = SGD([l1, l2])

with Context(TRAINING=1):
  X, Y = X_train[(samples:=Tensor.randint(128, high=X_train.shape[0]))], Y_train[samples]
  optim.zero_grad()
  model(X).sparse_categorical_crossentropy(Y).backward()
  optim.schedule_step()   # this will step the optimizer without running realize

# *****
# 3. Export and compile to a linear uop of compiled calls.

# The weight Tensors have been assigned to, but not yet realized. Everything is still lazy at this point
# l1.uop and l2.uop define a computation graph

# export gives the Tensors storage and returns the graph of stores into it as a function with no runtime state,
# its EXTERNs are the storage (and bound Variables) listed in externs
fn, externs = Tensor.export(l1, l2)

# the compiler turns the function into a linear of compiled calls, it only sees fn and the targets in DEV
linear = Tensor.compile(fn)

print(f"The compiled linear contains {len(linear.src)} calls.")
for call in linear.src: print(str(call)[:80])

# *****
# 4. Run the compiled linear on the storage, it's linked first.

Tensor.run(linear, externs)

# *****
# 5. Print the weight change

print("first weight change\n", l1.numpy()-l1n)
print("second weight change\n", l2.numpy()-l2n)
