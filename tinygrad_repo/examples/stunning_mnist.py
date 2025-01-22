# beautiful mnist in the new "one-shot" style
# one realize in the whole graph
# depends on:
#  - "big graph" UOp scheduling
#  - symbolic removal

from examples.beautiful_mnist import Model
from tinygrad import Tensor, nn, getenv, GlobalCounters
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import trange, DEBUG

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  print("*** got data")

  model = Model()
  print("*** got model")

  opt = nn.optim.Adam(nn.state.get_parameters(model))
  print("*** got optimizer")

  samples = Tensor.randint(getenv("STEPS", 10), getenv("BS", 512), high=X_train.shape[0])
  X_samp, Y_samp = X_train[samples], Y_train[samples]
  print("*** got samples")

  with Tensor.train():
    # TODO: this shouldn't be a for loop. something like: (contract is still up in the air)
    """
    i = UOp.range(samples.shape[0])  # TODO: fix range function on UOp
    losses = model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward().contract(i)
    opt.schedule_steps(i)
    """
    losses = []
    for i in range(samples.shape[0]):
      opt.zero_grad()
      losses.append(model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward())
      opt.schedule_step()
    # TODO: this stack currently breaks the "generator" aspect of losses. it probably shouldn't
    #losses = Tensor.stack(*losses)
  print("*** scheduled training")

  # evaluate the model
  with Tensor.test():
    test_acc = ((model(X_test).argmax(axis=1) == Y_test).mean()*100)
  print("*** scheduled eval")

  # NOTE: there's no kernels run in the scheduling phase
  assert GlobalCounters.kernel_count == 0, "kernels were run during scheduling!"

  # only actually do anything at the end
  if getenv("LOSS", 1):
    for i in (t:=trange(len(losses))): t.set_description(f"loss: {losses[i].item():6.2f}")
  print(f"test_accuracy: {test_acc.item():5.2f}%")
