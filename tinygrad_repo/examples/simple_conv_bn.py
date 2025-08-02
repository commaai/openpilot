from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.nn.state import get_parameters

if __name__ == "__main__":
  with Tensor.train():

    BS, C1, H, W = 4, 16, 224, 224
    C2, K, S, P = 64, 7, 2, 1

    x = Tensor.uniform(BS, C1, H, W)
    conv = Conv2d(C1, C2, kernel_size=K, stride=S, padding=P)
    bn = BatchNorm2d(C2, track_running_stats=False)
    for t in get_parameters([x, conv, bn]): t.realize()

    print("running network")
    x.sequential([conv, bn]).numpy()
