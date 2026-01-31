from tinygrad import Tensor, nn, Context, GlobalCounters

if __name__ == "__main__":
  conv = nn.Conv2d(64, 128, 3)
  img = Tensor.randn((1,64,128,128))
  with Context(DEBUG=0, BEAM=0):
    Tensor.realize(img, conv.weight, conv.bias)

  tst = conv(img).permute(0,2,3,1).realize()
  print(tst.shape)

  print("NEW")
  img_perm = img.permute(0,2,3,1).contiguous()
  print(img_perm.shape)
  pp = img_perm.permute(0,3,1,2)._pool((3,3)).permute(0,2,3,4,5,1)

  def hwio(pp, conv):
    pp = pp.unsqueeze(-1)
    weight = conv.weight.permute(2,3,1,0).contiguous()
    print(pp.shape, weight.shape, (pp*weight).shape)
    return (pp * weight).sum([-4,-3, -2])

  def ohwi(pp, conv):
    pp = pp.unsqueeze(-4)
    weight = conv.weight.permute(0,2,3,1).contiguous()
    print(pp.shape, weight.shape, (pp*weight).shape)
    return (pp * weight).sum([-3,-2,-1])

  for f in [hwio, ohwi]:
    GlobalCounters.reset()
    print("\n**************", f.__name__, "**************")
    out = f(pp, conv)
    out.realize()
    print(out.shape)

    with Context(DEBUG=0, BEAM=0):
      err = (tst-out).square()
      print(err.mean().item(), err.max().item())
