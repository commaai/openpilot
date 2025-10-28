from tinygrad import Tensor

def bit_extract(x: Tensor, e: int, s: int) -> Tensor:
  mask = (1 << (e - s + 1)) - 1
  return (x >> s) & mask

def u16_to_f16(x: Tensor) -> Tensor:
  sign = bit_extract(x, 15, 15).float()
  exponent = bit_extract(x, 14, 10).float()
  fraction = bit_extract(x, 9, 0).float()
  return sign.where(-1, 1) * exponent.where((exponent - 15.0).exp2() * (1 + fraction / 1024.0), 6.103515625e-5 * (fraction / 1024.0))

def u32_to_f16(oo: Tensor) -> Tensor:
  f1 = u16_to_f16(oo>>16)
  f2 = u16_to_f16(oo&0xFFFF)
  return Tensor.cat(f2.reshape(-1, 1), f1.reshape(-1, 1), dim=1).flatten()
