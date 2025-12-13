from tinygrad.dtype import AddrSpace

from extra.thunder.tiny.tk import WARP_THREADS

class GL:
  def __init__(self, shape, dtype, ker):
    self.shape, self.dtype = shape, dtype
    self._uop = ker.alloc(shape, dtype, AddrSpace.GLOBAL)

class ST:
  def __init__(self, shape, dtype, ker):
    self.shape, self.dtype = shape, dtype
    self._uop = ker.alloc(shape, dtype, AddrSpace.LOCAL)

class RT:
  TILE_ROW_DIM, TILE_COL_DIM = 16, 16
  BASE_TILE_NE = TILE_ROW_DIM * TILE_COL_DIM
  BASE_TILE_NEPT = BASE_TILE_NE // WARP_THREADS

  def __init__(self, shape, dtype, ker):
    assert len(shape) == 2
    assert shape[0] % RT.TILE_ROW_DIM == 0
    assert shape[1] % RT.TILE_COL_DIM == 0

    height = shape[0] // RT.TILE_ROW_DIM
    width = shape[1] // RT.TILE_COL_DIM

    self.shape, self.dtype = (height, width, self.BASE_TILE_NEPT), dtype
    self._uop = ker.alloc(self.shape, dtype, AddrSpace.REG)

class RV:
  def __init__(self, length, dtype, layout, ker):
    tiles = length // RT.TILE_ROW_DIM

    match layout:
      case "naive":
        inner_dim = 1
        outer_dim = (tiles + 1) // 2
      case "ortho":
        inner_dim = 1
        outer_dim = tiles
      case _: raise NotImplementedError(f"rv layout {layout} not implemented")

    self.shape, self.dtype = (outer_dim, inner_dim, 2), dtype
    self._uop = ker.alloc(self.shape, dtype, AddrSpace.REG)
