from tinygrad.ops import MovementOps, ProcessingOps

# input format is    N, H x W, C//4 x 4
# dweight format is  oc//4 x ch, cw x 4(oc)
# weight format is   oc//4 x ch, ic//4, cw, 4(oc) x 4(ic)
def preprocessing_op(x,w,C,make_image=True):
  w = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W))
  #print(x.shape, w.shape)

  if C.bs > 1 and C.py > 0:
    # explicitly add y-padding for batched inputs
    # N C H W
    xs = [(0, 0) for _ in x.shape]
    xs[2] = (C.py, C.py)
    x = x.movement_op(MovementOps.PAD, xs)
    C = C._replace(iy=C.iy + C.py*2, py=0)

  # hack for non multiples of 4 on C.cin
  if C.cin % 4 != 0 and not (C.cin == 1 and C.groups%4 == 0):
    to_add = 4 - (C.cin % 4)
    ws = [(0, 0) for _ in w.shape]
    ws[2] = (0, to_add)
    w = w.movement_op(MovementOps.PAD, ws)

    x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix))
    xs = [(0, 0) for _ in x.shape]
    xs[2] = (0, to_add)
    x = x.movement_op(MovementOps.PAD, xs)
    C = C._replace(cin = C.cin + to_add)
    x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups*C.cin, C.iy, C.ix))

  # hack for non multiples of 4 on C.rcout
  if C.rcout % 4 != 0 and not (C.rcout == 1 and C.groups%4 == 0):
    added_output_channels = 4 - (C.rcout % 4)
    ws = [(0, 0) for _ in w.shape]
    ws[1] = (0, added_output_channels)
    w = w.movement_op(MovementOps.PAD, ws)
    C = C._replace(rcout = C.rcout + added_output_channels, cout = C.groups * (C.rcout + added_output_channels))

  # packed
  assert (C.groups*C.cin) % 4 == 0
  #print(x.shape)
  x = x.movement_op(MovementOps.PERMUTE, (0,2,3,1))
  x = x.movement_op(MovementOps.RESHAPE, (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4))

  assert C.cout % 4 == 0
  if C.cin == 1:
    # depthwise
    w = w.movement_op(MovementOps.RESHAPE, (C.cout//4,4,C.H*C.W))
    w = w.movement_op(MovementOps.PERMUTE, (0,2,1))
  else:
    w = w.movement_op(MovementOps.RESHAPE, (C.cout//4,4,C.cin//4,4,C.H,C.W))
    w = w.movement_op(MovementOps.PERMUTE, (0,4,2,5,1,3))
    w = w.movement_op(MovementOps.RESHAPE, (C.cout//4, C.H * C.cin//4 * C.W * 4, 4))

  C = C._replace(out_shape = (C.bs*C.oy, C.ox*C.cout//4, 4))
  #x = contiguous(ctx, x, x.shapetracker) if not x.shapetracker.contiguous else x
  #w = contiguous(ctx, w, w.shapetracker) if not w.shapetracker.contiguous else w

  # contiguous before image, always
  x = x.contiguous()
  w = w.contiguous()

  # early realize on the weights
  bw = w
  while getattr(bw, 'op', None) and len(bw.op.src) == 1:
    bw = bw.op.src[0]
  if bw.realized:
    # weights are static
    wr = w.realize() #.image
    if make_image:
      wr.image
  return x,w,C

def postprocessing_op(ret, C, C_initial):
  added_output_channels = C.rcout - C_initial.rcout

  # undo hack for non multiples of 4 on C.rcout
  if added_output_channels != 0:
    ret = ret.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.groups, C.rcout))
    xs = [(0, s) for s in ret.shape]
    xs[4] = (0, ret.shape[4]-added_output_channels)
    ret = ret.movement_op(MovementOps.SHRINK, xs)
    C = C._replace(rcout = C.rcout - added_output_channels, cout = C.groups * (C.rcout - added_output_channels))

  ret = ret.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.cout))
  ret = ret.movement_op(MovementOps.PERMUTE, (0,3,1,2))
  return ret
