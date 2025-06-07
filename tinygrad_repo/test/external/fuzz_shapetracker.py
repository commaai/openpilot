import random
from tinygrad.helpers import DEBUG, getenv
from test.unit.test_shapetracker import CheckingShapeTracker

def do_permute(st):
  perm = list(range(0, len(st.shape)))
  random.shuffle(perm)
  perm = tuple(perm)
  if DEBUG >= 1: print("st.permute(", perm, ")")
  st.permute(perm)

def do_pad(st):
  c = random.randint(0, len(st.shape)-1)
  pad = tuple((random.randint(0,2), random.randint(0,2)) if i==c else (0,0) for i in range(len(st.shape)))
  if DEBUG >= 1: print("st.pad(", pad, ")")
  st.pad(pad)

def do_reshape_split_one(st):
  c = random.randint(0, len(st.shape)-1)
  poss = [n for n in [1,2,3,4,5] if st.shape[c]%n == 0]
  spl = random.choice(poss)
  shp = st.shape[0:c] + (st.shape[c]//spl, spl) + st.shape[c+1:]
  if DEBUG >= 1: print("st.reshape(", shp, ")")
  st.reshape(shp)

def do_reshape_combine_two(st):
  if len(st.shape) < 2: return
  c = random.randint(0, len(st.shape)-2)
  shp = st.shape[:c] + (st.shape[c] * st.shape[c+1], ) + st.shape[c+2:]
  if DEBUG >= 1: print("st.reshape(", shp, ")")
  st.reshape(shp)

def do_shrink(st):
  c = random.randint(0, len(st.shape)-1)
  while 1:
    shrink = tuple((random.randint(0,s), random.randint(0,s)) if i == c else (0,s) for i,s in enumerate(st.shape))
    if all(x<y for (x,y) in shrink): break
  if DEBUG >= 1: print("st.shrink(", shrink, ")")
  st.shrink(shrink)

def do_flip(st):
  flip = tuple(random.random() < 0.5 for _ in st.shape)
  if DEBUG >= 1: print("st.flip(", flip, ")")
  st.flip(flip)

def do_expand(st):
  c = [i for i,s in enumerate(st.shape) if s==1]
  if len(c) == 0: return
  c = random.choice(c)
  expand = tuple(random.choice([2,3,4]) if i==c else s for i,s in enumerate(st.shape))
  if DEBUG >= 1: print("st.expand(", expand, ")")
  st.expand(expand)

shapetracker_ops = [do_permute, do_pad, do_shrink, do_reshape_split_one, do_reshape_combine_two, do_flip, do_expand]

if __name__ == "__main__":
  random.seed(42)
  for _ in range(getenv("CNT", 200)):
    st = CheckingShapeTracker((random.randint(2, 10), random.randint(2, 10), random.randint(2, 10)))
    for i in range(8): random.choice(shapetracker_ops)(st)
    st.assert_same()
