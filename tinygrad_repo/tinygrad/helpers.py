import os, math, functools
from typing import Tuple, Union, List

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def prod(x:Union[List[int], Tuple[int, ...]]) -> int: return math.prod(x)
def argfix(*x): return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], (tuple, list)) else tuple(x)
def argsort(x): return sorted(range(len(x)), key=x.__getitem__) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items) if len(items) > 0 else True
def colored(st, color, background=False, bright=False): return f"\u001b[{10*background+60*bright+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m"  # replace the termcolor library with one line
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l): return [item for sublist in l for item in sublist]
def mnum(i) -> str: return str(i) if i >= 0 else f"m{-i}"

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG, IMAGE = getenv("DEBUG", 0), getenv("IMAGE", 0)
