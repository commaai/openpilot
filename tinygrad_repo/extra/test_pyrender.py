from extra.optimization.helpers import load_worlds, ast_str_to_ast
from tinygrad.helpers import tqdm
from tinygrad.uop.ops import pyrender, UOp, Ops
from tinygrad import dtypes
from tinygrad.shape.shapetracker import ShapeTracker, View
inf, nan = float('inf'), float('nan')

if __name__ == "__main__":
  ast_strs = load_worlds()
  for i, ast_str in enumerate(tqdm(ast_strs)):
    good_ast = ast_str_to_ast(ast_str)
    code = '\n'.join(pyrender(good_ast))
    print("\n***************\n\n"+code)
    exec(code)
    if str(good_ast) != str(ast):
      print(code)
      print("MISMATCH")
      print(good_ast)
      print(ast)
      break