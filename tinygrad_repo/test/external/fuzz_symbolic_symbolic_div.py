import random
import z3
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.validate import uops_to_z3
from tinygrad.helpers import DEBUG, Context, colored

seed = random.randint(0, 100)
print(f"Seed: {seed}")
random.seed(seed)

def get_random_term(ranges, factors):
  # 10% chance of nesting
  if random.randint(0,9) == 0: return get_random_expr(ranges, factors)
  return random.choice(ranges)*random.choice(factors)*random.choice([1, 1, 1, -1])

def get_random_expr(ranges, factors):
  num_terms = random.randint(2,4)
  x = UOp.sum(*[get_random_term(ranges, factors) for _ in range(num_terms)])
  return x.alu(random.choice([Ops.IDIV, Ops.MOD]), x.ufix(random.choice(factors)*random.choice([1, 1, 1, -1])))

if __name__ == "__main__":
  skipped = 0
  for i in range(700):
    if i % 100 == 0:
      print(f"Running test {i}")
    upper_bounds = [*list(range(1, 4)), 16, 33, 53, 64, 256]
    variable_names = ["i", "j", "k"]
    variables = [UOp.variable(s, 1, random.choice(upper_bounds)) for s in variable_names]
    factors = variables+upper_bounds
    # add some products
    for _ in range(2): factors.append(random.choice(variables)*random.choice(variables))
    # add some adds
    for _ in range(2): factors.append(random.choice(variables)+random.choice(factors))
    num_ranges = 4
    ranges = [UOp.range(random.choice(factors), i) for i in range(num_ranges)]
    variable_names += [f"r{i}" for i in range(num_ranges)]
    expr = get_random_expr(ranges, factors)

    with Context(CORRECT_DIVMOD_FOLDING=1):
      simplified_expr = expr.simplify()

    if DEBUG>=1:
      print(expr.render(simplify=False), "  -->  ", simplified_expr.render(simplify=False))

    solver = z3.Solver()
    solver.set(timeout=3000)  # some expressions take very long verify, but its very unlikely they actually return sat
    z3_expr, z3_simplified_expr, *z3_vars = uops_to_z3(solver, expr, simplified_expr, *variables, *ranges)
    check = solver.check(z3_simplified_expr != z3_expr)
    if check == z3.unknown and DEBUG>=1:
      skipped += 1
      print("skipped z3 verification due to timeout")
    elif check == z3.sat:
      print(colored("simplify INCORRECT!", "red"))
      print(solver.model())
      var_vals = {s:solver.model()[z] for s,z in zip(variable_names, z3_vars)}
      print("reproduce with:")
      print("var_vals = ", var_vals)
      print("globals = var_vals|{'cdiv':cdiv,'cmod':cmod}")
      print("expr = ast.simplify()")
      print("assert eval(ast.render(pm=renderer_infer, simplify=False),globals) == eval(expr.render(pm=renderer_infer, simplify=False),globals)")
      print()

      assert False

    if DEBUG >= 2: print(f"validated {expr.render()}")
  print(f"Skipped {skipped} expressions due to timeout")
