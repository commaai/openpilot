import random, operator
import z3
from tinygrad import Variable, dtypes
from tinygrad.uop.ops import UOp, graph_rewrite
from tinygrad.uop.spec import z3_renderer
from tinygrad.helpers import DEBUG, Context

seed = random.randint(0, 100)
print(f"Seed: {seed}")
random.seed(seed)

unary_ops = [lambda a:a+random.randint(-4, 4), lambda a: a*random.randint(-4, 4),
            lambda a: a//random.randint(1, 9), lambda a: a%random.randint(1, 9),
            lambda a:a.maximum(random.randint(-10, 10)), lambda a:a.minimum(random.randint(-10, 10))]
binary_ops = [lambda a,b: a+b, lambda a,b: a*b, lambda a,b:a.maximum(b), lambda a,b:a.minimum(b)]
comp_ops = [operator.lt, operator.le, operator.gt, operator.ge]

def random_or_sub_expression_int(depth, expr):
  sub_expr = random.choice([e for e in expr.toposort() if e.dtype is not dtypes.bool])
  return random.choice([random_int_expr(depth-1), sub_expr])

def random_int_expr(depth=10):
  if depth <= 0: return random.choice(v)
  expr1 = random_int_expr(depth-1)

  # we give more weight to arithmatic ops than to minimum and maximum
  ops = [
    lambda: random.choices(unary_ops, weights=[4, 4, 4, 4, 1, 1])[0](expr1),
    # for the second operand its either another random exprssion or some subexpression of the first operand
    lambda: random.choices(binary_ops, [8, 1, 1, 1])[0](expr1, random_or_sub_expression_int(depth-1, expr1)),
    lambda: random_bool_expr(3, random_or_sub_expression_int(depth-1, expr1)).where(expr1, random_or_sub_expression_int(depth-1, expr1)),
  ]
  # we give weight proportional to the amount of ops in each branch
  return random.choices(ops, weights=[6, 4, 1])[0]()

def random_bool_expr(depth=10, expr1=None):
  if depth == 0: return True
  if expr1 is None: expr1 = random_int_expr(depth-1)
  expr2 = random.choice([random_or_sub_expression_int(depth-1, expr1), UOp.const(dtypes.int, random.randint(-10, 10))])
  return random.choice(comp_ops)(expr1, expr2)


if __name__ == "__main__":
  skipped = 0
  for i in range(10000):
    if i % 1000 == 0:
      print(f"Running test {i}")
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256]
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    v = [u1,u2,u3]
    expr = random_int_expr(6)

    with Context(CORRECT_DIVMOD_FOLDING=1):
      simplified_expr = expr.simplify()

    solver = z3.Solver()
    solver.set(timeout=5000)  # some expressions take very long verify, but its very unlikely they actually return sat
    z3_sink = graph_rewrite(expr.sink(simplified_expr, u1, u2, u3), z3_renderer, ctx=(solver, {}))
    z3_expr, z3_simplified_expr = z3_sink.src[0].arg, z3_sink.src[1].arg
    check = solver.check(z3_simplified_expr != z3_expr)
    if check == z3.unknown and DEBUG>=1:
      skipped += 1
      print("Skipped due to timeout or interrupt:\n" +
            f"v1=Variable(\"{u1.arg[0]}\", {u1.arg[1]}, {u1.arg[2]})\n" +
            f"v2=Variable(\"{u2.arg[0]}\", {u2.arg[1]}, {u2.arg[2]})\n" +
            f"v3=Variable(\"{u3.arg[0]}\", {u3.arg[1]}, {u3.arg[2]})\n" +
            f"expr = {expr.render(simplify=False)}\n")
    elif check == z3.sat:
      m = solver.model()
      v1, v2, v3 = z3_sink.src[2].arg, z3_sink.src[3].arg, z3_sink.src[4].arg
      n1, n2, n3 = m[v1], m[v2], m[v3]
      u1_val, u2_val, u3_val = u1.const_like(n1.as_long()), u2.const_like(n2.as_long()), u3.const_like(n3.as_long())
      with Context(CORRECT_DIVMOD_FOLDING=1):
        num = expr.simplify().substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
        rn = expr.substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
        if num==rn: print("z3 found a mismatch but the expressions are equal!!")
      assert False, f"mismatched {expr.render()} at v1={m[v1]}; v2={m[v2]}; v3={m[v3]} = {num} != {rn}\n" +\
            "Reproduce with:\n" +\
            f"v1=Variable(\"{u1.arg[0]}\", {u1.arg[1]}, {u1.arg[2]})\n" +\
            f"v2=Variable(\"{u2.arg[0]}\", {u2.arg[1]}, {u2.arg[2]})\n" +\
            f"v3=Variable(\"{u3.arg[0]}\", {u3.arg[1]}, {u3.arg[2]})\n" +\
            f"expr = {expr}\n" +\
            f"v1_val, v2_val, v3_val = UOp.const(dtypes.int, {n1.as_long()}), UOp.const(dtypes.int, {n2.as_long()})," +\
                f"UOp.const(dtypes.int, {n3.as_long()})\n" +\
            "num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()\n" +\
            "rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()\n" +\
            "assert num==rn, f\"{num} != {rn}\"\n"

    if DEBUG >= 2: print(f"validated {expr.render()}")
  print(f"Skipped {skipped} expressions due to timeout")
