#!/usr/bin/env python3
import argparse
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, namedtuple
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_c as ts_c
import tree_sitter as ts


ROOT = Path(__file__).resolve().parents[3]
SAFETY_DIR = ROOT / "opendbc" / "safety"
SAFETY_TESTS_DIR = ROOT / "opendbc" / "safety" / "tests"
SAFETY_C_REL = Path("opendbc/safety/tests/libsafety/safety.c")

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"

COMPARISON_OPERATOR_MAP = {
  "==": "!=",
  "!=": "==",
  ">": "<=",
  ">=": "<",
  "<": ">=",
  "<=": ">",
}

MUTATOR_FAMILIES = {
  "increment": ("update_expression", {"++": "--"}),
  "decrement": ("update_expression", {"--": "++"}),
  "comparison": ("binary_expression", COMPARISON_OPERATOR_MAP),
  "boundary": ("number_literal", {}),
  "bitwise_assignment": ("assignment_expression", {"&=": "|=", "|=": "&=", "^=": "&="}),
  "bitwise": ("binary_expression", {"&": "|", "|": "&", "^": "&"}),
  "arithmetic_assignment": ("assignment_expression", {"+=": "-=", "-=": "+=", "*=": "/=", "/=": "*=", "%=": "*="}),
  "arithmetic": ("binary_expression", {"+": "-", "-": "+", "*": "/", "/": "*", "%": "*"}),
  "remove_negation": ("unary_expression", {"!": ""}),
}


_RawSite = namedtuple('_RawSite', 'expr_start expr_end op_start op_end line original_op mutated_op mutator')


@dataclass(frozen=True)
class MutationSite:
  site_id: int
  expr_start: int
  expr_end: int
  op_start: int
  op_end: int
  line: int
  original_op: str
  mutated_op: str
  mutator: str
  origin_file: Path
  origin_line: int


@dataclass(frozen=True)
class MutantResult:
  site: MutationSite
  outcome: str  # killed | survived | infra_error
  test_sec: float
  details: str


def colorize(text, color):
  term = os.environ.get("TERM", "")
  if not sys.stdout.isatty() or term in ("", "dumb") or "NO_COLOR" in os.environ:
    return text
  return f"{color}{text}{ANSI_RESET}"


def format_mutation(original_op, mutated_op):
  return colorize(f"{original_op}->{mutated_op}", ANSI_RED)


def _parse_int_literal(token):
  m = re.fullmatch(r"([0-9][0-9a-fA-FxX]*)([uUlL]*)", token)
  if m is None:
    return None
  body, suffix = m.groups()
  try:
    value = int(body, 0)
  except ValueError:
    return None
  base = "hex" if body.lower().startswith("0x") else "dec"
  return value, base, suffix


def _site_key(site):
  return (site.op_start, site.op_end, site.mutator)


def _is_in_constexpr_context(node):
  """Check if a node is inside a static or file-scope variable initializer."""
  current = node.parent
  while current is not None:
    if current.type == "init_declarator":
      decl = current.parent
      if decl and decl.type == "declaration":
        for child in decl.children:
          if child.type == "storage_class_specifier" and child.text == b"static":
            return True
        if decl.parent and decl.parent.type == "translation_unit":
          return True
    current = current.parent
  return False


def _prepare_for_parsing(txt):
  """Blank line markers and replace __typeof__() for tree-sitter. Preserves byte offsets."""
  result = re.sub(
    r'^[ \t]*#[ \t]+\d+[ \t]+"[^\n]*',
    lambda m: " " * len(m.group()),
    txt,
    flags=re.MULTILINE,
  )
  # Replace __typeof__(...) with padded int (handle nested parens)
  parts = []
  i = 0
  for m in re.finditer(r"(?:__typeof__|typeof)\s*\(", result):
    if m.start() < i:
      continue  # skip nested typeof inside already-replaced region
    parts.append(result[i:m.start()])
    depth = 1
    j = m.end()
    while j < len(result) and depth > 0:
      if result[j] == "(":
        depth += 1
      elif result[j] == ")":
        depth -= 1
      j += 1
    parts.append("int" + " " * (j - m.start() - 3))
    i = j
  parts.append(result[i:])
  return "".join(parts)


def enumerate_sites(input_source, preprocessed_file):
  subprocess.run([
    "cc", "-E", "-std=gnu11", "-nostdlib", "-fno-builtin", "-DALLOW_DEBUG",
    f"-I{ROOT}", f"-I{ROOT / 'opendbc/safety/board'}",
    str(input_source), "-o", str(preprocessed_file),
  ], cwd=ROOT, capture_output=True, check=True)

  txt = preprocessed_file.read_text()

  # Build line map from preprocessor directives
  line_map = {}
  current_map_file = None
  current_map_line = None
  directive_re = re.compile(r'^\s*#\s*(\d+)\s+"([^"]+)"')
  for pp_line_num, pp_line in enumerate(txt.splitlines(keepends=True), start=1):
    m = directive_re.match(pp_line)
    if m:
      current_map_line = int(m.group(1))
      current_map_file = Path(m.group(2)).resolve()
      continue
    if current_map_file is not None and current_map_line is not None:
      line_map[pp_line_num] = (current_map_file, current_map_line)
      current_map_line += 1

  # Parse with tree-sitter
  parser = ts.Parser(ts.Language(ts_c.language()))
  tree = parser.parse(_prepare_for_parsing(txt).encode())

  # Build rule map
  rule_map = {}
  counts = {}
  for mutator, (node_kind, op_map) in MUTATOR_FAMILIES.items():
    counts[mutator] = 0
    if mutator == "boundary":
      continue
    for original_op, mutated_op in op_map.items():
      rule_map.setdefault((node_kind, original_op), []).append((mutator, original_op, mutated_op))

  # Walk tree to find mutation sites
  deduped = {}
  build_incompatible_keys = set()
  stack = [tree.root_node]
  while stack:
    node = stack.pop()
    kind = node.type

    # Boundary mutations: find number_literals inside comparison operands
    if kind == "binary_expression":
      cmp_op = node.child_by_field_name("operator")
      if cmp_op and cmp_op.type in COMPARISON_OPERATOR_MAP:
        lit_stack = []
        for field in ("left", "right"):
          operand = node.child_by_field_name(field)
          if operand:
            lit_stack.append(operand)
        while lit_stack:
          n = lit_stack.pop()
          if n.type == "number_literal":
            token = txt[n.start_byte:n.end_byte]
            parsed = _parse_int_literal(token)
            if parsed:
              value, base, suffix = parsed
              mutated = f"0x{value + 1:X}{suffix}" if base == "hex" else f"{value + 1}{suffix}"
              line = n.start_point[0] + 1
              bsite = _RawSite(n.start_byte, n.end_byte, n.start_byte, n.end_byte, line, token, mutated, "boundary")
              key = _site_key(bsite)
              deduped[key] = bsite
              if _is_in_constexpr_context(n):
                build_incompatible_keys.add(key)
          lit_stack.extend(n.children)

    # Operator mutations: any node with an operator child
    op_child = node.child_by_field_name("operator")
    if op_child:
      for mutator, original_op, mutated_op in rule_map.get((kind, op_child.type), []):
        line = node.start_point[0] + 1
        site = _RawSite(node.start_byte, node.end_byte, op_child.start_byte, op_child.end_byte, line, original_op, mutated_op, mutator)
        key = _site_key(site)
        deduped[key] = site
        if _is_in_constexpr_context(node):
          build_incompatible_keys.add(key)

    stack.extend(node.children)

  sites = sorted(deduped.values(), key=lambda s: (s.op_start, s.mutator))
  out = []
  build_incompatible_site_ids = set()
  for s in sites:
    mapped = line_map.get(s.line)
    if mapped is None:
      continue
    origin_file, origin_line = mapped
    if SAFETY_DIR not in origin_file.parents and origin_file != SAFETY_DIR:
      continue
    site_id = len(out)
    site = MutationSite(
      site_id=site_id, expr_start=s.expr_start, expr_end=s.expr_end,
      op_start=s.op_start, op_end=s.op_end, line=s.line,
      original_op=s.original_op, mutated_op=s.mutated_op, mutator=s.mutator,
      origin_file=origin_file, origin_line=origin_line,
    )
    if _site_key(s) in build_incompatible_keys:
      build_incompatible_site_ids.add(site_id)
    out.append(site)
    counts[s.mutator] += 1
  return out, counts, build_incompatible_site_ids, txt


def _build_core_tests(catalog):
  """Build test ordering for core (non-mode) files.

  One test per unique method name from evenly-spaced modules,
  ordered by how widely each method is shared. Methods inherited by many
  classes exercise the most fundamental safety logic and run first.
  """
  MAX_PER_METHOD = 5
  method_freq = {}
  method_by_module = {}
  for name in sorted(catalog.keys()):
    for test_id in catalog[name]:
      method = test_id.rsplit(".", 1)[-1]
      method_freq[method] = method_freq.get(method, 0) + 1
      if method not in method_by_module:
        method_by_module[method] = {}
      if name not in method_by_module[method]:
        method_by_module[method][name] = test_id
  # Pick evenly-spaced modules for each method to maximize configuration diversity
  method_ids = {}
  for method, module_map in method_by_module.items():
    modules = sorted(module_map.keys())
    n = len(modules)
    if n <= MAX_PER_METHOD:
      method_ids[method] = [module_map[m] for m in modules]
    else:
      step = n / MAX_PER_METHOD
      method_ids[method] = [module_map[modules[int(i * step)]] for i in range(MAX_PER_METHOD)]
  # Round-robin: first instance of each method (by freq), then second, etc.
  # This ensures diverse early coverage with failfast.
  sorted_methods = sorted(method_freq, key=lambda m: -method_freq[m])
  ordered = []
  for round_idx in range(MAX_PER_METHOD):
    for m in sorted_methods:
      ids = method_ids.get(m, [])
      if round_idx < len(ids):
        ordered.append(ids[round_idx])
  return ordered


def build_priority_tests(site, catalog, core_tests):
  """Build an ordered list of test IDs for a mutation site.

  For mode files: all tests from the matching test_<mode>.py module.
  For core files: uses the pre-computed core_tests ordering.
  """
  src = site.origin_file
  rel_parts = src.relative_to(ROOT).parts
  is_mode = len(rel_parts) >= 4 and rel_parts[:3] == ("opendbc", "safety", "modes")

  if is_mode:
    mode_file = f"test_{src.stem}.py"
    return list(catalog.get(mode_file, []))
  return core_tests


def format_site_snippet(site, context_lines=2):
  source = site.origin_file
  text = source.read_text()
  lines = text.splitlines()
  display_ln = site.origin_line
  line_idx = display_ln - 1
  start = max(0, line_idx - context_lines)
  end = min(len(lines), line_idx + context_lines + 1)

  line_text = lines[line_idx]
  rel_start = line_text.find(site.original_op)
  if rel_start < 0:
    rel_start = 0
  rel_end = rel_start + len(site.original_op)

  snippet_lines = []
  width = len(str(end))
  for idx in range(start, end):
    num = idx + 1
    prefix = ">" if idx == line_idx else " "
    line = lines[idx]
    if idx == line_idx:
      marker = colorize(f"[[{site.original_op}->{site.mutated_op}]]", ANSI_RED)
      line = f"{line[:rel_start]}{marker}{line[rel_end:]}"
    snippet_lines.append(f"{prefix} {num:>{width}} | {line}")
  return "\n".join(snippet_lines)


def render_progress(completed, total, killed, survived, infra, elapsed_sec):
  bar_width = 30
  filled = int((completed / total) * bar_width)
  bar = "#" * filled + "-" * (bar_width - filled)

  rate = completed / elapsed_sec if elapsed_sec > 0 else 0.0
  remaining = total - completed
  eta = (remaining / rate) if rate > 0 else 0.0

  killed_text = colorize(f"k:{killed}", ANSI_GREEN)
  survived_text = colorize(f"s:{survived}", ANSI_RED)
  infra_text = colorize(f"i:{infra}", ANSI_YELLOW)

  return f"[{bar}] {completed}/{total} {killed_text} {survived_text} {infra_text} mps:{rate:.2f} elapsed:{elapsed_sec:.1f}s eta:{eta:.1f}s"


def print_live_status(text, *, final=False):
  if sys.stdout.isatty():
    print("\r" + text, end="\n" if final else "", flush=True)
  else:
    print(text, flush=True)


def _discover_test_catalog():
  loader = unittest.TestLoader()
  catalog = {}
  for test_file in sorted(SAFETY_TESTS_DIR.glob("test_*.py")):
    module_name = ".".join(test_file.relative_to(ROOT).with_suffix("").parts)
    suite = loader.loadTestsFromName(module_name)
    catalog[test_file.name] = [t.id() for group in suite for t in group]
  return catalog


def run_unittest(targets, lib_path, mutant_id, verbose):
  from opendbc.safety.tests.libsafety import libsafety_py
  libsafety_py.load(lib_path)
  libsafety_py.libsafety.mutation_set_active_mutant(mutant_id)

  if verbose:
    print("Running unittest targets:", ", ".join(targets), flush=True)

  loader = unittest.TestLoader()
  stream = io.StringIO()
  runner = unittest.TextTestRunner(stream=stream, verbosity=0, failfast=True)

  suite = unittest.TestSuite()
  for target in targets:
    suite.addTests(loader.loadTestsFromName(target))
  result = runner.run(suite)
  if result.failures:
    return result.failures[0][0].id()
  if result.errors:
    return result.errors[0][0].id()
  return None


def _instrument_source(source, sites):
  # Sort by start ascending, end descending (outermost first when same start)
  sorted_sites = sorted(sites, key=lambda s: (s.expr_start, -s.expr_end))

  # Build containment forest using a stack
  roots = []
  stack = []
  for site in sorted_sites:
    while stack and stack[-1][0].expr_end <= site.expr_start:
      stack.pop()
    node = [site, []]
    if stack:
      stack[-1][1].append(node)
    else:
      roots.append(node)
    stack.append(node)

  def build_replacement(site, children):
    parts = []
    pos = site.expr_start
    op_rel = None
    running_len = 0

    for child_site, child_children in children:
      seg = source[pos : child_site.expr_start]
      if op_rel is None and site.op_start >= pos and site.op_start < child_site.expr_start:
        op_rel = running_len + (site.op_start - pos)
      parts.append(seg)
      running_len += len(seg)

      child_repl = build_replacement(child_site, child_children)
      parts.append(child_repl)
      running_len += len(child_repl)
      pos = child_site.expr_end

    seg = source[pos : site.expr_end]
    if op_rel is None and site.op_start >= pos:
      op_rel = running_len + (site.op_start - pos)
    parts.append(seg)

    expr_text = "".join(parts)
    op_len = site.op_end - site.op_start
    assert op_rel is not None and expr_text[op_rel : op_rel + op_len] == site.original_op, (
      f"Operator mismatch (site_id={site.site_id}): expected {site.original_op!r} at offset {op_rel}"
    )
    mutated_expr = f"{expr_text[:op_rel]}{site.mutated_op}{expr_text[op_rel + op_len :]}"
    return f"((__mutation_active_id == {site.site_id}) ? ({mutated_expr}) : ({expr_text}))"

  result_parts = []
  pos = 0
  for site, children in roots:
    result_parts.append(source[pos : site.expr_start])
    result_parts.append(build_replacement(site, children))
    pos = site.expr_end
  result_parts.append(source[pos:])
  return "".join(result_parts)


def compile_mutated_library(preprocessed_source, sites, output_so):
  instrumented = _instrument_source(preprocessed_source, sites)

  prelude = """
    static int __mutation_active_id = -1;
    void mutation_set_active_mutant(int id) { __mutation_active_id = id; }
    int mutation_get_active_mutant(void) { return __mutation_active_id; }
  """
  marker_re = re.compile(r'^\s*#\s+\d+\s+"[^\n]*\n?', re.MULTILINE)
  instrumented = prelude + marker_re.sub("", instrumented)

  mutation_source = output_so.with_suffix(".c")
  mutation_source.write_text(instrumented)

  subprocess.run([
    "cc", "-shared", "-fPIC", "-w", "-fno-builtin", "-std=gnu11",
    "-g0", "-O0", "-DALLOW_DEBUG",
    str(mutation_source), "-o", str(output_so),
  ], cwd=ROOT, check=True)


def eval_mutant(site, targets, lib_path, verbose):
  try:
    t0 = time.perf_counter()
    failed_test = run_unittest(targets, lib_path, mutant_id=site.site_id, verbose=verbose)
    duration = time.perf_counter() - t0
    if failed_test is not None:
      return MutantResult(site, "killed", duration, "")
    return MutantResult(site, "survived", duration, "")
  except Exception as exc:
    return MutantResult(site, "infra_error", 0.0, str(exc))


def main():
  parser = argparse.ArgumentParser(description="Run strict safety mutation")
  parser.add_argument("-j", type=int, default=max((os.cpu_count() or 1) - 1, 1), help="parallel mutants to run")
  parser.add_argument("--max-mutants", type=int, default=0, help="optional limit for debugging (0 means all)")
  parser.add_argument("--list-only", action="store_true", help="list discovered candidates and exit")
  parser.add_argument("--verbose", action="store_true", help="print extra debug output")
  args = parser.parse_args()

  start = time.perf_counter()

  with tempfile.TemporaryDirectory(prefix="mutation-op-run-") as run_tmp_dir:
    preprocessed_file = Path(run_tmp_dir) / "safety_preprocessed.c"
    sites, mutator_counts, build_incompatible_ids, preprocessed_source = enumerate_sites(ROOT / SAFETY_C_REL, preprocessed_file)
    assert len(sites) > 0

    if args.max_mutants > 0:
      sites = sites[: args.max_mutants]

    mutator_summary = ", ".join(f"{name} ({c})" for name in MUTATOR_FAMILIES if (c := mutator_counts.get(name, 0)) > 0)
    print(f"Found {len(sites)} unique candidates: {mutator_summary}", flush=True)
    if args.list_only:
      for site in sites:
        mutation = format_mutation(site.original_op, site.mutated_op)
        print(f"  #{site.site_id:03d} {site.origin_file.relative_to(ROOT)}:{site.origin_line} [{site.mutator}] {mutation}")
      return 0

    print(f"Running {len(sites)} mutants with {args.j} workers", flush=True)

    discovered_count = len(sites)
    selected_site_ids = {s.site_id for s in sites}
    build_incompatible_ids &= selected_site_ids
    pruned_compile_sites = len(build_incompatible_ids)
    if pruned_compile_sites > 0:
      sites = [s for s in sites if s.site_id not in build_incompatible_ids]
      print(f"Pruned {pruned_compile_sites} build-incompatible mutants from constant-expression initializers", flush=True)
    if not sites:
      print("Failed to build mutation library: all sites were pruned as build-incompatible", flush=True)
      return 2

    mutation_lib = Path(run_tmp_dir) / "libsafety_mutation.so"
    compile_mutated_library(preprocessed_source, sites, mutation_lib)

    # Discover all tests by importing modules in the main process.
    # Forked workers inherit these imports, eliminating per-worker import cost.
    catalog = _discover_test_catalog()

    # Baseline smoke check
    baseline_ids = catalog.get("test_defaults.py", [])[:5]
    baseline_failed = run_unittest(baseline_ids, mutation_lib, mutant_id=-1, verbose=args.verbose)
    if baseline_failed is not None:
      print("Baseline smoke failed with mutant_id=-1; aborting to avoid false kill signals.", flush=True)
      print(f"  failed_test: {baseline_failed}", flush=True)
      return 2

    # Pre-compute test targets per mutation site
    core_tests = _build_core_tests(catalog)
    site_targets = {site.site_id: build_priority_tests(site, catalog, core_tests) for site in sites}

    results = []
    counts = Counter()

    with ProcessPoolExecutor(max_workers=args.j) as pool:
      future_map = {
        pool.submit(eval_mutant, site, site_targets[site.site_id], mutation_lib, args.verbose): site for site in sites
      }
      print_live_status(render_progress(0, len(sites), 0, 0, 0, 0.0))
      try:
        for fut in as_completed(future_map):
          try:
            res = fut.result()
          except Exception:
            site = future_map[fut]
            res = MutantResult(site, "killed", 0.0, "worker process crashed")
          results.append(res)
          counts[res.outcome] += 1
          elapsed_now = time.perf_counter() - start
          done = len(results) == len(sites)
          print_live_status(render_progress(len(results), len(sites), counts["killed"], counts["survived"],
                                            counts["infra_error"], elapsed_now), final=done)
      except Exception:
        # Pool broken — mark all unfinished mutants as killed (crash = behavioral change detected)
        completed_ids = {r.site.site_id for r in results}
        for site in sites:
          if site.site_id not in completed_ids:
            results.append(MutantResult(site, "killed", 0.0, "pool broken"))
            counts["killed"] += 1
        elapsed_now = time.perf_counter() - start
        print_live_status(render_progress(len(results), len(sites), counts["killed"], counts["survived"], counts["infra_error"], elapsed_now), final=True)

    survivors = sorted((r for r in results if r.outcome == "survived"), key=lambda r: r.site.site_id)
    if survivors:
      print("", flush=True)
      print(colorize("Surviving mutants", ANSI_RED), flush=True)
      for res in survivors:
        loc = f"{res.site.origin_file.relative_to(ROOT)}:{res.site.origin_line}"
        mutation = format_mutation(res.site.original_op, res.site.mutated_op)
        print(f"- #{res.site.site_id} {loc} [{res.site.mutator}] {mutation}", flush=True)
        print(format_site_snippet(res.site), flush=True)

    infra_results = sorted((r for r in results if r.outcome == "infra_error"), key=lambda r: r.site.site_id)
    if infra_results:
      print("", flush=True)
      print(colorize("Infra errors", ANSI_YELLOW), flush=True)
      for res in infra_results:
        loc = f"{res.site.origin_file.relative_to(ROOT)}:{res.site.origin_line}"
        detail = res.details.splitlines()[0] if res.details else "unknown error"
        print(f"- #{res.site.site_id} {loc}: {detail}", flush=True)

    elapsed = time.perf_counter() - start
    total_test_sec = sum(r.test_sec for r in results)
    print("", flush=True)
    print(colorize("Mutation summary", ANSI_BOLD), flush=True)
    print(f"  discovered: {discovered_count}", flush=True)
    print(f"  pruned_build_incompatible: {pruned_compile_sites}", flush=True)
    print(f"  total: {len(sites)}", flush=True)
    print(f"  killed: {colorize(str(counts['killed']), ANSI_GREEN)}", flush=True)
    print(f"  survived: {colorize(str(counts['survived']), ANSI_RED)}", flush=True)
    print(f"  infra_error: {colorize(str(counts['infra_error']), ANSI_YELLOW)}", flush=True)
    print(f"  test_time_sum: {total_test_sec:.2f}s", flush=True)
    print(f"  avg_test_per_mutant: {total_test_sec / len(results):.3f}s", flush=True)
    print(f"  mutants_per_second: {len(sites) / elapsed:.2f}", flush=True)
    print(f"  elapsed: {elapsed:.2f}s", flush=True)

    if counts["infra_error"] > 0:
      return 2

    # TODO: fix these surviving mutants and delete this block
    known_survivors = {
      ("opendbc/safety/helpers.h", 40, "arithmetic"),
      ("opendbc/safety/lateral.h", 105, "boundary"),
      ("opendbc/safety/lateral.h", 195, "boundary"),
      ("opendbc/safety/lateral.h", 239, "boundary"),
      ("opendbc/safety/lateral.h", 337, "arithmetic"),
    }
    survivors = [r for r in survivors if (str(r.site.origin_file.relative_to(ROOT)), r.site.origin_line, r.site.mutator) not in known_survivors]

    if survivors:
      return 1
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
