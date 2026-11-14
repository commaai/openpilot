import ast
from pathlib import Path

MICI = Path(__file__).parents[1]


def is_status(node: ast.AST) -> bool:
  return isinstance(node, ast.Attribute) and node.attr == "status" and isinstance(node.value, ast.Name) and node.value.id == "ui_state"


def violations(src: str):
  for node in ast.walk(ast.parse(src)):
    # switching on engaged vs override, e.g. ui_state.status == UIStatus.ENGAGED
    if isinstance(node, ast.Compare | ast.Match) and any(isinstance(n, ast.Attribute) and n.attr in ("ENGAGED", "OVERRIDE") and
                                                         isinstance(n.value, ast.Name) and n.value.id == "UIStatus" for n in ast.walk(node)):
      yield node.lineno
    # looking up per status values, e.g. COLORS[ui_state.status]
    elif isinstance(node, ast.Subscript) and is_status(node.slice) or isinstance(node, ast.Call) and any(is_status(a) for a in node.args):
      yield node.lineno


def test_status_fade():
  """engaged and override flap every frame, so they must only be told apart through status_fade (mici/onroad/status.py)"""
  found = [f"{path.relative_to(MICI)}:{line}" for path in sorted(MICI.rglob("*.py")) if "tests" not in path.parts
           for line in violations(path.read_text())]
  assert not found, f"blend engaged and override through status_fade, or compare against UIStatus.DISENGAGED: {found}"


def test_catches():
  assert list(violations("x = ui_state.status == UIStatus.ENGAGED")) == [1]
  assert list(violations("x = ui_state.status in (UIStatus.ENGAGED, UIStatus.OVERRIDE)")) == [1]
  assert list(violations("x = COLORS[ui_state.status]")) == [1]
  assert list(violations("x = COLORS.get(ui_state.status)")) == [1]
  assert list(violations("x = ui_state.status != UIStatus.DISENGAGED")) == []
  assert list(violations("x = status_fade.color({UIStatus.ENGAGED: a, UIStatus.OVERRIDE: b})")) == []
