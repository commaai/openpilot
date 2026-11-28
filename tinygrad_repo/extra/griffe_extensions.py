import ast
from pathlib import Path
import griffe

_AUTOGEN = Path(__file__).resolve().parents[1] / "tinygrad/runtime/autogen"

class SkipAutogen(griffe.Extension):
  def on_module_node(self, *, node, agent, **kwargs):
    # No generated bindings are documented. Keep their modules, but don't collect their members.
    if isinstance(node, ast.Module) and _AUTOGEN in agent.filepath.resolve().parents: node.body.clear()
