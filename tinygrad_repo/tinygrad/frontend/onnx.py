# type: ignore
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
try:
  from extra.onnx import OnnxRunner # noqa: F401 # pylint: disable=unused-import
except ImportError as e: raise ImportError("onnx frontend not in release\nTo fix, install tinygrad from a git checkout with pip install -e .") from e