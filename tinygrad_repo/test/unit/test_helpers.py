import gzip, unittest
from PIL import Image
from tinygrad import Variable
from tinygrad.helpers import Context, ContextVar
from tinygrad.helpers import merge_dicts, strip_parens, prod, round_up, fetch, fully_flatten, from_mv, to_mv, polyN
from tinygrad.tensor import get_shape
from tinygrad.codegen.lowerer import get_contraction
import numpy as np

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  # Ensuring that the test does not modify variables outside the tests.
  ctx = Context()
  def setUp(self): TestContextVars.ctx.__enter__()
  def tearDown(self): TestContextVars.ctx.__exit__()

  def test_initial_value_is_set(self):
    _TMP = ContextVar("_TMP", 5)
    self.assertEqual(_TMP.value, 5)

  @unittest.expectedFailure
  def test_multiple_creation_ignored(self):
    _TMP2 = ContextVar("_TMP2", 1)
    _TMP2 = ContextVar("_TMP2", 2)
    self.assertEqual(_TMP2.value, 1)

  @unittest.expectedFailure
  def test_new_var_inside_context(self):
    # Creating a _new_ variable inside a context should not have any effect on its scope (?)
    with Context(VARIABLE=1):
      _TMP3 = ContextVar("_TMP3", 1)
    _TMP3 = ContextVar("_TMP3", 2)
    self.assertEqual(_TMP3.value, 1)

  @unittest.expectedFailure
  def test_value_accross_modules(self):
    # Mocking module import by invoking the code but not in our globals().
    exec('from tinygrad.helpers import ContextVar;C = ContextVar("C", 13)', {}) # pylint:disable=exec-used
    # It should not matter that the first creation was in another module.
    C = ContextVar("C", 0)
    self.assertEqual(C.value, 13)

  @unittest.expectedFailure
  def test_assignment_across_modules(self):
    B = ContextVar("B", 1)
    # local assignment
    B.value = 2
    self.assertEqual(B.value, 2)
    # Assignment in another module.
    exec('from tinygrad.helpers import ContextVar;B = ContextVar("B", 0);B.value = 3;', {}) # pylint:disable=exec-used
    # Assignment in another module should affect this one as well.
    self.assertEqual(B.value, 3)

  def test_context_assignment(self):
    with Context(VARIABLE=1):
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_unknown_param_to_context(self):
    with self.assertRaises(KeyError):
      with Context(SOMETHING_ELSE=1):
        pass

  @unittest.expectedFailure
  def test_inside_context_assignment(self):
    with Context(VARIABLE=4):
      # What you can and cannot do inside a context.
      # 1. This type of statement has no effect.
      VARIABLE = ContextVar("VARIABLE", 0)
      self.assertTrue(VARIABLE >= 4, "ContextVars inside contextmanager may not set a new value")

      # 2. The call syntax however has a local effect.
      VARIABLE.value = 13
      self.assertTrue(VARIABLE.value == 13, "Call syntax however works inside a contextmanager.")

    # Related to 2. above. Note that VARIABLE is back to 0 again as expected.
    self.assertEqual(VARIABLE.value, 0)

  @unittest.expectedFailure
  def test_new_var_inside_context_other_module(self):
    with Context(VARIABLE=1):
      _NEW2 = ContextVar("_NEW2", 0)
    _NEW2 = ContextVar("_NEW2", 1)
    self.assertEqual(_NEW2.value, 0)

    code = """\
from tinygrad.helpers import Context, ContextVar
with Context(VARIABLE=1):
  _NEW3 = ContextVar("_NEW3", 0)"""
    exec(code, {})  # pylint:disable=exec-used
    # While _NEW3 was created in an outside scope it should still work the same as above.
    _NEW3 = ContextVar("_NEW3", 1)
    self.assertEqual(_NEW3.value, 0)

  def test_nested_context(self):
    with Context(VARIABLE=1):
      with Context(VARIABLE=2):
        with Context(VARIABLE=3):
          self.assertEqual(VARIABLE.value, 3)
        self.assertEqual(VARIABLE.value, 2)
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_decorator(self):
    @Context(VARIABLE=1, DEBUG=4)
    def test():
      self.assertEqual(VARIABLE.value, 1)

    self.assertEqual(VARIABLE.value, 0)
    test()
    self.assertEqual(VARIABLE.value, 0)

  def test_context_exit_reverts_updated_values(self):
    D = ContextVar("D", 1)
    D.value = 2
    with Context(D=3):
      ...
    assert D.value == 2, f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."

class TestMergeDicts(unittest.TestCase):
  def test_merge_dicts(self):
    a = {"a": 1, "b": 2}
    b = {"a": 1, "c": 3}
    c = {}
    d = {"a": 2, "b": 2}
    assert merge_dicts([a, b]) == {"a": 1, "b": 2, "c": 3}
    assert merge_dicts([a, c]) == a
    assert merge_dicts([a, b, c]) == {"a": 1, "b": 2, "c": 3}
    with self.assertRaises(AssertionError):
      merge_dicts([a, d])

class TestStripParens(unittest.TestCase):
  def test_simple(self): self.assertEqual("1+2", strip_parens("(1+2)"))
  def test_nested(self): self.assertEqual("1+(2+3)", strip_parens("(1+(2+3))"))
  def test_casted_no_strip(self): self.assertEqual("(int)(1+2)", strip_parens("(int)(1+2)"))

class TestProd(unittest.TestCase):
  def test_empty(self): self.assertEqual(1, prod(tuple()))
  def test_ints(self): self.assertEqual(30, prod((2, 3, 5)))
  def test_variable(self): self.assertEqual("(a*12)", prod((Variable("a", 1, 5), 3, 4)).render())
  def test_variable_order(self): self.assertEqual("(a*12)", prod((3, 4, Variable("a", 1, 5))).render())

class TestRoundUp(unittest.TestCase):
  def test_round_up(self):
    self.assertEqual(round_up(-3,4), 0)
    self.assertEqual(round_up(-4,4), -4)
    self.assertEqual(round_up(6,4), 8)
    self.assertEqual(round_up(8,4), 8)
    self.assertEqual(round_up(232, 24984), 24984)
    self.assertEqual(round_up(24984, 232), 25056)

@unittest.skip("no fetch tests because they need internet")
class TestFetch(unittest.TestCase):
  def test_fetch_bad_http(self):
    self.assertRaises(Exception, fetch, 'http://www.google.com/404', allow_caching=False)

  def test_fetch_small(self):
    assert (len(fetch('https://google.com', allow_caching=False).read_bytes())>0)

  def test_fetch_img(self):
    img = fetch("https://avatars.githubusercontent.com/u/132956020", allow_caching=False)
    with Image.open(img) as pimg:
      assert pimg.size == (77, 77), pimg.size

  def test_fetch_subdir(self):
    img = fetch("https://avatars.githubusercontent.com/u/132956020", allow_caching=False, subdir="images")
    with Image.open(img) as pimg:
      assert pimg.size == (77, 77), pimg.size
    assert img.parent.name == "images"

  def test_fetch_gunzip_valid(self):
    # compare fetch(gunzip=True) to fetch(gunzip=False) plus decompressing afterwards
    gzip_url: str = 'https://ftp.gnu.org/gnu/gzip/gzip-1.13.tar.gz'
    fp_gz = fetch(gzip_url, gunzip=True)
    fp_no_gz = fetch(gzip_url, gunzip=False)
    with open(fp_gz, 'rb') as f: content_gz = f.read()
    with open(fp_no_gz, 'rb') as f: content_no_gz = gzip.decompress(f.read())
    assert fp_gz.stat().st_size > fp_no_gz.stat().st_size
    assert isinstance(content_gz, bytes) and isinstance(content_no_gz, bytes)
    assert len(content_gz) == len(content_no_gz)
    assert content_gz == content_no_gz

  def test_fetch_gunzip_invalid(self):
    # given a non-gzipped file, fetch(gunzip=True) fails
    no_gzip_url: str = 'https://ftp.gnu.org/gnu/gzip/gzip-1.13.zip'
    with self.assertRaises(gzip.BadGzipFile):
      fetch(no_gzip_url, gunzip=True)

class TestFullyFlatten(unittest.TestCase):
  def test_fully_flatten(self):
    self.assertEqual(fully_flatten([[1, 3], [1, 2]]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten(((1, 3), (1, 2))), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[[1], [3]], [[1], [2]]]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[[[1], 2], 3], 4]), [1, 2, 3, 4])
    self.assertEqual(fully_flatten([[1, 2, [3, 4]], [5, 6], 7]), [1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(fully_flatten([[1, "ab"], [True, None], [3.14, [5, "b"]]]), [1, "ab", True, None, 3.14, 5, "b"])

  def test_fully_flatten_numpy(self):
    self.assertEqual(fully_flatten([np.array([])]), [])
    self.assertEqual(fully_flatten([np.array(3)]), [3])
    self.assertEqual(fully_flatten([np.array([3])]), [3])
    self.assertEqual(fully_flatten([np.array([[3]])]), [3])
    self.assertEqual(fully_flatten([np.array([1, 3]), np.array([1, 2])]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten((np.array([1, 3]), np.array([1, 2]))), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([np.array([[1], [3]]), np.array([[1], [2]])]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[1, "ab"], [True, None], np.array([[3.14], [6.28]])]), [1, "ab", True, None, 3.14, 6.28])

class TestMemoryview(unittest.TestCase):
  def test_from_mv_to_mv(self):
    base = memoryview(bytearray(b"\x11\x22\x33"*40))
    ct = from_mv(base)
    mv = to_mv(ct, len(base))
    mv[0] = 2
    assert base[0] == 2

class TestGetContraction(unittest.TestCase):
  def test_contraction(self):
    r = get_contraction((1,2,3,4), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3]])

    r = get_contraction((2,1,3,4), (2,3,4))
    self.assertEqual(r, [[0], [1, 2], [3]])

    r = get_contraction((1,2,3,1,4), (1,2,3,4))
    self.assertEqual(r, [[], [0, 1], [2], [3, 4]])

    r = get_contraction((1,2,3,1,4,1,1), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3, 4, 5, 6]])

    r = get_contraction((1,2,3,4), (1,2,3*4))
    self.assertEqual(r, [[], [0, 1], [2, 3]])

    r = get_contraction((1,2,3,4), (2,1,3,4))
    self.assertEqual(r, [[0, 1], [], [2], [3]])

    r = get_contraction((1,2,3,4), (1,1,2*3*4,1))
    self.assertEqual(r, [[], [], [0,1,2,3], []])

    r = get_contraction((2,1,3,4), (1,2,3,4))
    self.assertEqual(r, [[], [0], [1, 2], [3]])

    r = get_contraction((1,2,3,4), (2*3*4,1,1,1))
    self.assertEqual(r, [[0, 1, 2, 3], [], [], []])

    r = get_contraction((4,4,4,4), (16,1,16))
    self.assertEqual(r, [[0, 1], [], [2, 3]])

    r = get_contraction((1,2,3,4,1,1,1), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3, 4, 5, 6]])

    r = get_contraction((1,2,3,4), (1,2,3,4,1))
    self.assertEqual(r, [[], [0, 1], [2], [3], []])

    r = get_contraction((14,1,384,14,1,1,1,1), (1,14,384,14))
    self.assertEqual(r, [[], [0], [1,2], [3,4,5,6,7]])

    r = get_contraction((14,1,384,1,14,1,1,1,1), (1,14,384,14))
    self.assertEqual(r, [[], [0], [1,2], [3,4,5,6,7,8]])

    r = get_contraction((512, 512), (1, 1, 512, 1, 1, 1, 1, 512))
    self.assertEqual(r, [[], [], [0], [], [], [], [], [1]])

    r = get_contraction((1,2,3,4), (1,2,6,2))
    self.assertEqual(r, None)

  def test_contraction_ones(self):
    r = get_contraction((1,), (1,1,1))
    self.assertEqual(r, [[], [], [0]])

    r = get_contraction((1,1), (1,1,1))
    self.assertEqual(r, [[], [], [0, 1]])

    r = get_contraction((1,1,1,1), (1,))
    self.assertEqual(r, [[0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1))
    self.assertEqual(r, [[], [0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1,1))
    self.assertEqual(r, [[], [], [0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1,1,1))
    self.assertEqual(r, [[], [], [], [0,1,2,3]])

class TestGetShape(unittest.TestCase):
  def test_get_shape(self):
    assert get_shape(2) == ()
    assert get_shape([]) == (0,)
    assert get_shape([[]]) == (1, 0)
    assert get_shape([[1, 2]]) == (1, 2)
    assert get_shape([[1, 2], (3, 4)]) == (2, 2)

  def test_inhomogeneous_shape(self):
    with self.assertRaises(ValueError): get_shape([[], [1]])
    with self.assertRaises(ValueError): get_shape([[1, [2]], [1]])

class TestPolyN(unittest.TestCase):
  def test_float(self):
    np.testing.assert_allclose(polyN(1.0, [1.0, -2.0, 1.0]), 0.0)
    np.testing.assert_allclose(polyN(2.0, [1.0, -2.0, 1.0]), 1.0)
    np.testing.assert_allclose(polyN(3.0, [1.0, -2.0, 1.0]), 4.0)
    np.testing.assert_allclose(polyN(4.0, [1.0, -2.0, 1.0]), 9.0)

  def test_tensor(self):
    from tinygrad.tensor import Tensor
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

  def test_uop(self):
    from tinygrad.dtype import dtypes
    from tinygrad.ops import UOp
    from test.helpers import eval_uop
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 1.0), [1.0, -2.0, 1.0])), 0.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 2.0), [1.0, -2.0, 1.0])), 1.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 3.0), [1.0, -2.0, 1.0])), 4.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 4.0), [1.0, -2.0, 1.0])), 9.0)

if __name__ == '__main__':
  unittest.main()
