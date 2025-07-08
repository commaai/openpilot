import time, random, unittest, itertools
from unittest.mock import patch
from io import StringIO
from collections import namedtuple
from tqdm import tqdm
from tinygrad.helpers import tqdm as tinytqdm, trange as tinytrange
import numpy as np

def _get_iter_per_second(raw:str) -> float:
  # raw might have unit scale
  if raw.endswith("k"): return float(raw[:-1])*1e3
  if raw.endswith("M"): return float(raw[:-1])*1e6
  return float(raw)

# TODO: _get_iter_per_second in test_unit_scale might fail if lower bound is too small
NCOLS_RANGE = [80, 240]

class TestProgressBar(unittest.TestCase):
  def _compare_bars(self, bar1, bar2):
    prefix1, prog1, suffix1 = bar1.split("|")
    prefix2, prog2, suffix2 = bar2.split("|")

    self.assertEqual(len(bar1), len(bar2))
    self.assertEqual(prefix1, prefix2)

    def parse_timer(timer): return sum(int(x) * y for x, y in zip(timer.split(':')[::-1], (1, 60, 3600)))

    if "?" not in suffix1 and "?" not in suffix2:
      # allow for few sec diff in timers (removes flakiness)
      timer1, rm1 = [parse_timer(timer) for timer in suffix1.split("[")[-1].split(",")[0].split("<")]
      timer2, rm2 = [parse_timer(timer) for timer in suffix2.split("[")[-1].split(",")[0].split("<")]
      np.testing.assert_allclose(timer1, timer2, atol=5, rtol=1e-2)
      np.testing.assert_allclose(rm1, rm2, atol=5, rtol=1e-2)

      # get suffix without timers
      suffix1 = suffix1.split("[")[0] + suffix1.split(",")[1]
      suffix2 = suffix2.split("[")[0] + suffix2.split(",")[1]
      self.assertEqual(suffix1, suffix2)
    else:
      self.assertEqual(suffix1, suffix2)

    diff = sum([c1 != c2 for c1, c2 in zip(prog1, prog2)])  # allow 1 char diff to be less flaky, but it should match
    assert diff <= 1, f"{diff=}\n{prog1=}\n{prog2=}"

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_output_iter(self, mock_terminal_size, mock_stderr):
    for _ in range(10):
      total, ncols = random.randint(5, 30), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      # compare bars at each iteration (only when tinytqdm bar has been updated)
      for n in (bar := tinytqdm(range(total), desc="Test")):
        if bar.i % bar.skip != 0: continue
        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
        elapsed = n/iters_per_sec if n>0 else 0
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
        self._compare_bars(tinytqdm_output, tqdm_output)

      # compare final bars
      tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
      iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
      elapsed = total/iters_per_sec if n>0 else 0
      tqdm_output = tqdm.format_meter(n=total, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
      self._compare_bars(tinytqdm_output, tqdm_output)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_unit_scale(self, mock_terminal_size, mock_stderr):
    for unit_scale in [True, False]:
      # NOTE: numpy comparison raises TypeError if exponent > 22
      for exponent in range(1, 22, 3):
        low, high = 10 ** exponent, 10 ** (exponent+1)
        for _ in range(5):
          total, ncols = random.randint(low, high), random.randint(*NCOLS_RANGE)
          mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
          mock_stderr.truncate(0)

          # compare bars at each iteration (only when tinytqdm bar has been updated)
          # setting high rate to make sure it does not skip
          for n in tinytqdm(range(total), desc="Test", total=total, unit_scale=unit_scale, rate=10**9):
            tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()

            if n:
              iters_per_sec = _get_iter_per_second(tinytqdm_output.split("it/s")[-2].split(" ")[-1])
              elapsed = n/iters_per_sec
            else:
              elapsed = 0
            tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test", unit_scale=unit_scale)
            self._compare_bars(tinytqdm_output, tqdm_output)
            if n > 3: break

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_unit_scale_exact(self, mock_terminal_size, mock_stderr):
    unit_scale = True
    ncols = 80
    mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
    mock_stderr.truncate(0)

    total = 10
    with patch('time.perf_counter', side_effect=[0]+list(range(100))):  # one more 0 for the init call
      # compare bars at each iteration (only when tinytqdm bar has been updated)
      for n in tinytqdm(range(total), desc="Test", total=total, unit_scale=unit_scale, rate=10**9):
        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        elapsed = n
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test", unit_scale=unit_scale)
        self._compare_bars(tinytqdm_output, tqdm_output)
        if n > 5: break

    total = 10
    k=0.001000001
    # regression test for
    # E   AssertionError: ' 1.00/10.0  1000it/s]' != ' 1.00/10.0  1.00kit/s]'
    # E   -  1.00/10.0  1000it/s]
    # E   ?                ^
    # E   +  1.00/10.0  1.00kit/s]
    # E   ?              +  ^
    with patch('time.perf_counter', side_effect=[0, *[i*k for i in range(100)]]):  # one more 0 for the init call
      # compare bars at each iteration (only when tinytqdm bar has been updated)
      for n in tinytqdm(range(total), desc="Test", total=total, unit_scale=unit_scale, rate=10**9):
        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        elapsed = n*k
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test", unit_scale=unit_scale)
        self._compare_bars(tinytqdm_output, tqdm_output)
        if n > 5: break

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_set_description(self, mock_terminal_size, mock_stderr):
    for _ in range(10):
      total, ncols = random.randint(5, 30), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      expected_prefix = "Test"
      # compare bars at each iteration (only when tinytqdm bar has been updated)
      for i,n in enumerate(bar := tinytqdm(range(total), desc="Test")):
        if bar.i % bar.skip != 0: continue
        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
        elapsed = n/iters_per_sec if n>0 else 0
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=expected_prefix)
        expected_prefix = desc = f"Test {i}" if i % 2 == 0 else ""
        bar.set_description(desc)
        self._compare_bars(tinytqdm_output, tqdm_output)

      # compare final bars
      tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
      iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
      elapsed = total/iters_per_sec if n>0 else 0
      tqdm_output = tqdm.format_meter(n=total, total=total, elapsed=elapsed, ncols=ncols, prefix=expected_prefix)
      self._compare_bars(tinytqdm_output, tqdm_output)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_trange_output_iter(self, mock_terminal_size, mock_stderr):
    for _ in range(5):
      total, ncols = random.randint(5, 30), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      # compare bars at each iteration (only when tinytqdm bar has been updated)
      for n in (bar := tinytrange(total, desc="Test")):
        if bar.i % bar.skip != 0: continue
        tiny_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        iters_per_sec = float(tiny_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
        elapsed = n/iters_per_sec if n>0 else 0
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
        self._compare_bars(tiny_output, tqdm_output)

      # compare final bars
      tiny_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
      iters_per_sec = float(tiny_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
      elapsed = total/iters_per_sec if n>0 else 0
      tqdm_output = tqdm.format_meter(n=total, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
      self._compare_bars(tiny_output, tqdm_output)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_output_custom(self, mock_terminal_size, mock_stderr):
    for _ in range(10):
      total, ncols = random.randint(10000, 1000000), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      # compare bars at each iteration (only when tinytqdm bar has been updated)
      bar = tinytqdm(total=total, desc="Test")
      n = 0
      while n < total:
        incr = (total // 100) + random.randint(0, 1000)
        if n + incr > total: incr = total - n
        bar.update(incr, close=n+incr==total)
        n += incr
        if bar.i % bar.skip != 0: continue

        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
        elapsed = n/iters_per_sec if n>0 else 0
        tqdm_output = tqdm.format_meter(n=n, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
        self._compare_bars(tinytqdm_output, tqdm_output)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_output_custom_0_total(self, mock_terminal_size, mock_stderr):
    for _ in range(10):
      total, ncols = random.randint(10000, 100000), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      # compare bars at each iteration (only when tinytqdm bar has been updated)
      bar = tinytqdm(total=0, desc="Test")
      n = 0
      while n < total:
        incr = (total // 10) + random.randint(0, 100)
        if n + incr > total: incr = total - n
        bar.update(incr, close=n+incr==total)
        n += incr
        if bar.i % bar.skip != 0: continue

        tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
        iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1]) if n>0 else 0
        elapsed = n/iters_per_sec if n>0 else 0
        tqdm_output = tqdm.format_meter(n=n, total=0, elapsed=elapsed, ncols=ncols, prefix="Test")
        self.assertEqual(tinytqdm_output, tqdm_output)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_output_custom_nolen_total(self, mock_terminal_size, mock_stderr):
    for unit_scale in [True, False]:
      for _ in range(5):
        gen = itertools.count(0)
        ncols = random.randint(*NCOLS_RANGE)
        mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
        mock_stderr.truncate(0)

        # compare bars at each iteration (only when tinytqdm bar has been updated)
        # setting high rate to make sure it does not skip
        for n,g in enumerate(tinytqdm(gen, desc="Test", unit_scale=unit_scale, rate=10**9)):
          assert g == n
          tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
          if n:
            iters_per_sec = _get_iter_per_second(tinytqdm_output.split("it/s")[-2].split(" ")[-1])
            elapsed = n/iters_per_sec
          else:
            elapsed = 0
          tqdm_output = tqdm.format_meter(n=n, total=0, elapsed=elapsed, ncols=ncols, prefix="Test", unit_scale=unit_scale)
          self.assertEqual(tinytqdm_output, tqdm_output)
          if n > 5: break

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_write(self, mock_terminal_size, mock_stderr):
    for _ in range(5):
      ncols, tqdm_fp = random.randint(*NCOLS_RANGE), StringIO()
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)
      tqdm_fp.truncate(0)
      for i in tinytqdm(range(10)):
        time.sleep(0.01)
        tinytqdm.write(str(i))
        tqdm.write(str(i), file=tqdm_fp)
        tinytqdm_out, tqdm_out = mock_stderr.getvalue(), tqdm_fp.getvalue()
        self.assertEqual(tinytqdm_out.split("\r\033[K")[-1], tqdm_out.split(f"{i-1}\n")[-1])
      self.assertEqual(tinytqdm_out, tinytqdm_out)

  @patch('sys.stderr', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_context_manager(self, mock_terminal_size, mock_stderr):
    for _ in range(10):
      total, ncols = random.randint(5, 30), random.randint(*NCOLS_RANGE)
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(ncols)
      mock_stderr.truncate(0)

      with tinytqdm(desc="Test", total=total) as bar:
        for _ in range(total):
          bar.update(1)

      tinytqdm_output = mock_stderr.getvalue().split("\r")[-1].rstrip()
      iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1])
      elapsed = total/iters_per_sec
      tqdm_output = tqdm.format_meter(n=total, total=total, elapsed=elapsed, ncols=ncols, prefix="Test")
      self._compare_bars(tinytqdm_output, tqdm_output)

  def test_tqdm_perf(self):
    st = time.perf_counter()
    for _ in tqdm(range(100)): pass
    tqdm_time = time.perf_counter() - st

    st = time.perf_counter()
    for _ in tinytqdm(range(100)): pass
    tinytqdm_time = time.perf_counter() - st

    assert tinytqdm_time < 2 * tqdm_time

  def test_tqdm_perf_high_iter(self):
    st = time.perf_counter()
    for _ in tqdm(range(10^7)): pass
    tqdm_time = time.perf_counter() - st

    st = time.perf_counter()
    for _ in tinytqdm(range(10^7)): pass
    tinytqdm_time = time.perf_counter() - st

    assert tinytqdm_time < 5 * tqdm_time

if __name__ == '__main__':
  unittest.main()
