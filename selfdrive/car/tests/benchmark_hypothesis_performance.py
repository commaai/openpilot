"""
Benchmark script to diagnose hypothesis performance regression.
Issue: https://github.com/commaai/openpilot/issues/32693

Usage:
  pip install hypothesis==6.47.0 && python benchmark_hypothesis.py
  pip install hypothesis==6.103.1 && python benchmark_hypothesis.py
"""

import time
import os
import sys
import hypothesis
from hypothesis import given, settings, Phase, strategies as st

print(f"Python version: {sys.version}")
print(f"Hypothesis version: {hypothesis.__version__}")
print()

# Test 1: Simple integer generation
def test_simple_integers():
    @settings(max_examples=100, phases=(Phase.reuse, Phase.generate, Phase.shrink), deadline=None)
    @given(st.integers())
    def inner(x):
        pass
    
    start = time.time()
    inner()
    return time.time() - start

# Test 2: Fixed dictionaries (used in openpilot)
def test_fixed_dictionaries():
    strategy = st.fixed_dictionaries({
        'a': st.integers(),
        'b': st.booleans(),
        'c': st.sampled_from([1, 2, 3, 4, 5]),
    })
    
    @settings(max_examples=100, phases=(Phase.reuse, Phase.generate, Phase.shrink), deadline=None)
    @given(strategy)
    def inner(d):
        pass
    
    start = time.time()
    inner()
    return time.time() - start

# Test 3: sampled_from (critical for openpilot)
DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
ALL_ECUS = list(range(100))  # Simulated ECU list

def test_sampled_from():
    strategy = st.sampled_from(DLC_TO_LEN)
    
    @settings(max_examples=1000, phases=(Phase.reuse, Phase.generate, Phase.shrink), deadline=None)
    @given(strategy)
    def inner(x):
        pass
    
    start = time.time()
    inner()
    return time.time() - start

# Test 4: Complex strategy similar to openpilot
def test_complex_strategy():
    fingerprint_strategy = st.fixed_dictionaries({
        0: st.dictionaries(
            st.integers(min_value=0, max_value=0x800),
            st.sampled_from(DLC_TO_LEN),
            max_size=10
        )
    })
    
    @settings(max_examples=60, phases=(Phase.reuse, Phase.generate, Phase.shrink), deadline=None)
    @given(fingerprint_strategy)
    def inner(d):
        pass
    
    start = time.time()
    inner()
    return time.time() - start

# Test 5: st.data() usage (the actual bottleneck in openpilot)
def test_data_strategy():
    @settings(max_examples=60, phases=(Phase.reuse, Phase.generate, Phase.shrink), deadline=None)
    @given(st.data())
    def inner(data):
        # Multiple draws like in openpilot
        for _ in range(10):
            data.draw(st.integers())
            data.draw(st.booleans())
            data.draw(st.sampled_from(DLC_TO_LEN))
    
    start = time.time()
    inner()
    return time.time() - start

if __name__ == "__main__":
    print("Running benchmarks...")
    print("-" * 50)
    
    results = []
    
    for name, func in [
        ("Simple integers", test_simple_integers),
        ("Fixed dictionaries", test_fixed_dictionaries),
        ("sampled_from (1000 examples)", test_sampled_from),
        ("Complex strategy", test_complex_strategy),
        ("st.data() multiple draws", test_data_strategy),
    ]:
        try:
            elapsed = func()
            results.append((name, elapsed))
            print(f"{name}: {elapsed:.3f}s")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
    
    print("-" * 50)
    print("\nSummary:")
    for name, elapsed in results:
        print(f"  {name}: {elapsed:.3f}s")
    
    total = sum(elapsed for _, elapsed in results)
    print(f"\nTotal: {total:.3f}s")
