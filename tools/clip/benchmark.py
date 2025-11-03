#!/usr/bin/env python3
"""
Minimal benchmark script for clip.py performance measurement.
"""
import subprocess
import sys
import time
from pathlib import Path


def run_benchmark():
    clip_script = Path('tools/clip/run.py')
    output_file = 'benchmark_output.mp4'

    if not clip_script.exists():
        print(f"Error: {clip_script} not found")
        return None

    print(f"Running benchmark: {clip_script}")

    cmd = [sys.executable, str(clip_script), '--demo', '--output', output_file]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()

        total_time = end_time - start_time

        success = result.returncode == 0
        file_size = Path(output_file).stat().st_size / (1024 * 1024) if Path(output_file).exists() else 0

        print(f"\n{'='*50}")
        print(f"Results:")
        print(f"  Success: {success}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Output Size: {file_size:.2f} MB")
        if not success:
            print(f"  Return Code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
        print(f"{'='*50}\n")

        return {
            'success': success,
            'time': total_time,
            'file_size_mb': file_size,
            'return_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        print("Benchmark timed out after 10 minutes")
        return None


if __name__ == '__main__':
    result = run_benchmark()
    sys.exit(0 if result and result['success'] else 1)

