#!/bin/bash

# CI Speed Test Script
# Measures the total CI setup speed on GitHub Actions free runners

set -euo pipefail

# Check required commands are available
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed." >&2; exit 1; }

# Record start time
start_time=$(python3 -c 'import time; print(time.time())')

echo "🚀 Starting CI speed test..."

# Clean any existing environment (use explicit HOME variable)
rm -rf "${HOME}/.venv" /tmp/.openpilot_* 2>/dev/null || true

# Check if setup script exists
if [[ ! -f "./setup_parallel_ci.sh" ]]; then
    echo "Error: setup_parallel_ci.sh not found in current directory" >&2
    exit 1
fi

# Run the preferred CI setup script
chmod +x ./setup_parallel_ci.sh
./setup_parallel_ci.sh

# Record end time and calculate duration
end_time=$(python3 -c 'import time; print(time.time())')
duration=$(python3 -c "import sys; start=float(sys.argv[1]); end=float(sys.argv[2]); print(f'{end-start:.6f}')" "${start_time}" "${end_time}")

echo "⚡ CI setup completed in ${duration}s"

# Verify Python environment
if python -c 'import sys; print(f"Python version: {sys.version}")' >/dev/null 2>&1; then
    echo "✅ Python environment is ready"
else
    echo "❌ Python environment not ready"
    exit 1
fi

# Verify key packages
if python -c 'import pytest, numpy, PIL, psutil' >/dev/null 2>&1; then
    echo "✅ Required packages are installed"
else
    echo "❌ Package verification failed"
    exit 1
fi

echo "✅ All verification checks passed"
exit 0
