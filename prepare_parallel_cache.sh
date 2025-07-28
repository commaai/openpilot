#!/bin/bash

# Prepare Parallel Cache for CI optimization
# This script sets up optimized caching for parallel CI execution

set -e

echo "🚀 Preparing parallel cache for CI optimization..."

# Create cache directory
mkdir -p ~/.openpilot_ci_cache

# Cache system information
echo "System Info:" > ~/.openpilot_ci_cache/system_info.txt
echo "OS: $(uname -a)" >> ~/.openpilot_ci_cache/system_info.txt
echo "CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')" >> ~/.openpilot_ci_cache/system_info.txt
echo "Python: $(python3 --version)" >> ~/.openpilot_ci_cache/system_info.txt

# Create minimal virtual environment for CI
if [ ! -d ~/.venv ]; then
    echo "📦 Creating optimized virtual environment..."
    python3 -m venv ~/.venv
fi

# Activate environment
source ~/.venv/bin/activate

# Upgrade pip for speed
pip install --upgrade pip setuptools wheel

# Cache essential packages that are commonly used
echo "📥 Pre-caching essential packages..."
pip install pytest pytest-xdist numpy pillow psutil 2>/dev/null || true

echo "✅ Parallel cache preparation completed!"
echo "Cache location: ~/.openpilot_ci_cache"
echo "Virtual env: ~/.venv"
