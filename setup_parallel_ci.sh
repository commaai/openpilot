#!/bin/bash

# Parallel CI Setup Script - Optimized for speed
# Target: Under 20 seconds setup

set -e

echo "⚡ Starting parallel CI setup..."

# Use cached environment if available
if [ -d ~/.venv ]; then
    echo "🎯 Using cached virtual environment"
    source ~/.venv/bin/activate
else
    echo "📦 Creating new virtual environment"
    python3 -m venv ~/.venv
    source ~/.venv/bin/activate
    pip install --upgrade pip setuptools wheel
fi

# Install only essential dependencies for CI
echo "📥 Installing essential CI dependencies..."
pip install --quiet pytest pytest-xdist 2>/dev/null || true

# Set CI mode environment variable
export CI=true
export OPENPILOT_PREFIX=/tmp/.openpilot_ci

# Create minimal openpilot prefix for CI
mkdir -p $OPENPILOT_PREFIX

echo "✅ Parallel CI setup completed!"
echo "Environment: $VIRTUAL_ENV"
echo "CI mode: $CI"
echo "Prefix: $OPENPILOT_PREFIX"
