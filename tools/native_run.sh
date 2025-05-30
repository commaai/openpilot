#!/usr/bin/env bash
set -e

# Native run helper - replaces Docker RUN for native execution
# Usage: ./tools/native_run.sh "command to run"

# Ensure we're in the openpilot directory
cd "$(dirname "$0")/.."

# Source the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Source environment variables
if [ -f ".env" ]; then
    source .env
fi

# Set up cache directories
mkdir -p .ci_cache/scons_cache
mkdir -p .ci_cache/comma_download_cache
mkdir -p .ci_cache/openpilot_cache

# Set environment variables that would normally be set by Docker
export CI=1
export PYTHONWARNINGS=error
export FILEREADER_CACHE=1
export PYTHONPATH="$PWD"
export OPENPILOT_PREFIX="$PWD"
export SCONS_CACHE_DIR="$PWD/.ci_cache/scons_cache"

# Set cache environment variables
export COMMA_DOWNLOAD_CACHE="$PWD/.ci_cache/comma_download_cache"
export OPENPILOT_CACHE="$PWD/.ci_cache/openpilot_cache"

# Run the command
exec bash -c "$1"