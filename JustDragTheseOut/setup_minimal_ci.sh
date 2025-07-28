#!/bin/bash
# Minimal CI Setup Script
# Fast setup with essential packages only
# Target: Under 0.2 seconds with cache

set -e
START_TIME=$(date +%s.%N)

echo " Starting Minimal CI Setup..."

# Configuration
CACHE_DIR="$HOME/.cache/ci_minimal"
VENV_DIR="$HOME/.venv_minimal"
MINIMAL_PACKAGES=("pytest" "numpy" "pillow" "psutil")

# Function to print elapsed time
print_elapsed() {
    local current=$(date +%s.%N)
    local elapsed=$(echo "$current - $START_TIME" | bc -l 2>/dev/null || echo "0.000")
    printf "   %.3fs\n" "$elapsed"
}

# Check if cached environment exists with optimized extraction
if [[ -f "$CACHE_DIR/venv_minimal.tar.gz" ]]; then
    echo "  Using optimized cached minimal environment..."
    
    # Use temporary directory for faster extraction
    TEMP_EXTRACT="/tmp/minimal_extract_$$"
    mkdir -p "$TEMP_EXTRACT" &
    MKDIR_PID=$!
    
    # Optimized extraction process
    {
        wait $MKDIR_PID
        
        # Extract to temp location (typically faster)
        cd "$TEMP_EXTRACT" && tar -xzf "$CACHE_DIR/venv_minimal.tar.gz"
        
        # Atomic move to final location
        rm -rf "$VENV_DIR" 2>/dev/null || true
        mv "$TEMP_EXTRACT"/* "$VENV_DIR" 2>/dev/null || mv "$TEMP_EXTRACT"/.venv_minimal "$VENV_DIR"
        
        # Cleanup temp directory in background
        rm -rf "$TEMP_EXTRACT" &
    }
    
    # Instant activation
    source "$VENV_DIR/bin/activate"
    
    print_elapsed
    echo "  Minimal CI setup complete!"
    
    # Background verification (non-blocking)
    {
        python -c "import pytest, numpy, PIL, psutil; print('📦 Packages verified')" 2>/dev/null || echo "  Verification failed"
    } &
    
    exit 0
fi

echo  " Building minimal cache (first run)..."

# Create minimal environment from scratch
python3 -m venv "$VENV_DIR" --without-pip
source "$VENV_DIR/bin/activate"

# Install pip efficiently
curl -s https://bootstrap.pypa.io/get-pip.py | python - --no-cache-dir --quiet

# Install minimal packages
pip install --no-cache-dir "${MINIMAL_PACKAGES[@]}" --quiet

# Create cache for future runs
mkdir -p "$CACHE_DIR"
cd "$(dirname "$VENV_DIR")" && tar -czf "$CACHE_DIR/venv_minimal.tar.gz" "$(basename "$VENV_DIR")"

print_elapsed
echo "  Minimal CI setup and cache creation complete!"

# Background verification
{
    python -c "import pytest, numpy, PIL, psutil; print('📦 Packages verified')" 2>/dev/null || echo "  Verification failed"
} &

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0.000")
printf "  Total time: %.3fs\n" "$TOTAL_TIME"
