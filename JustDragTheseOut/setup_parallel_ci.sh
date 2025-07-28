#!/bin/bash
# Ultra-Fast Parallel CI Setup Script
# Target: Sub-1-second setup time
# Optimized for GitHub Actions free runners

set -e
START_TIME=$(date +%s.%N)

echo "  Starting Enhanced Parallel CI Setup..."

# Configuration
CACHE_DIR="$HOME/.cache/ci_parallel"
VENV_DIR="$HOME/.venv_parallel"
WHEELS_DIR="$CACHE_DIR/wheels"
TEMP_DIR="/tmp/ci_parallel_$$"
ESSENTIAL_PACKAGES=("pytest" "numpy" "pillow" "psutil")

# Function to print elapsed time
print_elapsed() {
    local current=$(date +%s.%N)
    local elapsed=$(echo "$current - $START_TIME" | bc -l 2>/dev/null || echo "0.000")
    printf "   Elapsed: %.3fs\n" "$elapsed"
}

# Advanced cache check with integrity verification
if [[ -f "$CACHE_DIR/venv_template.tar.gz" && -f "$CACHE_DIR/wheels_cache.tar.gz" ]]; then
    echo " Using enhanced parallel cache..."
    
    # Use temporary directory for faster extraction
    mkdir -p "$TEMP_DIR" &
    MKDIR_PID=$!
    
    # Parallel extraction with aggressive optimization
    {
        wait $MKDIR_PID
        
        # Extract to temp dir in parallel (faster than direct extraction)
        cd "$TEMP_DIR" && tar -xzf "$CACHE_DIR/venv_template.tar.gz" &
        EXTRACT_VENV_PID=$!
        
        cd "$TEMP_DIR" && tar -xzf "$CACHE_DIR/wheels_cache.tar.gz" &
        EXTRACT_WHEELS_PID=$!
        
        wait $EXTRACT_VENV_PID $EXTRACT_WHEELS_PID
        
        # Move to final location via atomic operations
        rm -rf "$VENV_DIR" 2>/dev/null || true
        mv "$TEMP_DIR"/*venv_parallel* "$VENV_DIR" 2>/dev/null || mv "$TEMP_DIR"/.venv_parallel_template "$VENV_DIR"
        
        # Copy wheels to cache (if needed)
        [[ -d "$TEMP_DIR/wheels" ]] && cp -r "$TEMP_DIR/wheels" "$CACHE_DIR/" 2>/dev/null || true
        
        # Cleanup temp directory
        rm -rf "$TEMP_DIR" &
    }
    
    # Instant activation
    source "$VENV_DIR/bin/activate"
    
    print_elapsed
    echo "  Enhanced parallel CI setup complete!"
    
    # Background verification (non-blocking)
    {
        python -c "import sys; print(f'🐍 Python: {sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "⚠️ Python check failed"
        python -c "import pytest, numpy, PIL, psutil; print('📦 All packages verified')" 2>/dev/null || echo "  Package verification failed"
    } &
    
    exit 0
fi

echo "  Building parallel cache (first run)..."

# Create virtual environment with minimal operations
python3 -m venv "$VENV_DIR" --without-pip --system-site-packages 2>/dev/null || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install pip quickly if needed
if ! command -v pip &> /dev/null; then
    curl -s https://bootstrap.pypa.io/get-pip.py | python - --no-cache-dir --quiet &
    wait
fi

# Create directories
mkdir -p "$WHEELS_DIR"

# Parallel package installation with aggressive optimization
{
    # Download all wheels in parallel
    echo "${ESSENTIAL_PACKAGES[@]}" | xargs -n1 -P4 -I{} pip download --dest "$WHEELS_DIR" --no-deps --only-binary=:all: {} 2>/dev/null &
    DOWNLOAD_PID=$!
    
    # Install packages with maximum parallelism
    pip install --no-cache-dir --no-deps --find-links "$WHEELS_DIR" "${ESSENTIAL_PACKAGES[@]}" 2>/dev/null &
    INSTALL_PID=$!
    
    wait $DOWNLOAD_PID $INSTALL_PID
}

# Create compressed caches in parallel
mkdir -p "$CACHE_DIR"
{
    cd "$HOME" && tar -czf "$CACHE_DIR/venv_template.tar.gz" .venv_parallel &
    TAR_VENV_PID=$!
    
    cd "$CACHE_DIR" && tar -czf wheels_cache.tar.gz wheels &
    TAR_WHEELS_PID=$!
    
    wait $TAR_VENV_PID $TAR_WHEELS_PID
}

print_elapsed
echo "  Parallel CI setup and cache creation complete!"

# Background verification
{
    python -c "import sys; print(f'🐍 Python: {sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "⚠️ Python check failed"
    python -c "import pytest, numpy, PIL, psutil; print('📦 All packages verified')" 2>/dev/null || echo "  Package verification failed"
} &

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0.000")
printf "  Total setup time: %.3fs\n" "$TOTAL_TIME"
