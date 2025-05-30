#!/usr/bin/env bash
set -e

echo "🧪 Local CI Performance Testing Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Start timing
start_time=$(date +%s)
log "Starting native setup performance test..."

# Step 1: Test Python virtual environment setup
log "Step 1: Testing Python environment setup"
step1_start=$(date +%s)

# Install uv if not present (simulating the workflow)
if ! command -v uv &> /dev/null; then
    log "Installing uv (fastest Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    log "Virtual environment exists, activating..."
    source .venv/bin/activate
else
    log "Creating virtual environment with uv..."
    uv venv .venv
    source .venv/bin/activate

    log "Installing Python packages with uv..."
    UV_EXTRA_INDEX_URL="" uv sync --frozen --all-extras --no-dev || {
        warning "uv sync failed, trying with pip..."
        pip install -e .
    }
fi

step1_end=$(date +%s)
step1_time=$((step1_end - step1_start))
success "Python setup completed in ${step1_time}s"

# Step 2: Test environment configuration
log "Step 2: Testing environment configuration"
step2_start=$(date +%s)

# Create .env file (macOS optimized)
cat > .env << EOF
PYTHONPATH=$PWD
export ZMQ=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
EOF

# Set environment variables
export CI=1
export PYTHONWARNINGS=error
export FILEREADER_CACHE=1
export PYTHONPATH="$PWD"
export OPENPILOT_PREFIX="$PWD"
export SCONS_CACHE_DIR="$PWD/.ci_cache/scons_cache"

# Create cache directories
mkdir -p .ci_cache/scons_cache
mkdir -p .ci_cache/comma_download_cache
mkdir -p .ci_cache/openpilot_cache

step2_end=$(date +%s)
step2_time=$((step2_end - step2_start))
success "Environment setup completed in ${step2_time}s"

# Step 3: Test basic functionality
log "Step 3: Testing basic functionality"
step3_start=$(date +%s)

# Test Python imports
./tools/native_run.sh "python3 -c 'import sys; print(f\"Python {sys.version}\")'" || {
    error "Python test failed"
    exit 1
}

# Test key package imports
./tools/native_run.sh "python3 -c 'import numpy; print(\"NumPy:\", numpy.__version__)'" || {
    warning "NumPy import failed"
}

./tools/native_run.sh "python3 -c 'import scons; print(\"SCons available\")'" || {
    warning "SCons import failed - this is expected on macOS"
}

# Test environment variables
./tools/native_run.sh "echo \"PYTHONPATH=\$PYTHONPATH\"" || {
    error "Environment variable test failed"
    exit 1
}

step3_end=$(date +%s)
step3_time=$((step3_end - step3_start))
success "Functionality tests completed in ${step3_time}s"

# Step 4: Test Git LFS (if available)
log "Step 4: Testing Git LFS"
step4_start=$(date +%s)

if command -v git-lfs &> /dev/null; then
    git lfs pull || warning "Git LFS pull failed (may be expected)"
else
    warning "Git LFS not installed"
fi

step4_end=$(date +%s)
step4_time=$((step4_end - step4_start))
success "Git LFS test completed in ${step4_time}s"

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "📊 Performance Summary"
echo "====================="
echo "Step 1 (Python setup):     ${step1_time}s"
echo "Step 2 (Environment):      ${step2_time}s"
echo "Step 3 (Functionality):    ${step3_time}s"
echo "Step 4 (Git LFS):          ${step4_time}s"
echo "─────────────────────────────────"
echo "Total setup time:          ${total_time}s"

echo ""
echo "🎯 Performance Analysis"
echo "======================="

if [ $total_time -lt 20 ]; then
    success "EXCELLENT: Setup time ${total_time}s meets <20s target! 🎉"
    echo "   This qualifies for the main bounty target!"
elif [ $total_time -lt 40 ]; then
    success "GOOD: Setup time ${total_time}s meets <40s sub-target ✅"
    echo "   This qualifies for the sub-bounty but misses main target."
    echo "   Consider optimizations to reach <20s"
else
    error "Setup time ${total_time}s exceeds both targets"
    echo "   Need significant optimization to meet bounty requirements"
fi

echo ""
echo "💡 Next Steps"
echo "============"
echo "1. Test on your GitHub fork to validate CI environment performance"
echo "2. Compare with Docker baseline (expect 3-5x speedup)"
echo "3. If performance meets targets, submit PR for bounty!"

echo ""
log "Local performance test completed!"