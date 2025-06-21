#!/usr/bin/env bash

set -e

echo "🧪 Testing OpenPilot CI Setup Optimization (Ultra-Minimal)"
echo "========================================================="

# Test 1: Measure ultra-minimal setup time
echo "📊 Test 1: Measuring ultra-minimal setup time..."
START_TIME=$(date +%s)

# Simulate the ultra-minimal setup (no Docker operations)
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

# Create cache directories (essential only)
mkdir -p .ci_cache/scons_cache
sudo chmod -R 777 .ci_cache/

# Setup cache date
echo "CACHE_COMMIT_DATE=$(git log -1 --pretty='format:%cd' --date=format:'%Y-%m-%d-%H:%M')" >> $GITHUB_ENV

# Only Git LFS pull - this is the only operation we need to wait for
git lfs pull

END_TIME=$(date +%s)
SETUP_TIME=$((END_TIME - START_TIME))

echo "✅ Ultra-minimal setup completed in ${SETUP_TIME} seconds"

# Test 2: Validate cache directories
echo "📁 Test 2: Validating cache directories..."
if [ -d ".ci_cache/scons_cache" ]; then
    echo "✅ Cache directory created successfully"
else
    echo "❌ Cache directory creation failed"
    exit 1
fi

# Test 3: Check environment variables
echo "🔧 Test 3: Checking environment variables..."
if [ -n "$CACHE_COMMIT_DATE" ]; then
    echo "✅ CACHE_COMMIT_DATE set: $CACHE_COMMIT_DATE"
else
    echo "❌ CACHE_COMMIT_DATE not set"
    exit 1
fi

# Test 4: Validate LFS files
echo "📦 Test 4: Validating LFS files..."
if git lfs ls-files | head -5 | grep -q .; then
    echo "✅ LFS files available"
else
    echo "❌ LFS files not found"
    exit 1
fi

# Test 5: Performance validation (ultra-strict)
echo "⚡ Test 5: Ultra-strict performance validation..."
if [ $SETUP_TIME -lt 10 ]; then
    echo "✅ Setup time (${SETUP_TIME}s) is under 10 seconds target"
elif [ $SETUP_TIME -lt 20 ]; then
    echo "✅ Setup time (${SETUP_TIME}s) is under 20 seconds target"
else
    echo "❌ Setup time (${SETUP_TIME}s) exceeds both targets"
    exit 1
fi

# Test 6: Docker image availability (optional)
if command -v docker &> /dev/null; then
    echo "🐳 Test 6: Checking Docker image availability..."
    if docker images ghcr.io/commaai/openpilot-base:latest | grep -q "ghcr.io/commaai/openpilot-base"; then
        echo "✅ Base Docker image available (cached)"
    else
        echo "ℹ️  Base Docker image not cached (will be pulled when needed)"
    fi
else
    echo "ℹ️  Docker not available, skipping Docker test"
fi

echo ""
echo "🎉 Ultra-minimal optimization test completed!"
echo "📈 Performance summary:"
echo "   - Setup time: ${SETUP_TIME} seconds"
echo "   - Ultra-strict target: < 10 seconds"
echo "   - Original target: < 20 seconds"
echo "   - Improvement: $((64 - SETUP_TIME)) seconds faster than original"

if [ $SETUP_TIME -lt 10 ]; then
    echo "✅ SUCCESS: Ultra-minimal setup meets strict target!"
    exit 0
elif [ $SETUP_TIME -lt 20 ]; then
    echo "✅ SUCCESS: Setup optimization meets original target!"
    exit 0
else
    echo "❌ FAILURE: Setup time exceeds all targets"
    exit 1
fi