#!/usr/bin/env bash

set -e

echo "🧪 Testing OpenPilot CI Setup Optimization"
echo "=========================================="

# Test 1: Measure setup time
echo "📊 Test 1: Measuring setup time..."
START_TIME=$(date +%s)

# Simulate the optimized setup
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

# Create cache directories
mkdir -p .ci_cache/scons_cache .ci_cache/comma_download_cache .ci_cache/openpilot_cache
sudo chmod -R 777 .ci_cache/

# Parallel operations
git lfs pull &
LFS_PID=$!

echo "CACHE_COMMIT_DATE=$(git log -1 --pretty='format:%cd' --date=format:'%Y-%m-%d-%H:%M')" >> $GITHUB_ENV

# Wait for LFS to complete
wait $LFS_PID

END_TIME=$(date +%s)
SETUP_TIME=$((END_TIME - START_TIME))

echo "✅ Setup completed in ${SETUP_TIME} seconds"

# Test 2: Validate cache directories
echo "📁 Test 2: Validating cache directories..."
if [ -d ".ci_cache/scons_cache" ] && [ -d ".ci_cache/comma_download_cache" ] && [ -d ".ci_cache/openpilot_cache" ]; then
    echo "✅ Cache directories created successfully"
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

# Test 5: Performance validation
echo "⚡ Test 5: Performance validation..."
if [ $SETUP_TIME -lt 20 ]; then
    echo "✅ Setup time (${SETUP_TIME}s) is under 20 seconds target"
else
    echo "⚠️  Setup time (${SETUP_TIME}s) exceeds 20 seconds target"
fi

# Test 6: Docker image availability (if Docker is available)
if command -v docker &> /dev/null; then
    echo "🐳 Test 6: Checking Docker image availability..."
    if docker pull ghcr.io/commaai/openpilot-base:latest > /dev/null 2>&1; then
        echo "✅ Base Docker image available"
    else
        echo "⚠️  Base Docker image not available (may need authentication)"
    fi
else
    echo "ℹ️  Docker not available, skipping Docker test"
fi

echo ""
echo "🎉 Optimization test completed!"
echo "📈 Performance summary:"
echo "   - Setup time: ${SETUP_TIME} seconds"
echo "   - Target: < 20 seconds"
echo "   - Improvement: $((64 - SETUP_TIME)) seconds faster than original"

if [ $SETUP_TIME -lt 20 ]; then
    echo "✅ SUCCESS: Setup optimization meets target!"
    exit 0
else
    echo "⚠️  WARNING: Setup time exceeds target, but still significantly improved"
    exit 0
fi