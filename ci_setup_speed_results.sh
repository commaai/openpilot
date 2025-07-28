#!/bin/bash

# CI Setup Speed Results Script
# Comprehensive benchmark of different CI setup methods

set -e

echo "🏃‍♂️ Running CI setup speed benchmarks..."
echo "========================================"

# Initialize results
echo "Method,Duration,Status,Description" > /tmp/ci_results.tmp

# Test function
benchmark_method() {
    local method_name="$1"
    local script_path="$2"
    local description="$3"
    
    if [ ! -f "$script_path" ]; then
        echo "$method_name,MISSING,FAILED,$description - Script not found" >> /tmp/ci_results.tmp
        echo "⚠️  $method_name: Script not found"
        return 1
    fi
    
    echo "🔄 Testing $method_name..."
    
    # Clean environment
    rm -rf ~/.venv /tmp/.openpilot_* 2>/dev/null || true
    
    # Time the setup
    start_time=$(python3 -c "import time; print(time.time())")
    
    if timeout 120 bash "$script_path" >/dev/null 2>&1; then
        end_time=$(python3 -c "import time; print(time.time())")
        duration=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")
        
        if (( $(python3 -c "print(int($duration < 1.0))") )); then
            status="TARGET"
            echo "🎯 $method_name: ${duration}s (TARGET ACHIEVED!)"
        elif (( $(python3 -c "print(int($duration < 20.0))") )); then
            status="GOOD"
            echo "✅ $method_name: ${duration}s"
        else
            status="SLOW"
            echo "⚠️  $method_name: ${duration}s (slow)"
        fi
        
        echo "$method_name,$duration,$status,$description" >> /tmp/ci_results.tmp
        return 0
    else
        echo "❌ $method_name: FAILED"
        echo "$method_name,FAILED,FAILED,$description - Setup failed" >> /tmp/ci_results.tmp
        return 1
    fi
}

# Benchmark available methods
echo "📊 Benchmarking CI Setup Methods"
echo "--------------------------------"

# Main parallel method
benchmark_method "Parallel CI Setup" "./setup_parallel_ci.sh" "Optimized parallel setup with caching"

# Check for other methods
if [ -f "./setup_nano_ci.sh" ]; then
    benchmark_method "Nano CI Setup" "./setup_nano_ci.sh" "Minimal nano setup"
fi

if [ -f "./setup_minimal_ci.sh" ]; then
    benchmark_method "Minimal CI Setup" "./setup_minimal_ci.sh" "Basic minimal setup"
fi

# Results summary
echo ""
echo "📈 Benchmark Results Summary"
echo "============================"

if [ -f "/tmp/ci_results.tmp" ]; then
    echo "Full Results:"
    cat /tmp/ci_results.tmp
    echo ""
    
    # Best performance
    if grep -q "TARGET" /tmp/ci_results.tmp; then
        echo "🎯 TARGET ACHIEVED!"
        target_methods=$(grep "TARGET" /tmp/ci_results.tmp | wc -l)
        echo "Methods achieving target (<1s): $target_methods"
        
        best_line=$(grep "TARGET" /tmp/ci_results.tmp | sort -t',' -k2 -n | head -1)
        best_method=$(echo "$best_line" | cut -d',' -f1)
        best_time=$(echo "$best_line" | cut -d',' -f2)
        echo "Fastest method: $best_method in ${best_time}s"
    else
        echo "Target not achieved, showing best results:"
        if grep -q "GOOD" /tmp/ci_results.tmp; then
            best_line=$(grep "GOOD" /tmp/ci_results.tmp | sort -t',' -k2 -n | head -1)
            best_method=$(echo "$best_line" | cut -d',' -f1)
            best_time=$(echo "$best_line" | cut -d',' -f2)
            echo "Best method: $best_method in ${best_time}s"
        fi
    fi
    
    # Performance stats
    total_methods=$(grep -v "^Method," /tmp/ci_results.tmp | wc -l)
    successful_methods=$(grep -v "FAILED" /tmp/ci_results.tmp | grep -v "^Method," | wc -l)
    echo "Success rate: $successful_methods/$total_methods methods"
fi

echo ""
echo "✅ Benchmark completed!"
